"""Run serve measurement over a file and fold the numbers into the AI shot result."""
from __future__ import annotations

import importlib.util
import time
from pathlib import Path

from analysis.shot_analysis import Cancelled, aggregate, pick_source

MATCH_WINDOW = 0.8  # s between a Gemini serve and a measured contact to call them the same serve


def tracking_ready() -> dict:
    missing = [m for m in ("torch", "cv2", "numpy") if importlib.util.find_spec(m) is None]
    return {"ready": not missing, "missing": missing}


def measure_file(project: dict, file_entry: dict, root: Path, shots: dict | None,
                 report=None, cancelled=None) -> dict:
    from analysis.tracknet import BallDetector, active_weights
    from analysis.serve_speed import ServeMeter, Court

    report = report or (lambda d: None)
    cancelled = cancelled or (lambda: False)
    cal = (file_entry.get("calibration") or {}).get("points") or []
    if len(cal) < 4:
        raise RuntimeError("코트 4점 캘리브레이션이 필요해. 영상 위에서 먼저 코트 모서리 4곳을 찍어줘.")
    fid = file_entry["id"]
    points = sorted((p for p in project.get("points", []) if p.get("file_id") == fid), key=lambda p: p["start"])
    gem_serves = [e for e in (shots or {}).get("events", []) if e["shot"] == "serve" and e.get("file_id") == fid]
    if not points and not gem_serves:
        raise RuntimeError("측정할 서브를 찾을 기준이 없어. 포인트를 라벨링하거나 AI 샷 분석을 먼저 해줘.")

    weights = active_weights(root, report)
    report({"percent": 2.0, "message": "공 추적 모델 불러오는 중..."})
    meter = ServeMeter(BallDetector(weights), Court(cal),
                       pick_source(file_entry, "original", 0)[0],
                       float((file_entry.get("metadata") or {}).get("fps") or 59.4))

    jobs = [("point", p) for p in points]
    serves: list[dict] = []
    for k, (kind, p) in enumerate(jobs):
        if cancelled():
            raise Cancelled()
        report({"percent": 3 + 45 * k / max(1, len(jobs)),
                "message": f"서브 공 추적 중 · 포인트 {k + 1}/{len(jobs)}"})
        for n, s in enumerate(meter.at_point(p["start"], p["end"])):
            s.update(point_id=p["id"], point_no=k + 1, serve_number=n + 1)
            serves.append(s)

    # Serves the AI saw but no labeled point covered (or that were missed there). Not after a serve
    # that went in within the same point: from then on it's a rally, and a rally has no serves.
    def after_good_serve(e):
        return any(s.get("point_id") == e.get("point_id") and e.get("point_id") and s["call"] in ("in", "close")
                   and s["contact_t"] < e["t"] - 0.5 for s in serves)
    leftovers = [e for e in gem_serves if not any(abs(e["t"] - s["contact_t"]) <= MATCH_WINDOW for s in serves)
                 and not after_good_serve(e)]
    for k, e in enumerate(leftovers):
        if cancelled():
            raise Cancelled()
        report({"percent": 93 + 6 * k / max(1, len(leftovers)), "message": f"AI가 찾은 서브 추적 중 · {k + 1}/{len(leftovers)}"})
        s = meter.near(e["t"])
        if s and not any(abs(s["contact_t"] - x["contact_t"]) < 0.3 for x in serves):
            s.update(point_id=e.get("point_id"), point_no=e.get("point_no"), serve_number=e.get("serve_number") or 0)
            serves.append(s)
    serves.sort(key=lambda s: s["contact_t"])
    audio = refine_with_audio(serves, meter, report)
    add_serve_types(serves, meter, report)
    add_motion_types(serves, project, file_entry, meter, root, report, cancelled)
    for s in serves:
        s.pop("_flight", None)
    rally = find_point_hits(points, serves, meter, report, cancelled)

    return {
        "file_id": fid,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": "TrackNet + court homography + drag model",
        "points_checked": len(points),
        "serves": serves,
        "warnings": server_end_warnings(project, serves),
        "audio": audio,
        "rally_hits": rally,
    }


def find_point_hits(points, serves, meter, report=None, cancelled=None) -> dict:
    """Every racket hit in each labelled point from the ball track (analysis/hit_detect.py)."""
    from analysis.hit_detect import find_rally_hits
    report = report or (lambda d: None)
    anchors = [{"t": s["contact_t"], "side": s["server_end"], "fault": s["call"] in ("net", "out")} for s in serves]
    hits, failed = [], []
    for k, p in enumerate(points):
        if cancelled and cancelled():
            raise Cancelled()
        report({"percent": 50 + 45 * k / max(1, len(points)), "message": f"랠리 타구 찾는 중 · 포인트 {k + 1}/{len(points)}"})
        try:
            found = find_rally_hits(meter, meter.src, p["start"], p["end"], anchors)
        except Exception as exc:
            failed.append(f"#{k + 1}: {exc}")
            continue
        for h in found:
            h.update(point_id=p["id"], point_no=k + 1)
        hits += found
    rally = [h for h in hits if not h.get("serve")]
    return {"hits": hits, "failed": failed,
            "stats": {"points": len(points), "rally_hits": len(rally),
                      "near": sum(h["side"] == "near" for h in rally), "far": sum(h["side"] == "far" for h in rally),
                      "with_sound": sum(1 for h in rally if h.get("audio"))}}


def add_motion_types(serves, project, file_entry, meter, root, report=None, cancelled=None):
    """Serve type from the server's follow-through, learnt per server from labelled serves
    (analysis/serve_motion.py). Fills only serves the ball's bend left unclassified."""
    from analysis.overlay_video import ensure_pose_weights, pose_ready
    if not serves or not pose_ready():
        return
    from ultralytics import YOLO
    from analysis.serve_motion import leg_split, learn_types
    meta = file_entry.get("metadata") or {}
    W, H = int(meta.get("width") or 1920), int(meta.get("height") or 1080)
    model = YOLO(str(ensure_pose_weights(root, report)))
    for k, s in enumerate(serves):
        if cancelled and cancelled():
            raise Cancelled()
        if report:
            report({"percent": 48.0 + 2.0 * k / len(serves), "message": f"서브 동작 측정 중 · {k + 1}/{len(serves)}"})
        try:
            s["leg_split"] = leg_split(model, meter.src, meter.court, W, H, s)
        except Exception:
            s["leg_split"] = None
    labels = [l for l in project.get("shot_labels", []) if l.get("file_id") == file_entry["id"]]
    learn_types(serves, labels, project.get("points", []))
    for s in serves:
        if not s.get("serve_type_measured") and s.get("serve_type_motion"):
            s["serve_type_measured"] = s["serve_type_motion"]
            s["serve_type_how"] = "motion"
        elif s.get("serve_type_measured"):
            s["serve_type_how"] = "flight"


def add_serve_types(serves, meter, report=None):
    """Serve type from how the measured flight bends (analysis/serve_spin.py). Right-handed servers assumed."""
    from analysis.serve_spin import measure_spin
    from analysis.serve_speed import COURT_L
    if meter.cam is None:
        return
    if report:
        report({"percent": 99.5, "message": "서브 궤적 휘어짐으로 구질 판정 중..."})
    for s in serves:
        flight = s.get("_flight")
        if not flight or s.get("method") != "bounce" or not s.get("contact_height_m") or not s.get("bounce_court"):
            continue
        p0 = (s["server_court"][0], 0.0 if s["server_end"] == "near" else COURT_L, s["contact_height_m"])
        try:
            r = measure_spin(meter.cam, p0, s["bounce_court"], s["bounce_t"] - s["contact_t"], flight,
                             s["contact_t"], s["server_end"])
        except Exception:
            r = None
        if r:
            s.update(r)


def refine_with_audio(serves, meter, report=None) -> dict:
    """Far-side contacts onto the racket sound, with the A/V offset learnt from near-side serves."""
    from analysis.serve_audio import av_offset, refine_far_contacts
    if meter.cam is None:
        return {"used": False, "reason": "no camera model"}
    if report:
        report({"percent": 99.0, "message": "타구음으로 먼 쪽 서브 임팩트 보정 중..."})
    try:
        offset, samples = av_offset(serves, meter.src, meter.cam)
    except Exception as exc:
        return {"used": False, "reason": f"audio read failed: {exc}"}
    if offset is None:
        return {"used": False, "reason": "not enough consistent near-side serves to learn the A/V offset",
                "samples_ms": [round(r * 1000) for r in samples]}
    moved = refine_far_contacts(serves, meter.src, meter.cam, offset)
    return {"used": True, "av_offset_ms": round(offset * 1000), "samples_ms": [round(r * 1000) for r in samples],
            "far_contacts_moved": moved}


def server_end_warnings(project: dict, serves: list[dict]) -> list[str]:
    """The score tells which end the server should be on; the ball says where they really were."""
    by_id = {p["id"]: p for p in project.get("points", [])}
    bad = []
    for s in serves:
        p = by_id.get(s.get("point_id"))
        st = (p or {}).get("score_before") or {}
        server, a_end = st.get("server_id"), st.get("team_a_end")
        if not server or a_end not in ("near", "far"):
            continue
        expected = a_end if server[0].upper() == "A" else ("far" if a_end == "near" else "near")
        if s["server_end"] != expected:
            bad.append(s.get("point_no"))
    if not bad:
        return []
    nos = ", ".join(f"#{n}" for n in sorted({b for b in bad if b}))
    if len(bad) == len([s for s in serves if s.get("point_id")]):
        return [f"포인트 {nos}: 공 추적상 서버가 모두 스코어 설정과 반대쪽 코트에 있었어. "
                f"경기 설정의 ‘Team A 시작 위치’를 반대로 바꾸면 팀 구분이 맞아져."]
    return [f"포인트 {nos}: 공 추적상 서버는 반대쪽 코트에 있었어. 경기 설정의 서브 순서나 ‘Team A 시작 위치’를 확인해줘."]


def summarize(serves: list[dict]) -> dict:
    sp = [s["speed_kmh"] for s in serves]
    calls = {c: sum(1 for s in serves if s["call"] == c) for c in ("in", "out", "net", "close")}
    return {"count": len(serves), "avg": round(sum(sp) / len(sp)) if sp else None,
            "max": max(sp) if sp else None, "calls": calls}


def apply_measurements(result: dict, measured: dict, settings: dict) -> dict:
    """Replace AI-estimated serve speeds with measured ones and add serves the AI missed."""
    serves = list(measured.get("serves") or [])
    used = set()
    events = result.get("events", [])
    for e in events:
        if e["shot"] != "serve":
            continue
        best, best_d = None, None
        for i, s in enumerate(serves):
            d = abs(e["t"] - s["contact_t"])
            if i not in used and d <= MATCH_WINDOW and (best_d is None or d < best_d):
                best, best_d = i, d
        if best is None:
            e.setdefault("speed_source", "ai")
            continue
        used.add(best)
        _fold(e, serves[best])
    for i, s in enumerate(serves):
        if i in used:
            continue
        ev = {
            "t": s["contact_t"], "file_id": result.get("file_id") or measured.get("file_id"),
            "point_id": s.get("point_id"), "point_no": s.get("point_no"),
            "shot": "serve", "side": s["server_end"], "lane": "unknown", "spin": "unknown",
            "serve_number": s.get("serve_number") or 0, "serve_type": "unknown",
            "serve_direction": s["direction"], "result": "unknown", "speed_kmh": None,
            "confidence": 70, "team": None, "player": None, "fixes": ["added_by_tracking"],
        }
        _fold(ev, s)
        events.append(ev)
    events.sort(key=lambda e: e["t"])
    result["events"] = events
    result["summary"] = aggregate(events, settings)
    result["measured_serves"] = summarize(serves)
    result["measure_warnings"] = list(measured.get("warnings") or [])
    return result


def _fold(e: dict, s: dict):
    if e.get("speed_source") != "measured":
        e["speed_est"] = e.get("speed_kmh")
    e.update(
        speed_kmh=s["speed_kmh"], speed_source="measured", speed_err_kmh=s["speed_err_kmh"],
        bounce_court=s["bounce_court"], serve_call=s["call"], measured_direction=s["direction"],
        server_end=s["server_end"], flight_s=s["flight_s"],
    )
    if s["call"] in ("out", "net"):
        e["result"] = "fault"
    elif s["call"] == "in" and e.get("result") in ("fault", "unknown"):
        e["result"] = "in"
    if e.get("serve_direction") in (None, "unknown"):
        e["serve_direction"] = s["direction"]
    if s.get("serve_type_measured"):  # trajectory physics beats the AI's guess (see analysis/serve_spin.py)
        if e.get("serve_type_source") != "measured":
            e["serve_type_ai"] = e.get("serve_type")
        e.update(serve_type=s["serve_type_measured"], serve_type_source="measured")
    for k in ("spin_side_ms2", "spin_up_ms2", "spin_fit_px"):
        if k in s:
            e[k] = s[k]


def serves_only_result(measured: dict, file_entry: dict, settings: dict) -> dict:
    """A minimal shot result built from measured serves, so the dashboard works without Gemini."""
    res = {"file_id": file_entry["id"], "file_name": file_entry.get("name"), "events": [],
           "scope": "points", "source": "original", "model": "ball-tracking", "analyzed_at": measured["measured_at"],
           "analyzed_seconds": 0, "segment_count": measured.get("points_checked", 0), "overall_confidence": 0,
           "warnings": [], "observations": [], "serves_only": True}
    return apply_measurements(res, measured, settings)
