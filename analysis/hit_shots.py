"""Shot types one hit at a time: short clips cut at tracked hit moments, judged by Gemini.

The hit moments come from the ball track (analysis/hit_detect.py, saved by the serve
measurement as cache/<file>_hits.json). For each one a 1.3 s clip of the hitter's half of the
court is cut from the original video at native resolution, and many clips go inline in one
request (the free tier allows about 20 requests per model per day), each with a narrow
question about its single hit. Serves come from the serve measurement itself.

Forehand vs backhand is then settled by the hitter's swing measured with pose estimation
(analysis/swing_side.py) whenever that is clear, since Gemini's call on it is close to a coin
flip at this camera distance (4-6 of 10 on the test points, vs 10/10 from the swing), and
volley vs ground stroke by whether the tracked ball bounced first (hit_detect.mark_bounces).

On the hand-labelled points 1-6 of the test video the detector found all 21 labelled hits
on the right side, and Gemini 3.5 Flash named 8 of the 14 rally strokes from the detected
moments (9/14 from the label times), with the net volleys right where the whole-video
prompt kept missing them.
"""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from analysis import gemini_client as gemini
from analysis.serve_speed import COURT_L, COURT_W, NET_Y, Court
from analysis.shot_analysis import (FALLBACK_MODEL, SHOTS, Cancelled, aggregate, build_segments, _attribute,
                                    _clamp, _enum)
from analysis.swing_side import stroke
from core.media import exe

BEFORE, AFTER = 0.8, 0.5   # s of clip around the hit
FAR_SOUND_S = 0.15         # a far-side hit is heard about this much after it happens (distance + A/V offset)
CLIP_FPS = 10              # frames per second Gemini samples from each clip
BATCH = 20                 # clips per request
SERVE_MATCH_S = 0.8        # a detected rally opener this close to a measured serve is that serve
VERSION = 2
SWING_VERSION = 1

HIT = {
    "type": "OBJECT",
    "properties": {
        "clip": {"type": "INTEGER"},
        "found_hit": {"type": "BOOLEAN"},
        "hit_time": {"type": "NUMBER"},
        "hitter_position": {"type": "STRING", "enum": ["baseline", "midcourt", "net", "unknown"]},
        "bounced_before_hit": {"type": "STRING", "enum": ["yes", "no", "unknown"]},
        "racket_side": {"type": "STRING", "enum": ["forehand", "backhand", "overhead", "unknown"]},
        "shot": {"type": "STRING", "enum": ["serve", "forehand", "backhand", "volley", "smash", "lob", "other"]},
        "confidence": {"type": "INTEGER"},
        "reason": {"type": "STRING"},
    },
    "required": ["clip", "found_hit", "bounced_before_hit", "racket_side", "shot", "confidence"],
}
SCHEMA = {"type": "OBJECT", "properties": {"hits": {"type": "ARRAY", "items": HIT}}, "required": ["hits"]}


def crop_box(court: Court, side: str, W: int, H: int):
    """Pixel box (x, y, w, h) covering one half of the court plus room for the players' height."""
    if side == "far":
        ground = [(-3, NET_Y - 1), (COURT_W + 3, NET_Y - 1), (COURT_W + 3, COURT_L + 4), (-3, COURT_L + 4)]
        head_y = COURT_L + 2
    else:
        ground = [(-3, -4), (COURT_W + 3, -4), (COURT_W + 3, NET_Y + 1), (-3, NET_Y + 1)]
        head_y = NET_Y - 1
    pts = [court.to_image(x, y) for x, y in ground]
    xs = [p[0] * W for p in pts]
    ys = [p[1] * H for p in pts]
    tall = court.px_per_m(COURT_W / 2, head_y) * (W / 640) * 3.0  # a player + raised racket
    x0, x1 = max(0, min(xs)), min(W, max(xs))
    y0, y1 = max(0, min(ys) - tall), min(H, max(ys))
    w, h = int(x1 - x0) // 2 * 2, int(y1 - y0) // 2 * 2
    return int(x0), int(y0), w, h


def cut(src, t, box, out: Path):
    from core.media import hw_decode_args
    x, y, w, h = box
    args = ["-nostdin", "-loglevel", "error", "-y", "-ss", f"{max(0, t - BEFORE):.3f}",
            "-i", src, "-t", f"{BEFORE + AFTER:.3f}", "-vf", f"crop={w}:{h}:{x}:{y},scale='min(1280,iw)':-2",
            "-r", "30", "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "96k", str(out)]
    hw = hw_decode_args()
    if hw and subprocess.run([exe("ffmpeg"), *hw, *args], capture_output=True).returncode == 0:
        return
    subprocess.run([exe("ffmpeg"), *args], check=True)


def prompt(doubles: bool, hands: str = "") -> str:
    return "\n".join([
        "아래는 같은 테니스 경기에서 잘라낸 짧은 클립(각 약 1.3초)들입니다. 각 클립은 한쪽 코트 절반만 잘라낸 화면입니다.",
        "각 클립 바로 앞의 설명에 적힌 순간, 그 한 번의 타구만 판정해 hits 배열에 클립마다 하나씩 넣으세요 (clip 번호 포함)."
        + (" 복식이라 같은 코트에 두 명이 있을 수 있습니다." if doubles else ""),
        "- hit_time: 클립 기준 임팩트 시각(초). 타구음과 공의 방향 전환을 함께 보세요.",
        "- hitter_position: 친 선수 위치 (baseline 베이스라인 근처 / midcourt / net 네트 근처)",
        "- bounced_before_hit: 이 타구 직전에 공이 이 선수 쪽 코트에 바운스했는지 (yes / no). 바운스 없이 친 공은 발리입니다.",
        "- racket_side: 선수의 라켓 쥔 손 쪽에서 쳤으면 forehand, 반대쪽(몸 앞을 가로질러)이면 backhand, 머리 위면 overhead.",
        "  라켓을 쥔 손이 오른손인지 왼손인지 먼저 확인하세요. 양손 백핸드는 두 손이 라켓에 있습니다.",
        "  방향 주의: 카메라 쪽(가까운) 선수는 등을 보이므로 선수의 오른쪽 = 화면 오른쪽,"
        " 먼 쪽 선수는 카메라를 마주 보므로 선수의 오른쪽 = 화면 왼쪽입니다."
        " 오른손잡이는 공을 자기 몸 오른쪽에서 치면 forehand, 왼쪽에서 치면 backhand (왼손잡이는 반대).",
        *([hands] if hands else []),
        "- shot: 바운스 후 타구면 racket_side에 따라 forehand / backhand, 바운스 전 타구면 volley, 머리 위 강타는 smash,"
        " 높게 띄우면 lob, 베이스라인 뒤에서 공을 토스해 치는 첫 타구는 serve.",
        "- confidence 0~100, reason 한 문장(한국어).",
        "타구가 안 보이면 found_hit=false. 순수 JSON만 출력하세요.",
    ])


def clip_line(k: int, side: str, note: str = "") -> str:
    who = "카메라에서 먼 쪽 코트" if side == "far" else "카메라 쪽(가까운) 코트"
    return f"클립 {k}: 시작 후 약 {BEFORE:.1f}초 무렵(±0.5초), {who}의 선수 한 명이 라켓으로 공을 칩니다." + (
        f" {note}" if note else "")


def _side_players(settings: dict, seg: dict, side: str) -> list[str] | None:
    a_end = seg.get("team_a_end")
    if a_end not in ("near", "far") or side not in ("near", "far"):
        return None
    names = [n for n in (settings.get("team_a") if side == a_end else settings.get("team_b")) or [] if n]
    return names[:1] if (settings.get("mode") or "doubles") == "singles" else names


def side_hand(settings: dict, seg: dict, side: str) -> str | None:
    """"right"/"left" when every player on `side` holds the racket in that hand, None when mixed."""
    lefties = {n.strip() for n in settings.get("left_handed") or [] if n and n.strip()}
    names = _side_players(settings, seg, side)
    if not names:
        return "right"  # players not entered: most are right-handed
    hands = {"left" if n.strip() in lefties else "right" for n in names}
    return hands.pop() if len(hands) == 1 else None


def _server_end(seg: dict) -> str | None:
    a_end, server = seg.get("team_a_end"), seg.get("server_id")
    if a_end not in ("near", "far") or not server:
        return None
    return a_end if server[0].upper() == "A" else ("far" if a_end == "near" else "near")


def hand_note(settings: dict, seg: dict, side: str) -> str:
    """Who plays on `side` in this point and which hand they hold the racket with.

    Players not listed in settings["left_handed"] are taken as right-handed, the usual case."""
    names = _side_players(settings, seg, side)
    if not names:
        return ""
    lefties = {n.strip() for n in settings.get("left_handed") or [] if n and n.strip()}
    hands = [f"{n} {'왼손잡이' if n.strip() in lefties else '오른손잡이'}" for n in names]
    if not lefties & {n.strip() for n in names}:
        return "이 코트 선수는 모두 오른손잡이입니다."
    return "이 코트 선수: " + ", ".join(hands) + "."


def clip_time(hit: dict) -> float:
    """Where to centre the clip: far hits are heard a little after they happen."""
    return hit["t"] - (FAR_SOUND_S if hit["side"] == "far" and hit.get("audio") else 0.0)


def classify(api_key: str, model: str, src: str, court: Court, W: int, H: int, hits: list[dict], work: Path,
             doubles: bool = True, hands: str = "", fps: float = CLIP_FPS) -> dict:
    """One request for all `hits` ([{"t", "side", "note"?, ...}]). Returns {"answers": {index: answer}, "usage"}."""
    work.mkdir(parents=True, exist_ok=True)

    def make(item):
        k, h = item
        clip = work / f"hit_{k:03d}_{clip_time(h):.2f}.mp4"
        cut(src, clip_time(h), crop_box(court, h["side"], W, H), clip)
        return clip

    with ThreadPoolExecutor(4) as pool:
        clips = list(pool.map(make, enumerate(hits)))
    parts = [{"text": prompt(doubles, hands)}]
    for k, (h, clip) in enumerate(zip(hits, clips), 1):
        parts.append({"text": clip_line(k, h["side"], h.get("note") or "")})
        parts.append({"inlineData": {"mimeType": "video/mp4", "data": base64.b64encode(clip.read_bytes()).decode()},
                      "videoMetadata": {"fps": fps}})
    try:
        r = gemini.generate_json(api_key, model, parts, SCHEMA, media_resolution="MEDIA_RESOLUTION_HIGH")
    finally:
        for c in clips:
            try:
                c.unlink()
            except OSError:
                pass
    answers = {}
    for a in (r["json"] or {}).get("hits", []) if isinstance(r.get("json"), dict) else []:
        try:
            k = int(a.get("clip")) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= k < len(hits):
            answers[k] = a
    return {"answers": answers, "usage": r.get("usage") or {}}


def hits_path(cache_dir: Path, fid: str) -> Path:
    return Path(cache_dir) / f"{fid}_hits.json"


class HitShotJob:
    """Same result shape as ShotAnalysisJob, built from tracked hits instead of whole-video chunks."""

    def __init__(self, *, project: dict, file_entry: dict, api_key: str, model: str, cache_dir: Path,
                 report=None, cancelled=None):
        self.project, self.f = project, file_entry
        self.api_key, self.model = api_key, model or gemini.DEFAULT_MODEL
        self.cache_dir = Path(cache_dir)
        self.report = report or (lambda d: None)
        self.cancelled = cancelled or (lambda: False)

    def _check(self):
        if self.cancelled():
            raise Cancelled()

    def run(self) -> dict:
        fid = self.f["id"]
        settings = self.project.get("settings") or {}
        fp = hits_path(self.cache_dir, fid)
        if not fp.exists():
            raise RuntimeError("타구 시점이 아직 없어. ‘서브 실측’을 먼저 돌려줘 (랠리 타구도 같이 찾아).")
        tracked = json.loads(fp.read_text(encoding="utf-8"))
        measured_fp = self.cache_dir / f"{fid}_serves.json"
        measured = json.loads(measured_fp.read_text(encoding="utf-8")) if measured_fp.exists() else {"serves": []}
        serve_ts = [s["contact_t"] for s in measured.get("serves", [])]
        everything = tracked.get("hits", [])
        for i, h in enumerate(everything):  # straight after a serve with no rally between: a 2nd serve the meter missed?
            prev = everything[i - 1] if i else None
            h["_after_serve"] = bool(prev and prev.get("serve") and prev.get("point_id") == h.get("point_id"))
        hits = [h for h in everything if not h.get("serve")
                and not (h.get("first") and any(abs(h["t"] - t) <= SERVE_MATCH_S for t in serve_ts))]
        if not hits:
            raise RuntimeError("찾은 랠리 타구가 없어. 포인트 구간과 코트 캘리브레이션을 확인해줘.")

        meta = self.f.get("metadata") or {}
        W, H = int(meta.get("width") or 1920), int(meta.get("height") or 1080)
        court = Court(self.f["calibration"]["points"])
        src = self.f["path"]
        doubles = (settings.get("mode") or "doubles") == "doubles"
        segs = build_segments(self.f, self.project.get("points") or [], "points")
        hits = [dict(h, note=hand_note(settings, self._seg(segs, h), h["side"])) for h in hits]
        warnings: list[str] = []
        batches = [hits[i:i + BATCH] for i in range(0, len(hits), BATCH)]
        answers: dict[int, dict] = {}
        usage = {"prompt": 0, "output": 0}
        used_models = set()
        work = self.cache_dir / "hits_work" / fid
        # Gemini answers in the background while the pose model measures the swings here:
        # neither needs the other until the events are put together
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(1) as pool:
            pending = [pool.submit(self._request, batch, court, src, W, H, doubles, work) for batch in batches]
            swings = self._swings(hits, court, src, W, H, fid, warnings)
            for b, (batch, fut) in enumerate(zip(batches, pending)):
                base = b * BATCH
                while not fut.done():
                    self._check()
                    self.report({"percent": round(50 + 45 * b / len(batches), 1),
                                 "message": f"타구 {base + 1}~{base + len(batch)} / {len(hits)} 판정 결과 기다리는 중 (요청 {b + 1}/{len(batches)})"})
                    time.sleep(0.5)
                try:
                    res, model = fut.result()
                except gemini.GeminiError as exc:
                    if exc.status in (400, 401, 403, 404, 499):
                        raise
                    # Gemini overloaded or out of quota: keep going on the ball and the swing alone
                    warnings.append(f"Gemini가 응답하지 않아 ({exc.status}) 타구 {base + 1}~{base + len(batch)}는 "
                                    "공 궤적·스윙 방향으로만 판정했어. 나중에 다시 돌리면 보완돼.")
                    continue
                used_models.add(model)
                for k, a in res["answers"].items():
                    answers[base + k] = a
                usage["prompt"] += int(res["usage"].get("promptTokenCount") or 0)
                usage["output"] += int(res["usage"].get("candidatesTokenCount") or 0)
        if any(m != self.model for m in used_models):
            warnings.append(f"{self.model} 과부하·하루 한도로 일부 요청은 {FALLBACK_MODEL}로 판정했어.")

        events = []
        unknown = 0
        for k, h in enumerate(hits):
            a = answers.get(k) or {}
            seg = self._seg(segs, h)
            shot = _enum(a.get("shot"), SHOTS, "other") if a else "other"
            # only a rally's opening hit from the server's end can be a serve; anything else overhead is a smash
            overhead_in_rally = shot == "serve" and (not h.get("first") or _server_end(seg) not in (None, h["side"]))
            if overhead_in_rally:
                shot = "smash"
            missed_serve = bool(h.get("first") and h.get("_after_serve") and _server_end(seg) == h["side"])
            if missed_serve:
                shot = "serve"
            ev = {
                "t": round(clip_time(h), 2), "file_id": fid,
                "point_id": seg.get("point_id"), "point_no": seg.get("point_no"),
                "shot": shot, "side": h["side"], "lane": "unknown", "spin": "unknown",
                "serve_number": 0, "serve_type": "unknown" if shot == "serve" else None,
                "serve_direction": "unknown" if shot == "serve" else None,
                "result": "unknown", "speed_kmh": None,
                "confidence": _clamp(a.get("confidence")) if a else 0,
                "team": None, "player": None, "fixes": ["tracked_hit"],
                "hit_source": "tracking", "hit_audio": h.get("audio"),
                "bounced_before_hit": a.get("bounced_before_hit"), "racket_side": a.get("racket_side"),
                "hitter_position": a.get("hitter_position"), "reason": a.get("reason"),
            }
            sw = swings.get(k)
            if sw:
                hand = side_hand(settings, seg, h["side"])
                side_by_swing = stroke(sw["swing"], hand)
                ev["swing"] = sw["swing"]
                if side_by_swing:
                    ev["racket_side"] = side_by_swing
                    if shot in ("forehand", "backhand"):
                        if shot != side_by_swing:
                            ev["fixes"].append("swing_side")
                        ev["shot"] = side_by_swing
            if h.get("bounced") is not None:  # the ball track settles volley vs ground stroke (smashes and lobs stay)
                ev["bounced_before_hit"] = "yes" if h["bounced"] else "no"
                if not h["bounced"] and ev["shot"] in ("forehand", "backhand"):
                    ev["shot"] = "volley"
                    ev["fixes"].append("ball_no_bounce")
                elif h["bounced"] and ev["shot"] == "volley" and ev.get("racket_side") in ("forehand", "backhand"):
                    ev["shot"] = ev["racket_side"]
                    ev["fixes"].append("ball_bounced")
            if ev["shot"] == "other" and h.get("bounced") is not None:  # the AI saw no hit; ball + sound say there was
                if not h["bounced"]:
                    ev["shot"] = "volley"
                elif ev.get("racket_side") in ("forehand", "backhand"):
                    ev["shot"] = ev["racket_side"]
                if ev["shot"] != "other":
                    ev["fixes"].append("tracked_hit_type")
            if missed_serve:
                ev["fixes"].append("second_serve_after_fault")
                ev["serve_number"] = 2
            if overhead_in_rally:
                ev["fixes"].append("serve_mid_rally")
            if a and a.get("found_hit") is False:
                ev["fixes"].append("ai_saw_no_hit")
            if not a and ev["shot"] == "other":
                unknown += 1
            _attribute(ev, seg, settings)
            events.append(ev)
        if unknown:
            warnings.append(f"AI 답도 공 궤적 단서도 없어 종류를 모르는 타구 {unknown}개는 ‘기타’로 뒀어.")

        result = {
            "version": VERSION, "prompt_variant": "hits", "serve_anchors": len(serve_ts),
            "file_id": fid, "file_name": self.f.get("name"),
            "analyzed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": self.model,
            "scope": "points", "source": "original",
            "analyzed_seconds": round(sum(s["end"] - s["start"] for s in segs), 1),
            "segment_count": len(segs), "chunk_count": len(batches), "failed_chunks": 0,
            "video_context": None,
            "overall_confidence": round(sum(e["confidence"] for e in events) / len(events)) if events else 0,
            "summary": aggregate(events, settings), "events": events, "rule_fixes": {},
            "observations": [], "warnings": warnings, "usage": usage,
            "hit_detection": tracked.get("stats"),
        }
        if measured.get("serves"):
            from analysis.serve_measure_job import apply_measurements
            apply_measurements(result, measured, settings)
        return result

    def _swings(self, hits, court, src, W, H, fid, warnings) -> dict[int, dict]:
        """Pose-measured forward swing of each hit's hitter (cached per hit), {index: result}."""
        from analysis.overlay_video import ensure_pose_weights, pose_ready
        if not pose_ready():
            warnings.append("ultralytics가 없어 포/백을 스윙으로 확인하지 못했어 (AI 판정만 씀).")
            return {}
        from analysis.swing_side import SwingJudge
        fp = self.cache_dir / "shots" / f"{fid}_swings.json"
        try:
            cache = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else {}
        except Exception:
            cache = {}
        judge, out = None, {}
        for k, h in enumerate(hits):
            self._check()
            key = f"{SWING_VERSION}:{h['t']:.2f}:{h['side']}"
            if key not in cache:
                self.report({"percent": round(5 + 45 * k / len(hits), 1),
                             "message": f"타구 {k + 1}/{len(hits)} 스윙 방향 측정 중 (포즈 인식)"})
                if judge is None:
                    root = Path(__file__).resolve().parents[1]
                    judge = SwingJudge(ensure_pose_weights(root, self.report), src, court, W, H)
                cache[key] = judge(h)
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            if cache[key]:
                out[k] = cache[key]
        return out

    @staticmethod
    def _seg(segs, h):
        return next((s for s in segs if s["start"] - 0.5 <= h["t"] <= s["end"] + 0.5), None) or {}

    def _request(self, batch, court, src, W, H, doubles, work):
        key = hashlib.sha1(json.dumps({
            "v": VERSION, "model": "", "src": src, "fps": CLIP_FPS,
            "hits": [(round(clip_time(h), 3), h["side"], h.get("note") or "") for h in batch],
        }, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        model, attempt = self.model, 0
        while True:
            cached = self.cache_dir / "shots" / f"hits_{model}_{key}.json"
            if cached.exists():
                try:
                    data = json.loads(cached.read_text(encoding="utf-8"))
                    return {"answers": {int(k): v for k, v in data["answers"].items()}, "usage": {}}, model
                except Exception:
                    pass
            attempt += 1
            try:
                self._check()
                res = classify(self.api_key, model, src, court, W, H, batch, work, doubles)
                break
            except gemini.GeminiError as exc:
                if exc.status == 499:
                    raise Cancelled()
                if exc.status in (400, 401, 403, 404):
                    raise
                if attempt >= 3 or getattr(exc, "daily_quota", False):
                    if exc.status in (429, 500, 503) and model != FALLBACK_MODEL:
                        model, attempt = FALLBACK_MODEL, 0
                        continue
                    raise
                for _ in range(int(10 * attempt * 2)):
                    self._check()
                    time.sleep(0.5)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps({"answers": res["answers"]}, ensure_ascii=False), encoding="utf-8")
        return res, model
