"""Dashboard export (1920x1080), SwingVision-style: the match footage top-left with the
scoreboard and a card for each shot on it, and the ball's path on the whole court around it:
a top view on the right (where it flew, where it landed, where it was hit), a side view and a
front view below (how high, over the net). The views are drawn by analysis/court_views.py from
the 3D ball arcs the serve measurement saves, 15 times a second, and piped straight into FFmpeg;
the footage itself is never decoded in Python. Look: analysis/render_style.py.
"""
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path

from core.media import exe, _run_ffmpeg_with_progress
from analysis.shot_analysis import extract_clip, _make_chunk, pick_source, Cancelled, _has_audio
from analysis import render_style as rs
from analysis.render_style import (ACCENT, ACCENT_INK, BAD, BG, DIR_KO, LINE, LINE_2, OK, SHOT_KO, SPIN_KO,
                                   SURFACE_3, TEAM, TEXT, TEXT_2, TEXT_3, rgba)

W, VH, FPS = 1344, 756, 30   # footage area of the 1920x1080 frame (analysis/court_views.py lays out the rest)
VIEW_FPS = 15                # court views redrawn this often (the ball dot moves smoothly enough)
POPUP_SEC = 1.6
POINT_PAD_BEFORE, POINT_PAD_AFTER = 1.5, 1.0


def ensure_pillow(report=None):
    try:
        return importlib.import_module("PIL")
    except ImportError:
        if report:
            report({"message": "Pillow 설치 중 (최초 1회)..."})
        subprocess.run([sys.executable, "-m", "pip", "install", "pillow", "--disable-pip-version-check"],
                       check=True, capture_output=True)
        importlib.invalidate_caches()
        return importlib.import_module("PIL")


Fonts = rs.Fonts


# ── Timeline ──────────────────────────────────────────────────────

def build_segments(project: dict, file_entry: dict, result: dict) -> list[dict]:
    fid = file_entry["id"]
    duration = float((file_entry.get("metadata") or {}).get("duration") or 0) or 1e9
    pts = sorted((p for p in project.get("points", []) if p.get("file_id") == fid), key=lambda p: p["start"])
    if pts:
        ranges = [(max(0.0, p["start"] - POINT_PAD_BEFORE), min(duration, p["end"] + POINT_PAD_AFTER), p) for p in pts]
    else:
        # No labels: one segment per burst of shots.
        ranges, cur = [], []
        for e in sorted(result.get("events", []), key=lambda e: e["t"]):
            if cur and e["t"] - cur[-1]["t"] > 6.0:
                ranges.append((max(0.0, cur[0]["t"] - 2.5), min(duration, cur[-1]["t"] + 2.0), None))
                cur = []
            cur.append(e)
        if cur:
            ranges.append((max(0.0, cur[0]["t"] - 2.5), min(duration, cur[-1]["t"] + 2.0), None))
    segs = []
    for a, b, p in ranges:
        if segs and a <= segs[-1]["end"]:
            segs[-1]["end"] = max(segs[-1]["end"], b)
            continue
        segs.append({"start": round(a, 3), "end": round(b, 3), "point": p})
    return [s for s in segs if s["end"] - s["start"] > 0.3]


def map_events(events: list[dict], chunk: dict) -> list[dict]:
    """Attach output-video time to every event that falls inside an exported segment."""
    out = []
    for e in sorted(events, key=lambda e: e["t"]):
        for s in chunk["segments"]:
            if s["start"] <= e["t"] < s["end"]:
                out.append({**e, "out_t": s["clip_start"] + (e["t"] - s["start"])})
                break
    for i, e in enumerate(out, 1):
        e["no"] = i
    return out


def court_side(point: dict | None) -> str | None:
    """deuce/ad from the score before the point (even points played → deuce court)."""
    if not point:
        return None
    sc = point.get("score_before") or {}
    if sc.get("tiebreak"):
        n = int(sc.get("tiebreak_a") or 0) + int(sc.get("tiebreak_b") or 0)
    else:
        idx = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}
        n = idx.get(str(sc.get("point_a")), 0) + idx.get(str(sc.get("point_b")), 0)
    return "deuce" if n % 2 == 0 else "ad"


# ── Drawing ───────────────────────────────────────────────────────

def render_popup(e: dict, settings: dict, fonts):
    """Card naming the shot just hit, top-left over the footage."""
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (W, VH), (0, 0, 0, 0))
    _draw_popup(ImageDraw.Draw(img), e, settings, fonts)
    return img


def _draw_popup(d, e: dict, settings: dict, fonts):
    if e["shot"] == "serve":
        head = ["", "1st ", "2nd "][e.get("serve_number") or 0] + "서브"
        tail = " · ".join(x for x in (SPIN_KO.get(e.get("serve_type")), DIR_KO.get(e.get("serve_direction"))) if x)
    else:
        head = SHOT_KO[e["shot"]]
        tail = SPIN_KO.get(e.get("spin"), "")
    who, team = _who(e, settings)
    call = {"out": "아웃", "net": "네트"}.get(e.get("serve_call"))
    outcome = call or {"ace": "에이스", "winner": "위너", "error": "에러", "fault": "폴트"}.get(e.get("result"))
    f_no, f_head, f_tail, f_meta, f_spd = (fonts.get("SemiBold", 22), fonts.get("ExtraBold", 34),
                                           fonts.get("Medium", 22), fonts.get("Medium", 20), fonts.get("Bold", 20))
    no = f"#{e['no']}"
    spd = f"{e['speed_kmh']} km/h" if e.get("speed_kmh") else None
    row1 = d.textlength(no, font=f_no) + 12 + d.textlength(head, font=f_head)
    if tail:
        row1 += d.textlength("   " + tail, font=f_tail)
    if spd:
        row1 += 20 + d.textlength(spd, font=f_spd) + 24
    meta = [m for m in (who, outcome) if m]
    row2 = (18 if team else 0) + sum(d.textlength(m, font=f_meta) for m in meta) + 32 * max(0, len(meta) - 1)
    bw, bh = int(max(row1, row2) + 44), (104 if meta else 70)
    x, y = 28, 28
    d.rounded_rectangle([x + 2, y + 6, x + bw + 2, y + bh + 6], radius=16, fill=(0, 0, 0, 90))
    d.rounded_rectangle([x, y, x + bw, y + bh], radius=16, fill=rgba(BG, 232), outline=rgba(LINE_2), width=1)
    cx = x + 22
    d.text((cx, y + 22), no, font=f_no, fill=rgba(TEXT_3))
    cx += d.textlength(no, font=f_no) + 12
    d.text((cx, y + 12), head, font=f_head, fill=rgba(TEXT))
    cx += d.textlength(head, font=f_head)
    if tail:
        d.text((cx, y + 22), "   " + tail, font=f_tail, fill=rgba(TEXT_2))
        cx += d.textlength("   " + tail, font=f_tail)
    if spd:
        sx = cx + 20
        sw = d.textlength(spd, font=f_spd) + 24
        measured = e.get("speed_source") == "measured"
        d.rounded_rectangle([sx, y + 18, sx + sw, y + 52], radius=17, fill=rgba(ACCENT if measured else SURFACE_3))
        d.text((sx + 12, y + 22), spd, font=f_spd, fill=rgba(ACCENT_INK if measured else TEXT))
    if meta:
        mx = x + 22
        if team:
            d.ellipse([mx, y + 73, mx + 10, y + 83], fill=rgba(TEAM[team]))
            mx += 18
        for m in meta:
            color = TEXT_2 if m == who else (OK if m in ("에이스", "위너") else BAD)
            d.text((mx, y + 64), m, font=f_meta, fill=rgba(color))
            mx += d.textlength(m, font=f_meta) + 32


def _who(e, settings):
    """(label, team letter or None) for the player or side that hit the shot."""
    team = (e.get("team") or (e.get("player") or " ")[0]).upper()
    team = team if team in TEAM else None
    if e.get("player"):
        pid = e["player"]
        names = settings.get("team_a" if pid[0] == "A" else "team_b") or []
        try:
            name = names[int(pid[1:]) - 1]
        except (ValueError, IndexError):
            name = ""
        return (name or pid), team
    if team:
        names = " · ".join(x for x in settings.get(f"team_{team.lower()}") or [] if x)
        return (names or f"Team {team}"), team
    return {"near": "카메라 쪽", "far": "먼 쪽"}.get(e.get("side"), ""), None


def render_overlay(point: dict | None, e: dict | None, settings: dict, fonts):
    """What sits on the footage at a moment: the scoreboard of the point (top-right) and the
    card of the shot just hit (top-left), on a transparent frame."""
    from PIL import Image, ImageDraw
    from analysis import scoreboard
    img = scoreboard.render(point, settings, (W, VH), fonts, corner="tr") if point else Image.new("RGBA", (W, VH), (0, 0, 0, 0))
    if e is not None:
        _draw_popup(ImageDraw.Draw(img), e, settings, fonts)
    return img


def render_tally(done: list[dict], fonts):
    """The shot count line over the top view (BGR patch)."""
    import numpy as np
    from PIL import Image, ImageDraw
    from analysis.court_views import TOP
    counts = {}
    for e in done:
        counts[e["shot"]] = counts.get(e["shot"], 0) + 1
    img = Image.new("RGB", (TOP["w"], 44), BG)
    d = ImageDraw.Draw(img)
    big, small = fonts.get("ExtraBold", 30), fonts.get("Medium", 17)
    x = 0
    d.text((x, 2), str(len(done)), font=big, fill=TEXT)
    x += d.textlength(str(len(done)), font=big) + 6
    d.text((x, 14), "타구", font=small, fill=TEXT_3)
    x += d.textlength("타구", font=small) + 22
    for k in ("forehand", "backhand", "volley", "serve", "smash", "lob"):
        if not counts.get(k):
            continue
        part = f"{SHOT_KO[k]} {counts[k]}"
        if x + d.textlength(part, font=small) > TOP["w"]:
            break
        d.text((x, 14), part, font=small, fill=TEXT_2)
        x += d.textlength(part, font=small) + 16
    return np.array(img)[:, :, ::-1].copy()


def _ffconcat(path: Path, entries: list[tuple[str, float]]):
    lines = ["ffconcat version 1.0"]
    for name, dur in entries:
        lines += [f"file '{name}'", f"duration {dur:.4f}"]
    lines.append(f"file '{entries[-1][0]}'")  # concat demuxer ignores the last duration otherwise
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _overlay(clip, chunk, src, src_kind, file_entry, root, work, has_audio, ball, pose, report, cancelled):
    """Ball trail / skeleton pass over the footage (35-65%). Returns the annotated clip and notes."""
    from analysis import overlay_video as ov
    from analysis.serve_speed import Court
    notes = []
    cal = (file_entry.get("calibration") or {}).get("points") or []
    if pose and len(cal) < 4:
        notes.append("코트 캘리브레이션이 없어 스켈레톤은 건너뜀")
        pose = False
    n_frames = int(round(chunk["clip_duration"] * ov.FPS)) + 2
    track = people = None
    if ball:
        from analysis.tracknet import BallDetector, active_weights
        report({"percent": 35.0, "message": "공 궤적 추적 중..."})
        det = BallDetector(active_weights(root, report))
        track = ov.ball_tracks(clip, n_frames, det, cancelled=cancelled, court=Court(cal) if len(cal) >= 4 else None,
                               report=lambda f: report({"percent": 35 + 8 * f, "message": "공 궤적 추적 중"}))
        del det
    if pose:
        if not ov.pose_ready():
            notes.append("ultralytics가 없어 스켈레톤은 건너뜀 (AI 기능 설치 필요)")
        else:
            from ultralytics import YOLO
            report({"percent": 43.0, "message": "선수 자세 추정 중..."})
            model = YOLO(str(ov.ensure_pose_weights(root, report)))
            meta = file_entry.get("metadata") or {}
            w, h = (int(meta.get("width") or 1920), int(meta.get("height") or 1080))
            if src_kind == "proxy":
                w, h = 1280, int(round(1280 * h / max(1, w))) // 2 * 2
            people = ov.poses(src, chunk["segments"], Court(cal), model, w, h, n_frames, cancelled=cancelled,
                              report=lambda f: report({"percent": 43 + 17 * f, "message": "선수 자세 추정 중"}))
            del model
    out = work / "footage_overlay.mp4"
    report({"percent": 60.0, "message": "공 궤적·스켈레톤 그리는 중..."})
    ov.annotate(clip, out, track, people, has_audio, show_pose=people is not None, show_ball=track is not None,
                cancelled=cancelled, report=lambda f: report({"percent": 60 + 5 * f, "message": "공 궤적·스켈레톤 그리는 중"}))
    return out, notes


def _ball_paths(root: Path, fid: str, chunk: dict, events: list[dict]):
    """Per point id: the ball's path (analysis/court_views.py) from the cached measurement."""
    import json
    from analysis import court_views as cv
    cache = Path(root or ".") / "cache"
    try:
        hits = json.loads((cache / f"{fid}_hits.json").read_text(encoding="utf-8"))
    except Exception:
        hits = {}
    try:
        serves = json.loads((cache / f"{fid}_serves.json").read_text(encoding="utf-8")).get("serves", [])
    except Exception:
        serves = []
    arcs = hits.get("arcs") or {}
    paths = {}
    for s in chunk["segments"]:
        p = s.get("point")
        if not p:
            continue
        sv = [x for x in serves if p["start"] - 1.6 <= x["contact_t"] <= p["end"] + 0.5]
        hit_times = [(e["t"], e.get("side")) for e in events
                     if e.get("point_id") == p["id"] and e["shot"] != "serve"]
        paths[p["id"]] = cv.build_path(arcs.get(p["id"]) or [], sv, hit_times)
    return paths, bool(arcs)


def _team_of_side(point: dict | None) -> dict:
    a_end = ((point or {}).get("score_before") or {}).get("team_a_end")
    if a_end not in ("near", "far"):
        return {}
    other = "far" if a_end == "near" else "near"
    return {a_end: "A", other: "B"}


def export_dashboard(project: dict, file_entry: dict, result: dict, out_path: Path, work: Path,
                     quality: str = "auto", report=None, cancelled=None, root: Path | None = None,
                     ball: bool = False, pose: bool = False) -> dict:
    """1920x1080: footage top-left with the scoreboard and shot cards on it; top view of the whole
    court on the right; side and front views below. The views follow the ball frame by frame."""
    import numpy as np
    from analysis import court_views as cv
    report = report or (lambda d: None)
    cancelled = cancelled or (lambda: False)
    ensure_pillow(report)
    settings = project.get("settings") or {}

    segs = build_segments(project, file_entry, result)
    if not segs:
        raise RuntimeError("내보낼 구간이 없어. 샷 분석 결과에 타구가 없거나 포인트가 없어.")
    total_src = sum(s["end"] - s["start"] for s in segs)
    src, src_kind = pick_source(file_entry, quality, total_src)
    has_audio = bool((file_entry.get("metadata") or {}).get("audio_codec"))
    if src_kind == "proxy":
        has_audio = has_audio and _has_audio(src)

    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    try:
        # 1) footage: labeled points back to back (0-55%)
        chunk = _make_chunk([{"start": s["start"], "end": s["end"], "point": s["point"]} for s in segs])
        clip = work / "footage.mp4"
        report({"percent": 1.0, "message": f"영상 구간 {len(segs)}개 추출 중 ({'원본' if src_kind == 'original' else '프록시'})"})
        extract_clip(src, chunk, clip, has_audio, crf=18, cancelled=cancelled,
                     progress=lambda p: report({"percent": p * (0.3 if ball or pose else 0.5),
                                                 "message": f"영상 구간 {len(segs)}개 추출 중"}))
        duration = chunk["clip_duration"]
        notes = []
        if ball or pose:
            clip, notes = _overlay(clip, chunk, src, src_kind, file_entry, root, work, has_audio,
                                   ball, pose, report, cancelled)

        # 2) what sits on the footage: scoreboard per point + shot cards (55%)
        report({"percent": 55.0, "message": "스코어·타구 카드 그리는 중..."})
        events = map_events(result.get("events", []), chunk)
        fonts = Fonts()
        marks = {0.0, duration}
        for s in chunk["segments"]:
            marks |= {s["clip_start"], s["clip_end"]}
        for e in events:
            marks |= {e["out_t"], min(duration, e["out_t"] + POPUP_SEC)}
        marks = sorted(m for m in marks if 0.0 <= m <= duration)
        entries, cache_imgs = [], {}
        for a, b in zip(marks, marks[1:]):
            if b - a < 0.5 / FPS:
                continue
            mid = (a + b) / 2
            seg = next((s for s in chunk["segments"] if s["clip_start"] <= mid < s["clip_end"]), None)
            point = seg.get("point") if seg else None
            card = next((e for e in reversed(events) if e["out_t"] <= mid < e["out_t"] + POPUP_SEC), None)
            key = ((point or {}).get("id"), card["no"] if card else None)
            if key not in cache_imgs:
                name = f"ov_{len(cache_imgs):04d}.png"
                render_overlay(point, card, settings, fonts).save(work / name)
                cache_imgs[key] = name
            entries.append((cache_imgs[key], b - a))
        _ffconcat(work / "overlay.txt", entries or [(render_overlay(None, None, settings, fonts).save(work / "ov_blank.png") or "ov_blank.png", duration)])

        # 3) court views, drawn frame by frame and piped into the compose (55-100%)
        paths, have_arcs = _ball_paths(root, file_entry["id"], chunk, events)
        if not have_arcs:
            notes.append("랠리 공 궤적은 서브 실측을 다시 돌리면 나와요 (지금은 서브만 그려요)")
        bg = cv.background(fonts)
        order = [s["point"]["id"] for s in chunk["segments"] if s.get("point")]
        n_frames = int(np.ceil(duration * VIEW_FPS)) + 1
        ffmpeg = exe("ffmpeg")
        fc = (f"[1:v]scale={W}:{VH}:force_original_aspect_ratio=decrease,pad={W}:{VH}:(ow-iw)/2:(oh-ih)/2,"
              f"setsar=1,fps={FPS}[v];[2:v]fps={FPS},format=rgba[o];[v][o]overlay=0:0:eof_action=pass[vo];"
              f"[0:v]fps={FPS}[bg];[bg][vo]overlay=0:0:eof_action=pass,format=yuv420p[out]")
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{cv.OUT_W}x{cv.OUT_H}", "-framerate", str(VIEW_FPS), "-i", "-",
               "-i", str(clip), "-f", "concat", "-safe", "0", "-i", str(work / "overlay.txt"),
               "-filter_complex", fc, "-map", "[out]"]
        if has_audio:
            cmd += ["-map", "1:a:0", "-c:a", "aac", "-b:a", "128k"]
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-t", f"{duration:.3f}",
                "-movflags", "+faststart", str(out_path)]
        err = (work / "compose.log").open("wb")
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err)
        tally_key, tally = None, None
        try:
            for k in range(n_frames):
                if cancelled():
                    raise Cancelled()
                tau = k / VIEW_FPS
                seg = next((s for s in chunk["segments"] if s["clip_start"] <= tau < s["clip_end"]), chunk["segments"][-1])
                t = seg["start"] + (tau - seg["clip_start"])
                point = seg.get("point")
                pid = (point or {}).get("id")
                done = [e for e in events if e["out_t"] <= tau]
                shot = next((e["t"] for e in reversed(done) if e.get("point_id") == pid), None)
                history = [b for q in order[:order.index(pid)] if pid in order for b in paths.get(q, cv.PointPath([], [], [])).landings]
                frame = cv.draw_dynamic(bg.copy(), paths.get(pid), t, shot, history, _team_of_side(point))
                if tally_key != len(done):
                    tally_key, tally = len(done), render_tally(done, fonts)
                frame[cv.TOP["y"] - 96:cv.TOP["y"] - 52, cv.TOP["x"]:cv.TOP["x"] + cv.TOP["w"]] = tally
                proc.stdin.write(frame.tobytes())
                if k % 15 == 0:
                    report({"percent": 55 + 45 * k / max(1, n_frames), "message": "코트 뷰 그리며 합성 중"})
            proc.stdin.close()
            rc = proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        finally:
            err.close()
        if rc != 0 or not out_path.exists():
            raise RuntimeError(f"대시보드 영상 합성 실패: {(work / 'compose.log').read_text(errors='replace')[-600:]}")
        return {"path": str(out_path), "duration": duration, "events": len(events), "segments": len(segs),
                "notes": notes}
    finally:
        shutil.rmtree(work, ignore_errors=True)
