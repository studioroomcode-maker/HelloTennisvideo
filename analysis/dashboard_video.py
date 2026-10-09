"""Dashboard export: match footage on top, live stats panel underneath (1280x1280).

The panel follows the app's look (analysis/render_style.py): near-black surface, Korean
labels, one accent for what is measured or newest. Top row: shots so far · forehand vs
backhand · speed of the latest shot. Bottom: speed of each shot (serves in the accent) and
where the serves landed in the receiver's boxes. The panel only changes when a shot happens,
so one PNG is drawn per shot and FFmpeg holds each until the next; a pop-up card names each
shot as it is hit. The video itself is never decoded in Python.
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

W, VH, PH, FPS = 1280, 720, 560, 30
TOP_H = 290
PAD = 32
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

def _avg(vals):
    vals = [v for v in vals if v]
    return round(sum(vals) / len(vals)) if vals else None


def _label(d, xy, text, fonts):
    d.text(xy, text, font=fonts.get("SemiBold", 21), fill=TEXT_2)


def _mix(c, toward, f):
    return tuple(int(a * (1 - f) + b * f) for a, b in zip(c, toward))


def render_panel(done: list[dict], fonts, side_of_point: dict, serves_only: bool = False):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (W, PH), BG)
    d = ImageDraw.Draw(img)
    big_y = 70
    col = (W - 2 * PAD) / 3
    xs = [PAD + i * col for i in range(3)]
    for x in xs[1:]:
        d.line([x - 18, PAD, x - 18, TOP_H - PAD + 4], fill=LINE, width=1)
    d.line([PAD, TOP_H, W - PAD, TOP_H], fill=LINE, width=1)
    sub = fonts.get("Medium", 19)
    big = fonts.get("ExtraBold", 104)
    foot = TOP_H - PAD - 24

    counts = {}
    for e in done:
        counts[e["shot"]] = counts.get(e["shot"], 0) + 1
    last = done[-1] if done else None

    # 1) shots so far
    x = xs[0]
    _label(d, (x, PAD), "타구", fonts)
    d.text((x, big_y), str(len(done)), font=big, fill=TEXT)
    extra = [f"{SHOT_KO[k]} {counts[k]}" for k in ("serve", "volley", "smash", "lob", "dropshot", "other") if counts.get(k)]
    d.text((x, foot), " · ".join(extra) or "아직 없음", font=sub, fill=TEXT_3)

    # 2) forehand vs backhand
    x = xs[1]
    _label(d, (x, PAD), "포핸드 · 백핸드", fonts)
    fh, bh = counts.get("forehand", 0), counts.get("backhand", 0)
    if serves_only:
        d.text((x, big_y), "--", font=big, fill=TEXT_3)
        d.text((x, foot), "타구 판정을 돌리면 나와요", font=sub, fill=TEXT_3)
    else:
        d.text((x, big_y), str(fh), font=big, fill=TEXT)
        wx = x + d.textlength(str(fh), font=big)
        d.text((wx + 16, big_y + 30), "·", font=fonts.get("Bold", 56), fill=TEXT_3)
        d.text((wx + 46, big_y), str(bh), font=big, fill=TEXT_2)
        bw = col - 52
        by = foot - 22
        d.rounded_rectangle([x, by, x + bw, by + 8], radius=4, fill=SURFACE_3)
        if fh + bh:
            d.rounded_rectangle([x, by, x + max(8, bw * fh / (fh + bh)), by + 8], radius=4, fill=TEXT)
            d.text((x, foot), f"포핸드 {round(100 * fh / (fh + bh))}%", font=sub, fill=TEXT_3)

    # 3) latest shot speed
    x = xs[2]
    _label(d, (x, PAD), "타구 속도", fonts)
    sp = last.get("speed_kmh") if last else None
    measured = bool(last and last.get("speed_source") == "measured")
    num = str(sp) if sp else "--"
    d.text((x, big_y), num, font=big, fill=ACCENT if measured else (TEXT if sp else TEXT_3))
    d.text((x + d.textlength(num, font=big) + 10, big_y + 66), "km/h", font=fonts.get("SemiBold", 24), fill=TEXT_2)
    if last:
        if last["shot"] == "serve":
            parts = [["", "1st ", "2nd "][last.get("serve_number") or 0] + "서브",
                     SPIN_KO.get(last.get("serve_type"), ""), "실측" if measured else ""]
        else:
            parts = [SHOT_KO[last["shot"]], "추정" if sp else ""]
        d.text((x, foot - 26), " · ".join(p for p in parts if p), font=sub, fill=TEXT_2)
    speeds = [e.get("speed_kmh") for e in done if e.get("speed_kmh")]
    srv = _avg(e.get("speed_kmh") for e in done if e["shot"] == "serve")
    stats = [f"최고 {max(speeds)}" if speeds else "", f"서브 평균 {srv}" if srv else ""]
    d.text((x, foot), " · ".join(s for s in stats if s) or " ", font=sub, fill=TEXT_3)

    _speed_history(d, done, fonts)
    _placement(d, done, fonts, side_of_point)
    return img


def _speed_history(d, done, fonts):
    top0 = TOP_H + 26
    _label(d, (PAD, top0), "타구별 속도", fonts)
    tiny = fonts.get("Medium", 15)
    shots = [e for e in done if e.get("speed_kmh")][-24:]
    left, right, top, bottom = PAD + 40, 612, top0 + 54, PH - 44
    top_val = max([e["speed_kmh"] for e in shots] or [0])
    scale = max(60, int(top_val * 1.15 / 20 + 1) * 20) if top_val else 100
    for v in (0, scale // 2, scale):
        y = bottom - (bottom - top) * v / scale
        d.line([left, y, right, y], fill=LINE_2 if v == 0 else LINE, width=1)
        d.text((left - 10 - d.textlength(str(v), font=tiny), y - 9), str(v), font=tiny, fill=TEXT_3)
    # legend
    lx = right - d.textlength("서브", font=tiny) - d.textlength("스트로크", font=tiny) - 60
    for color, name in ((ACCENT, "서브"), (TEXT_2, "스트로크")):
        d.rounded_rectangle([lx, top0 + 8, lx + 10, top0 + 18], radius=2, fill=color)
        d.text((lx + 16, top0 + 2), name, font=tiny, fill=TEXT_3)
        lx += 16 + d.textlength(name, font=tiny) + 18
    if not shots:
        d.text((left + 12, (top + bottom) / 2 - 10), "속도가 잡힌 타구가 아직 없어요", font=fonts.get("Medium", 18), fill=TEXT_3)
        return
    slot = (right - left - 8) / max(len(shots), 10)
    bw = min(30, slot * 0.66)
    for e in shots:
        i = shots.index(e)
        cx = left + 8 + slot * (i + 0.5)
        h = max(3, (bottom - top) * min(e["speed_kmh"], scale) / scale)
        latest = e is shots[-1]
        color = ACCENT if e["shot"] == "serve" else TEXT_2
        if not latest:
            color = _mix(color, BG, 0.35)
        d.rounded_rectangle([cx - bw / 2, bottom - h, cx + bw / 2, bottom], radius=3, fill=color)
        if latest:
            val = str(e["speed_kmh"])
            f = fonts.get("Bold", 16)
            d.text((cx - d.textlength(val, font=f) / 2, bottom - h - 24), val, font=f, fill=TEXT)
        lab = str(e["no"])
        d.text((cx - d.textlength(lab, font=tiny) / 2, bottom + 6), lab, font=tiny, fill=TEXT_3)


def _placement(d, done, fonts, side_of_point):
    """Serve landing spots in the receiver's service boxes, seen from the server (net at the bottom)."""
    x0 = 660
    top0 = TOP_H + 26
    _label(d, (x0, top0), "서브 위치", fonts)
    tiny = fonts.get("Medium", 15)
    L, R, T, B = 700, W - PAD - 40, top0 + 54, PH - 40
    mid = (L + R) / 2
    d.rectangle([L, T, R, B], outline=LINE_2, width=2)
    d.line([mid, T, mid, B], fill=LINE_2, width=2)
    d.line([L - 14, B, R + 14, B], fill=TEXT_3, width=4)
    d.text((L + 10, T + 8), "듀스", font=tiny, fill=TEXT_3)
    d.text((R - 10 - d.textlength("애드", font=tiny), T + 8), "애드", font=tiny, fill=TEXT_3)
    d.text((R + 8, B + 6), "네트", font=tiny, fill=TEXT_3)
    lx = R + 30 - d.textlength("최근", font=tiny) - d.textlength("폴트", font=tiny) - 48
    d.ellipse([lx, top0 + 7, lx + 11, top0 + 18], fill=ACCENT)
    d.text((lx + 16, top0 + 2), "최근", font=tiny, fill=TEXT_3)
    lx += 16 + d.textlength("최근", font=tiny) + 16
    d.ellipse([lx, top0 + 7, lx + 11, top0 + 18], outline=TEXT_3, width=2)
    d.text((lx + 16, top0 + 2), "폴트", font=tiny, fill=TEXT_3)
    serves = [e for e in done if e["shot"] == "serve"
              and (e.get("bounce_court") or e.get("serve_direction") in ("T", "body", "wide"))][-24:]
    bw = mid - L
    num = fonts.get("Bold", 15)
    for e in serves:
        if e.get("bounce_court"):
            cx, cy = e["bounce_court"]
            lx = (10.97 - cx) if e.get("server_end") == "far" else cx
            depth = abs(cy - 11.885)
            x = L + (lx - 1.37) / (9.60 - 1.37) * (R - L)
            y = B - depth / 6.40 * (B - T)
            x, y = min(max(x, L - 30), R + 30), min(max(y, top0 + 40), B + 8)
        else:
            box = side_of_point.get(e.get("point_id")) or "deuce"
            frac = {"T": 0.84, "body": 0.5, "wide": 0.16}[e["serve_direction"]]
            x = L + bw * frac if box == "deuce" else R - bw * frac
            x += ((e["no"] * 53) % 5 - 2) * 7
            y = T + (B - T) * 0.32 + ((e["no"] * 37) % 5 - 2) * 9
        latest = e is serves[-1]
        fault = e.get("result") == "fault" or e.get("serve_call") in ("out", "net")
        r = 14
        if latest:
            d.ellipse([x - r, y - r, x + r, y + r], fill=ACCENT)
            ink = ACCENT_INK
        elif fault:
            d.ellipse([x - r, y - r, x + r, y + r], fill=BG, outline=TEXT_3, width=2)
            ink = TEXT_3
        else:
            d.ellipse([x - r, y - r, x + r, y + r], fill=SURFACE_3, outline=LINE_2, width=1)
            ink = TEXT
        lab = str(e["no"])
        d.text((x - d.textlength(lab, font=num) / 2, y - 10), lab, font=num, fill=ink)


def render_popup(e: dict, settings: dict, fonts):
    """Card naming the shot just hit, top-left over the footage."""
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (W, VH), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
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
        d.rounded_rectangle([sx, y + 18, sx + sw, y + 52], radius=17, fill=rgba(ACCENT if measured else (35, 40, 45)))
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
    return img


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


# ── Export ────────────────────────────────────────────────────────

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


def export_dashboard(project: dict, file_entry: dict, result: dict, out_path: Path, work: Path,
                     quality: str = "auto", report=None, cancelled=None, root: Path | None = None,
                     ball: bool = False, pose: bool = False) -> dict:
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
        # 1) footage: labeled points back to back (0-65%)
        chunk = _make_chunk([{"start": s["start"], "end": s["end"], "point": s["point"]} for s in segs])
        clip = work / "footage.mp4"
        report({"percent": 1.0, "message": f"영상 구간 {len(segs)}개 추출 중 ({'원본' if src_kind == 'original' else '프록시'})"})
        extract_clip(src, chunk, clip, has_audio, crf=18, cancelled=cancelled,
                     progress=lambda p: report({"percent": p * (0.35 if ball or pose else 0.65),
                                                 "message": f"영상 구간 {len(segs)}개 추출 중"}))
        duration = chunk["clip_duration"]
        notes = []
        if ball or pose:
            clip, notes = _overlay(clip, chunk, src, src_kind, file_entry, root, work, has_audio,
                                   ball, pose, report, cancelled)

        # 2) panels + popups (65-70%)
        report({"percent": 65.0, "message": "대시보드 그리는 중..."})
        events = map_events(result.get("events", []), chunk)
        point_side = {}
        for s in chunk["segments"]:
            if s.get("point"):
                point_side[s["point"]["id"]] = court_side(s["point"])
        fonts = Fonts()
        from PIL import Image
        Image.new("RGBA", (W, VH), (0, 0, 0, 0)).save(work / "blank.png")
        panels, pops = [], []
        t_prev, p_prev = 0.0, "blank.png"
        serves_only = bool(result.get("serves_only"))
        render_panel([], fonts, point_side, serves_only).save(work / "panel_000.png")
        panel_prev = "panel_000.png"
        for i, e in enumerate(events, 1):
            if cancelled():
                raise Cancelled()
            t = min(max(e["out_t"], 0.0), duration)
            panels.append((panel_prev, t - t_prev))
            panel_prev = f"panel_{i:03d}.png"
            render_panel(events[:i], fonts, point_side, serves_only).save(work / panel_prev)
            # popup timeline: previous popup ends at this shot at the latest
            pops.append((p_prev, t - t_prev))
            p_prev = f"pop_{i:03d}.png"
            render_popup(e, settings, fonts).save(work / p_prev)
            t_prev = t
        panels.append((panel_prev, duration - t_prev))
        # popups last POPUP_SEC, then blank until the next shot
        pop_entries = []
        for (name, dur) in pops + [(p_prev, duration - t_prev)]:
            if name == "blank.png" or dur <= POPUP_SEC:
                pop_entries.append((name, dur))
            else:
                pop_entries += [(name, POPUP_SEC), ("blank.png", dur - POPUP_SEC)]
        min_d = 1.0 / FPS / 2
        panels = [(n, d) for n, d in panels if d > min_d] or [("panel_000.png", duration)]
        pop_entries = [(n, d) for n, d in pop_entries if d > min_d] or [("blank.png", duration)]
        _ffconcat(work / "panels.txt", panels)
        _ffconcat(work / "pops.txt", pop_entries)

        # 3) compose (70-100%)
        n_frames = int(round(duration * FPS))
        base = (f"[0:v]scale={W}:{VH}:force_original_aspect_ratio=decrease,pad={W}:{VH}:(ow-iw)/2:(oh-ih)/2,"
                f"setsar=1,fps={FPS}[v];[1:v]fps={FPS},format=rgba[p];[v][p]overlay=0:0:eof_action=pass[vo];"
                f"[2:v]fps={FPS},scale={W}:{PH},format=rgb24[pn];[vo][pn]vstack=inputs=2")
        variants = [base + ",format=yuv420p[out]"]
        ffmpeg = exe("ffmpeg")
        tail = ""
        for fc in variants:
            cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                   "-i", str(clip),
                   "-f", "concat", "-safe", "0", "-i", str(work / "pops.txt"),
                   "-f", "concat", "-safe", "0", "-i", str(work / "panels.txt"),
                   "-filter_complex", fc, "-map", "[out]"]
            if has_audio:
                cmd += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "128k"]
            cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-t", f"{duration:.3f}",
                    "-movflags", "+faststart", "-progress", "pipe:1", str(out_path)]

            def on_prog(info):
                if info.get("progress") != "heartbeat":
                    report({"percent": 70 + 0.3 * float(info.get("percent") or 0),
                            "message": "대시보드 영상 합성 중"})

            rc, tail = _run_ffmpeg_with_progress(cmd, duration, progress_callback=on_prog,
                                                 attempt_name="대시보드 합성", startup_timeout=120,
                                                 stall_timeout=120, should_cancel=cancelled)
            if cancelled():
                raise Cancelled()
            if rc == 0 and out_path.exists():
                break
        else:
            raise RuntimeError(f"대시보드 영상 합성 실패: {tail[-600:]}")
        return {"path": str(out_path), "duration": duration, "events": len(events), "segments": len(segs),
                "notes": notes}
    finally:
        shutil.rmtree(work, ignore_errors=True)
