"""Ball trail + player skeleton overlay for the dashboard footage.

* Ball: TrackNet on the exported 720p footage (30 fps, what it was trained on),
  static blobs removed and detections linked into tracks; only tracks of a few
  frames or more are drawn, as a fading trail with a ring on the current ball.
* Skeleton: YOLO pose on the 4K source. Far-court players are only ~60 px tall,
  so a second pass runs on a native-resolution crop of the far court. People
  whose feet aren't on (or right around) the calibrated court are dropped, which
  removes the neighbouring courts and benches.
"""
from __future__ import annotations

import math
import subprocess
import urllib.request
from pathlib import Path

from core.media import exe
from analysis.shot_analysis import Cancelled

OUT_W, OUT_H, FPS = 1280, 720, 30
POSE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m-pose.pt"
POSE_SIZE_MIN = 30_000_000
TRAIL = 14
MIN_TRACK = 6
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)]


def ensure_pose_weights(root: Path, report=None) -> Path:
    path = Path(root) / "models" / "yolo11m-pose.pt"
    if path.exists() and path.stat().st_size >= POSE_SIZE_MIN:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if report:
        report({"message": "자세 추정 모델(YOLO pose, 40MB) 다운로드 중 (최초 1회)..."})
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(POSE_URL, timeout=120) as resp, tmp.open("wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    if tmp.stat().st_size < POSE_SIZE_MIN:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("자세 추정 모델 다운로드가 불완전해.")
    tmp.replace(path)
    return path


def _frames(cmd, w, h):
    """Yield raw BGR frames from an ffmpeg command writing rawvideo to stdout."""
    import numpy as np
    size = w * h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


# ── Ball ─────────────────────────────────────────────────────────

def ball_tracks(footage: Path, n_frames: int, detector, report=None, cancelled=None, court=None):
    """Per output frame: (x, y) in 1280x720 of a linked ball, or None.

    With the court calibration, only balls of this court's game are kept (not the next
    court's, nor ones bouncing behind the fence)."""
    from analysis.serve_speed import heat_to_blobs, remove_static, link_tracks
    cmd = [exe("ffmpeg"), "-nostdin", "-loglevel", "error", "-i", str(footage),
           "-vf", "scale=640:360:flags=area", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    blobs, hist = [], []
    batch = []

    def flush(final=False):
        nonlocal hist
        if len(batch) < 1:
            return
        frames = hist + batch
        heats = detector.heat_levels(frames)[len(hist):]
        blobs.extend(heat_to_blobs(h) for h in heats)
        hist = frames[-2:]
        batch.clear()

    for i, fr in enumerate(_frames(cmd, 640, 360)):
        if cancelled and cancelled():
            raise Cancelled()
        batch.append(fr)
        if len(batch) >= 240:
            flush()
            if report:
                report(i / max(1, n_frames))
    flush()
    # first two frames have no history in TrackNet; heat_levels returns None for them
    times = [i / FPS for i in range(len(blobs))]
    tracks = link_tracks(remove_static(blobs, FPS), times, max_gap=4)
    per_frame = [None] * len(blobs)
    if court is not None:
        from analysis.hit_detect import own_court
        tracks = [tr for tr in tracks if own_court(tr, court)]
    for tr in tracks:
        if len(tr) < MIN_TRACK:
            continue
        for t, x, y in tr:
            per_frame[int(round(t * FPS))] = (x * 2.0, y * 2.0)
    return per_frame


# ── Pose ─────────────────────────────────────────────────────────

def far_crop(court) -> tuple[float, float, float, float]:
    """Normalized crop around the far half of the court (players behind the baseline too)."""
    pts = [court.to_image(x, y) for x, y in ((-1.5, 11.0), (12.5, 11.0), (12.5, 27.5), (-1.5, 27.5))]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1 = max(0.0, min(xs) - 0.02), min(1.0, max(xs) + 0.02)
    # bodies stand up from the ground: extend upward by ~ a player's height
    y0, y1 = max(0.0, min(ys) - 0.10), min(1.0, max(ys) + 0.02)
    return x0, x1, y0, y1


def _on_court(court, kps, box) -> bool:
    ank = [k for k in (kps[15], kps[16]) if k[2] > 0.3]
    if ank:
        fx, fy = sum(k[0] for k in ank) / len(ank), sum(k[1] for k in ank) / len(ank)
    else:
        fx, fy = (box[0] + box[2]) / 2, box[3]
    cx, cy = court.to_court(fx, fy)
    return -2.5 <= cx <= 13.5 and -5.0 <= cy <= 28.8


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _people(rt, rb, court, fw, fh, src_w, src_h, cx0, cy0):
    """Merge the full-frame and far-crop detections of one frame into on-court people."""
    people = []
    for res, sx, sy, ox, oy in ((rt, 1 / fw, 1 / fh, 0.0, 0.0), (rb, 1 / src_w, 1 / src_h, cx0, cy0)):
        if res.keypoints is None:
            continue
        for kp, b, c in zip(res.keypoints.data.cpu().numpy(), res.boxes.xyxy.cpu().numpy(),
                            res.boxes.conf.cpu().numpy()):
            kps = [(float(p[0]) * sx + ox, float(p[1]) * sy + oy, float(p[2])) for p in kp]
            box = (b[0] * sx + ox, b[1] * sy + oy, b[2] * sx + ox, b[3] * sy + oy)
            if not _on_court(court, kps, box):
                continue
            dup = next((p for p in people if _iou(p["box"], box) > 0.35), None)
            if dup is None:
                people.append({"box": box, "kps": kps, "conf": float(c)})
            elif c > dup["conf"]:
                dup.update(box=box, kps=kps, conf=float(c))
    return [p["kps"] for p in people]


def poses(src: str, segments: list[dict], court, model, src_w: int, src_h: int,
          n_frames: int, report=None, cancelled=None):
    """Per footage frame: list of people, each 17 keypoints (x, y, conf) normalized to 0..1."""
    cx0, cx1, cy0, cy1 = far_crop(court)
    # each source frame arrives as one image: full frame (1080p) on top, native far crop below
    fw, fh = 1920, 1080
    crop_w = int((cx1 - cx0) * src_w) // 2 * 2
    crop_h = int((cy1 - cy0) * src_h) // 2 * 2
    stack_w = max(fw, crop_w)
    out: list[list] = [[] for _ in range(n_frames)]
    done = 0
    for s in segments:
        base = int(round(s["clip_start"] * FPS))  # first footage frame of this segment
        length = s["clip_end"] - s["clip_start"]
        expect = int(round(length * FPS))
        fc = (f"[0:v]fps={FPS},split=2[a][b];[a]scale={fw}:{fh},pad={stack_w}:{fh}[top];"
              f"[b]crop={crop_w}:{crop_h}:{int(cx0 * src_w)}:{int(cy0 * src_h)},pad={stack_w}:{crop_h}[bot];"
              f"[top][bot]vstack=inputs=2")
        cmd = [exe("ffmpeg"), "-nostdin", "-loglevel", "error", "-ss", f"{s['start']:.3f}", "-t", f"{length:.3f}",
               "-i", src, "-filter_complex", fc, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        k = 0
        batch = []
        frames = _frames(cmd, stack_w, fh + crop_h)
        while True:
            fr = next(frames, None)
            if fr is not None and k + len(batch) < expect:
                if cancelled and cancelled():
                    raise Cancelled()
                batch.append(fr)
                if len(batch) < 8:
                    continue
            if not batch:
                break
            r_top = model.predict([b[:fh, :fw] for b in batch], imgsz=1280, conf=0.3, verbose=False)
            r_bot = model.predict([b[fh:fh + crop_h, :crop_w] for b in batch], imgsz=1536, conf=0.2, verbose=False)
            for rt, rb in zip(r_top, r_bot):
                if 0 <= base + k < n_frames:
                    out[base + k] = _people(rt, rb, court, fw, fh, src_w, src_h, cx0, cy0)
                k += 1
            batch = []
            if report:
                report((done + k) / max(1, n_frames))
            if fr is None:
                break
        frames.close()
        done += expect
    return out


# ── Drawing ──────────────────────────────────────────────────────

ACCENT_BGR = (106, 230, 205)   # app accent (ball yellow-green), see analysis/render_style.py
DIM_BGR = (40, 70, 62)


def _trail_color(f):
    """0 (oldest) -> 1 (newest): from a dim olive into the app's accent."""
    return tuple(int(a + (b - a) * f) for a, b in zip(DIM_BGR, ACCENT_BGR))


def draw_frame(img, people, trail, show_pose=True, show_ball=True):
    import cv2
    if show_pose:
        for kps in people:
            pts = [(int(k[0] * OUT_W), int(k[1] * OUT_H), k[2]) for k in kps]
            for a, b in SKELETON:
                if pts[a][2] > 0.3 and pts[b][2] > 0.3:
                    cv2.line(img, pts[a][:2], pts[b][:2], (18, 16, 14), 4, cv2.LINE_AA)
                    cv2.line(img, pts[a][:2], pts[b][:2], (242, 240, 237), 2, cv2.LINE_AA)
            vis = [p for p in pts[5:7] if p[2] > 0.3]
            if pts[0][2] > 0.3 and vis:
                sx = sum(p[0] for p in vis) // len(vis)
                sy = sum(p[1] for p in vis) // len(vis)
                cv2.line(img, (sx, sy), pts[0][:2], (242, 240, 237), 2, cv2.LINE_AA)
            for p in pts[5:]:
                if p[2] > 0.3:
                    cv2.circle(img, p[:2], 3, (242, 240, 237), -1, cv2.LINE_AA)
    if show_ball and trail:
        layer = img.copy()
        n = len(trail)
        for i in range(1, n):
            a, b = trail[i - 1], trail[i]
            if a is None or b is None:
                continue
            f = i / max(1, n - 1)
            cv2.line(layer, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), _trail_color(f),
                     max(1, int(1 + 4 * f)), cv2.LINE_AA)
        cur = trail[-1]
        if cur is not None:
            cv2.circle(layer, (int(cur[0]), int(cur[1])), 12, (18, 16, 14), 4, cv2.LINE_AA)
            cv2.circle(layer, (int(cur[0]), int(cur[1])), 12, ACCENT_BGR, 2, cv2.LINE_AA)
        cv2.addWeighted(layer, 0.9, img, 0.1, 0, dst=img)


def annotate(footage: Path, out: Path, ball, people, has_audio: bool, show_pose=True, show_ball=True,
             report=None, cancelled=None):
    """Draw overlays on every footage frame and re-encode (audio copied from the footage)."""
    ffmpeg = exe("ffmpeg")
    enc = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{OUT_W}x{OUT_H}", "-r", str(FPS), "-i", "-", "-i", str(footage),
           "-map", "0:v", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
    if has_audio:
        enc += ["-map", "1:a:0", "-c:a", "copy"]
    enc += ["-shortest", str(out)]
    proc = subprocess.Popen(enc, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    dec = [ffmpeg, "-nostdin", "-loglevel", "error", "-i", str(footage), "-vf",
           f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    total = max(len(ball or []), len(people or []), 1)
    try:
        for i, fr in enumerate(_frames(dec, OUT_W, OUT_H)):
            if cancelled and cancelled():
                raise Cancelled()
            img = fr.copy()
            trail = None
            if ball is not None and i < len(ball):
                window = ball[max(0, i - TRAIL + 1): i + 1]
                # only draw while the ball is (or was just) visible
                if any(p is not None for p in window[-3:]):
                    trail = window
            draw_frame(img, people[i] if people is not None and i < len(people) else [],
                       trail, show_pose, show_ball)
            proc.stdin.write(img.tobytes())
            if report and i % 60 == 0:
                report(i / total)
        proc.stdin.close()
        rc = proc.wait()
        err = proc.stderr.read().decode("utf-8", errors="replace")
    except BaseException:
        proc.kill()
        raise
    if rc != 0 or not out.exists():
        raise RuntimeError(f"주석 영상 인코딩 실패: {err[-500:]}")


def pose_ready() -> bool:
    import importlib.util
    return importlib.util.find_spec("ultralytics") is not None
