"""Measured serve speed from ball tracking (step 2).

TrackNet finds the ball in every source frame around a serve. The serve flight is
the stretch of trajectory between racket contact and the first bounce:

* bounce  — the ball's vertical velocity flips at the ground, which shows up in
            the image as a sharp upward kink of the y-velocity;
* contact — where the flight starts: either the end of the (mostly vertical) toss
            track, or the start of the flight track when the toss isn't seen.

The 4-point court calibration maps the bounce and the server's baseline position
to metres. Flight time comes from the source frame timestamps (~60 fps here), and
the launch speed (what a radar gun shows) is solved from distance and time with
quadratic air drag: d = ln(1 + k·v0·T) / k.
"""
from __future__ import annotations

import math
import re
import subprocess
import tempfile
from pathlib import Path

from core.media import exe

COURT_W, COURT_L = 10.97, 23.77
SINGLES_L, SINGLES_R = 1.37, 9.60
CENTER_X = 5.485
NET_Y = 11.885
SVC_NEAR, SVC_FAR = 5.485, 18.285
K_DRAG = 0.0202          # ρ·Cd·A / 2m for a tennis ball (Cd≈0.55), 1/m
DEFAULT_CONTACT_H = 2.6  # m, used when the contact height can't be read from the image
IN_W, IN_H = 640, 360
STATIC_RADIUS = 2.5
KINK_MIN_MS = 4.0        # m/s upward change of vertical velocity that counts as a bounce
LINE_MARGIN = 0.25       # m; closer than this to a line → "close" instead of in/out


# ── Geometry ─────────────────────────────────────────────────────

class Court:
    """Homography between normalized image coords and court metres (y=0 near baseline)."""

    def __init__(self, calibration_points: list[dict]):
        import numpy as np
        import cv2
        self.np, self.cv2 = np, cv2
        img = np.float32([[p["x"], p["y"]] for p in calibration_points[:4]])
        court = np.float32([[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]])
        self.img = img
        self.H = cv2.getPerspectiveTransform(img, court)
        self.Hinv = cv2.getPerspectiveTransform(court, img)

    def to_court(self, x, y):
        p = self.cv2.perspectiveTransform(self.np.float32([[[x, y]]]), self.H)[0, 0]
        return float(p[0]), float(p[1])

    def to_image(self, cx, cy):
        p = self.cv2.perspectiveTransform(self.np.float32([[[cx, cy]]]), self.Hinv)[0, 0]
        return float(p[0]), float(p[1])

    def baseline_y(self, x, end):
        """Image y of the near/far baseline at normalized image x."""
        a, b = (self.img[0], self.img[1]) if end == "near" else (self.img[3], self.img[2])
        return float(a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0]))

    def px_per_m(self, cx, cy):
        """Image pixels (640-wide frame) per metre across the court at a court point."""
        x1, y1 = self.to_image(cx - 0.5, cy)
        x2, y2 = self.to_image(cx + 0.5, cy)
        return math.hypot((x2 - x1) * IN_W, (y2 - y1) * IN_H)


def launch_speed(distance_m: float, flight_s: float) -> float:
    """Initial speed (m/s) that covers distance_m in flight_s under quadratic drag."""
    return (math.exp(K_DRAG * distance_m) - 1.0) / (K_DRAG * flight_s)


def serve_box(server_end: str, server_x: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Diagonal service box (x-range, y-range) the serve must land in."""
    # Near server faces +y: their right (deuce side) is +x. Far server faces -y: deuce is -x.
    deuce = server_x > CENTER_X if server_end == "near" else server_x < CENTER_X
    if server_end == "near":
        xr = (SINGLES_L, CENTER_X) if deuce else (CENTER_X, SINGLES_R)
        yr = (NET_Y, SVC_FAR)
    else:
        xr = (CENTER_X, SINGLES_R) if deuce else (SINGLES_L, CENTER_X)
        yr = (SVC_NEAR, NET_Y)
    return xr, yr


def judge_in(bx: float, by: float, xr, yr) -> str:
    inside = xr[0] <= bx <= xr[1] and yr[0] <= by <= yr[1]
    edge = min(abs(bx - xr[0]), abs(bx - xr[1]), abs(by - yr[0]), abs(by - yr[1]))
    if edge < LINE_MARGIN:
        return "close"
    return "in" if inside else "out"


def placement(bx: float, xr) -> str:
    """T / body / wide by thirds of the 4.115 m wide box, measured from the centre line."""
    dc = abs(bx - CENTER_X)
    if dc < 1.37:
        return "T"
    return "body" if dc < 2.74 else "wide"


# ── Frames ───────────────────────────────────────────────────────

_GPU_DECODE = None  # decided on first use: can FFmpeg decode on the NVIDIA GPU here?


def _gpu_decode_ok() -> bool:
    global _GPU_DECODE
    if _GPU_DECODE is None:
        import os
        if os.environ.get("HT_GPU_DECODE", "1") == "0":
            _GPU_DECODE = False
        else:
            try:
                import torch
                _GPU_DECODE = bool(torch.cuda.is_available())
            except Exception:
                _GPU_DECODE = False
    return _GPU_DECODE


_PIX_FMT: dict[str, str] = {}


def _pix_fmt(src: str) -> str:
    if src not in _PIX_FMT:
        try:
            out = subprocess.run([exe("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                                  "stream=pix_fmt", "-of", "csv=p=0", src], capture_output=True, text=True).stdout
            _PIX_FMT[src] = out.strip()
        except Exception:
            _PIX_FMT[src] = ""
    return _PIX_FMT[src]


def read_window(src: str, t0: float, t1: float):
    """Decode [t0, t1) of the source at 640x360 with real presentation timestamps.

    On an NVIDIA GPU the video is decoded there and brought back at full size and bit depth for
    the same area downscale on the CPU: the frames are bit-identical to a CPU decode (a GPU-side
    downscale was 0.9/255 off and lost a borderline net-fault serve) and 2.3x faster on the 4K
    10-bit HEVC test video. Falls back to the CPU when the GPU path gives nothing."""
    import numpy as np
    global _GPU_DECODE
    ffmpeg = exe("ffmpeg")
    t0 = max(0.0, t0)
    area = f"scale={IN_W}:{IN_H}:flags=area,showinfo"
    variants = []
    if _gpu_decode_ok():
        if "10" in _pix_fmt(src):
            variants.append((["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"], "hwdownload,format=p010le," + area))
        variants.append((["-hwaccel", "cuda"], area))
    variants.append(([], area))
    for k, (pre, vf) in enumerate(variants):
        cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "info", *pre, "-ss", f"{t0:.3f}", "-copyts",
               "-i", src, "-to", f"{t1:.3f}", "-an", "-vf", vf,
               "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        with tempfile.TemporaryFile() as err:
            cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=err)
            err.seek(0)
            log = err.read().decode("utf-8", errors="replace")
        size = IN_W * IN_H * 3
        raw = cp.stdout
        frames = [np.frombuffer(raw, np.uint8, size, j).reshape(IN_H, IN_W, 3) for j in range(0, len(raw) - size + 1, size)]
        times = [float(m) for m in re.findall(r"\bpts_time:\s*([-\d.]+)", log)]
        n = min(len(frames), len(times))
        if n:
            return frames[:n], times[:n]
        if pre and not variants[k + 1][0]:
            _GPU_DECODE = False  # this machine's FFmpeg/GPU can't; stay on the CPU from now on
    raise RuntimeError(f"프레임을 읽지 못했어 ({t0:.1f}s~{t1:.1f}s)")


# ── Detections → tracks ──────────────────────────────────────────

def heat_to_blobs(heat, thresh: int = 127, max_n: int = 3):
    import cv2
    import numpy as np
    if heat is None:
        return []
    mask = (heat > thresh).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 2 or area > 400:
            continue
        ys, xs = np.nonzero(labels == i)
        w = heat[ys, xs].astype(np.float32)
        blobs.append((float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum()), float(w.sum())))
    blobs.sort(key=lambda b: -b[2])
    return blobs[:max_n]


def remove_static(dets: list[list[tuple]], fps: float) -> list[list[tuple]]:
    """Drop blobs that sit in one spot for a quarter second or more (lights, parked balls)."""
    pts = [(i, b) for i, bl in enumerate(dets) for b in bl]
    static = []
    need = max(8, int(fps * 0.25))
    for i, b in pts:
        if any(math.hypot(b[0] - s[0], b[1] - s[1]) < STATIC_RADIUS for s in static):
            continue
        frames = {j for j, c in pts if math.hypot(b[0] - c[0], b[1] - c[1]) < STATIC_RADIUS}
        if len(frames) >= need:
            static.append(b)
    return [[b for b in bl if not any(math.hypot(b[0] - s[0], b[1] - s[1]) < STATIC_RADIUS + 1 for s in static)]
            for bl in dets]


def link_tracks(dets: list[list[tuple]], times: list[float], max_gap: int = 6) -> list[list[tuple]]:
    """Greedy constant-velocity linking. Returns tracks of (t, x, y)."""
    active: list[dict] = []
    done: list[list[tuple]] = []
    for i, blobs in enumerate(dets):
        t = times[i]
        still = []
        for tr in active:
            if i - tr["last_i"] > max_gap:
                done.append(tr["pts"])
            else:
                still.append(tr)
        active = still
        pairs = []
        for bi, b in enumerate(blobs):
            for ti, tr in enumerate(active):
                lt, lx, ly = tr["pts"][-1]
                dt = t - lt
                gap = i - tr["last_i"]
                if tr["v"] is None:
                    gate = 30.0 * gap
                    px, py = lx, ly
                else:
                    vx, vy = tr["v"]
                    px, py = lx + vx * dt, ly + vy * dt
                    gate = 8.0 + 3.0 * gap + 0.25 * math.hypot(vx, vy) * dt
                d = math.hypot(b[0] - px, b[1] - py)
                if d <= gate:
                    pairs.append((d, bi, ti))
        used_b, used_t = set(), set()
        for d, bi, ti in sorted(pairs):
            if bi in used_b or ti in used_t:
                continue
            used_b.add(bi)
            used_t.add(ti)
            tr = active[ti]
            lt, lx, ly = tr["pts"][-1]
            b = blobs[bi]
            dt = max(1e-3, t - lt)
            tr["v"] = ((b[0] - lx) / dt, (b[1] - ly) / dt)
            tr["pts"].append((t, b[0], b[1]))
            tr["last_i"] = i
        for bi, b in enumerate(blobs):
            if bi not in used_b:
                active.append({"pts": [(t, b[0], b[1])], "v": None, "last_i": i})
    done += [tr["pts"] for tr in active]
    return [p for p in done if len(p) >= 3]


def bridge_tracks(tracks: list[list[tuple]], max_gap: float = 0.3) -> list[list[tuple]]:
    """Join a track to one that starts shortly after it where its motion says the ball went.

    The ball is often lost for a few frames right around the bounce, which splits the
    one trajectory that holds the bounce kink into two tracks.
    """
    tracks = sorted(tracks, key=lambda t: t[0][0])
    used = set()
    out = []
    for i, a in enumerate(tracks):
        if i in used:
            continue
        chain = list(a)
        while True:
            tail = chain[-6:]
            vx, _ = _slope(tail, 1)
            vy, _ = _slope(tail, 2)
            if vx is None:
                break
            best, best_d = None, None
            for j, b in enumerate(tracks):
                if j in used or j == i or b is a:
                    continue
                dt = b[0][0] - chain[-1][0]
                if not (0 < dt <= max_gap):
                    continue
                px, py = chain[-1][1] + vx * dt, chain[-1][2] + vy * dt
                d = math.hypot(b[0][1] - px, b[0][2] - py)
                if d <= 15 + 0.6 * math.hypot(vx, vy) * dt and (best_d is None or d < best_d):
                    best, best_d = j, d
            if best is None:
                break
            used.add(best)
            chain += tracks[best]
        out.append(chain)
    return out


# ── Flight analysis ──────────────────────────────────────────────

def _slope(pts, axis=2):
    n = len(pts)
    if n < 2:
        return None, None
    mt = sum(p[0] for p in pts) / n
    mv = sum(p[axis] for p in pts) / n
    den = sum((p[0] - mt) ** 2 for p in pts)
    if den <= 0:
        return None, None
    b = sum((p[0] - mt) * (p[axis] - mv) for p in pts) / den
    return b, mv - b * mt


def find_bounces(track: list[tuple], court: Court, win: float = 0.1):
    """Upward kinks of the image y-velocity, scaled to m/s at the ball's court position."""
    out = []
    for i in range(2, len(track) - 2):
        t = track[i][0]
        before = [p for p in track if t - win <= p[0] <= t]
        after = [p for p in track if t <= p[0] <= t + win]
        if len(before) < 3 or len(after) < 3:
            continue
        b1, a1 = _slope(before)
        b2, a2 = _slope(after)
        if b1 is None or b2 is None:
            continue
        cx, cy = court.to_court(track[i][1] / IN_W, track[i][2] / IN_H)
        scale = max(4.0, court.px_per_m(min(max(cx, 0), COURT_W), min(max(cy, 0), COURT_L)))
        kink = (b2 - b1) / scale
        if kink < -KINK_MIN_MS:
            tb = (a2 - a1) / (b1 - b2) if abs(b1 - b2) > 1e-6 else t
            if abs(tb - t) > win:
                tb = t
            xb = _interp(track, tb, 1)
            yb = _interp(track, tb, 2)
            out.append({"i": i, "t": tb, "x": xb, "y": yb, "kink": kink})
    # kinks hidden inside a hole in the track: compare the motion on either side of it
    for i in range(1, len(track)):
        a, b = track[i - 1], track[i]
        if b[0] - a[0] < 0.05:
            continue
        before = [p for p in track[max(0, i - 6):i] if a[0] - p[0] <= 0.25]
        after = [p for p in track[i:i + 6] if p[0] - b[0] <= 0.25]
        if len(before) < 3 or len(after) < 3:
            continue
        b1, a1 = _slope(before)
        b2, a2 = _slope(after)
        if b1 is None or b2 is None or abs(b1 - b2) < 1e-6:
            continue
        tb = (a2 - a1) / (b1 - b2)
        if not (a[0] - 0.03 <= tb <= b[0] + 0.03):
            continue
        xb = _interp(track, tb, 1)
        yb = a1 + b1 * tb
        cx, cy = court.to_court(xb / IN_W, yb / IN_H)
        scale = max(4.0, court.px_per_m(min(max(cx, 0), COURT_W), min(max(cy, 0), COURT_L)))
        kink = (b2 - b1) / scale
        if kink < -KINK_MIN_MS:
            out.append({"i": i, "t": tb, "x": xb, "y": yb, "kink": kink})
    # keep the strongest kink of each cluster
    out.sort(key=lambda b: b["t"])
    merged = []
    for b in out:
        if merged and b["t"] - merged[-1]["t"] < 0.08:
            if b["kink"] < merged[-1]["kink"]:
                merged[-1] = b
        else:
            merged.append(b)
    return merged


def _interp(track, t, axis):
    for a, b in zip(track, track[1:]):
        if a[0] <= t <= b[0]:
            f = (t - a[0]) / max(1e-6, b[0] - a[0])
            return a[axis] + (b[axis] - a[axis]) * f
    return min(track, key=lambda p: abs(p[0] - t))[axis]


def _toss_before(tracks, run, bounce):
    """The toss: a mostly-vertical track that ends where the flight begins.

    Right after a near-side contact the ball is fast and blurred and TrackNet usually
    loses it until it is well into the far court, so the flight may only reappear up to
    ~0.6 s later. It must then lie on the way from the toss end to the bounce.
    """
    start = run[0]
    best = None
    for tr in tracks:
        end = tr[-1]
        gap = start[0] - end[0]
        if not (-0.02 <= gap <= 0.6):
            continue
        seg = [p for p in tr if p[0] >= end[0] - 0.3]
        if len(seg) < 3:
            continue
        dx = abs(seg[-1][1] - seg[0][1])
        dy = abs(seg[-1][2] - seg[0][2])
        if dy < 8 or dx > max(4.0, 0.5 * dy):
            continue
        ex, ey = start[1] - end[1], start[2] - end[2]
        bx, by = bounce["x"] - end[1], bounce["y"] - end[2]
        if gap > 0.12:
            nb = math.hypot(bx, by)
            ne = math.hypot(ex, ey)
            if nb < 1 or ne >= nb or (ex * bx + ey * by) / max(1e-6, ne * nb) < 0.8:
                continue
        elif math.hypot(ex, ey) > 80:
            continue
        if best is None or end[0] > best[-1][0]:
            best = tr
    return best


def measure_flights(tracks: list[list[tuple]], court: Court, t_from: float, t_to: float) -> list[dict]:
    """All serve-like flights whose contact falls in [t_from, t_to], earliest first."""
    found = []
    for tr in bridge_tracks(tracks):
        if len(tr) < 8:
            continue
        for b in find_bounces(tr, court):
            # the flight side comes from the unbridged track holding the last point before
            # the bounce, so a bridge can't pull an earlier toss or rally ball into it
            last = max((p for p in tr if p[0] < b["t"] - 0.01), default=None, key=lambda p: p[0])
            if last is None:
                continue
            own = next((o for o in tracks if last in o), tr)
            pre = [p for p in own if p[0] < b["t"] - 0.01]
            if len(pre) < 5:
                continue
            # flight = pre-bounce part of the track; the linker already vouched for continuity,
            # so only a long hole (the ball lost far away) splits it
            run = [pre[-1]]
            for p in reversed(pre[:-1]):
                if run[-1][0] - p[0] > 0.2:
                    break
                run.append(p)
            run.reverse()
            toss = _toss_before(tracks, run, b)
            if toss is not None:
                gap = run[0][0] - toss[-1][0]
                # the toss is tracked until the racket meets it: contact is just after its last point
                tc = toss[-1][0] + (gap / 2 if gap <= 0.12 else 0.008)
                cpt = toss[-1]
                contact_src = "toss"
            else:
                tc = run[0][0]
                cpt = run[0]
                contact_src = "flight_start"
            if not (t_from <= tc <= t_to):
                continue
            T = b["t"] - tc
            if contact_src == "flight_start" and court.to_court(b["x"] / IN_W, b["y"] / IN_H)[1] > NET_Y:
                # lands in the far half, so it came from the near side; a near-side serve always
                # shows its toss to this camera, so a flight with no toss is a rally shot
                continue
            if not (0.3 <= T <= 1.5):
                continue
            res = _evaluate(court, cpt, b, T, need_height=contact_src == "flight_start")
            if res:
                res.update(contact_t=round(tc, 3), bounce_t=round(b["t"], 3), contact_src=contact_src,
                           points=len(run), _contact_px=(cpt[1], cpt[2]), _flight=run)
                found.append(res)
    found.sort(key=lambda r: r["contact_t"])
    # one flight per contact
    uniq = []
    for r in found:
        if not uniq or r["contact_t"] - uniq[-1]["contact_t"] > 0.2:
            uniq.append(r)
    return uniq


def _evaluate(court: Court, cpt, b, T, need_height=False):
    bx, by = court.to_court(b["x"] / IN_W, b["y"] / IN_H)
    if not (-1.5 <= bx <= COURT_W + 1.5 and -1.0 <= by <= COURT_L + 1.0):
        return None
    # A kink right at the net line is the ball hitting the net (net fault), not a bounce.
    at_net = abs(by - NET_Y) < 0.6
    if at_net:
        server_end = "near" if _interp_side(cpt, court) == "near" else "far"
        by = NET_Y
    else:
        server_end = "far" if by < NET_Y else "near"
    cxn = cpt[1] / IN_W
    base_y = court.baseline_y(cxn, server_end)
    sx, sy = court.to_court(cxn, base_y)
    if not (-2.0 <= sx <= COURT_W + 2.0):
        return None
    # contact height: pixels above the server's baseline at the local metre scale
    scale = court.px_per_m(min(max(sx, 0), COURT_W), sy)
    h = (base_y * IN_H - cpt[2]) / scale if scale > 0 else DEFAULT_CONTACT_H
    if need_height and not 1.6 <= h <= 3.4:
        return None  # track starts mid-flight, not at the racket
    h_used = h if 1.6 <= h <= 3.4 else DEFAULT_CONTACT_H
    horiz = math.hypot(bx - sx, by - sy)
    if not ((10.5 if at_net else 11.0) <= horiz <= 30.0):
        return None
    dist = math.hypot(horiz, h_used - (0.95 if at_net else 0.0))
    v0 = launch_speed(dist, T) * 3.6
    if not (50 <= v0 <= 240):
        return None
    xr, yr = serve_box(server_end, sx)
    frame = 1 / 59.4
    rel = math.hypot(1.5 * frame / T, 0.4 / dist)
    return {
        "speed_kmh": round(v0),
        "speed_err_kmh": round(v0 * rel),
        "avg_kmh": round(dist / T * 3.6),
        "flight_s": round(T, 3),
        "distance_m": round(dist, 2),
        "contact_height_m": round(h, 2),
        "server_end": server_end,
        "server_court": [round(sx, 2), round(sy, 2)],
        "bounce_court": [round(bx, 2), round(by, 2)],
        "bounce_px": [round(b["x"], 1), round(b["y"], 1)],
        "box": {"x": list(xr), "y": list(yr)},
        "call": "net" if at_net else judge_in(bx, by, xr, yr),
        "direction": placement(bx, xr),
        "side": "deuce" if (sx > CENTER_X) == (server_end == "near") else "ad",
    }


def _interp_side(cpt, court: Court) -> str:
    """Which baseline the contact point stands over: the one whose image y is just below it."""
    xn = cpt[1] / IN_W
    near_y = court.baseline_y(xn, "near") * IN_H
    return "near" if cpt[2] > (court.baseline_y(xn, "far") * IN_H + near_y) / 2 - 40 else "far"


def near_serves_3d(tracks, court: Court, cam, t_from: float, t_to: float) -> list[dict]:
    """Near-side serves whose bounce wasn't seen: fit the 3D flight from the toss onwards.

    Only near-side serves: the ball crosses the view as it flies away, which pins the
    trajectory down. For far-side serves (ball coming at the camera) depth and height
    trade off and the extrapolated bounce isn't trustworthy.
    """
    from analysis.flight3d import fit_serve
    out = []
    bridged = bridge_tracks(tracks)
    for toss in tracks:
        end = toss[-1]
        if not (t_from <= end[0] <= t_to):
            continue
        seg = [p for p in toss if p[0] >= end[0] - 0.3]
        if len(seg) < 3:
            continue
        dx, dy = abs(seg[-1][1] - seg[0][1]), abs(seg[-1][2] - seg[0][2])
        if dy < 8 or dx > max(4.0, 0.5 * dy):
            continue
        p0 = cam.on_plane_y(end[1], end[2], 0.0)
        if p0 is None or not (1.6 <= p0[2] <= 3.4 and -1.5 <= p0[0] <= COURT_W + 1.5):
            continue
        tc = end[0] + 0.008
        for tr in sorted(bridged, key=lambda t: t[0][0]):
            first = tr[0]
            if not (tc < first[0] <= tc + 0.6) or len(tr) < 8 or first[2] >= end[2]:
                continue
            pts = [p for p in tr if p[0] <= tc + 1.5]
            # stop at the bounce if it is visible: image y stops growing and turns back
            ymax = max(range(len(pts)), key=lambda i: pts[i][2])
            if ymax < len(pts) - 1 and pts[-1][2] < pts[ymax][2] - 3:
                pts = pts[:ymax + 1]
            fit = fit_serve(cam, (end[1], end[2]), tc, "near", pts)
            if not fit or not fit.get("ok"):
                continue
            bx, by = fit["bounce_court"]
            sx = float(p0[0])
            xr, yr = serve_box("near", sx)
            # does the fitted flight clear the net (0.914 m centre, 1.07 m posts)?
            from analysis.flight3d import crossing
            cr = crossing(fit["contact_3d"], fit["v0"], NET_Y)
            at_net = cr is None or cr[2] < 1.0
            if at_net:
                bx, by = (cr[1], NET_Y) if cr else (bx, NET_Y)
                call = "net"
            elif not (NET_Y <= by <= COURT_L + 3 and -2 <= bx <= COURT_W + 2):
                continue
            else:
                call = judge_in(bx, by, xr, yr)
            # an extrapolated bounce is less certain than a seen one: widen the doubt band
            if call not in ("close", "net") and min(abs(bx - xr[0]), abs(bx - xr[1]), abs(by - yr[0]), abs(by - yr[1])) < 0.6:
                call = "close"
            T = (cr[0] if at_net and cr else fit["bounce_t"] - tc)
            out.append({
                "speed_kmh": fit["speed_kmh"], "speed_err_kmh": max(6, round(fit["speed_kmh"] * 0.06)),
                "avg_kmh": round(math.hypot(bx - sx, by) / max(0.2, T) * 3.6), "flight_s": round(T, 3),
                "distance_m": round(math.hypot(math.hypot(bx - sx, by), float(p0[2])), 2),
                "contact_height_m": round(float(p0[2]), 2), "server_end": "near",
                "server_court": [round(sx, 2), 0.0], "bounce_court": [round(bx, 2), round(by, 2)],
                "bounce_px": None, "box": {"x": list(xr), "y": list(yr)}, "call": call,
                "direction": placement(bx, xr), "side": "deuce" if sx > CENTER_X else "ad",
                "contact_t": round(tc, 3), "bounce_t": round(tc + T, 3), "contact_src": "toss",
                "net_height_m": round(cr[2], 2) if cr else None,
                "points": fit["n_points"], "method": "3d", "fit_rms_px": fit["rms_px"],
            })
            break
    out.sort(key=lambda r: r["contact_t"])
    return out


# ── Runner ───────────────────────────────────────────────────────

class ServeMeter:
    def __init__(self, detector, court: Court, src: str, fps: float = 59.4):
        self.det, self.court, self.src, self.fps = detector, court, src, fps
        # ball detections per frame, kept so a stretch of video is decoded and run through the
        # detector once even when the serve, 2nd-serve and rally passes all ask for it
        self._blobs: dict[float, list] = {}
        self._covered: list[tuple[float, float]] = []
        try:
            from analysis.flight3d import Camera
            self.cam = Camera(court)
        except Exception:
            self.cam = None

    def _flights(self, tracks, t_from, t_to):
        found = measure_flights(tracks, self.court, t_from, t_to)
        for f in found:
            f.setdefault("method", "bounce")
        if self.cam is not None:
            for f in near_serves_3d(tracks, self.court, self.cam, t_from, t_to):
                if not any(abs(f["contact_t"] - g["contact_t"]) < 0.3 for g in found):
                    found.append(f)
        found.sort(key=lambda r: r["contact_t"])
        for f in found:
            f.pop("_contact_px", None)  # "_flight" stays: the serve-type fit needs it after the audio fix
        return found

    def _missing(self, t0: float, t1: float) -> list[tuple[float, float]]:
        gaps, cur = [], t0
        for a, b in sorted(self._covered):
            if b <= cur or a >= t1:
                continue
            if a > cur:
                gaps.append((cur, a))
            cur = max(cur, b)
        if cur < t1:
            gaps.append((cur, t1))
        return [(a, b) for a, b in gaps if b - a > 0.5 / self.fps]

    def _detect(self, t0: float, t1: float, frames_times=None):
        """Decode [t0, t1) and store each frame's ball blobs. The detector looks at three frames,
        so two extra frames before t0 are decoded and only used as history."""
        frames, times = frames_times or read_window(self.src, t0 - 2.5 / self.fps, t1)
        heats = self.det.heat_levels(frames)
        for t, h in zip(times, heats):
            if h is not None and t >= t0 - 0.25 / self.fps:
                self._blobs[round(t, 4)] = heat_to_blobs(h)
        self._covered.append((t0, t1))

    def prefetch(self, spans, progress=None, cancelled=None, slice_s: float = 8.0):
        """Run the detector over all `spans` up front, FFmpeg decoding the next slice while the GPU
        works on this one (decoding was a third of the tracking time and needs no GPU). Later
        tracks() calls over these spans are then served from memory."""
        from concurrent.futures import ThreadPoolExecutor
        merged = []
        for a, b in sorted((max(0.0, a), b) for a, b in spans):
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        slices = []
        for a, b in merged:
            for ga, gb in self._missing(a, b):
                while ga < gb - 0.5 / self.fps:
                    slices.append((ga, min(gb, ga + slice_s)))
                    ga = slices[-1][1]
        if not slices:
            return
        lead = 2.5 / self.fps
        with ThreadPoolExecutor(1) as pool:
            fut = pool.submit(read_window, self.src, slices[0][0] - lead, slices[0][1])
            for k, (a, b) in enumerate(slices):
                if cancelled and cancelled():
                    from analysis.shot_analysis import Cancelled
                    raise Cancelled()
                got = fut.result()
                if k + 1 < len(slices):
                    fut = pool.submit(read_window, self.src, slices[k + 1][0] - lead, slices[k + 1][1])
                self._detect(a, b, got)
                if progress:
                    progress((k + 1) / len(slices))

    def tracks(self, t0: float, t1: float):
        for a, b in self._missing(max(0.0, t0), t1):
            self._detect(a, b)
        times = sorted(t for t in self._blobs if t0 <= t < t1)
        dets = remove_static([self._blobs[t] for t in times], self.fps)
        return link_tracks(dets, times)

    def at_point(self, start: float, end: float) -> list[dict]:
        """1st serve near START; if it was out (or too close to call), look for the 2nd serve later
        in the point. A 2nd serve from the same end means the 1st was a fault after all."""
        out = []
        tracks = self.tracks(start - 1.5, start + 3.0)
        first = self._flights(tracks, start - 1.2, start + 1.8)
        if not first:
            return out
        out.append(first[0])
        if first[0]["call"] in ("out", "net", "close") and end > first[0]["bounce_t"] + 3:
            t0 = first[0]["bounce_t"] + 2.0
            t1 = min(end + 1.0, t0 + 25.0)
            # 2nd serve: scan in 8 s slices so memory stays bounded
            while t0 < t1 and len(out) < 2:
                tracks = self.tracks(t0, min(t1, t0 + 8.0))
                nxt = [f for f in self._flights(tracks, t0, min(t1, t0 + 7.0))
                       if f["server_end"] == first[0]["server_end"]]
                if nxt:
                    out.append(nxt[0])
                t0 += 7.0
            if len(out) == 2 and first[0]["call"] == "close":
                first[0]["call"] = "out"
                first[0]["call_note"] = "close, but a 2nd serve followed"
        return out

    def near(self, t: float) -> dict | None:
        """The serve whose contact is closest to an approximate time (e.g. from Gemini)."""
        tracks = self.tracks(t - 1.5, t + 2.5)
        flights = self._flights(tracks, t - 1.0, t + 1.0)
        return min(flights, key=lambda f: abs(f["contact_t"] - t)) if flights else None
