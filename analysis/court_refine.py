"""Refine the 4-click court calibration against the painted lines.

1. A median of frames spread over the video gives an empty court (players move, lines don't).
2. White paint = bright, low-saturation pixels.
3. Every model line (baselines, doubles/singles sidelines, service lines, centre service
   line) is sampled in court metres, projected with the current homography, and the paint
   centre is searched along the image normal at each sample.
4. A homography is fitted to all (court point, paint point) pairs with RANSAC, and the
   loop repeats a few times with a shrinking search band.

The refined homography is turned back into four corner points, so the rest of the app is
unchanged. Alignment is reported as the median distance of projected line samples from
the paint before and after, in source pixels.
"""
from __future__ import annotations

import subprocess

import cv2
import numpy as np

from core.media import exe

W_D, L, SL, SR, C, SVC_N, SVC_F, NET = 10.97, 23.77, 1.37, 9.60, 5.485, 5.485, 18.285, 11.885
LINES = [  # (name, (x0, y0), (x1, y1)) in court metres, y = 0 at the near baseline
    ("near baseline", (0, 0), (W_D, 0)),
    ("far baseline", (0, L), (W_D, L)),
    ("left doubles sideline", (0, 0), (0, L)),
    ("right doubles sideline", (W_D, 0), (W_D, L)),
    ("left singles sideline", (SL, 0), (SL, L)),
    ("right singles sideline", (SR, 0), (SR, L)),
    ("near service line", (SL, SVC_N), (SR, SVC_N)),
    ("far service line", (SL, SVC_F), (SR, SVC_F)),
    ("centre service line", (C, SVC_N), (C, SVC_F)),
]


def empty_court(src: str, duration: float, w: int, h: int, n: int = 15) -> np.ndarray:
    frames = []
    for k in range(n):
        t = duration * (k + 0.5) / n
        raw = subprocess.run([exe("ffmpeg"), "-nostdin", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", src,
                              "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                             capture_output=True).stdout
        if len(raw) == w * h * 3:
            frames.append(np.frombuffer(raw, np.uint8).reshape(h, w, 3))
    if not frames:
        raise RuntimeError("프레임을 읽지 못했어")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def paint_mask(img: np.ndarray) -> np.ndarray:
    """Line paint: bright and grey-white, relative to the local court colour."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    local = cv2.blur(v, (61, 61))
    return ((v > local + 25) & (s < 60)).astype(np.float32) * (v - local)


def _project(H, pts):
    return cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), H).reshape(-1, 2)


def samples(H, step=0.25):
    """Court-metre samples along every line with their projected pixel and image normal."""
    out = []
    for name, a, b in LINES:
        a, b = np.float32(a), np.float32(b)
        n = max(2, int(np.linalg.norm(b - a) / step))
        pts = [a + (b - a) * (i / n) for i in range(n + 1)]
        px = _project(H, pts)
        d = px[-1] - px[0]
        nrm = np.float32([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
        for cpt, ipt in zip(pts, px):
            out.append((name, cpt, ipt, nrm))
    return out


def find_paint(weight, ipt, nrm, band):
    """Weighted centre of paint along the normal through ipt within +-band px, or None."""
    h, w = weight.shape
    offs = np.arange(-band, band + 1, dtype=np.float32)
    xs = ipt[0] + offs * nrm[0]
    ys = ipt[1] + offs * nrm[1]
    ok = (xs >= 0) & (xs < w - 1) & (ys >= 0) & (ys < h - 1)
    if ok.sum() < band:
        return None
    vals = np.zeros_like(offs)
    vals[ok] = weight[ys[ok].astype(int), xs[ok].astype(int)]
    if vals.max() <= 0:
        return None
    # take the strongest connected run only, so a neighbouring line doesn't pull the centre
    k = int(np.argmax(vals))
    lo, hi = k, k
    while lo > 0 and vals[lo - 1] > 0:
        lo -= 1
    while hi < len(vals) - 1 and vals[hi + 1] > 0:
        hi += 1
    seg = vals[lo:hi + 1]
    c = float((offs[lo:hi + 1] * seg).sum() / seg.sum())
    return ipt + c * nrm, abs(c)


def alignment(weight, H, band):
    dists = [r[1] for _, _, ipt, nrm in samples(H) if (r := find_paint(weight, ipt, nrm, band))]
    return float(np.median(dists)) if dists else None, len(dists)


def refine(img: np.ndarray, H0: np.ndarray, iters: int = 4):
    """H0: court metres -> source pixels. Returns (H, report)."""
    weight = paint_mask(img)
    H = H0.copy()
    before = alignment(weight, H, 12)
    for band in [int(b) for b in np.linspace(30, 8, iters)]:
        src, dst = [], []
        for _, cpt, ipt, nrm in samples(H):
            r = find_paint(weight, ipt, nrm, band)
            if r:
                src.append(cpt)
                dst.append(r[0])
        if len(src) < 20:
            break
        Hn, inl = cv2.findHomography(np.float32(src), np.float32(dst), cv2.RANSAC, 3.0)
        if Hn is None:
            break
        H = Hn
    after = alignment(weight, H, 12)
    return H, {"before_px": before[0], "after_px": after[0], "samples": after[1]}


def refine_calibration(file_entry: dict) -> dict:
    """Refine a file's 4-point calibration in place-ready form. Returns the new calibration dict."""
    meta = file_entry.get("metadata") or {}
    w, h = int(meta.get("width") or 0), int(meta.get("height") or 0)
    cal = file_entry.get("calibration") or {}
    manual = cal.get("manual_points") or cal.get("points") or []
    if len(manual) < 4 or not w or not h:
        raise RuntimeError("코트 4점 캘리브레이션을 먼저 해줘.")
    img = empty_court(file_entry["path"], float(meta.get("duration") or 60), w, h)
    court = np.float32([[0, 0], [W_D, 0], [W_D, L], [0, L]])
    imgpts = np.float32([[p["x"] * w, p["y"] * h] for p in manual[:4]])
    H0 = cv2.getPerspectiveTransform(court, imgpts)
    H1, rep = refine(img, H0)
    if rep["after_px"] is None or (rep["before_px"] is not None and rep["after_px"] > rep["before_px"]):
        raise RuntimeError(f"라인 보정이 더 나빠져서 적용하지 않았어 ({rep})")
    corners = _project(H1, court)
    return {
        "points": [{"x": float(x / w), "y": float(y / h)} for x, y in corners],
        "manual_points": manual[:4],
        "order": "near-left,near-right,far-right,far-left",
        "refined": {"line_error_px_before": round(rep["before_px"], 2), "line_error_px_after": round(rep["after_px"], 2),
                    "samples": rep["samples"]},
    }
