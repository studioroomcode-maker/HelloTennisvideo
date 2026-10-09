"""Ad-hoc frame viewer for manual labelling.

  python tools/gt_view.py strip OUT.jpg T0 T1 X Y W H [STEP] [SCALE]
      every STEP-th native frame in [T0, T1], cropped to (X, Y, W, H) source px, tiled with timestamps
  python tools/gt_view.py ball T X Y [R]
      precise ball centre near (X, Y) in the native frame closest to T: yellow-green blob search
      within radius R (source px). Prints "t x y area".
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = str(ROOT / "20260708_203015.mp4")


def grab(t0, t1, crop):
    x, y, w, h = crop
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info", "-ss", f"{t0:.4f}", "-copyts", "-i", SRC,
           "-to", f"{t1:.4f}", "-an", "-vf", f"crop={w}:{h}:{x}:{y},showinfo", "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    with tempfile.TemporaryFile() as err:
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=err).stdout
        err.seek(0)
        ts = [float(m) for m in re.findall(r"\bpts_time:\s*([-\d.]+)", err.read().decode("utf-8", "replace"))]
    size = w * h * 3
    fr = [np.frombuffer(out, np.uint8, size, k).reshape(h, w, 3) for k in range(0, len(out) - size + 1, size)]
    return list(zip(ts, fr))


def main():
    mode = sys.argv[1]
    if mode == "strip":
        out = sys.argv[2]
        t0, t1 = float(sys.argv[3]), float(sys.argv[4])
        crop = tuple(int(v) for v in sys.argv[5:9])
        step = int(sys.argv[9]) if len(sys.argv) > 9 else 1
        scale = float(sys.argv[10]) if len(sys.argv) > 10 else 1.0
        pairs = grab(t0 - 0.02, t1 + 0.02, crop)
        pairs = [p for p in pairs if t0 - 0.009 <= p[0] <= t1 + 0.009][::step]
        tiles = []
        for t, im in pairs:
            im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            if scale >= 2:  # fine grid every 10 source px, labelled every 50
                x0, y0 = crop[0], crop[1]
                for gx in range((x0 // 10 + 1) * 10, x0 + crop[2], 10):
                    X = int((gx - x0) * scale)
                    cv2.line(im, (X, im.shape[0] - (10 if gx % 50 else 22)), (X, im.shape[0]), (0, 255, 255), 1)
                    if gx % 50 == 0:
                        cv2.putText(im, str(gx), (X + 2, im.shape[0] - 24), 0, 0.35, (0, 255, 255), 1)
                for gy in range((y0 // 10 + 1) * 10, y0 + crop[3], 10):
                    Y = int((gy - y0) * scale)
                    cv2.line(im, (0, Y), (10 if gy % 50 else 22, Y), (0, 255, 255), 1)
                    if gy % 50 == 0:
                        cv2.putText(im, str(gy), (24, Y + 4), 0, 0.35, (0, 255, 255), 1)
            cv2.putText(im, f"{t:.3f}", (3, 16), 0, 0.5, (0, 255, 255), 1)
            tiles.append(im)
        cols = int(sys.argv[11]) if len(sys.argv) > 11 else 8
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))
        print(len(pairs), "frames")
    elif mode == "ball":
        t, x, y = float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        r = int(sys.argv[5]) if len(sys.argv) > 5 else 40
        pairs = grab(t - 0.03, t + 0.03, (x - r, y - r, 2 * r, 2 * r))
        tt, im = min(pairs, key=lambda p: abs(p[0] - t))
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        # tennis-ball yellow-green, bright
        mask = cv2.inRange(hsv, (25, 60, 120), (50, 255, 255))
        n, lab, st, cen = cv2.connectedComponentsWithStats(mask)
        best = max(range(1, n), key=lambda i: st[i, cv2.CC_STAT_AREA], default=None)
        if best is None:
            print(f"{tt:.3f} none")
            return
        cx, cy = cen[best]
        print(f"{tt:.3f} {x - r + cx:.1f} {y - r + cy:.1f} {st[best, cv2.CC_STAT_AREA]}")


def trace(tm, half, x, y, r=50, box=700):
    """Colour-only ball trace (no learned model) around tm +- half s, seeded at (x, y) in the frame nearest tm
    and followed forwards and backwards from there."""
    pairs = grab(tm - half - 0.01, tm + half + 0.01, (x - box // 2, y - box // 2, box, box))
    ox, oy = x - box // 2, y - box // 2

    def find(im, cx, cy, rr):
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (22, 50, 110), (52, 255, 255))
        n, lab, st, cen = cv2.connectedComponentsWithStats(mask)
        cands = [(i, cen[i]) for i in range(1, n) if 3 <= st[i, cv2.CC_STAT_AREA] <= 900
                 and abs(cen[i][0] + ox - cx) <= rr and abs(cen[i][1] + oy - cy) <= rr]
        if not cands:
            return None
        i, c = min(cands, key=lambda ic: (ic[1][0] + ox - cx) ** 2 + (ic[1][1] + oy - cy) ** 2)
        return c[0] + ox, c[1] + oy, int(st[i, cv2.CC_STAT_AREA])

    k0 = min(range(len(pairs)), key=lambda k: abs(pairs[k][0] - tm))
    res = {}
    for rng in (range(k0, len(pairs)), range(k0, -1, -1)):
        cx, cy, miss = x, y, 0
        for k in rng:
            f = find(pairs[k][1], cx, cy, r * (1 + miss))
            if f is None:
                miss += 1
                res.setdefault(k, (pairs[k][0], None, None, 0))
                continue
            miss = 0
            cx, cy = f[0], f[1]
            res[k] = (pairs[k][0], f[0], f[1], f[2])
    return [res[k] for k in sorted(res)]


if __name__ == "__main__":
    if sys.argv[1] == "trace":
        for t, x, y, a in trace(float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]),
                                int(sys.argv[6]) if len(sys.argv) > 6 else 50):
            print(f"{t:.3f} " + ("-" if x is None else f"{x:.1f} {y:.1f} {a}"))
        sys.exit(0)
    main()
