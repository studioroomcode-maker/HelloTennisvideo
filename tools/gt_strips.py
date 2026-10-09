"""Render frame strips around each serve's contact and bounce for manual ground-truth labelling.

Usage: python tools/gt_strips.py OUT_DIR
For every serve in cache/<file>_serves.json, writes
  <n>_contact.jpg : 13 native frames around the measured contact, cropped around the server
  <n>_bounce.jpg  : 13 native frames around the measured bounce, cropped around the landing spot,
                    with a pixel grid in 4K source coordinates so positions can be read off.
Each tile is labelled with its exact source timestamp. The measured times only pick *where to
look*; the labels are read from the pictures, so they don't depend on any ball detector.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.serve_speed import Court  # noqa: E402
from analysis.flight3d import Camera  # noqa: E402

HALF = 6  # frames each side


def frames_at(src, t_center, crop, fps=59.4):
    """Native frames (with pts) in [t-HALF, t+HALF] frames, cropped to crop=(x, y, w, h) in source px."""
    x, y, w, h = crop
    t0 = t_center - (HALF + 1.5) / fps
    t1 = t_center + (HALF + 1.5) / fps
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info", "-ss", f"{t0:.4f}", "-copyts", "-i", src,
           "-to", f"{t1:.4f}", "-an", "-vf", f"crop={w}:{h}:{x}:{y},showinfo", "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    with tempfile.TemporaryFile() as err:
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=err).stdout
        err.seek(0)
        log = err.read().decode("utf-8", "replace")
    ts = [float(m) for m in re.findall(r"\bpts_time:\s*([-\d.]+)", log)]
    size = w * h * 3
    fr = [np.frombuffer(out, np.uint8, size, k).reshape(h, w, 3) for k in range(0, len(out) - size + 1, size)]
    n = min(len(fr), len(ts))
    pairs = sorted(zip(ts[:n], fr[:n]), key=lambda p: abs(p[0] - t_center))[:2 * HALF + 1]
    return sorted(pairs, key=lambda p: p[0])


def strip(pairs, crop, scale, grid=False, cols=7):
    x0, y0, w, h = crop
    tiles = []
    for t, im in pairs:
        im = cv2.resize(im, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        if grid:
            step = 20
            for gx in range((x0 // step + 1) * step, x0 + w, step):
                X = int((gx - x0) * scale)
                cv2.line(im, (X, 0), (X, 6 if gx % 100 else 14), (0, 255, 255), 1)
                if gx % 100 == 0:
                    cv2.putText(im, str(gx), (X + 2, 26), 0, 0.35, (0, 255, 255), 1)
            for gy in range((y0 // step + 1) * step, y0 + h, step):
                Y = int((gy - y0) * scale)
                cv2.line(im, (0, Y), (6 if gy % 100 else 14, Y), (0, 255, 255), 1)
                if gy % 100 == 0:
                    cv2.putText(im, str(gy), (16, Y + 4), 0, 0.35, (0, 255, 255), 1)
        cv2.putText(im, f"{t:.3f}", (4, im.shape[0] - 6), 0, 0.5, (255, 255, 255), 2)
        tiles.append(im)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    p = json.loads((ROOT / "projects" / "current_project.json").read_text(encoding="utf-8"))
    f = p["files"][0]
    W, H = int(f["metadata"]["width"]), int(f["metadata"]["height"])
    court = Court(f["calibration"]["points"])
    cam = Camera(court)
    serves = json.loads((ROOT / "cache" / f"{f['id']}_serves.json").read_text(encoding="utf-8"))["serves"]
    for k, s in enumerate(serves, 1):
        sx, sy = s["server_court"]
        h = s["contact_height_m"] if 1.5 < s["contact_height_m"] < 3.5 else 2.6
        u, v = cam.project(np.array([sx, sy, h]))[0]
        cx, cy = int(u / 640 * W), int(v / 360 * H)
        near = s["server_end"] == "near"
        cw, ch = (700, 700) if near else (360, 360)
        crop = (max(0, min(W - cw, cx - cw // 2)), max(0, min(H - ch, cy - ch // 2)), cw, ch)
        pairs = frames_at(f["path"], s["contact_t"], crop)
        cv2.imwrite(str(out / f"{k:02d}_contact.jpg"), strip(pairs, crop, 260 / cw))
        bx, by = s["bounce_court"]
        bu, bv = court.to_image(bx, by)
        bw, bh = (520, 300) if near else (700, 400)
        bcrop = (max(0, min(W - bw, int(bu * W) - bw // 2)), max(0, min(H - bh, int(bv * H) - bh // 2)), bw, bh)
        pairs = frames_at(f["path"], s["bounce_t"], bcrop)
        cv2.imwrite(str(out / f"{k:02d}_bounce.jpg"), strip(pairs, bcrop, 400 / bw, grid=True))
        print(k, s.get("point_no"), s.get("serve_number"), s["server_end"], s["contact_t"], s["bounce_t"], crop, bcrop, flush=True)


if __name__ == "__main__":
    main()
