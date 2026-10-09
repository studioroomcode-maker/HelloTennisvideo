"""Build a pseudo-labelled ball dataset from match footage for fine-tuning TrackNet.

Usage:
    python tools/ball_dataset.py OUT_DIR VIDEO[:START-END] [VIDEO[:START-END] ...]

For windows sampled across each video (native ~60 fps, 640x360):
* confident labels — points of long linked tracks (what the stock model already sees);
* hard labels      — frames inside short holes of a bridged track, filled from the
                     motion on both sides (blur / occlusion frames the model misses).
Unlabelled frames are not used as "no ball" targets: a miss is not evidence of absence.
Writes OUT_DIR/frames/*.jpg and OUT_DIR/labels.jsonl (one sample per labelled frame).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.tracknet import BallDetector, ensure_weights  # noqa: E402
from analysis.serve_speed import (read_window, heat_to_blobs, remove_static, link_tracks,  # noqa: E402
                                  bridge_tracks, _slope)

WINDOW = 3.0
STRIDE = 12.0
LONG_TRACK = 12
MAX_FILL = 8  # frames


def _fit_at(pts, t, axis):
    """Value at t from a line through up to 5 points (time, value)."""
    b, a = _slope(pts, axis)
    return a + b * t if b is not None else None


def label_window(frames, times, det):
    heats = det.heat_levels(frames)
    blobs = [heat_to_blobs(h) for h in heats]
    tracks = link_tracks(remove_static(blobs, 59.4), times)
    labels = {}
    # confident: long tracks
    for tr in tracks:
        if len(tr) >= LONG_TRACK:
            for t, x, y in tr:
                labels[t] = (x, y, "track")
    # hard: holes in bridged tracks
    for tr in bridge_tracks(tracks):
        if len(tr) < LONG_TRACK:
            continue
        for a, b in zip(tr, tr[1:]):
            missing = [t for t in times if a[0] < t < b[0]]
            if not (1 <= len(missing) <= MAX_FILL):
                continue
            before = [p for p in tr if p[0] <= a[0]][-5:]
            after = [p for p in tr if p[0] >= b[0]][:5]
            if len(before) < 3 or len(after) < 3:
                continue
            for t in missing:
                # each side's motion extrapolated, blended by distance in time
                w = (t - a[0]) / (b[0] - a[0])
                xa, ya = _fit_at(before, t, 1), _fit_at(before, t, 2)
                xb, yb = _fit_at(after, t, 1), _fit_at(after, t, 2)
                if None in (xa, ya, xb, yb) or abs(xa - xb) > 12 or abs(ya - yb) > 12:
                    continue  # the two sides disagree (bounce/hit inside the hole): skip
                labels[t] = (xa * (1 - w) + xb * w, ya * (1 - w) + yb * w, "fill")
    return labels


def main():
    import cv2
    out = Path(sys.argv[1])
    (out / "frames").mkdir(parents=True, exist_ok=True)
    det = BallDetector(ensure_weights(ROOT))
    n_track = n_fill = 0
    with (out / "labels.jsonl").open("a", encoding="utf-8") as fp:
        for spec in sys.argv[2:]:
            path, _, rng = spec.partition(":")
            vid = Path(path).stem
            import subprocess
            dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                        "csv=p=0", path], capture_output=True, text=True).stdout.strip())
            t0, t1 = (float(v) for v in rng.split("-")) if rng else (0.0, dur)
            t = t0
            while t + WINDOW <= t1:
                frames, times = read_window(path, t, t + WINDOW)
                labels = label_window(frames, times, det)
                keep = set()
                for i, tt in enumerate(times):
                    if tt in labels and i >= 2:
                        keep.update((i, i - 1, i - 2))
                names = {}
                for i in sorted(keep):
                    name = f"{vid}_{times[i]:.3f}.jpg"
                    names[i] = name
                    cv2.imwrite(str(out / "frames" / name), frames[i], [cv2.IMWRITE_JPEG_QUALITY, 92])
                for i, tt in enumerate(times):
                    if tt in labels and i >= 2:
                        x, y, kind = labels[tt]
                        fp.write(json.dumps({"frames": [names[i], names[i - 1], names[i - 2]], "x": round(x, 2),
                                             "y": round(y, 2), "kind": kind, "video": vid, "t": tt}) + "\n")
                        n_track += kind == "track"
                        n_fill += kind == "fill"
                print(f"{vid} {t:7.1f}s  labels so far: track {n_track}  fill {n_fill}", flush=True)
                t += STRIDE


if __name__ == "__main__":
    main()
