"""Compare ball-tracking weights on the labelled points of the project (held out from training).

Usage: python tools/eval_serves.py WEIGHTS [WEIGHTS ...]
Prints, per weights file: serves measured, and how much of each point's first 3 s the
ball is on a linked track (coverage) and how many holes those tracks have.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.tracknet import BallDetector  # noqa: E402
from analysis.wasb import WASBDetector, EnsembleDetector  # noqa: E402
from analysis.serve_speed import (ServeMeter, Court, read_window, heat_to_blobs, remove_static,  # noqa: E402
                                  link_tracks)


def main():
    p = json.loads((ROOT / "projects" / "current_project.json").read_text(encoding="utf-8"))
    f = p["files"][0]
    court = Court(f["calibration"]["points"])
    pts = [x for x in p["points"] if x["file_id"] == f["id"]]
    for w in sys.argv[1:]:
        if "+" in w:  # "a.pt+b.pth.tar": max-ensemble
            det = EnsembleDetector(*[WASBDetector(Path(x)) if "wasb" in Path(x).name else BallDetector(Path(x))
                                     for x in w.split("+")])
        else:
            det = WASBDetector(Path(w)) if "wasb" in Path(w).name else BallDetector(Path(w))
        meter = ServeMeter(det, court, f["path"], f["metadata"]["fps"])
        found, cov, holes, rows = 0, [], 0, []
        for n, pt in enumerate(pts, 1):
            res = meter.at_point(pt["start"], pt["end"])
            found += bool(res)
            rows.append(f"#{n}:" + (",".join(f"{r['speed_kmh']}{r['call'][0]}" for r in res) or "-"))
            frames, times = read_window(f["path"], pt["start"] - 0.5, pt["start"] + 3.0)
            tracks = [t for t in link_tracks(remove_static([heat_to_blobs(h) for h in det.heat_levels(frames)], 59.4),
                                             times) if len(t) >= 6]
            on = sum(len(t) for t in tracks)
            cov.append(on / max(1, len(times)))
            for t in tracks:
                holes += sum(1 for a, b in zip(t, t[1:]) if b[0] - a[0] > 0.03)
        print(f"{Path(w).name}: serves {found}/{len(pts)}  coverage {sum(cov) / len(cov):.3f}  holes {holes}  "
              + " ".join(rows), flush=True)


if __name__ == "__main__":
    main()
