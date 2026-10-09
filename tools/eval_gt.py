"""Score serve measurement against the hand-verified ground truth (cache/<file>_serves_gt.json).

Usage: python tools/eval_gt.py [--no-audio] DETECTOR [DETECTOR ...]
DETECTOR: a weights file (tracknet .pt or wasb .pth.tar) or "a+b" for a max-ensemble.
Per serve: contact/bounce timing error, speed error vs the ground-truth speed, call agreement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.tracknet import BallDetector  # noqa: E402
from analysis.wasb import WASBDetector, EnsembleDetector  # noqa: E402
from analysis.serve_speed import ServeMeter, Court  # noqa: E402
from analysis.serve_measure_job import refine_with_audio  # noqa: E402


def detector(spec):
    def one(w):
        return WASBDetector(Path(w)) if "wasb" in Path(w).name else BallDetector(Path(w))
    parts = spec.split("+")
    return one(parts[0]) if len(parts) == 1 else EnsembleDetector(*[one(w) for w in parts])


def main():
    args = sys.argv[1:]
    use_audio = "--no-audio" not in args
    specs = [a for a in args if not a.startswith("--")]
    p = json.loads((ROOT / "projects" / "current_project.json").read_text(encoding="utf-8"))
    f = p["files"][0]
    gt = json.loads((ROOT / "cache" / f"{f['id']}_serves_gt.json").read_text(encoding="utf-8"))
    court = Court(f["calibration"]["points"])
    pts = sorted((x for x in p["points"] if x["file_id"] == f["id"]), key=lambda x: x["start"])
    for spec in specs:
        meter = ServeMeter(detector(spec), court, f["path"], f["metadata"]["fps"])
        serves = []
        for n, pt in enumerate(pts, 1):
            for k, s in enumerate(meter.at_point(pt["start"], pt["end"]), 1):
                s.update(point_no=n, serve_number=k)
                serves.append(s)
        audio = refine_with_audio(serves, meter) if use_audio else {"used": False}
        rows, sp_err, calls_ok, matched = [], [], 0, set()
        for g in gt["serves"]:
            m = min((s for s in serves if abs(s["contact_t"] - g["contact_t"]) < 0.5),
                    key=lambda s: abs(s["contact_t"] - g["contact_t"]), default=None)
            if m is None:
                rows.append(f"#{g['point_no']}.{g['serve_number']} missed (gt {g.get('speed_kmh')} {g['call']})")
                continue
            matched.add(id(m))
            dc = (m["contact_t"] - g["contact_t"]) * 1000
            call_ok = m["call"] == g["call"] or {m["call"], g["call"]} <= {"in", "close"}
            calls_ok += call_ok
            if g.get("speed_kmh"):
                e = m["speed_kmh"] - g["speed_kmh"]
                sp_err.append(abs(e))
                rows.append(f"#{g['point_no']}.{g['serve_number']} {m['speed_kmh']:>3} vs {g['speed_kmh']:>3} ({e:+d}) "
                            f"contact {dc:+.0f}ms [{m['contact_src']}] call {m['call']}/{g['call']}")
            else:
                rows.append(f"#{g['point_no']}.{g['serve_number']} call {m['call']}/{g['call']} (no gt speed)")
        extra = [s for s in serves if id(s) not in matched]
        print(f"== {'+'.join(Path(x).name for x in spec.split('+'))} | audio {audio}")
        print(f"   speed MAE {sum(sp_err) / max(1, len(sp_err)):.1f} km/h over {len(sp_err)} | calls {calls_ok}/{len(gt['serves'])}"
              f" | extra (not serves) {len(extra)}")
        for r in rows:
            print("   " + r)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
