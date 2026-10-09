"""Experiment: how much does each measured serve curve sideways / dip, with both ends pinned?

For every serve in cache/<file>_serves.json the contact point (3D, from the camera ray and
the server's baseline) and the bounce (court plane, homography) are already measured. The
flight in between is fitted with gravity + drag + a constant extra acceleration
(a_side, a_up) -- a stand-in for the Magnus force -- while forcing the trajectory through
the measured bounce at the measured time. With both ends pinned the single-camera depth
ambiguity is gone, so the extra acceleration is what bends the observed image path.

a_side is reported in the server's frame (+ = server's right). A right-handed slice curves
to the server's left (a_side < 0); a kick dips (a_up < 0) and kicks right after the bounce.

    python tools/serve_spin.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.flight3d import Camera, G, K_DRAG  # noqa: E402
from analysis.serve_speed import Court, ServeMeter, measure_flights, COURT_L  # noqa: E402
from analysis.shot_analysis import pick_source  # noqa: E402
from analysis.tracknet import BallDetector, active_weights  # noqa: E402

PX_PER_M = 40.0  # weight of the bounce constraint: 1 m off the measured bounce costs like 40 px


def fly(p0, v0, acc, times):
    """Positions at the given times with gravity, drag and a constant extra acceleration."""
    dt = 0.002
    p, v = np.array(p0, float), np.array(v0, float)
    order = np.argsort(times)
    out = [None] * len(times)
    t, k = 0.0, 0
    tmax = max(times)
    while k < len(order):
        while k < len(order) and times[order[k]] <= t + 1e-9:
            out[order[k]] = p.copy()
            k += 1
        a = np.array([acc[0], 0.0, -G + acc[1]]) - K_DRAG * np.linalg.norm(v) * v
        v = v + a * dt
        p = p + v * dt
        t += dt
        if t > tmax + 0.01:
            break
    return np.array([o if o is not None else p for o in out])


def fit(cam, p0, bounce, T, pts, spin=True, start=None):
    from scipy.optimize import least_squares
    ts = np.array([q[0] for q in pts])
    uv = np.array([[q[1], q[2]] for q in pts])
    v_guess = (np.array([bounce[0], bounce[1], 0.0]) - p0) / T
    v_guess[2] += 0.5 * G * T

    def resid(x):
        acc = x[3:5] if spin else (0.0, 0.0)
        pos = fly(p0, x[:3], acc, list(ts) + [T])
        img = (cam.project(pos[:-1]) - uv).ravel()
        end = (pos[-1] - np.array([bounce[0], bounce[1], 0.0])) * PX_PER_M
        return np.r_[img, end]

    x0 = (np.r_[v_guess, 0.0, 0.0] if spin else v_guess) if start is None else np.asarray(start, float)
    lo = [-60, -80, -40] + ([-25, -25] if spin else [])
    hi = [60, 80, 40] + ([25, 25] if spin else [])
    sol = least_squares(resid, x0, loss="soft_l1", f_scale=2.0, bounds=(lo, hi))
    r = sol.fun[:-3].reshape(-1, 2)
    sol.bounce_miss_m = float(np.linalg.norm(sol.fun[-3:]) / PX_PER_M)
    return sol.x, float(np.sqrt((r ** 2).sum(1).mean())), sol


def main():
    fid = sys.argv[1] if len(sys.argv) > 1 else "2d9348b629"
    project = json.loads((ROOT / "projects" / "current_project.json").read_text(encoding="utf-8"))
    f = next(x for x in project["files"] if x["id"] == fid)
    court = Court(f["calibration"]["points"])
    cam = Camera(court)
    meter = ServeMeter(BallDetector(active_weights(ROOT, lambda d: None)), court,
                       pick_source(f, "original", 0)[0], float(f["metadata"]["fps"]))
    serves = json.loads((ROOT / "cache" / f"{fid}_serves.json").read_text(encoding="utf-8"))["serves"]
    labels = [l for l in project.get("shot_labels", []) if l["file_id"] == fid and l["shot"] == "serve"]
    rows = []
    for s in serves:
        tc, tb = s["contact_t"], s["bounce_t"]
        tracks = meter.tracks(tc - 1.5, tb + 0.6)
        fl = [g for g in measure_flights(tracks, court, tc - 0.6, tc + 0.6) if abs(g["bounce_t"] - tb) < 0.08]
        if not fl or not s.get("contact_height_m") or not s.get("bounce_court"):
            print(f"{tc:7.2f} no flight", flush=True)
            continue
        run = fl[0]["_flight"]
        base_y = 0.0 if s["server_end"] == "near" else COURT_L
        p0 = np.array([s["server_court"][0], base_y, s["contact_height_m"]])
        T = tb - tc
        pts = [(q[0] - tc, q[1], q[2]) for q in run if tc + 0.03 < q[0] < tb - 0.01]
        if len(pts) < 6:
            print(f"{tc:7.2f} too few points ({len(pts)})", flush=True)
            continue
        x0, rms0, sol0 = fit(cam, p0, s["bounce_court"], T, pts, spin=False)
        # several starts (the fit can settle in a wrong valley); keep the best
        best = None
        for ax in (0.0, -6.0, 6.0):
            for az in (0.0, -6.0):
                x, r, sol = fit(cam, p0, s["bounce_court"], T, pts, spin=True, start=np.r_[x0, ax, az])
                if best is None or sol.cost < best[2].cost:
                    best = (x, r, sol)
        x1, rms1, sol1 = best
        sgn = 1.0 if s["server_end"] == "near" else -1.0  # server's right = +x for the near server
        a_side, a_up = sgn * x1[3], x1[4]
        lab = next((l for l in labels if abs(l["t"] - tc) < 1.2), None)
        row = {"t": tc, "end": s["server_end"], "speed": s["speed_kmh"], "call": s["call"], "n": len(pts),
               "rms_nospin": round(rms0, 2), "rms_spin": round(rms1, 2),
               "a_side": round(float(a_side), 1), "a_up": round(float(a_up), 1),
               "miss_nospin_m": round(sol0.bounce_miss_m, 2), "miss_spin_m": round(sol1.bounce_miss_m, 2),
               "label": (lab or {}).get("spin")}
        rows.append(row)
        print(f"{tc:7.2f} {s['server_end']:4s} {s['speed_kmh']:3d}km/h {s['call']:5s} n={len(pts):2d} "
              f"rms {rms0:5.2f} -> {rms1:5.2f}px  바운스 빗나감 {sol0.bounce_miss_m:4.2f} -> {sol1.bounce_miss_m:4.2f}m  옆 {a_side:+6.1f}  위 {a_up:+6.1f} m/s²  정답 {row['label']}", flush=True)
    out = ROOT / "cache" / f"{fid}_serve_spin.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
