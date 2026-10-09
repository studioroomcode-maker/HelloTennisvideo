"""Ground-truth serve flights for the sample match, independent of any ball detector.

Contact times were read frame by frame from the 4K source (racket meets ball; strips made with
tools/gt_view.py). The contact *position* and the bounce are measured here with a colour-only
ball trace (tools/gt_view.trace) seeded at points picked by eye. Speeds use the same physics
as the app (distance / flight time with quadratic drag), so differences come from the inputs.

Usage: python tools/gt_serves.py  -> writes cache/<file>_serves_gt.json and prints the table
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from analysis.serve_speed import Court, launch_speed, serve_box, judge_in  # noqa: E402
from analysis.flight3d import Camera  # noqa: E402
from gt_view import trace  # noqa: E402

# point no, serve no, server end, contact time (manual), contact seed px (4K), bounce seed (t, px)
LABELS = [
    (1, 1, "far", 34.668, (1835, 767), (35.539, (2111, 1199))),
    (2, 1, "far", 48.500, (2096, 761), (49.530, (1702, 1217))),
    (3, 1, "far", 65.490, (1839, 769), (66.429, (2178, 1217))),
    (4, 1, "far", 81.610, (2074, 764), (82.612, (1496, 1306))),
    (5, 1, "far", 104.245, (1882, 767), (105.231, (2353, 1199))),
    (6, 1, "near", 124.651, None, (125.475, (1834, 977))),
    (7, 1, "near", 144.645, None, (145.433, (2180, 935))),
    (7, 2, "near", 149.520, None, (150.321, None)),
    (8, 1, "near", 165.352, None, (166.122, (1933, 1003))),
    (9, 1, "near", 179.872, None, (180.548, None)),
    (10, 1, "near", 200.280, None, (200.834, None)),
]
NOT_SERVES = [(3, 69.052, "rally forehand, not a serve")]


def bounce_from_trace(pts):
    """Bounce = intersection of the lines through the 3 points before and after the lowest point."""
    pts = [p for p in pts if p[1] is not None]
    if len(pts) < 7:
        return None
    k = max(range(len(pts)), key=lambda i: pts[i][2])
    pre, post = pts[max(0, k - 3):k + 1], pts[k:k + 4]
    if len(pre) < 3 or len(post) < 3:
        return None
    t1, y1 = np.array([p[0] for p in pre]), np.array([p[2] for p in pre])
    t2, y2 = np.array([p[0] for p in post]), np.array([p[2] for p in post])
    b1, a1 = np.polyfit(t1, y1, 1)
    b2, a2 = np.polyfit(t2, y2, 1)
    tb = (a2 - a1) / (b1 - b2) if b1 != b2 else pts[k][0]
    if not (pre[0][0] <= tb <= post[-1][0]):
        tb = pts[k][0]
    xs = [p for p in pts if abs(p[0] - tb) < 0.03]
    xb = float(np.interp(tb, [p[0] for p in xs], [p[1] for p in xs])) if len(xs) >= 2 else pts[k][1]
    yb = float(a1 + b1 * tb)
    return tb, xb, yb, b2 - b1


def main():
    p = json.loads((ROOT / "projects" / "current_project.json").read_text(encoding="utf-8"))
    f = p["files"][0]
    W, H = int(f["metadata"]["width"]), int(f["metadata"]["height"])
    court = Court(f["calibration"]["points"])
    cam = Camera(court)
    serves = json.loads((ROOT / "cache" / f"{f['id']}_serves.json").read_text(encoding="utf-8"))["serves"]
    rows = []
    for pno, sno, end, tc, cseed, (tb0, bseed) in LABELS:
        meas = next((s for s in serves if s.get("point_no") == pno and s.get("serve_number") == sno), None)
        # contact pixel
        if cseed is None:
            sx, sy = meas["server_court"]
            h = meas["contact_height_m"] if 1.5 < meas["contact_height_m"] < 3.5 else 2.6
            u, v = cam.project(np.array([sx, sy, h]))[0]
            cseed = (int(u / 640 * W), int(v / 360 * H))
        ct = trace(tc, 0.02, cseed[0], cseed[1], r=90, box=500)
        cpt = min((q for q in ct if q[1] is not None), key=lambda q: abs(q[0] - tc), default=None)
        base_y = 0.0 if end == "near" else 23.77
        p0 = cam.on_plane_y(cpt[1] / W * 640, cpt[2] / H * 360, base_y) if cpt else None
        # bounce
        if bseed is None:
            bx, by = meas["bounce_court"]
            u, v = court.to_image(bx, by)
            bseed = (int(u * W), int(v * H))
        b = bounce_from_trace(trace(tb0, 0.12, bseed[0], bseed[1], r=45))
        row = {"point_no": pno, "serve_number": sno, "server_end": end, "contact_t": tc}
        if b and p0 is not None:
            tb, xb, yb, kink = b
            bcx, bcy = court.to_court(xb / W, yb / H)
            horiz = math.hypot(bcx - p0[0], bcy - p0[1])
            dist = math.hypot(horiz, float(p0[2]))
            T = tb - tc
            xr, yr = serve_box(end, float(p0[0]))
            row.update(bounce_t=round(tb, 3), flight_s=round(T, 3), contact_3d=[round(float(c), 2) for c in p0],
                       bounce_court=[round(bcx, 2), round(bcy, 2)], distance_m=round(dist, 2),
                       speed_kmh=round(launch_speed(dist, T) * 3.6), call=judge_in(bcx, bcy, xr, yr))
        rows.append(row)
        print(row, flush=True)
    out = ROOT / "cache" / f"{f['id']}_serves_gt.json"
    out.write_text(json.dumps({"serves": rows, "not_serves": NOT_SERVES}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
