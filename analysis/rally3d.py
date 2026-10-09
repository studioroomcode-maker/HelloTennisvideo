"""Rally hits the net-tape crossings miss, from 3D ball flights.

hit_detect.py reads the rally from where the ball crosses the net tape's row in the image.
A lob, and the smash that answers it, can stay above that row the whole way (the ball is
high up, so it shows high in the image on either side of the net), and then the exchange is
invisible to it. Here each tracked ball path is split into arcs that each fit one ballistic
flight in 3D (gravity + drag, through the calibrated camera). Between two arcs whose
velocity along the court has opposite signs the ball was hit, by the end it flies away
from: an arc heading for the far end was hit by a near player. The moment is the racket
sound nearest to where the two arcs, run towards each other, meet.

On the test video this found all seven hits of the 4:01-4:12 lob exchange (crossings found
three, one of them 0.9 s late). Its arcs are sparser elsewhere, so it only fills gaps.
"""
from __future__ import annotations

import numpy as np

from analysis.flight3d import Camera, G
from analysis.serve_speed import K_DRAG, NET_Y, Court

DT = 1 / 120                      # s, integration step (1/240 found the same hits on the test video, 40% slower)
LO = np.array([-8, -8, -0.5, -70, -70, -40.0])  # start position (m) and velocity (m/s) bounds
HI = np.array([19, 32, 15, 70, 70, 40.0])
MAX_RMS = 1.5                     # px (640x360) for a stretch of track to count as one flight
GOOD_RMS = 1.0                    # px for an arc to be used
MIN_VY = 3.0                      # m/s along the court; slower arcs are bounces in place or noise
MAX_GAP = 2.5                     # s between two arcs for them to be one hit apart
NEAR_LAG, FAR_LAG = 0.12, 0.17    # s the racket sound comes after the hit
FILL_S = 1.0                      # a 3D hit this close to a crossing hit is the same one


def fly(p0, v0, rel):
    """Positions at sorted times rel (s, >= 0) after the state (p0, v0)."""
    n = int(np.ceil(rel[-1] / DT)) + 1
    out = np.empty((n + 1, 3))
    p, v = np.array(p0, float), np.array(v0, float)
    g = np.array([0.0, 0.0, -G])
    for k in range(n + 1):
        out[k] = p
        v = v + (g - K_DRAG * np.sqrt(v @ v) * v) * DT
        p = p + v * DT
    f = rel / DT
    i = np.minimum(f.astype(int), n - 1)
    w = (f - i)[:, None]
    return out[i] * (1 - w) + out[i + 1] * w


def fly_back(p0, v0, seconds: float):
    p, v = np.array(p0, float), np.array(v0, float)
    g = np.array([0.0, 0.0, -G])
    for _ in range(int(seconds / DT)):
        a = g - K_DRAG * np.sqrt(v @ v) * v
        p = p - v * DT
        v = v - a * DT
    return p


class ArcFitter:
    def __init__(self, court: Court):
        self.cam = Camera(court)

    def _solve(self, rel, uv, x0):
        from scipy.optimize import least_squares

        def resid(x):
            return (self.cam.project(fly(x[:3], x[3:], rel)) - uv).ravel()
        sol = least_squares(resid, np.clip(x0, LO + 1e-3, HI - 1e-3), loss="soft_l1", f_scale=1.5,
                            bounds=(LO, HI))
        r = sol.fun.reshape(-1, 2)
        return sol, float(np.sqrt((r ** 2).sum(1).mean()))

    def _guesses(self, uv, ts):
        def at(u, v, z):
            o, d = self.cam.ray(u, v)
            s = (z - o[2]) / d[2] if abs(d[2]) > 1e-6 else 10.0
            return o + max(s, 1.0) * d
        for z0 in (1.0, 3.5):
            pa, pb = at(*uv[0], z0), at(*uv[-1], z0)
            yield np.r_[pa, (pb - pa) / max(1e-3, ts[-1] - ts[0])]

    def arc(self, pts, warm=None) -> dict:
        ts = np.array([p[0] for p in pts])
        uv = np.array([[p[1], p[2]] for p in pts])
        rel = ts - ts[0]
        best = None
        for x0 in ([warm] if warm is not None else []) + list(self._guesses(uv, ts)):
            sol, rms = self._solve(rel, uv, x0)
            if best is None or rms < best[1]:
                best = (sol, rms)
            if warm is not None and rms < GOOD_RMS:
                break
        sol, rms = best
        return {"t0": float(ts[0]), "t1": float(ts[-1]), "p": sol.x[:3], "v": sol.x[3:], "rms": rms, "n": len(pts)}

    def split(self, track, min_n: int = 8, step: int = 6) -> list[dict]:
        """Greedy split of one track into arcs that each fit one flight."""
        out, i = [], 0
        while len(track) - i >= min_n:
            j = i + min_n
            cur = self.arc(track[i:j])
            if cur["rms"] > MAX_RMS:
                i += 3
                continue
            while j < len(track):
                j2 = min(len(track), j + step)
                nxt = self.arc(track[i:j2], warm=np.r_[cur["p"], cur["v"]])
                if nxt["rms"] > MAX_RMS:
                    break
                cur, j = nxt, j2
            out.append(cur)
            i = j
        return out


def _good(a) -> bool:
    p, v = a["p"], a["v"]
    return (a["rms"] <= GOOD_RMS and -7.5 < p[0] < 18.5 and -7.5 < p[1] < 31.5 and p[2] < 14.5
            and abs(v[1]) >= MIN_VY)


def _meet(a, b):
    """(time, position) where arc a run forward and arc b run backward come closest."""
    best = None
    for t in np.arange(a["t1"] - 0.1, b["t0"] + 0.02, 0.02):
        pa = fly(a["p"], a["v"], np.array([max(0.0, t - a["t0"])]))[0]
        pb = fly_back(b["p"], b["v"], max(0.0, b["t0"] - t))
        d = float(np.linalg.norm(pa - pb))
        if best is None or d < best[0]:
            best = (d, float(t), (pa + pb) / 2)
    return best


def arc_hits(tracks, court: Court, onsets=None) -> list[dict]:
    """[{"t", "side", "window": [t_a, t_b], "meet": [x, y, z], "on_side", "plausible"}] from own-court tracks.

    on_side: the arcs meet over the hitter's half; plausible: also at a height a racket reaches."""
    fitter = ArcFitter(court)
    arcs = []
    for tr in tracks:
        if len(tr) >= 16:
            arcs += fitter.split(tr[::2])
    arcs = sorted((a for a in arcs if _good(a)), key=lambda a: a["t0"])
    sounds = [o for o in (onsets or []) if o["strength"] >= 3.0]
    out = []
    for a, b in zip(arcs, arcs[1:]):
        if not -0.05 <= b["t0"] - a["t1"] <= MAX_GAP or np.sign(a["v"][1]) == np.sign(b["v"][1]):
            continue
        side = "near" if b["v"][1] > 0 else "far"
        lag = NEAR_LAG if side == "near" else FAR_LAG
        _, tm, pm = _meet(a, b)
        cand = [o for o in sounds if a["t1"] - 0.1 <= o["t"] - lag <= b["t0"] + 0.05]
        if cand:
            o = max(cand, key=lambda o: o["strength"] / (1 + abs(o["t"] - lag - tm) / 0.3))
            t, s = o["t"], o["strength"]
        else:
            t, s = tm + lag, None
        on_side = pm[1] < NET_Y + 1.0 if side == "near" else pm[1] > NET_Y - 1.0
        out.append({"t": round(t, 3), "side": side, "audio": s, "window": [round(a["t1"], 2), round(b["t0"], 2)],
                    "meet": [round(float(c), 2) for c in pm], "on_side": bool(on_side),
                    "plausible": bool(on_side and -1.0 <= pm[2] <= 4.5)})
    return out


def merge(hits: list[dict], extra: list[dict], t_from: float, t_to: float, serves=None) -> list[dict]:
    """Fill crossing-based `hits` with 3D ones: re-time a crossing hit the 3D gap places elsewhere,
    add plausible 3D hits where no crossing hit is near. Serve tosses and faults are respected."""
    serves = serves or []
    faults = [(s["t"], min([u["t"] for u in serves if u["t"] > s["t"]] or [s["t"] + 4.0]))
              for s in serves if s.get("fault")]
    hits = [dict(h) for h in hits]
    for e in extra:
        if not t_from <= e["t"] <= t_to or any(f0 < e["t"] < f1 for f0, f1 in faults) \
                or any(s["t"] - 1.5 <= e["t"] <= s["t"] + 0.3 for s in serves):
            continue
        lag = NEAR_LAG if e["side"] == "near" else FAR_LAG
        same = [h for h in hits if not h.get("serve") and not h.get("arc_3d") and h["side"] == e["side"]
                and abs(h["t"] - e["t"]) <= 1.2]
        if same:
            h = min(same, key=lambda h: abs(h["t"] - e["t"]))
            outside = not e["window"][0] - 0.15 <= h["t"] - lag <= e["window"][1] + 0.1
            if outside and e["audio"] and e["on_side"]:
                h.update(t=e["t"], audio=e["audio"], retimed_3d=True)
            continue
        if e["plausible"] and not any(abs(h["t"] - e["t"]) <= FILL_S for h in hits):
            hits.append({"t": e["t"], "side": e["side"], "audio": e["audio"], "hole": e["window"],
                         "cross_t": None, "first": False, "serve": False, "arc_3d": True})
    return sorted(hits, key=lambda h: h["t"])
