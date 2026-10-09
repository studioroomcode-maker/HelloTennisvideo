"""Serve type from the measured flight: how much the ball bends between two pinned ends.

The contact point (camera ray meeting the server's baseline plane) and the bounce (court
plane) are already measured, so the flight between them is fitted with gravity + drag + a
constant extra acceleration (a_side, a_up), standing in for the Magnus force, while the
trajectory is forced through the measured bounce at the measured time. Pinning both ends
removes the single-camera depth ambiguity that made a free spin term unstable before.

a_side is in the server's frame (+ = the server's right). For a right-handed server a
slice bends to the server's left and a kick dips and breaks right. On the hand-labelled
serves of the test video the three far-side slices measured a_side -4.4 to -5.1 m/s^2
with the fit error falling from 4-5 px to under 1.1 px; fits worse than MAX_RMS (mostly
near-side serves flying straight away from the camera) are left unclassified.

The contact point itself is only estimated (the toss top / racket sound and an assumed
height), and for near serves an error there of a few decimetres spoiled the fit (3-6 px).
So it may move within CONTACT_FREE metres, at a small cost (CONTACT_COST px per metre): all
serves with a bounce then fit within 1 px, and 7 of the 9 hand-labelled slices measured as
slices (3 before). A fit that needs the contact moved to the edge of that box is not trusted.
"""
from __future__ import annotations

import numpy as np

from analysis.flight3d import G, K_DRAG

BOUNCE_WEIGHT = 40.0  # 1 m away from the measured bounce costs as much as 40 px of image error
MAX_RMS = 1.5         # px (640x360): only fits this tight are classified
SIDE_MS2 = 2.5        # sideways bend that counts as sidespin
FLAT_MS2 = 0.0        # "flat" is not called yet: the test video has no flat serve with a measured flight,
                      # and its near-side slices bent only 0.7-1.9 m/s^2 (one read as flat at 1.2)
DIP_MS2 = 3.0         # extra downward pull that counts as topspin (kick)
CONTACT_FREE = (1.5, 1.0, 0.6)  # m the contact point may move (sideways, along the court, up)
CONTACT_COST = 3.0    # px of image error per metre the contact point moves


def fly(p0, v0, acc, times):
    """Positions at the given times (s after contact): gravity, drag and constant extra acceleration."""
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


def _fit(cam, p0, bounce, T, pts, spin, start):
    from scipy.optimize import least_squares
    ts = [q[0] for q in pts]
    uv = np.array([[q[1], q[2]] for q in pts])
    end = np.array([bounce[0], bounce[1], 0.0])

    n = 5 if spin else 3  # x = velocity (3), [extra acceleration (2)], contact shift (3)

    def resid(x):
        pos = fly(np.asarray(p0) + x[n:], x[:3], x[3:5] if spin else (0.0, 0.0), ts + [T])
        return np.r_[(cam.project(pos[:-1]) - uv).ravel(), (pos[-1] - end) * BOUNCE_WEIGHT, x[n:] * CONTACT_COST]

    free = list(CONTACT_FREE)
    lo = [-60, -80, -40] + ([-25, -25] if spin else []) + [-f for f in free]
    hi = [60, 80, 40] + ([25, 25] if spin else []) + free
    start = np.r_[start, np.zeros(3)] if len(start) == n else np.asarray(start)
    sol = least_squares(resid, start, loss="soft_l1", f_scale=2.0, bounds=(lo, hi))
    r = sol.fun[:2 * len(pts)].reshape(-1, 2)
    miss = float(np.linalg.norm(sol.fun[2 * len(pts):2 * len(pts) + 3]) / BOUNCE_WEIGHT)
    return sol, float(np.sqrt((r ** 2).sum(1).mean())), miss


def measure_spin(cam, p0, bounce, T, flight, contact_t, server_end, right_handed=True):
    """flight: [(t, u, v, ...)] image points (640x360) between contact and bounce. Returns a dict or None."""
    pts = [(q[0] - contact_t, q[1], q[2]) for q in flight if 0.03 < q[0] - contact_t < T - 0.01]
    if len(pts) < 8 or T <= 0.2:
        return None
    p0 = np.asarray(p0, float)
    v_guess = (np.array([bounce[0], bounce[1], 0.0]) - p0) / T
    v_guess[2] += 0.5 * G * T
    sol0, rms0, miss0 = _fit(cam, p0, bounce, T, pts, False, v_guess)
    best = None
    for ax in (0.0, -6.0, 6.0):  # the fit can settle in a wrong valley; keep the best start
        cand = _fit(cam, p0, bounce, T, pts, True, np.r_[sol0.x[:3], ax, 0.0, sol0.x[3:]])
        if best is None or cand[0].cost < best[0].cost:
            best = cand
    sol1, rms1, miss1 = best
    sgn = 1.0 if server_end == "near" else -1.0  # server's right is +x for the near server
    a_side, a_up = float(sgn * sol1.x[3]), float(sol1.x[4])
    shift = sol1.x[5:8]
    at_edge = any(abs(d) >= f - 0.02 for d, f in zip(shift, CONTACT_FREE))
    kind = None
    if rms1 <= MAX_RMS and miss1 <= 0.3 and not at_edge:
        hand = 1.0 if right_handed else -1.0
        side = a_side * hand  # + = bends to the hitter's forehand side
        if side <= -SIDE_MS2:
            kind = "slice"
        elif a_up <= -DIP_MS2 or (side >= SIDE_MS2 and a_up <= -1.0):
            kind = "kick"
        elif abs(side) < FLAT_MS2 and abs(a_up) < DIP_MS2:
            kind = "flat"
    return {"spin_side_ms2": round(a_side, 1), "spin_up_ms2": round(a_up, 1),
            "spin_fit_px": round(rms1, 2), "nospin_fit_px": round(rms0, 2),
            "spin_bounce_miss_m": round(miss1, 2), "nospin_bounce_miss_m": round(miss0, 2),
            "contact_shift_m": [round(float(d), 2) for d in shift],
            "serve_type_measured": kind}
