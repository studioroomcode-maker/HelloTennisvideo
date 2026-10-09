"""3D serve flight from one calibrated camera.

The 4-point court homography is decomposed into a pinhole camera (focal length,
rotation, position) — the standard planar calibration with square pixels and the
principal point at the image centre. The serve contact is then a real 3D point:
where the camera ray through the contact pixel meets the vertical plane of the
server's baseline. Launch velocity is fitted so that a ball under gravity and
quadratic drag, started there, reprojects onto the tracked image points. That
gives the launch speed directly and the bounce (z = 0) even when the frames
around the bounce were lost.
"""
from __future__ import annotations

import math

import numpy as np

from analysis.serve_speed import IN_W, IN_H, K_DRAG, COURT_L

G = 9.81


class Camera:
    def __init__(self, court, w: int = IN_W, h: int = IN_H):
        self.w, self.h = w, h
        self.cx, self.cy = w / 2, h / 2
        S = np.diag([w, h, 1.0])
        T = np.array([[1, 0, -self.cx], [0, 1, -self.cy], [0, 0, 1.0]])
        Hc = T @ S @ np.asarray(court.Hinv, dtype=np.float64)
        h1, h2, h3 = Hc[:, 0], Hc[:, 1], Hc[:, 2]
        cands = []
        den = h1[2] * h2[2]
        if abs(den) > 1e-12:
            cands.append(-(h1[0] * h2[0] + h1[1] * h2[1]) / den)
        den = h1[2] ** 2 - h2[2] ** 2
        if abs(den) > 1e-12:
            cands.append(-((h1[0] ** 2 + h1[1] ** 2) - (h2[0] ** 2 + h2[1] ** 2)) / den)
        good = [c for c in cands if c > 0]
        if not good:
            raise ValueError("캘리브레이션에서 초점거리를 구할 수 없어")
        self.f = math.sqrt(sum(good) / len(good))
        Kinv = np.diag([1 / self.f, 1 / self.f, 1.0])
        a1, a2, a3 = Kinv @ h1, Kinv @ h2, Kinv @ h3
        lam = 2.0 / (np.linalg.norm(a1) + np.linalg.norm(a2))
        for sign in (1, -1):
            r1, r2, t = sign * lam * a1, sign * lam * a2, sign * lam * a3
            R = np.column_stack([r1, r2, np.cross(r1, r2)])
            U, _, Vt = np.linalg.svd(R)
            R = U @ Vt
            C = -R.T @ t
            if C[2] > 0:  # camera above the court
                break
        self.R, self.t, self.C = R, t, C
        self._refine(court)

    def _refine(self, court):
        """Pinhole fitted to the court plane as the homography sees it.

        With the principal point pinned to the image centre a homography can't always be
        reproduced (the plain decomposition was ~25 px off at a near corner), so the
        principal point may move, softly pulled back to the centre. The fit uses a grid
        over the whole court, which the homography is known to match well."""
        import cv2
        from scipy.optimize import least_squares
        gx, gy = np.meshgrid(np.linspace(0, 10.97, 7), np.linspace(0, COURT_L, 9))
        grid = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
        target = np.array([[a * self.w, b * self.h] for a, b in (court.to_image(x, y) for x, y, _ in grid)])
        rvec, _ = cv2.Rodrigues(self.R)

        def unpack(x):
            R, _ = cv2.Rodrigues(x[1:4])
            return x[0], R, x[4:7], x[7], x[8]

        def resid(x):
            f, R, t, cx, cy = unpack(x)
            Xc = grid @ R.T + t
            uv = np.column_stack([f * Xc[:, 0] / Xc[:, 2] + cx, f * Xc[:, 1] / Xc[:, 2] + cy])
            prior = np.array([(cx - self.w / 2) * 0.1, (cy - self.h / 2) * 0.1])
            return np.r_[(uv - target).ravel(), prior]

        sol = least_squares(resid, np.r_[self.f, rvec.ravel(), self.t, self.cx, self.cy],
                            x_scale=np.r_[self.f, 1, 1, 1, 1, 1, 1, 50, 50])
        f, R, t, cx, cy = unpack(sol.x)
        self.f, self.R, self.t, self.cx, self.cy = float(f), R, np.asarray(t, float), float(cx), float(cy)
        self.C = -self.R.T @ self.t
        r = sol.fun[:-2].reshape(-1, 2)
        self.calib_rms = float(np.sqrt((r ** 2).sum(1).mean()))
        self.calib_max = float(np.sqrt((r ** 2).sum(1)).max())

    def _proj(self, X, f, R, t):
        Xc = np.atleast_2d(X) @ R.T + t
        return np.column_stack([f * Xc[:, 0] / Xc[:, 2] + self.cx, f * Xc[:, 1] / Xc[:, 2] + self.cy])

    def project(self, X):
        return self._proj(X, self.f, self.R, self.t)

    def ray(self, u, v):
        d = self.R.T @ np.array([(u - self.cx) / self.f, (v - self.cy) / self.f, 1.0])
        return self.C, d / np.linalg.norm(d)

    def on_plane_y(self, u, v, y):
        """Point where the ray through pixel (u, v) crosses the vertical plane at court y."""
        o, d = self.ray(u, v)
        if abs(d[1]) < 1e-9:
            return None
        s = (y - o[1]) / d[1]
        return o + s * d if s > 0 else None


def simulate(p0, v0, times, spin_az=0.0):
    """Positions at the given times (s after contact) and the ground-hit time (or None)."""
    dt = 0.002
    p, v = np.array(p0, float), np.array(v0, float)
    out, t, k = [], 0.0, 0
    order = np.argsort(times)
    res = [None] * len(times)
    hit = None
    tmax = max(max(times, default=0), 2.0)
    while t <= tmax + dt:
        while k < len(order) and times[order[k]] <= t + 1e-9:
            res[order[k]] = p.copy()
            k += 1
        if hit is None and p[2] <= 0 and t > 0.05:
            hit = t
        a = np.array([0, 0, -G + spin_az]) - K_DRAG * np.linalg.norm(v) * v
        v = v + a * dt
        p = p + v * dt
        t += dt
    return [r if r is not None else p.copy() for r in res], hit


def ground_hit(p0, v0, spin_az=0.0):
    """(time, x, y) of the first ground contact, interpolated to the millisecond."""
    dt = 0.001
    p, v, t = np.array(p0, float), np.array(v0, float), 0.0
    while t < 3.0:
        a = np.array([0, 0, -G + spin_az]) - K_DRAG * np.linalg.norm(v) * v
        nv = v + a * dt
        npos = p + nv * dt
        if npos[2] <= 0 < p[2]:
            f = p[2] / (p[2] - npos[2])
            q = p + (npos - p) * f
            return t + dt * f, float(q[0]), float(q[1])
        p, v, t = npos, nv, t + dt
    return None


def fit_serve(cam: Camera, contact_px, contact_t: float, server_end: str, points, max_rms: float = 3.0):
    """Fit launch velocity to flight points [(t, u, v)] (640x360 px). Returns a dict or None.

    Points after the fitted bounce belong to the bounce/return, so they are dropped and the
    fit repeated (twice at most)."""
    pts = list(points)
    res = None
    for _ in range(3):
        res = _fit_once(cam, contact_px, contact_t, server_end, pts, max_rms)
        if not res or "bounce_t" not in res:
            return res
        keep = [p for p in pts if p[0] < res["bounce_t"] - 0.01]
        if len(keep) == len(pts) or len(keep) < 6:
            return res
        pts = keep
    return res


def _fit_once(cam: Camera, contact_px, contact_t: float, server_end: str, points, max_rms: float):
    from scipy.optimize import least_squares
    base_y = 0.0 if server_end == "near" else COURT_L
    p0 = cam.on_plane_y(contact_px[0], contact_px[1], base_y)
    if p0 is None or not (1.3 <= p0[2] <= 3.6) or not (-2.5 <= p0[0] <= 13.5):
        return None
    pts = [p for p in points if p[0] > contact_t + 0.02]
    if len(pts) < 6:
        return None
    ts = np.array([p[0] - contact_t for p in pts])
    uv = np.array([[p[1], p[2]] for p in pts])
    # initial guess: reach the ray of the last point near the ground, in the observed time
    last = cam.on_plane_y(uv[-1][0], uv[-1][1], base_y + (14.0 if server_end == "near" else -14.0))
    tgt = last if last is not None else p0 + np.array([0, 18.0 if server_end == "near" else -18.0, -2.0])
    v_guess = (tgt - p0) / max(0.2, ts[-1])

    # x = vx, vy, vz, contact-time shift. No spin term: with one camera it trades off
    # against depth and made the extrapolated bounce wander by metres in testing.
    def resid(x):
        sim, _ = simulate(p0, x[:3], list(ts + x[3]))
        return (cam.project(np.array(sim)) - uv).ravel()

    sol = least_squares(resid, np.r_[v_guess, 0.0], loss="soft_l1", f_scale=2.0,
                        bounds=([-60, -80, -40, -0.08], [60, 80, 40, 0.08]))
    r = sol.fun.reshape(-1, 2)
    rms = float(np.sqrt((r ** 2).sum(1).mean()))
    v0 = sol.x[:3]
    hit = ground_hit(p0, v0)
    if hit is None:
        return {"ok": False, "rms_px": round(rms, 2)}
    tb, bx, by = hit
    if rms > max_rms:
        return {"ok": False, "rms_px": round(rms, 2), "bounce_t": float(contact_t - sol.x[3] + tb)}
    return {"ok": True, "speed_kmh": round(float(np.linalg.norm(v0)) * 3.6), "rms_px": round(rms, 2),
            "bounce_t": round(contact_t - sol.x[3] + tb, 3), "bounce_court": [round(bx, 2), round(by, 2)],
            "contact_3d": [round(float(c), 2) for c in p0], "v0": [round(float(c), 2) for c in v0],
            "n_points": len(pts)}


def crossing(p0, v0, y_plane: float):
    """(t, x, z) where the flight crosses the vertical plane y = y_plane, or None."""
    dt = 0.001
    p, v, t = np.array(p0, float), np.array(v0, float), 0.0
    while t < 3.0:
        a = np.array([0, 0, -G]) - K_DRAG * np.linalg.norm(v) * v
        nv = v + a * dt
        npos = p + nv * dt
        if (p[1] - y_plane) * (npos[1] - y_plane) <= 0 and npos[1] != p[1]:
            f = (y_plane - p[1]) / (npos[1] - p[1])
            q = p + (npos - p) * f
            return t + dt * f, float(q[0]), float(q[2])
        if npos[2] < 0:
            return None
        p, v, t = npos, nv, t + dt
    return None
