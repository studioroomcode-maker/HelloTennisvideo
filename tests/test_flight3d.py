import math
import unittest

import numpy as np

from analysis.serve_speed import Court, COURT_L
from analysis.flight3d import Camera, simulate, ground_hit, fit_serve


def synthetic_camera():
    """Pinhole roughly where the sample match was filmed: behind the near baseline, 3.6 m up."""
    import cv2
    f, cx, cy = 320.0, 320.0, 180.0
    C = np.array([6.2, -5.5, 3.6])
    target = np.array([5.5, 14.0, 0.0])
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    t = -R @ C

    def project(X):
        Xc = np.atleast_2d(X) @ R.T + t
        return np.column_stack([f * Xc[:, 0] / Xc[:, 2] + cx, f * Xc[:, 1] / Xc[:, 2] + cy])
    return project, C


class Flight3DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        project, cls.C = synthetic_camera()
        cls.project = staticmethod(project)
        corners = np.array([[0, 0, 0], [10.97, 0, 0], [10.97, COURT_L, 0], [0, COURT_L, 0]], float)
        px = cls.project(corners)
        cls.court = Court([{"x": u / 640, "y": v / 360} for u, v in px])
        cls.cam = Camera(cls.court)

    def test_camera_recovered_from_four_corners(self):
        self.assertLess(np.linalg.norm(self.cam.C - self.C), 0.15)
        X = np.array([[5.0, 12.0, 1.0], [2.0, 20.0, 2.5]])
        self.assertLess(np.abs(self.cam.project(X) - self.project(X)).max(), 1.0)

    def test_ground_hit_under_gravity_and_drag(self):
        tb, bx, by = ground_hit([5.0, 0.0, 2.5], [0.0, 30.0, -1.0])
        self.assertTrue(0.4 < tb < 0.9)
        self.assertAlmostEqual(bx, 5.0, delta=1e-6)
        self.assertGreater(by, 12.0)

    def test_near_serve_fit_recovers_speed_and_bounce(self):
        p0 = np.array([9.0, 0.0, 2.5])
        v0 = np.array([-6.0, 33.0, -1.5])  # ~123 km/h, cross-court
        times = list(np.arange(0.25, 0.6, 1 / 59.4))  # flight seen only once it is past the net
        sim, _ = simulate(p0, v0, times)
        uv = self.project(np.array(sim))
        contact_px = self.project(p0)[0]
        pts = [(t, u, v) for t, (u, v) in zip(times, uv)]
        fit = fit_serve(self.cam, contact_px, 0.0, "near", pts)
        self.assertTrue(fit and fit["ok"], fit)
        self.assertAlmostEqual(fit["speed_kmh"], np.linalg.norm(v0) * 3.6, delta=3)
        tb, bx, by = ground_hit(p0, v0)
        self.assertAlmostEqual(fit["bounce_court"][0], bx, delta=0.3)
        self.assertAlmostEqual(fit["bounce_court"][1], by, delta=0.4)
        self.assertAlmostEqual(fit["bounce_t"], tb, delta=0.03)


if __name__ == "__main__":
    unittest.main()


class NetCrossingTests(unittest.TestCase):
    def test_low_serve_hits_the_net(self):
        from analysis.flight3d import crossing
        low = crossing([9.0, 0.0, 2.5], [-6.0, 30.0, -2.5], 11.885)
        high = crossing([9.0, 0.0, 2.5], [-6.0, 30.0, 0.0], 11.885)
        self.assertLess(low[2], 0.914)
        self.assertGreater(high[2], 0.914)
        # lands before it even reaches the net: no crossing (treated as a net fault by the caller)
        self.assertIsNone(crossing([9.0, 0.0, 2.5], [-6.0, 30.0, -6.0], 11.885))
