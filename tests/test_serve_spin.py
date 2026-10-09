import unittest

import numpy as np

from analysis.flight3d import Camera
from analysis.serve_speed import Court, COURT_L
from analysis.serve_spin import fly, measure_spin

# same synthetic camera as the other 3D tests: corners of the court in a 640x360 view
CAL = [{"x": 0.16, "y": 0.92}, {"x": 0.84, "y": 0.92}, {"x": 0.62, "y": 0.42}, {"x": 0.38, "y": 0.42}]


class ServeSpinTests(unittest.TestCase):
    def _serve(self, side_acc, up_acc=0.0):
        cam = Camera(Court(CAL))
        p0 = np.array([4.0, COURT_L, 2.6])          # far server, deuce side
        v0 = np.array([1.0, -26.0, -1.5])
        times = list(np.arange(0.02, 0.9, 1 / 60))
        pos = fly(p0, v0, (side_acc, up_acc), times + [2.0])
        hit = next(i for i, q in enumerate(pos) if q[2] <= 0)  # first sample on/under the ground
        T = times[hit - 1] if hit > 0 else times[-1]
        pts = [(t, *cam.project(np.array([q]))[0]) for t, q in zip(times, pos) if t < T]
        bounce = pos[hit - 1][:2] if hit > 0 else pos[-2][:2]
        return measure_spin(cam, p0, bounce, T, pts, 0.0, "far")

    def test_slice_bends_to_servers_left(self):
        # far server faces the camera, so their left is +x in court coordinates
        r = self._serve(side_acc=+5.0)
        self.assertIsNotNone(r)
        self.assertLess(r["spin_side_ms2"], -3.0)
        self.assertEqual(r["serve_type_measured"], "slice")

    def test_straight_serve_is_not_called_slice(self):
        r = self._serve(side_acc=0.0)
        self.assertIsNotNone(r)
        self.assertLess(abs(r["spin_side_ms2"]), 1.0)
        self.assertIsNone(r["serve_type_measured"])  # measured straight, but "flat" is not called until validated


if __name__ == "__main__":
    unittest.main()
