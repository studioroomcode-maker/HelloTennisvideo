import unittest

import numpy as np

from analysis import rally3d as r3
from analysis.flight3d import Camera
from analysis.serve_speed import Court

CAL = [{"x": 0.058, "y": 0.838}, {"x": 0.936, "y": 0.841}, {"x": 0.605, "y": 0.420}, {"x": 0.419, "y": 0.430}]
FPS = 60.0


def flight(cam, p0, v0, t0, t1):
    ts = np.arange(0, t1 - t0, 1 / FPS)
    uv = cam.project(r3.fly(p0, v0, ts))
    return [(t0 + t, float(u), float(v)) for t, (u, v) in zip(ts, uv)]


class Rally3dTests(unittest.TestCase):
    def setUp(self):
        self.court = Court(CAL)
        self.cam = Camera(self.court)

    def test_lob_answered_by_the_near_player(self):
        # far player lobs towards the near court; the near player hits it back deep at t = 1.3
        lob = flight(self.cam, (5, 22, 1.0), (0, -13, 9), 0.0, 1.3)
        back = flight(self.cam, (5, 4.5, 1.6), (0, 18, 5), 1.3, 2.3)
        hits = r3.arc_hits([lob, back], self.court, [{"t": 1.42, "strength": 6.0}, {"t": 0.6, "strength": 3.5}])
        self.assertEqual([h["side"] for h in hits], ["near"])
        self.assertAlmostEqual(hits[0]["t"], 1.42, delta=0.01)
        self.assertTrue(hits[0]["plausible"])

    def test_merge_fills_gaps_and_keeps_existing_hits(self):
        crossing = [{"t": 1.0, "side": "far", "serve": False}, {"t": 5.0, "side": "far", "serve": False}]
        extra = [{"t": 1.2, "side": "far", "audio": 5.0, "window": [0.9, 1.1], "on_side": True, "plausible": True},
                 {"t": 3.0, "side": "near", "audio": 6.0, "window": [2.7, 2.9], "on_side": True, "plausible": True},
                 {"t": 4.0, "side": "near", "audio": 6.0, "window": [3.7, 3.9], "on_side": True, "plausible": False}]
        out = r3.merge(crossing, extra, 0.0, 10.0)
        self.assertEqual([(h["t"], h["side"]) for h in out], [(1.0, "far"), (3.0, "near"), (5.0, "far")])

    def test_merge_retimes_a_late_crossing_hit(self):
        crossing = [{"t": 2.0, "side": "far", "serve": False}]
        extra = [{"t": 1.2, "side": "far", "audio": 5.0, "window": [0.6, 1.1], "on_side": True, "plausible": False}]
        self.assertEqual(r3.merge(crossing, extra, 0.0, 10.0)[0]["t"], 1.2)


if __name__ == "__main__":
    unittest.main()
