import unittest

import numpy as np

from analysis import hit_detect as hd
from analysis.flight3d import Camera
from analysis.serve_speed import Court

# calibration of the sample match: near-left, near-right, far-right, far-left (normalized)
CAL = [{"x": 0.058, "y": 0.838}, {"x": 0.936, "y": 0.841}, {"x": 0.605, "y": 0.420}, {"x": 0.419, "y": 0.430}]
FPS = 60.0


def arc(p0, p1, t0, t1, apex):
    """Ball positions (t, x, y, z) on a parabola from p0 to p1 between t0 and t1."""
    out = []
    for t in np.arange(t0, t1, 1 / FPS):
        s = (t - t0) / (t1 - t0)
        x = p0[0] + (p1[0] - p0[0]) * s
        y = p0[1] + (p1[1] - p0[1]) * s
        z = p0[2] + (p1[2] - p0[2]) * s + 4 * apex * s * (1 - s)
        out.append((t, x, y, z))
    return out


class HitDetectTests(unittest.TestCase):
    def setUp(self):
        self.court = Court(CAL)
        self.cam = Camera(self.court)

    def track(self, pts):
        uv = self.cam.project(np.array([[x, y, z] for _, x, y, z in pts]))
        return [(t, float(u), float(v)) for (t, *_), (u, v) in zip(pts, uv)]

    def test_rally_sides_and_times(self):
        # near player hits at 0.0 (to the far baseline), far player at 1.2 (back to the near baseline)
        out = arc((5, 0.5, 1.0), (5, 21, 1.0), 0.0, 1.2, 1.8)
        back = arc((5, 21, 1.0), (5, 1.5, 1.0), 1.2, 2.4, 1.6)
        tracks = [self.track(out[2:]), self.track(back[3:])]  # the ball is lost for a few frames at each racket
        onsets = [{"t": 0.0, "strength": 8.0}, {"t": 1.35, "strength": 5.0}, {"t": 0.6, "strength": 4.0}]
        hits = hd.detect_hits(tracks, self.court, onsets)
        self.assertEqual([h["side"] for h in hits], ["near", "far"])
        self.assertAlmostEqual(hits[0]["t"], 0.0, delta=0.01)
        self.assertAlmostEqual(hits[1]["t"], 1.35, delta=0.01)  # the far hit's sound, not the bounce at 0.6

    def test_far_service_box_bounce_is_not_a_hit(self):
        # near serve lands short in the far box (below the tape in the image), bounces up,
        # and the far player returns it from the baseline
        serve = arc((7, 0, 2.6), (3, 15.5, 0.0), 0.0, 0.6, 0.3)
        up = arc((3, 15.5, 0.0), (2.5, 22, 1.0), 0.6, 1.2, 0.8)
        ret = arc((2.5, 22, 1.0), (6, 2, 1.0), 1.25, 2.4, 1.5)
        tracks = [self.track(serve[2:] + up), self.track(ret)]
        hits = hd.detect_hits(tracks, self.court, [{"t": 1.37, "strength": 5.0}],
                              serves=[{"t": 0.0, "side": "near", "fault": False}])
        self.assertEqual([(h["side"], h["serve"]) for h in hits], [("near", True), ("far", False)])
        self.assertAlmostEqual(hits[1]["t"], 1.37, delta=0.01)

    def test_nothing_counts_after_a_fault_until_the_next_serve(self):
        fault = arc((7, 0, 2.6), (4, 9, 0.9), 0.0, 0.4, 0.1)        # into the net
        tap = arc((4, 20, 1.0), (6, 3, 1.0), 1.5, 2.5, 1.5)          # receiver taps the ball back
        second = arc((7, 0, 2.6), (3, 17, 0.0), 4.0, 4.7, 0.4)
        tracks = [self.track(fault), self.track(tap), self.track(second)]
        hits = hd.detect_hits(tracks, self.court, [], serves=[{"t": 0.0, "side": "near", "fault": True},
                                                              {"t": 4.0, "side": "near", "fault": False}])
        self.assertEqual([h["t"] for h in hits], [0.0, 4.0])

    def test_slow_tracks_are_ignored(self):
        rolling = [(t, 300 + 20 * t, 250.0) for t in np.arange(0, 1.0, 1 / FPS)]
        self.assertEqual(hd.detect_hits([rolling], self.court, []), [])

    def test_bounce_before_a_ground_stroke_not_a_volley(self):
        # far player hits at 0.0; the ball bounces at the near baseline area at 1.0, near player hits at 1.5
        down = arc((5, 22, 1.0), (5, 4, 0.0), 0.0, 1.0, 1.5)
        up = arc((5, 4, 0.0), (5, 0.5, 1.0), 1.0, 1.5, 0.6)
        stroke = [{"t": 0.0, "side": "far"}, {"t": 1.5, "side": "near"}]
        hd.mark_bounces(stroke, [self.track(down + up)], self.court)
        self.assertTrue(stroke[1]["bounced"])
        self.assertAlmostEqual(stroke[1]["bounce_t"], 1.0, delta=0.05)
        # the same ball taken out of the air at the service line
        volley = [{"t": 0.0, "side": "far"}, {"t": 0.75, "side": "near"}]
        hd.mark_bounces(volley, [self.track(arc((5, 22, 1.0), (5, 6.4, 1.2), 0.0, 0.75, 1.5))], self.court)
        self.assertFalse(volley[1]["bounced"])

    def test_balls_on_the_next_court_are_not_ours(self):
        ours = self.track(arc((5, 22, 1.0), (5, 2, 1.0), 0.0, 1.0, 1.5))
        nxt = self.track(arc((5 + 14, 22, 1.0), (5 + 14, 2, 1.0), 0.0, 1.0, 1.5))  # next court to the right
        self.assertTrue(hd.own_court(ours, self.court))
        self.assertFalse(hd.own_court(nxt, self.court))

    def test_a_serve_return_has_bounced(self):
        hits = [{"t": 0.0, "side": "near", "serve": True}, {"t": 1.2, "side": "far"}]
        hd.mark_bounces(hits, [], self.court)
        self.assertTrue(hits[1]["bounced"])


if __name__ == "__main__":
    unittest.main()
