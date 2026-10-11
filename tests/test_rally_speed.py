import math
import unittest

from analysis.hit_detect import BOUNCE_GAP, bounce_after, contact_time, mark_landings
from analysis.rally_speed import ball_at_hitter, shot_speed
from analysis.serve_speed import IN_H, IN_W, Court, launch_speed

CAL = [{"x": 0.058, "y": 0.838}, {"x": 0.936, "y": 0.841}, {"x": 0.605, "y": 0.420}, {"x": 0.419, "y": 0.430}]
W, H = 3840, 2160
DT = 1 / 60


def px(court, cx, cy):
    u, v = court.to_image(cx, cy)
    return u * IN_W, v * IN_H


def bounce_track(court, tb, cx=5.0, cy=20.0, n=12):
    """A ball coming down to (cx, cy) at tb and going back up: the image y peaks there."""
    x, y = px(court, cx, cy)
    return [(tb + k * DT, x + 2.0 * k, y - 1.2 * abs(k)) for k in range(-n, n + 1)]


class BounceAfterTests(unittest.TestCase):
    def setUp(self):
        self.court = Court(CAL)

    def test_finds_the_bounce_on_the_far_side(self):
        t, (x, y) = bounce_after([bounce_track(self.court, 11.0)], self.court, 10.2, 12.0, "near")
        self.assertAlmostEqual(t, 11.0, places=3)
        cx, cy = self.court.to_court(x / IN_W, y / IN_H)
        self.assertAlmostEqual(cx, 5.0, delta=0.05)
        self.assertAlmostEqual(cy, 20.0, delta=0.05)

    def test_not_on_the_hitters_own_side(self):
        self.assertIsNone(bounce_after([bounce_track(self.court, 11.0)], self.court, 10.2, 12.0, "far"))

    def test_a_second_ball_seen_at_once_is_no_bounce(self):
        # the tracker follows the ball down, then jumps to a white cap lower in the image while the
        # ball is still in the air: next to the ball's points that one looks like a bounce
        x, y = px(self.court, 5.0, 20.0)
        ball = [(11.0 + k * DT, x + 2.0 * k, y - 1.2 * abs(k)) for k in range(-24, -5)]
        jumped = ball + [(11.0 - 5 * DT, x + 150, y + 25)]
        t, _ = bounce_after([bounce_track(self.court, 11.0), jumped], self.court, 10.2, 12.0, "near")
        self.assertAlmostEqual(t, 11.0, places=3)


class MarkLandingsTests(unittest.TestCase):
    def setUp(self):
        self.court = Court(CAL)

    def test_landing_before_the_next_hit_only(self):
        tracks = [bounce_track(self.court, 11.0)]
        hits = [{"t": 10.12, "side": "near", "audio": 5.0}, {"t": 12.5, "side": "far", "audio": 5.0}]
        mark_landings(hits, tracks, self.court)
        self.assertAlmostEqual(hits[0]["landed"]["t"], 11.0, places=3)
        self.assertEqual(hits[0]["landed"]["court"], [5.0, 20.0])
        # the other side hit it right after: that low point was the racket, not a bounce
        hits = [{"t": 10.12, "side": "near", "audio": 5.0}, {"t": 11.0 + BOUNCE_GAP, "side": "far", "audio": 5.0}]
        mark_landings(hits, tracks, self.court)
        self.assertNotIn("landed", hits[0])


class ShotSpeedTests(unittest.TestCase):
    def setUp(self):
        self.court = Court(CAL)
        u, v = self.court.to_image(5.0, 0.0)
        fx, fy = u * W, v * H
        self.box = [fx - 60, fy - 400, fx + 60, fy]
        self.hit = {"t": 10.12, "side": "near", "audio": 6.0,
                    "balls": [[10.02, fx / W * IN_W + 8, (fy - 200) / H * IN_H]],
                    "landed": {"t": 11.0, "px": [0, 0], "court": [5.0, 20.0]}}

    def test_distance_over_flight_time(self):
        sp = shot_speed(self.hit, {"box": self.box}, "forehand", self.court, W, H)
        c = contact_time(self.hit)
        d = math.dist((5.0, 20.0, 0.0), (5.0, 0.0, 1.0))
        self.assertAlmostEqual(sp["speed_kmh"], launch_speed(d, 11.0 - c) * 3.6, delta=1.5)
        self.assertAlmostEqual(sp["distance_m"], d, delta=0.1)

    def test_needs_the_sound_the_box_and_the_ball_at_the_hitter(self):
        self.assertIsNone(shot_speed(dict(self.hit, audio=None), {"box": self.box}, "forehand", self.court, W, H))
        self.assertIsNone(shot_speed(self.hit, None, "forehand", self.court, W, H))
        far_ball = dict(self.hit, balls=[[10.02, 50.0, 50.0]])
        self.assertIsNone(shot_speed(far_ball, {"box": self.box}, "forehand", self.court, W, H))

    def test_ball_at_hitter_only_around_contact(self):
        balls = self.hit["balls"]
        self.assertTrue(ball_at_hitter(balls, 10.0, self.box, W, H))
        self.assertFalse(ball_at_hitter(balls, 10.5, self.box, W, H))


if __name__ == "__main__":
    unittest.main()
