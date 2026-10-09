import unittest

from analysis import swing_side as ss


def person(xs, side_x=0.0, bh=100.0, top=0.0):
    """Rows of one person whose wrists sit at torso + x * body height in each frame (30 fps)."""
    rows = {}
    for i, x in enumerate(xs):
        kp = [[0.0, 0.0, 0.0] for _ in range(17)]
        for k in (5, 6, 11, 12):
            kp[k] = [side_x, top + 50.0, 0.9]
        kp[9] = kp[10] = [side_x + x * bh, top + 50.0, 0.9]
        rows[i] = (kp, [side_x - 20, top, side_x + 20, top + bh])
    return rows


def sweep(a, b, n=23, start=10, length=6):
    """Hands at `a` until frame `start`, moving to `b` over `length` frames (contact is frame 13.5)."""
    return [a if i < start else b if i >= start + length else a + (b - a) * (i - start) / length for i in range(n)]


class SwingSideTests(unittest.TestCase):
    def test_right_to_left_sweep_is_a_forehand_for_a_right_hander(self):
        swing = ss.forward_swing(ss.lateral(person(sweep(0.3, -0.2)), "near", ss.FPS))
        self.assertLess(swing, -0.3)
        self.assertEqual(ss.stroke(swing, "right"), "forehand")
        self.assertEqual(ss.stroke(swing, "left"), "backhand")

    def test_far_players_face_the_camera(self):
        # image left -> right for a far player is his right -> left: a forehand
        swing = ss.forward_swing(ss.lateral(person(sweep(-0.3, 0.2)), "far", ss.FPS))
        self.assertEqual(ss.stroke(swing, "right"), "forehand")

    def test_small_swing_or_mixed_hands_names_nothing(self):
        self.assertIsNone(ss.stroke(0.05, "right"))
        self.assertIsNone(ss.stroke(0.4, None))

    def test_hitter_is_the_player_the_ball_reaches(self):
        hitter = {"rows": person(sweep(-0.3, 0.3), side_x=500.0)}
        partner = {"rows": person(sweep(0.4, -0.4), side_x=100.0)}  # bigger move, but far from the ball
        tc = 1.0
        balls = [[tc, 500 * 640 / 3840, 50 * 360 / 2160]]
        res = ss.judge_people([partner, hitter], balls, tc, "near", (0, 0, 3840, 2160), 3840, 2160, ss.FPS)
        self.assertEqual(res["by"], "ball")
        self.assertGreater(res["swing"], 0.3)
        # without the ball the biggest swing is taken
        res = ss.judge_people([partner, hitter], [], tc, "near", (0, 0, 3840, 2160), 3840, 2160, ss.FPS)
        self.assertLess(res["swing"], -0.5)


if __name__ == "__main__":
    unittest.main()
