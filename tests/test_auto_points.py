import unittest

from analysis.auto_points import segment


def serve(t, end="far", call="in", h=2.4, x=5.0):
    return {"contact_t": t, "server_end": end, "call": call, "contact_height_m": h, "server_court": [x, 23.77]}


def hits(*ts, first="near"):
    other = {"near": "far", "far": "near"}
    out, side = [], first
    for t in ts:
        out.append({"t": t, "side": side})
        side = other[side]
    return out


class SegmentTests(unittest.TestCase):
    def test_rally_after_a_serve_is_one_point(self):
        pts = segment([serve(10.0)], hits(10.0, 11.6, 13.0, 14.5, first="far"), 60.0)
        self.assertEqual(len(pts), 1)
        self.assertAlmostEqual(pts[0]["start"], 9.0)
        self.assertAlmostEqual(pts[0]["end"], 16.0)

    def test_quiet_spell_splits_points(self):
        pts = segment([serve(10.0), serve(30.0)], hits(11.6, 13.0) + hits(31.5, 33.0), 60.0)
        self.assertEqual([p["serves"] for p in pts], [[10.0], [30.0]])

    def test_serve_like_shot_inside_a_rally_is_not_a_serve(self):
        # a far player's deep shot two seconds after the return
        pts = segment([serve(10.0), serve(13.2, end="far")], hits(11.6, 15.0), 60.0)
        self.assertEqual(len(pts), 1)
        self.assertEqual(pts[0]["serves"], [10.0])

    def test_fault_then_second_serve_same_point(self):
        pts = segment([serve(10.0, end="near", call="out"), serve(15.0, end="near")], hits(16.6, 18.0, first="far"), 60.0)
        self.assertEqual(len(pts), 1)
        self.assertEqual(pts[0]["serves"], [10.0, 15.0])

    def test_lone_fault_is_not_a_point(self):
        # a ball fed back to the server looked like a serve going out, and nothing followed
        self.assertEqual(segment([serve(10.0, call="out")], [], 60.0), [])

    def test_implausible_contact_height_is_not_a_serve(self):
        self.assertEqual(segment([serve(10.0, h=1.6)], [], 60.0), [])

    def test_missed_serve_rally_still_makes_a_point(self):
        pts = segment([], hits(20.0, 21.5, 23.0), 60.0)
        self.assertEqual(len(pts), 1)
        self.assertEqual(pts[0]["serves"], [])

    def test_two_balls_fed_back_are_not_a_point(self):
        self.assertEqual(segment([], hits(20.0, 21.5), 60.0), [])

    def test_points_never_overlap(self):
        pts = segment([serve(10.0), serve(17.5)], hits(11.6, 13.0, 14.5, 16.0) + hits(19.0, 20.5), 60.0)
        for a, b in zip(pts, pts[1:]):
            self.assertLessEqual(a["end"], b["start"])


if __name__ == "__main__":
    unittest.main()
