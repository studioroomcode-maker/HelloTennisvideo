import unittest
from unittest import mock

from analysis import point_winner
from analysis.hit_shots import rally_results
from analysis.point_winner import call_shot, decide, evidence, fill_winners, judge_flight

SETTINGS = {"server_rotation": ["A1", "B1"], "team_a_start_end": "near", "change_ends": False, "no_ad": True}


def flight(x, y, net_z=1.5, t=10.0):
    return {"x": x, "y": y, "t": t, "net_z": net_z}


def serve(t, end="near", call="in"):
    return {"contact_t": t, "server_end": end, "call": call}


class JudgeFlightTests(unittest.TestCase):
    def test_in(self):
        self.assertEqual(judge_flight("near", flight(5.0, 20.0)), ("won", "in"))

    def test_long_and_wide_are_out(self):
        self.assertEqual(judge_flight("near", flight(5.0, 24.5)), ("lost", "out"))
        self.assertEqual(judge_flight("far", flight(11.6, 6.0)), ("lost", "out"))

    def test_on_the_line_is_in(self):
        self.assertEqual(judge_flight("near", flight(10.99, 23.79))[0], "won")

    def test_short_of_the_net_or_into_the_tape(self):
        self.assertEqual(judge_flight("near", flight(5.0, 10.0, net_z=None)), ("lost", "net"))
        self.assertEqual(judge_flight("far", flight(5.0, 6.0, net_z=0.4)), ("lost", "net"))

    def test_crossed_before_the_fitted_stretch(self):
        self.assertEqual(judge_flight("near", flight(4.0, 20.0, net_z=None)), ("won", "in"))


class CallShotTests(unittest.TestCase):
    def call(self, f, nxt=None, singles=False):
        with mock.patch.object(point_winner, "shot_flight", return_value=f):
            return call_shot({"t": 9.0, "side": "near"}, nxt, [], singles)

    def test_in_out_close(self):
        self.assertEqual(self.call(flight(5.0, 20.0))["call"], "in")
        self.assertEqual(self.call(flight(5.0, 24.5))["call"], "out")
        self.assertEqual(self.call(flight(10.85, 20.0))["call"], "close")

    def test_singles_lines(self):
        self.assertEqual(self.call(flight(10.0, 20.0))["call"], "in")
        self.assertEqual(self.call(flight(10.0, 20.0), singles=True)["call"], "out")

    def test_taken_in_the_air_is_not_called(self):
        self.assertEqual(self.call(flight(5.0, 25.0, t=10.0), {"t": 10.05, "side": "far"}), {"call": None, "volleyed": True})

    def test_net_ball_played_back_was_a_bad_fit(self):
        self.assertIsNone(self.call(flight(5.0, 10.0, net_z=None), {"t": 11.0, "side": "far"}))
        self.assertEqual(self.call(flight(5.0, 10.0, net_z=None))["call"], "net")


class RallyResultTests(unittest.TestCase):
    def ev(self, t, call, team="A"):
        return {"t": t, "point_id": "p", "shot": "forehand", "team": team, "result": "unknown",
                "landing": {"call": call, "x": 5.0, "y": 20.0}}

    def test_results(self):
        evs = [self.ev(1, "in"), self.ev(2, "out", "B"), self.ev(3, "in")]
        rally_results(evs)
        self.assertEqual([e["result"] for e in evs], ["in", "error", "winner"])

    def test_no_winner_for_the_team_that_lost_the_point(self):
        evs = [self.ev(1, "in", "A")]
        rally_results(evs, {"p": "B"})
        self.assertEqual(evs[0]["result"], "in")


class EvidenceTests(unittest.TestCase):
    def test_double_fault_goes_to_the_receiver(self):
        e = evidence([serve(10.0, call="net"), serve(15.0, call="out")], [], [])
        self.assertEqual(e["how"], "double_fault")
        self.assertLess(e["p_near"], 0.5)

    def test_serve_with_no_return(self):
        e = evidence([serve(10.0, end="far")], [], [])
        self.assertEqual(e["how"], "ace")
        self.assertLess(e["p_near"], 0.5)

    def test_last_ball_unseen_counts_as_the_hitters_miss(self):
        hits = [{"t": 11.5, "side": "far"}, {"t": 13.0, "side": "near"}]
        e = evidence([serve(10.0)], hits, [])
        self.assertEqual(e["how"], "error_unseen")
        self.assertLess(e["p_near"], 0.5)


class DecideTests(unittest.TestCase):
    def test_server_change_closes_the_game(self):
        # four points lean A, one leans B a little; the 5th is served from the far end, which is B's
        # serve, so the game was over after four points: A won all four
        pts = [{"id": str(i), "winner": None} for i in range(5)]
        seen = {"0": {"p_near": 0.7}, "1": {"p_near": 0.45}, "2": {"p_near": 0.7}, "3": {"p_near": 0.7},
                "4": {"p_near": 0.5, "server_end": "far"}}
        self.assertEqual([decide(pts, SETTINGS, seen)[str(i)] for i in range(4)], ["A"] * 4)
        del seen["4"]["server_end"]
        self.assertEqual(decide(pts, SETTINGS, seen)["1"], "B")

    def test_users_winner_is_kept(self):
        pts = [{"id": "0", "winner": "B"}]
        self.assertEqual(decide(pts, SETTINGS, {"0": {"p_near": 0.99}})["0"], "B")


class FillTests(unittest.TestCase):
    def test_fills_blank_keeps_users_and_cleared(self):
        ev = {"p_near": 0.8, "how": "winner"}
        project = {"settings": SETTINGS, "files": [{"id": "f"}], "points": [
            {"id": "a", "file_id": "f", "start": 1, "winner": None, "winner_evidence": ev},
            {"id": "b", "file_id": "f", "start": 2, "winner": "B", "winner_evidence": ev},
            {"id": "c", "file_id": "f", "start": 3, "winner": None, "winner_cleared": True, "winner_evidence": ev},
            {"id": "d", "file_id": "f", "start": 4, "winner": None},
        ]}
        fill_winners(project)
        a, b, c, d = project["points"]
        self.assertEqual((a["winner"], a["winner_auto"]), ("A", True))
        self.assertEqual(b["winner"], "B")
        self.assertFalse(b.get("winner_auto"))
        self.assertIsNone(c["winner"])
        self.assertIsNone(d["winner"])


if __name__ == "__main__":
    unittest.main()
