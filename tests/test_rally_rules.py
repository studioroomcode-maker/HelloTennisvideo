import unittest

from analysis.rally_rules import apply_rally_rules
from analysis.shot_analysis import fix_serve_numbers

SETTINGS = {"mode": "doubles", "team_a": ["a1", "a2"], "team_b": ["b1", "b2"]}


def ev(t, shot, side, conf=80, result="in", pid="p1", a_end="near", server="A1", winner=None):
    team = None
    if shot == "serve" and server:
        team = server[0]
    elif side in ("near", "far") and a_end:
        team = "A" if side == a_end else "B"
    return {
        "t": t, "shot": shot, "side": side, "confidence": conf, "result": result,
        "point_id": pid, "point_no": 1, "spin": "unknown", "lane": "unknown",
        "serve_number": 1 if shot == "serve" else 0,
        "serve_type": "flat" if shot == "serve" else None,
        "serve_direction": "T" if shot == "serve" else None,
        "speed_kmh": None, "team": team, "player": server if shot == "serve" else None,
        "fixes": [], "_side_raw": side,
        "_ctx": {"a_end": a_end, "server": server if pid else None, "winner": winner},
    }


def run(events):
    warnings = []
    out, fixes = apply_rally_rules(events, SETTINGS, fix_serve_numbers, warnings)
    return out, fixes


class RallyRuleTests(unittest.TestCase):
    def test_impossible_same_side_sequence_is_decoded(self):
        out, fixes = run([
            ev(1.0, "serve", "near"),
            ev(3.0, "backhand", "near", conf=40),   # return must come from the far side
            ev(4.0, "forehand", "near", conf=85),
            ev(5.0, "forehand", "far", conf=80),
        ])
        self.assertEqual([e["side"] for e in out[1:]], ["far", "near", "far"])
        self.assertEqual(out[1]["team"], "B")
        self.assertEqual(fixes["side_flip"], 1)
        self.assertNotIn("_ctx", out[0])

    def test_missed_shot_allowed_when_time_permits(self):
        out, fixes = run([
            ev(1.0, "serve", "near"),
            ev(3.0, "forehand", "far"),
            ev(4.0, "forehand", "near"),
            ev(6.5, "backhand", "near", conf=90),  # 2.5s later: opponent shot was missed
            ev(7.5, "forehand", "far"),
        ])
        self.assertNotIn("side_flip", fixes)
        self.assertEqual([e["side"] for e in out[1:]], ["far", "near", "near", "far"])

    def test_duplicate_swing_removed(self):
        out, fixes = run([
            ev(1.0, "serve", "near"),
            ev(3.0, "forehand", "far", conf=60),
            ev(3.3, "backhand", "far", conf=90),
        ])
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1]["shot"], "backhand")
        self.assertEqual(fixes["duplicate"], 1)

    def test_serve_mid_rally_becomes_smash(self):
        out, fixes = run([
            ev(1.0, "serve", "near"),
            ev(3.0, "forehand", "far"),
            ev(4.2, "serve", "near"),
        ])
        self.assertEqual(out[2]["shot"], "smash")
        self.assertEqual(out[2]["team"], "A")
        self.assertIsNone(out[2]["player"])
        self.assertEqual(fixes["serve_to_smash"], 1)

    def test_fault_ball_tap_removed_and_serves_numbered(self):
        out, fixes = run([
            ev(1.0, "serve", "near", result="in"),
            ev(2.0, "forehand", "far"),             # receiver taps the fault ball back
            ev(14.0, "serve", "near", result="fault"),
            ev(15.5, "backhand", "far"),
        ])
        self.assertEqual([e["shot"] for e in out], ["serve", "serve", "backhand"])
        self.assertEqual([e["serve_number"] for e in out[:2]], [1, 2])
        self.assertEqual(out[0]["result"], "fault")
        self.assertEqual(out[1]["result"], "in")  # it was returned
        self.assertEqual(fixes["fault_return"], 1)

    def test_results_only_end_the_rally(self):
        out, fixes = run([
            ev(1.0, "serve", "near", result="ace"),
            ev(3.0, "volley", "far", result="winner"),
            ev(4.0, "forehand", "near", result="winner"),
        ])
        self.assertEqual(out[0]["result"], "in")
        self.assertEqual(out[1]["shot"], "other")       # a return can't be a volley
        self.assertEqual(out[1]["result"], "in")
        self.assertEqual(out[2]["result"], "winner")
        self.assertEqual(fixes["return_volley"], 1)

    def test_result_contradicting_labeled_winner_is_cleared(self):
        out, fixes = run([
            ev(1.0, "serve", "near", winner="B"),
            ev(3.0, "forehand", "far", winner="B"),
            ev(4.0, "forehand", "near", result="winner", winner="B"),  # Team A "winner" but B won
        ])
        self.assertEqual(out[2]["result"], "unknown")
        self.assertEqual(fixes["result_vs_label"], 1)

    def test_full_video_mode_splits_rallies_at_serves(self):
        evs = [ev(t, s, side, pid=None, a_end=None, server=None) for t, s, side in [
            (0.0, "serve", "far"), (1.5, "forehand", "far", ), (2.6, "forehand", "far"),
            (20.0, "serve", "near"), (21.5, "backhand", "far"),
        ]]
        out, fixes = run(evs)
        self.assertEqual([e["side"] for e in out], ["far", "near", "far", "near", "far"])
        self.assertEqual(fixes["side_flip"], 1)


if __name__ == "__main__":
    unittest.main()
