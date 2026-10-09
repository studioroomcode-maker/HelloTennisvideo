import unittest

from analysis.hit_shots import clip_line, hand_note

SETTINGS = {"mode": "doubles", "team_a": ["Kim", "Lee"], "team_b": ["Park", "Choi"]}


class HandNoteTests(unittest.TestCase):
    def test_players_default_to_right_handed(self):
        self.assertEqual(hand_note(SETTINGS, {"team_a_end": "far"}, "far"), "이 코트 선수는 모두 오른손잡이입니다.")

    def test_lefty_named_on_the_side_that_holds_their_team(self):
        s = dict(SETTINGS, left_handed=["Park"])
        self.assertEqual(hand_note(s, {"team_a_end": "far"}, "near"), "이 코트 선수: Park 왼손잡이, Choi 오른손잡이.")
        self.assertEqual(hand_note(s, {"team_a_end": "near"}, "near"), "이 코트 선수는 모두 오른손잡이입니다.")

    def test_unknown_ends_give_no_note(self):
        self.assertEqual(hand_note(SETTINGS, {}, "near"), "")
        self.assertNotIn("오른손", clip_line(1, "near"))


if __name__ == "__main__":
    unittest.main()
