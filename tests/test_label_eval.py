import unittest

from analysis.label_eval import evaluate

POINTS = [{"id": "p1", "start": 10.0, "end": 20.0}, {"id": "p2", "start": 30.0, "end": 40.0}]


class LabelEvalTests(unittest.TestCase):
    def test_scores_only_labelled_points(self):
        labels = [{"t": 10.5, "shot": "serve"}, {"t": 12.0, "shot": "forehand"}, {"t": 14.0, "shot": "backhand"},
                  {"t": 16.0, "shot": "forehand"}]
        events = [{"t": 10.3, "shot": "serve"}, {"t": 12.4, "shot": "forehand"},
                  {"t": 14.2, "shot": "forehand"},          # backhand read as forehand
                  {"t": 18.0, "shot": "volley"},            # nothing there
                  {"t": 33.0, "shot": "forehand"}]          # unlabelled point: ignored
        r = evaluate(labels, events, POINTS)
        self.assertEqual(r["points_scored"], 1)
        self.assertEqual(r["ai_events"], 4)
        self.assertEqual(r["correct"], 2)
        self.assertEqual(r["recall"], 50)
        self.assertEqual(r["confusion"], {"backhand->forehand": 1})
        self.assertEqual([m["t"] for m in r["missed"]], [16.0])
        self.assertEqual(r["per_type"]["forehand"]["labelled"], 2)
        self.assertEqual(r["per_type"]["forehand"]["correct"], 1)

    def test_stroke_tolerance_and_serve_spin(self):
        labels = [{"t": 10.5, "shot": "serve", "spin": "slice", "side": "far", "tol": 0.5},
                  {"t": 12.0, "shot": "stroke", "side": "near", "tol": 1.2},
                  {"t": 13.0, "shot": "volley", "tol": 1.2}]
        events = [{"t": 10.2, "shot": "serve", "serve_type": "flat", "side": "far"},
                  {"t": 13.0, "shot": "backhand", "side": "near"},   # stroke label, 1.0 s off, any side of body
                  {"t": 13.4, "shot": "forehand"}]                   # extra
        r = evaluate(labels, events, POINTS)
        self.assertEqual(r["correct"], 2)
        self.assertEqual(r["serve_spin"], {"checked": 1, "correct": 0, "pairs": ["slice->flat"]})
        self.assertEqual(r["side"], {"checked": 2, "correct": 2})
        self.assertEqual(r["missed"], [])  # matched in time, wrong type -> confusion, not a miss
        self.assertEqual(r["confusion"], {"volley->forehand": 1})

    def test_high_volley_and_smash_score_alike(self):
        labels = [{"t": 12.0, "shot": "volley"}, {"t": 15.0, "shot": "smash"}]
        events = [{"t": 12.1, "shot": "smash"}, {"t": 15.1, "shot": "volley"}]
        self.assertEqual(evaluate(labels, events, POINTS)["correct"], 2)


if __name__ == "__main__":
    unittest.main()
