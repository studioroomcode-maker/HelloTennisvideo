import unittest

from analysis.serve_motion import learn_types

POINTS = [{"id": "p1", "score_before": {"server_id": "B1"}}, {"id": "p2", "score_before": {"server_id": "A1"}}]


def serve(t, split, point="p1"):
    return {"contact_t": t, "leg_split": split, "point_id": point}


class ServeMotionTests(unittest.TestCase):
    def test_learnt_per_server_without_its_own_label(self):
        serves = [serve(10, 0.22), serve(20, 0.24), serve(30, 0.10), serve(40, 0.09), serve(50, 0.23), serve(60, 0.30, "p2")]
        labels = [{"shot": "serve", "t": t, "spin": k} for t, k in
                  ((10, "flat"), (20, "flat"), (30, "slice"), (40, "slice"), (60, "slice"))]
        learn_types(serves, labels, POINTS)
        self.assertEqual([s.get("serve_type_motion") for s in serves], ["flat", "flat", "slice", "slice", "flat", None])

    def test_overlapping_classes_are_not_used(self):
        serves = [serve(10, 0.20), serve(20, 0.12), serve(30, 0.18)]
        labels = [{"shot": "serve", "t": 10, "spin": "flat"}, {"shot": "serve", "t": 20, "spin": "slice"}]
        learn_types(serves, labels, POINTS)
        self.assertEqual(serves[2]["serve_type_motion"], "flat")
        self.assertIsNone(serves[0].get("serve_type_motion"))  # its own label left out: one class remains


if __name__ == "__main__":
    unittest.main()
