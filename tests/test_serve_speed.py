import math
import unittest

from analysis import serve_speed as ss
from analysis.serve_measure_job import apply_measurements, server_end_warnings

# calibration of the sample match: near-left, near-right, far-right, far-left (normalized)
CAL = [{"x": 0.058, "y": 0.838}, {"x": 0.936, "y": 0.841}, {"x": 0.605, "y": 0.420}, {"x": 0.419, "y": 0.430}]


class PhysicsTests(unittest.TestCase):
    def test_launch_speed_matches_drag_model(self):
        v0 = ss.launch_speed(20.0, 0.55)
        # integrate dv/dt = -k v^2 and check the distance flown
        v, x, dt = v0, 0.0, 1e-4
        for _ in range(int(0.55 / dt)):
            x += v * dt
            v -= ss.K_DRAG * v * v * dt
        self.assertAlmostEqual(x, 20.0, delta=0.05)
        self.assertGreater(v0, 20.0 / 0.55)  # launch speed is above the average speed

    def test_service_boxes_are_diagonal(self):
        # near server on the right (deuce) serves into the far-left box
        self.assertEqual(ss.serve_box("near", 8.0), ((ss.SINGLES_L, ss.CENTER_X), (ss.NET_Y, ss.SVC_FAR)))
        # far server on their right (court x small) serves into the near box on court-right
        self.assertEqual(ss.serve_box("far", 3.0), ((ss.CENTER_X, ss.SINGLES_R), (ss.SVC_NEAR, ss.NET_Y)))

    def test_calls_and_placement(self):
        box = ((ss.CENTER_X, ss.SINGLES_R), (ss.SVC_NEAR, ss.NET_Y))
        self.assertEqual(ss.judge_in(7.0, 8.0, *box), "in")
        self.assertEqual(ss.judge_in(7.0, 4.0, *box), "out")
        self.assertEqual(ss.judge_in(7.0, ss.SVC_NEAR + 0.1, *box), "close")
        self.assertEqual(ss.placement(5.9, box[0]), "T")
        self.assertEqual(ss.placement(7.5, box[0]), "body")
        self.assertEqual(ss.placement(9.3, box[0]), "wide")


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.court = ss.Court(CAL)

    def test_static_blobs_removed_and_moving_ball_linked(self):
        dets, times = [], []
        for i in range(60):
            t = i / 60
            dets.append([(100.0, 50.0, 2000.0), (300 + 4 * i, 200 - 1.5 * i, 2500.0)])  # light + ball
            times.append(t)
        dets = ss.remove_static(dets, 60)
        self.assertTrue(all(len(d) == 1 for d in dets))
        tracks = ss.link_tracks(dets, times)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(len(tracks[0]), 60)

    def test_bounce_kink_found_in_continuous_and_gapped_track(self):
        # ball coming toward the camera in the near court: y grows, then flips up at the bounce
        def pt(t):
            tb = 0.5
            y = 170 + 120 * min(t, tb) - 40 * max(0.0, t - tb)
            return (t, 330 + 40 * t, y)
        track = [pt(i / 60) for i in range(60)]
        b = ss.find_bounces(track, self.court)
        self.assertEqual(len(b), 1)
        self.assertAlmostEqual(b[0]["t"], 0.5, delta=0.02)
        gapped = [p for p in track if not 0.45 < p[0] < 0.6]
        b = ss.find_bounces(gapped, self.court)
        self.assertEqual(len(b), 1)
        self.assertAlmostEqual(b[0]["t"], 0.5, delta=0.03)

    def test_bridge_joins_track_split_by_lost_frames(self):
        a = [(i / 60, 400 - 4 * i, 160.0) for i in range(30)]
        b = [(i / 60, 400 - 4 * i, 160.0) for i in range(38, 60)]
        joined = ss.bridge_tracks([a, b])
        self.assertEqual(len(joined), 1)
        self.assertEqual(len(joined[0]), 52)


class MergeTests(unittest.TestCase):
    def test_measured_speed_replaces_ai_estimate(self):
        result = {"file_id": "f1", "events": [
            {"t": 10.0, "file_id": "f1", "shot": "serve", "side": "far", "spin": "unknown", "serve_number": 1,
             "serve_type": "flat", "serve_direction": "unknown", "result": "in", "speed_kmh": 150,
             "confidence": 80, "team": "A", "player": "A1", "point_id": "p1", "point_no": 1, "fixes": []},
        ]}
        measured = {"file_id": "f1", "serves": [
            {"contact_t": 10.3, "bounce_t": 10.95, "speed_kmh": 112, "speed_err_kmh": 5, "bounce_court": [6.9, 7.4],
             "call": "in", "direction": "body", "server_end": "far", "flight_s": 0.65, "point_id": "p1", "point_no": 1},
            {"contact_t": 30.0, "bounce_t": 30.8, "speed_kmh": 95, "speed_err_kmh": 4, "bounce_court": [4.0, 19.0],
             "call": "out", "direction": "body", "server_end": "near", "flight_s": 0.8, "point_id": "p2", "point_no": 2},
        ], "warnings": []}
        apply_measurements(result, measured, {})
        ev = result["events"]
        self.assertEqual(ev[0]["speed_kmh"], 112)
        self.assertEqual(ev[0]["speed_est"], 150)
        self.assertEqual(ev[0]["speed_source"], "measured")
        self.assertEqual(ev[0]["serve_direction"], "body")
        self.assertEqual(ev[1]["result"], "fault")  # added from tracking, called out
        self.assertEqual(result["summary"]["serve"]["count"], 2)
        self.assertEqual(result["measured_serves"]["max"], 112)

    def test_server_end_mismatch_is_reported(self):
        project = {"points": [{"id": "p1", "score_before": {"server_id": "A1", "team_a_end": "near"}}]}
        w = server_end_warnings(project, [{"point_id": "p1", "point_no": 1, "server_end": "far"}])
        self.assertTrue(w and "#1" in w[0])
        self.assertEqual(server_end_warnings(project, [{"point_id": "p1", "point_no": 1, "server_end": "near"}]), [])


if __name__ == "__main__":
    unittest.main()
