import unittest

from analysis import dashboard_video as dv

SETTINGS = {"mode": "doubles", "team_a": ["선수A1", "선수A2"], "team_b": ["선수B1", "선수B2"]}


class DashboardTimelineTests(unittest.TestCase):
    def test_points_become_padded_segments_and_events_map_to_output_time(self):
        f = {"id": "f1", "metadata": {"duration": 100.0}}
        project = {"points": [
            {"id": "p1", "file_id": "f1", "start": 10.0, "end": 15.0},
            {"id": "p2", "file_id": "f1", "start": 40.0, "end": 44.0},
        ]}
        segs = dv.build_segments(project, f, {"events": []})
        self.assertEqual([(s["start"], s["end"]) for s in segs], [(8.5, 16.0), (38.5, 45.0)])
        chunk = dv._make_chunk(segs)
        evs = dv.map_events([{"t": 41.0, "shot": "forehand"}, {"t": 11.0, "shot": "serve"},
                             {"t": 30.0, "shot": "lob"}], chunk)
        self.assertEqual([e["shot"] for e in evs], ["serve", "forehand"])  # 30s is outside any point
        self.assertAlmostEqual(evs[0]["out_t"], 2.5)
        self.assertAlmostEqual(evs[1]["out_t"], 7.5 + 2.5)
        self.assertEqual([e["no"] for e in evs], [1, 2])

    def test_unlabeled_events_grouped_into_bursts(self):
        f = {"id": "f1", "metadata": {"duration": 100.0}}
        segs = dv.build_segments({"points": []}, f, {"events": [{"t": 20.0}, {"t": 22.0}, {"t": 60.0}]})
        self.assertEqual([(s["start"], s["end"]) for s in segs], [(17.5, 24.0), (57.5, 62.0)])

    def test_court_side_from_score(self):
        self.assertEqual(dv.court_side({"score_before": {"point_a": "15", "point_b": "0"}}), "ad")
        self.assertEqual(dv.court_side({"score_before": {"point_a": "40", "point_b": "40"}}), "deuce")
        self.assertEqual(dv.court_side({"score_before": {"point_a": "AD", "point_b": "40"}}), "ad")
        self.assertEqual(dv.court_side({"score_before": {"tiebreak": True, "tiebreak_a": 2, "tiebreak_b": 1}}), "ad")


class DashboardRenderTests(unittest.TestCase):
    def setUp(self):
        try:
            dv.ensure_pillow()
        except Exception:
            self.skipTest("Pillow unavailable")

    def test_overlay_card_and_scoreboard(self):
        fonts = dv.Fonts()
        e = {"no": 1, "shot": "serve", "serve_number": 1, "serve_type": "flat", "serve_direction": "T",
             "speed_kmh": 150, "point_id": "p1", "result": "in", "player": "A1", "team": "A"}
        point = {"id": "p1", "score_before": {"point_a": "15", "point_b": "0", "server_id": "A1"}}
        img = dv.render_overlay(point, e, SETTINGS, fonts)
        self.assertEqual((img.size, img.mode), ((dv.W, dv.VH), "RGBA"))
        self.assertGreater(img.getpixel((60, 60))[3], 0)            # shot card, top-left
        self.assertGreater(img.getpixel((dv.W - 60, 60))[3], 0)     # scoreboard, top-right
        self.assertEqual(img.getpixel((dv.W // 2, dv.VH - 10))[3], 0)
        self.assertEqual(dv.render_tally([e], fonts).shape[1], __import__("analysis.court_views").court_views.TOP["w"])


class CourtViewTests(unittest.TestCase):
    def test_serve_and_rally_landings(self):
        from analysis import court_views as cv
        serve = {"contact_t": 10.0, "bounce_t": 10.6, "server_end": "far", "server_court": [3.0, 23.77],
                 "contact_height_m": 2.6, "bounce_court": [6.5, 7.0], "call": "in"}
        # the return: hit near the near baseline at 11.2, flying long over the far baseline
        ret = {"t0": 11.2, "t1": 11.9, "p": [5.0, 1.0, 1.0], "v": [0.0, 34.0, 4.0]}
        path = cv.build_path([ret], [serve], [(11.2, "near")])
        self.assertEqual([round(b["t"], 1) for b in path.landings][:1], [10.6])
        self.assertTrue(path.landings[0]["in"])
        far = [b for b in path.landings if b["t"] > 11.2]
        self.assertTrue(far)  # flown on past its last tracked point until it lands...
        self.assertGreater(far[0]["y"], 23.77)
        self.assertFalse(far[0]["in"])  # ...beyond the far baseline: out
        x, y, z = path.at(10.3)
        self.assertLess(abs(y - (23.77 + (7.0 - 23.77) * 0.5)), 0.2)  # halfway along the serve
        self.assertEqual(len(path.hits), 1)

    def test_an_arc_starting_below_ground_does_not_blow_up(self):
        from analysis import court_views as cv
        path = cv.build_path([{"t0": 0.0, "t1": 0.4, "p": [4.6, 24.0, -0.5], "v": [1.3, 5.6, 5.7]}], [], [])
        self.assertTrue(all(-10 < q[1] < 25 and -10 < q[2] < 40 and q[0] >= 0 for q in path.samples))

    def test_frame_draws_on_the_background(self):
        try:
            dv.ensure_pillow()
        except Exception:
            self.skipTest("Pillow unavailable")
        from analysis import court_views as cv
        bg = cv.background(dv.Fonts())
        self.assertEqual(bg.shape, (cv.OUT_H, cv.OUT_W, 3))
        path = cv.build_path([{"t0": 0.0, "t1": 1.0, "p": [5.0, 2.0, 1.0], "v": [0.0, 18.0, 4.0]}], [], [])
        frame = cv.draw_dynamic(bg.copy(), path, 0.5, 0.0, [])
        self.assertGreater(int((frame != bg).any(axis=2).sum()), 100)  # path and ball drawn


if __name__ == "__main__":
    unittest.main()
