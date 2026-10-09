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
    def test_panel_and_popup_render(self):
        try:
            dv.ensure_pillow()
        except Exception:
            self.skipTest("Pillow unavailable")
        fonts = dv.Fonts()
        events = [
            {"no": 1, "shot": "serve", "serve_number": 1, "serve_type": "flat", "serve_direction": "T",
             "speed_kmh": 150, "point_id": "p1", "result": "in", "player": "A1", "team": "A"},
            {"no": 2, "shot": "forehand", "spin": "topspin", "speed_kmh": 104, "team": "B", "point_id": "p1"},
        ]
        panel = dv.render_panel(events, fonts, {"p1": "deuce"})
        self.assertEqual(panel.size, (dv.W, dv.PH))
        self.assertEqual(dv.render_panel([], fonts, {}).size, (dv.W, dv.PH))
        pop = dv.render_popup(events[0], SETTINGS, fonts)
        self.assertEqual(pop.mode, "RGBA")
        self.assertEqual(pop.getpixel((dv.W - 5, dv.VH - 5))[3], 0)  # transparent outside the card

    def test_scoreboard_rows_and_size(self):
        from analysis import scoreboard
        point = {"score_before": {"point_a": "30", "point_b": "40", "set_games_a": 1, "set_games_b": 0,
                                  "server_id": "B1", "team_a_end": "far"}}
        rows = scoreboard.score_rows(point, dict(SETTINGS, scoreboard_position_mode="court"))
        self.assertEqual([r["team"] for r in rows], ["B", "A"])  # camera-side team on top
        self.assertEqual([(r["points"], r["games"], r["serving"]) for r in rows], [("40", 0, True), ("30", 1, False)])
        rows = scoreboard.score_rows(point, dict(SETTINGS, scoreboard_position_mode="team"))
        self.assertEqual(rows[0]["team"], "A")
        try:
            dv.ensure_pillow()
        except Exception:
            self.skipTest("Pillow unavailable")
        img = scoreboard.render(point, SETTINGS, (1920, 1080))
        self.assertEqual((img.size, img.mode), ((1920, 1080), "RGBA"))
        self.assertGreater(img.getpixel((60, 60))[3], 0)    # card in the top-left corner
        self.assertEqual(img.getpixel((1900, 1060))[3], 0)  # transparent elsewhere


if __name__ == "__main__":
    unittest.main()
