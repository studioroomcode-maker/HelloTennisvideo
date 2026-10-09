import json
import shutil
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from analysis import gemini_client
from analysis import shot_analysis as sa

SETTINGS = {"mode": "doubles", "team_a": ["선수A1", "선수A2"], "team_b": ["선수B1", "선수B2"]}


def point(pid, start, end, server="A1", a_end="near"):
    return {"id": pid, "file_id": "f1", "start": start, "end": end,
            "score_before": {"server_id": server, "team_a_end": a_end}}


FILE = {"id": "f1", "name": "m.mp4", "metadata": {"duration": 600.0}}


class SegmentTests(unittest.TestCase):
    def test_points_are_padded_and_overlaps_split(self):
        segs = sa.build_segments(FILE, [point("p1", 10, 20), point("p2", 21, 30)], "points")
        self.assertEqual(segs[0]["start"], 8.0)
        # p1 padded end (21) overlaps p2 padded start (19) -> cut halfway between 20 and 21
        self.assertAlmostEqual(segs[0]["end"], 20.5)
        self.assertAlmostEqual(segs[1]["start"], 20.5)
        self.assertEqual(segs[1]["end"], 31.0)
        self.assertEqual(segs[1]["point_no"], 2)

    def test_full_video_windows_cover_everything_once(self):
        segs = sa.build_segments(FILE, [], "full")
        self.assertEqual(segs[0]["own_start"], 0.0)
        self.assertEqual(segs[-1]["own_end"], 600.0)
        for a, b in zip(segs, segs[1:]):
            self.assertEqual(a["own_end"], b["own_start"])
            self.assertGreater(a["end"], b["start"])  # windows overlap

    def test_chunks_and_time_mapping(self):
        segs = sa.build_segments(FILE, [point("p1", 10, 20), point("p2", 100, 110)], "points")
        chunks = sa.group_chunks(segs)
        self.assertEqual(len(chunks), 1)
        ch = chunks[0]
        self.assertAlmostEqual(ch["clip_duration"], 26.0)
        t, seg = sa.clip_to_source(ch, 15.0)  # 13s into clip seg 1 (len 13) -> 2s into seg 2
        self.assertEqual(seg["point_id"], "p2")
        self.assertAlmostEqual(t, 100.0)

    def test_many_points_split_into_chunks(self):
        pts = [point(f"p{i}", i * 30, i * 30 + 20) for i in range(20)]
        chunks = sa.group_chunks(sa.build_segments(FILE, pts, "points"))
        self.assertGreater(len(chunks), 1)
        for ch in chunks:
            self.assertLessEqual(ch["clip_duration"], sa.CHUNK_MAX_SEC)


class EventTests(unittest.TestCase):
    def setUp(self):
        segs = sa.build_segments(FILE, [point("p1", 10, 20, "B1", "far"), point("p2", 100, 110)], "points")
        self.chunk = sa.group_chunks(segs)[0]

    def norm(self, events):
        warnings = []
        return sa.normalize_events(events, self.chunk, "f1", SETTINGS, warnings), warnings

    def test_serve_attributed_to_server_and_strokes_to_team(self):
        evs, _ = self.norm([
            {"t": 2.5, "shot": "serve", "side": "near", "serve_type": "kick", "speed_kmh": 120, "confidence": 80},
            {"t": 4.0, "shot": "forehand", "side": "near", "spin": "topspin", "speed_kmh": 90, "confidence": 70},
            {"t": 5.0, "shot": "backhand", "side": "far", "spin": "slice", "speed_kmh": 0, "confidence": 70},
        ])
        serve, fh, bh = evs
        self.assertEqual(serve["player"], "B1")
        self.assertEqual(serve["side"], "near")  # Team B is near because Team A is far
        self.assertAlmostEqual(serve["t"], 10.5)
        self.assertEqual(fh["team"], "B")
        self.assertEqual(bh["team"], "A")
        self.assertIsNone(bh["speed_kmh"])

    def test_unrealistic_speed_dropped_with_warning(self):
        evs, warnings = self.norm([{"t": 2.0, "shot": "serve", "side": "near", "speed_kmh": 400, "confidence": 50}])
        self.assertIsNone(evs[0]["speed_kmh"])
        self.assertTrue(warnings)

    def test_serve_numbers_fixed_within_point_and_aggregate(self):
        evs, warnings = self.norm([
            {"t": 2.0, "shot": "serve", "side": "near", "serve_number": 1, "serve_type": "flat",
             "result": "in", "speed_kmh": 160, "confidence": 80},
            {"t": 4.0, "shot": "serve", "side": "near", "serve_number": 1, "serve_type": "kick",
             "result": "in", "speed_kmh": 120, "confidence": 80},
            {"t": 5.2, "shot": "forehand", "side": "far", "spin": "topspin", "confidence": 80, "speed_kmh": 100},
            {"t": 5.3, "shot": "forehand", "side": "far", "spin": "topspin", "confidence": 60, "speed_kmh": 0},
        ])
        evs = sa.dedupe_events(evs)
        sa.fix_serve_numbers(evs, warnings)
        serves = [e for e in evs if e["shot"] == "serve"]
        self.assertEqual([s["serve_number"] for s in serves], [1, 2])
        self.assertEqual(serves[0]["result"], "fault")
        summary = sa.aggregate(evs, SETTINGS)
        self.assertEqual(summary["counts"]["forehand"], 1)  # near-duplicate removed
        self.assertEqual(summary["serve"]["by_type"]["flat"]["speed"]["max"], 160)
        self.assertEqual(summary["serve"]["by_number"]["1"]["in_rate"], 0)
        self.assertEqual(summary["servers"]["B1"]["label"], "B1(선수B1)")
        self.assertEqual(summary["groups"]["teamA"]["counts"]["forehand"], 1)


# ── End-to-end with real FFmpeg and a fake Gemini endpoint ────────

class FakeGemini(BaseHTTPRequestHandler):
    prompts: list = []

    def log_message(self, *a):
        pass

    def _send(self, obj, headers=None):
        body = json.dumps(obj).encode()
        self.send_response(200)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        assert self.headers.get("x-goog-api-key") == "test-key"
        if self.path.startswith("/upload/v1beta/files"):
            port = self.server.server_address[1]
            return self._send({}, {"X-Goog-Upload-URL": f"http://127.0.0.1:{port}/session/1"})
        if self.path.startswith("/session/"):
            assert len(body) > 1000
            return self._send({"file": {"name": "files/x1", "uri": "https://fake/files/x1", "state": "PROCESSING"}})
        if ":generateContent" in self.path:
            req = json.loads(body)
            prompt = req["contents"][0]["parts"][1]["text"]
            FakeGemini.prompts.append(prompt)
            events = [
                {"t": 1.0, "shot": "serve", "side": "near", "serve_number": 1, "serve_type": "slice",
                 "serve_direction": "wide", "result": "in", "speed_kmh": 135, "confidence": 85},
                {"t": 2.5, "shot": "backhand", "side": "far", "spin": "slice", "result": "in",
                 "speed_kmh": 70, "confidence": 75},
                {"t": 8.0, "shot": "forehand", "side": "near", "spin": "topspin", "result": "winner",
                 "speed_kmh": 110, "confidence": 80},
            ]
            text = json.dumps({"videoContext": {"cameraAngle": "베이스라인 뒤"}, "events": events,
                               "observations": ["테스트 관찰"]})
            return self._send({"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
                               "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 200}})
        self.send_error(404)

    def do_GET(self):
        if self.path.startswith("/v1beta/files/"):
            return self._send({"name": "files/x1", "uri": "https://fake/files/x1", "state": "ACTIVE"})
        self.send_error(404)

    def do_DELETE(self):
        self._send({})


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")
class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.video = self.tmp / "match.mp4"
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=40",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=40",
                        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest",
                        str(self.video)], check=True)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeGemini)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.old_base = gemini_client.API_BASE
        gemini_client.API_BASE = f"http://127.0.0.1:{self.server.server_address[1]}"
        FakeGemini.prompts = []

    def tearDown(self):
        gemini_client.API_BASE = self.old_base
        self.server.shutdown()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_points_pipeline_maps_events_to_source_time(self):
        f = {"id": "f1", "name": "match.mp4", "path": str(self.video), "proxy": None,
             "metadata": {"duration": 40.0, "audio_codec": "aac"}}
        project = {"settings": SETTINGS, "files": [f],
                   "points": [point("p1", 5, 10, "A1", "near"), point("p2", 25, 30, "B1", "far")]}
        updates = []
        job = sa.ShotAnalysisJob(project=project, file_entry=f, api_key="test-key", model="gemini-test",
                                 scope="points", quality="auto", high_res=False, cache_dir=self.tmp,
                                 report=updates.append)
        res = job.run()
        self.assertEqual(res["chunk_count"], 1)
        self.assertIn("포인트 #2", FakeGemini.prompts[0])
        self.assertIn("서버 B1(선수B1)", FakeGemini.prompts[0])
        ts = [e["t"] for e in res["events"]]
        # clip: p1 = src 3..11 (8s), p2 = src 23..31. clip t=8.0 -> p1 end region? 8.0 >= 8 -> p2 at src 23.0
        self.assertAlmostEqual(ts[0], 4.0, delta=0.15)
        self.assertAlmostEqual(ts[2], 23.0, delta=0.2)
        self.assertEqual(res["events"][2]["team"], "B")  # near side while Team A is far
        self.assertEqual(res["summary"]["counts"]["forehand"], 1)
        self.assertEqual(res["summary"]["serve"]["by_type"]["slice"]["speed"]["avg"], 135)
        self.assertEqual(updates[-1]["percent"], 100.0)

        # Second run is served from the per-chunk cache without calling Gemini again.
        FakeGemini.prompts = []
        res2 = sa.ShotAnalysisJob(project=project, file_entry=f, api_key="test-key", model="gemini-test",
                                  scope="points", quality="auto", high_res=False, cache_dir=self.tmp).run()
        self.assertEqual(FakeGemini.prompts, [])
        self.assertEqual([e["t"] for e in res2["events"]], ts)


if __name__ == "__main__":
    unittest.main()
