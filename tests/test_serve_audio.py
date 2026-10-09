import unittest
from unittest import mock

import numpy as np

from analysis import serve_audio
from analysis.audio_onset import SR, onset_envelope, strongest_onset


def clicks(times, dur=1.5, noise=0.002):
    rng = np.random.default_rng(0)
    x = rng.normal(0, noise, int(dur * SR)).astype(np.float32)
    for t in times:
        k = int(t * SR)
        x[k:k + 80] += rng.normal(0, 0.5, 80).astype(np.float32) * np.hanning(80).astype(np.float32)
    return x


class OnsetTests(unittest.TestCase):
    def test_click_onset_found_to_a_few_ms(self):
        env = onset_envelope(clicks([0.6]))
        t, prom = strongest_onset(env, 0.0, 0.4, 0.8)
        self.assertAlmostEqual(t, 0.6, delta=0.004)
        self.assertGreater(prom, 10)


class FarContactTests(unittest.TestCase):
    def test_far_contact_moved_onto_the_sound(self):
        cam = mock.Mock()
        cam.C = np.array([6.2, -5.6, 3.6])
        serve = {"contact_src": "flight_start", "server_end": "far", "contact_t": 10.20, "bounce_t": 11.10,
                 "server_court": [4.0, 23.77], "contact_height_m": 2.3, "distance_m": 17.0, "speed_kmh": 100}
        delay = float(np.linalg.norm(np.array([4.0, 23.77, 2.3]) - cam.C)) / 343.0
        offset = 0.09
        true_contact = 10.00
        audio_t0 = serve["contact_t"] - serve_audio.SEARCH_BEFORE - 0.2

        def fake_read(src, t0, t1):
            return clicks([true_contact + delay + offset - t0], dur=t1 - t0)

        with mock.patch.object(serve_audio, "read_audio", side_effect=fake_read):
            moved = serve_audio.refine_far_contacts([serve], "x.mp4", cam, offset)
        self.assertEqual(moved, 1)
        self.assertAlmostEqual(serve["contact_t"], true_contact, delta=0.006)
        self.assertEqual(serve["contact_src"], "audio")
        self.assertLess(serve["speed_kmh"], 100)  # longer flight than the tracker thought
        self.assertAlmostEqual(audio_t0, 9.6, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
