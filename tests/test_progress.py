import unittest
from core.media import _parse_ffmpeg_clock, _parse_speed


class ProgressParsingTests(unittest.TestCase):
    def test_clock(self):
        self.assertAlmostEqual(_parse_ffmpeg_clock("00:02:03.500000"), 123.5, places=3)
        self.assertAlmostEqual(_parse_ffmpeg_clock("01:00:00.000000"), 3600.0, places=3)

    def test_speed(self):
        self.assertAlmostEqual(_parse_speed("7.42x"), 7.42, places=2)
        self.assertEqual(_parse_speed("N/A"), 0.0)


if __name__ == "__main__":
    unittest.main()
