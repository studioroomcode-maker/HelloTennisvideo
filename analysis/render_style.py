"""One look for everything rendered into exported video: dashboard panel, shot pop-ups,
broadcast scoreboard, ball trail. It matches the app (web/styles.css): near-black surfaces,
one accent (ball yellow-green) for what is live or measured, team A sky / team B apricot,
Pretendard for Korean and numbers.

Pretendard is fetched once into models/fonts (it is what the app's web page loads); without
a network the system Korean fonts are used instead.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / "models" / "fonts"
PRETENDARD_URL = "https://cdn.jsdelivr.net/npm/pretendard@1.3.9/dist/public/static/Pretendard-{w}.otf"
WEIGHTS = ("Medium", "SemiBold", "Bold", "ExtraBold")

# RGB
BG = (14, 16, 18)
SURFACE = (21, 24, 27)
SURFACE_2 = (27, 31, 35)
SURFACE_3 = (35, 40, 45)
LINE = (42, 48, 54)
LINE_2 = (54, 61, 68)
TEXT = (237, 240, 242)
TEXT_2 = (169, 178, 186)
TEXT_3 = (123, 132, 141)
ACCENT = (205, 230, 106)
ACCENT_INK = (21, 26, 5)
OK = (103, 199, 147)
BAD = (238, 128, 128)
WARN = (227, 184, 100)
TEAM = {"A": (132, 195, 236), "B": (238, 164, 137)}

SHOT_KO = {"serve": "서브", "forehand": "포핸드", "backhand": "백핸드", "volley": "발리", "smash": "스매시",
           "lob": "로브", "dropshot": "드롭샷", "other": "타구"}
SPIN_KO = {"flat": "플랫", "slice": "슬라이스", "kick": "킥", "topspin": "탑스핀"}
DIR_KO = {"T": "T", "body": "바디", "wide": "와이드"}

_FALLBACK = {
    "Medium": ["malgun.ttf", "NanumBarunGothic.otf", "AppleSDGothicNeo.ttc", "NotoSansCJK-Regular.ttc"],
    "SemiBold": ["malgunbd.ttf", "NanumBarunGothicBold.otf", "AppleSDGothicNeo.ttc", "NotoSansCJK-Bold.ttc"],
    "Bold": ["malgunbd.ttf", "NanumBarunGothicBold.otf", "AppleSDGothicNeo.ttc", "NotoSansCJK-Bold.ttc"],
    "ExtraBold": ["malgunbd.ttf", "NanumGothicExtraBold.ttf", "AppleSDGothicNeo.ttc", "NotoSansCJK-Bold.ttc"],
}
_SYS_DIRS = [Path("C:/Windows/Fonts"), Path.home() / "AppData/Local/Microsoft/Windows/Fonts",
             Path("/System/Library/Fonts"), Path("/Library/Fonts"), Path("/usr/share/fonts/opentype/noto"),
             Path("/usr/share/fonts/truetype")]


def ensure_fonts() -> dict[str, Path | None]:
    """Paths for each weight: Pretendard if it is (or can be) downloaded, else a system Korean font."""
    out = {}
    for w in WEIGHTS:
        p = FONT_DIR / f"Pretendard-{w}.otf"
        if not p.exists():
            try:
                FONT_DIR.mkdir(parents=True, exist_ok=True)
                tmp = p.with_suffix(".part")
                with urllib.request.urlopen(PRETENDARD_URL.format(w=w), timeout=20) as r, tmp.open("wb") as f:
                    f.write(r.read())
                if tmp.stat().st_size > 100_000:
                    tmp.replace(p)
                else:
                    tmp.unlink(missing_ok=True)
            except Exception:
                pass
        if p.exists():
            out[w] = p
            continue
        out[w] = next((d / n for n in _FALLBACK[w] for d in _SYS_DIRS if (d / n).exists()), None)
    return out


class Fonts:
    """Cached Pillow fonts: fonts.get("Bold", 28)."""

    def __init__(self):
        from PIL import ImageFont
        self._IF = ImageFont
        self.paths = ensure_fonts()
        self._cache = {}

    def get(self, weight: str, size: int):
        key = (weight, size)
        if key not in self._cache:
            path = self.paths.get(weight)
            self._cache[key] = self._IF.truetype(str(path), size) if path else self._IF.load_default(size)
        return self._cache[key]


def rgba(c, a=255):
    return tuple(c) + (a,)
