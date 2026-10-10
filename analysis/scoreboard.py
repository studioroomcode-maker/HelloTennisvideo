"""Broadcast scoreboard drawn as an image per point (same look as the app and the dashboard).

Two rows, the serving team marked with a ball-coloured dot: team colour, names, games, and
the point score in a raised cell. Sets appear once a set has been won. Rows follow the
court (the camera-side team on top, as in the app) or stay A over B, per the scoreboard setting. Drawing
it ourselves instead of burning ASS subtitles keeps the look and drops the need for an
FFmpeg built with libass.
"""
from __future__ import annotations

from analysis.render_style import ACCENT, BG, LINE_2, SURFACE_3, TEAM, TEXT, TEXT_2, Fonts, rgba

BASE_W = 1920  # layout is designed at 1080p and scaled to the video width


def score_rows(point: dict, settings: dict) -> list[dict]:
    st = point.get("score_before") or {}
    tb = bool(st.get("tiebreak"))
    rows = []
    for t in ("A", "B"):
        k = t.lower()
        names = [n for n in settings.get(f"team_{k}") or [] if n]
        rows.append({"team": t, "name": " · ".join(names) or f"Team {t}",
                     "sets": int(st.get(f"sets_{k}") or 0), "games": int(st.get(f"set_games_{k}") or 0),
                     "points": str(st.get(f"tiebreak_{k}") if tb else st.get(f"point_{k}", "0")),
                     "serving": str(st.get("server_id") or "").upper().startswith(t)})
    if settings.get("scoreboard_position_mode", "court") == "court" and st.get("team_a_end", "near") == "far":
        rows.reverse()  # camera-side team on the top row, as in the app's scoreboard
    return rows


def render(point: dict, settings: dict, size: tuple[int, int], fonts: Fonts | None = None, corner: str = "tl"):
    """Transparent RGBA frame of the video's size with the scoreboard in a top corner ("tl" or "tr")."""
    from PIL import Image, ImageDraw
    fonts = fonts or Fonts()
    W, H = size
    s = W / BASE_W
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    rows = score_rows(point, settings)
    show_sets = any(r["sets"] for r in rows)
    f_name, f_num, f_pts = fonts.get("Bold", round(26 * s)), fonts.get("Bold", round(28 * s)), fonts.get("ExtraBold", round(30 * s))
    pad, row_h, gap = 18 * s, 50 * s, 2 * s
    name_w = max(d.textlength(r["name"], font=f_name) for r in rows)
    col = 46 * s
    pts_w = 70 * s
    width = pad + 14 * s + name_w + 30 * s + (col if show_sets else 0) + col + 10 * s + pts_w
    x0, y0 = (W - 40 * s - width if corner == "tr" else 40 * s), 40 * s
    height = row_h * 2 + gap
    d.rounded_rectangle([x0 + 2 * s, y0 + 6 * s, x0 + width + 2 * s, y0 + height + 6 * s], radius=14 * s, fill=(0, 0, 0, 80))
    d.rounded_rectangle([x0, y0, x0 + width, y0 + height], radius=14 * s, fill=rgba(BG, 232))
    d.line([x0 + pad, y0 + row_h + gap / 2, x0 + width - pts_w, y0 + row_h + gap / 2], fill=rgba(LINE_2), width=max(1, round(s)))
    px = x0 + width - pts_w
    d.rounded_rectangle([px, y0, x0 + width, y0 + height], radius=14 * s, fill=rgba(SURFACE_3))
    d.rectangle([px, y0, px + 14 * s, y0 + height], fill=rgba(SURFACE_3))
    for i, r in enumerate(rows):
        cy = y0 + i * (row_h + gap) + row_h / 2
        x = x0 + pad
        d.ellipse([x, cy - 5 * s, x + 10 * s, cy + 5 * s], fill=rgba(TEAM[r["team"]]))
        x += 24 * s
        d.text((x, cy), r["name"], font=f_name, fill=rgba(TEXT), anchor="lm")
        if r["serving"]:
            bx = x + d.textlength(r["name"], font=f_name) + 14 * s
            d.ellipse([bx, cy - 6 * s, bx + 12 * s, cy + 6 * s], fill=rgba(ACCENT))
        x = x0 + pad + 14 * s + name_w + 30 * s
        if show_sets:
            d.text((x + col / 2, cy), str(r["sets"]), font=f_num, fill=rgba(TEXT_2), anchor="mm")
            x += col
        d.text((x + col / 2, cy), str(r["games"]), font=f_num, fill=rgba(TEXT), anchor="mm")
        d.text((px + pts_w / 2, cy), r["points"], font=f_pts, fill=rgba(TEXT), anchor="mm")
    return img
