"""The ball's path on the whole court, seen from above, from behind the camera-side baseline and
from the side, for the dashboard video (SwingVision-style).

Where the ball went comes from the 3D flight arcs fitted to the tracked ball (analysis/rally3d.py,
saved per point by the serve measurement) and, for serves, from the measured contact, bounce and
flight time. Between two arcs the earlier one is flown on until it lands or the next one starts;
from a landing to the next arc the ball is joined by a straight line. A landing is where a path
reaches the ground; it is "in" when it falls inside the doubles court (serves use the measured call).

Drawing: the court, axes and labels are drawn once; each video frame copies that and adds the
point's paths, landings and the moving ball with OpenCV (anti-aliased), so a frame costs a few ms.
"""
from __future__ import annotations

import math

import numpy as np

from analysis import render_style as rs
from analysis.serve_speed import COURT_L, COURT_W, NET_Y

G = 9.81
STEP = 1 / 60                 # s between path samples
JOIN_MAX = 1.6                # s: arcs further apart than this are not joined
NET_H = 0.914
SINGLES_IN = 1.372
SERVICE_Y = (NET_Y - 6.40, NET_Y + 6.40)

# canvas layout (1920x1080): footage top-left, views around it
OUT_W, OUT_H = 1920, 1080
FOOT_W, FOOT_H = 1344, 756
X_RANGE = (-2.5, COURT_W + 2.5)
Y_RANGE = (-3.5, COURT_L + 3.5)
TOP = {"x": 1344 + 24, "y": 104, "w": 576 - 48, "h": 956}
SIDE = {"x": 24, "y": 812, "w": 820, "h": 232, "zmax": 6.0}
FRONT = {"x": 884, "y": 812, "w": 436, "h": 232, "zmax": 6.0}
JOIN_GAP = 0.6                # s: only flights this close are joined by a straight line
JOIN_DIST = 4.0               # m: ... and only when the line is this short


def _bgr(c):
    return (int(c[2]), int(c[1]), int(c[0]))


def _mix(c, toward, f):
    return tuple(int(a * (1 - f) + b * f) for a, b in zip(c, toward))


# ── ball path ─────────────────────────────────────────────────────

def _fly(p, v, t):
    """Ballistic position t s after (p, v), with the same drag the arcs were fitted with."""
    from analysis.rally3d import fly
    return fly(p, v, np.array([max(0.0, t)]))[0]


def _sample_arc(p, v, t0, t1, step=STEP):
    from analysis.rally3d import fly
    n = max(2, int(math.ceil((t1 - t0) / step)) + 1)
    rel = np.linspace(0.0, max(1e-3, t1 - t0), n)
    pos = fly(np.asarray(p, float), np.asarray(v, float), rel)
    return [(t0 + r, *q) for r, q in zip(rel, pos)]


class PointPath:
    """One point's ball path: samples [(t, x, y, z)], landings and hit spots."""

    def __init__(self, samples, landings, hits):
        self.samples = samples
        self.landings = landings          # [{"t", "x", "y", "in"}]
        self.hits = hits                  # [{"t", "x", "y", "z", "side"}]
        self._ts = np.array([s[0] for s in samples]) if samples else np.zeros(0)

    def upto(self, t):
        if not self.samples:
            return []
        k = int(np.searchsorted(self._ts, t, side="right"))
        return self.samples[:k]

    def at(self, t):
        if not self.samples or t < self._ts[0] - 0.05 or t > self._ts[-1] + 0.05:
            return None
        k = int(np.clip(np.searchsorted(self._ts, t), 1, len(self._ts) - 1))
        a, b = self.samples[k - 1], self.samples[k]
        f = 0.0 if b[0] == a[0] else min(1.0, max(0.0, (t - a[0]) / (b[0] - a[0])))
        return tuple(a[i] + (b[i] - a[i]) * f for i in range(1, 4))


BOUNDS = ((-4.0, COURT_W + 4.0), (-6.0, COURT_L + 6.0), (-0.5, 9.0))  # where a real ball can be (m)
MAX_SPEED = 55.0              # m/s: faster "arcs" are bad fits


def _plausible(sample) -> bool:
    _, x, y, z = sample
    return all(lo <= c <= hi for c, (lo, hi) in zip((x, y, z), BOUNDS))


def _arc_ok(a) -> bool:
    """An arc is drawn only if it stays where a ball can be while it was tracked, at a ball's speed."""
    if float(np.linalg.norm(a["v"])) > MAX_SPEED:
        return False
    return all(_plausible(q) for q in _sample_arc(a["p"], a["v"], a["t0"], a["t1"], step=0.05))


def _in_court(x, y, tol=0.05):
    return -tol <= x <= COURT_W + tol and -tol <= y <= COURT_L + tol


def build_path(arcs: list[dict], serves: list[dict], hit_times: list[tuple[float, str]]) -> PointPath:
    """arcs: [{"t0","t1","p","v"}] of one point; serves: measured serves in it; hit_times: [(t, side)]."""
    pieces = []  # (t0, t1, p, v, kind)
    for s in serves:
        if not s.get("bounce_court") or not s.get("contact_height_m"):
            continue
        T = float(s["bounce_t"]) - float(s["contact_t"])
        if T <= 0.15:
            continue
        x0 = float(s["server_court"][0])
        y0 = 0.0 if s["server_end"] == "near" else COURT_L
        z0 = float(s["contact_height_m"])
        bx, by = s["bounce_court"]
        v = [(bx - x0) / T, (by - y0) / T, (0.0 - z0 + 0.5 * G * T * T) / T]
        pieces.append({"t0": float(s["contact_t"]), "t1": float(s["bounce_t"]), "p": [x0, y0, z0], "v": v,
                       "kind": "serve", "call": s.get("call")})
    serve_spans = [(p["t0"] - 0.1, p["t1"] + 0.05) for p in pieces]
    for a in arcs:
        if any(lo <= a["t0"] <= hi and a["t1"] <= hi + 0.3 for lo, hi in serve_spans):
            continue  # the measured serve already draws this flight
        if not _arc_ok(a):
            continue
        pieces.append({**a, "kind": "arc"})
    pieces.sort(key=lambda q: q["t0"])

    samples, landings = [], []
    for k, pc in enumerate(pieces):
        nxt = pieces[k + 1] if k + 1 < len(pieces) else None
        gap_end = nxt["t0"] if nxt and nxt["t0"] - pc["t1"] <= JOIN_MAX else pc["t1"] + 0.3
        if pc["kind"] == "serve":
            # gravity only (no drag) for the reconstructed serve, so it lands where it was measured
            n = max(2, int(math.ceil((pc["t1"] - pc["t0"]) / STEP)) + 1)
            seg = []
            for r in np.linspace(0.0, pc["t1"] - pc["t0"], n):
                seg.append((pc["t0"] + r, pc["p"][0] + pc["v"][0] * r, pc["p"][1] + pc["v"][1] * r,
                            max(0.0, pc["p"][2] + pc["v"][2] * r - 0.5 * G * r * r)))
            land = (pc["t1"], seg[-1][1], seg[-1][2], pc.get("call") in ("in", "close"))
            if pc.get("call") != "net":
                landings.append({"t": land[0], "x": land[1], "y": land[2], "in": land[3], "serve": True})
            end_t = pc["t1"]
        else:
            seg = _sample_arc(pc["p"], pc["v"], pc["t0"], max(pc["t1"], min(gap_end, pc["t1"] + 0.6)))
            bad = next((i for i, q in enumerate(seg) if q[0] > pc["t1"] and not _plausible(q)), None)
            if bad is not None:  # flown on past its tracked part, it left the court area: stop there
                seg = seg[:bad]
            # a landing is the ball coming down through the ground from above it
            cut = next((i for i in range(1, len(seg)) if seg[i][3] <= 0.0 < seg[i - 1][3]
                        and seg[i][0] > pc["t0"] + 0.05), None)
            if cut is not None:
                a, b = seg[cut - 1], seg[cut]
                f = min(1.0, max(0.0, a[3] / max(1e-6, a[3] - b[3])))
                land = tuple(a[i] + (b[i] - a[i]) * f for i in range(4))
                seg = seg[:cut] + [(land[0], land[1], land[2], 0.0)]
                landings.append({"t": land[0], "x": land[1], "y": land[2], "in": _in_court(land[1], land[2]),
                                 "serve": False})
            end_t = seg[-1][0]
        if samples and seg and seg[0][0] <= samples[-1][0]:
            seg = [q for q in seg if q[0] > samples[-1][0]]
        samples += seg
        p1 = nxt["p"] if nxt else None
        close = (nxt is not None and samples and nxt["t0"] - end_t <= JOIN_GAP
                 and math.dist(samples[-1][1:], p1) <= JOIN_DIST)
        if close:
            # straight join from where this flight ended (often a landing) to where the next starts
            q0 = samples[-1]
            n = max(1, int((nxt["t0"] - q0[0]) / STEP))
            for i in range(1, n):
                f = i / n
                samples.append((q0[0] + (nxt["t0"] - q0[0]) * f, q0[1] + (p1[0] - q0[1]) * f,
                                q0[2] + (p1[1] - q0[2]) * f, max(0.0, q0[3] + (p1[2] - q0[3]) * f)))
    path = PointPath(samples, _dedupe(landings), [])
    for t, side in hit_times:
        pos = path.at(t)
        if pos is not None:
            path.hits.append({"t": t, "x": pos[0], "y": pos[1], "z": pos[2], "side": side})
    return path


def _dedupe(landings):
    out = []
    for b in sorted(landings, key=lambda b: b["t"]):
        if out and b["t"] - out[-1]["t"] < 0.2:
            continue
        out.append(b)
    return out


# ── drawing ───────────────────────────────────────────────────────

def _top_xy(x, y):
    w, h = TOP["w"], TOP["h"]
    sx = w / (X_RANGE[1] - X_RANGE[0])
    sy = h / (Y_RANGE[1] - Y_RANGE[0])
    s = min(sx, sy)
    ox = TOP["x"] + (w - s * (X_RANGE[1] - X_RANGE[0])) / 2
    oy = TOP["y"] + (h - s * (Y_RANGE[1] - Y_RANGE[0])) / 2
    return ox + (x - X_RANGE[0]) * s, oy + (Y_RANGE[1] - y) * s  # near baseline at the bottom


def _front_xy(x, z):
    s = min(FRONT["w"] / (X_RANGE[1] - X_RANGE[0]), FRONT["h"] / FRONT["zmax"])
    ox = FRONT["x"] + (FRONT["w"] - s * (X_RANGE[1] - X_RANGE[0])) / 2
    return ox + (x - X_RANGE[0]) * s, FRONT["y"] + FRONT["h"] - z * s


def _side_xy(y, z):
    s = min(SIDE["w"] / (Y_RANGE[1] - Y_RANGE[0]), SIDE["h"] / SIDE["zmax"])
    return SIDE["x"] + (y - Y_RANGE[0]) * s, SIDE["y"] + SIDE["h"] - z * s


def _pt(xy):
    return int(round(xy[0] * 4)), int(round(xy[1] * 4))  # cv2 shift=2 (quarter-pixel)


def background(fonts=None):
    """The static canvas: dark surface, the three court views and their labels (RGB uint8)."""
    import cv2
    from PIL import Image, ImageDraw
    fonts = fonts or rs.Fonts()
    img = Image.new("RGB", (OUT_W, OUT_H), rs.BG)
    d = ImageDraw.Draw(img)
    lab = fonts.get("SemiBold", 20)
    sub = fonts.get("Medium", 15)
    d.text((TOP["x"], 64), "탑뷰", font=lab, fill=rs.TEXT_2)
    d.text((TOP["x"] + d.textlength("탑뷰", font=lab) + 10, 68), "위에서 · 카메라 쪽이 아래", font=sub, fill=rs.TEXT_3)
    d.text((FRONT["x"], FRONT["y"] - 34), "프론트뷰", font=lab, fill=rs.TEXT_2)
    d.text((FRONT["x"] + d.textlength("프론트뷰", font=lab) + 10, FRONT["y"] - 30), "카메라 쪽에서 · 높이", font=sub, fill=rs.TEXT_3)
    d.text((SIDE["x"], SIDE["y"] - 34), "사이드뷰", font=lab, fill=rs.TEXT_2)
    d.text((SIDE["x"] + d.textlength("사이드뷰", font=lab) + 10, SIDE["y"] - 30), "옆에서 · 왼쪽이 카메라 쪽", font=sub, fill=rs.TEXT_3)
    d.line([1344, 0, 1344, OUT_H], fill=rs.LINE, width=1)
    d.line([0, FOOT_H, 1344, FOOT_H], fill=rs.LINE, width=1)
    d.line([FRONT["x"] - 20, FOOT_H + 24, FRONT["x"] - 20, OUT_H - 24], fill=rs.LINE, width=1)
    arr = np.array(img)[:, :, ::-1].copy()  # BGR for OpenCV

    line, net = _bgr(rs.TEXT_3), _bgr(rs.TEXT_2)
    # top view: court surface, lines, net
    c = [_top_xy(0, 0), _top_xy(COURT_W, 0), _top_xy(COURT_W, COURT_L), _top_xy(0, COURT_L)]
    cv2.fillPoly(arr, [np.array([_pt(p) for p in c], np.int32)], _bgr(rs.SURFACE_2), cv2.LINE_AA, shift=2)
    segs = [((0, 0), (COURT_W, 0)), ((0, COURT_L), (COURT_W, COURT_L)), ((0, 0), (0, COURT_L)),
            ((COURT_W, 0), (COURT_W, COURT_L)), ((SINGLES_IN, 0), (SINGLES_IN, COURT_L)),
            ((COURT_W - SINGLES_IN, 0), (COURT_W - SINGLES_IN, COURT_L)),
            ((SINGLES_IN, SERVICE_Y[0]), (COURT_W - SINGLES_IN, SERVICE_Y[0])),
            ((SINGLES_IN, SERVICE_Y[1]), (COURT_W - SINGLES_IN, SERVICE_Y[1])),
            ((COURT_W / 2, SERVICE_Y[0]), (COURT_W / 2, SERVICE_Y[1])),
            ((COURT_W / 2, 0), (COURT_W / 2, 0.3)), ((COURT_W / 2, COURT_L), (COURT_W / 2, COURT_L - 0.3))]
    for a, b in segs:
        cv2.line(arr, _pt(_top_xy(*a)), _pt(_top_xy(*b)), line, 1, cv2.LINE_AA, shift=2)
    cv2.line(arr, _pt(_top_xy(-0.9, NET_Y)), _pt(_top_xy(COURT_W + 0.9, NET_Y)), net, 2, cv2.LINE_AA, shift=2)
    # front view: ground, sidelines, net
    cv2.line(arr, _pt(_front_xy(X_RANGE[0], 0)), _pt(_front_xy(X_RANGE[1], 0)), line, 1, cv2.LINE_AA, shift=2)
    for x in (0, SINGLES_IN, COURT_W - SINGLES_IN, COURT_W):
        cv2.line(arr, _pt(_front_xy(x, 0)), _pt(_front_xy(x, -0.25)), line, 1, cv2.LINE_AA, shift=2)
    net_poly = [_front_xy(-0.9, 0), _front_xy(-0.9, 1.07), _front_xy(COURT_W / 2, NET_H),
                _front_xy(COURT_W + 0.9, 1.07), _front_xy(COURT_W + 0.9, 0)]
    cv2.fillPoly(arr, [np.array([_pt(p) for p in net_poly], np.int32)], _bgr(rs.SURFACE_3), cv2.LINE_AA, shift=2)
    cv2.polylines(arr, [np.array([_pt(p) for p in net_poly[1:4]], np.int32)], False, net, 2, cv2.LINE_AA, shift=2)
    # side view: ground with baselines/service lines, net
    cv2.line(arr, _pt(_side_xy(Y_RANGE[0], 0)), _pt(_side_xy(Y_RANGE[1], 0)), line, 1, cv2.LINE_AA, shift=2)
    for y in (0, SERVICE_Y[0], SERVICE_Y[1], COURT_L):
        cv2.line(arr, _pt(_side_xy(y, 0)), _pt(_side_xy(y, -0.25)), line, 1, cv2.LINE_AA, shift=2)
    cv2.line(arr, _pt(_side_xy(NET_Y, 0)), _pt(_side_xy(NET_Y, NET_H)), net, 3, cv2.LINE_AA, shift=2)
    for z in (1, 2, 3, 4, 5):
        y0 = _side_xy(Y_RANGE[0], z)[1]
        cv2.line(arr, (SIDE["x"] * 4, int(y0 * 4)), ((SIDE["x"] + 6) * 4, int(y0 * 4)), line, 1, cv2.LINE_AA, shift=2)
    img = Image.fromarray(arr[:, :, ::-1])
    d = ImageDraw.Draw(img)
    tiny = fonts.get("Medium", 13)
    for z in (1, 3, 5):
        d.text((SIDE["x"] + 10, _side_xy(Y_RANGE[0], z)[1] - 8), f"{z}m", font=tiny, fill=rs.TEXT_3)
    for name, y in (("베이스라인", 0), ("네트", NET_Y), ("베이스라인", COURT_L)):
        x = _side_xy(y, 0)[0]
        d.text((x - d.textlength(name, font=tiny) / 2, _side_xy(y, 0)[1] + 8), name, font=tiny, fill=rs.TEXT_3)
    return np.array(img)[:, :, ::-1].copy()


def draw_dynamic(arr, path: PointPath | None, t: float, shot_start: float | None, history: list[dict],
                 hitter_team: dict | None = None):
    """Draw one frame's moving parts onto `arr` (BGR, the background copy).

    path: the current point's path; t: source time; shot_start: time of the latest hit (its path is
    the accent one); history: landings of earlier points (faint dots in the top view)."""
    import cv2
    faint = _bgr(rs.LINE_2)
    for b in history:
        cv2.circle(arr, _pt(_top_xy(b["x"], b["y"])), 3 * 4, faint, -1, cv2.LINE_AA, shift=2)
    if path is None:
        return arr
    done = path.upto(t)
    acc, dim = _bgr(rs.ACCENT), _bgr(_mix(rs.TEXT_3, rs.BG, 0.15))

    def poly(samples, pick, to_xy, color, width):
        """Draw each unbroken run of samples; a gap in time (the ball unseen) leaves a gap on screen."""
        runs, cur = [], []
        for q in samples:
            if cur and q[0] - cur[-1][0] > 3 * STEP:
                runs.append(cur)
                cur = []
            cur.append(q)
        runs.append(cur)
        lines = [np.array([_pt(to_xy(*pick(q))) for q in r], np.int32) for r in runs if len(r) >= 2]
        if lines:
            cv2.polylines(arr, lines, False, color, width, cv2.LINE_AA, shift=2)

    cut = shot_start if shot_start is not None else -1e9
    old = [q for q in done if q[0] < cut]
    new = [q for q in done if q[0] >= cut - STEP]
    for pts, color, width in ((old, dim, 1), (new, acc, 2)):
        poly(pts, lambda q: (q[1], q[2]), _top_xy, color, width)
        poly(pts, lambda q: (q[1], q[3]), _front_xy, color, width)
        poly(pts, lambda q: (q[2], q[3]), _side_xy, color, width)
    # landings so far: filled = in, ring = out
    for b in path.landings:
        if b["t"] > t:
            continue
        color = _bgr(rs.TEXT) if b["in"] else _bgr(rs.BAD)
        latest = b["t"] >= cut
        r = 6 if latest else 4
        for xy in (_top_xy(b["x"], b["y"]),):
            if b["in"]:
                cv2.circle(arr, _pt(xy), r * 4, _bgr(rs.ACCENT) if latest else color, -1, cv2.LINE_AA, shift=2)
            else:
                cv2.circle(arr, _pt(xy), r * 4, color, 2, cv2.LINE_AA, shift=2)
        sx = _side_xy(b["y"], 0)
        cv2.circle(arr, _pt(sx), 4 * 4, color if not latest else _bgr(rs.ACCENT), -1, cv2.LINE_AA, shift=2)
        fx = _front_xy(b["x"], 0)
        cv2.circle(arr, _pt(fx), 4 * 4, color if not latest else _bgr(rs.ACCENT), -1, cv2.LINE_AA, shift=2)
    # hit spots: small rings in the hitter's team colour
    for h in path.hits:
        if h["t"] > t:
            continue
        team = (hitter_team or {}).get(h["side"])
        color = _bgr(rs.TEAM.get(team, rs.TEXT_2))
        cv2.circle(arr, _pt(_top_xy(h["x"], h["y"])), 5 * 4, color, 2, cv2.LINE_AA, shift=2)
    # the ball now
    pos = path.at(t)
    if pos is not None and done:
        x, y, z = pos
        for xy in (_top_xy(x, y), _front_xy(x, z), _side_xy(y, z)):
            cv2.circle(arr, _pt(xy), 8 * 4, _bgr(rs.BG), -1, cv2.LINE_AA, shift=2)
            cv2.circle(arr, _pt(xy), 6 * 4, acc, -1, cv2.LINE_AA, shift=2)
        # its shadow on the ground, so height reads in the top view too
        cv2.circle(arr, _pt(_side_xy(y, 0)), 3 * 4, dim, -1, cv2.LINE_AA, shift=2)
    return arr
