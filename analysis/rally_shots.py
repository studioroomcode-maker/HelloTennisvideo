"""Rally shot speeds: hit -> bounce segments inside a point.

Every sharp change in the ball's image velocity is a candidate event. One that happens
at a player (the ball inside the player's body box, widened by racket reach) is a hit;
an upward kick away from players is a bounce. A hit followed by a bounce on the other
side of the net is a measurable shot: contact = camera ray through the ball pixel meeting
the vertical plane at the hitter's depth (feet on the court), bounce = homography, and
the launch speed comes from distance and flight time with the same drag model as serves.
"""
from __future__ import annotations

import math

from analysis.serve_speed import (IN_W, IN_H, COURT_W, COURT_L, NET_Y, _slope, _interp, launch_speed,
                                  bridge_tracks)

KINK_MS = 3.5
REACH = 0.55        # racket reach, as a fraction of the player's height, beyond the body box
MAX_FLIGHT = 2.2


def kinks(track, court, win=0.1):
    """All sharp velocity changes along a (bridged) track: (t, x, y, dvy_ms, dvx_ms)."""
    out = []
    idx = list(range(2, len(track) - 2))
    for i in idx:
        t = track[i][0]
        before = [p for p in track if t - win <= p[0] <= t]
        after = [p for p in track if t <= p[0] <= t + win]
        if len(before) < 3 or len(after) < 3:
            continue
        out.append(_kink_at(track, court, before, after, t))
    for i in range(1, len(track)):  # inside holes
        a, b = track[i - 1], track[i]
        if b[0] - a[0] < 0.05:
            continue
        before = [p for p in track[max(0, i - 6):i] if a[0] - p[0] <= 0.25]
        after = [p for p in track[i:i + 6] if p[0] - b[0] <= 0.25]
        if len(before) < 3 or len(after) < 3:
            continue
        b1, a1 = _slope(before)
        b2, a2 = _slope(after)
        if b1 is None or b2 is None or abs(b1 - b2) < 1e-6:
            continue
        tb = (a2 - a1) / (b1 - b2)
        if a[0] - 0.03 <= tb <= b[0] + 0.03:
            out.append(_kink_at(track, court, before, after, tb))
    out = [k for k in out if k and (abs(k[3]) >= KINK_MS or abs(k[4]) >= KINK_MS)]  # (t,x,y,dvy,dvx,vy0,vy1)
    out.sort(key=lambda k: k[0])
    merged = []
    for k in out:  # strongest of each 80 ms cluster
        if merged and k[0] - merged[-1][0] < 0.08:
            if abs(k[3]) + abs(k[4]) > abs(merged[-1][3]) + abs(merged[-1][4]):
                merged[-1] = k
        else:
            merged.append(k)
    return merged


def _kink_at(track, court, before, after, t):
    b1, a1 = _slope(before)
    b2, a2 = _slope(after)
    bx1, _ = _slope(before, 1)
    bx2, _ = _slope(after, 1)
    if None in (b1, b2, bx1, bx2):
        return None
    x, y = _interp(track, t, 1), _interp(track, t, 2)
    cx, cy = court.to_court(x / IN_W, y / IN_H)
    scale = max(4.0, court.px_per_m(min(max(cx, 0), COURT_W), min(max(cy, 0), COURT_L)))
    return (t, x, y, (b2 - b1) / scale, (bx2 - bx1) / scale, b1 / scale, b2 / scale)


def player_at(people, x, y):
    """The player whose (racket-widened) box contains the ball pixel, as (feet_court, box)."""
    best = None
    for p in people:
        pts = [(k[0] * IN_W, k[1] * IN_H) for k in p["kps"] if k[2] > 0.3]
        if len(pts) < 5:
            continue
        x0, x1 = min(q[0] for q in pts), max(q[0] for q in pts)
        y0, y1 = min(q[1] for q in pts), max(q[1] for q in pts)
        h = max(8.0, y1 - y0)
        if x0 - REACH * h <= x <= x1 + REACH * h and y0 - 0.45 * h <= y <= y1 + 0.1 * h:
            d = abs((x0 + x1) / 2 - x) + abs((y0 + y1) / 2 - y)
            if best is None or d < best[0]:
                best = (d, p["feet"], (x0, y0, x1, y1))
    return None if best is None else best[1:]


def rally_events(tracks, court, people_at):
    """Hits and bounces in time order. people_at(t) -> [{"kps": [...], "feet": (cx, cy)}]."""
    events = []
    for tr in bridge_tracks(tracks):
        if len(tr) < 8:
            continue
        for t, x, y, dvy, dvx, vy0, vy1 in kinks(tr, court):
            pl = player_at(people_at(t), x, y)
            # a hit sends the ball back: a far player turns an up-screen ball down-screen (toward the
            # camera), a near player the opposite. A ball merely flying past a player keeps its direction.
            if pl is not None:
                far = pl[0][1] >= NET_Y
                reversed_ = (vy0 < 1.0 and vy1 > 1.0) if far else (vy0 > -1.0 and vy1 < -1.0)
                if not reversed_:
                    pl = None
            if pl is not None:
                events.append({"type": "hit", "t": t, "px": (x, y), "feet": pl[0], "box": pl[1],
                               "side": "near" if pl[0][1] < NET_Y else "far"})
            elif dvy < -KINK_MS:
                bx, by = court.to_court(x / IN_W, y / IN_H)
                if -2 <= bx <= COURT_W + 2 and -3 <= by <= COURT_L + 3:
                    events.append({"type": "bounce", "t": t, "px": (x, y), "court": (bx, by)})
    events.sort(key=lambda e: e["t"])
    dedup = []
    for e in events:
        if dedup and e["type"] == dedup[-1]["type"] and e["t"] - dedup[-1]["t"] < 0.15:
            continue
        dedup.append(e)
    return dedup


def shots_from_events(events, cam):
    """Hit followed by a bounce on the other side of the net within MAX_FLIGHT seconds."""
    shots = []
    for i, h in enumerate(events):
        if h["type"] != "hit":
            continue
        nxt = next((e for e in events[i + 1:] if e["t"] - h["t"] > 0.15), None)
        if nxt is None or nxt["type"] != "bounce" or nxt["t"] - h["t"] > MAX_FLIGHT:
            continue
        bx, by = nxt["court"]
        if (h["side"] == "near") == (by < NET_Y):
            continue  # must land on the other side
        p0 = cam.on_plane_y(h["px"][0], h["px"][1], h["feet"][1])
        if p0 is None or not (0.1 <= p0[2] <= 3.5):
            continue
        horiz = math.hypot(bx - p0[0], by - p0[1])
        dist = math.hypot(horiz, float(p0[2]))
        T = nxt["t"] - h["t"]
        v0 = launch_speed(dist, T) * 3.6
        if not (20 <= v0 <= 200):
            continue
        shots.append({"hit_t": round(h["t"], 3), "bounce_t": round(nxt["t"], 3), "side": h["side"],
                      "contact_3d": [round(float(c), 2) for c in p0], "bounce_court": [round(bx, 2), round(by, 2)],
                      "flight_s": round(T, 3), "distance_m": round(dist, 2), "speed_kmh": round(v0)})
    return shots
