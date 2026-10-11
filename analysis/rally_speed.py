"""Rally shot speed, measured the way serves are: distance over flight time.

The ball leaves the racket at the hit's contact time (the racket sound, less its delay) from
where the hitter stands (the feet of the pose box that made the swing), and lands where the
tracked ball is seen bouncing on the other side (hit_detect.mark_landings). The launch speed
that covers that distance in that time under drag is the shot's speed, radar-style, like the
serve's.

Checked frame by frame on the sample match: the contact time and bounce came within a frame
(17 ms) on the shots looked at, and the same pieces give the serves' speeds within 2 km/h of
the hand-measured ones on 7 of 8. A shot is only measured when the ball is seen at the hitter
around contact: a hit put at the wrong moment (or on the wrong player) otherwise turns into a
confident wrong speed.
"""
from __future__ import annotations

import math

from analysis.hit_detect import contact_time
from analysis.serve_speed import IN_H, IN_W, Court, launch_speed

CONTACT_Z = {"smash": 2.5, "volley": 1.2}  # m; contact height by shot, else DEFAULT_Z
DEFAULT_Z = 1.0
AT_HITTER = 0.5        # the ball within this many box heights of the hitter's box near contact
AROUND = (-0.1, 0.2)   # s around contact the ball is looked for at the hitter
MIN_KMH, MAX_KMH = 15, 200


def ball_at_hitter(balls: list, c: float, box: list, W: int, H: int) -> bool:
    """Was the ball (640x360 px) seen at the hitter's pose box (source px) around contact c?"""
    sx, sy = IN_W / W, IN_H / H
    x0, y0, x1, y1 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
    bh = max(1.0, y1 - y0)
    for t, x, y in balls or []:
        if c + AROUND[0] <= t <= c + AROUND[1]:
            dx = max(x0 - x, 0.0, x - x1)
            dy = max(y0 - 0.6 * bh - y, 0.0, y - y1)  # the racket reaches above the head
            if math.hypot(dx, dy) <= AT_HITTER * bh:
                return True
    return False


def shot_speed(hit: dict, swing: dict | None, shot: str, court: Court, W: int, H: int) -> dict | None:
    """{"speed_kmh", "distance_m", "flight_s"} for one rally hit, or None when it cannot be measured."""
    land = hit.get("landed")
    box = (swing or {}).get("box")
    if not land or not box or not hit.get("audio"):  # without the sound the contact time is a guess
        return None
    c = contact_time(hit)
    if not ball_at_hitter(hit.get("balls"), c, box, W, H):
        return None
    fx, fy = court.to_court((box[0] + box[2]) / 2 / W, box[3] / H)
    d = math.dist((land["court"][0], land["court"][1], 0.0), (fx, fy, CONTACT_Z.get(shot, DEFAULT_Z)))
    T = land["t"] - c
    if T <= 0.1:
        return None
    kmh = launch_speed(d, T) * 3.6
    if not MIN_KMH <= kmh <= MAX_KMH:
        return None
    return {"speed_kmh": round(kmh), "distance_m": round(d, 1), "flight_s": round(T, 3)}
