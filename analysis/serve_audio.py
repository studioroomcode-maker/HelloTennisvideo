"""Put far-side serve contacts on the racket sound.

Why: for a serve from the far baseline the ball first flies almost straight at the
camera, so in the image it barely moves and the tracker only picks it up 75-230 ms
after the racket hit (checked frame by frame on the sample match). That made the
flight look shorter and the serve 6-29 km/h faster than it was.

The hit is loud, though. Two delays sit between hit and onset in the soundtrack:
  * sound travel, distance(contact, camera) / 343 m/s (~85 ms from the far baseline);
  * the recording's own audio/video offset (~90 ms on this phone), which is measured
    per video from the near-side serves, whose toss-based contact times are reliable.
The far contact is then the strongest onset in a window before the tracked flight start.
"""
from __future__ import annotations

import statistics

import numpy as np

from analysis.audio_onset import read_audio, onset_envelope, strongest_onset, SPEED_OF_SOUND

SEARCH_BEFORE = 0.40   # s before the tracked flight start where the real contact can be
MIN_PROMINENCE_OFFSET = 4.0
MIN_PROMINENCE_FAR = 3.0
CONTACT_H = 2.4


def _delay(cam, s) -> float:
    sx, sy = s["server_court"]
    h = s.get("contact_height_m") or CONTACT_H
    h = h if 1.5 <= h <= 3.5 else CONTACT_H
    return float(np.linalg.norm(np.array([sx, sy, h]) - cam.C)) / SPEED_OF_SOUND


def av_offset(serves, src, cam):
    """Median (onset - sound delay - contact) over serves whose contact is toss-based."""
    res = []
    for s in serves:
        if s.get("contact_src") != "toss":
            continue
        tc, dl = s["contact_t"], _delay(cam, s)
        t0 = tc - 0.3
        env = onset_envelope(read_audio(src, t0, tc + 0.6))
        o = strongest_onset(env, t0, tc + dl - 0.10, tc + dl + 0.25)
        if o and o[1] >= MIN_PROMINENCE_OFFSET:
            res.append(o[0] - dl - tc)
    if len(res) < 2:
        return None, res
    med = statistics.median(res)
    mad = statistics.median(abs(r - med) for r in res)
    return (med if mad <= 0.025 else None), res


def refine_far_contacts(serves, src, cam, offset: float) -> int:
    """Move flight-start contacts onto the racket sound. Returns how many were moved."""
    from analysis.serve_speed import launch_speed
    moved = 0
    for s in serves:
        if s.get("contact_src") != "flight_start" or s.get("server_end") != "far":
            continue
        tv = s["contact_t"]
        dl = _delay(cam, s)
        t0 = tv - SEARCH_BEFORE - 0.2
        env = onset_envelope(read_audio(src, t0, tv + 0.3))
        o = strongest_onset(env, t0, tv - SEARCH_BEFORE + dl + offset, tv + 0.02 + dl + offset)
        if not o or o[1] < MIN_PROMINENCE_FAR:
            continue
        tc = o[0] - dl - offset
        T = s["bounce_t"] - tc
        if not (0.3 <= T <= 1.6):
            continue
        dist = s["distance_m"]
        s.update(contact_t_vision=tv, contact_t=round(tc, 3), contact_src="audio", flight_s=round(T, 3),
                 speed_kmh=round(launch_speed(dist, T) * 3.6), avg_kmh=round(dist / T * 3.6),
                 audio_prominence=round(o[1], 1))
        s["speed_err_kmh"] = max(3, round(s["speed_kmh"] * 0.04))
        moved += 1
    return moved
