"""Racket-hit moments in a rally from the ball track, timed by the impact sound.

Every shot sends the ball over the net, so the rally is read from net crossings: the ball's
image path passing the row of the net tape, downward (toward the camera, the far team hit it)
or upward (the near team hit it). Two crossings the opposite way round have one hit between
them, by the team on the side the ball came into. Crossings, not the raw up/down motion, decide
the side because over the far court the image motion is mostly the ball falling and bouncing,
which looks like coming toward the camera.

The moment of the hit is where the ball's run toward the second crossing began: the start of
the unbroken stretch of image motion in that direction, back from the crossing. The tracker
often loses the ball for a few frames at the racket, so the exact time is taken from the
strongest impact sound in the hole between the incoming and the outgoing stretch.
"""
from __future__ import annotations

from analysis.serve_speed import COURT_L, COURT_W, IN_H, IN_W, NET_Y, Court, _slope

NET_TOP_M = 1.0       # m; the tape height used for the crossing row
STEP = 0.1            # s; direction is read from line fits over this much track
MIN_SPEED_MS = 2.0    # m/s of image-y motion (scaled at the ball) for a stretch to have a direction
RUN_GAP = 0.15        # s; holes up to this long don't break a run in one direction
AUDIO_PAD = 0.12      # s; impact sounds this far outside the hole still count
MIN_AUDIO = 3.0       # onset strength that counts as an impact sound
TAIL_AUDIO = 4.0      # a last near-side hit with no crossing after it (into the net) needs this loud a sound
RESTART_S = 2.5       # s; a same-way crossing this long after the last one is a new shot from that side
MIN_APART = 0.3       # s; a hit is never put this close after the previous one
MIN_TRACK_MS = 2.5    # m/s; slower tracks are a ball rolling or a shoe, not a shot
FAR_SURE_AUDIO = 3.5  # a far hit whose ball never showed up past the net's foot needs this loud a sound
DIP_S = 0.7           # s; longest dip under the tape that is still a far-court bounce
SAME_CROSS = 0.3      # s; crossings this close in the same direction are one
NEAR_DELAY, FAR_DELAY = 0.12, 0.17  # s; a hit is heard this long after it happens (A/V offset + distance)
BOUNCE_PX = 3.0       # image px (640x360) the ball must come down to and go back up from a bounce
BOUNCE_GAP = 0.15     # s; a "bounce" closer than this to the contact is the racket itself
SEEN_GAP = 0.4        # s; "no bounce" needs the ball seen this close to the contact, else it is unknown


def _on_court(court: Court, x: float, y: float) -> bool:
    """Inside this court's sidelines (with room for wide balls) at the ball's image row.

    A ball in the air maps far off the court through the ground homography, so the test is
    made in the image: the sidelines' x at the ball's row, the far end's row also standing
    for every row above it (balls high over the far court).
    """
    (lnx, lny), (rnx, rny) = court.to_image(0, 0), court.to_image(COURT_W, 0)
    (lfx, lfy), (rfx, rfy) = court.to_image(0, COURT_L), court.to_image(COURT_W, COURT_L)
    yn, yf = (lny + rny) / 2 * IN_H, (lfy + rfy) / 2 * IN_H
    if y > yn + 0.15 * IN_H:
        return False
    f = min(1.0, max(0.0, (y - yf) / max(1.0, yn - yf)))
    left = (lfx + (lnx - lfx) * f) * IN_W
    right = (rfx + (rnx - rfx) * f) * IN_W
    pad = 0.25 * (right - left)
    return left - pad <= x <= right + pad


def own_court(track, court: Court) -> bool:
    """A moving ball of this court's game, not one on the next court or behind the fence."""
    return (len(track) >= 6 and _speed(track, court) >= MIN_TRACK_MS
            and sum(_on_court(court, p[1], p[2]) for p in track) >= 0.6 * len(track))


def _speed(track, court: Court) -> float:
    """Mean image speed of the track in m/s, scaled at its middle point."""
    import math
    path = sum(math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(track, track[1:]))
    mid = track[len(track) // 2]
    return path / max(1e-3, track[-1][0] - track[0][0]) / _scale(court, mid[1], mid[2])


def _scale(court: Court, x: float, y: float) -> float:
    cx, cy = court.to_court(x / IN_W, y / IN_H)
    return max(4.0, court.px_per_m(min(max(cx, 0), COURT_W), min(max(cy, 0), COURT_L)))


def net_rows(court: Court):
    """(x px) -> (tape row, ground row) of the net in the image, straight lines between the posts."""
    xa, ya = court.to_image(-0.9, NET_Y)
    xb, yb = court.to_image(COURT_W + 0.9, NET_Y)
    lift_a = NET_TOP_M * court.px_per_m(0, NET_Y) * (IN_W / 640)
    lift_b = NET_TOP_M * court.px_per_m(COURT_W, NET_Y) * (IN_W / 640)
    x0, x1 = xa * IN_W, xb * IN_W
    g0, g1 = ya * IN_H, yb * IN_H

    def rows(x):
        f = (x - x0) / (x1 - x0 or 1.0)
        g = g0 + (g1 - g0) * f
        return g - (lift_a + (lift_b - lift_a) * f), g
    return rows


def pieces(track, court: Court) -> list[tuple]:
    """(t0, t1, dir) for each STEP of the track; dir +1 = image y growing (toward the camera)."""
    out = []
    k, t1 = track[0][0], track[-1][0]
    while k < t1:
        seg = [p for p in track if k <= p[0] < k + STEP]
        if len(seg) >= 3:
            v, _ = _slope(seg)
            if v is not None:
                s = v / _scale(court, seg[len(seg) // 2][1], seg[len(seg) // 2][2])
                d = 0 if abs(s) < MIN_SPEED_MS else (1 if s > 0 else -1)
                out.append((seg[0][0], seg[-1][0], d))
        k += STEP
    return out


def _zone(rows, x, y):
    """'far' above the net tape, 'near' below the net's foot, None in the band between.

    The band matters: with the camera this low the whole far service box lies below the
    tape in the image, so a ball bouncing there dips under the tape without coming over."""
    tape, ground = rows(x)
    return "far" if y < tape else ("near" if y > ground else None)


def crossings(track, rows) -> list[tuple]:
    """(t, dir) where the track passes the net-tape row; +1 = downward in the image (toward the camera)."""
    out = []
    for p, q in zip(track, track[1:]):
        da, db = p[2] - rows(p[1])[0], q[2] - rows(q[1])[0]
        if da < 0 <= db or db < 0 <= da:
            f = da / (da - db) if da != db else 0.5
            out.append((p[0] + f * (q[0] - p[0]), 1 if db >= 0 else -1))
    return out


def _dips(cross, tracks, rows, max_s: float = DIP_S) -> list[tuple]:
    """Drop down-then-up pairs where the ball never got past the net's foot in between.

    With the camera this low the far service box lies below the tape in the image, so a
    ball bouncing there dips under the tape and comes back without having come over."""
    out = list(cross)
    k = 0
    while k + 1 < len(out):
        (t1, d1), (t2, d2) = out[k], out[k + 1]
        if d1 > 0 and d2 < 0 and t2 - t1 <= max_s:
            pts = [p for tr in tracks for p in tr if t1 <= p[0] <= t2]
            if not any(p[2] > rows(p[1])[1] for p in pts):
                del out[k:k + 2]
                k = max(0, k - 1)
                continue
        k += 1
    return out


def _end_zone(track, rows, from_end: bool):
    """Which side of the tape the track ends (or starts) on."""
    t, x, y = track[-1] if from_end else track[0]
    return ("far" if y < rows(x)[0] else "near"), t, x


def gap_crossings(tracks, rows, max_gap: float = 0.35, max_dx: float = 60.0) -> list[tuple]:
    """Crossings hidden in the hole between a track that ends in one zone and one that starts in the other."""
    out = []
    for a in tracks:
        za, ta, xa = _end_zone(a, rows, True)
        if za is None:
            continue
        for b in tracks:
            if b is a:
                continue
            zb, tb, xb = _end_zone(b, rows, False)
            if zb is None or zb == za or not (0 < tb - ta <= max_gap) or abs(xb - xa) > max_dx:
                continue
            out.append(((ta + tb) / 2, 1 if zb == "near" else -1))
    return out


def detect_hits(tracks, court: Court, onsets=None, t_from=None, t_to=None, serves=None) -> list[dict]:
    """tracks: [[(t, x, y), ...]] at 640x360; onsets: [{"t", "strength"}] from audio_onset.hit_candidates;
    serves: [{"t", "side", "fault"}] measured serve contacts (ServeMeter), which open a rally;
    after a fault nothing counts until the next serve.

    Returns [{"t", "side": "near"|"far", "audio", "hole": [t_a, t_b], "first", "serve"}] in time
    order; side is the team that hit the ball. A near server's toss rises past the net row in the
    image, so crossings in the 1.5 s before a measured serve are dropped.
    """
    rows = net_rows(court)
    tracks = [tr for tr in tracks if own_court(tr, court)]
    lo_t = t_from if t_from is not None else -1e9
    hi_t = t_to if t_to is not None else 1e9
    serves = [s for s in (serves or []) if lo_t - 1.0 <= s["t"] <= hi_t]
    cross = sorted(c for c in [c for tr in tracks for c in crossings(tr, rows)] + gap_crossings(tracks, rows)
                   if lo_t <= c[0] <= hi_t and not any(s["t"] - 1.5 <= c[0] <= s["t"] + 0.05 for s in serves))
    cross = _dips(cross, tracks, rows)
    faults = [(s["t"], min([u["t"] for u in serves if u["t"] > s["t"]] or [s["t"] + 4.0]))
              for s in serves if s.get("fault")]
    cross = [c for c in cross if not any(f0 < c[0] < f1 for f0, f1 in faults)]
    merged: list[tuple] = []
    for c in cross:
        if merged and merged[-1][1] == c[1] and c[0] - merged[-1][0] < SAME_CROSS:
            continue
        merged.append(c)
    events = sorted([(c[0], c[1], False) for c in merged] +
                    [(s["t"], -1 if s["side"] == "near" else 1, True) for s in serves])
    runs = sorted((p for tr in tracks for p in pieces(tr, court)), key=lambda p: p[0])
    sounds = [o for o in (onsets or []) if o["strength"] >= MIN_AUDIO]

    def hole_before(tc, d, floor):
        """Start of the unbroken run in direction d that reaches tc, and the end of the motion before it."""
        mine = [p for p in runs if p[2] == d and p[0] <= tc and p[0] >= floor]
        if not mine:
            return max(floor, tc - 0.5), tc
        start = None
        for p in reversed(mine):
            if p[0] > tc:
                continue
            if start is None:
                if tc - p[1] > RUN_GAP + STEP:
                    break
                start = p[0]
            elif start - p[1] <= RUN_GAP:
                start = p[0]
            else:
                break
        if start is None:
            return max(floor, tc - 0.5), tc
        other = [p for p in runs if p[2] == -d and floor <= p[1] <= start + 0.02]
        before = max((p[1] for p in other), default=max(floor, start - 0.3))
        return min(before, start), start

    def reached_near(tc):
        return any(tc <= p[0] <= tc + 0.8 and p[2] > rows(p[1])[1] for tr in tracks for p in tr)

    hits: list[dict] = []
    last = None  # (t, dir) of the shot the ball is currently on
    for tc, d, is_serve in events:
        side = "far" if d > 0 else "near"
        if is_serve:
            hits.append({"t": round(tc, 3), "side": side, "audio": None, "hole": None, "cross_t": None,
                         "first": True, "serve": True})
            last = (tc, d)
            continue
        if any(f0 < tc < f1 for f0, f1 in faults):
            continue
        if last and last[1] == d and tc - last[0] < RESTART_S:
            continue  # the same ball crossing again (or the serve's own flight)
        restart = last is None or last[1] == d
        a, b = hole_before(tc, d, max(lo_t, tc - 1.5) if restart else last[0])
        if d > 0 and not restart:
            # Over the far court the image barely shows the turn: a ball hit up rises on screen
            # just like one still going away, so the run toward the camera may start only at the
            # top of the arc. Any time since the ball went over is possible; the sound decides.
            a = min(a, last[0] + 0.2)
        after = hits[-1]["t"] + MIN_APART if hits else -1e9
        near = [o for o in sounds if max(after, a - AUDIO_PAD) <= o["t"] <= b + AUDIO_PAD]
        if near:
            o = max(near, key=lambda o: o["strength"] / (1 + abs(o["t"] - b) / 0.5))
            t, s = o["t"], o["strength"]
        else:
            t, s = max(b, after), None
        if d > 0 and (s or 0) < FAR_SURE_AUDIO and not reached_near(tc):
            continue  # dipped under the tape and was lost there (a far-court bounce), not hit back
        hits.append({"t": round(t, 3), "side": side, "audio": s, "hole": [round(a, 2), round(b, 2)],
                     "cross_t": round(tc, 2), "first": restart, "serve": False})
        last = (tc, d)
    if last and last[1] > 0:
        back = [p for p in runs if p[2] < 0 and last[0] - 0.2 <= p[0] <= hi_t]
        for p in back:
            near = [o for o in sounds if p[0] - 0.3 <= o["t"] <= p[0] + 0.15 and o["strength"] >= TAIL_AUDIO]
            if near:
                o = max(near, key=lambda o: o["strength"])
                hits.append({"t": round(o["t"], 3), "side": "near", "audio": o["strength"],
                             "hole": [round(p[0] - 0.3, 2), round(p[0], 2)], "cross_t": None, "first": False,
                             "serve": False, "tail": True})
                break
    return hits


def contact_time(hit: dict) -> float:
    """When the racket met the ball: the hit time is the sound, which comes a little later."""
    if not hit.get("audio"):
        return hit["t"]
    return hit["t"] - (NEAR_DELAY if hit["side"] == "near" else FAR_DELAY)


def bounce_before(tracks, court: Court, t0: float, t1: float):
    """Time of the last bounce in (t0, t1 - BOUNCE_GAP), or None.

    Whichever way the ball travels, a bounce is the lowest point of its image path: it comes
    down to the court and goes back up, so the image y peaks there. A ball volleyed out of the
    air keeps moving one way (down the image towards a near player, up it towards a far one).
    """
    best = None
    for tr in tracks:
        if len(tr) < 6 or _speed(tr, court) < MIN_TRACK_MS:
            continue
        ours = own_court(tr, court)  # else the bounce itself must be on this court (wide balls leave the band)
        pts = [p for p in tr if t0 <= p[0] <= t1]
        for i, (t, x, y) in enumerate(pts):
            if t > t1 - BOUNCE_GAP:
                break
            before = [p[2] for p in pts[:i] if p[0] >= t - 0.3]
            after = [p[2] for p in pts[i + 1:] if p[0] <= t + 0.3]
            if (before and after and y >= max(before + after) and y - min(before) >= BOUNCE_PX
                    and y - min(after) >= BOUNCE_PX and (ours or _on_court(court, x, y))
                    and (best is None or t > best)):
                best = t
    return best


def mark_bounces(hits: list[dict], tracks, court: Court):
    """Set "bounced" on every rally hit: did the ball bounce on the hitter's side first?"""
    for k, h in enumerate(hits):
        if h.get("serve") or k == 0:
            continue
        prev = hits[k - 1]
        # a rally opener straight after a serve marker (or opening the point) is a serve the meter missed
        prev_serve = prev.get("serve") or (prev.get("first") and (k < 2 or hits[k - 2].get("serve")))
        if prev_serve:
            h["bounced"] = True  # a serve has to bounce before it may be returned
            continue
        t0, t1 = prev["t"] + BOUNCE_GAP, contact_time(h)
        b = bounce_before(tracks, court, t0, t1)
        if b is not None:
            h["bounced"], h["bounce_t"] = True, round(b, 3)
            continue
        seen = [p[0] for tr in tracks if own_court(tr, court) for p in tr if t0 <= p[0] <= t1]
        # a ball lost on its way down may have bounced unseen
        h["bounced"] = False if seen and max(seen) >= t1 - SEEN_GAP else None


def find_rally_hits(meter, src: str, start: float, end: float, serves=None) -> list[dict]:
    """Track the ball over one point (in 8 s slices so memory stays bounded) and detect its hits.

    meter: ServeMeter (its detector and court); serves: [{"t", "side", "fault"}] in or near the point.
    """
    from analysis.audio_onset import hit_candidates
    t0, t1 = start - 1.5, end + 1.0
    tracks, s = [], t0
    while s < t1:
        e = min(t1, s + 8.0)
        tracks += meter.tracks(s, e + 0.3)
        s = e
    onsets = hit_candidates(src, [(t0, t1)], min_ratio=2.0)
    hits = detect_hits(tracks, meter.court, onsets, start - 1.0, end + 0.5, serves)
    own = [tr for tr in tracks if own_court(tr, meter.court)]
    from analysis.rally3d import arc_hits, merge
    try:  # lobs and smashes that never cross the tape row in the image
        hits = merge(hits, arc_hits(own, meter.court, onsets), start - 1.0, end + 0.5, serves)
    except Exception:
        pass
    mark_bounces(hits, tracks, meter.court)
    for h in hits:  # the ball around the hit, so the hitter can be told from the other players later
        pts = sorted(p for tr in own for p in tr if h["t"] - 0.6 <= p[0] <= h["t"] + 0.4)
        h["balls"] = [[round(p[0], 3), round(p[1], 1), round(p[2], 1)] for p in pts[::2]]
    return hits
