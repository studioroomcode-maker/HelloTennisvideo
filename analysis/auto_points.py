"""Find the points of a match on their own: the ball tracked over the whole video, serve-like
flights (ServeMeter) and rally hits (hit_detect) read in sequence like an umpire would.

A point opens with a serve after a quiet spell, keeps going while the ball is hit back and forth,
and ends when it goes quiet again. Many things look like a serve to the flight detector: a far
player's deep rally shot, a ball fed back to the server between points. They are told apart by
where they fall in the sequence (no serve after a serve went in; a fault is always followed by a
second serve from the same end) and by the contact height a real serve has.
"""
from __future__ import annotations

SCAN_S = 7.5           # s; serve search step (each window is tracked with 1.5 s before, 1.5 s after)
HITS_S = 20.0          # s; rally-hit search window
GAP_END = 3.3          # s without a hit after which a rally is over (balls fed back come later)
SECOND_MIN_S = 2.0     # s; a 2nd serve comes at least this long after the 1st
SECOND_MAX_S = 12.0    # s; ... and at most this long
ORPHAN_HITS = 3        # hits back and forth with no serve seen that still make a point (serve missed)
ORPHAN_JOIN_S = 6.0    # s; such a run this close after a point is the rest of that point
SERVE_H = (1.95, 3.05)  # m; contact height of a real serve (a feed or a smash is lower or higher)
SERVE_X = (0.3, 10.67)  # m across the court; a server stands between the doubles sidelines
LEAD_S = 1.0           # s before the serve contact a point starts
TAIL_S = 1.5           # s after the last hit a point ends
FAULTS = ("out", "net")


def scan(meter, duration: float, progress=None, cancelled=None) -> tuple[list[dict], list[dict]]:
    """Serve-like flights and rally hits over the whole video (the meter's ball blobs must cover
    it, see ServeMeter.prefetch). Hits are found without serve anchors: which flights are serves
    is only decided in segment()."""
    from analysis.hit_detect import find_rally_hits
    serves: list[dict] = []
    a = 0.0
    while a < duration:
        if cancelled and cancelled():
            from analysis.shot_analysis import Cancelled
            raise Cancelled()
        tracks = meter.tracks(max(0.0, a - 1.5), min(duration, a + SCAN_S + 1.5))
        for s in meter._flights(tracks, a, a + SCAN_S):
            if not any(abs(s["contact_t"] - x["contact_t"]) < 0.3 for x in serves):
                s.pop("_flight", None)
                serves.append(s)
        a += SCAN_S
        if progress:
            progress(0.3 * min(1.0, a / duration))
    hits: list[dict] = []
    a = 0.0
    while a < duration:
        if cancelled and cancelled():
            from analysis.shot_analysis import Cancelled
            raise Cancelled()
        for h in find_rally_hits(meter, meter.src, a, min(duration, a + HITS_S)):
            if a <= h["t"] < a + HITS_S and not any(abs(h["t"] - x["t"]) < 0.2 for x in hits):
                hits.append({"t": h["t"], "side": h["side"]})
        a += HITS_S
        if progress:
            progress(0.3 + 0.7 * min(1.0, a / duration))
    serves.sort(key=lambda s: s["contact_t"])
    hits.sort(key=lambda h: h["t"])
    return serves, hits


def plausible_serve(s: dict) -> bool:
    h = s.get("contact_height_m")
    x = (s.get("server_court") or [None])[0]
    if h is not None and not (SERVE_H[0] <= h <= SERVE_H[1]):
        return False
    if x is not None and not (SERVE_X[0] <= x <= SERVE_X[1]):
        return False
    return True


def segment(serves: list[dict], hits: list[dict], duration: float) -> list[dict]:
    """[{"start", "end", "serves": [contact_t...], "server_end", "hits": n, "how"}] in time order."""
    cands = [s for s in serves if plausible_serve(s)]
    # a serve's own racket hit shows up as a hit too
    rally = [h for h in hits if not any(abs(h["t"] - s["contact_t"]) < 0.35 for s in serves)]
    events = sorted([(s["contact_t"], "serve", s) for s in cands] + [(h["t"], "hit", h) for h in rally],
                    key=lambda e: e[0])

    points: list[dict] = []
    cur = None       # the open point
    orphan: list = []  # hits with no serve before them

    def close():
        nonlocal cur
        if cur is not None:
            points.append(cur)
        cur = None

    def flush_orphan():
        nonlocal orphan
        sides = [h["side"] for _, h in orphan]
        switches = sum(1 for a, b in zip(sides, sides[1:]) if a != b)
        if len(orphan) >= ORPHAN_HITS and switches >= ORPHAN_HITS - 1:
            prev = points[-1] if points else None
            if prev is not None and orphan[0][0] - prev["last"] <= ORPHAN_JOIN_S:
                prev["last"] = orphan[-1][0]
                prev["hits"] += len(orphan)
            else:
                points.append({"first": orphan[0][0], "last": orphan[-1][0], "serves": [], "server_end": None,
                               "hits": len(orphan), "phase": "rally", "how": "rally"})
        orphan = []

    for t, kind, e in events:
        if cur is not None:
            waiting = cur["phase"] == "fault"
            alive = t - cur["last"] <= (SECOND_MAX_S if waiting else GAP_END)
            if kind == "serve" and waiting and alive and e["server_end"] == cur["server_end"] \
                    and t - cur["serves"][-1] >= SECOND_MIN_S:
                cur["serves"].append(t)
                cur["last"] = t
                cur["phase"] = "done" if e["call"] in FAULTS else "rally"  # double fault ends it
                continue
            if alive and cur["phase"] == "rally":
                cur["last"] = t  # a "serve" in the middle of a rally is a deep rally shot
                cur["hits"] += 1
                continue
            if alive and waiting and kind == "hit":
                # the fault hit back, or the 2nd serve's rally when the 2nd serve itself was missed;
                # a hit from the server's own end a while after the fault is that missed 2nd serve
                cur["fault_hits"].append(t)
                late = t - cur["serves"][-1] >= SECOND_MIN_S + 1.0
                if late and e["side"] == cur["server_end"] or \
                        late and len(cur["fault_hits"]) >= 2 and cur["fault_hits"][-1] - cur["fault_hits"][-2] <= GAP_END:
                    cur["phase"] = "rally"
                    cur["last"] = t
                    cur["hits"] += len(cur["fault_hits"])
                continue
            close()
        if kind == "serve":
            if orphan:
                flush_orphan()
            cur = {"first": t, "last": t, "serves": [t], "server_end": e["server_end"], "hits": 0,
                   "phase": "fault" if e["call"] in FAULTS or e["call"] == "close" else "rally",
                   "call": e["call"], "fault_hits": [], "how": "serve"}
        else:
            if orphan and t - orphan[-1][0] > GAP_END:
                flush_orphan()
            orphan.append((t, e))
    close()
    if orphan:
        flush_orphan()

    # a "serve" from the other end right after a point is that point's rally (a deep shot from the
    # far side), the server's end only changes between games
    merged: list[dict] = []
    for p in sorted(points, key=lambda p: p["first"]):
        prev = merged[-1] if merged else None
        if prev is not None and p["server_end"] and prev["server_end"] and p["server_end"] != prev["server_end"]                 and p["first"] - prev["last"] <= ORPHAN_JOIN_S and prev["phase"] == "rally":
            prev["last"] = max(prev["last"], p["last"], *p.get("fault_hits", []))
            prev["hits"] += p["hits"] + 1 + len(p.get("fault_hits", []))
            continue
        merged.append(p)

    out = []
    for p in merged:
        if p.get("phase") == "fault" and p["call"] in FAULTS:
            continue  # a fault with no 2nd serve and no rally after it was not a serve
        if p.get("phase") == "fault" and not p["fault_hits"]:
            p["phase"] = "rally"  # "close" with nothing after: an ace, or a call we can't make
        last = p["last"] if p["phase"] != "fault" else max([p["last"]] + p["fault_hits"])
        out.append({"start": round(max(0.0, p["first"] - LEAD_S), 2),
                    "end": round(min(duration, last + TAIL_S), 2),
                    "serves": p["serves"], "server_end": p["server_end"], "hits": p["hits"], "how": p["how"]})
    # neighbouring points can't overlap
    for a, b in zip(out, out[1:]):
        if a["end"] > b["start"]:
            a["end"] = round(b["start"] - 0.05, 2)
    return out
