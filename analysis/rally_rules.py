"""Rally-structure and physics rules applied to Gemini's shot events.

Adapted from Tennis-Vision (utils/rally_decode.py, utils/shot_physics.py): the model
labels each contact independently, so impossible sequences slip through — the same
side hitting twice, a "serve" in the middle of a rally, a winner followed by more
shots. These rules repair what the rally structure makes certain and leave the rest.

Every change is recorded on the event (`fixes`) and counted in the returned dict.
"""
from __future__ import annotations

from collections import Counter

SIDES = ("near", "far")
MID_RALLY_GAP = 2.5    # an overhead swing this soon after a rally shot is a smash, not a serve
RALLY_SPLIT_GAP = 6.0  # full-video mode: this much silence ends a rally
MIN_ALT_GAP = 0.3      # fastest plausible exchange between opposite sides (net volleys)
MIN_MISSED_GAP = 1.2   # same side twice in a row needs time for an unseen opponent shot
DUP_GAP = 0.5          # same side twice faster than this is one swing recorded twice
MISS_COST = 1.0        # cost of assuming the model missed a shot
FAST_ALT_COST = 0.8    # opposite sides but faster than MIN_ALT_GAP: unlikely, not impossible
INF = float("inf")


def _other(side):
    return {"near": "far", "far": "near"}.get(side)


def _retag(ev: dict, settings: dict):
    """Recompute team/player after the side of a non-serve event changed."""
    a_end = ev["_ctx"].get("a_end")
    ev["team"] = ev["player"] = None
    if a_end in SIDES and ev["side"] in SIDES:
        ev["team"] = "A" if ev["side"] == a_end else "B"
        if (settings.get("mode") or "doubles") == "singles":
            ev["player"] = f"{ev['team']}1"


def _fix(ev, code, counter):
    ev["fixes"].append(code)
    counter[code] += 1


def _groups(events: list[dict]) -> list[list[dict]]:
    """One group per labeled point; without labels, split the stream into rallies."""
    by_point: dict[str, list] = {}
    loose = []
    for e in events:
        (by_point.setdefault(e["point_id"], []) if e.get("point_id") else loose).append(e)
    groups = [sorted(g, key=lambda e: e["t"]) for g in by_point.values()]
    cur: list[dict] = []
    for e in sorted(loose, key=lambda e: e["t"]):
        if cur:
            gap = e["t"] - cur[-1]["t"]
            if gap >= RALLY_SPLIT_GAP or (e["shot"] == "serve" and gap >= MID_RALLY_GAP):
                groups.append(cur)
                cur = []
        cur.append(e)
    if cur:
        groups.append(cur)
    return groups


def decode_sides(rally: list[dict], receiver: str | None) -> list[str]:
    """Viterbi over near/far: consecutive shots alternate sides unless a shot was missed.

    Emission cost is the model's own side label weighted by its confidence; a repeated
    side costs MISS_COST and is only allowed when there was time for an unseen shot.
    """
    def emit(e, s):
        lab = e["side"]
        return 0.0 if lab not in SIDES or lab == s else 0.3 + e["confidence"] / 100

    cost = [{s: (0.0 if receiver in (None, s) else MISS_COST) + emit(rally[0], s) for s in SIDES}]
    back: list[dict] = [{}]
    for i in range(1, len(rally)):
        dt = rally[i]["t"] - rally[i - 1]["t"]
        row, br = {}, {}
        for s in SIDES:
            best, arg = INF, None
            for p in SIDES:
                if p != s:
                    tr = 0.0 if dt >= MIN_ALT_GAP else FAST_ALT_COST
                else:
                    tr = MISS_COST if dt >= MIN_MISSED_GAP else INF
                c = cost[-1][p] + tr
                if c < best:
                    best, arg = c, p
            row[s], br[s] = best + emit(rally[i], s), arg
        cost.append(row)
        back.append(br)
    s = min(SIDES, key=lambda k: cost[-1][k])
    path = [s]
    for i in range(len(rally) - 1, 0, -1):
        s = back[i][s]
        path.append(s)
    return path[::-1]


def apply_rally_rules(events: list[dict], settings: dict, fix_serve_numbers, warnings: list):
    fixes: Counter = Counter()
    groups = _groups(events)
    removed: set[int] = set()

    # Pass 1 — serve structure.
    for g in groups:
        for i, e in enumerate(g):
            prev = g[i - 1] if i else None
            if (e["shot"] == "serve" and prev and prev["shot"] != "serve"
                    and e["t"] - prev["t"] < MID_RALLY_GAP):
                # An overhead swing in the flow of a rally is a smash.
                e.update(shot="smash", serve_number=0, serve_type=None, serve_direction=None,
                         side=e["_side_raw"])
                _retag(e, settings)
                _fix(e, "serve_to_smash", fixes)
        serve_idx = [i for i, e in enumerate(g) if e["shot"] == "serve"]
        if not serve_idx:
            continue
        first, last = serve_idx[0], serve_idx[-1]
        # Nothing is played before the serve; between serves only the receiver tapping
        # back a fault ball happens. Neither belongs in the shot counts.
        for e in g[:first]:
            removed.add(id(e))
            fixes["before_serve"] += 1
        between = [e for e in g[first:last] if e["shot"] != "serve"]
        if 0 < len(between) <= 2 * (len(serve_idx) - 1):
            for e in between:
                removed.add(id(e))
                fixes["fault_return"] += 1

    events = [e for e in events if id(e) not in removed]
    fix_serve_numbers(events, warnings)
    groups = [[e for e in g if id(e) not in removed] for g in groups]

    # Pass 2 — rally sequence and results.
    for g in groups:
        if not g:
            continue
        serve_idx = [i for i, e in enumerate(g) if e["shot"] == "serve"]
        last_serve = g[serve_idx[-1]] if serve_idx else None
        rally = [e for e in g[(serve_idx[-1] + 1 if serve_idx else 0):] if e["shot"] != "serve"]

        # Same side twice within DUP_GAP: one swing recorded twice.
        kept: list[dict] = []
        for e in rally:
            p = kept[-1] if kept else None
            if p and e["side"] in SIDES and e["side"] == p["side"] and e["t"] - p["t"] < DUP_GAP:
                if e["confidence"] > p["confidence"]:
                    kept[-1] = e
                    removed.add(id(p))
                else:
                    removed.add(id(e))
                fixes["duplicate"] += 1
                continue
            kept.append(e)
        rally = kept

        if rally:
            receiver = _other(last_serve["side"]) if last_serve else None
            anchored = receiver is not None or any(e["side"] in SIDES for e in rally)
            for e, s in zip(rally, decode_sides(rally, receiver)):
                if e["side"] in SIDES and e["side"] != s:
                    e["side"] = s
                    _fix(e, "side_flip", fixes)
                elif e["side"] not in SIDES and anchored:
                    e["side"] = s
                    _fix(e, "side_fill", fixes)
                else:
                    continue
                _retag(e, settings)

            # The return must bounce first, so it cannot be a volley or smash.
            ret = rally[0]
            if last_serve and ret["shot"] in ("volley", "smash"):
                ret["shot"] = "other"
                _fix(ret, "return_volley", fixes)
            # Only the last shot of a rally can end the point.
            for e in rally[:-1]:
                if e["result"] in ("winner", "error"):
                    e["result"] = "in"
                    _fix(e, "result_mid_rally", fixes)
            # A returned serve was in.
            if last_serve and last_serve["result"] in ("fault", "ace"):
                last_serve["result"] = "in"
                _fix(last_serve, "serve_result", fixes)

        # The labeled point winner is ground truth for the shot that ended the point.
        end = (rally or g)[-1]
        winner = end["_ctx"].get("winner")
        if winner in ("A", "B") and end["team"] and end["result"] in ("winner", "ace", "error"):
            scored = end["team"] if end["result"] in ("winner", "ace") else ("B" if end["team"] == "A" else "A")
            if scored != winner:
                end["result"] = "unknown"
                _fix(end, "result_vs_label", fixes)

    events = [e for e in events if id(e) not in removed]
    for e in events:
        e.pop("_side_raw", None)
        e.pop("_ctx", None)
    return events, dict(fixes)
