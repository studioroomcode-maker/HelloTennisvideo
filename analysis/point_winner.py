"""Who won each point, from what the ball did, checked against the rules of scoring.

Each point gets a chance that the near end won it, from its last shot:
- two serve faults in a row and nothing after: a double fault, the server lost it
- a rally: the last hitter's ball, followed in 3D (rally3d arcs) to where it lands. Short of the
  net or below the tape, or landing outside the doubles court: the hitter lost it. In: the
  hitter won it (nobody got it back). No flight seen: the hitter most likely missed, as most
  club points end on an error.
- a serve in with no return: the server won it.

These are then read as a sequence through the score (core.scoring). A game ends on its last point
and the next game is served by the other team, so where the serve is seen to change ends tells
which points closed a game; the most likely winners consistent with that are chosen. Winners the
user has set stay as they are.
"""
from __future__ import annotations

import copy
import math

import numpy as np

COURT_W = 10.97        # m, doubles width
SINGLES_X = (1.37, 9.60)  # m, the singles sidelines
CLOSE_M = 0.25         # m from a line: too close to call from the fit (as for serves)
BOUNCE_TO_HIT_S = 0.15  # s; a ball hit sooner than this after its landing time was taken in the air
COURT_L = 23.77        # m
NET_Y = COURT_L / 2
NET_H = 0.85           # m; a flight crossing lower went into the tape (0.914 in the middle; the fit is ±0.1)
LINE_SLACK = 0.05      # m; the fit's landing point this far out still counts as on the line
CLEAR_OUT = 0.3        # m out for a ball the other side still hit back to end the point (club players
                       # often play a ball that is out; a fit this far off is rare)
ARC_BEFORE, ARC_AFTER = 0.35, 0.6   # s around a hit for the flight it started
MIN_VY = 3.0           # m/s along the court
SERVE_HIT_S = 0.35     # s; a hit this close to a serve is the serve itself
FAULTS = ("out", "net")

P_DOUBLE_FAULT = 0.9
P_NET = 0.85           # the hitter lost it
P_OUT = 0.8
P_IN = 0.7             # the hitter won it
P_UNSEEN = 0.65        # the hitter lost it
P_ACE = 0.75
P_SERVE_SEEN = 0.95    # the serve's end matches the score's server
BEAM = 256


def landing(arc: dict) -> dict:
    """Where an arc's flight first touches the ground, and how high it crossed the net."""
    from analysis.rally3d import fly
    rel = np.arange(0.0, 4.0, 0.005)
    xs = fly(np.array(arc["p"], float), np.array(arc["v"], float), rel)
    below = np.where(xs[:, 2] <= 0)[0]
    k = int(below[0]) if len(below) else len(rel) - 1
    side = np.sign(xs[:k + 1, 1] - NET_Y)
    cross = np.where(np.diff(side) != 0)[0]
    return {"x": float(xs[k, 0]), "y": float(xs[k, 1]), "t": float(arc["t0"] + rel[k]),
            "net_z": float(xs[cross[0], 2]) if len(cross) else None}


def shot_flight(hit: dict, arcs: list[dict]) -> dict | None:
    """The flight a hit started: the first arc right after it heading away from the hitter, run on
    to the ground; else where its ball was seen bouncing beyond the net (hit_detect's "landed").
    The arc goes first: a bounce is only tied to the hit before it, and when the other side's
    next hit was missed (a quick volley at the net) it belongs to that one instead."""
    away = 1.0 if hit["side"] == "near" else -1.0
    for a in sorted(arcs, key=lambda a: a["t0"]):
        if hit["t"] - ARC_BEFORE <= a["t0"] <= hit["t"] + ARC_AFTER and a["v"][1] * away >= MIN_VY:
            return landing(a)
    land = hit.get("landed")
    if land:
        return {"x": land["court"][0], "y": land["court"][1], "t": land["t"], "net_z": None}
    return None


def out_by(f: dict, singles: bool = False) -> float:
    """How far (m) outside the court the ball landed, 0 when in."""
    x0, x1 = SINGLES_X if singles else (0.0, COURT_W)
    return max(0.0, x0 - f["x"], f["x"] - x1, -f["y"], f["y"] - COURT_L)


def inside_by(f: dict, singles: bool = False) -> float:
    """How far (m) inside the nearest line of the half it landed in, 0 when out."""
    x0, x1 = SINGLES_X if singles else (0.0, COURT_W)
    end = f["y"] if f["y"] < NET_Y else COURT_L - f["y"]
    return max(0.0, min(f["x"] - x0, x1 - f["x"], end))


def judge_flight(side: str, f: dict, singles: bool = False) -> tuple[str, str]:
    """('won' | 'lost', why) for the hitter at `side` whose ball flew as f."""
    far_half = f["y"] > NET_Y
    # no crossing in the fitted stretch but landing over the net: it crossed before the arc began
    if far_half != (side == "near") or (f["net_z"] is not None and f["net_z"] < NET_H):
        return "lost", "net"
    return ("lost", "out") if out_by(f, singles) > LINE_SLACK else ("won", "in")


def call_shot(hit: dict, nxt: dict | None, arcs: list[dict], singles: bool = False) -> dict | None:
    """The line call on one rally shot: {"call": in | out | net | close, "x", "y", "t"} where the
    ball landed, {"call": None, "volleyed": True} when the other side took it in the air, or None
    when its flight was not seen."""
    f = shot_flight(hit, arcs)
    if f is None:
        return None
    if nxt is not None and nxt["t"] < f["t"] + BOUNCE_TO_HIT_S:
        return {"call": None, "volleyed": True}
    where = {"x": round(f["x"], 2), "y": round(f["y"], 2), "t": round(f["t"], 3)}
    why = judge_flight(hit["side"], f, singles)[1]
    if why == "net":
        # a ball into the net is not played back; when the other side did, the fit was off
        return None if nxt is not None and nxt["side"] != hit["side"] else dict(where, call="net")
    margin = out_by(f, singles) or inside_by(f, singles)
    return dict(where, call="close" if margin < CLOSE_M else why)


def evidence(serves: list[dict], hits: list[dict], arcs: list[dict], singles: bool = False) -> dict:
    """{"p_near": chance the near end won, "how": reason} for one point."""
    serves = sorted(serves, key=lambda s: s["contact_t"])
    rally = sorted((h for h in hits if not h.get("serve")
                    and not any(abs(h["t"] - s["contact_t"]) < SERVE_HIT_S for s in serves)),
                   key=lambda h: h["t"])
    last_serve = serves[-1] if serves else None
    after = [h for h in rally if last_serve is None or h["t"] > last_serve["contact_t"]]

    def toward(side: str, p: float, how: str) -> dict:
        return {"p_near": p if side == "near" else 1 - p, "how": how}

    other = {"near": "far", "far": "near"}
    if last_serve is not None and not after:
        server = last_serve["server_end"]
        if len(serves) >= 2 and all(s["call"] in FAULTS for s in serves[-2:]):
            return toward(other[server], P_DOUBLE_FAULT, "double_fault")
        if last_serve["call"] in ("in", "close"):
            return toward(server, P_ACE, "ace")
        return toward(other[server], P_OUT, "fault")  # the other fault was not seen
    if not after:
        return {"p_near": 0.5, "how": "unknown"}
    # a ball clearly out ends the point even when the other side plays it back
    for h, nxt in zip(after, after[1:]):
        f = shot_flight(h, arcs)
        if f is not None and out_by(f, singles) >= CLEAR_OUT and f["t"] < nxt["t"] and judge_flight(h["side"], f, singles)[1] == "out":
            return toward(other[h["side"]], P_OUT, "out")
    last = after[-1]
    f = shot_flight(last, arcs)
    if f is None:
        return toward(other[last["side"]], P_UNSEEN, "error_unseen")
    result, why = judge_flight(last["side"], f, singles)
    if result == "won":
        return toward(last["side"], P_IN, "winner")
    return toward(other[last["side"]], P_NET if why == "net" else P_OUT, why)


def _state(sc) -> tuple:
    return (tuple(sc.sets), tuple(sc.games), tuple(sc.points), sc.adv, sc.tiebreak, tuple(sc.tb),
            sc.completed_games, sc.game_server_index, sc.tb_points_played, sc.team_a_near)


def decide(points: list[dict], settings: dict, seen: dict) -> dict:
    """Winners for every point (time order across the project): {point_id: "A" | "B"}.

    seen[point_id] = {"p_near": ..., "server_end": "near" | "far" | None}. Points with a winner
    already set keep it; the rest are chosen to fit both the evidence and the score's server."""
    from core.scoring import MatchScorer
    scorer = MatchScorer(server_rotation=settings.get("server_rotation") or ["A1", "B1"],
                         no_ad=bool(settings.get("no_ad", False)),
                         tiebreak_at_6_all=bool(settings.get("tiebreak_at_6_all", True)),
                         team_a_start_end=settings.get("team_a_start_end") or "near",
                         change_ends=settings.get("change_ends", True) is not False)
    beam = [(0.0, scorer, [])]
    for p in points:
        e = seen.get(p.get("id")) or {}
        nxt = []
        for score, sc, picks in beam:
            snap = sc.snapshot()
            a_near = snap.team_a_end == "near"
            if e.get("server_end"):
                serving_a = sc._server_team(snap.server_id) == 0
                a_end = "near" if a_near else "far"
                ok = (e["server_end"] == a_end) == serving_a
                score += math.log(P_SERVE_SEEN if ok else 1 - P_SERVE_SEEN)
            options = [p["winner"]] if p.get("winner") in ("A", "B") else ["A", "B"]
            for w in options:
                pn = e.get("p_near", 0.5)
                pa = pn if a_near else 1 - pn
                s2 = score + (0.0 if len(options) == 1 else math.log(max(1e-6, pa if w == "A" else 1 - pa)))
                c = copy.deepcopy(sc)
                c.award_point(w)
                nxt.append((s2, c, picks + [w]))
        nxt.sort(key=lambda b: -b[0])
        best_at: dict = {}  # two ways to the same score: only the likelier one matters from here
        for b in nxt:
            best_at.setdefault(_state(b[1]), b)
        beam = list(best_at.values())[:BEAM]
    best = beam[0][2] if beam else []
    return {p.get("id"): w for p, w in zip(points, best)}


def point_evidence(points: list[dict], serves: list[dict], rally: dict | None, singles: bool = False) -> dict:
    """{point_id: {"p_near", "how", "server_end"}} for the points measure_file just tracked."""
    out = {}
    for p in points:
        sv = [s for s in serves if p["start"] <= s["contact_t"] <= p["end"]]
        hs = [h for h in (rally or {}).get("hits", []) if p["start"] <= h["t"] <= p["end"]]
        e = evidence(sv, hs, (rally or {}).get("arcs", {}).get(p["id"], []), singles)
        e["server_end"] = sv[0]["server_end"] if sv else None
        out[p["id"]] = e
    return out


def fill_winners(project: dict) -> int:
    """Chooses the winners the analysis may set (points with evidence whose winner is blank or was
    chosen here before) so the whole project's score holds together; the user's own stay. Returns
    how many winners changed."""
    def ours(p):
        return p.get("winner_evidence") and (p.get("winner_auto") or (p.get("winner") is None and not p.get("winner_cleared")))
    order = {f.get("id"): i for i, f in enumerate(project.get("files", []))}
    pts = sorted((p for p in project.get("points", []) if p.get("winner") in ("A", "B") or ours(p)),
                 key=lambda p: (order.get(p.get("file_id"), 999999), p.get("start", 0)))
    mine = {p["id"] for p in pts if ours(p)}
    free = [dict(p, winner=None) if p["id"] in mine else p for p in pts]
    picks = decide(free, project.get("settings") or {}, {p["id"]: p["winner_evidence"] for p in pts if p.get("winner_evidence")})
    changed = 0
    for p in pts:
        if p["id"] in mine:
            changed += p.get("winner") != picks.get(p["id"])
            p["winner"] = picks.get(p["id"])
            p["winner_auto"] = True
    return changed
