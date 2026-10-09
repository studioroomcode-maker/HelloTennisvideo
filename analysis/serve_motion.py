"""Serve type from the server's own motion, learnt per server from the user's labels.

The ball's bend (analysis/serve_spin.py) shows a far-side slice clearly, but the near
server's slices bent only 0.7-1.9 m/s^2 on the test video. His motion told them apart
instead: after a flat first serve he drove through so hard that the back leg kicked up
(ankles 0.21-0.24 body heights apart within 0.45 s of contact) while after slices they
stayed 0.08-0.12 apart. That is one player's habit, not a rule (the far server's slices
ranged 0.13-0.36), so the cut is learnt per server from serves the user labelled with a
type, and a serve's own label is left out when it is judged so scoring stays honest.
"""
from __future__ import annotations

import numpy as np

FPS = 30
AFTER_S = 0.45      # s after contact over which the follow-through is watched
MIN_GAP = 0.04      # leg-split difference between the two classes' nearest examples needed to trust the cut
LABEL_S = 1.0       # s between a labelled serve and a measured contact to pair them


def leg_split(model, src: str, court, W: int, H: int, serve: dict):
    """Largest vertical gap between the server's ankles after contact, in body heights (or None)."""
    from analysis.hit_shots import crop_box
    from analysis import swing_side as ss
    from analysis.serve_speed import COURT_L
    side, tc = serve["server_end"], serve["contact_t"]
    box = crop_box(court, side, W, H)
    x, y, w, h = box
    from core.media import decode_raw
    raw = decode_raw(["-nostdin", "-loglevel", "error", "-ss", f"{max(0.0, tc - 0.1):.3f}", "-i", src,
                      "-t", f"{AFTER_S + 0.1:.3f}", "-vf", f"crop={w}:{h}:{x}:{y},fps={FPS}", "-f", "rawvideo",
                      "-pix_fmt", "bgr24", "-"])
    n = len(raw) // (w * h * 3)
    frames = np.frombuffer(raw[:n * w * h * 3], np.uint8).reshape(n, h, w, 3)
    dets = []
    for i, img in enumerate(frames):
        r = model.predict(img, imgsz=2560 if side == "far" else 1280, conf=0.25, verbose=False)[0]
        if r.keypoints is not None and len(r.keypoints):
            dets.append((i, list(zip(r.keypoints.data.cpu().numpy().tolist(), r.boxes.xyxy.cpu().numpy().tolist()))))
    people = ss.track_people(dets, court, side, box, W, H)
    if not people:
        return None
    u, _ = court.to_image(serve["server_court"][0], 0.0 if side == "near" else COURT_L)
    sx = u * W - x
    i0 = int(0.1 * FPS)

    def centre(p):
        k = min(p["rows"], key=lambda k: abs(k - i0))
        bb = p["rows"][k][1]
        return (bb[0] + bb[2]) / 2

    server = min(people, key=lambda p: abs(centre(p) - sx))
    best = None
    for i, (kp, bb) in server["rows"].items():
        if i < i0 + 2:
            continue
        ankles = [kp[k] for k in (15, 16) if kp[k][2] > 0.3]
        if len(ankles) == 2:
            gap = abs(ankles[0][1] - ankles[1][1]) / max(1.0, bb[3] - bb[1])
            best = gap if best is None else max(best, gap)
    return None if best is None else round(float(best), 3)


def _server_of(serve, points) -> str | None:
    p = next((p for p in points if p["id"] == serve.get("point_id")), None)
    return ((p or {}).get("score_before") or {}).get("server_id")


def learn_types(serves: list[dict], labels: list[dict], points: list[dict]):
    """Set serve["serve_type_motion"] from the same server's labelled serves (own label left out)."""
    labelled = []
    for s in serves:
        lab = min((l for l in labels if l.get("shot") == "serve" and l.get("spin")
                   and abs(l["t"] - s["contact_t"]) <= LABEL_S), key=lambda l: abs(l["t"] - s["contact_t"]), default=None)
        s["_label_spin"] = lab["spin"] if lab else None
        s["_server"] = _server_of(s, points)
        if lab and s.get("leg_split") is not None:
            labelled.append(s)
    for s in serves:
        if s.get("leg_split") is None or not s["_server"]:
            continue
        train = [o for o in labelled if o is not s and o["_server"] == s["_server"]]
        kinds = {}
        for o in train:
            kinds.setdefault(o["_label_spin"], []).append(o["leg_split"])
        if len(kinds) < 2:
            continue  # need examples of at least two types from this server
        # two classes along one number: the cut is valid only if they don't overlap
        (ka, va), (kb, vb) = sorted(((k, sorted(v)) for k, v in kinds.items()), key=lambda kv: np.mean(kv[1]))[:2]
        if vb[0] - va[-1] < MIN_GAP:
            continue
        cut = (va[-1] + vb[0]) / 2
        s["serve_type_motion"] = ka if s["leg_split"] < cut else kb
        s["serve_type_motion_cut"] = round(cut, 3)
    for s in serves:
        s.pop("_label_spin", None)
        s.pop("_server", None)
