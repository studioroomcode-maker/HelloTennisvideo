"""Forehand or backhand from the hitter's swing, measured with pose estimation.

Around each tracked hit, every person standing in the hitter's half of the court is followed
with YOLO pose for 0.75 s. The hitter is the one the ball comes closest to (or, when the ball
is lost near the player, the one with the biggest swing). Its wrists are measured sideways
from the torso, in the player's own frame: the near players show their backs, so their right
is the image right; the far players face the camera, so their right is the image left.

The forward swing is the fastest sideways move of the hands through contact. For a
right-hander a forehand sweeps the hands from right to left, a backhand from left to right
(both wrists, one- or two-handed alike); a left-hander is the mirror image.

On points 1-6 of the test video this named all 10 hand-labelled ground strokes correctly,
where Gemini managed 4-6 of them from the same moments.
"""
from __future__ import annotations


from analysis.hit_detect import contact_time
from analysis.serve_speed import COURT_L, COURT_W, NET_Y, Court
from core.media import decode_raw

FPS = 30
PRE, POST = 0.45, 0.30          # s of video around the contact moment
SWING_MIN = 0.10                # forward swing, in body heights, needed to name a side
BALL_NEAR = 1.0                 # ball within this many body heights of a player marks the hitter
MIN_ROWS = 8                    # frames a person needs to be judged


def _frames(src: str, t: float, box, fps: int):
    import numpy as np
    x, y, w, h = box
    raw = decode_raw(["-nostdin", "-loglevel", "error", "-ss", f"{max(0.0, t - PRE):.3f}", "-i", src,
                      "-t", f"{PRE + POST:.3f}", "-vf", f"crop={w}:{h}:{x}:{y},fps={fps}", "-f", "rawvideo",
                      "-pix_fmt", "bgr24", "-"])
    n = len(raw) // (w * h * 3)
    return np.frombuffer(raw[:n * w * h * 3], np.uint8).reshape(n, h, w, 3)


def _on_side(court: Court, side: str, fx: float, fy: float) -> bool:
    if not -3 <= fx <= COURT_W + 3:
        return False
    return NET_Y + 0.5 <= fy <= COURT_L + 7 if side == "far" else -7 <= fy <= NET_Y - 0.5


def track_people(detections, court: Court, side: str, box, W: int, H: int) -> list[dict]:
    """Link per-frame detections [(i, [(keypoints, xyxy), ...]), ...] into people on `side`."""
    import numpy as np
    people = []
    for i, dets in detections:
        for kp, bb in dets:
            c = np.array([(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2])
            bh = max(1.0, bb[3] - bb[1])
            fx, fy = court.to_court((c[0] + box[0]) / W, (bb[3] + box[1]) / H)
            if not _on_side(court, side, fx, fy):
                continue
            best = None
            for p in people:
                j = max(p["rows"])
                d = float(np.hypot(*(c - p["c"]))) / bh
                if i not in p["rows"] and i - j <= 8 and d < 0.4 + 0.1 * (i - j) and (best is None or d < best[0]):
                    best = (d, p)
            if best:
                best[1]["rows"][i] = (kp, bb)
                best[1]["c"] = c
            else:
                people.append({"rows": {i: (kp, bb)}, "c": c})
    return [p for p in people if len(p["rows"]) >= MIN_ROWS]


def lateral(rows: dict, side: str, fps: int) -> list[tuple[float, float]]:
    """[(time from contact, hands' sideways offset from the torso in body heights, + = player's right)]."""
    sign = 1 if side == "near" else -1
    out = []
    for i, (kp, bb) in sorted(rows.items()):
        bh = max(1.0, bb[3] - bb[1])
        torso = [kp[k] for k in (5, 6, 11, 12) if kp[k][2] > 0.3]
        wrists = [kp[k] for k in (9, 10) if kp[k][2] > 0.3]
        if torso and wrists:
            tx = sum(p[0] for p in torso) / len(torso)
            wx = sum(p[0] for p in wrists) / len(wrists)
            out.append((i / fps - PRE, (wx - tx) * sign / bh))
    return out


def forward_swing(series) -> float:
    """Largest sideways move of the hands over 0.15-0.25 s centred within 0.25 s of contact."""
    best = 0.0
    for t0, u0 in series:
        for t1, u1 in series:
            if 0.15 <= t1 - t0 <= 0.25 and abs((t0 + t1) / 2) <= 0.25 and abs(u1 - u0) > abs(best):
                best = u1 - u0
    return best


def ball_distance(person: dict, balls, tc: float, box, W: int, H: int, fps: int):
    """Closest approach (body heights) of the ball to the person's box, from the ball points near contact."""
    best = None
    for t, x, y in balls or []:
        if abs(t - tc) > 0.4:
            continue
        i = min(person["rows"], key=lambda k: abs(k - (PRE + t - tc) * fps))
        bb = person["rows"][i][1]
        bh = max(1.0, bb[3] - bb[1])
        bx, by = x * W / 640 - box[0], y * H / 360 - box[1]
        dx = max(bb[0] - bx, 0, bx - bb[2])
        dy = max(bb[1] - 0.5 * bh - by, 0, by - bb[3])
        d = (dx * dx + dy * dy) ** 0.5 / bh
        if best is None or d < best:
            best = d
    return best


def judge_people(people: list[dict], balls, tc, side, box, W, H, fps) -> dict | None:
    if not people:
        return None
    cands = []
    for p in people:
        series = lateral(p["rows"], side, fps)
        i = min(p["rows"], key=lambda k: abs(k - PRE * fps))
        bb = p["rows"][i][1]
        cands.append({"swing": forward_swing(series), "dist": ball_distance(p, balls, tc, box, W, H, fps),
                      "frames": len(series), "box": [round(bb[0] + box[0]), round(bb[1] + box[1]),
                                                     round(bb[2] + box[0]), round(bb[3] + box[1])]})
    close = [c for c in cands if c["dist"] is not None and c["dist"] < BALL_NEAR]
    pick = min(close, key=lambda c: c["dist"]) if close else max(cands, key=lambda c: abs(c["swing"]))
    return dict(pick, by="ball" if close else "swing", people=len(cands))


def stroke(swing: float, hand: str | None) -> str | None:
    """forehand/backhand from a forward swing; None when the swing is too small or the hand unknown."""
    if hand not in ("right", "left") or abs(swing) < SWING_MIN:
        return None
    right_to_left = swing < 0
    return "forehand" if right_to_left == (hand == "right") else "backhand"


class SwingJudge:
    def __init__(self, model_path, src: str, court: Court, W: int, H: int, fps: int | None = None):
        from ultralytics import YOLO
        import torch
        self.model = YOLO(str(model_path))
        self.gpu = torch.cuda.is_available()
        self.fps = fps or (FPS if self.gpu else 15)
        self.src, self.court, self.W, self.H = src, court, W, H

    def __call__(self, hit: dict) -> dict | None:
        from analysis.hit_shots import crop_box  # hit_shots imports this module
        side = hit["side"]
        tc = contact_time(hit)
        box = crop_box(self.court, side, self.W, self.H)
        frames = _frames(self.src, tc, box, self.fps)
        # far players are ~100 px tall in 4K; pose finds them reliably only when upscaled
        size = 2560 if side == "far" else 1280
        dets = []
        for i, img in enumerate(frames):
            r = self.model.predict(img, imgsz=size, conf=0.25, verbose=False)[0]
            if r.keypoints is None or not len(r.keypoints):
                continue
            dets.append((i, list(zip(r.keypoints.data.cpu().numpy().tolist(), r.boxes.xyxy.cpu().numpy().tolist()))))
        people = track_people(dets, self.court, side, box, self.W, self.H)
        res = judge_people(people, hit.get("balls"), tc, side, box, self.W, self.H, self.fps)
        if res:
            res["swing"] = round(res["swing"], 3)
            res["dist"] = None if res["dist"] is None else round(res["dist"], 2)
        return res
