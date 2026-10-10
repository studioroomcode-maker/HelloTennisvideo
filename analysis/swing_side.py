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


def pose_size(side: str) -> int:
    """YOLO input size: far players are ~100 px tall in 4K and are found reliably only when upscaled."""
    return 2560 if side == "far" else 1280


def decode_clip(src: str, t0: float, dur: float, box, fps: int, size: int):
    """Frames of the crop `box` over [t0, t0 + dur) and the factor back to crop pixels. A crop
    bigger than the pose input is shrunk here (YOLO would shrink it anyway): the full-width near
    crop of a 4K video was 17 MB a frame through the pipe, and decoding cost 3x the pose itself."""
    import numpy as np
    x, y, w, h = box
    vf = f"crop={w}:{h}:{x}:{y},fps={fps}"
    k = max(w, h) / size
    if k > 1:
        w, h = round(w / k / 2) * 2, round(h / k / 2) * 2
        vf += f",scale={w}:{h}:flags=area"
    else:
        k = 1.0
    raw = decode_raw(["-nostdin", "-loglevel", "error", "-ss", f"{max(0.0, t0):.3f}", "-i", src,
                      "-t", f"{dur:.3f}", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"])
    n = len(raw) // (w * h * 3)
    return np.frombuffer(raw[:n * w * h * 3], np.uint8).reshape(n, h, w, 3), k


def detect_poses(model, frames, size: int, k: float = 1.0, batch: int = 8) -> list:
    """[(frame index, [(keypoints, xyxy), ...]), ...] in crop pixels, frames run through YOLO in batches."""
    import torch
    half = torch.cuda.is_available()
    dets = []
    for b in range(0, len(frames), batch):
        rs = model.predict(list(frames[b:b + batch]), imgsz=size, conf=0.25, half=half, verbose=False)
        for j, r in enumerate(rs):
            if r.keypoints is None or not len(r.keypoints):
                continue
            kps = r.keypoints.data.cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy()
            if k != 1.0:
                kps[:, :, :2] *= k
                boxes *= k
            dets.append((b + j, list(zip(kps.tolist(), boxes.tolist()))))
    return dets


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

    def _clip(self, hit: dict):
        from analysis.hit_shots import crop_box  # hit_shots imports this module
        side = hit["side"]
        tc = contact_time(hit)
        box = crop_box(self.court, side, self.W, self.H)
        size = pose_size(side)
        return tc, box, size, decode_clip(self.src, tc - PRE, PRE + POST, box, self.fps, size)

    def _judge(self, hit: dict, clip) -> dict | None:
        tc, box, size, (frames, k) = clip
        side = hit["side"]
        people = track_people(detect_poses(self.model, frames, size, k), self.court, side, box, self.W, self.H)
        res = judge_people(people, hit.get("balls"), tc, side, box, self.W, self.H, self.fps)
        if res:
            res["swing"] = round(res["swing"], 3)
            res["dist"] = None if res["dist"] is None else round(res["dist"], 2)
        return res

    def __call__(self, hit: dict) -> dict | None:
        return self._judge(hit, self._clip(hit))

    def many(self, hits: list[dict], workers: int = 3):
        """Yield (index, result) in order. FFmpeg cuts the next clips while the GPU runs the pose
        model on this one: decoding was the slow half, and it runs off the GPU."""
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(workers) as pool:
            ahead = workers + 1
            futs = {k: pool.submit(self._clip, h) for k, h in enumerate(hits[:ahead])}
            for k, h in enumerate(hits):
                clip = futs.pop(k).result()
                if k + ahead < len(hits):
                    futs[k + ahead] = pool.submit(self._clip, hits[k + ahead])
                yield k, self._judge(h, clip)
