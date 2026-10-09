"""TrackNet ball detector (architecture from yastrebksv/TrackNet, model.py).

Three consecutive 640x360 frames in, a 256-level heatmap out. Weights are fetched
once from the original author's release and loaded with `weights_only=True`, so
the checkpoint can only contain tensors, never code.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

IN_W, IN_H = 640, 360
WEIGHTS_URL = ("https://drive.usercontent.google.com/download?"
               "id=1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl&export=download&confirm=t")
WEIGHTS_SIZE_MIN = 40_000_000


def weights_path(root: Path) -> Path:
    return Path(root) / "models" / "tracknet.pt"


def ensure_weights(root: Path, report=None) -> Path:
    path = weights_path(root)
    if path.exists() and path.stat().st_size >= WEIGHTS_SIZE_MIN:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    if report:
        report({"message": "공 추적 모델(TrackNet, 43MB) 다운로드 중 (최초 1회)..."})
    with urllib.request.urlopen(WEIGHTS_URL, timeout=120) as resp, tmp.open("wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    if tmp.stat().st_size < WEIGHTS_SIZE_MIN:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("TrackNet 가중치 다운로드가 불완전해. 인터넷 연결을 확인하고 다시 시도해줘.")
    tmp.replace(path)
    return path


def active_weights(root: Path, report=None) -> Path:
    """Fine-tuned weights when they were evaluated as better (models/tracknet_ft.json says so)."""
    import json
    ft = Path(root) / "models" / "tracknet_ft.pt"
    meta = ft.with_suffix(".json")
    try:
        if ft.exists() and json.loads(meta.read_text(encoding="utf-8")).get("use"):
            return ft
    except Exception:
        pass
    return ensure_weights(root, report)


def build_model():
    import torch.nn as nn

    class ConvBlock(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.block = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(cout))

        def forward(self, x):
            return self.block(x)

    class BallTrackerNet(nn.Module):
        def __init__(self, out_channels=256):
            super().__init__()
            self.out_channels = out_channels
            self.conv1, self.conv2 = ConvBlock(9, 64), ConvBlock(64, 64)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv3, self.conv4 = ConvBlock(64, 128), ConvBlock(128, 128)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.conv5, self.conv6, self.conv7 = ConvBlock(128, 256), ConvBlock(256, 256), ConvBlock(256, 256)
            self.pool3 = nn.MaxPool2d(2, 2)
            self.conv8, self.conv9, self.conv10 = ConvBlock(256, 512), ConvBlock(512, 512), ConvBlock(512, 512)
            self.ups1 = nn.Upsample(scale_factor=2)
            self.conv11, self.conv12, self.conv13 = ConvBlock(512, 256), ConvBlock(256, 256), ConvBlock(256, 256)
            self.ups2 = nn.Upsample(scale_factor=2)
            self.conv14, self.conv15 = ConvBlock(256, 128), ConvBlock(128, 128)
            self.ups3 = nn.Upsample(scale_factor=2)
            self.conv16, self.conv17 = ConvBlock(128, 64), ConvBlock(64, 64)
            self.conv18 = ConvBlock(64, out_channels)

        def forward(self, x):
            for name in ("conv1", "conv2", "pool1", "conv3", "conv4", "pool2", "conv5", "conv6", "conv7",
                         "pool3", "conv8", "conv9", "conv10", "ups1", "conv11", "conv12", "conv13", "ups2",
                         "conv14", "conv15", "ups3", "conv16", "conv17", "conv18"):
                x = getattr(self, name)(x)
            return x  # (B, 256, H, W) logits over heat levels

    return BallTrackerNet()


class BallDetector:
    def __init__(self, weights: Path, device: str | None = None):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model()
        state = torch.load(str(weights), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        self.model = model.to(self.device).eval()
        if self.device == "cuda":
            self.model = self.model.half()
            # same results at batch 16, ~25% faster once cuDNN has picked its kernels
            torch.backends.cudnn.benchmark = True

    def heat_levels(self, frames_bgr_small: list, batch: int = 16):
        """frames: list of 640x360 BGR uint8 arrays. Returns per-frame uint8 heat maps (first two are None)."""
        import numpy as np
        torch = self.torch
        out = [None, None]
        idx = list(range(2, len(frames_bgr_small)))
        for k in range(0, len(idx), batch):
            group = idx[k:k + batch]
            stack = np.stack([np.concatenate([frames_bgr_small[i], frames_bgr_small[i - 1],
                                              frames_bgr_small[i - 2]], axis=2) for i in group])
            if self.device == "cuda" and len(group) < batch:
                # keep one batch shape: cuDNN re-tunes (slowly) for every new shape it meets
                stack = np.concatenate([stack, np.repeat(stack[-1:], batch - len(group), axis=0)])
            x = torch.from_numpy(stack).to(self.device).permute(0, 3, 1, 2)
            x = x.half() / 255.0 if self.device == "cuda" else x.float() / 255.0
            with torch.no_grad():
                levels = self.model(x).argmax(dim=1).to(torch.uint8).cpu().numpy()
            out.extend(list(levels[:len(group)]))
        return out


def heat_to_point(heat, thresh: int = 127):
    """Centroid of the strongest blob above `thresh` in a 640x360 heat map → (x, y, score) or None."""
    import cv2
    import numpy as np
    if heat is None:
        return None
    mask = (heat > thresh).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    best, best_score = None, 0.0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 2 or area > 400:
            continue
        score = float(heat[labels == i].astype(np.float32).sum())
        if score > best_score:
            best, best_score = i, score
    if best is None:
        return None
    ys, xs = np.nonzero(labels == best)
    w = heat[ys, xs].astype(np.float32)
    return float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum()), best_score
