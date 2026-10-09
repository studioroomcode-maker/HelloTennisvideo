"""WASB ball detector (nttcom/WASB-SBDT, BMVC 2023) with its tennis weights.

Drop-in alternative to analysis.tracknet.BallDetector: `heat_levels(frames)` takes
640x360 BGR frames and returns one 640x360 uint8 heat map per frame (sigmoid x 255),
so the existing blob/track code works unchanged (its 127 threshold = WASB's 0.5).

The network sees 3 consecutive frames (RGB, ImageNet-normalized, 512x288) and
predicts a heat map for each of them. Windows slide one frame at a time and the up
to three predictions a frame gets are averaged.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

INP_W, INP_H = 512, 288
OUT_W, OUT_H = 640, 360
WEIGHTS_URL = ("https://drive.usercontent.google.com/download?"
               "id=14AeyIOCQ2UaQmbZLNQJa1H_eSwxUXk7z&export=download&confirm=t")
WEIGHTS_SIZE_MIN = 5_000_000

# src/configs/model/wasb.yaml
CFG = {
    "frames_in": 3, "frames_out": 3, "out_scales": [0],
    "MODEL": {"EXTRA": {
        "FINAL_CONV_KERNEL": 1, "PRETRAINED_LAYERS": ["*"],
        "STEM": {"INPLANES": 64, "STRIDES": [1, 1]},
        "STAGE1": {"NUM_MODULES": 1, "NUM_BRANCHES": 1, "BLOCK": "BOTTLENECK", "NUM_BLOCKS": [1],
                   "NUM_CHANNELS": [32], "FUSE_METHOD": "SUM"},
        "STAGE2": {"NUM_MODULES": 1, "NUM_BRANCHES": 2, "BLOCK": "BASIC", "NUM_BLOCKS": [2, 2],
                   "NUM_CHANNELS": [16, 32], "FUSE_METHOD": "SUM"},
        "STAGE3": {"NUM_MODULES": 1, "NUM_BRANCHES": 3, "BLOCK": "BASIC", "NUM_BLOCKS": [2, 2, 2],
                   "NUM_CHANNELS": [16, 32, 64], "FUSE_METHOD": "SUM"},
        "STAGE4": {"NUM_MODULES": 1, "NUM_BRANCHES": 4, "BLOCK": "BASIC", "NUM_BLOCKS": [2, 2, 2, 2],
                   "NUM_CHANNELS": [16, 32, 64, 128], "FUSE_METHOD": "SUM"},
        "DECONV": {"NUM_DECONVS": 0, "KERNEL_SIZE": [], "NUM_BASIC_BLOCKS": 2},
    }, "INIT_WEIGHTS": True},
}


class _Cfg(dict):
    """dict that also answers attribute access, like the omegaconf config hrnet.py expects."""

    def __init__(self, d):
        super().__init__({k: _Cfg(v) if isinstance(v, dict) else v for k, v in d.items()})

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as exc:
            raise AttributeError(k) from exc


def weights_path(root: Path) -> Path:
    return Path(root) / "models" / "wasb_tennis.pth.tar"


def ensure_weights(root: Path, report=None) -> Path:
    path = weights_path(root)
    if path.exists() and path.stat().st_size >= WEIGHTS_SIZE_MIN:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if report:
        report({"message": "공 검출 모델(WASB 테니스, 6MB) 다운로드 중 (최초 1회)..."})
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(WEIGHTS_URL, timeout=120) as resp, tmp.open("wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    if tmp.stat().st_size < WEIGHTS_SIZE_MIN:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("WASB 가중치 다운로드가 불완전해.")
    tmp.replace(path)
    return path


class WASBDetector:
    def __init__(self, weights: Path, device: str | None = None):
        import torch
        from analysis.wasb_hrnet import HRNet
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = HRNet(_Cfg(CFG))
        ck = torch.load(str(weights), map_location="cpu", weights_only=True)
        model.load_state_dict(ck.get("model_state_dict", ck))
        self.model = model.to(self.device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.stride = 2  # serve windows are read at the native ~60 fps; set 1 for 30 fps footage

    def _prep(self, frames):
        import cv2
        import numpy as np
        torch = self.torch
        rgb = np.stack([cv2.resize(f, (INP_W, INP_H), interpolation=cv2.INTER_LINEAR)[:, :, ::-1] for f in frames])
        x = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        return (x - self.mean) / self.std  # (N, 3, H, W)

    def heat_levels(self, frames, batch: int = 32, stride: int | None = None, dense: bool = False):
        """stride: frame gap inside a window (2 for ~60 fps input, matching the 30 fps training data).
        dense: one window per frame (averaging up to 3 predictions) instead of disjoint windows."""
        import cv2
        torch = self.torch
        n = len(frames)
        s_ = stride if stride is not None else self.stride
        if n < 2 * s_ + 1:
            return [None] * n
        x = self._prep(frames)
        acc = torch.zeros((n, INP_H, INP_W), device=self.device)
        cnt = torch.zeros(n, device=self.device)
        last = n - 2 * s_
        if dense:
            starts = list(range(0, last))
        else:
            # disjoint windows: with stride s, s interleaved chains of non-overlapping triples
            starts = [b + k * 3 * s_ for b in range(s_) for k in range((last - b + 3 * s_ - 1) // (3 * s_))]
            starts = [st for st in starts if st < last]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            for k in range(0, len(starts), batch):
                st = torch.tensor(starts[k:k + batch], device=self.device)
                inp = torch.cat([x[st], x[st + s_], x[st + 2 * s_]], dim=1)  # oldest first, as in training
                hm = self.model(inp)[0].float().sigmoid()  # (B, 3, H, W): one map per input frame
                for j in range(3):
                    acc.index_add_(0, st + j * s_, hm[:, j])
                    cnt.index_add_(0, st + j * s_, torch.ones(len(st), device=self.device))
        heat = (acc / cnt.clamp(min=1).view(-1, 1, 1) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        return [cv2.resize(h, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR) if c > 0 else None
                for h, c in zip(heat, cnt.cpu().numpy())]


class EnsembleDetector:
    """Pixel-wise max of two detectors' heat maps (e.g. TrackNet + WASB)."""

    def __init__(self, *detectors):
        self.detectors = detectors

    def heat_levels(self, frames):
        import numpy as np
        out = None
        for d in self.detectors:
            hs = d.heat_levels(frames)
            if out is None:
                out = list(hs)
                continue
            out = [b if a is None else a if b is None else np.maximum(a, b) for a, b in zip(out, hs)]
        return out
