"""Fine-tune TrackNet on the pseudo-labelled dataset from tools/ball_dataset.py.

Usage:
    python tools/finetune_tracknet.py DATASET_DIR [--steps 6000] [--out models/tracknet_ft.pt]

Same target as the original training (yastrebksv/TrackNet): a Gaussian heat map
(variance 10 at 1280x720, i.e. 2.5 at 640x360) quantized to 256 levels, per-pixel
cross-entropy. Starts from models/tracknet.pt with a small learning rate so it
adapts to these courts without forgetting what it knew. Holds out 10% of the
windows (by video time) and reports precision/recall at 4 px before and after.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from analysis.tracknet import build_model, ensure_weights  # noqa: E402
from analysis.serve_speed import heat_to_blobs  # noqa: E402

W, H = 640, 360
VAR = 2.5
SIZE = 6


def gaussian():
    ax = np.arange(-SIZE, SIZE + 1)
    g = np.exp(-(ax[None, :] ** 2 + ax[:, None] ** 2) / (2 * VAR))
    return (g * 255).astype(np.int64)


G = gaussian()


def target(x, y):
    t = np.zeros((H, W), np.int64)
    xi, yi = int(round(x)), int(round(y))
    y0, y1 = max(0, yi - SIZE), min(H, yi + SIZE + 1)
    x0, x1 = max(0, xi - SIZE), min(W, xi + SIZE + 1)
    if y0 < y1 and x0 < x1:
        t[y0:y1, x0:x1] = G[y0 - (yi - SIZE):y1 - (yi - SIZE), x0 - (xi - SIZE):x1 - (xi - SIZE)]
    return t


class BallSet(Dataset):
    def __init__(self, root: Path, rows, augment: bool):
        self.root, self.rows, self.augment = root, rows, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        imgs = [cv2.imread(str(self.root / "frames" / n)) for n in r["frames"]]
        x, y = r["x"], r["y"]
        if self.augment:
            if random.random() < 0.5:
                imgs = [im[:, ::-1] for im in imgs]
                x = W - 1 - x
            gain = random.uniform(0.8, 1.2)
            bias = random.uniform(-15, 15)
            imgs = [np.clip(im.astype(np.float32) * gain + bias, 0, 255) for im in imgs]
        stack = np.concatenate(imgs, axis=2).astype(np.float32) / 255.0
        return torch.from_numpy(stack.transpose(2, 0, 1).copy()), torch.from_numpy(target(x, y)), \
            torch.tensor([x, y], dtype=torch.float32)


def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for xb, _, xy in loader:
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                lv = model(xb.to(device)).argmax(1).to(torch.uint8).cpu().numpy()
            for heat, (gx, gy) in zip(lv, xy.numpy()):
                b = heat_to_blobs(heat)
                if not b:
                    fn += 1
                elif math.hypot(b[0][0] - gx, b[0][1] - gy) <= 4:
                    tp += 1
                else:
                    fp += 1
                    fn += 1
    model.train()
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return prec, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--fill-weight", type=float, default=3.0, help="oversampling of hard (gap-filled) frames")
    ap.add_argument("--out", default=str(ROOT / "models" / "tracknet_ft.pt"))
    a = ap.parse_args()
    root = Path(a.dataset)
    rows = [json.loads(l) for l in (root / "labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    # hold out every 10th window (window = video + 3 s bucket) so val frames never neighbour train frames
    def bucket(r):
        return zlib.crc32(f"{r['video']}:{int(r['t'] // 12)}".encode()) % 10
    train = [r for r in rows if bucket(r) != 0]
    val = [r for r in rows if bucket(r) == 0]
    val_fill = [r for r in val if r["kind"] == "fill"]
    print(f"samples: train {len(train)} (fill {sum(r['kind'] == 'fill' for r in train)}), "
          f"val {len(val)} (fill {len(val_fill)})", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model()
    model.load_state_dict(torch.load(str(ensure_weights(ROOT)), map_location="cpu", weights_only=True))
    model.to(device).train()

    weights = [a.fill_weight if r["kind"] == "fill" else 1.0 for r in train]
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=a.steps * a.batch, replacement=True)
    tl = DataLoader(BallSet(root, train, True), batch_size=a.batch, sampler=sampler, num_workers=4,
                    persistent_workers=True)
    vl = DataLoader(BallSet(root, val, False), batch_size=8, num_workers=2)
    vfl = DataLoader(BallSet(root, val_fill, False), batch_size=8, num_workers=2) if val_fill else None

    p, r = evaluate(model, vl, device)
    pf, rf = evaluate(model, vfl, device) if vfl else (0, 0)
    print(f"before: val P {p:.3f} R {r:.3f} | hard-frames P {pf:.3f} R {rf:.3f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    crit = torch.nn.CrossEntropyLoss()
    st = time.time()
    losses = []
    for step, (xb, yb, _) in enumerate(tl, 1):
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            loss = crit(model(xb.to(device)), yb.to(device))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        losses.append(loss.item())
        if step % 200 == 0:
            print(f"step {step}/{a.steps} loss {np.mean(losses[-200:]):.5f} {time.time() - st:.0f}s", flush=True)
        if step % 2000 == 0 or step == a.steps:
            p, r = evaluate(model, vl, device)
            pf, rf = evaluate(model, vfl, device) if vfl else (0, 0)
            print(f"  val P {p:.3f} R {r:.3f} | hard-frames P {pf:.3f} R {rf:.3f}", flush=True)
        if step >= a.steps:
            break
    model.eval()
    torch.save({k: v.float().cpu() for k, v in model.state_dict().items()}, a.out)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
