#!/usr/bin/env python3
"""Option 1 - fine-tune a style directly into the SRVGG weights.

Instead of post-hoc color statistics (option 2), we fine-tune the pretrained
realesr-animevideov3 (or style-blend) weights on paired data:

    LR = bicubic-4x downscale of HR (+ degradation)
    HR target = the SAME HR frame precolored toward data/style statistics
                (the net must learn to reproduce that palette/cel look)

The learned style is baked into the resulting .pth - no runtime style ref
needed. Trains a 621k-param student at 4x scale, renders the exact
20:45:00-20:55:00 clip with the new weights and returns IQA + timing.

  python scripts/step_style_train.py --iters 4000 --out results/style_ft
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parent.parent
from apps.anime_upscaler_gui.anime_upscaler_gui.arch_registry import build
import gzip  # noqa keep imports explicit




def color_transfer2(img, m_r, s_r, strength=1.0):
    m, sd = img.mean((0, 1)), img.std((0, 1)) + 1e-8
    tgt_m = m + strength * (m_r - m)
    tgt_s = sd * (1.0 + strength * (s_r / sd - 1.0))
    return np.clip((img - m) / sd * tgt_s + tgt_m, 0, 1)


def load_style_stats(style_dir):
    chunks = []
    for p in sorted(Path(style_dir).glob("*")):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            a = np.asarray(imageio.imread(str(p)), dtype=np.float32) / 255.0
            if a.ndim == 2:
                a = np.repeat(a[..., None], 3, axis=2)
            chunks.append(a[..., :3].reshape(-1, 3))
    if not chunks:
        return None, None
    ref = np.concatenate(chunks, axis=0)
    return ref.mean(0), ref.std(0) + 1e-8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr-dir", nargs="*", default=[str(ROOT / "data" / "val_hr")])
    ap.add_argument("--style", default=str(ROOT / "data" / "style"),
                    help="style reference folder whose palette gets baked in")
    ap.add_argument("--style-strength", type=float, default=1.0)
    ap.add_argument("--init", default=str(ROOT / "results/style_mix/weights/stylemix_a0.50.pth"))
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-crop", type=int, default=64)
    ap.add_argument("--out", default=str(ROOT / "results" / "style_ft"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    DEV = torch.device("cuda")

    hr_files = []
    for d in args.hr_dir:
        hr_files += sorted(p for p in Path(d).glob("**/*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not hr_files:
        raise SystemExit("no HR training images found")
    random.Random(0).shuffle(hr_files)
    print(f"[data] {len(hr_files)} HR images", flush=True)

    m_ref, s_ref = load_style_stats(args.style)
    if m_ref is None:
        print("[style] no style images - plain SR fine-tune", flush=True)

    ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
    sd = ckpt.get("params", ckpt)
    model = build("srvgg", args.init)
    model.load_state_dict(sd, strict=True)
    model.train().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters, eta_min=1e-6)

    rng = random.Random(7)
    cache = {}
    LR = args.lr_crop
    HRLR = LR * 4

    def sample():
        p = hr_files[rng.randrange(len(hr_files))]
        img = cache.get(p)
        if img is None:
            a = np.asarray(imageio.imread(str(p)), np.float32) / 255.0
            a = a if a.ndim == 3 else np.repeat(a[..., None], 3, axis=2)
            img = torch.from_numpy(a[..., :3]).permute(2, 0, 1)
            cache[p] = img
            if len(cache) > 256:  # bounded cache
                cache.pop(next(iter(cache)))
        _, H, W = img.shape
        if H < HRLR + 8 or W < HRLR + 8:
            return None
        y = rng.randrange(H - HRLR - 8)
        x = rng.randrange(W - HRLR - 8)
        hr = img[:, y:y + HRLR, x:x + HRLR].unsqueeze(0)
        hrn = hr[0].permute(1, 2, 0).numpy()
        styled = color_transfer2(hrn, m_ref, s_ref, args.style_strength)
        hr_s = torch.from_numpy(np.ascontiguousarray(styled)).permute(2, 0, 1).unsqueeze(0)
        lr = F.interpolate(hr, scale_factor=0.25, mode="bicubic", antialias=True)
        # mild degradation: tiny blur + noise
        lr = F.avg_pool2d(lr, 3, 1, 1)
        lr = (lr + 0.01 * torch.randn_like(lr)).clamp(0, 1)
        return lr.to(DEV), hr_s.to(DEV)

    t0 = time.time()
    losses = []
    torch.backends.cudnn.benchmark = True
    for it in range(1, args.iters + 1):
        pair = None
        while pair is None:
            pair = sample()
        lr, hr = pair
        pred = model(lr)
        loss = F.l1_loss(pred, hr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if it % 250 == 0 or it == 1:
            print(f"[train] iter {it} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"({(time.time()-t0)/(it and it):.3f}s/iter avg {(time.time()-t0)/it:.3f})", flush=True)

    torch.save({"params": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "scale": 4},
               out / "styled_student.pth")
    (out / "train_log.json").write_text(json.dumps(
        {"iters": args.iters, "batch": args.batch, "lr": args.lr, "init": args.init,
         "style": args.style, "style_strength": args.style_strength,
         "loss_first": losses[0], "time_s": round(time.time() - t0, 1)}, indent=2))
    print("[saved]", out / "styled_student.pth", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
