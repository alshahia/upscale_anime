#!/usr/bin/env python3
"""Style-dial examples on a single image (powered step 2, option 2).

Takes an HR frame, downscales it 4x (simulating 480p), upscales it back with
alpha blends of realesr-animevideov3 (the anime-video style, alpha=1) and
4xLSDIRCompactv2 (the cleaner/milder LSDIR style, alpha=0), then applies the
color statistics of the reference images in data/style/.

Usage:
  python scripts/step_style_image.py --image "data/val_hr/<file>.png" \
      --alphas 0.05,0.10,0.20,0.30,0.40,0.50 [--style-strength 1.0]

Outputs in results/style_image/:
  style_dial.png  - labeled grid: bicubic LR baseline | each blend w/ style
  <label>.png     - the full-res styled tile for each alpha
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent


def color_transfer(img_float, ref_float, strength=1.0):
    """Per-channel mean/std shift of img toward ref (both float [0,1])."""
    m, sd = img_float.mean((0, 1)), img_float.std((0, 1)) + 1e-8
    mr, sr = ref_float.mean((0, 1)), ref_float.std((0, 1)) + 1e-8
    tgt_m = m + strength * (mr - m)
    tgt_s = sd * (1.0 + strength * (sr / sd - 1.0))
    return np.clip((img_float - m) / sd * tgt_s + tgt_m, 0, 1)


def tile_with_label(arr_u8, text, label_h=40):
    im = Image.fromarray(arr_u8)
    W, H = im.size
    out = Image.new("RGB", (W, H + label_h), (18, 18, 18))
    d = ImageDraw.Draw(out)
    d.text((10, 10), text, fill=(255, 255, 255))
    out.paste(im, (0, label_h))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="HR source frame (native 1920x1080)")
    ap.add_argument("--style", default=str(ROOT / "data" / "style"),
                    help="style reference image or folder of images")
    ap.add_argument("--alphas", default="0.05,0.10,0.20,0.30,0.40,0.50",
                    help="blend weights; 1.0 = pure animevideov3, 0.0 = pure LSDIR")
    ap.add_argument("--style-strength", type=float, default=1.0)
    ap.add_argument("--out", default=str(ROOT / "results" / "style_image"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "apps" / "anime_upscaler_gui"))
    from anime_upscaler_gui.archs import build

    DEV = torch.device("cuda")

    # HR -> LR (simulated 480p downscale, like the Vimeo pipeline)
    hr = np.asarray(imageio.imread(args.image), dtype=np.uint8)[..., :3]
    hf = torch.from_numpy(hr).permute(2, 0, 1)[None].float() / 255.0
    lr = F.interpolate(hf, scale_factor=0.25, mode="bicubic", antialias=True).clamp(0, 1)

    # style reference statistics: combine all images in the folder
    ref_paths = sorted(p for p in Path(args.style).glob("*")
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    if not ref_paths:
        raise SystemExit(f"no style reference images in {args.style}")
    chunks = []
    for p in ref_paths:
        a = np.asarray(imageio.imread(str(p)), dtype=np.float32) / 255.0
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=2)
        chunks.append(a[..., :3].reshape(-1, 3))
    ref_all = np.concatenate(chunks, axis=0)
    ref_mean_px = ref_all[np.linspace(0, ref_all.shape[0] - 1, 512 * 512).astype(np.int64)]
    ref_img = ref_mean_px.reshape(512, 512, 3)
    print(f"[style] {len(ref_paths)} reference image(s) combined: {[p.name for p in ref_paths]}", flush=True)

    sd_a = torch.load(str(ROOT / "pretrained" / "realesr-animevideov3.pth"),
                      map_location="cpu", weights_only=False)
    sd_a = sd_a.get("params", sd_a)
    sd_b = torch.load(str(ROOT / "pretrained" / "4xLSDIRCompactv2.pth"),
                      map_location="cpu", weights_only=False)
    sd_b = sd_b.get("params", sd_b)
    assert set(sd_a) == set(sd_b)

    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    model = build("srvgg", str(ROOT / "pretrained" / "realesr-animevideov3.pth")).to(DEV).eval().half()
    lr_t = lr.to(DEV).half()

    with torch.inference_mode():
        tiles = []
        bicubic = (F.interpolate(lr, scale_factor=4, mode="bicubic", antialias=True)
                   [0].permute(1, 2, 0).numpy())
        tiles.append(tile_with_label((bicubic * 255).round().astype(np.uint8),
                                     "bicubic 4x (no SR)"))
        cur_alpha = None
        for alpha in alphas:
            if cur_alpha != alpha:
                sd = {k: (alpha * sd_a[k].float() + (1 - alpha) * sd_b[k].float()).to(sd_a[k].dtype)
                      for k in sd_a}
                torch.save({"params": sd, "scale": 4}, str(out / "_tmp_ckpt.pth"))
                model = build("srvgg", str(out / "_tmp_ckpt.pth")).to(DEV).eval().half()
                cur_alpha = alpha
            y = model(lr_t).clamp(0, 1)
            up = (y[0].permute(1, 2, 0).float().cpu().numpy())
            styled = (color_transfer(up, ref_img, args.style_strength) * 255).round().astype(np.uint8)
            imageio.imwrite(str(out / f"styled_a{alpha:.2f}.png"), styled)
            tiles.append(tile_with_label(styled, f"alpha={alpha:.2f} + style"))
            print(f"[alpha {alpha:.2f}] done", flush=True)
        (out / "_tmp_ckpt.pth").unlink()

    # 2-row grid (tiles sorted), each full 1920x1080+label -> keep native size
    W0 = max(t.size[0] for t in tiles)
    grid_rows = []
    per_row = 3
    for i in range(0, len(tiles), per_row):
        row = tiles[i:i + per_row]
        row = [t for t in row] + [Image.new("RGB", (W0, row[0].size[1]), (10, 10, 10))
                                  for _ in range(per_row - len(row))]
        grid_rows.append(np.concatenate([np.asarray(t) for t in row], axis=1))
    grid = np.concatenate(grid_rows, axis=0)
    grid_path = out / "style_dial.png"
    imageio.imwrite(str(grid_path), grid)
    print("[done]", grid_path, grid.shape, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
