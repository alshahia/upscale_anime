#!/usr/bin/env python3
"""Pre-train sanity check for a downloaded neosr/SPAN pretrained checkpoint.

Loads a single pretrained .pth, runs forward pass on one image, and reports
whether the checkpoint looks healthy. Use BEFORE starting a finetune to confirm
you have a working warm-start and not the flat-blue variant.

Detects the warm-start taint documented in AGENTS.md:
  - upsampler.0.bias mean ~0.43 (clean pretrain: mean ~0.06)
  - forward-pass output dominated by the bias -> flat image

Usage:
  python scripts/eval_pretrained.py --checkpoint pretrained/span_pix_pretrain_4x.pth --input data/test_mini_sr/2.png
  python scripts/eval_pretrained.py --checkpoint pretrained/span_mssim_pretrain_4x.pth --input data/test_mini_sr/2.png
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from anime_sr.models.span import create_neosr_span  # noqa: E402

TAINT_BIAS_MEAN_MIN = 0.35
TAINT_BIAS_MEAN_MAX = 0.45
TAINT_BIAS_STD_MAX = 0.05
FLAT_OUTPUT_STD_MAX = 0.03


def verdict(bias_mean: float, bias_std: float, output_std: float) -> str:
    if TAINT_BIAS_MEAN_MIN <= bias_mean <= TAINT_BIAS_MEAN_MAX and bias_std < TAINT_BIAS_STD_MAX:
        return "BROKEN (warm-start taint, bias_mean~0.43)"
    if output_std < FLAT_OUTPUT_STD_MAX:
        return "BROKEN (forward output is flat)"
    return "OK"


def main() -> int:
    p = argparse.ArgumentParser(description="Sanity-check a downloaded SPAN pretrained checkpoint")
    p.add_argument("--checkpoint", required=True, help="Path to pretrained .pth")
    p.add_argument("--input", required=True, help="Path to LR input image")
    p.add_argument("--output", default="results/eval_pretrained", help="Where to save SR PNG")
    p.add_argument("--scale", type=int, default=4)
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    img_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        print(f"[FATAL] checkpoint not found: {ckpt_path}")
        return 1
    if not img_path.exists():
        print(f"[FATAL] input image not found: {img_path}")
        return 1

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"[FATAL] cv2 could not read image: {img_path}")
        return 1

    print(f"[Eval] checkpoint: {ckpt_path}")
    print(f"[Eval] input image: {img_path} ({img_bgr.shape[1]}x{img_bgr.shape[0]})")

    model = create_neosr_span({"type": "neosr_span", "scale": args.scale})
    info = model.load_neosr_weights(str(ckpt_path), strict=False)
    print(f"[Eval] loaded {info['loaded']}/{info['total_ckpt_keys']} params, skipped {info['skipped']}, mismatched {info['mismatched']}")

    sd = model.state_dict()
    bias = sd.get("upsampler.0.bias")
    if bias is None:
        print("[WARN] model has no upsampler.0.bias -- taint check skipped")
        bias_mean = bias_std = float("nan")
    else:
        bias_mean = float(bias.mean())
        bias_std = float(bias.std())
        print(f"[Eval] upsampler.0.bias  mean={bias_mean:.4f}  std={bias_std:.4f}")

    model.eval()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        out = model(t).clamp(0, 1)

    out_std = float(out.std())
    out_mean = float(out.mean())
    out_min = float(out.min())
    out_max = float(out.max())
    print(f"[Eval] forward output  mean={out_mean:.4f}  std={out_std:.4f}  min={out_min:.4f}  max={out_max:.4f}")
    print(f"[Eval] output shape: {tuple(out.shape)}")

    sr_bgr = cv2.cvtColor((out.squeeze(0).permute(1, 2, 0).numpy() * 255).astype("uint8"), cv2.COLOR_RGB2BGR)
    sr_path = out_dir / f"{ckpt_path.stem}_{img_path.stem}.png"
    cv2.imwrite(str(sr_path), sr_bgr)
    print(f"[Eval] saved SR: {sr_path}")

    v = verdict(bias_mean, bias_std, out_std)
    print(f"\n[Verdict] {v}")
    if v.startswith("BROKEN"):
        print("[Verdict] DO NOT use this checkpoint as a warm-start. Re-download or pick a different file.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())