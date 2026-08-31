#!/usr/bin/env python3
"""Compare cascade 2x + 2x vs single 4x quality on the val split.

Pass criterion: cascade PSNR >= single 4x - 0.3 dB.

Usage:
    python scripts/eval_cascade_vs_single.py \
        --ckpt-2x pretrained/RFDN_distill_v2_2x_student.pth \
        --ckpt-4x pretrained/RFDN_distill_v1_4x_student.pth \
        --val-dir data/anime_video_frames \
        --val-batches 50
"""
import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Locate package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "anime_upscaler"))
from dataset import AnimePairDataset, denorm01  # noqa: E402
from student import RFDN  # noqa: E402


def _load_student(ckpt_path: Path, scale: int) -> RFDN:
    """Load an RFDN checkpoint, auto-sniffing the scale if it doesn't match.

    The 4x v1 checkpoint encodes `Conv2d(52, 48, 3, 3)` for the upsampler;
    the future 2x v2 checkpoint encodes `Conv2d(52, 12, 3, 3)`. We read
    `upsampler.0.weight.shape[0]` and pick the closest legal scale (2/3/4/8)
    so the script accepts both checkpoints regardless of the `--ckpt-2x`
    vs `--ckpt-4x` flag.
    """
    sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for k in ("student", "params", "state_dict", "params_ema",
                  "model_state_dict"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    ups_w = sd.get("upsampler.0.weight") if isinstance(sd, dict) else None
    if ups_w is not None and hasattr(ups_w, "shape") and len(ups_w.shape) == 4:
        out_ch = int(ups_w.shape[0])
        # out_ch = num_out_ch * scale**2 = 3 * scale**2 => scale = sqrt(out_ch/3)
        ratio = out_ch // 3
        detected = int(round(math.sqrt(max(ratio, 1))))
        if detected * detected == ratio and detected in (2, 3, 4, 8):
            scale = detected
    m = RFDN(scale=scale)
    m.load_state_dict(sd, strict=False)
    return m.eval()


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a.clamp(0, 1) - b.clamp(0, 1)) ** 2).mean(dim=(1, 2, 3)).clamp(min=1e-10)
    return float((-10 * torch.log10(mse)).mean().item())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-dir", default="data/anime_video_frames",
                    help="folder of HR frames; split=val used for eval")
    ap.add_argument("--ckpt-2x", required=True,
                    help="2x student checkpoint (the cascade inner model)")
    ap.add_argument("--ckpt-4x", required=True,
                    help="4x student checkpoint (single-shot baseline)")
    ap.add_argument("--val-batches", type=int, default=50,
                    help="number of val minibatches to evaluate (>=1)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not Path(args.ckpt_2x).exists():
        print(f"ERROR: --ckpt-2x not found: {args.ckpt_2x}")
        return 2
    if not Path(args.ckpt_4x).exists():
        print(f"ERROR: --ckpt-4x not found: {args.ckpt_4x}")
        return 2

    print(f"loading 2x student {args.ckpt_2x}")
    m2 = _load_student(Path(args.ckpt_2x), 2).to(args.device)
    print(f"loading 4x student {args.ckpt_4x}")
    m4 = _load_student(Path(args.ckpt_4x), 4).to(args.device)
    print(f"loading val split from {args.val_dir}")
    val_ds = AnimePairDataset(args.val_dir, "val", scale=4,
                              max_files=args.val_batches * args.batch_size)
    dl = DataLoader(val_ds, batch_size=args.batch_size,
                    num_workers=0, shuffle=False)

    sum_bic, sum_4x, sum_casc = 0.0, 0.0, 0.0
    n = 0
    with torch.no_grad():
        for lr_n, hr_n in dl:
            lr = denorm01(lr_n).to(args.device)
            hr = denorm01(hr_n).to(args.device)
            # bicubic 4x baseline (sanity floor)
            bic = F.interpolate(lr, scale_factor=4, mode="bicubic",
                                align_corners=False).clamp(0, 1)
            # single 4x
            y4 = m4(lr).clamp(0, 1)
            if y4.shape[-2:] != hr.shape[-2:]:
                y4 = F.interpolate(y4, size=hr.shape[-2:], mode="bicubic",
                                    align_corners=False).clamp(0, 1)
            # cascade 2x + 2x (using the 2x model)
            y1 = m2(lr).clamp(0, 1)
            yc = m2(y1).clamp(0, 1)
            if yc.shape[-2:] != hr.shape[-2:]:
                yc = F.interpolate(yc, size=hr.shape[-2:], mode="bicubic",
                                    align_corners=False).clamp(0, 1)
            sum_bic += _psnr(bic, hr)
            sum_4x += _psnr(y4, hr)
            sum_casc += _psnr(yc, hr)
            n += 1
            if n >= args.val_batches:
                break
    if n == 0:
        print("ERROR: no val batches evaluated")
        return 1

    p_bic, p_4x, p_casc = sum_bic / n, sum_4x / n, sum_casc / n
    delta = p_4x - p_casc  # positive -> cascade is WORSE
    print()
    print(f"val batches      : {n}")
    print(f"bicubic 4x       : {p_bic:6.3f} dB  (sanity floor)")
    print(f"single 4x        : {p_4x:6.3f} dB  (production baseline)")
    print(f"cascade 2x2x     : {p_casc:6.3f} dB  (Phase 2 candidate)")
    print(f"delta vs single  : {delta:+.3f} dB  (positive = cascade LOSES)")
    print()
    if delta <= 0.3:
        print("OK: cascade within 0.3 dB tolerance")
        return 0
    else:
        print(f"REGRESSION: cascade loses {delta:.3f} dB vs single 4x (>0.3 dB)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
