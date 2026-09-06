#!/usr/bin/env python3
"""Phase 5#6 (MambaIRv2 pure-PyTorch) test-split PSNR + SSIM harness.

Mirrors scripts/eval_v3_ckpt.py but for the mambair arch. Loads
`student_best_ema.pt`, restores MambaIRv2Student, runs the FULL held-out
test split with animevideov3 teacher (same teacher the trainer used),
prints PSNR + SSIM of (bicubic, student, teacher) and writes a tiny results
markdown table.

Usage:
  .venv/Scripts/python.exe tmp/eval_mambair_test_psnr.py \
      --ckpt runs/distill_mambair_v1_10ep_v3/student_best_ema.pt \
      --split test
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(r"E:\python projects\upscale_anime")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "anime_upscaler"))

from student_mambair import MambaIRv2Student  # noqa: E402
from teacher import TEACHERS                   # noqa: E402
from torch.utils.data import DataLoader       # noqa: E402
import dataset as _ds                          # noqa: E402
AnimePairDataset = _ds.AnimePairDataset

# Reuse distill.py's metrics + denorm so we match what the trainer
# prints in train_log.csv (same ANIME convention: tensors in [-1, 1],
# PSNR/SSIM computed on [0, 1]). torchmetrics is NOT installed in this venv.
from distill import (  # noqa: E402
    set_seed, seed_worker, psnr01, ssim01, denorm01,
)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--data", default=str(REPO / "data" / "anime_video_frames"))
ap.add_argument("--batch-size", type=int, default=8)
ap.add_argument("--num-workers", type=int, default=0,
                help="0 = single-process (avoids Windows fork issues with this script).")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--split", default="test", choices=["val", "test"])
args = ap.parse_args()

set_seed(args.seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out_dir = Path(args.ckpt).resolve().parent
tag = Path(args.ckpt).stem

# ---- Dataset ----
ds = AnimePairDataset(args.data, args.split)
dl = DataLoader(
    ds, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=True,
    worker_init_fn=seed_worker,
    persistent_workers=args.num_workers > 0,
)

# ---- Teacher (must match what the trainer used) ----
# TEACHERS["animevideov3"] = (RealESRTeacher, {"ckpt_path": ...}); unpack the
# (Class, kwargs) tuple and pass device explicitly, mirroring distill.py:467.
cls, tkwargs = TEACHERS["animevideov3"]
teacher = cls(**tkwargs, device=DEVICE).to(DEVICE)
print(f"[teacher] {teacher.__class__.__name__}")

# ---- Student ----
state = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
ck_args = state.get("args", {}) or {}
student = MambaIRv2Student(
    num_in_ch=3, num_out_ch=3,
    embed_dim=ck_args.get("mambair_embed_dim", 48),
    num_blocks=ck_args.get("mambair_num_blocks", 8),
    d_state=ck_args.get("mambair_d_state", 16),
    scale=ck_args.get("scale", 4),
).to(DEVICE).eval()
missing, unexpected = student.load_state_dict(state["student"], strict=False)
if missing or unexpected:
    print(f"[load] missing={missing} unexpected={unexpected}", flush=True)
print(f"[student] arch=mambair params={sum(p.numel() for p in student.parameters()):,}")

psnr_student, ssim_student = [], []
psnr_bicubic, ssim_bicubic = [], []
psnr_teacher, ssim_teacher = [], []
teacher.eval()

# distill.py convention: AnimePairDataset returns normalised tensors (in
# [-1, 1]); call denorm01 to map back to [0, 1] before computing metrics.
n_done = 0
t0 = time.perf_counter()
with torch.no_grad():
    for lr_n, hr_n in dl:
        lr = denorm01(lr_n).to(DEVICE, non_blocking=True)
        hr = denorm01(hr_n).to(DEVICE, non_blocking=True)
        bicubic_hr = F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1)
        t_out = teacher(lr).clamp(0, 1)
        s_out = student(lr).clamp(0, 1)
        psnr_bicubic.append(psnr01(bicubic_hr, hr))
        psnr_student.append(psnr01(s_out, hr))
        psnr_teacher.append(psnr01(t_out, hr))
        ssim_bicubic.append(ssim01(bicubic_hr, hr))
        ssim_student.append(ssim01(s_out, hr))
        ssim_teacher.append(ssim01(t_out, hr))
        n_done += lr.size(0)
elapsed = time.perf_counter() - t0

psnr_b, psnr_s, psnr_t = np.mean(psnr_bicubic), np.mean(psnr_student), np.mean(psnr_teacher)
ssim_b, ssim_s, ssim_t = np.mean(ssim_bicubic), np.mean(ssim_student), np.mean(ssim_teacher)

gain_b = psnr_s - psnr_b
gain_t = psnr_t - psnr_b
retention = 100.0 * gain_b / gain_t if gain_t > 0 else float("nan")

lines = [
    f"| model | PSNR (dB) | SSIM |",
    f"|---|---:|---:|",
    f"| bicubic | {psnr_b:.2f} | {ssim_b:.4f} |",
    f"| student | {psnr_s:.2f} | {ssim_s:.4f} |",
    f"| teacher (animevideov3) | {psnr_t:.2f} | {ssim_t:.4f} |",
    f"| student - bicubic | {gain_b:+.2f} dB | |",
    f"| student retention of teacher gain | {retention:.1f}% | |",
    f"| n={n_done} batch={args.batch_size} elapsed={elapsed:.1f}s ({elapsed/n_done*1000:.0f}ms/sample) | | |",
]
table = "\n".join(lines)
print("\n=== EVAL " + args.split.upper() + f" (ckpt={tag}) ===\n" + table)

md_path = out_dir / f"eval_{tag}_{args.split}_results_table.md"
md_path.write_text(table + "\n")
print(f"[eval] wrote {md_path}")
