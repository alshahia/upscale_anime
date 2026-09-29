#!/usr/bin/env python3
"""Latency sanity check for I1 vs I1-bicubic-mismatch."""
import sys, time
from pathlib import Path
import cv2, numpy as np, torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from anime_upscaler.student import RFDN

DEVICE = torch.device("cuda")
CKPT = REPO_ROOT / "runs" / "distill_i1_nearest_residual" / "student_best.pt"
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
if isinstance(sd, dict):
    for k in ("student", "params", "params_ema", "state_dict", "model_state_dict"):
        if k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break

# Test input: 854x480 LR -> 3416x1920 SR
lr = torch.randn(1, 3, 480, 854, device=DEVICE)

for mode in ("nearest", "bicubic"):
    m = RFDN(scale=4, shortcut_mode=mode).to(DEVICE).eval()
    m.load_state_dict(sd, strict=False)
    # Warmup
    with torch.no_grad():
        for _ in range(20):
            _ = m(lr)
    torch.cuda.synchronize()
    # Measure
    t0 = time.perf_counter()
    N = 30
    with torch.no_grad():
        for _ in range(N):
            _ = m(lr)
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000 / N
    print(f"mode={mode:8s}  mean latency: {dt_ms:6.1f} ms/frame @480x854 LR")
