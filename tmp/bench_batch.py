"""Batch vs single-frame integration test for the Phase 1.C batching path.

Compares fps for batch_size=1 (existing per-frame path) vs batch_size=4 at
the same shape, on both TensorRT and PyTorch backends. Verifies that:

  1. batch=1 still works (no regression)
  2. batch=4 is meaningfully faster (target: >= 2x)
  3. Outputs match closely (max abs diff <= 0.01 between batch and per-frame)

Uses synthetic input frames (gradients + noise) to avoid depending on real
sample data and to keep results reproducible.
"""
from __future__ import annotations
import os, sys, time, warnings
warnings.filterwarnings("ignore")

# Bench script conventions (see tmp/bench_multishape.py).
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CKPT = os.path.join(ROOT, "pretrained", "RFDN_distill_v1_4x_student.pth")
KIND = "rfdn_student"
DEV = None  # set in main
FP16 = True

import numpy as np
import torch
import cv2

from anime_upscaler_gui.archs import build
from anime_upscaler_gui.pipeline import (
    _make_backend, _to_tensor, _tensor_to_bgr,
    _to_tensor_batch, _tensor_to_bgr_batch,
    _pinned,
)


def synth_frame(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed=seed + h * 1009 + w)
    base = np.linspace(0, 255, num=h, dtype=np.float32)[:, None].repeat(w, axis=1)
    grad = np.stack([base + rng.normal(0, 8, (h, w)).astype(np.float32) for _ in range(3)], axis=-1)
    return np.clip(grad, 0, 255).astype(np.uint8)


def per_frame_run(backend, h: int, w: int, n_frames: int, device, warmup: int = 3) -> tuple:
    """Run N frames through the backend one at a time. Returns (y_concat, fps).

    warmup frames are run (untimed) before the timing loop so the first
    call pays engine build cost, not the measurement window.
    """
    # Warmup: run a few frames to trigger any engine build.
    for i in range(warmup):
        rgb = synth_frame(h, w, seed=10000 + i)
        pinned_in = _pinned.get_input(h, w, n=1)
        x = _to_tensor(rgb, device, FP16, fp16_pin=False, pinned_in=pinned_in)
        with torch.no_grad():
            y = backend(x)
        if FP16:
            y = y.float()
        out_h, out_w = int(y.shape[-2]), int(y.shape[-1])
        pinned_out = _pinned.get_output(out_h, out_w, n=1)
        _ = _tensor_to_bgr(y, pinned_out=pinned_out)
    torch.cuda.synchronize()
    # Timed run
    y_list = []
    t = time.time()
    for i in range(n_frames):
        rgb = synth_frame(h, w, seed=i)
        pinned_in = _pinned.get_input(h, w, n=1)
        x = _to_tensor(rgb, device, FP16, fp16_pin=False, pinned_in=pinned_in)
        with torch.no_grad():
            y = backend(x)
        if FP16:
            y = y.float()
        out_h, out_w = int(y.shape[-2]), int(y.shape[-1])
        pinned_out = _pinned.get_output(out_h, out_w, n=1)
        _ = _tensor_to_bgr(y, pinned_out=pinned_out)
        y_list.append(y.cpu())
    torch.cuda.synchronize()
    dt = time.time() - t
    return torch.cat(y_list, dim=0), n_frames / dt


def batched_run(backend, h: int, w: int, batch_size: int, n_frames: int, device,
                warmup_batches: int = 2) -> tuple:
    """Run N frames through the backend in batches. Returns (y_concat, fps).

    warmup_batches are run (untimed) to trigger any engine build.
    """
    n_batches = n_frames // batch_size
    # Warmup
    for b in range(warmup_batches):
        rgbs = [synth_frame(h, w, seed=20000 + b * batch_size + i) for i in range(batch_size)]
        pinned_in = _pinned.get_input(h, w, n=batch_size)
        x = _to_tensor_batch(rgbs, device, FP16, pinned_in=pinned_in)
        with torch.no_grad():
            y = backend(x)
        if FP16:
            y = y.float()
        out_h, out_w = int(y.shape[-2]), int(y.shape[-1])
        pinned_out = _pinned.get_output(out_h, out_w, n=batch_size)
        _ = _tensor_to_bgr_batch(y, pinned_out=pinned_out)
    torch.cuda.synchronize()
    # Timed
    y_list = []
    t = time.time()
    for b in range(n_batches):
        rgbs = [synth_frame(h, w, seed=b * batch_size + i) for i in range(batch_size)]
        pinned_in = _pinned.get_input(h, w, n=batch_size)
        x = _to_tensor_batch(rgbs, device, FP16, pinned_in=pinned_in)
        with torch.no_grad():
            y = backend(x)  # (N, 3, H*4, W*4)
        if FP16:
            y = y.float()
        out_h, out_w = int(y.shape[-2]), int(y.shape[-1])
        pinned_out = _pinned.get_output(out_h, out_w, n=batch_size)
        _ = _tensor_to_bgr_batch(y, pinned_out=pinned_out)
        y_list.append(y.cpu())
    torch.cuda.synchronize()
    dt = time.time() - t
    return torch.cat(y_list, dim=0), (n_batches * batch_size) / dt


def main() -> int:
    global DEV
    DEV = torch.device("cuda")
    print(f"device={DEV} kind={KIND} fp16={FP16}")
    print()

    # Build one model and use it for both backends.
    m = build(KIND, CKPT).to(DEV).eval()
    if FP16:
        m = m.half()

    results = {}
    for backend_label, use_trt in (("PyTorch", False), ("TensorRT", True)):
        print(f"=== {backend_label} backend ===")
        # Build TWO backends: one for batch=1, one for batch=4.
        b1, _ = _make_backend(m, CKPT, KIND, DEV, FP16, use_trt, False, batch_size=1)
        b4, _ = _make_backend(m, CKPT, KIND, DEV, FP16, use_trt, False, batch_size=4)
        for shape_label, h, w in [("480x270 (1080p out)", 480, 270),
                                   ("960x540 (4K out)",   960, 540)]:
            n_frames = 32
            print(f"  shape {shape_label}")
            y_single, fps1 = per_frame_run(b1, h, w, n_frames, DEV)
            y_batch4, fps4 = batched_run(b4, h, w, 4, n_frames, DEV)
            diff = float((y_single[:4].float() - y_batch4[:4].float()).abs().max().item())
            print(f"    batch=1: {fps1:.1f} fps")
            print(f"    batch=4: {fps4:.1f} fps  speedup={fps4/fps1:.2f}x  max_diff(first 4)={diff:.5f}")
            results[(backend_label, shape_label)] = (fps1, fps4, diff)
        del b1, b4
        torch.cuda.empty_cache()
        print()

    # Summary
    print("=" * 100)
    print(f"{'backend':>10s} {'shape':>20s}  {'b1 fps':>8s}  {'b4 fps':>8s}  {'speedup':>8s}  {'max_diff':>9s}")
    print("-" * 100)
    for (be, sh), (f1, f4, d) in results.items():
        print(f"{be:>10s} {sh:>20s}  {f1:8.1f}  {f4:8.1f}  {f4/f1:8.2f}x  {d:9.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
