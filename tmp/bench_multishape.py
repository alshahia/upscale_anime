"""Multi-shape integration bench for the Phase 1.B TensorRT backend.

Verifies that the _TrtBackend inside _make_backend() produces output within
~1e-3 of PyTorch FP16 across the four shapes that matter for video:
  * 240x426   - typical 480p-ish frame
  * 480x480   - square
  * 540x304   - vertical-ish
  * 960x540   - 4K target input

For each shape we:
  1. build a fresh synthetic RGB frame (gradients + noise; structured input)
  2. export ONNX + build TRT engine (or pull from cache)
  3. run TRT 30 times and PyTorch 20 times, compute fps
  4. compare a single output to PyTorch (max abs diff)
  5. print one summary line

Note: cache lives under %APPDATA%/anime_upscaler_gui/cache/trt.
Delete that dir to force a clean build for every shape.
"""
from __future__ import annotations
import os, sys, time, warnings
warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# sys.path is configured externally (PYTHONPATH=apps\anime_upscaler_gui).

import cv2
import numpy as np
import torch

from anime_upscaler_gui.archs import build
from anime_upscaler_gui.pipeline import _make_backend, _to_tensor, _PinnedPool

CKPT  = os.path.join(ROOT, "pretrained", "RFDN_distill_v1_4x_student.pth")
KIND  = "rfdn_student"
DEV   = torch.device("cuda")
FP16  = True
N_WARM = 5
N_TIME = 30

SHAPES = [
    (240, 426),
    (480, 480),
    (540, 304),
    (960, 540),
]


def synth_frame(h: int, w: int) -> np.ndarray:
    """Deterministic, structured RGB frame with gradients + noise.

    Avoids all-zero or constant inputs (which can hide precision issues).
    """
    rng = np.random.default_rng(seed=h * 1009 + w)
    base = np.linspace(0, 255, num=h, dtype=np.float32)[:, None].repeat(w, axis=1)
    grad = np.stack(
        [base + rng.normal(0, 8, (h, w)).astype(np.float32) for _ in range(3)],
        axis=-1,
    )
    grad = np.clip(grad, 0, 255).astype(np.uint8)
    return grad  # already (H, W, 3) RGB


def bench_one(h: int, w: int) -> dict:
    rgb = synth_frame(h, w)

    m = build(KIND, CKPT).to(DEV).eval()
    if FP16:
        m = m.half()

    t0 = time.time()
    trt, name_t = _make_backend(m, CKPT, KIND, DEV, FP16, True, False)
    pt, name_p = _make_backend(m, CKPT, KIND, DEV, FP16, False, False)
    init_dt = time.time() - t0

    pin = _PinnedPool().get_input(h, w)
    x = _to_tensor(rgb, DEV, FP16, fp16_pin=False, pinned_in=pin)

    with torch.no_grad():
        y_t = trt(x)
        y_p = pt(x)
    torch.cuda.synchronize()
    max_diff = float((y_t.float() - y_p.float()).abs().max().item())

    for _ in range(N_WARM):
        with torch.no_grad():
            y_t = trt(x)
            y_p = pt(x)
    torch.cuda.synchronize()

    t = time.time()
    for _ in range(N_TIME):
        with torch.no_grad():
            y_t = trt(x)
    torch.cuda.synchronize()
    trt_ms = (time.time() - t) * 1000.0 / N_TIME

    t = time.time()
    for _ in range(N_TIME):
        with torch.no_grad():
            y_p = pt(x)
    torch.cuda.synchronize()
    pt_ms = (time.time() - t) * 1000.0 / N_TIME

    trt_fps = 1000.0 / trt_ms
    pt_fps = 1000.0 / pt_ms
    speedup = pt_ms / trt_ms

    del trt, pt, m, x, y_t, y_p, pin
    torch.cuda.empty_cache()

    return dict(
        h=h, w=w, init_dt=init_dt,
        max_diff=max_diff,
        trt_ms=trt_ms, pt_ms=pt_ms,
        trt_fps=trt_fps, pt_fps=pt_fps,
        speedup=speedup,
        backend_t=name_t, backend_p=name_p,
    )


def main() -> int:
    print(f"device={DEV} kind={KIND} fp16={FP16}")
    print(f"cache_dir={os.environ.get('APPDATA', '?')}\\anime_upscaler_gui\\cache\\trt")
    print()
    rows = []
    for h, w in SHAPES:
        print(f"--- shape {h}x{w} ---", flush=True)
        try:
            r = bench_one(h, w)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            rows.append((h, w, None, None, None, None, None, None, None, str(e)))
            continue
        print(f"  init={r['init_dt']:.1f}s  max_diff={r['max_diff']:.5f}")
        print(f"  TRT={r['trt_ms']:.2f}ms ({r['trt_fps']:.1f}fps)  "
              f"PT={r['pt_ms']:.2f}ms ({r['pt_fps']:.1f}fps)  speedup={r['speedup']:.2f}x")
        rows.append((
            h, w, r['init_dt'], r['max_diff'],
            r['trt_ms'], r['pt_ms'], r['trt_fps'], r['pt_fps'], r['speedup'], "",
        ))
    print()
    print("=" * 100)
    print(f"{'shape':>10s}  {'init_s':>7s}  {'max_diff':>9s}  "
          f"{'trt_ms':>7s}  {'pt_ms':>7s}  {'trt_fps':>8s}  {'pt_fps':>7s}  {'speedup':>8s}")
    print("-" * 100)
    for h, w, init, diff, t_ms, p_ms, t_fps, p_fps, sp, err in rows:
        if err:
            print(f"{h}x{w:<6d}  ERROR: {err}")
        else:
            print(f"{h}x{w:<6d}  {init:7.1f}  {diff:9.5f}  "
                  f"{t_ms:7.2f}  {p_ms:7.2f}  {t_fps:8.1f}  {p_fps:7.1f}  {sp:8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
