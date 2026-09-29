#!/usr/bin/env python3
"""One-off full-frame eval of the I1 student using the CORRECT nearest shortcut.

The canonical compare_students_vs_pretrained.py loads rfdn_student through
apps/anime_upscaler_gui/.../archs.py which is a vendored RFDN with mode="bicubic"
hardcoded. The I1 student was trained with mode="nearest" so loading it through
the vendored RFDN would evaluate with the WRONG shortcut and produce
misleading numbers.

This script loads the I1 student via anime_upscaler.student.RFDN with
shortcut_mode="nearest", then runs the same metric (cv2.Laplacian ksize=3 CV_32F)
on the same source frame (tmp/real_video_1sec.mp4 frame 8) so the numbers are
comparable to the canonical memory anchors (v1=21.0, animevideov3=58.1).
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from anime_upscaler.student import RFDN  # the real one (has shortcut_mode)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIDEO = REPO_ROOT / "tmp" / "real_video_1sec.mp4"
FRAME_IDX = 8
CKPT = REPO_ROOT / "runs" / "distill_i1_nearest_residual" / "student_best.pt"


def extract_frame(video, frame_idx):
    out = REPO_ROOT / "tmp" / "i1_eval_frame.png"
    if out.exists() and out.stat().st_size > 1000:
        return out
    import subprocess
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-i", str(video),
           "-vf", f"select=eq(n\\,{frame_idx})",
           "-vframes", "1", str(out)]
    subprocess.run(cmd, check=True, timeout=30)
    return out


def laplacian_variance(gray):
    f = np.float32(gray)
    lap = cv2.Laplacian(f, cv2.CV_32F, ksize=3)
    return float(lap.var())


def channel_stats(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return dict(
        mean=float(bgr.mean()),
        std=float(bgr.std()),
        lap_var=laplacian_variance(gray),
    )


def load_student(ckpt_path, shortcut_mode="nearest"):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # KD pipeline ckpt wraps student in dict; canonical archs loader also handles this
    if isinstance(sd, dict):
        for k in ("student", "params", "params_ema", "state_dict", "model_state_dict"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    m = RFDN(scale=4, shortcut_mode=shortcut_mode)
    m.load_state_dict(sd, strict=False)
    m.eval().to(DEVICE)
    return m


def upscale(model, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(t).clamp(0, 1)
    arr = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def main():
    if not VIDEO.exists():
        print(f"[FAIL] video not found: {VIDEO}")
        return
    if not CKPT.exists():
        print(f"[FAIL] I1 ckpt not found: {CKPT}")
        return
    out_dir = REPO_ROOT / "tmp" / "i1_correct_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_png = extract_frame(VIDEO, FRAME_IDX)
    bgr = cv2.imread(str(frame_png), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"[FAIL] could not read frame {frame_png}")
        return
    h, w = bgr.shape[:2]
    print(f"[src] {w}x{h}  ({VIDEO.name} frame {FRAME_IDX})")

    # --- bicubic baseline ---
    bicubic = cv2.resize(bgr, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
    bic_stats = channel_stats(bicubic)
    cv2.imwrite(str(out_dir / "bicubic_4x.png"), bicubic)
    print(f"[bicubic  ] lap_var={bic_stats['lap_var']:7.1f}  mean={bic_stats['mean']:.3f} std={bic_stats['std']:.3f}")

    # --- I1 student with CORRECT nearest shortcut ---
    t0 = time.perf_counter()
    model = load_student(CKPT, shortcut_mode="nearest")
    sr = upscale(model, bgr)
    dt_ms = (time.perf_counter() - t0) * 1000
    sr_stats = channel_stats(sr)
    cv2.imwrite(str(out_dir / "i1_nearest_4x.png"), sr)
    print(f"[I1 NEAREST  ] lap_var={sr_stats['lap_var']:7.1f}  mean={sr_stats['mean']:.3f} std={sr_stats['std']:.3f}  infer={dt_ms:6.1f}ms")

    # --- Sanity: same weights with BICUBIC shortcut to show the eval-time risk ---
    t0 = time.perf_counter()
    model_b = load_student(CKPT, shortcut_mode="bicubic")
    sr_b = upscale(model_b, bgr)
    dt_ms_b = (time.perf_counter() - t0) * 1000
    sr_b_stats = channel_stats(sr_b)
    cv2.imwrite(str(out_dir / "i1_bicubic_4x.png"), sr_b)
    print(f"[I1 BICUBIC-mismatch (sanity) ] lap_var={sr_b_stats['lap_var']:7.1f}  infer={dt_ms_b:6.1f}ms")

    # --- Save metrics CSV ---
    with open(out_dir / "metrics.csv", "w") as f:
        f.write("label,lap_var,mean,std,dt_ms\n")
        f.write(f"bicubic,{bic_stats['lap_var']:.1f},{bic_stats['mean']:.3f},{bic_stats['std']:.3f},\n")
        f.write(f"i1_nearest,{sr_stats['lap_var']:.1f},{sr_stats['mean']:.3f},{sr_stats['std']:.3f},{dt_ms:.1f}\n")
        f.write(f"i1_bicubic_mismatch,{sr_b_stats['lap_var']:.1f},{sr_b_stats['mean']:.3f},{sr_b_stats['std']:.3f},{dt_ms_b:.1f}\n")
    print(f"\n[done] outputs in {out_dir}/")


if __name__ == "__main__":
    main()
