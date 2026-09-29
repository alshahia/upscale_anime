#!/usr/bin/env python3
"""Minimal real-video upscaler: animevideov3 on tmp/real_video_1sec.mp4.

Reads the 18-frame 1-second clip, runs each frame through realesr-animevideov3,
writes the upscaled video + middle frame PNG. Output goes to tmp/animevideov3_real/.

Phase 3 follow-up: gives the user a real-video artifact to compare against the
single-frame crop comparison from scripts/compare_students_vs_pretrained.py.
"""
import cv2
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "apps" / "anime_upscaler_gui"))

import torch  # noqa: E402
from anime_upscaler_gui.archs import build  # noqa: E402

INPUT  = REPO / "tmp" / "real_video_1sec.mp4"
OUTDIR = REPO / "tmp" / "animevideov3_real"
OUTDIR.mkdir(parents=True, exist_ok=True)
CKPT   = REPO / "pretrained" / "realesr-animevideov3.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    if not INPUT.exists():
        print(f"[FAIL] missing input: {INPUT}")
        return 1
    if not CKPT.exists():
        print(f"[FAIL] missing ckpt: {CKPT}")
        return 1

    print(f"[load] {CKPT.name}");
    model = build("srvgg", str(CKPT)).to(DEVICE).eval()
    print(f"[load] {sum(p.numel() for p in model.parameters()):,} params");

    cap = cv2.VideoCapture(str(INPUT))
    if not cap.isOpened():
        print(f"[FAIL] cv2 could not open {INPUT}");
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 18.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[in] {INPUT.name}: {w}x{h} @ {fps:.2f} fps, {n} frames");

    out_video = OUTDIR / "animevideov3.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w * 4, h * 4))

    t0 = time.perf_counter()
    n_done = 0
    infer_ms_total = 0.0
    frames_to_save = sorted({n // 2, 0, n - 1})  # middle, first, last
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE) / 255.0
        tT = time.perf_counter()
        with torch.no_grad():
            y = model(x).clamp(0, 1)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - tT) * 1000.0
        infer_ms_total += infer_ms
        sr = (y.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        sr_bgr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
        writer.write(sr_bgr)
        # Save sample frames
        if n_done in frames_to_save:
            tag = ["first", "middle", "last"][[0, n // 2, n - 1].index(n_done)] if n_done in [0, n // 2, n - 1] else str(n_done)
            p = OUTDIR / f"frame_{n_done:03d}_{tag}.png"
            cv2.imwrite(str(p), sr_bgr)
            print(f"  [frame {n_done}] saved {p.name} ({sr_bgr.shape[1]}x{sr_bgr.shape[0]})");
        n_done += 1

    cap.release()
    writer.release()
    dt = time.perf_counter() - t0
    print();
    print(f"[done] {n_done} frames in {dt:.2f}s -> {out_video.name}");
    print(f"       mean fps: {n_done/dt:.2f}, mean infer ms: {infer_ms_total/n_done:.1f}");
    print(f"       output:   {out_video}");
    return 0


if __name__ == "__main__":
    sys.exit(main())
