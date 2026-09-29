#!/usr/bin/env python3
"""Minimal real-video upscaler: v3 student on tmp/real_video_1sec.mp4.

Same shape as test_animevideov3_real_video.py but loads the v3 student from
runs/distill_v3_4x_v3_epoch18_ema_unpromoted.pth (the un-promoted Phase 3 ckpt).
Useful for visual comparison against animevideov3 on the same 1s clip.
"""
import cv2, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "apps" / "anime_upscaler_gui"))
sys.path.insert(0, str(REPO / "anime_upscaler"))

import torch  # noqa: E402
from anime_upscaler_gui.archs import build  # noqa: E402

INPUT  = REPO / "tmp" / "real_video_1sec.mp4"
OUTDIR = REPO / "tmp" / "v3_student_real"
OUTDIR.mkdir(parents=True, exist_ok=True)
CKPT   = REPO / "runs" / "distill_v3_4x_v3_epoch18_ema_unpromoted.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    if not INPUT.exists():
        print(f"[FAIL] missing input: {INPUT}")
        return 1
    if not CKPT.exists():
        print(f"[FAIL] missing v3 ckpt: {CKPT}")
        return 1

    print(f"[load] {CKPT.name}");
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["student"] if isinstance(ck, dict) and "student" in ck else ck
    model = build("rfdn_student", str(CKPT)).to(DEVICE).eval()
    print(f"[load] {sum(p.numel() for p in model.parameters()):,} params");

    cap = cv2.VideoCapture(str(INPUT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 18.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[in] {INPUT.name}: {w}x{h} @ {fps:.2f} fps, {n} frames");

    out_video = OUTDIR / "v3_student.mp4"
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 4, h * 4))
    t0 = time.perf_counter()
    n_done = 0
    infer_ms_total = 0.0
    save_idx = sorted({n // 2, 0, n - 1})
    save_tags = {0: "first", n // 2: "middle", n - 1: "last"}
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
        infer_ms_total += (time.perf_counter() - tT) * 1000.0
        sr = (y.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        sr_bgr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
        writer.write(sr_bgr)
        if n_done in save_idx:
            tag = save_tags.get(n_done, str(n_done))
            p = OUTDIR / f"frame_{n_done:03d}_{tag}.png"
            cv2.imwrite(str(p), sr_bgr)
            print(f"  [frame {n_done}] saved {p.name}");
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
