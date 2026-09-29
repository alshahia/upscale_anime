#!/usr/bin/env python3
"""Phase 4 I3 full-frame lap_var evaluation on the canonical harness frame.

Reuses tmp/eval_i2a_full_frame.py's pattern (already validated for I2a/I2b):
tmp/real_video_1sec.mp4 frame 8, 3416x1920 output, cv2.Laplacian ksize=3
CV_32F variance. The I3 ckpt is a TinySRVGGStudent (not RFDN), so the
student build path switches on `--arch`.

Usage:
  .venv/Scripts/python.exe tmp/eval_i3_full_frame.py \
      --ckpt runs/distill_i3_srvgg_body/student_best_ema.pt \
      --label i3_srvgg \
      --out-dir tmp/i3_eval
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, cv2, torch

sys.path.insert(0, r"E:\\python projects\\upscale_anime")
REPO = Path(r"E:\\python projects\\upscale_anime")
sys.path.insert(0, str(REPO / "anime_upscaler"))

from student import TinySRVGGStudent  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--video", default=str(REPO / "tmp" / "real_video_1sec.mp4"))
ap.add_argument("--frame", type=int, default=8)
ap.add_argument("--out-dir", default=str(REPO / "tmp" / "i3_eval"))
args = ap.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

# Load ckpt (distill.py saves {epoch, student, adapters, D, val_psnr, args})
state = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
arch = state.get("args", {}).get("arch", "rfdn")
if arch == "srvgg":
    student = TinySRVGGStudent(scale=4).to(DEVICE).eval()
else:
    # Default to RFDN for safety (no I3 RFDN ckpt expected).
    from student import RFDN
    sm = state.get("args", {}).get("shortcut_mode", "bicubic")
    student = RFDN(scale=4, shortcut_mode=sm).to(DEVICE).eval()
student.load_state_dict(state["student"])
print(f"[arch] {arch}  params: {sum(p.numel() for p in student.parameters()):,}")

# Read frame
cap = cv2.VideoCapture(args.video)
cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
ok, bgr = cap.read()
assert ok and bgr is not None
cap.release()
print(f"[lr] {bgr.shape}")
h, w = bgr.shape[:2]

# Bicubic baseline (for comparison)
bic = cv2.resize(bgr, (w*4, h*4), interpolation=cv2.INTER_CUBIC)

# Run student
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
# warmup
with torch.no_grad():
    for _ in range(5):
        _ = student(t).clamp(0, 1)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(30):
        sr = student(t).clamp(0, 1)
    torch.cuda.synchronize()
ms_per_frame = (time.perf_counter() - t0) / 30 * 1000.0

sr_np = (sr.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
sr_bgr = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)

# Metrics: lap_var on grayscale, per-frame
def lap_var(bgr_img):
    g = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(np.float32(g), cv2.CV_32F, ksize=3).var())

lap_bic = lap_var(bic)
lap_sr = lap_var(sr_bgr)
print(f"[{args.label}] full-frame lap_var: bicubic={lap_bic:.2f}  student={lap_sr:.2f}")
print(f"[{args.label}] latency: {ms_per_frame:.1f} ms/frame @ {w}x{h} LR (4x = {w*4}x{h*4})")

# Write outputs
cv2.imwrite(str(out_dir / f"{args.label}_bicubic_4x.png"), bic)
cv2.imwrite(str(out_dir / f"{args.label}_student_4x.png"), sr_bgr)
with open(out_dir / f"{args.label}_metrics.csv", "w") as f:
    f.write("metric,value\n")
    f.write(f"lap_var_bicubic,{lap_bic}\n")
    f.write(f"lap_var_student,{lap_sr}\n")
    f.write(f"ms_per_frame,{ms_per_frame}\n")
print(f"[{args.label}] wrote outputs to {out_dir}")
