#!/usr/bin/env python3
"""Phase 5#6 (MambaIRv2 pure-PyTorch) full-frame eval harness.

Mirrors tmp/eval_i3_full_frame.py (used for SRVGG I3) and
tmp/eval_i2a_full_frame.py, but for MambaIRv2Student. The state-space model
has its own arch switch in distill.py -- sniff arch from state["args"], build
the right student, run tmp/real_video_1sec.mp4 frame 8 at 4x, compute
lap_var + per-frame latency, write PNGs + metrics CSV.

Usage:
  .venv/Scripts/python.exe tmp/eval_mambair_full_frame.py \
      --ckpt runs/distill_mambair_v1_10ep_v3/student_best_ema.pt \
      --label mambair_v3 \
      --out-dir tmp/mambair_eval
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, cv2, torch

REPO = Path(r"E:\python projects\upscale_anime")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "anime_upscaler"))

# MambaIRv2Student lives at anime_upscaler/student_mambair.py and writes
# its state_dict under key "student" in distill.py's mambair build branch
# (line 487). Mirror distill's build path here.
from student_mambair import MambaIRv2Student  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--video", default=str(REPO / "tmp" / "real_video_1sec.mp4"))
ap.add_argument("--frame", type=int, default=8)
ap.add_argument("--out-dir", default=str(REPO / "tmp" / "mambair_eval"))
args = ap.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ---- Load checkpoint ----
state = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
ck_args = state.get("args", {}) or {}
arch = ck_args.get("arch", "mambair")
print(f"[arch] sniffed arch={arch} from ckpt args")

if arch == "mambair":
    student = MambaIRv2Student(
        num_in_ch=3, num_out_ch=3,
        embed_dim=ck_args.get("mambair_embed_dim", 48),
        num_blocks=ck_args.get("mambair_num_blocks", 8),
        d_state=ck_args.get("mambair_d_state", 16),
        scale=ck_args.get("scale", 4),
    ).to(DEVICE).eval()
else:
    raise SystemExit(f"unexpected arch in ckpt args: {arch!r} -- rerun with right ckpt")

missing, unexpected = student.load_state_dict(state["student"], strict=False)
if missing or unexpected:
    print(f"[load] missing={missing}  unexpected={unexpected}", flush=True)
print(f"[arch] params: {sum(p.numel() for p in student.parameters()):,}")

# ---- Disable gradient checkpointing for inference ----
# .eval() already gates it on self.training=True, but be defensive in case
# someone reuses this with .train() mode accidentally.
for blk in (student.body if hasattr(student, "body") else []):
    if hasattr(blk, "use_checkpoint"):
        blk.use_checkpoint = False

# ---- Read frame ----
cap = cv2.VideoCapture(args.video)
cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
ok, bgr = cap.read()
assert ok and bgr is not None, f"failed to read frame {args.frame} from {args.video}"
cap.release()
print(f"[lr] shape: {bgr.shape}")
h, w = bgr.shape[:2]

bic = cv2.resize(bgr, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)

# ---- Run student ----
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    for _ in range(5):
        _ = student(t).clamp(0, 1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        sr = student(t).clamp(0, 1)
    torch.cuda.synchronize()
ms_per_frame = (time.perf_counter() - t0) / 30 * 1000.0

sr_np = (sr.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
sr_bgr = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)


def lap_var(bgr_img):
    g = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(np.float32(g), cv2.CV_32F, ksize=3).var())


lap_bic = lap_var(bic)
lap_sr = lap_var(sr_bgr)
print(f"[{args.label}] full-frame lap_var: bicubic={lap_bic:.2f}  student={lap_sr:.2f}")
print(f"[{args.label}] latency: {ms_per_frame:.1f} ms/frame @ {w}x{h} LR (4x = {w*4}x{h*4})")

cv2.imwrite(str(out_dir / f"{args.label}_bicubic_4x.png"), bic)
cv2.imwrite(str(out_dir / f"{args.label}_student_4x.png"), sr_bgr)
with open(out_dir / f"{args.label}_metrics.csv", "w") as f:
    f.write("metric,value\n")
    f.write(f"lap_var_bicubic,{lap_bic}\n")
    f.write(f"lap_var_student,{lap_sr}\n")
    f.write(f"ms_per_frame,{ms_per_frame}\n")
    f.write(f"val_psnr_ema_from_ckpt,{state.get('val_psnr_ema', float('nan'))}\n")
print(f"[{args.label}] wrote outputs to {out_dir}")
