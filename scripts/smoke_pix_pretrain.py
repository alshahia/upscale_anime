"""Smoke test for the pretrain-taint fix.

Loads span_pix_pretrain_4x.pth via the trainer's load_neosr_weights path,
runs a forward pass on a reference image, and writes a smoke SR PNG.
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
warnings.filterwarnings("ignore")

import torch, cv2, numpy as np
from models.span import create_neosr_span

PRETRAIN = Path("pretrained/span_pix_pretrain_4x.pth")
REF_IMG = Path("data/test_mini_sr/2.png")
OUT_PNG = Path("results/audit/v4_smoke_pix_pretrain.png")

m = create_neosr_span({"type": "neosr_span", "scale": 4})
info = m.load_neosr_weights(str(PRETRAIN), strict=False)
print(f"[smoke] pretrain load: {info['loaded']}/{info['total_ckpt_keys']} loaded, "
      f"{info['skipped']} skipped, {info['mismatched']} mismatched")
m.eval()

img = cv2.imread(str(REF_IMG))
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
t = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
with torch.no_grad():
    out = m(t)
print(f"[smoke] output: shape={tuple(out.shape)} "
      f"min={out.min().item():.4f} max={out.max().item():.4f} "
      f"mean={out.mean().item():.4f} std={out.std().item():.4f}")

# Check the upsampler bias that was the taint signature
b = m.upsampler[0].bias
print(f"[smoke] upsampler.0.bias: mean={b.mean().item():.4f} std={b.std().item():.4f}")
TAINTED = (0.35 <= b.mean().item() <= 0.45) and (b.std().item() < 0.05)
print(f"[smoke] taint check: {'TAINTED' if TAINTED else 'CLEAN'}")

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
out_np = (out.squeeze(0).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
cv2.imwrite(str(OUT_PNG), out_bgr)
print(f"[smoke] saved: {OUT_PNG}")
print(f"[smoke] {'PASS' if not TAINTED else 'FAIL'}")
sys.exit(1 if TAINTED else 0)