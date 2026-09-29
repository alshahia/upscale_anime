import sys, warnings
warnings.filterwarnings("ignore")
import torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

for name in ["stylemix_a0.50.pth", "stylemix_a0.00.pth", "stylemix_a1.00.pth"]:
    sd = torch.load(str(ROOT/"results/style_mix/weights"/name), map_location="cpu", weights_only=False)
    sd = sd.get("params", sd)
    n_bad = sum(1 for k, v in sd.items() if torch.isnan(v.float()).any() or torch.isinf(v.float()).any())
    print(name, "tensors with NaN/inf:", n_bad, "total", len(sd))

# also check source ckpts
import os
for name in ["realesr-animevideov3.pth", "4xLSDIRCompactv2.pth"]:
    sd = torch.load(str(ROOT/"pretrained"/name), map_location="cpu", weights_only=False)
    sd = sd.get("params", sd)
    n_bad = sum(1 for k, v in sd.items() if torch.isnan(v.float()).any())
    print(name, "tensors with NaN:", n_bad, "total", len(sd))
