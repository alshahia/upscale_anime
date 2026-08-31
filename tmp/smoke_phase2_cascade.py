import sys
sys.path.insert(0, "anime_upscaler")
sys.path.insert(0, "apps")
sys.path.insert(0, "apps/anime_upscaler_gui")
import torch
from student import RFDN

# Cascade 2x with FRESH 2x weights: just check shapes
m2 = RFDN(scale=2).eval()
x = torch.randn(1, 3, 96, 96)
with torch.no_grad():
    y1 = m2(x); print(f"2x step1 (96->192): {tuple(y1.shape)}  expected (1,3,192,192)")
    y2 = m2(y1); print(f"2x step2 (192->384): {tuple(y2.shape)}  expected (1,3,384,384)")

# 4x single-shot from same LR
m4 = RFDN(scale=4).eval()
with torch.no_grad():
    y4 = m4(x); print(f"4x single (96->384): {tuple(y4.shape)}  expected (1,3,384,384)")

assert tuple(y2.shape) == tuple(y4.shape), f"cascade != single: {tuple(y2.shape)} vs {tuple(y4.shape)}"
print("CASCADE SHAPE SMOKE PASS")

# Now test the pipeline cascade helper
from anime_upscaler_gui.pipeline import _cascade_count
print(f"_cascade_count(m2={m2}) = {_cascade_count(m2)} (expected 2)")
print(f"_cascade_count(m4={m4}) = {_cascade_count(m4)} (expected 1)")
print("HELPER PASS")
