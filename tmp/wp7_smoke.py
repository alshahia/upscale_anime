"""WP-7 smoke tests: build_student arch detection + InferenceEngine fixes."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # for anime_upscaler
sys.path.insert(0, str(ROOT / "src"))  # for models, utils, inference

import numpy as np
import torch

from anime_upscaler.student import RFDN, TinySRVGGStudent, build_student

print("=== build_student tests ===")

# 1. RFDN full checkpoint with args
rfdn = RFDN(scale=4, shortcut_mode="bicubic")
ckpt = {"epoch": 10, "student": rfdn.state_dict(), "val_psnr": 30.0,
        "args": {"arch": "rfdn", "scale": 4, "shortcut_mode": "bicubic"}}
m = build_student(ckpt)
assert isinstance(m, RFDN), type(m)
m.load_state_dict(ckpt["student"])
print("1. RFDN via args arch: OK", type(m).__name__)

# 2. TinySRVGGStudent full checkpoint with args
srvgg = TinySRVGGStudent(num_feat=52, num_conv=12, scale=4)
ckpt2 = {"epoch": 5, "student": srvgg.state_dict(), "val_psnr": 28.0,
         "args": {"arch": "srvgg", "scale": 4}}
m2 = build_student(ckpt2)
assert isinstance(m2, TinySRVGGStudent), type(m2)
m2.load_state_dict(ckpt2["student"])
print("2. SRVGG via args arch: OK", type(m2).__name__)

# 3. bare RFDN state_dict (structural sniff)
m3 = build_student(rfdn.state_dict())
assert isinstance(m3, RFDN), type(m3)
print("3. RFDN via key sniff: OK")

# 4. bare SRVGG state_dict (structural sniff)
m4 = build_student(srvgg.state_dict())
assert isinstance(m4, TinySRVGGStudent), type(m4)
print("4. SRVGG via key sniff: OK")

# 5. explicit arch override
m5 = build_student(ckpt, arch="srvgg")
assert isinstance(m5, TinySRVGGStudent), type(m5)
print("5. explicit arch override: OK")

# 6. unknown arch raises
try:
    build_student(ckpt, arch="mambair")
    raise AssertionError("should have raised")
except ValueError:
    print("6. unknown arch raises ValueError: OK")

# 7. scale sniffing: RFDN scale=2
rfdn2 = RFDN(scale=2)
m7 = build_student(rfdn2.state_dict())
assert isinstance(m7, RFDN) and m7.scale == 2, m7.scale
print("7. RFDN scale=2 sniffed: OK")

# 8. SRVGG scale=2
srvgg2 = TinySRVGGStudent(scale=2)
m8 = build_student(srvgg2.state_dict())
assert isinstance(m8, TinySRVGGStudent) and m8.scale == 2
print("8. SRVGG scale=2 sniffed: OK")

# 9. shortcut_mode sniffing from args
rfdn_near = RFDN(shortcut_mode="nearest")
ckpt9 = {"student": rfdn_near.state_dict(),
         "args": {"arch": "rfdn", "shortcut_mode": "nearest"}}
m9 = build_student(ckpt9)
assert m9.shortcut_mode == "nearest"
print("9. shortcut_mode sniffed from args: OK")

# 10. num_feat/num_conv sniffing
srvgg_custom = TinySRVGGStudent(num_feat=32, num_conv=6, scale=4)
m10 = build_student(srvgg_custom.state_dict())
assert m10.num_feat == 32 and m10.num_conv == 6, (m10.num_feat, m10.num_conv)
print("10. num_feat/num_conv sniffed: OK")

print()
print("=== InferenceEngine tests ===")
from inference.engine import InferenceEngine

# 11. device='cpu' respected
eng = InferenceEngine(TinySRVGGStudent(), device="cpu")
assert eng.device.type == "cpu", eng.device
assert eng.use_amp is False
print("11. device='cpu' respected: OK", eng.device)

# 12. run on CPU
lr = np.random.rand(32, 32, 3).astype(np.float32)
sr = eng.run(lr)
assert sr.shape == (128, 128, 3), sr.shape
print("12. run() on CPU: OK", sr.shape)

# 13. run_batch same-shape images
imgs = [np.random.rand(32, 32, 3).astype(np.float32) for _ in range(3)]
srs = eng.run_batch(imgs)
assert len(srs) == 3 and all(s.shape == (128, 128, 3) for s in srs)
print("13. run_batch 3 same-shape: OK")

# 14. run_batch mixed shapes, order preserved (H, W order!)
mixed = [np.random.rand(32, 32, 3).astype(np.float32),
         np.random.rand(48, 24, 3).astype(np.float32),
         np.random.rand(32, 32, 3).astype(np.float32)]
srs_m = eng.run_batch(mixed)
assert srs_m[0].shape == (128, 128, 3), srs_m[0].shape
assert srs_m[1].shape == (192, 96, 3), srs_m[1].shape
assert srs_m[2].shape == (128, 128, 3), srs_m[2].shape
print("14. run_batch mixed shapes, order preserved: OK")

# 15. run_batch empty
assert eng.run_batch([]) == []
print("15. run_batch empty: OK")

# 16. benchmark scale from model
res = eng.benchmark(input_size=(3, 32, 32), num_runs=3, warmup=1)
assert res["output_resolution"] == "128x128", res["output_resolution"]
print("16. benchmark scale from model: OK", res["output_resolution"])

# 17. benchmark explicit scale
res17 = eng.benchmark(input_size=(3, 32, 32), num_runs=3, warmup=1, scale=2)
assert res17["output_resolution"] == "64x64", res17["output_resolution"]
print("17. benchmark explicit scale: OK", res17["output_resolution"])

# 18. from_ensemble signature: weights default is None
import inspect
sig = inspect.signature(InferenceEngine.from_ensemble)
assert sig.parameters["weights"].default is None
print("18. from_ensemble weights default None: OK")

# 19. run_tta on CPU
sr_tta = eng.run_tta(lr)
assert sr_tta.shape == (128, 128, 3)
print("19. run_tta on CPU: OK")

print()
print("ALL SMOKE TESTS PASSED")
