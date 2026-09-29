#!/usr/bin/env python3
"""Controlled TRT probe: fp16 ONNX export -> static 960x540 engine ->
parity vs torch fp16 eager + fps. Bypasses the GUI cache entirely."""
import sys
import time
import warnings

warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "apps" / "anime_upscaler_gui"))

import numpy as np
import torch
import tensorrt as trt
from PIL import Image

from anime_upscaler_gui.archs import build

torch.cuda.set_per_process_memory_fraction(0.9)
torch.backends.cudnn.benchmark = True
DEV = torch.device("cuda")

MODEL = build(
    "srvgg_student",
    str(ROOT / "pretrained" / "SRVGG_distill_v1_4x_student.pth"),
).to(DEV).half().eval()
print("[load]", sum(p.numel() for p in MODEL.parameters()), "params")

# --- real 960x540 LR frame ---
src = sorted((ROOT / "data" / "anime_fullframes").glob("*.jpg"))[7]
im = Image.open(src).convert("RGB")
lr_pil = im.resize((960, 540), Image.BICUBIC)
lr = torch.from_numpy(
    np.asarray(lr_pil, np.float32).transpose(2, 0, 1)[None] / 255.0
).half().to(DEV)

# --- 1) ONNX export (fp16 graph) ---
onnx_path = ROOT / "tmp" / "probe_fp16.onnx"
t0 = time.time()
with torch.inference_mode():
    torch.onnx.export(
        MODEL,
        torch.zeros(1, 3, 540, 960, device=DEV, dtype=torch.half),
        str(onnx_path),
        opset_version=17,
        input_names=["lr"],
        output_names=["hr"],
        dynamo=False,
    )
print(f"[onnx] exported in {time.time() - t0:.1f}s, "
      f"{onnx_path.stat().st_size / 1e6:.2f} MB")

# --- 2) build static engine ---
t0 = time.time()
logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network()  # TRT 11: explicit batch is the only mode
parser = trt.OnnxParser(network, logger)
with open(onnx_path, "rb") as f:
    ok = parser.parse(f.read())
if not ok:
    for i in range(parser.num_errors):
        print("parse err:", parser.get_error(i))
    sys.exit(1)
print(f"[trt] parsed; inputs={network.num_inputs} outputs={network.num_outputs};",
      f"in_dtype={network.get_input(0).dtype} out_dtype={network.get_output(0).dtype}")
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
prof = builder.create_optimization_profile()
prof.set_shape("lr", (1, 3, 540, 960), (1, 3, 540, 960), (1, 3, 540, 960))
config.add_optimization_profile(prof)
plan = builder.build_serialized_network(network, config)
assert plan is not None, "engine build failed"
engine_bytes = bytes(memoryview(plan).tobytes())
print(f"[trt] built in {time.time() - t0:.1f}s, {len(engine_bytes) / 1e6:.1f} MB")

runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(engine_bytes)
ctx = engine.create_execution_context()
ctx.set_input_shape("lr", (1, 3, 540, 960))
for i in range(engine.num_io_tensors):
    nm = engine.get_tensor_name(i)
    print("  tensor", nm, engine.get_tensor_dtype(nm), engine.get_tensor_mode(nm))

y = torch.empty(1, 3, 2160, 3840, device=DEV, dtype=torch.half)
stream = torch.cuda.Stream()
cur = torch.cuda.current_stream()

# --- 3) parity vs torch fp16 eager ---
with torch.inference_mode():
    y_torch = MODEL(lr).float()
ctx.set_tensor_address("lr", lr.data_ptr())
ctx.set_tensor_address("hr", y.data_ptr())
with torch.cuda.stream(stream):
    assert ctx.execute_async_v3(stream.cuda_stream)
cur.wait_stream(stream)
torch.cuda.synchronize()
d = (y.float() - y_torch).abs()
print(f"[parity] max={d.max().item():.5f} mean={d.mean().item():.6f}")

# --- 4) fps ---
def run_once():
    ctx.set_tensor_address("lr", lr.data_ptr())
    ctx.set_tensor_address("hr", y.data_ptr())
    with torch.cuda.stream(stream):
        ctx.execute_async_v3(stream.cuda_stream)
    cur.wait_stream(stream)
    torch.cuda.synchronize()

for _ in range(5):
    run_once()
t0 = time.perf_counter()
N = 30
for _ in range(N):
    run_once()
fps = N / (time.perf_counter() - t0)
print(f"[fps] trt static 960x540 -> 4K: {fps:.1f} (target 25)")

