import sys, time, warnings
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

# fp32 model (fp32 ONNX weights; TRT 11 picks kernels automatically)
MODEL = build("srvgg_student",
              str(ROOT / "pretrained" / "SRVGG_distill_v1_4x_student.pth")
              ).to(DEV).eval()
src = sorted((ROOT / "data" / "anime_fullframes").glob("*.jpg"))[7]
im = Image.open(src).convert("RGB")
lr_pil = im.resize((960, 540), Image.BICUBIC)
lr = torch.from_numpy(
    np.asarray(lr_pil, np.float32).transpose(2, 0, 1)[None] / 255.0
).to(DEV)

onnx_path = ROOT / "tmp" / "probe_fp32.onnx"
t0 = time.time()
with torch.inference_mode():
    torch.onnx.export(MODEL,
                      torch.zeros(1, 3, 540, 960, device=DEV),
                      str(onnx_path), opset_version=17,
                      input_names=["lr"], output_names=["hr"],
                      dynamo=False)
print(f"[onnx] {onnx_path.stat().st_size / 1e6:.2f} MB in {time.time() - t0:.1f}s", flush=True)

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network()
parser = trt.OnnxParser(network, logger)
with open(onnx_path, "rb") as f:
    ok = parser.parse(f.read())
if not ok:
    for i in range(parser.num_errors):
        print("parse err:", parser.get_error(i))
    sys.exit(1)
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
prof = builder.create_optimization_profile()
prof.set_shape("lr", (1, 3, 540, 960), (1, 3, 540, 960), (1, 3, 540, 960))
config.add_optimization_profile(prof)
t0 = time.time()
plan = builder.build_serialized_network(network, config)
assert plan is not None
engine_bytes = bytes(memoryview(plan).tobytes())
print(f"[trt] fp32-graph engine {len(engine_bytes) / 1e6:.1f} MB in {time.time() - t0:.1f}s", flush=True)

runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(engine_bytes)
ctx = engine.create_execution_context()
lrh = lr.half()
y = torch.empty(1, 3, 2160, 3840, device=DEV, dtype=torch.half)
stream = torch.cuda.Stream()
cur = torch.cuda.current_stream()
with torch.inference_mode():
    y_torch = MODEL(lr).float()
ctx.set_tensor_address("lr", lr.data_ptr())
ctx.set_tensor_address("hr", y.data_ptr())
with torch.cuda.stream(stream):
    ok2 = ctx.execute_async_v3(stream.cuda_stream)
cur.wait_stream(stream)
torch.cuda.synchronize()
d = (y.float() - y_torch).abs()
print(f"[parity] execute_ok={ok2} max={d.max().item():.5f} mean={d.mean().item():.6f}")
for _ in range(5):
    with torch.cuda.stream(stream):
        ctx.execute_async_v3(stream.cuda_stream)
    cur.wait_stream(stream)
    torch.cuda.synchronize()
t0 = time.perf_counter()
N = 30
for _ in range(N):
    with torch.cuda.stream(stream):
        ctx.execute_async_v3(stream.cuda_stream)
    cur.wait_stream(stream)
    torch.cuda.synchronize()
fps = N / (time.perf_counter() - t0)
print(f"[fps] trt fp32-graph static 4K: {fps:.1f} (target 25)")
