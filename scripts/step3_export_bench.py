#!/usr/bin/env python3
"""Step 3 (deploy): benchmark the Step-2 SRVGG student at deployment speed.

Backends compared (validated in tmp/debug_trt_probe*.py):
  * ORT-CUDA: fp16 ONNX (dynamic H/W, opset 17) via onnxruntime
    CUDAExecutionProvider with CUDA IOBinding (no per-frame host transfers).
  * torch eager fp16 + channels_last (memory-format optimized reference).

TensorRT 11.2's new compiler backend is BROKEN on this setup
(Turing/WDDM): every engine variant (fp16 weights static, fp16 weights
dynamic, fp32 weights static) produces garbage or NaN output; see
tmp/debug_trt_probe.py (max|diff|=1.26, mean|diff|=0.33) and
tmp/debug_trt_probe_fp32.py (NaN). ORT output matches torch fp16 eager to
9.8e-4 max diff, so ORT is the deployment backend.

For each shape: parity check, then 20 warmup + 60 timed iters, writes
results/step3_deploy_bench.csv.
"""
import csv
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "apps" / "anime_upscaler_gui"))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from anime_upscaler_gui.archs import build  # noqa: E402

CKPT = ROOT / "pretrained" / "SRVGG_distill_v1_4x_student.pth"
KIND = "srvgg_student"
WORK = ROOT / "tmp"
OUT_CSV = ROOT / "results" / "step3_deploy_bench.csv"
DEV = torch.device("cuda")
N_WARM = 20
N_TIME = 60

torch.cuda.set_per_process_memory_fraction(0.9)
torch.backends.cudnn.benchmark = True


def parse_shapes(s: str):
    out = []
    for part in s.split(","):
        h, w = part.strip().lower().split("x")
        out.append((int(h), int(w)))
    return out


def real_frame(h: int, w: int) -> np.ndarray:
    """Real anime LR frame (bicubic-halved source frame), (H, W, 3) uint8."""
    srcs = sorted((ROOT / "data" / "anime_fullframes").glob("*.jpg")) or \
           sorted((ROOT / "data" / "anime_fullframes").glob("*.png"))
    im = Image.open(srcs[7]).convert("RGB")
    hr = np.asarray(im)
    hi, wi = hr.shape[:2]
    y0 = max((hi - 4 * h) // 2, 0)
    x0 = max((wi - 4 * w) // 2, 0)
    crop = hr[y0:y0 + 4 * h, x0:x0 + 4 * w]
    pil = Image.fromarray(crop).resize((w, h), Image.BICUBIC)
    return np.asarray(pil)


def export_onnx(net16, shapes):
    """One fp16 ONNX with dynamic H/W covers every bench shape."""
    path = WORK / "step3_student_dyn_fp16.onnx"
    torch.onnx.export(
        net16,
        torch.zeros(1, 3, 16, 16, device=DEV, dtype=torch.half),
        str(path), opset_version=17,
        input_names=["lr"], output_names=["hr"],
        dynamic_axes={"lr": {2: "H", 3: "W"}, "hr": {2: "H", 3: "W"}},
        dynamo=False)
    return path


def make_ort_session(onnx_path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        str(onnx_path), sess_options=so,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    return sess


class OrtBackend:
    """ORT-CUDA backend with CUDA IOBinding (GPU-to-GPU, no host round trips)."""

    def __init__(self, sess):
        self.sess = sess

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.empty(
            1, 3, x.shape[2] * 4, x.shape[3] * 4,
            device=x.device, dtype=x.dtype)
        io = self.sess.io_binding()
        io.bind_input("lr", x.device.type, x.device.index or 0, np.float16,
                      tuple(x.shape), x.data_ptr())
        io.bind_output("hr", y.device.type, y.device.index or 0, np.float16,
                       tuple(y.shape), y.data_ptr())
        self.sess.run_with_iobinding(io)
        return y


def fps_of(fn, warmup=N_WARM, iters=N_TIME) -> float:
    for _ in range(warmup):
        with torch.inference_mode():
            fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.inference_mode():
            fn()
    torch.cuda.synchronize()
    return iters / (time.perf_counter() - t0)


def main() -> int:
    shapes = parse_shapes(sys.argv[sys.argv.index("--shapes") + 1]) \
        if "--shapes" in sys.argv else [(540, 960), (360, 640)]

    assert CKPT.exists(), f"missing ckpt: {CKPT}"
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(exist_ok=True)

    net16 = build(KIND, str(CKPT)).to(DEV).half().eval()
    print(f"[load] {sum(p.numel() for p in net16.parameters()):,} params", flush=True)
    sess = make_ort_session(export_onnx(net16, shapes))
    ort_backend = OrtBackend(sess)

    rows = []
    for (h, w) in shapes:
        print(f"\n=== shape {h}x{w} (LR) -> {h*4}x{w*4} (SR) ===", flush=True)
        model_cl = net16.to(memory_format=torch.channels_last)

        x_np = np.ascontiguousarray(
            real_frame(h, w).astype(np.float32).transpose(2, 0, 1)[None] / 255.0)
        x = torch.from_numpy(x_np).half().to(DEV)
        x_cl = x.to(memory_format=torch.channels_last)

        with torch.inference_mode():
            y_ort = ort_backend(x).float()
            y_eager = model_cl(x_cl).float()
        torch.cuda.synchronize()
        max_diff = float((y_ort - y_eager).abs().max().item())
        print(f"[parity] max|ort-eager|={max_diff:.5f}", flush=True)

        ort_fps = fps_of(lambda: ort_backend(x))
        eager_fps = fps_of(lambda: model_cl(x_cl))
        print(f"[fps] ort_cuda={ort_fps:.1f}  eager_fp16_CL={eager_fps:.1f}  "
              f"target_25={'PASS' if ort_fps >= 25 else 'FAIL'}", flush=True)

        rows.append(dict(shape=f"{h}x{w}", sr_out=f"{h*4}x{w*4}",
                         ort_ms=round(1000.0 / ort_fps, 2),
                         ort_fps=round(ort_fps, 2), eager_fps=round(eager_fps, 2),
                         max_abs_diff=round(max_diff, 5),
                         meets_25=ort_fps >= 25))

        del x, x_cl, y_ort, y_eager
        torch.cuda.empty_cache()

    with open(OUT_CSV, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n[done] wrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
