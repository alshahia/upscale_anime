"""Bounded latency experiment: TensorRT EP and torch.compile on the student.

Usage: .venv/Scripts/python.exe anime_upscaler/bench_deploy.py
Measures per-frame latency at 270x480 LR input (1080p output) on GPU.
"""
import sys
import time
import warnings

sys.path.insert(0, "anime_upscaler")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

LR_SHAPE = (1, 3, 270, 480)  # -> 1080 x 1920 SR output


def bench_ort_trt_fp16() -> None:
    try:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        t0 = time.perf_counter()
        sess = ort.InferenceSession(
            "anime_upscaler/export/student_fp16.onnx",
            sess_options=so,
            providers=[
                ("TensorrtExecutionProvider", {"trt_fp16_enable": "1"}),
                ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
                "CPUExecutionProvider",
            ],
        )
        build_s = time.perf_counter() - t0
        used = sess.get_providers()
        print(f"providers active: {used} (session build {build_s:.1f}s)")
        if "TensorrtExecutionProvider" not in used:
            print("ort-trt-fp16 : SKIPPED (TRT EP not active)")
            return
        name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name
        x = np.random.randn(*LR_SHAPE).astype(np.float16)
        for _ in range(20):
            sess.run([out_name], {name: x})
        t0 = time.perf_counter()
        for _ in range(50):
            sess.run([out_name], {name: x})
        dt = (time.perf_counter() - t0) / 50 * 1000
        print(f"ort-trt-fp16 : {dt:.2f} ms/frame")
    except Exception as exc:  # noqa: BLE001 - bounded experiment, report honestly
        print(f"ort-trt-fp16 : FAIL {type(exc).__name__}: {str(exc)[:200]}")


def bench_cudagraph_fp16() -> None:
    """Manual CUDA-graph replay of the eager fp16 model (no Triton needed)."""
    try:
        import torch

        from student import RFDN

        net = RFDN().half().cuda().eval()
        static_x = torch.randn(*LR_SHAPE, device="cuda", dtype=torch.float16)

        # warmup on a side stream (required before graph capture)
        s = torch.cuda.Stream()
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                net(static_x)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.no_grad():
            static_out = net(static_x)

        # eager fp16 baseline
        with torch.no_grad():
            for _ in range(10):
                net(static_x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(50):
                net(static_x)
            torch.cuda.synchronize()
        eager_ms = (time.perf_counter() - t0) / 50 * 1000

        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            g.replay()
        torch.cuda.synchronize()
        graph_ms = (time.perf_counter() - t0) / 200 * 1000

        assert torch.isfinite(static_out.float()).all(), "graph output not finite"
        print(f"eager-fp16   : {eager_ms:.2f} ms/frame")
        print(f"graph-fp16   : {graph_ms:.2f} ms/frame (manual CUDA graph)")

        # channels_last + cudnn autotune variant
        torch.backends.cudnn.benchmark = True
        net_cl = RFDN().half().cuda().eval().to(memory_format=torch.channels_last)
        xcl = static_x.to(memory_format=torch.channels_last)
        with torch.no_grad():
            for _ in range(15):
                net_cl(xcl)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(50):
                net_cl(xcl)
            torch.cuda.synchronize()
        cl_ms = (time.perf_counter() - t0) / 50 * 1000
        print(f"clast-fp16   : {cl_ms:.2f} ms/frame (channels_last+autotune)")
    except Exception as exc:  # noqa: BLE001
        print(f"cudagraph-fp16: FAIL {type(exc).__name__}: {str(exc)[:300]}")


def bench_ort_cuda_fp16() -> None:
    try:
        import onnxruntime as ort

        so = ort.SessionOptions()
        sess = ort.InferenceSession(
            "anime_upscaler/export/student_fp16.onnx",
            sess_options=so,
            providers=[("CUDAExecutionProvider", {}), "CPUExecutionProvider"],
        )
        used = sess.get_providers()
        if "CUDAExecutionProvider" not in used:
            print(f"ort-cuda-fp16 : SKIPPED (providers={used})")
            return
        name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name
        x = np.random.randn(*LR_SHAPE).astype(np.float16)  # fp16 model wants f16 io
        for _ in range(20):
            sess.run([out_name], {name: x})
        t0 = time.perf_counter()
        for _ in range(50):
            sess.run([out_name], {name: x})
        dt = (time.perf_counter() - t0) / 50 * 1000
        print(f"ort-cuda-fp16 : {dt:.2f} ms/frame")
    except Exception as exc:  # noqa: BLE001
        print(f"ort-cuda-fp16 : FAIL {type(exc).__name__}: {str(exc)[:200]}")


if __name__ == "__main__":
    bench_ort_cuda_fp16()
    bench_cudagraph_fp16()
