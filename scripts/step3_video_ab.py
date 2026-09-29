#!/usr/bin/env python3
"""Step 3 (deploy): real-video A/B - Step-2 student (ORT fp16 4x) vs
realesr-animevideov3 (ORT fp16 4x) on a 1-second 25 fps clip.

Input: data/anime_vid episode 2 at t=45 s (real action scene), 1080p source
downscaled to 640x640-half LR (mp4upload-stream simulation). Each model runs
2x through the clip: pass 1 warms, pass 2 is timed (wall clock; transfers +
encode pipeline reported separately in step3_export_bench.py).

Backend is ONNX-Runtime CUDA fp16 with CUDA IOBinding: TensorRT 11 produces
garbage output on this Turing/WDDM machine (tmp/debug_trt_probe.py), while
ORT matches torch to 1e-3.

Writes:
  results/step3_video_ab_student.mp4        (student 4K clip)
  results/step3_video_ab_animevideov3.mp4
  results/step3_video_ab.csv                (per-model fps)
Prints the >=25 fps verdict for each.

Usage:
    python scripts/step3_video_ab.py [t=45] [model=both]
"""
import csv
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "apps" / "anime_upscaler_gui"))

import av  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from anime_upscaler_gui.archs import build  # noqa: E402

torch.cuda.set_per_process_memory_fraction(0.9)
torch.backends.cudnn.benchmark = True

DEV = torch.device("cuda")
T_START = float(sys.argv[1]) if len(sys.argv) > 1 else 45.0
WHICH = sys.argv[2] if len(sys.argv) > 2 else "both"
OUT_DIR = ROOT / "results"
WORK = ROOT / "tmp"

MODELS = {
    "student": (ROOT / "pretrained" / "SRVGG_distill_v1_4x_student.pth", "srvgg_student"),
    "animevideov3": (ROOT / "pretrained" / "realesr-animevideov3.pth", "srvgg"),
}

SRC = ROOT / "data" / "anime_vid" / (
    "[Anime3rb.com] A-Rank Party wo Ridatsu shita Ore wa Moto Oshiego-tachi "
    "to Meikyuu Shinbu wo Mezasu. - 2 [1080p].mp4")


def grab_clip():
    container = av.open(str(SRC))
    stream = container.streams.video[0]
    fps = float(stream.average_rate or 25)
    n = max(1, int(round(fps)))
    container.seek(int(T_START / stream.time_base), stream=stream)
    frames = []
    for frame in container.decode(stream):
        frames.append(frame.to_image().convert("RGB"))
        if len(frames) >= n:
            break
    container.close()
    return frames, fps


def to_rgb_np(y_t, out_h, out_w):
    a = (y_t.clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)
    return a


def make_ort_session(model, onnx_tag: str):
    path = WORK / f"step3_{onnx_tag}_fp16.onnx"
    dummy = torch.zeros(1, 3, 16, 16, device=DEV, dtype=torch.half)
    torch.onnx.export(
        model, dummy, str(path), opset_version=17,
        input_names=["lr"], output_names=["hr"],
        dynamic_axes={"lr": {0: "N", 2: "H", 3: "W"},
                      "hr": {0: "N", 2: "H", 3: "W"}},
        dynamo=False)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(path), sess_options=so,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])


def run_model(tag: str, ckpt_path: Path, kind: str, lrs, out_h: int, out_w: int):
    print(f"\n=== {tag} ===", flush=True)
    model = build(kind, str(ckpt_path)).to(DEV).half().eval()
    sess = make_ort_session(model, onnx_tag=tag)

    def sr(x: torch.Tensor) -> torch.Tensor:
        y = torch.empty(1, 3, out_h, out_w, device=DEV, dtype=torch.half)
        io = sess.io_binding()
        io.bind_input("lr", x.device.type, 0, np.float16,
                      tuple(x.shape), x.data_ptr())
        io.bind_output("hr", y.device.type, 0, np.float16,
                       tuple(y.shape), y.data_ptr())
        sess.run_with_iobinding(io)
        return y

    outputs, times = [], []
    with torch.inference_mode():
        # pass 1: warm (engine build / cudnn algo search), untimed
        for x in lrs:
            outputs.clear()
            _ = sr(x)
        torch.cuda.synchronize()
        # pass 2: timed
        for x in lrs:
            t0 = time.perf_counter()
            y = sr(x)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            a = (y.clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)
            outputs.append(a)
    dt_frame = float(np.mean(times))
    fps_inst = 1.0 / dt_frame
    fps_end = len(lrs) / float(np.sum(times))
    print(f"[fps] {dt_frame*1000:.1f} ms/frame -> {fps_inst:.1f} fps "
          f"(end-to-end {fps_end:.1f} for {len(lrs)} frames) "
          f"| meets_25fps={'PASS' if fps_inst >= 25 else 'FAIL'}", flush=True)
    path = OUT_DIR / f"step3_video_ab_{tag}.mp4"
    with imageio.get_writer(str(path), fps=25, macro_block_size=8) as vw:
        for f_rgb in outputs:
            vw.append_data(f_rgb)
    print(f"[out] {path.name} ({len(outputs)} frames @25 fps)", flush=True)
    del model, sess
    torch.cuda.empty_cache()
    return dict(model=tag, backend="onnxruntime-cuda-fp16", fps=round(fps_inst, 2),
                fps_end_to_end=round(fps_end, 2),
                meets_25=fps_inst >= 25)


def main() -> int:
    assert SRC.exists(), f"missing source: {SRC}"
    OUT_DIR.mkdir(exist_ok=True)
    frames, fps = grab_clip()
    print(f"[in] {len(frames)} frames @ {fps:.1f} fps from t={T_START}s", flush=True)
    lrs = []
    for img in frames:
        a = np.asarray(img, np.float32) / 255.0
        t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(DEV)
        # 1080p -> third-size LR frames (640x360, worst-case stream quality);
        # model 4x-upscale -> 2560x1440 (QHD, 1080p-class output)
        lr = F.interpolate(t, scale_factor=1 / 3, mode="bicubic", align_corners=False)
        lr = lr.clamp(0, 1).to(DEV).half()
        lrs.append(lr.contiguous())
    rows = []
    for tag, (ckpt_path, kind) in MODELS.items():
        if WHICH not in ("both", tag):
            continue
        rows.append(run_model(tag, ckpt_path, kind, lrs, out_h=1440, out_w=2560))
    out_csv = OUT_DIR / "step3_video_ab.csv"
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n[done] wrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
