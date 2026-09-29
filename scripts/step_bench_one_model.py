#!/usr/bin/env python3
"""Single-model worker for scripts/step_bench_all.py.

Runs in its own subprocess so VRAM stats (peak) and OOM isolation are per
model. Loads frames from lr_frames/, upscales with one kind/ckpt, writes
upscaled.mp4, frames.csv (per-frame timing) and row.csv (summary row)."""
import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "apps" / "anime_upscaler_gui"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import imageio  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from anime_upscaler_gui.archs import build  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kind", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--bench-dir", required=True,
                    help="parent bench dir (contains lr_frames + meta.json)")
    args = ap.parse_args()
    bench = Path(args.bench_dir)
    meta = json.loads((bench / "meta.json").read_text())
    out_dir = bench / "models" / args.model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    row_path = out_dir / "row.csv"

    torch.cuda.set_per_process_memory_fraction(0.9)
    torch.backends.cudnn.benchmark = True
    DEV = torch.device("cuda")

    frames = sorted((bench / "lr_frames").glob("*.png"))
    src_fps = meta["fps"]
    def fail(msg: str):
        with open(row_path, "w", newline="") as f:
            csv.writer(f).writerow(["status", "error", msg])
        print(f"[FAIL] {args.model_id}: {msg}", flush=True)

    try:
        t0 = time.time()
        model = build(args.kind, args.ckpt).to(DEV).eval()
        model = model.half()
        load_s = time.time() - t0
        n_params = sum(p.numel() for p in model.parameters())
    except Exception as e:
        fail(f"load failed: {e!r}")
        return 1
    print(f"[load] {args.model_id}: {n_params:,} params in {load_s:.1f}s", flush=True)

    recurrent = args.kind == "animesr"
    pad4 = args.kind == "animesr"   # MSRSWVSR needs H,W multiples of 4
    oh0 = ow0 = 0
    frames_t = []
    for idx_f, p in enumerate(frames):
        a = np.asarray(imageio.imread(p), np.float32) / 255.0
        t = torch.from_numpy(a).permute(2, 0, 1)[None].to(DEV).half().contiguous()
        if idx_f == 0:
            oh0, ow0 = t.shape[2] * 4, t.shape[3] * 4
        if pad4:
            ph, pw = (4 - t.shape[2] % 4) % 4, (4 - t.shape[3] % 4) % 4
            if ph or pw:
                t = F.pad(t, (0, pw, 0, ph), mode="replicate")
        frames_t.append(t)

    def run_one(i: int) -> torch.Tensor:
        t = frames_t[i]
        if recurrent:
            n = len(frames_t)
            prev = frames_t[max(i - 1, 0)]
            cur = frames_t[i]
            nxt = frames_t[min(i + 1, n - 1)]
            t = torch.stack([prev, cur, nxt], dim=1)              # (1, 3, 3, H, W) N-frame stack
        y = model(t)
        if recurrent:
            y = y[:, 1]                                           # center frame
        if pad4:
            y = y[..., :oh0, :ow0]
        return y

    # pass 1: warmup (cudnn autotune), untimed
    t0 = time.time()
    try:
        with torch.inference_mode():
            for i in list(range(min(4, len(frames_t)))):
                y = run_one(i)
            del y
            torch.cuda.synchronize()
        warm_s = time.time() - t0
    except torch.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        fail(f"OOM during warmup: {e!r}")
        return 1
    except Exception as e:
        torch.cuda.empty_cache()
        fail(f"warmup failed: {e!r}")
        return 1
    torch.cuda.reset_peak_memory_stats()

    ms = []
    outs = []
    infer_fail = None
    try:
        with torch.inference_mode():
            for i in range(len(frames_t)):
                t1 = time.perf_counter()
                y = run_one(i)
                torch.cuda.synchronize()
                ms.append((time.perf_counter() - t1) * 1000.0)
                a = (y.clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)
                outs.append(a)
                del y
    except torch.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        fail(f"OOM during inference at frame {i}: {e!r}")
        return 1
    except Exception as e:
        torch.cuda.empty_cache()
        infer_fail = repr(e)

    if infer_fail:
        fail(f"inference failed: {infer_fail}")
        return 1

    # encode (timed, excluded from fps-clock of inference)
    mp4 = out_dir / "upscaled.mp4"
    t0 = time.time()
    with imageio.get_writer(str(mp4), fps=src_fps, macro_block_size=8) as vw:
        for a in outs:
            vw.append_data(a)
    encode_s = time.time() - t0

    with open(out_dir / "frames.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "ms"])
        for i, m in enumerate(ms):
            wr.writerow([i, round(m, 2)])

    ms_arr = np.array(ms)
    fps_inst = 1000.0 / float(ms_arr.mean())
    fps_window = len(ms_arr) / (float(ms_arr.sum()) / 1000.0)
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    oh, ow = outs[0].shape[:2]

    # quality (IQA on 8 sampled output frames)
    iqas = {"clipiqa": [], "niqe": []}
    try:
        import pyiqa
        for name in ("clipiqa", "niqe"):
            m = pyiqa.create_metric(name, device=str(DEV))
            with torch.inference_mode():
                for idx in np.linspace(0, len(outs) - 1, 8).astype(int):
                    v = m(torch.from_numpy(outs[idx]).permute(2, 0, 1)[None].to(DEV).float() / 255.0)
                    iqas[name].append(float(v.item()))
            iqas[name + "_n"] = len(iqas[name])
    except Exception as e:
        iqas["error"] = repr(e)
        print(f"[warn] iqa failed: {e!r}", flush=True)

    row = {
        "model_id": args.model_id,
        "kind": args.kind,
        "ckpt": Path(args.ckpt).name,
        "scale": args.scale,
        "params": n_params,
        "status": "ok",
        "backend": "pytorch-fp16" + ("-recurrent3" if recurrent else ""),
        "device": "cuda-RTX4000-ac",
        "lr_in": f"{frames_t[0].shape[3]}x{frames_t[0].shape[2]}",
        "sr_out": f"{ow}x{oh}",
        "n_frames": len(frames_t),
        "load_s": round(load_s, 2),
        "warm_s": round(warm_s, 2),
        "infer_total_s": round(ms_arr.sum() / 1000.0, 2),
        "mean_ms": round(float(ms_arr.mean()), 2),
        "min_ms": round(float(ms_arr.min()), 2),
        "max_ms": round(float(ms_arr.max()), 2),
        "p95_ms": round(float(np.percentile(ms_arr, 95)), 2),
        "ms_std": round(float(ms_arr.std()), 2),
        "fps_1_over_mean": round(fps_inst, 2),
        "fps_throughput": round(fps_window, 2),
        "encode_s": round(encode_s, 2),
        "peak_vram_GB": round(peak_gb, 2),
        "clipiqa_mean": round(np.mean(iqas.get("clipiqa", [np.nan])), 2) if iqas.get("clipiqa") else "",
        "clipiqa_std": round(np.std(iqas.get("clipiqa", [np.nan])), 2) if iqas.get("clipiqa") else "",
        "niqe_mean": round(np.mean(iqas.get("niqe", [np.nan])), 2) if iqas.get("niqe") else "",
        "niqe_n": iqas.get("niqe_n", ""),
    }
    import math
    row = {k: ("" if (isinstance(v, float) and math.isnan(v)) else v) for k, v in row.items()}
    with open(row_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(row.keys()))
        wr.writeheader()
        wr.writerows([row])
    print(f"[done] {args.model_id}: {row['fps_throughput']} fps thr, "
          f"{row['mean_ms']} ms/frame, peak GB {row['peak_vram_GB']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
