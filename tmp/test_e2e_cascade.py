#!/usr/bin/env python3
"""E2E bench for cascade 2x+2x mode (Phase 2).

Mirrors tmp/test_e2e.py but loads the v2_2x student; cascade auto-engages
because the model has scale=2. We measure:
  * E2E fps on .venv/test_4k_540p_input.mp4 -> 3416x1920 output
  * Per-frame inference time
  * Output file size + NVENC CPU usage (best-effort)

Usage:
    .venv/Scripts/python.exe tmp/test_e2e_cascade.py
    .venv/Scripts/python.exe tmp/test_e2e_cascade.py --config trt_b1_nvenc_cascade
"""
import argparse
import json
import os
import queue
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "apps" / "anime_upscaler_gui"))

from anime_upscaler_gui.pipeline import _RunJob, _JobEvent, _PipelineWorker

# --- Configs --------------------------------------------------------------- #
# (label, batch_size, use_tensorrt, use_nvenc, tta, model_path, model_scale)
# model_scale determines cascade depth automatically via _cascade_count().
CONFIGS = {
    "trt_b1_nvenc_cascade": (1, True,  True,  False, "pretrained/RFDN_distill_v2_2x_student.pth", 2),
    "trt_b4_nvenc_cascade": (4, True,  True,  False, "pretrained/RFDN_distill_v2_2x_student.pth", 2),
    "pt_b1_nvenc_cascade":  (1, False, True,  False, "pretrained/RFDN_distill_v2_2x_student.pth", 2),
    "pt_b4_nvenc_cascade":  (4, False, True,  False, "pretrained/RFDN_distill_v2_2x_student.pth", 2),
}

INPUT  = PROJECT / ".venv" / "test_4k_540p_input.mp4"
OUTDIR = PROJECT / "tmp" / "e2e_out"
OUTDIR.mkdir(parents=True, exist_ok=True)


def run_job(label, batch_size, use_tensorrt, use_nvenc, tta,
            model_path, model_scale, reuse_cache=True):
    out_path = OUTDIR / f"{label}.mp4"
    if out_path.exists():
        out_path.unlink()
    if not INPUT.exists():
        print(f"  [{label}] FAIL: input missing ({INPUT})")
        return None
    if not Path(model_path).exists():
        print(f"  [{label}] FAIL: model missing ({model_path})")
        return None

    if not use_tensorrt and not reuse_cache:
        cache_root = Path(os.environ.get("APPDATA", "")) / "anime_upscaler_gui" / "cache" / "trt"
        if cache_root.exists():
            for f in cache_root.glob("*.engine"):
                f.unlink()

    job = _RunJob(
        job_id=1,
        input_path=INPUT,
        output_path=out_path,
        is_video=True,
        model_filename=str(model_path),
        kind="rfdn_student",
        scale=model_scale,           # 2 for cascade; pipeline auto-cascades
        outscale=4.0,                # user's output is always 4x
        fp16=True,
        device="cuda",
        batch_size=batch_size,
        decode="cv2",
        prefetch="sync",
        pin_memory="auto",
        downscale_max_edge=0,
        gpu_guard_mode="off",
        tile_size=0,
        tile_overlap=32,
        cut_start_seconds=0.0,
        cut_end_seconds=0.0,
        video_crf=18,
        video_preset="medium",
        tta=tta,
        use_tensorrt=use_tensorrt,
        use_nvenc=use_nvenc,
        nvenc_preset="p1",
        nvenc_qp=18,
        on_frame_error=lambda idx, msg: "abort",
    )

    in_q: "queue.Queue" = queue.Queue()
    out_q: "queue.Queue" = queue.Queue()
    worker = _PipelineWorker(in_q, out_q)
    worker.start()

    print(f"  [{label}] starting: batch={batch_size} TRT={use_tensorrt} NVENC={use_nvenc} TTA={tta} model_scale={model_scale}")
    t0 = time.time()
    in_q.put(job)

    events = []
    fps_samples = []
    infer_ms_samples = []
    n_frames = 0
    rc = None
    while True:
        try:
            evt: _JobEvent = out_q.get(timeout=300)
        except queue.Empty:
            print(f"  [{label}] FAIL: timed out")
            worker.stop(); worker.join(timeout=10)
            return None
        events.append({"kind": evt.kind, "msg": evt.message[:200],
                       "fps": evt.fps, "infer_ms": evt.infer_ms,
                       "frame_idx": evt.frame_idx})
        if evt.kind == "started":
            t_started = time.time()
        elif evt.kind == "progress":
            if evt.fps > 0:
                fps_samples.append(evt.fps)
            if evt.infer_ms > 0:
                infer_ms_samples.append(evt.infer_ms)
            n_frames = max(n_frames, evt.frame_idx + 1)
        elif evt.kind in ("finished", "error", "fatal_error"):
            t_ended = time.time()
            rc = evt.kind
            break

    wall_s = t_ended - t_started if 't_started' in dir() else (t_ended - t0)

    worker.stop(); worker.join(timeout=10)

    if rc != "finished":
        print(f"  [{label}] FAIL: worker ended with kind={rc}; last={events[-1]['msg'] if events else '<none>'}")
        return None

    fps_mean = sum(fps_samples) / len(fps_samples) if fps_samples else 0.0
    fps_max = max(fps_samples) if fps_samples else 0.0
    inf_mean = sum(infer_ms_samples) / len(infer_ms_samples) if infer_ms_samples else 0.0
    out_size = out_path.stat().st_size if out_path.exists() else 0

    result = {
        "label": label,
        "config": {
            "batch_size": batch_size, "use_tensorrt": use_tensorrt,
            "use_nvenc": use_nvenc, "tta": tta,
            "model_scale": model_scale, "model_path": str(model_path),
        },
        "wall_seconds": round(wall_s, 3),
        "frames_processed": n_frames,
        "fps_reported_mean": round(fps_mean, 2),
        "fps_reported_max":  round(fps_max, 2),
        "infer_ms_mean": round(inf_mean, 2),
        "output_size_bytes": out_size,
        "output_path": str(out_path),
    }
    print(f"  [{label}] OK: fps_mean={fps_mean:.2f} (peak {fps_max:.2f}), "
          f"infer_mean={inf_mean:.2f}ms, frames={n_frames}, output={out_size/1024:.1f}KB")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="trt_b1_nvenc_cascade",
                    help="comma-separated list of CONFIGS keys")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    configs = list(CONFIGS.keys()) if args.all else [c.strip() for c in args.config.split(",")]
    results = []
    for cfg in configs:
        if cfg not in CONFIGS:
            print(f"unknown config: {cfg}")
            continue
        # Reuse engine cache across cascade runs for fair comparison
        results.append(run_job(cfg, *CONFIGS[cfg], reuse_cache=True))

    summary_path = OUTDIR / "results_cascade.json"
    with open(summary_path, "w") as f:
        json.dump([r for r in results if r], f, indent=2)
    print(f"results -> {summary_path}")
