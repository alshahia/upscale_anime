"""
P1.E2E end-to-end harness.

Drives PipelineWorker via queue.Queue, measures end-to-end fps for a video
job. Supports multi-config comparison: TRT vs PyTorch, batch=1 vs batch=4,
NVENC vs libx264.

Usage:
    python tmp/test_e2e.py [--config CONFIG] [--reuse-cache] [--engine-shape HW]

Reports elapsed wall time, fps (frames / wall), output file size, and CPU
snapshots. Designed for regression: same input, same model across configs.
"""
import argparse, json, os, queue, sys, threading, time
from dataclasses import asdict
from pathlib import Path

# Path bootstrap
PROJECT = Path(r"E:\python projects\upscale_anime")
sys.path.insert(0, str(PROJECT / "apps" / "anime_upscaler_gui"))

from anime_upscaler_gui import pipeline as P
from anime_upscaler_gui.pipeline import (
    PipelineWorker, RunJob, JobEvent,
)

# --- Configs ---------------------------------------------------------------
CONFIGS = {
    # (label, batch_size, use_tensorrt, use_nvenc, tta)
    "trt_b1_nvenc":  (1,  True,  True,  False),
    "trt_b4_nvenc":  (4,  True,  True,  False),   # known broken (TRT stream bug)
    "pt_b1_nvenc":   (1,  False, True,  False),
    "pt_b4_nvenc":   (4,  False, True,  False),
    "pt_b1_libx":    (1,  False, False, False),    # baseline (TRT off + libx264)
}

INPUT  = PROJECT / ".venv" / "test_4k_540p_input.mp4"
MODEL  = PROJECT / "pretrained" / "RFDN_distill_v1_4x_student.pth"
OUTDIR = PROJECT / "tmp" / "e2e_out"
OUTDIR.mkdir(parents=True, exist_ok=True)


def run_job(label, batch_size, use_tensorrt, use_nvenc, tta,
            reuse_cache=True, engine_shape=None):
    out_path = OUTDIR / f"{label}.mp4"
    if out_path.exists() and not os.environ.get("E2E_KEEP_OLD"):
        out_path.unlink()
    if not INPUT.exists():
        print(f"  [{label}] FAIL: input missing ({INPUT})")
        return None
    if not MODEL.exists():
        print(f"  [{label}] FAIL: model missing ({MODEL})")
        return None

    # Optionally delete engine cache for a shape to simulate cold start
    if engine_shape:
        import shutil
        cache_root = Path(os.environ.get("APPDATA", "")) / "anime_upscaler_gui" / "cache" / "trt"
        if cache_root.exists():
            for f in cache_root.glob(f"*_b{batch_size}_{engine_shape[0]}x{engine_shape[1]}_fp16.engine"):
                if not reuse_cache:
                    print(f"  [{label}] deleting cached engine: {f.name}")
                    f.unlink()

    # If use_tensorrt=False, optionally delete all engine caches to ensure PyTorch path is exercised
    if not use_tensorrt and not reuse_cache:
        import shutil
        cache_root = Path(os.environ.get("APPDATA", "")) / "anime_upscaler_gui" / "cache" / "trt"
        if cache_root.exists():
            for f in cache_root.glob("*.engine"):
                print(f"  [{label}] deleting cached engine: {f.name}")
                f.unlink()

    # Build job
    job = RunJob(
        job_id=1,
        input_path=INPUT,
        output_path=out_path,
        is_video=True,
        model_filename=str(MODEL),
        kind="rfdn_student",
        scale=4,
        outscale=4.0,
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
        on_frame_error=lambda idx, msg: "abort",   # fail-fast in headless mode
    )

    in_q: "queue.Queue" = queue.Queue()
    out_q: "queue.Queue" = queue.Queue()
    worker = PipelineWorker(in_q, out_q)
    worker.start()

    print(f"  [{label}] starting: batch={batch_size} TRT={use_tensorrt} NVENC={use_nvenc} TTA={tta}")
    t0 = time.time()
    in_q.put(job)

    events = []
    last_progress = 0.0
    fps_samples = []
    infer_ms_samples = []
    n_frames = 0
    rc = None
    while True:
        try:
            evt: JobEvent = out_q.get(timeout=300)  # 5-min safety
        except queue.Empty:
            print(f"  [{label}] FAIL: timed out waiting for events")
            worker.stop(); worker.join(timeout=10)
            return None
        events.append({"kind": evt.kind, "msg": evt.message[:200],
                       "progress": evt.progress, "fps": evt.fps,
                       "infer_ms": evt.infer_ms, "frame_idx": evt.frame_idx})
        if evt.kind == "started":
            t_started = time.time()
        elif evt.kind == "progress":
            last_progress = evt.progress
            if evt.fps > 0:
                fps_samples.append(evt.fps)
            if evt.infer_ms > 0:
                infer_ms_samples.append(evt.infer_ms)
            n_frames = max(n_frames, evt.frame_idx + 1)
        elif evt.kind in ("finished", "error", "fatal_error"):
            t_ended = time.time()
            rc = evt.kind
            break
        elif evt.kind == "frame_error":
            # Worker requested an action; we returned "abort"
            pass

    wall_s = t_ended - t_started if 't_started' in dir() else (t_ended - t0)
    total_s = time.time() - t0

    worker.stop(); worker.join(timeout=10)

    if rc != "finished":
        print(f"  [{label}] FAIL: worker ended with kind={rc}")
        print(f"          last message: {events[-1]['msg'] if events else '<none>'}")
        return None

    # Aggregate
    fps_mean = sum(fps_samples) / len(fps_samples) if fps_samples else 0.0
    fps_max  = max(fps_samples) if fps_samples else 0.0
    inf_mean = sum(infer_ms_samples) / len(infer_ms_samples) if infer_ms_samples else 0.0
    out_size = out_path.stat().st_size if out_path.exists() else 0

    result = {
        "label": label,
        "config": {
            "batch_size": batch_size, "use_tensorrt": use_tensorrt,
            "use_nvenc": use_nvenc, "tta": tta,
        },
        "wall_seconds": round(wall_s, 3),
        "total_seconds": round(total_s, 3),
        "frames_processed": n_frames,
        "fps_reported_mean": round(fps_mean, 2),
        "fps_reported_max":  round(fps_max, 2),
        "infer_ms_mean": round(inf_mean, 2),
        "output_size_bytes": out_size,
        "output_path": str(out_path),
        "reuse_cache": reuse_cache,
        "events_count": len(events),
    }
    print(f"  [{label}] OK: {wall_s:.2f}s wall, fps_mean={fps_mean:.2f} "
          f"(peak {fps_max:.2f}), infer_mean={inf_mean:.2f}ms, "
          f"frames={n_frames}, output={out_size/1024:.1f}KB")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="trt_b1_nvenc",
                    help="comma-separated list of config keys (see CONFIGS)")
    ap.add_argument("--reuse-cache", action="store_true", default=True)
    ap.add_argument("--no-cache", dest="reuse_cache", action="store_false")
    ap.add_argument("--all", action="store_true", help="run every config in CONFIGS")
    args = ap.parse_args()

    configs = list(CONFIGS.keys()) if args.all else [c.strip() for c in args.config.split(",")]
    results = []
    for cfg in configs:
        if cfg not in CONFIGS:
            print(f"unknown config: {cfg}; skipping")
            continue
        r = run_job(cfg, *CONFIGS[cfg], reuse_cache=args.reuse_cache)
        if r:
            results.append(r)
        print()

    # Summary
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'config':<20} {'batch':>5} {'TRT':>4} {'NVENC':>6} {'wall':>7} {'fps':>7} {'infer':>7} {'size':>8}")
    for r in results:
        c = r["config"]
        print(f"{r['label']:<20} {c['batch_size']:>5} {str(c['use_tensorrt']):>4} "
              f"{str(c['use_nvenc']):>6} {r['wall_seconds']:>7.2f} {r['fps_reported_mean']:>7.2f} "
              f"{r['infer_ms_mean']:>7.2f} {r['output_size_bytes']/1024:>7.1f}K")

    # Write JSON for downstream report
    report_path = OUTDIR / "results.json"
    report_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {report_path}")
