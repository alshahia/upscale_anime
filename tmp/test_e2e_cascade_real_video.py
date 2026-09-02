#!/usr/bin/env python3
"""Real-video cascade E2E test.

Runs all four cascade configs (pt_b1, pt_b4, trt_b1, trt_b4) on a real 1-second
anime video (tmp/real_video_1sec.mp4, 854x480, 18 frames, 1s). Also runs a
single-4x baseline for comparison. Extracts a middle frame PNG from each output
for visual inspection.

Usage:
    .venv/Scripts/python.exe tmp/test_e2e_cascade_real_video.py
"""
import argparse, json, os, queue, shutil, subprocess, sys, time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "apps" / "anime_upscaler_gui"))

from anime_upscaler_gui.pipeline import _RunJob, _PipelineWorker, _JobEvent

CONFIGS = {
    # (label, batch_size, use_tensorrt, use_nvenc, tta, model_path, scale)
    "single_pt_b1":  (1, False, True, False, "pretrained/RFDN_distill_v1_4x_student.pth", 4),
    "single_pt_b4":  (4, False, True, False, "pretrained/RFDN_distill_v1_4x_student.pth", 4),
    "single_trt_b1": (1, True, True, False, "pretrained/RFDN_distill_v1_4x_student.pth", 4),
    "cascade_pt_b1": (1, False, True, False, "pretrained/RFDN_distill_v2_2x_student.pth", 2),
    "cascade_pt_b4": (4, False, True, False, "pretrained/RFDN_distill_v2_2x_student.pth", 2),
    "cascade_trt_b1":(1, True, True, False, "pretrained/RFDN_distill_v2_2x_student.pth", 2),
}

INPUT  = PROJECT / "tmp" / "real_video_1sec.mp4"
OUTDIR = PROJECT / "tmp" / "e2e_real"
OUTDIR.mkdir(parents=True, exist_ok=True)


def extract_frame(video_path: Path, out_png: Path, frame_idx: int = 8) -> bool:
    """Use ffmpeg to pull a single frame from the output for visual evidence."""
    if not video_path.exists():
        return False
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", "select=eq(n," + str(frame_idx) + ")",
        "-vframes", "1",
        str(out_png),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=30)
        return out_png.exists()
    except Exception:
        return False


def run_job(label, batch_size, use_tensorrt, use_nvenc, tta,
            model_path, scale, reuse_cache=True):
    out_path = OUTDIR / f"{label}.mp4"
    if out_path.exists():
        out_path.unlink()
    if not INPUT.exists():
        print(f"  [{label}] FAIL: input missing ({INPUT})")
        return None
    if not Path(model_path).exists():
        print(f"  [{label}] FAIL: model missing ({model_path})")
        return None

    # Optional cache wipe to ensure clean TRT build
    if not reuse_cache:
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
        scale=scale,
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
        on_frame_error=lambda idx, msg: "abort",
    )

    in_q: "queue.Queue" = queue.Queue()
    out_q: "queue.Queue" = queue.Queue()
    worker = _PipelineWorker(in_q, out_q)
    worker.start()

    print(f"  [{label}] start: b={batch_size} TRT={use_tensorrt} NVENC={use_nvenc} scale={scale}")
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
            print(f"  [{label}] FAIL: timeout")
            worker.stop(); worker.join(timeout=10)
            return None
        events.append({"kind": evt.kind, "msg": evt.message[:160],
                       "fps": evt.fps, "infer_ms": evt.infer_ms,
                       "frame_idx": evt.frame_idx})
        if evt.kind == "started":
            t_started = time.time()
        elif evt.kind == "progress":
            if evt.fps > 0: fps_samples.append(evt.fps)
            if evt.infer_ms > 0: infer_ms_samples.append(evt.infer_ms)
            n_frames = max(n_frames, evt.frame_idx + 1)
        elif evt.kind in ("finished", "error", "fatal_error"):
            t_ended = time.time()
            rc = evt.kind
            break

    wall_s = t_ended - t_started if "t_started" in dir() else (t_ended - t0)
    worker.stop(); worker.join(timeout=10)

    if rc != "finished":
        last = events[-1]["msg"] if events else "<none>"
        print(f"  [{label}] FAIL: rc={rc}; last='{last}'")
        return None

    fps_mean = sum(fps_samples) / len(fps_samples) if fps_samples else 0.0
    fps_max  = max(fps_samples) if fps_samples else 0.0
    inf_mean = sum(infer_ms_samples) / len(infer_ms_samples) if infer_ms_samples else 0.0
    out_size = out_path.stat().st_size if out_path.exists() else 0

    # Extract a frame for visual evidence
    frame_png = OUTDIR / f"{label}_frame.png"
    extract_frame(out_path, frame_png, frame_idx=8)

    return {
        "label": label,
        "config": {
            "batch_size": batch_size, "use_tensorrt": use_tensorrt,
            "use_nvenc": use_nvenc, "tta": tta,
            "scale": scale, "model_path": str(model_path),
        },
        "wall_seconds": round(wall_s, 3),
        "frames_processed": n_frames,
        "fps_reported_mean": round(fps_mean, 2),
        "fps_reported_max": round(fps_max, 2),
        "infer_ms_mean": round(inf_mean, 2),
        "output_size_bytes": out_size,
        "frame_png": str(frame_png) if frame_png.exists() else None,
        "output_path": str(out_path),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="single_pt_b1,single_trt_b1,cascade_pt_b1,cascade_trt_b1",
                    help="comma-separated list of config keys")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-cache", dest="reuse_cache", action="store_false")
    args = ap.parse_args()

    chosen = list(CONFIGS.keys()) if args.all else [c.strip() for c in args.configs.split(",")]
    print(f"running: {chosen}")
    print(f"input:   {INPUT} ({INPUT.stat().st_size/1024:.1f}KB)")

    results = []
    for cfg in chosen:
        if cfg not in CONFIGS:
            print(f"unknown: {cfg}")
            continue
        r = run_job(cfg, *CONFIGS[cfg], reuse_cache=args.reuse_cache)
        if r:
            results.append(r)
        print()

    # Summary
    print("=" * 88)
    print("REAL-VIDEO CASCADE TEST SUMMARY  (1-second real anime clip)")
    print("=" * 88)
    print(f"{'config':<20} {'b':>2} {'TRT':>4} {'NVENC':>6} {'scale':>5} "
          f"{'wall':>7} {'fps':>7} {'peak':>6} {'infer':>7} {'size':>8}")
    for r in results:
        c = r["config"]
        print(f"{r['label']:<20} {c['batch_size']:>2} {str(c['use_tensorrt']):>4} "
              f"{str(c['use_nvenc']):>6} {c['scale']:>5} "
              f"{r['wall_seconds']:>7.2f} {r['fps_reported_mean']:>7.2f} "
              f"{r['fps_reported_max']:>6.2f} {r['infer_ms_mean']:>7.2f} "
              f"{r['output_size_bytes']/1024:>7.1f}K")

    report_path = OUTDIR / "results.json"
    report_path.write_text(json.dumps(results, indent=2))
    print(f"\nresults -> {report_path}")
    print(f"frames  -> {OUTDIR}/*_frame.png")
