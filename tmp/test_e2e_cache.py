import sys, json, time, os, shutil
from pathlib import Path
PROJECT = Path(r"E:\python projects\upscale_anime")
sys.path.insert(0, str(PROJECT / "apps" / "anime_upscaler_gui"))
from anime_upscaler_gui import pipeline as P
from anime_upscaler_gui.pipeline import PipelineWorker, RunJob, JobEvent
import queue

INPUT = PROJECT / ".venv" / "test_4k_540p_input.mp4"
MODEL = PROJECT / "pretrained" / "RFDN_distill_v1_4x_student.pth"
CACHE = Path(os.environ["APPDATA"]) / "anime_upscaler_gui" / "cache" / "trt"

def list_engines():
    return sorted(CACHE.glob("*_fp16.engine")) if CACHE.exists() else []

def run_once(label, build_engines=None):
    out = PROJECT / "tmp" / "e2e_out" / f"cache_{label}.mp4"
    if out.exists(): out.unlink()
    if build_engines:
        for e in build_engines:
            if e.exists():
                print(f"  [{label}] deleting cache: {e.name}")
                e.unlink()
    job = RunJob(
        job_id=1, input_path=INPUT, output_path=out, is_video=True,
        model_filename=str(MODEL), kind="rfdn_student", scale=4, outscale=4.0,
        fp16=True, device="cuda", batch_size=1,
        decode="cv2", prefetch="sync", pin_memory="auto",
        downscale_max_edge=0, gpu_guard_mode="off", tile_size=0, tile_overlap=32,
        tta=False, use_tensorrt=True, use_nvenc=True,
        nvenc_preset="p1", nvenc_qp=18,
        on_frame_error=lambda idx, msg: "abort",
    )
    in_q, out_q = queue.Queue(), queue.Queue()
    w = PipelineWorker(in_q, out_q); w.start()
    t0 = time.time()
    in_q.put(job)
    fps_samples = []
    while True:
        evt = out_q.get(timeout=600)
        if evt.kind == "progress" and evt.fps > 0: fps_samples.append(evt.fps)
        if evt.kind in ("finished", "error", "fatal_error"):
            wall = time.time() - t0
            if evt.kind != "finished":
                print(f"  [{label}] FAIL: {evt.message[:100]}")
                return None
            break
    w.stop()
    fps_mean = sum(fps_samples)/len(fps_samples) if fps_samples else 0
    print(f"  [{label}] OK: {wall:.2f}s, fps_mean={fps_mean:.2f}, frames={evt.message.split(chr(40))[0].split(chr(32))[-1] if chr(40) in evt.message else '?'}")
    return wall, fps_mean

print("=== E2E.4: Engine cache stress ===")
print(f"Cache dir: {CACHE}")
print()
print(f"Cache before: {[e.name for e in list_engines()]}")
print()

# Run 1: warm cache
print("--- Run 1: warm cache (reuse all engines) ---")
w1, f1 = run_once("warm")
print()

# Run 2: delete the 480x854 engine (our shape), force rebuild
print("--- Run 2: delete 480x854 engine, rebuild ---")
target = [e for e in list_engines() if "480x854" in e.name and "_b1_" in e.name]
print(f"  Deleting: {[e.name for e in target]}")
w2, f2 = run_once("rebuild", build_engines=target)
print()

# Run 3: warm again (newly-built engine should be reused)
print("--- Run 3: warm again (rebuilt engine reused) ---")
w3, f3 = run_once("rewarm")
print()

print(f"Cache after: {[e.name for e in list_engines()]}")
print()
print(f"=== Summary ===")
print(f"Run 1 (warm):     {w1:.2f}s wall, {f1:.2f} fps")
print(f"Run 2 (rebuild):  {w2:.2f}s wall, {f2:.2f} fps   <- engine built here, takes longer")
print(f"Run 3 (rewarm):   {w3:.2f}s wall, {f3:.2f} fps   <- rebuilt engine reused")
