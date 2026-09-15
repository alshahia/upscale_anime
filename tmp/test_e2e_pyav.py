import sys, json, time
from pathlib import Path
PROJECT = Path(r"E:\python projects\upscale_anime")
sys.path.insert(0, str(PROJECT / "apps" / "anime_upscaler_gui"))
from anime_upscaler_gui import pipeline as P
from anime_upscaler_gui.pipeline import PipelineWorker, RunJob, JobEvent
import queue

INPUT = PROJECT / ".venv" / "test_4k_540p_input.mp4"
MODEL = PROJECT / "pretrained" / "RFDN_distill_v1_4x_student.pth"
OUT = PROJECT / "tmp" / "e2e_out" / "trt_b1_pyav.mp4"
if OUT.exists(): OUT.unlink()

job = RunJob(
    job_id=1, input_path=INPUT, output_path=OUT, is_video=True,
    model_filename=str(MODEL), kind="rfdn_student", scale=4, outscale=4.0,
    fp16=True, device="cuda", batch_size=1,
    decode="pyav", prefetch="sync", pin_memory="auto",
    downscale_max_edge=0, gpu_guard_mode="off", tile_size=0, tile_overlap=32,
    tta=False, use_tensorrt=True, use_nvenc=True,
    nvenc_preset="p1", nvenc_qp=18,
    on_frame_error=lambda idx, msg: "abort",
)

in_q, out_q = queue.Queue(), queue.Queue()
w = PipelineWorker(in_q, out_q)
w.start()
print("[pyav] starting: decode=pyav TRT=ON NVENC=ON")
in_q.put(job)

t0 = time.time()
fps_samples = []
infer_samples = []
n_frames = 0
while True:
    evt = out_q.get(timeout=300)
    if evt.kind == "progress":
        if evt.fps > 0: fps_samples.append(evt.fps)
        if evt.infer_ms > 0: infer_samples.append(evt.infer_ms)
        n_frames = max(n_frames, evt.frame_idx + 1)
    if evt.kind in ("finished", "error", "fatal_error"):
        print(f"[pyav] {evt.kind}: {evt.message[:200]}")
        if evt.kind != "finished":
            print("[pyav] FAIL")
            sys.exit(1)
        break
w.stop(); stop_w = time.time()

wall = stop_w - t0
fps_mean = sum(fps_samples)/len(fps_samples) if fps_samples else 0
fps_peak = max(fps_samples) if fps_samples else 0
inf_mean = sum(infer_samples)/len(infer_samples) if infer_samples else 0
print(f"[pyav] OK: {wall:.2f}s, fps_mean={fps_mean:.2f} (peak {fps_peak:.2f}), infer={inf_mean:.2f}ms, frames={n_frames}")
print(f"[pyav] output: {OUT.stat().st_size/1024:.1f} KB")
