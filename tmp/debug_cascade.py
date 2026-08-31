import sys, queue, os, traceback
sys.path.insert(0, "apps/anime_upscaler_gui")
import cv2, torch
from anime_upscaler_gui.pipeline import _RunJob, _JobEvent, _PipelineWorker

job = _RunJob(
    job_id=1,
    input_path=__import__('pathlib').Path('E:/python projects/upscale_anime/.venv/test_4k_540p_input.mp4'),
    output_path=__import__('pathlib').Path('E:/python projects/upscale_anime/tmp/e2e_out/cascade_debug.mp4'),
    is_video=True,
    model_filename='E:/python projects/upscale_anime/pretrained/RFDN_distill_v2_2x_student.pth',
    kind='rfdn_student',
    scale=2,
    outscale=4.0,
    fp16=True,
    device='cuda',
    batch_size=1,
    decode='cv2',
    prefetch='sync',
    pin_memory='auto',
    downscale_max_edge=0,
    gpu_guard_mode='off',
    tile_size=0,
    tile_overlap=32,
    cut_start_seconds=0.0,
    cut_end_seconds=0.0,
    video_crf=18,
    video_preset='medium',
    tta=False,
    use_tensorrt=True,
    use_nvenc=True,
    nvenc_preset='p1',
    nvenc_qp=18,
)
in_q = queue.Queue(); out_q = queue.Queue()
w = _PipelineWorker(in_q, out_q); w.start()
in_q.put(job)
import time
t0 = time.time()
while time.time()-t0 < 60:
    try:
        evt = out_q.get(timeout=2)
    except queue.Empty:
        print("TIMEOUT")
        break
    print(f"{evt.kind}: {evt.message[:200]} fps={evt.fps} frame_idx={evt.frame_idx}")
    if evt.kind in ('finished','error','fatal_error'):
        break
w.stop(); w.join(timeout=10)
print("done")
