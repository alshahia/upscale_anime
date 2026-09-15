"""
Upscale ~2 seconds from the middle of the Yi Ren Zhi Xia - 1 [480p].mp4 video
using the last trained SRVGG distill student model:
    pretrained/SRVGG_distill_v1_4x_student.pth
(kind='srvgg_student', 4x native scale).

Mirrors tmp/mid2s_job.py (the existing RFDN-distill mid-2s job), but switches
the model checkpoint from RFDN to the SRVGG distill student, per the user
request.

Output filename is unique to avoid clobbering prior outputs:
  [Anime3rb.com] Yi Ren Zhi Xia - 1 [480p]_mid2s_x4_srvgg_distill.mp4
"""

import os
import queue
import sys
import time
import pathlib

# Make sure the project root is on sys.path so `apps` (top-level package
# marker at upscale_anime/apps/__init__.py) resolves. When this script is run
# from the project root with `python tmp\srvgg_distill_mid2s_job.py`,
# sys.path[0] becomes `tmp/` (the script's directory), NOT the project root,
# so we explicitly insert the project root here.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Reuse the GUI pipeline worker exactly as tmp/mid2s_job.py does.
from apps.anime_upscaler_gui.anime_upscaler_gui import pipeline as P

SRC = pathlib.Path(r'C:/Users/Ahmad Mahmoud/Downloads/Video/[Anime3rb.com] Yi Ren Zhi Xia - 1 [480p].mp4')
OUT = pathlib.Path(str(SRC.parent / '[Anime3rb.com] Yi Ren Zhi Xia - 1 [480p]_mid2s_x4_srvgg_distill.mp4'))

# Video duration ~1424.056s. Middle is ~712.028s; pick 712.03 -> 714.03 (2.0s window).
CUT_START = 712.03
CUT_END = 714.03

# Last trained SRVGG distill student checkpoint.
CKPT = r'E:/python projects/upscale_anime/pretrained/SRVGG_distill_v1_4x_student.pth'

# Sanity: don't overwrite existing outputs.
if OUT.exists():
    raise SystemExit(f'REFUSING to overwrite existing output: {OUT}')

if not pathlib.Path(CKPT).exists():
    raise SystemExit(f'checkpoint not found: {CKPT}')

print('OUTPUT:', OUT, flush=True)
print('CHECKPOINT:', CKPT, flush=True)
print(f'CUT: [{CUT_START}, {CUT_END}) = {CUT_END - CUT_START:.3f} seconds', flush=True)

in_q, out_q = queue.Queue(), queue.Queue()
worker = P.PipelineWorker(in_q, out_q)
worker.start()

job = P.RunJob(
    job_id=1,
    input_path=SRC,
    output_path=OUT,
    is_video=True,
    model_filename=CKPT,
    kind='srvgg_student',        # arch kind registered in apps/.../archs.py
    scale=4,
    outscale=4.0,
    fp16=True,
    device='cuda',
    batch_size=4,
    decode='pyav',
    prefetch='async',
    pin_memory='auto',
    downscale_max_edge=576,
    gpu_guard_mode='warn',
    tile_size=256,
    tile_overlap=32,
    cut_start_seconds=CUT_START,
    cut_end_seconds=CUT_END,
    video_crf=18,
    video_preset='medium',
    tta=False,
    use_tensorrt=False,          # SRVGG distill student has no compiled TRT engine here
    use_nvenc=False,             # RTX 4000 doesn't have NVENC enabled in this env
    nvenc_preset='p1',
    nvenc_qp=18,
)

in_q.put(job)

t0 = time.perf_counter()
evt = None
try:
    while True:
        evt = out_q.get(timeout=1800)        # 30-min cap per event
        elapsed = time.perf_counter() - t0
        print(f'[{elapsed:7.1f}s] {evt.kind}: {evt.message} progress={evt.progress} fps={evt.fps}',
              flush=True)
        if evt.kind in ('finished', 'error', 'fatal_error'):
            break
finally:
    worker.stop()
    worker.join(timeout=15)

size = os.path.getsize(OUT) if OUT.exists() else 'missing'
print('FINAL:', evt.kind if evt else 'no-event', 'size:', size, 'output:', OUT)
