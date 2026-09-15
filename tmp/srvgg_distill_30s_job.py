"""
Upscale a 30-second clip from the Yi Ren Zhi Xia - 1 [480p].mp4 video using
the last trained SRVGG distill student model:
    pretrained/SRVGG_distill_v1_4x_student.pth
(kind='srvgg_student', 4x native scale).

Cut window is 120.0 -> 150.0 seconds (30.0s) -- this lands in actual episode
content well after the opening theme has ended (OP usually ends around 85-95s
for this show's episode 1), so the resulting clip is narrative content rather
than title cards.

Output filename is unique to avoid clobbering prior outputs:
  [Anime3rb.com] Yi Ren Zhi Xia - 1 [480p]_30s_x4_srvgg_distill.mp4
"""

import os
import queue
import sys
import time
import pathlib

# Make sure the project root is on sys.path so the top-level 'apps' package
# marker (apps/__init__.py) resolves.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from apps.anime_upscaler_gui.anime_upscaler_gui import pipeline as P

SRC = pathlib.Path(r'C:/Users/Ahmad Mahmoud/Downloads/Video/[Anime3rb.com] Yi Ren Zhi Xia - 1 [480p].mp4')
OUT = pathlib.Path(str(SRC.parent / '[Anime3rb.com] Yi Ren Zhi Xia - 1 [480p]_30s_x4_srvgg_distill.mp4'))

CUT_START = 120.0
CUT_END = 150.0

CKPT = r'E:/python projects/upscale_anime/pretrained/SRVGG_distill_v1_4x_student.pth'

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
    kind='srvgg_student',
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
    use_tensorrt=False,
    use_nvenc=False,
    nvenc_preset='p1',
    nvenc_qp=18,
)

in_q.put(job)

t0 = time.perf_counter()
evt = None
try:
    while True:
        evt = out_q.get(timeout=3600)
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
