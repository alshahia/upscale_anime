import sys, queue, time, pathlib
# Phase A5 fix: insert the project root so 'apps' (top-level package) resolves.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, r'E:/python projects/upscale_anime/apps/anime_upscaler_gui')
from apps.anime_upscaler_gui.anime_upscaler_gui import pipeline as P
src = pathlib.Path(r'C:/Users/Ahmad Mahmoud/Downloads/Video/[Anime3rb.com] Yi Ren Zhi Xia - 1 [480p].mp4')
out = pathlib.Path(str(src.parent / 'YiRenZhiXia_E02_mid2s_x4.mp4'))
if out.exists(): out.unlink()
print('OUTPUT:', out, flush=True)
in_q, out_q = queue.Queue(), queue.Queue()
w = P.PipelineWorker(in_q, out_q)
w.start()
job = P.RunJob(
    job_id=1, input_path=src, output_path=out, is_video=True,
    model_filename=r'E:/python projects/upscale_anime/pretrained/SRVGG_distill_v1_4x_student.pth',
    kind='srvgg_student', scale=4, outscale=4.0, fp16=True, device='cuda',
    batch_size=4, decode='pyav', prefetch='async', pin_memory='auto',
    downscale_max_edge=576, gpu_guard_mode='warn', tile_size=256, tile_overlap=32,
    cut_start_seconds=712.03, cut_end_seconds=714.03,
    video_crf=18, video_preset='medium', tta=False,
    use_tensorrt=True, use_nvenc=True, nvenc_preset='p1', nvenc_qp=18,
)
in_q.put(job)
t0 = time.perf_counter()
evt = None
while True:
    evt = out_q.get(timeout=1200)
    el = time.perf_counter() - t0
    print(f'[{el:7.1f}s] {evt.kind}: {evt.message} progress={evt.progress} fps={evt.fps}', flush=True)
    if evt.kind in ('finished', 'error', 'fatal_error'):
        break
w.stop(); w.join(timeout=15)
import os
print('FINAL:', evt.kind, 'size:', os.path.getsize(out) if out.exists() else 'missing')