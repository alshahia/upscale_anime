import sys, queue, time
sys.path.insert(0, r'E:/python projects/upscale_anime/apps/anime_upscaler_gui')
import logging
logging.basicConfig(level=logging.INFO)
import pathlib
from apps.anime_upscaler_gui.anime_upscaler_gui import pipeline as P
src = pathlib.Path(r'C:/Users/Ahmad Mahmoud/Downloads/Video/[Anime3rb.com] Yi Ren Zhi Xia - 1 [480p].mp4')
out = src.with_name(src.stem + '__mid20s_x4_upscaled.mp4')
print('OUTPUT:', out)
in_q, out_q = queue.Queue(), queue.Queue()
w = P.PipelineWorker(in_q, out_q)
w.start()
job = P.RunJob(
    job_id=1, input_path=src, output_path=out, is_video=True,
    model_filename=r'E:/python projects/upscale_anime/pretrained/SRVGG_distill_v1_4x_student.pth',
    kind='srvgg_student', scale=4, outscale=4.0, fp16=True, device='cuda',
    batch_size=4, decode='pyav', prefetch='async', pin_memory='auto',
    downscale_max_edge=576, gpu_guard_mode='warn', tile_size=256, tile_overlap=32,
    cut_start_seconds=712.03, cut_end_seconds=732.03,
    video_crf=18, video_preset='medium', tta=False,
    use_tensorrt=True, use_nvenc=True, nvenc_preset='p1', nvenc_qp=18,
)
in_q.put(job)
t0 = time.perf_counter()
while True:
    evt = out_q.get(timeout=3600)
    el = time.perf_counter() - t0
    print(f'[{el:7.1f}s] {evt.kind}: {evt.message} progress={evt.progress} fps={evt.fps}', flush=True)
    if evt.kind in ('finished', 'error', 'fatal_error'):
        break
w.stop(); w.join(timeout=30)
print('DONE', evt.kind)