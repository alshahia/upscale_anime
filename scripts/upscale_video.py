#!/usr/bin/env python3
r"""Upscale an anime video file with one of the downloaded pretrained models.

Defaults to realesr-animevideov3 (anime-specific, BSD-3, real-time capable).
Use --model to swap. Outscale 2.0 is the practical sweet spot for 1080p anime.

Usage:
  .venv\Scripts\python.exe scripts\upscale_video.py --input anime.mp4 --output anime_2x.mp4
  .venv\Scripts\python.exe scripts\upscale_video.py --input anime.mp4 --output anime_2x.mp4 --model lsdir --outscale 2 --fp16
  .venv\Scripts\python.exe scripts\upscale_video.py --input anime.mp4 --output anime_4x.mp4 --model span --outscale 4
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from compare_pretrained_models import _build, _save_sr  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PRESETS = {
    'realesr-animevideov3': ('realesr-animevideov3.pth', 'srvgg', 4.0),
    'lsdir':                ('4xLSDIRCompactv2.pth',     'srvgg', 4.0),
    'span':                 ('span_pix_pretrain_4x.pth', 'span',  4.0),
    'animesr-v2':           ('AnimeSR_v2.pth',           'animesr', 4.0),
    'animesr-v1':           ('AnimeSR_v1-PaperModel.pth','animesr', 4.0),
}


def _resize_keep_ar(img, target_h, target_w):
    h, w = img.shape[:2]
    if (h, w) == (target_h, target_w):
        return img
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)


def _open_capture(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f'[FATAL] cannot open video: {path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, n, w, h


def _to_tensor(bgr, device, half):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    t = torch.from_numpy(np.ascontiguousarray(rgb)).float().unsqueeze(0) / 255.0
    if half:
        t = t.half()
    return t.to(device)


def _tensor_to_bgr(y):
    y = y.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    return cv2.cvtColor((y * 255).astype('uint8'), cv2.COLOR_RGB2BGR)


def _ffmpeg_pipe(out_path, fps, w, h, crf=18, preset='medium'):
    if shutil.which('ffmpeg') is None:
        return None
    cmd = ['ffmpeg', '-y', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'bgr24',
           '-s', f'{w}x{h}', '-r', f'{fps:.3f}',
           '-i', 'pipe:0',
           '-c:v', 'libx264', '-preset', preset, '-crf', str(crf),
           '-pix_fmt', 'yuv420p', str(out_path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', default='realesr-animevideov3',
                   choices=list(MODEL_PRESETS.keys()))
    p.add_argument('--outscale', type=float, default=2.0,
                   help='Final output scale; 2.0 = 2x, 4.0 = 4x')
    p.add_argument('--fp16', action='store_true',
                   help='Run model in FP16 (cuts VRAM ~half, ~1.5-2x faster on RTX)')
    p.add_argument('--max-frames', type=int, default=0,
                   help='0 = all; positive = process only this many frames (smoke test)')
    p.add_argument('--pretrained-dir', default='pretrained')
    args = p.parse_args()

    ckpt_name, kind, native_scale = MODEL_PRESETS[args.model]
    ckpt_path = Path(args.pretrained_dir) / ckpt_name
    if not ckpt_path.exists():
        raise SystemExit(f'[FATAL] checkpoint not found: {ckpt_path}')

    cap, fps, total, w, h = _open_capture(args.input)
    target_w = int(round(w * args.outscale))
    target_h = int(round(h * args.outscale))
    print(f'[Video] {args.input}: {w}x{h} @ {fps:.2f} fps, {total} frames')
    print(f'[Video] model={args.model} ({ckpt_name}), outscale={args.outscale}, fp16={args.fp16}')
    print(f'[Video] output: {target_w}x{target_h}')

    model = _build(kind, ckpt_path).to(DEVICE).eval()
    if args.fp16:
        model = model.half()

    ffmpeg_proc = _ffmpeg_pipe(args.output, fps, target_w, target_h)
    use_pipe = ffmpeg_proc is not None
    if not use_pipe:
        # fallback: cv2 mp4v (large files, no H.264 compression)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (target_w, target_h))
        if not writer.isOpened():
            raise SystemExit('[FATAL] cv2 VideoWriter failed to open')
        print('[Video] ffmpeg not found; falling back to mp4v (no H.264).')

    proc_frames = 0
    total_ms = 0.0
    t_start = time.perf_counter()
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            x = _to_tensor(frame, DEVICE, args.fp16)
            if kind == 'animesr':
                x = x.unsqueeze(1).expand(-1, 3, -1, -1, -1).contiguous()
                y = model(x)
                y = y[:, y.shape[1] // 2]
            else:
                y = model(x)
            sr = _tensor_to_bgr(y.float() if args.fp16 else y)
            sr = _resize_keep_ar(sr, target_h, target_w)
            if use_pipe:
                ffmpeg_proc.stdin.write(sr.tobytes())
            else:
                writer.write(sr)
            total_ms += (time.perf_counter() - t0) * 1000.0
            proc_frames += 1
            if args.max_frames and proc_frames >= args.max_frames:
                break
            if proc_frames % 30 == 0:
                elapsed = time.perf_counter() - t_start
                achieved = proc_frames / elapsed
                eta = (total - proc_frames) / max(achieved, 1e-6)
                print(f'  [{proc_frames}/{total}] {achieved:.2f} fps, ETA {eta:.0f}s')

    cap.release()
    if use_pipe:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
    else:
        writer.release()

    wall = time.perf_counter() - t_start
    avg_ms = total_ms / max(proc_frames, 1)
    achieved_fps = proc_frames / max(wall, 1e-6)
    real_time_ratio = achieved_fps / fps
    print(f'\n[Done] {proc_frames} frames in {wall:.1f}s -> {achieved_fps:.2f} fps')
    print(f'[Done] avg per-frame: {avg_ms:.1f} ms  (source {fps:.2f} fps; '
          f'>=1.0x = real-time capable)')
    print(f'[Done] output: {args.output}')


if __name__ == '__main__':
    main()