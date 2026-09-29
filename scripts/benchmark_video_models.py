#!/usr/bin/env python3
r"""Run every downloaded pretrained SR model on a real anime video and dump:
  - results/vid_benchmark/<model>/output.mp4     (H.264 via ffmpeg pipe)
  - results/vid_benchmark/<model>/metrics.csv    (per-frame: ms, stats, sharpness, NIQE, CLIPIQA)
  - results/vid_benchmark/summary.csv            (per-model aggregates)
  - results/vid_benchmark/summary.txt            (human-readable rank)

All models target the same final outscale (default 2.0); 4x models internally
upscale to 4x then bicubic-downsample to the target. ERANet is 2x native.

Usage:
  .venv\Scripts\python.exe scripts\benchmark_video_models.py
  .venv\Scripts\python.exe scripts\benchmark_video_models.py --max-frames 200 --outscale 2 --fp16
  .venv\Scripts\python.exe scripts\benchmark_video_models.py --cut-seconds 30 --cut-start-seconds 60 --no-iqa

By default the script processes the ENTIRE video (--max-frames 0 = all).
Use --max-frames N to clip to N frames for smoke testing.
"""
import argparse
import csv
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

# AGENTS.md: Windows terminal breaks on non-ASCII; force UTF-8 so torch.onnx
# export progress messages (which contain check marks) print cleanly.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    import av
    _HAS_PYAV = True
except Exception:
    _HAS_PYAV = False

try:
    import onnx
    import onnxruntime as ort
    _HAS_ORT = True
except Exception:
    _HAS_ORT = False

# onnxruntime's CUDA EP needs cublasLt at runtime; PyTorch ships it. Surface it.
if _HAS_ORT:
    try:
        import torch as _torch
        _torch_lib = str(Path(_torch.__file__).parent / 'lib')
        if _torch_lib not in os.environ.get('PATH', ''):
            os.environ['PATH'] = _torch_lib + os.pathsep + os.environ['PATH']
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from compare_pretrained_models import _build  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODELS = [
    ('realesr-animevideov3', 'realesr-animevideov3.pth',  'srvgg'),
    ('4xLSDIRCompactv2',     '4xLSDIRCompactv2.pth',      'srvgg'),
    ('AnimeSR_v2',           'AnimeSR_v2.pth',            'animesr'),
    ('SPAN',                 'span_pix_pretrain_4x.pth',  'span'),
    ('ERANet',               'eranet_N12_pretrain_325k.pth','era'),
]


# ---------------- Decode + backend helpers ----------------

class _Cv2Reader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f'cannot open {path}')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 24.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __iter__(self):
        while True:
            ok, bgr = self.cap.read()
            if not ok:
                break
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def release(self):
        self.cap.release()


class _PyAvReader:
    """Yields RGB uint8 (H, W, 3) frames + metadata. Tries NVDEC hwaccel when available."""
    def __init__(self, path):
        if not _HAS_PYAV:
            raise RuntimeError('PyAV not installed')
        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        # try hardware decode; falls back silently to software
        try:
            self.stream.codec_context.options = {'hwaccel': 'cuda'}
        except Exception:
            pass
        self.fps = float(self.stream.average_rate) or float(self.stream.base_rate) or 24.0
        self.total = self.stream.frames or 0
        self.w = self.stream.codec_context.width
        self.h = self.stream.codec_context.height

    def __iter__(self):
        for frame in self.container.decode(self.stream):
            yield frame.to_ndarray(format='rgb24')

    def release(self):
        self.container.close()


def _batched(frames_iter, batch_size, max_frames):
    """Yield (start_idx, list_of_rgb_frames) of length <= batch_size."""
    batch = []
    consumed = 0
    for rgb in frames_iter:
        batch.append(rgb)
        consumed += 1
        if len(batch) == batch_size:
            yield consumed - batch_size, batch
            batch = []
        if max_frames and consumed >= max_frames:
            break
    if batch:
        yield consumed - len(batch), batch


# ponytail: bounded queue + thread = backpressure if GPU falls behind, otherwise
# decode runs ahead of compute. queue.Queue blocks the producer when full, so
# the reader naturally throttles to model speed without dropping frames.
_SENTINEL = object()


class _AsyncReader:
    """Wraps any sync reader so frame decoding happens on a background thread.

    Iterating pulls frames from a bounded queue. Producer is a daemon thread
    that pushes until EOF or max_frames, then a sentinel to terminate the
    consumer cleanly. Errors in the producer surface as a (caught) log line;
    partial frames already in the queue are still yielded.
    """
    def __init__(self, sync_reader, prefetch=8):
        self._sync = sync_reader
        self._q = queue.Queue(maxsize=prefetch)

    def start(self, max_frames=0):
        def _run():
            n = 0
            try:
                for frame in self._sync:
                    self._q.put(frame)  # blocks if queue full -> backpressure
                    n += 1
                    if max_frames and n >= max_frames:
                        break
            except Exception as e:
                print(f'  [AsyncReader] producer error: {e}')
            finally:
                self._q.put(_SENTINEL)
        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is _SENTINEL:
                return
            yield item

    def release(self):
        self._sync.release()


class _PyTorchBackend:
    def __init__(self, ckpt_path, kind, fp16):
        self.model = _build(kind, ckpt_path).to(DEVICE).eval()
        if fp16:
            self.model = self.model.half()
        self.kind = kind

    def __call__(self, x):
        # x: (B, 3, H, W) on DEVICE
        with torch.no_grad():
            if self.kind == 'animesr':
                x = x.unsqueeze(1).expand(-1, 3, -1, -1, -1).contiguous()
                y = self.model(x)
                y = y[:, y.shape[1] // 2]
            else:
                y = self.model(x)
        return y

    def export_onnx(self, onnx_path, h, w, fp16):
        dummy = torch.randn(1, 3, h, w, device=DEVICE)
        if fp16:
            dummy = dummy.half()
        if self.kind == 'animesr':
            dummy = dummy.unsqueeze(1).expand(1, 3, 3, h, w).contiguous()
            dynamic = {'input': {0: 'B', 3: 'H', 4: 'W'}, 'output': {0: 'B', 3: 'H4', 4: 'W4'}}
        else:
            dynamic = {'input': {0: 'B', 2: 'H', 3: 'W'}, 'output': {0: 'B', 2: 'H4', 3: 'W4'}}
        # opset 18 (>= latest supported). dynamo=False uses the legacy exporter which
        # tolerates dynamic_axes without the constraints warning.
        torch.onnx.export(self.model, dummy, str(onnx_path), opset_version=18,
                          input_names=['input'], output_names=['output'],
                          dynamic_axes=dynamic, do_constant_folding=True,
                          dynamo=False)


class _OnnxBackend:
    def __init__(self, onnx_path, kind, fp16):
        if not _HAS_ORT:
            raise RuntimeError('onnxruntime not installed')
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=providers)
        self.kind = kind
        self.fp16 = fp16
        self.np_dtype = np.float16 if fp16 else np.float32

    def __call__(self, x):
        # x: (B, 3, H, W) torch tensor (may carry grad if caller forgot no_grad)
        with torch.no_grad():
            inp = x.detach().cpu().numpy().astype(self.np_dtype)
        if self.kind == 'animesr':
            inp = np.repeat(inp[:, None], 3, axis=1)  # (B,3,3,H,W)
        out = self.sess.run(None, {'input': inp})[0]
        if self.kind == 'animesr':
            out = out[:, out.shape[1] // 2]
        return torch.from_numpy(out).to(DEVICE)


def _ensure_onnx(pyt_backend, onnx_path, h, w, fp16):
    if not onnx_path.exists():
        print(f'  [ONNX] exporting -> {onnx_path.name} (one-time)')
        pyt_backend.export_onnx(onnx_path, h, w, fp16)
    return onnx_path


def _try_load_iqa():
    """Return dict of metric_name -> callable(img_uint8_rgb) -> float, or {}."""
    metrics = {}
    try:
        import pyiqa  # noqa: F401
        for name in ('niqe', 'clipiqa'):
            try:
                metrics[name] = pyiqa.create_metric(name, device=DEVICE)
            except Exception as e:
                print(f'[IQA] {name} unavailable: {e}')
    except Exception as e:
        print(f'[IQA] pyiqa not available: {e}')
    return metrics


def _to_tensor(bgr, half):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    t = torch.from_numpy(np.ascontiguousarray(rgb)).float().unsqueeze(0) / 255.0
    return t.to(DEVICE).half() if half else t.to(DEVICE)


def _tensor_to_bgr(y):
    y = y.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    return cv2.cvtColor((y * 255).astype('uint8'), cv2.COLOR_RGB2BGR)


def _resize_to(sr, target_h, target_w):
    h, w = sr.shape[:2]
    if (h, w) == (target_h, target_w):
        return sr
    return cv2.resize(sr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)


def _ffmpeg_pipe(out_path, fps, w, h, crf=18, preset='medium'):
    if shutil.which('ffmpeg') is None:
        return None
    return subprocess.Popen(
        ['ffmpeg', '-y', '-loglevel', 'error',
         '-f', 'rawvideo', '-pix_fmt', 'bgr24',
         '-s', f'{w}x{h}', '-r', f'{fps:.3f}',
         '-i', 'pipe:0',
         '-c:v', 'libx264', '-preset', preset, '-crf', str(crf),
         '-pix_fmt', 'yuv420p', str(out_path)],
        stdin=subprocess.PIPE)


def _sharpness_laplacian(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _safe_metric(fn, bgr):
    """Run an IQA metric on uint8 RGB tensor; return NaN on failure."""
    try:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
        with torch.no_grad():
            v = fn(t)
        return float(v.detach().cpu().item())
    except Exception:
        return float('nan')


def _summarize_ms(times):
    if not times:
        return {}
    s = sorted(times)
    return {
        'avg_ms': round(sum(s) / len(s), 2),
        'median_ms': round(s[len(s) // 2], 2),
        'p95_ms': round(s[int(len(s) * 0.95)], 2),
        'min_ms': round(s[0], 2),
        'max_ms': round(s[-1], 2),
        'fps_achieved': round(1000.0 / max(s[len(s) // 2], 1e-6), 3),
    }


def _run_model(model, kind, bgr, fp16):
    h0, w0 = bgr.shape[:2]
    x = _to_tensor(bgr, fp16)
    pad_h = pad_w = 0
    if kind == 'animesr':
        # AnimeSR has 2 stride-2 stages -> input must be divisible by 4
        pad_h = (4 - h0 % 4) % 4
        pad_w = (4 - w0 % 4) % 4
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        x = x.unsqueeze(1).expand(-1, 3, -1, -1, -1).contiguous()
        with torch.no_grad():
            y = model(x)
        y = y[:, y.shape[1] // 2]
        if pad_h or pad_w:
            y = y[..., :h0 * 4, :w0 * 4]
    else:
        with torch.no_grad():
            y = model(x)
    if fp16:
        y = y.float()
    return y


def _extract_cut(full_path, cut_path, start_seconds, cut_seconds, fps):
    """ffmpeg stream-copy extract a cut window from the already-encoded full mp4."""
    if shutil.which('ffmpeg') is None:
        return False
    cmd = ['ffmpeg', '-y', '-loglevel', 'error',
           '-ss', f'{start_seconds:.3f}',
           '-i', str(full_path),
           '-t', f'{cut_seconds:.3f}',
           '-c', 'copy', str(cut_path)]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0


def _benchmark_one(label, ckpt_path, kind, src_path, out_dir, outscale, max_frames, fp16,
                   iqa_metrics, frame_stride, cut_seconds, cut_start_seconds,
                   batch_size, decode, backend, prefetch, prefetch_size):
    print(f'\n=== {label}  (decode={decode}, batch={batch_size}, backend={backend}, prefetch={prefetch}) ===')
    try:
        if decode == 'pyav':
            sync_reader = _PyAvReader(src_path)
        else:
            sync_reader = _Cv2Reader(src_path)
    except Exception as e:
        print(f'  [SKIP] cannot open {src_path}: {e}')
        return None
    fps, total, w, h = sync_reader.fps, sync_reader.total, sync_reader.w, sync_reader.h
    if prefetch == 'async':
        reader = _AsyncReader(sync_reader, prefetch=prefetch_size)
        print(f'  [prefetch] async reader started (queue={prefetch_size})')
    else:
        reader = sync_reader
    target_w = int(round(w * outscale))
    target_h = int(round(h * outscale))
    limit = max_frames or total
    if prefetch == 'async':
        reader.start(max_frames=limit)
    full_minutes = (total / fps) / 60
    print(f'  src: {w}x{h} @ {fps:.2f} fps, {total} frames ({full_minutes:.1f} min)')
    print(f'  out: {target_w}x{target_h}, max_frames={limit}')

    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / 'output_full.mp4'
    cut_path = out_dir / 'output_cut.mp4'
    pipe = _ffmpeg_pipe(full_path, fps, target_w, target_h)
    if pipe is None:
        print('  [WARN] ffmpeg missing; using cv2 mp4v fallback (no H.264, no cut extraction)')

    pyt = _PyTorchBackend(ckpt_path, kind, fp16)
    if backend == 'onnx':
        onnx_dir = Path('pretrained/onnx')
        onnx_dir.mkdir(exist_ok=True, parents=True)
        onnx_path = onnx_dir / f'{label}.onnx'
        _ensure_onnx(pyt, onnx_path, h, w, fp16)
        try:
            model = _OnnxBackend(onnx_path, kind, fp16)
        except Exception as e:
            print(f'  [WARN] ONNX load failed ({e}); falling back to PyTorch')
            model = pyt
    else:
        model = pyt

    cut_start_frame = int(round(cut_start_seconds * fps))
    cut_end_frame = cut_start_frame + int(round(cut_seconds * fps))
    write_to_cut = pipe is not None and cut_seconds > 0

    per_frame = []
    proc = 0
    t_wall = time.perf_counter()
    last_report = t_wall

    for start_idx, rgb_batch in _batched(iter(reader), batch_size, limit):
        np_batch = np.stack(rgb_batch, axis=0).astype(np.float32) / 255.0
        np_batch = np.transpose(np_batch, (0, 3, 1, 2))  # NHWC -> NCHW
        x = torch.from_numpy(np.ascontiguousarray(np_batch))
        # ponytail: pinned + non_blocking only pays off when the next model call
        # can overlap the transfer. With sync decode there's nothing to overlap
        # with, so pin_memory just adds an extra host->host copy (~5 ms/batch on
        # this hw). Async decode keeps the queue full, so pin becomes a win.
        if DEVICE.type == 'cuda' and prefetch == 'async':
            x = x.pin_memory().to(DEVICE, non_blocking=True)
        else:
            x = x.to(DEVICE)
        if fp16:
            x = x.half()
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            y = model(x)
        except Exception as e:
            print(f'  [ERR] batch starting at frame {start_idx}: {e}')
            break
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        batch_ms = (time.perf_counter() - t0) * 1000.0
        per_frame_ms = batch_ms / max(x.shape[0], 1)
        if fp16:
            y = y.float()

        for b in range(x.shape[0]):
            src_idx = start_idx + b
            yi = y[b:b + 1]
            sr = _tensor_to_bgr(yi)
            sr = _resize_to(sr, target_h, target_w)
            if pipe:
                pipe.stdin.write(sr.tobytes())
                if write_to_cut and cut_start_frame <= src_idx < cut_end_frame:
                    pipe.stdin.write(sr.tobytes())
            sharp = _sharpness_laplacian(sr)
            row = {'frame_idx': src_idx, 'infer_ms': round(per_frame_ms, 2),
                   'out_mean': round(float(yi.mean()), 4), 'out_std': round(float(yi.std()), 4),
                   'out_min': round(float(yi.min()), 4), 'out_max': round(float(yi.max()), 4),
                   'sharpness': round(sharp, 2)}
            for name, fn in iqa_metrics.items():
                row[name] = round(_safe_metric(fn, sr), 4)
            per_frame.append(row)
            proc += 1
        if max_frames and proc >= max_frames:
            break
        now = time.perf_counter()
        if now - last_report > 10.0:
            elapsed = now - t_wall
            achieved = proc / elapsed
            remaining = (limit - proc) / max(achieved, 1e-6)
            med = sorted([r['infer_ms'] for r in per_frame])[len(per_frame) // 2]
            print(f'  [{proc}/{limit}] {achieved:.2f} fps '
                  f'(median {med:.1f} ms) ETA {remaining / 60:.1f} min')
            last_report = now
    reader.release()
    if pipe:
        pipe.stdin.close(); pipe.wait()

    wall = time.perf_counter() - t_wall
    csv_path = out_dir / 'metrics.csv'
    if per_frame:
        with open(csv_path, 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=list(per_frame[0].keys()))
            wr.writeheader(); wr.writerows(per_frame)

    times = [r['infer_ms'] for r in per_frame]
    stats = _summarize_ms(times)
    stats.update({
        'frames_processed': len(per_frame),
        'wall_seconds': round(wall, 2),
        'avg_sharpness': round(sum(r['sharpness'] for r in per_frame) / max(len(per_frame), 1), 2),
        'avg_out_mean': round(sum(r['out_mean'] for r in per_frame) / max(len(per_frame), 1), 4),
        'avg_out_std': round(sum(r['out_std'] for r in per_frame) / max(len(per_frame), 1), 4),
        'decode': decode, 'batch_size': batch_size, 'backend': backend,
        'prefetch': prefetch, 'prefetch_size': prefetch_size,
    })
    for name in iqa_metrics:
        vals = [r[name] for r in per_frame if not np.isnan(r[name])]
        if vals:
            stats[f'avg_{name}'] = round(sum(vals) / len(vals), 4)
            stats[f'{name}_n'] = len(vals)
        else:
            stats[f'avg_{name}'] = float('nan')
            stats[f'{name}_n'] = 0

    if pipe and cut_seconds > 0:
        ok = _extract_cut(full_path, cut_path, cut_start_seconds, cut_seconds, fps)
        stats['cut_path'] = str(cut_path.relative_to(out_dir.parent)) if ok else 'FAILED'

    real_time = stats['fps_achieved'] / fps if fps else 0
    print(f'  -> {stats["frames_processed"]} frames in {wall:.1f}s '
          f'({stats["fps_achieved"]:.2f} fps, real-time x{real_time:.2f})')
    print(f'  full: {full_path.name}  cut: {cut_path.name if pipe and cut_seconds > 0 else "(skipped)"}')
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default=r'data\anime_vid_lr\Yi Ren Zhi Xia - 1 [480p].mp4')
    p.add_argument('--output-root', default='results/vid_benchmark')
    p.add_argument('--outscale', type=float, default=2.0)
    p.add_argument('--max-frames', type=int, default=0,
                   help='0 = process entire video; >0 = clip to N frames (smoke test)')
    p.add_argument('--frame-stride', type=int, default=1)
    p.add_argument('--cut-seconds', type=float, default=30.0,
                   help='Length of the trimmed cut mp4 per model (0 = skip cut)')
    p.add_argument('--cut-start-seconds', type=float, default=0.0,
                   help='Where the cut starts in the original video')
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--pretrained-dir', default='pretrained')
    p.add_argument('--no-iqa', action='store_true',
                   help='Skip pyiqa metrics (NIQE/CLIPIQA) for pure speed numbers')
    p.add_argument('--models', nargs='*', default=None,
                   help='Subset of model presets to run (default: all 5)')
    p.add_argument('--decode', choices=['cv2', 'pyav'], default='pyav' if _HAS_PYAV else 'cv2',
                   help='Video decoder (pyav is faster, supports NVDEC hwaccel)')
    p.add_argument('--batch', type=int, default=4, dest='batch_size',
                   help='Frames per model.forward call (1 = no batching)')
    p.add_argument('--backend', choices=['pytorch', 'onnx'], default='pytorch',
                   help='Inference backend (onnx = onnxruntime-gpu; auto-exports .onnx on first run)')
    p.add_argument('--prefetch', choices=['sync', 'async'], default='sync',
                   help='sync = decode on main thread (current behavior); '
                        'async = background-thread reader so GPU does not wait for decode')
    p.add_argument('--prefetch-size', type=int, default=8,
                   help='Async queue depth (frames buffered ahead of the GPU)')
    args = p.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f'[FATAL] input not found: {src}')
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f'[VidBench] input={src}  outscale={args.outscale}  max_frames={args.max_frames}  fp16={args.fp16}')

    iqa_metrics = {} if args.no_iqa else _try_load_iqa()
    if iqa_metrics:
        print(f'[VidBench] IQA enabled: {list(iqa_metrics.keys())}')
    else:
        print('[VidBench] IQA disabled (sharpness-only).')

    chosen = MODELS
    if args.models:
        wanted = set(args.models)
        chosen = [(l, c, k) for (l, c, k) in MODELS if l in wanted]
        if not chosen:
            raise SystemExit(f'[FATAL] --models matched none of {[l for l, _, _ in MODELS]}')

    summaries = []
    for label, ckpt_name, kind in chosen:
        ckpt_path = Path(args.pretrained_dir) / ckpt_name
        if not ckpt_path.exists():
            print(f'\n=== {label} ==='); print(f'  [SKIP] {ckpt_path} missing')
            continue
        out_dir = out_root / label
        try:
            s = _benchmark_one(label, ckpt_path, kind, src, out_dir,
                               args.outscale, args.max_frames, args.fp16,
                               iqa_metrics, args.frame_stride,
                               args.cut_seconds, args.cut_start_seconds,
                               args.batch_size, args.decode, args.backend,
                               args.prefetch, args.prefetch_size)
        except Exception as e:
            print(f'  [FAIL] {label}: {e}'); continue
        if s:
            s['model'] = label; s['checkpoint'] = ckpt_name
            summaries.append(s)

    if not summaries:
        print('[VidBench] no models ran successfully.')
        return
    csv_path = out_root / 'summary.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader(); w.writerows(summaries)
    (out_root / 'summary.json').write_text(json.dumps(summaries, indent=2, default=str))

    print(f'\n[Rank by median fps]')
    by_speed = sorted(summaries, key=lambda s: s.get('fps_achieved', 0), reverse=True)
    for s in by_speed:
        niqe = s.get('avg_niqe', float('nan'))
        clip = s.get('avg_clipiqa', float('nan'))
        print(f'  {s["model"]:24} {s["fps_achieved"]:6.2f} fps  '
              f'sharp={s["avg_sharpness"]:7.1f}  '
              f'NIQE={niqe:5.2f}  CLIPIQA={clip:5.3f}  '
              f'out_std={s["avg_out_std"]:.3f}')

    print(f'\n[VidBench] per-model layout: {out_root}/<model>/{{output_full.mp4, output_cut.mp4, metrics.csv}}')
    print(f'[VidBench] summary:           {csv_path}')


if __name__ == '__main__':
    main()