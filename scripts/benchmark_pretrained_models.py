#!/usr/bin/env python3
r"""Honest speed benchmark for the downloaded pretrained SR models.

For each model x image: warm-up once, then run N trials and report median ms
plus the output statistics (mean, std, min, max). Healthy SR outputs cluster
around mean ~0.5-0.7 with std ~0.20-0.28; a near-zero std signals a broken
weight load or a wrong arch.

Outputs:
  - results/benchmark/bench.csv          (per model x image stats)
  - results/benchmark/<label>/*.png      (best-of-N SR PNG per image)
  - prints a summary table sorted by median ms

Usage:
  .venv\Scripts\python.exe scripts\benchmark_pretrained_models.py --trials 5
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from compare_pretrained_models import MODELS, _build, _save_sr  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _bench_one(model, x, trials):
    """Warm up once, then time `trials` forward passes. Returns (median_ms, runs)."""
    runs = []
    with torch.no_grad():
        for _ in range(1 + trials):
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            y = model(x)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            # AnimeSR returns (B, N, C, H, W); collapse to middle frame for stats + save
            if y.dim() == 5:
                y = y[:, y.shape[1] // 2]
            runs.append((y, (time.perf_counter() - t0) * 1000.0))
    # drop warm-up, take median ms
    times = sorted(r[1] for r in runs[1:])
    median = times[len(times) // 2]
    return median, runs[1:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', nargs='+', default=['results/val_sr/2.png', 'results/val_sr/1_sr.png'])
    p.add_argument('--trials', type=int, default=5)
    p.add_argument('--output', default='results/benchmark')
    p.add_argument('--pretrained-dir', default='pretrained')
    args = p.parse_args()

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    inputs = [Path(x) for x in args.input]
    print(f'[Bench] device={DEVICE}  trials={args.trials}  inputs={len(inputs)}  models={len(MODELS)}\n')

    rows = []
    for label, ckpt_name, kind, scale, mode in MODELS:
        ckpt_path = Path(args.pretrained_dir) / ckpt_name
        if not ckpt_path.exists():
            print(f'[SKIP] {label}: {ckpt_path} not found')
            continue
        try:
            m = _build(kind, ckpt_path).to(DEVICE)
        except Exception as e:
            print(f'[FAIL] {label}: {e}')
            continue

        print(f'=== {label}  ({ckpt_name}, {scale}x) ===')
        for img_path in inputs:
            if not img_path.exists():
                continue
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE) / 255.0
            if mode == 'video':
                x = x.unsqueeze(1).expand(-1, 3, -1, -1, -1).contiguous()

            median_ms, runs = _bench_one(m, x, args.trials)
            ys = torch.stack([r[0] for r in runs])  # (trials, 1, 3, H, W)
            # bench already collapsed 5D -> 4D inside _bench_one
            last = ys[-1]
            mean = float(last.mean())
            std = float(last.std())
            vmin = float(last.min())
            vmax = float(last.max())
            # consistency across trials (deterministic models should be ~0)
            consistency = float(ys.std(dim=0).max())
            print(f'  {img_path.name:<10} -> {last.shape[3]}x{last.shape[2]}  '
                  f'median={median_ms:6.1f} ms  mean={mean:.3f} std={std:.3f} '
                  f'min={vmin:.2f} max={vmax:.2f}  cross-trial std={consistency:.2e}')

            best_idx = len(runs) - 1
            sr_path = out_root / label / img_path.name
            _save_sr(runs[best_idx][0], sr_path)

            rows.append({
                'model': label, 'checkpoint': ckpt_name, 'input': img_path.name,
                'output_px': f'{last.shape[3]}x{last.shape[2]}',
                'median_ms': round(median_ms, 2),
                'min_ms': round(min(r[1] for r in runs), 2),
                'max_ms': round(max(r[1] for r in runs), 2),
                'mean_val': round(mean, 4),
                'std_val': round(std, 4),
                'min_val': round(vmin, 4),
                'max_val': round(vmax, 4),
                'cross_trial_std': f'{consistency:.2e}',
                'verdict': 'OK' if 0.10 < std < 0.35 and abs(mean) < 0.9 else 'CHECK',
            })

    csv_path = out_root / 'bench.csv'
    if rows:
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f'\n[Rank by median ms over fastest input]')
    by_speed = sorted(rows, key=lambda r: r['median_ms'])
    for r in by_speed:
        print(f'  {r["model"]:24} {r["input"]:12}  {r["median_ms"]:6.1f} ms  '
              f'std={r["std_val"]:.3f}  {r["verdict"]}')

    print(f'\n[Bench] CSV: {csv_path}')
    print(f'[Bench] SR PNGs: {out_root}/<model>/<input>.png')


if __name__ == '__main__':
    main()