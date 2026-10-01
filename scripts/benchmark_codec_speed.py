"""
Benchmark of the v7 in-process codec pipeline (compression_modules.py).

Run from the project root with:
  python scripts/benchmark_codec_speed.py                 # human-readable table
  python scripts/benchmark_codec_speed.py --json          # JSON to stdout
  python scripts/benchmark_codec_speed.py --json -o out.json
  python scripts/benchmark_codec_speed.py --sizes 32 64   # custom sizes
  python scripts/benchmark_codec_speed.py --trials 10     # custom trial count

Empirical numbers on Quadro RTX 4000, in-process path (post-refactor):

| Codec   | 32x32 | 64x64 | 128x128 |
|---------|------:|------:|--------:|
| AVIF    |  7.04 |  6.74 |   18.87 |
| h264    | 10.94 | 10.90 |   13.65 |
| h265    | 45.89 | 47.55 |   56.43 |

For comparison, the prior imageio_ffmpeg subprocess path measured
0.01-0.07 s/frame *plus* per-call subprocess spawn (50-200 ms cold).
Net: the codec step in v7 training is now ~5-10x faster wall-clock.

Used in CI to track codec perf regressions. Run before and after
changes to compression_modules.py to confirm no perf regression.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from anime_sr.data.compression_modules import (
    JPEGCompression, WebPCompression, AVIFCompression,
    PyAVVideoCompression,
)


def _bench(fn, n):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / n


def measure(sizes, trials):
    rng = np.random.default_rng(0)
    results = {"sizes": list(sizes), "trials": trials, "ms_per_frame": {}}
    for sz in sizes:
        img = rng.integers(0, 256, size=(sz, sz, 3), dtype=np.uint8)
        avif = AVIFCompression((30, 90))
        h264 = PyAVVideoCompression('h264', (18, 35))
        h265 = PyAVVideoCompression('h265', (18, 35))
        results["ms_per_frame"][str(sz)] = {
            "avif": round(_bench(lambda: avif(img), trials), 3),
            "h264": round(_bench(lambda: h264(img), trials), 3),
            "h265": round(_bench(lambda: h265(img), trials), 3),
        }
    return results


def print_table(results):
    sizes = results["sizes"]
    print(f"Size  | AVIF  | h264  | h265  (trials={results['trials']})")
    print("-" * 50)
    for sz in sizes:
        row = results["ms_per_frame"][str(sz)]
        print(f"{sz:4d}  | {row['avif']:5.2f} | {row['h264']:5.2f} | {row['h265']:5.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--json', action='store_true', help='Emit JSON to stdout (or -o file)')
    p.add_argument('-o', '--output', help='Write JSON to this file (default: stdout with --json)')
    p.add_argument('--sizes', type=int, nargs='+', default=[32, 64, 128],
                   help='Frame sizes to benchmark (default: 32 64 128)')
    p.add_argument('--trials', type=int, default=5,
                   help='Iterations per codec per size (default: 5)')
    args = p.parse_args()

    results = measure(args.sizes, args.trials)

    if args.json:
        text = json.dumps(results, indent=2)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(text)
                f.write('\n')
            print(f"Wrote {args.output}")
        else:
            print(text)
    else:
        print_table(results)


if __name__ == "__main__":
    main()
