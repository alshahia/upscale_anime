"""
Benchmark of the v7 in-process codec pipeline (compression_modules.py).

Run from the project root with:
  python scripts_temp/test_codec_speed.py

Empirical numbers on Quadro RTX 4000, in-process path (post-refactor):

| Codec   | 32x32 | 64x64 | 128x128 |
|---------|------:|------:|--------:|
| AVIF    |  7.04 |  6.74 |   18.87 |
| h264    | 10.94 | 10.90 |   13.65 |
| h265    | 45.89 | 47.55 |   56.43 |

For comparison, the prior imageio_ffmpeg subprocess path measured
0.01-0.07 s/frame *plus* per-call subprocess spawn (50-200 ms cold).
Net: the codec step in v7 training is now ~5-10x faster wall-clock.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from data.compression_modules import (
    JPEGCompression, WebPCompression, AVIFCompression,
    PyAVVideoCompression,
)


def _bench(fn, n=5):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / n


def main():
    print("Size  | AVIF  | h264  | h265")
    print("-" * 35)
    for sz in (32, 64, 128):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, size=(sz, sz, 3), dtype=np.uint8)
        avif = AVIFCompression((30, 90))
        h264 = PyAVVideoCompression('h264', (18, 35))
        h265 = PyAVVideoCompression('h265', (18, 35))
        a = _bench(lambda: avif(img))
        h = _bench(lambda: h264(img))
        v = _bench(lambda: h265(img))
        print(f"{sz:4d}  | {a:5.2f} | {h:5.2f} | {v:5.2f}")


if __name__ == "__main__":
    main()
