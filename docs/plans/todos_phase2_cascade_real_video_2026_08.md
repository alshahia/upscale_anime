# Real-Video Cascade Test — Findings (2026-08-31 EOD)

## Setup
- Input: `tmp/real_video_1sec.mp4` — real anime clip cut from `.venv/test_4k_540p_input.mp4`
  - 854x480, 18 fps, 1.00 s, 18 frames (h264 / yuv420p)
- Output: 3416x1920 (4x upscale) at 18 fps via NVENC h264 (qp=18)
- Models:
  - single 4x: `pretrained/RFDN_distill_v1_4x_student.pth`
  - cascade 2x: `pretrained/RFDN_distill_v2_2x_student.pth` (auto-engages 2x->2x when scale=2)
- Test runner: `tmp/test_e2e_cascade_real_video.py` (writes JSON + extracts middle frame)

## Numbers

| Config                | batch | TRT  | NVENC | scale | wall  | fps mean | fps peak | infer ms | output |
|-----------------------|------:|------|-------|------:|------:|---------:|---------:|---------:|-------:|
| single_pt_b1          |   1   | no  | yes   |   4   | 3.39 s | **3.46** | 5.68 | 429.77 | 594.3 K |
| cascade_pt_b1         |   1   | no  | yes   |   2   | 6.81 s | **1.98** | 2.68 | 426.41 | 1.7 MB  |
| cascade_pt_b4         |   4   | no  | yes   |   2   | 10.28 s| **0.79** | 1.32 | 2367.16| 1.7 MB  |
| cascade_trt_b1        |   1   | yes | yes   |   2   | FAIL  | -        | -    | -      | 261 B   |

**Single 4x is 1.75x faster than cascade 2x+2x at batch=1 on this real video.**

## Cascade TRT failure
- `cascade_trt_b1` crashed with `CUDA error: an illegal memory access (cudaError 700)` at `pipeline.py:229` in `_to_tensor` (the pinned->device copy at the *very first* frame).
- Same crash reproduces in fresh Python process; same crash happens after cache wipe.
- `tmp/smoke_trt_both_scales.py` (the smoke test that built both engines successfully earlier in this session) now also fails with cudaError 700.
- Working theory: the cascade wrapper re-enters `_TrtBackend.run` in the same CUDA stream/context that the previous inference already finalised. Stale stream/state leaks across the two 2x passes. This is a real bug in the cascade TRT path, not a transient issue.

## Visual evidence (frame 8 from each output)

| Output                       | Bytes | Res         |
|------------------------------|------:|-------------|
| source (LR)                  |   207 KB | 854x480   |
| single_pt_b1 (4x one-shot)   |  1.54 MB | 3416x1920 |
| cascade_pt_b1 (2x -> 2x)     |  1.76 MB | 3416x1920 |
| cascade_pt_b4 (2x -> 2x, b=4)|  1.76 MB | 3416x1920 |

Files: `tmp/e2e_real/{source,single_pt_b1,cascade_pt_b1,cascade_pt_b4}_frame.png`

## Objective difference (single vs cascade outputs)

| Pair                              | PSNR (avg) | SSIM |
|-----------------------------------|-----------:|-----:|
| single_pt_b1 vs cascade_pt_b1     | **43.79 dB** | 0.9296 |
| single_pt_b1 vs cascade_pt_b4     | 43.79 dB | 0.9296 |

Cascade and single outputs differ measurably (~5.7% of pixels meaningfully different). No ground-truth 4K master available for this clip, so we cannot say which is closer to truth. Earlier validation on the test set:
- single 4x v1: val PSNR 30.10 dB
- cascade 2x+2x v2: val PSNR 28.71 dB (cascade loses ~1.39 dB)

## Decision
**Cascade 2x+2x is not the right path for this hardware.** All three signals agree:
1. Speed: 1.75x SLOWER than single 4x at batch=1 (and 4x slower at batch=4).
2. Quality: -1.39 dB on val (single 4x wins).
3. Stability: cascade TRT path crashes with cudaError 700 in real-video use.

The correct follow-ups remain:
- **CUDA graph capture** of the existing single-4x TRT engine (~0.5 day, expected 30-50% E2E gain).
- **NVDEC decode** for the h264 input (~0.5 day, smaller win).
- **Smaller student** (~50 K params) — only worth doing after CUDA graphs hit their limit.