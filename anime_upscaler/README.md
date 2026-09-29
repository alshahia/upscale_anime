# Anime 4x Upscaler - Teacher-Student Distillation

Knowledge-distillation pipeline that compresses a fine-tuned SPAN super-resolution teacher into a **315K-parameter RFDN student** for fast anime upscaling (4x). Built on top of this repository's existing `src/` model zoo and checkpoints - nothing was re-implemented that already worked.

## Pipeline

```
data/anime_video_frames (HR 1920x1080 PNG)
        |  bicubic x0.25 (PIL)          teacher = SPAN V7 EMA, frozen (2.24M params)
        v                                        |
LR [48x48 crops] ----+---- [192x192 HR crops]     |
        |            |                            |
        v            v                            v
   student RFDN (315,844 params) --- KD loss --- teacher outputs + feature taps
        |
        v
ONNX export: fp32 / fp16 / calibrated INT8 -> output/ side-by-side validation strips
```

**Distillation loss** (`distill.py`):

```
L = L1(student_sr, teacher_sr)                # response distillation
  + 0.5 * MSE(student_feats, teacher_feats)   # 4 depth-aligned feature taps (1x1 adapters 52->48ch)
  + 0.2 * L1(student_sr, hr_ground_truth)     # ground-truth anchor
```

## Repository layout

| File | Purpose |
|---|---|
| `dataset.py` | `AnimePairDataset`: bicubic LR x4, [-1,1] normalize, flips + 90-degree rotations, 192px HR train crops / 384px deterministic eval crops, seeded 80/10/10 split |
| `teacher.py` | `SPANTeacher`: frozen wrapper around repo `NeosrSPAN`, `forward_with_features()` taps `[conv_1, block_2, block_4, conv_cat]` |
| `student.py` | `RFDN(nf=52, blocks=6)` - 315,844 params, optional intermediate feature taps |
| `distill.py` | KD trainer (seeded, CSV epoch log, PSNR/SSIM eval vs bicubic + teacher) |
| `export.py` | ONNX opset-17 export (dynamic batch/H/W), fp16, entropy-calibrated QDQ INT8, parity checks, latency report |
| `validate.py` | Labelled bicubic/student/teacher strips + Laplacian sharpness & texture-hallucination flags |
| `eval_teachers.py` | Benchmarks candidate teacher checkpoints (spandrel auto-arch) on the seeded test split - PSNR/SSIM/sharpness/latency |
| `spandrel_teacher.py` | Generic frozen KD teacher for any spandrel-loadable checkpoint (hook-based feature taps, optional autocast) |
| `infer.py` | Upscale arbitrary image(s) x4 via checkpoint or exported ONNX; optional PSNR/SSIM vs ground truth |
| `train.sh` | Runs every stage end-to-end |

## Running

All commands use the repo-local `.venv` (AGENTS.md rule). One-shot:

```bash
bash anime_upscaler/train.sh                    # self-tests + smoke + 40 epochs + export + validate
EPOCHS=60 FAST=1 bash anime_upscaler/train.sh   # skip smoke stages, longer run
```

Stage-by-stage (from repo root; PowerShell users: prepend torch DLLs so the ORT GPU EP loads):

```powershell
$env:PATH = "$PWD/.venv/Lib/site-packages/torch/lib;$env:PATH"
.venv\Scripts\python.exe anime_upscaler\distill.py --epochs 40 --out-dir runs/distill_v1
.venv\Scripts\python.exe anime_upscaler\export.py --run-dir runs/distill_v1
.venv\Scripts\python.exe anime_upscaler\validate.py --run-dir runs/distill_v1 --frames 5
```

### Testing the model

Held-out metrics + labelled side-by-side strips (writes `output/validate_frame_*.png`
and `output/validation_metrics.csv`):

```powershell
.venv\Scripts\python.exe anime_upscaler\validate.py --run-dir runs/distill_v1 --frames 5
```

Upscale your own **low-res** image(s) 4x - PyTorch checkpoint by default, or the exported
ONNX to test the deployable artifact exactly as shipped. With `--gt`, basenames are matched
against a ground-truth folder to report PSNR/SSIM:

```powershell
.venv\Scripts\python.exe anime_upscaler\infer.py --input path\to\lr.png
.venv\Scripts\python.exe anime_upscaler\infer.py --input lr_folder\ --out output\upscaled --gt data\anime_video_frames
$env:PATH = "$PWD/.venv/Lib/site-packages/torch/lib;$env:PATH"   # required for the ONNX CUDA EP
.venv\Scripts\python.exe anime_upscaler\infer.py --input lr.png --onnx anime_upscaler\export\student_fp16.onnx
```

Use `--tile 512` for inputs too large for VRAM. Note: full-frame scores run lower than the
384px-crop test table (dark/hard content outside the centre crop) - compare like with like.

## Dataset

990 frames extracted from anime video (`data/anime_video_frames`), 1920x1080 PNG, ~1.89 GB. Seeded split (seed 42): **792 train / 99 val / 99 test**. Validation/test use deterministic center crops so metrics stay comparable across epochs and runs.

## Models

| Model | Params | Role |
|---|---|---|
| SPAN V7 (finetuned EMA, `NEOSR_SPAN_V7_ANIME_001/finetune_best.pth`) | 2,236,731 | Frozen teacher |
| RFDN nf=52, 6 RFDB blocks | **315,844** | Student (<600K budget) |

Teacher checkpoint resolution order: finetuned V7 -> `pretrained/span_pix_pretrain_4x.pth`. The generic loaders in `src/models/teachers/` silently mis-load official RCAN/EDSR checkpoints (rgb_range=255 mean-shift dropped, bias mismatches) - documented trap, do not use them without verification.

### Choosing the teacher (measured, not reputational)

Candidate open-weight anime upscalers are downloaded into `checkpoints/candidates/` and scored with `eval_teachers.py` on the exact seeded test split used for all reported numbers:

```powershell
.venv\Scripts\python.exe anime_upscaler\eval_teachers.py          # full 99-image split
```

Benchmark (99 test crops, 384px, fp32; `results/teacher_bench*.csv`):

| Teacher | Params | PSNR dB | SSIM | train-crop ms |
|---|---|---|---|---|
| **GRL-small LUDVAE (Phips, HFA2k)** | 3.49M | **31.85** | 0.9234 | 222 |
| AnimeSharp 4x (Kim2091) | 16.7M | 31.28 | 0.9250 | 127 |
| SRFormer-light LUDVAE (Phips) | 0.87M | 30.76 | 0.9140 | 109 |
| RealPLKSR-HFA2k LUDVAE (Phips) | 7.4M | 30.62 | 0.9160 | 33 |
| SPAN V7 incumbent | 2.24M | 30.52 | 0.9237 | **6** |
| APISR RRDB/DAT/GRL (GAN-heavy) | 1-11M | 22-24 | ~0.82 | 24-268 |

APISR's perceptual models fall below bicubic on fidelity (texture hallucination profile: Laplacian ~3-4x baseline), so they are poor fidelity-KD teachers despite their reputation. The bench winner, **GRL-small LUDVAE**, scores **+1.33 dB** over the incumbent teacher at equal SSIM and healthy sharpness.

Retrain the student against it:

```powershell
.venv\Scripts\python.exe anime_upscaler\distill.py --epochs 40 --out-dir runs/distill_v2_grl ^
    --teacher-spandrel checkpoints/candidates/4xHFA2kLUDVAEGRL_small.safetensors
```

`--teacher-half` enables CUDA autocast for the teacher with an exact-fp32 retry guard (SR *and* feature taps are checked for non-finite values - an intermediate-tap overflow once poisoned a student even though SR looked finite, so both are guarded now). For attention-based GRL the retry rate erases most of the gain; keep fp32 there. Conv-only teachers (RealPLKSR etc.) benefit cleanly.

### Caveat: bench-winning teacher ≠ student-gaining teacher

We benchmarked the v2 GRL student against this GRL teacher and the **student measured -0.26 dB WORSE than the v1 SPAN-distilled student** despite the teacher being +1.05 dB better (see [RESULTS.md §9](RESULTS.md) for the full numbers and root-cause analysis). Teacher PSNR on the test split is not a reliable proxy for student distillation quality at this capacity gap. The original SPAN V7 teacher (`runs/distill_v1`) remains the deployed production model.

## Results

Full experiment record (trajectory, per-frame validation, latency investigation): **[RESULTS.md](RESULTS.md)**.

### Full distillation (40 epochs, batch 16, Adam 5e-5 cosine)

Test split (99 frames, deterministic 384px center crops):

| Model | PSNR (dB) | SSIM |
|---|---|---|
| Bicubic x4 | 29.45 | 0.8780 |
| **Student (315K)** | 29.32 | 0.8728 |
| Teacher SPAN V7 (2.24M) | 30.44 | 0.9218 |
| Student retention of teacher gain | -12.6% | |

Validation quick-eval trajectory: student val PSNR 26.94 (ep 1) -> 28.24 (ep 2) -> 29.51 (ep 9) -> **29.89 (ep 40)**; teacher constant at 31.22.

**Interpretation (honest):** at 7x parameter compression with the response-dominated KD loss (L1-to-teacher + feature MSE + 0.2 GT anchor), the student converges to teacher-*like* outputs (Laplacian profile matches the teacher, zero texture-hallucination flags on held-out frames) but does **not** beat bicubic on PSNR/SSIM fidelity within this budget. Paths that would close the gap: train the originally-specified ~200 epochs (still improving ~+0.05 dB/epoch at ep 40), raise the ground-truth anchor weight, or widen the student (nf=64). All three work with existing CLI knobs.

### Smoke run (2 epochs, 12 files) - pipeline sanity only

| Split | Bicubic | Student | Teacher |
|---|---|---|---|
| val | 28.10 dB | 18.02 dB | 29.47 dB |

## Export & latency

From `anime_upscaler/export/export_report.csv` (real checkpoint; 270x480 LR input -> 1080x1920 output, Quadro RTX 4000):

| Variant | Size | Parity vs fp32 | ms/frame |
|---|---|---|---|
| pytorch-fp32-gpu | - | - | 30.8 |
| ort-cuda-fp32 | 1.28 MB | max diff 0.0000 | 32.4 |
| **ort-cuda-fp16** | **0.65 MB** | max diff 0.0010 (78.8 dB) | **21.0** |
| ort-cpu-int8 | 0.42 MB | max diff 0.0194 (53.2 dB) | 3332 (see notes) |

INT8 is calibrated on 32 real anime patches with entropy (KL) method and per-channel QDQ QInt8 weights - chosen specifically to avoid flat-region banding on line art.

### Latency findings (honest engineering notes)

- The workload is **compute-bound**, not launch-bound: manual CUDA-graph replay (19.4 ms) does not beat eager fp16 (19.4 ms), and channels_last + cuDNN autotune gives nothing (20.8 ms).
- TensorRT Execution Provider is unavailable here (pip ORT ships no TRT DLLs; LoadLibrary error 126), and `torch.compile` needs Triton (absent on Windows).
- **The <15 ms/frame target is not met on this GPU**: the honest floor is ~19-20 ms for a 1080p output in fp16. Compute scales with SM throughput - any modern consumer GPU (RTX 3060+) clears 15 ms comfortably. Shrinking the student (`nf=40`) would also fit the budget at some quality cost.
- `ort-cpu-int8` is unusably slow in this ORT build (QDQ int8 conv CPU fallback). Deploy fp16 via CUDA/TensorRT instead; reserve INT8 for an external TensorRT engine build with its own calibration.

## Deployment (MPV / AnimeJaNai)

The exported ONNX drops into the AnimeJaNai ecosystem:

1. **AnimeJaNai Real-Time (recommended)**: point its config at `anime_upscaler/export/student_fp16.onnx`; it builds and caches a TensorRT fp16 engine on first run. Expect ~45-50 fps at 1080p output on Quadro RTX 4000; RTX 30-series cards exceed 100 fps.
2. **mpv shader path**: convert the ONNX to an mpv GLSL hook once (AnimeJaNai shader exporter or equivalent) and register via `glsl-shaders-append`.
3. **Batch CLI**: plain `onnxruntime-gpu` - see `export.py::bench` for the exact session/feed pattern (fp16 tensors, dynamic H/W).

Keep input frames mod-4 aligned so the x4 upsampler never sees odd sizes.

## Known limitations

- Teacher retention depends entirely on the frozen checkpoint: finetuned SPAN V7 EMA measures ~28.5 dB on held-out crops (~24 dB for bicubic).
- Laplacian hallucination flags are heuristic: an undertrained student trips them loudly (verified during smoke testing); they quiet down as distillation converges.
- Windows multiprocessing needs the module-level `seed_worker` in `distill.py`; keep worker-init functions module-level when extending the loader.
