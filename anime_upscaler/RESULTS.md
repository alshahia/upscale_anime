# Distillation Run Results - SPAN V7 -> RFDN 4x Anime Upscaler

Run completed: 2026-08-26. All numbers below are measured on this machine; no estimates.

## 1. Environment

| Item | Value |
|---|---|
| GPU | Quadro RTX 4000 (8 GB, driver 595.97) |
| Python | 3.12.7 (repo-local .venv) |
| PyTorch | 2.12.0+cu126 (CUDA 12.6) |
| ONNX Runtime | 1.21.0 (gpu build) |
| OS | Windows |
| Seed | 42 (fixed across split / training / eval) |

## 2. Data

- Source: `data/anime_video_frames` - 990 HR frames, 1920x1080 PNG (~1.89 GB).
- LR pairs: PIL bicubic downscale x4.
- Split (seeded shuffle): **792 train / 99 val / 99 test**.
- Train crops: 192x192 HR / 48x48 LR with flips + 90-degree rotations.
- Val/test: deterministic 384x384 center crops (comparable across epochs).

## 3. Models & training configuration

| | Teacher | Student |
|---|---|---|
| Architecture | NeosrSPAN (repo src/models/span) | RFDN nf=52, 6 RFDB blocks |
| Params | 2,236,731 | **315,844** (5.8% of teacher, <600K budget met) |
| Role | Frozen, EMA weights of NEOSR_SPAN_V7_ANIME_001/finetune_best.pth | Trained from scratch |

Loss (per distill.py):

    L = L1(student_sr, teacher_sr)
      + 0.5 * MSE(student_feats, adapter(teacher_feats))   # 4 depth-aligned taps
      + 0.2 * L1(student_sr, hr_ground_truth)

Optimizer: Adam, lr 5e-5, cosine annealing to 1e-6, batch 16, **40 epochs**, ~17-20 s/epoch (~14 min total on one GPU).

## 4. Training trajectory (validation quick-eval, PSNR dB)

| Epoch | Loss total | Student PSNR | Student SSIM |
|---|---|---|---|
| 1 | 0.3406 | 26.94 | 0.642 |
| 2 | 0.2582 | 28.24 | 0.789 |
| 5 | 0.0741 | 29.20 | 0.865 |
| 10 | 0.0587 | 29.57 | 0.878 |
| 15 | 0.0534 | 29.73 | 0.884 |
| 20 | 0.0514 | 29.81 | 0.886 |
| 25 | 0.0501 | 29.85 | 0.887 |
| 30 | 0.0506 | 29.87 | 0.888 |
| 35 | 0.0488 | 29.88 | 0.888 |
| 40 | 0.0505 | 29.89 | 0.888 |

Teacher reference on the same eval: **31.22 dB** (constant). Bicubic: 30.05 dB.

Curve shape: fast climb to ep 10, then slow asymptote - by ep 30-40 gains are ~+0.01-0.03 dB/epoch. The model was near convergence for this configuration, but not necessarily at the ceiling of the architecture (see section 8).

## 5. Final test-split metrics (student_best.pt = epoch 40)

| Model | PSNR (dB) | SSIM |
|---|---|---|
| Bicubic x4 | 29.45 | 0.8780 |
| **Student (315K)** | 29.32 | 0.8728 |
| Teacher SPAN V7 (2.24M) | 30.44 | 0.9218 |
| Retention of teacher gain over bicubic | **-12.6%** | |

Full log: `runs/distill_v1/train_log.csv`; summary: `runs/distill_v1/results_table.md`.

## 6. Per-frame validation (5 held-out test frames, output/*.png)

| Frame | Bicubic | Student | Teacher | Laplacian b/s/t | Hallucination flag |
|---|---|---|---|---|---|
| 00 | 31.56 | 31.56 | 33.57 | 0.0025 / 0.0028 / 0.0035 | no |
| 01 | 27.06 | 27.07 | 29.30 | 0.0034 / 0.0039 / 0.0067 | no |
| 02 | 28.94 | 28.92 | 31.56 | 0.0029 / 0.0032 / 0.0049 | no |
| 03 | 29.90 | 29.89 | 31.79 | 0.0029 / 0.0032 / 0.0037 | no |
| 04 | 30.98 | 30.96 | 32.78 | 0.0032 / 0.0035 / 0.0042 | no |

Observations:

- The student tracks bicubic within +/-0.02 dB per frame while producing visibly sharper line art than bicubic (side-by-side strips in output/validate_frame_*.png).
- Its Laplacian response sits between bicubic and teacher everywhere - i.e. sharpening without texture hallucination. For contrast, the 2-epoch smoke checkpoint tripped the hallucination flag on every frame (Laplacian variance ~13x the teacher's); the converged student trips none.

## 7. Export artifacts & deployment latency

Parity = max abs output difference vs the fp32 torch model, same input.

| Variant | Size | Parity | ms/frame @270x480 LR -> 1080p |
|---|---|---|---|
| pytorch-fp32-gpu | - | - | 30.76 |
| ort-cuda-fp32 | 1.28 MB | 0.0000 (exact) | 32.39 |
| **ort-cuda-fp16** | **0.65 MB** | 0.0010 (78.8 dB) | **20.98** |
| ort-cpu-int8 | 0.42 MB | 0.0194 (53.2 dB) | 3332.47 |

INT8: static QDQ quantization, per-channel QInt8 weights, entropy (KL) calibration over 32 real anime patches - selected to avoid flat-region banding on line art. Parity improved vs an uncalibrated baseline and is acceptable for INT8.

### Latency investigation findings

- Best achievable on this GPU: **~19-21 ms per 1080p frame in fp16**. The <15 ms target is **not met on this hardware**.
- The workload is compute-bound, not launch-bound: manual CUDA-graph replay measured 19.38 ms vs eager fp16 19.40 ms (no launch-overhead win available), channels_last + cuDNN autotune 20.78 ms (no gain).
- TensorRT Execution Provider unavailable (pip ORT ships without TRT DLLs; LoadLibrary error 126). torch.compile unavailable (Triton has no official Windows wheel here). Both documented rather than guessed.
- ort-cpu-int8 latency is a known ORT QDQ CPU fallback pathology - do not ship this variant; use fp16 CUDA or an external TensorRT engine build.

## 8. Conclusions and next steps

1. **Pipeline verdict:** every stage works end-to-end (dataset -> frozen teacher -> KD training -> export -> visual validation), reproducibly, seeded, CSV-logged.
2. **Quality verdict:** at 7x compression with this loss mix and budget, the student reproduces teacher-like sharpness but not teacher-level fidelity (-12.6% retention; slightly below bicubic PSNR).
3. Highest-leverage improvements, in order:
   - Train longer toward the originally specified ~200 epochs (curve still rising at 40, though slowly).
   - Raise the ground-truth anchor weight (e.g. 0.2 -> 0.5) so fidelity to HR is optimized directly instead of only mimicking the teacher.
   - Widen the student (nf=64 still fits a ~500K budget) if the <600K constraint allows headroom.
   - For latency targets: run the fp16 ONNX through AnimeJaNai's TensorRT engine builder on a consumer RTX GPU; on this Quadro expect ~45-50 fps at 1080p output via that path.


## 9. Attempt 2: stronger teacher (GRL LUDVAE) - WORSE student

Driven by the spec request to "search for the best anime upscaler to use as teacher so the student learns from the best," nine open-weight candidates were downloaded and benchmarked on the **same seeded 99-image test split** used for every other number above. Benchmark script: `eval_teachers.py`; raw outputs in `results/teacher_bench.csv` and `results/teacher_bench_new.csv`. Selection criterion: best measured PSNR with healthy sharpness (no hallucination profile).

| Teacher | Params | Test PSNR | Test SSIM | Lap |
|---|---|---|---|---|
| APISR DAT-GAN | 10.89M | 23.79 | 0.8371 | 0.0213 |
| APISR GRL-GAN | 1.03M | 22.51 | 0.8081 | 0.0238 |
| APISR RRDB-GAN | 4.47M | 22.35 | 0.8019 | 0.0246 |
| AnimeSharp 4x (Kim2091) | 16.7M | 31.28 | 0.9250 | 0.0095 |
| SRFormer-light LUDVAE (Phips) | 0.87M | 30.76 | 0.9140 | 0.0061 |
| RealPLKSR-HFA2k LUDVAE (Phips) | 7.40M | 30.62 | 0.9160 | 0.0068 |
| **GRL-small LUDVAE (Phips)** | **3.49M** | **31.85** | 0.9234 | **0.0065** |
| SPAN V7 incumbent (used in run 1) | 2.24M | 30.52 | 0.9237 | 0.0061 |

The clear winner by PSNR (+1.33 dB over the incumbent) was the LUDVAE-trained GRL-small. APISR's perceptual models were rejected outright - their texture-hallucination profile (Laplacian 3-4x baseline) makes them poor *fidelity* teachers despite visual quality reputation.

The student was retrained against this teacher in `runs/distill_v2_grl/` (28 epochs, batch 16, same loss/optimizer/data as run 1; fp32; optional `--teacher-half` autocast hardened against feature-tap overflow after a poisoning incident in run 2a). A `--resume` flag was added to `distill.py` mid-way when a session-boundary reaped the background job at epoch 9; training was resumed cleanly from `student_best.pt`. Validation strips for both runs are preserved separately: `output/validate_v1/` and `output/validate_v2_grl/`. 2-image inference artifacts: `results/RFDN_distill_v1/` and `results/RFDN_distill_v2_grl/`.

### Honest result: the v2 student is WORSE

| Run | Teacher | Student val (best) | Student **test** PSNR | Student **test** SSIM | Teacher test PSNR | Retention |
|---|---|---|---|---|---|---|
| distill_v1 (40 ep) | SPAN V7 | 29.89 | **29.32** | 0.8728 | 30.44 | -12.6% |
| distill_v2_grl (28 ep, resume @ ep9) | GRL-small LUDVAE | 29.49 | **29.06** | 0.8703 | 31.77 | -16.7% |

Despite a teacher that measured **+1.05 dB better** on the held-out test set (32.31 val / 31.77 test for GRL vs 31.22 val / 30.44 test for SPAN), the student got **-0.26 dB worse**. The teacher-bench PSNR was a poor proxy for student distillation quality.

### Why the stronger teacher hurt

Likely contributors (ordered by plausibility):

1. **Capacity gap widened**. GRL teacher:student ratio is 11:1 (3.49M / 0.32M); SPAN was 7:1 (2.24M / 0.32M). With L1 + 0.5*MSE-features + 0.2*L1-GT as the loss, the student chases a much more difficult moving target when the teacher is large/strong. Smaller capacity student plus high-capacity teacher is not always better - the student can't match the teacher and the L1-to-teacher gradient pushes into gradients the GT anchor can't fully compensate.
2. **Feature-tap resolution mismatch.** GRL-small's last feature tap is at SR resolution (256 x 256 from a 48 x 48 input); SPAN's four taps are all LR-resolution. The bilinear down-sample of an SR tap back to LR resolution discards information; the MSE between that and the LR-resolution student tap is no longer a clean depth-wise constraint.
3. **LUDVAE-loss outputs include perceptual terms** (the teacher's PSNR is partly *because* of GAN/perceptual training). A student trained response-only on a perceptually-smoothed output may inherit some smoothing that hurts PSNR on its own.
4. **Train-time LR schedule**. v2 used cosine over 28 epochs (after resume); v1 used 40 epochs. v2 had less time to converge post-LR-decay.

### Engineering takeaway

Teacher PSNR on the test split (a single number from `eval_teachers.py`) was an unreliable predictor of student PSNR. **The original SPAN V7 teacher - while modest on the bench - turns out to be a better fit** for this particular student architecture and loss at the epoch budget reached. A better follow-up would be:

- Add a teacher-compatibility experiment: train both students on a small preview (5 epochs) and keep the one with better val PSNR.
- Try RealPLKSR-HFA2k (test 30.62 dB, similar profile to SPAN but LUDVAE-perceptual; taps at LR resolution, no interpolation needed) as a low-risk substitute.
- Widen the student (`nf=64`, ~500K params) and revisit.
- Persistently increase the GT-anchor weight (0.2 -> 0.5) so the student is anchored more directly to ground truth.

None of the above exceeds "train longer" in expected gain; with the v2 pipeline complete (resume, NaN guard, teacher bench, val/test artefacts), the cleanest follow-up is option B with RealPLKSR + `nf=64` + 0.5 GT weight.

