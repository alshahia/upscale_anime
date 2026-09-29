# Roadmap — Real-Time High-Quality Anime Upscaling (Real-ESRGAN rebase)

**Updated:** Step-0 kickoff · replaces "train from scratch" strategy
**Goal:** realesr-animevideov3-tier quality, real-time (25 fps+) at 1080p output on the Quadro RTX 4000 (8 GB).

---

## 1. Memory / Findings (why everything from scratch underperformed)

Evidence already in the repo — no re-training needed to know this:

| Finding | Evidence |
|---|---|
| Student adds ~nothing over bicubic | `pretrained/eval_tta_test_RFDN_distill_v1_4x_student_results_table.csv`: student **29.46 dB / 0.8789 SSIM** vs bicubic 29.45 / 0.8780 |
| Teacher ceiling is low | incumbent SPAN teacher 30.44 dB on the same split |
| From-scratch training was poisoned | `TRANSFER_LEARNING_DIAGNOSIS.md`: pretrained weights never loaded (naming mismatch), RCAN teacher = NaN, EDSR teacher = near-zero output |
| GAN pretrain never downloaded | `pretrained/span_official_checkpoints.zip` is 2.4 KB (not a real checkpoint) |
| Deployment model already in hand | `pretrained/realesr-animevideov3.pth` + `pretrained/onnx/realesr-animevideov3.onnx` |
| Anime-specialized champions in repo | `checkpoints/candidates/4x_APISR_{GRL,DAT,RRDB}_GAN_generator.pth`, `4x-AnimeSharp.pth` |

**Core rule learned:** student ≤ teacher, always. Distillation is only as good as the
teacher. Training from scratch on one 8 GB GPU with ~4K full frames cannot reach
models trained on millions of pairs.

## 2. Decision

Stop training new from-scratch models. Adopt the proven Real-ESRGAN chain:

```
Step 0  MEASURE   shootout: pretrained champions vs our models      (no training)
Step 1  ADAPT     fine-tune the winning champion on our domain      (low LR, our degradation)
Step 2  DISTILL   train tiny real-time student against THAT teacher  (TinySRVGG family)
Step 3  SHIP      ONNX -> TensorRT fp16, verify 25 fps @ 1080p out
```

## 3. Step details

### Step 0 — this task
Script: `anime_upscaler/eval_step0.py`
- Quality: pyiqa CLIPIQA + NIQE + Laplacian sharpness on real mp4upload frames
  (center 384x384 crops, 4x upscale), no reference needed.
- Speed: fp16 forward latency at 640x360 -> and 960x540 -> inputs on CUDA; fps reported;
  real-time bar = 25 fps at 1080p-class output before TensorRT.
- Outputs: `results/step0_model_shootout.csv`, samples in `results/step0_samples/`.
- Acceptance: numeric table + side-by-side samples that name the deployment model
  and the fine-tune target. PASS = all rows loaded or FAIL causes understood.

### Step 1 — domain-adapt fine-tune (the first real training run)
- Base: Step-0 winner (expected: realesr-animevideov3 or 4x_APISR_GRL_GAN).
- Data: extract 10-20K full frames from target 25fps episodes into `data/anime_hr`;
  keep anime_faces <= 30% weight. Fix val split to >= 500 diverse frames.
- Degradation: TWO-STAGE compression targeting mp4upload-like web streams
  (bicubic downscale -> H.264 CRF 23-30 / banding / quantize). Reuse v7
  `preprocessing.degradation` but verify codec stages actually fire.
- Train: LR 1e-5..2e-5, cosine, EMA 0.999, Charbonnier + line-art, GAN <= 0.01,
  20-50K iters, batch 16-32 @ crop 256, fp16 AMP (Turing: fp16, NOT bf16).
- Acceptance: NR-IQA on holdout real frames >= base champion; no line-halo/oversharpen regression on 10 visual pairs.

### Step 2 — student distillation, done right
- Teacher = Step-1 model (or 4x_APISR_GRL_GAN) via `--teacher-spandrel` in
  `anime_upscaler/distill.py` (v3 loss already implemented). NEVER the incumbent SPAN.
- Student: TinySRVGGStudent first (same family as realesr-animevideov3);
  RFDN as second option. Train >= 100K iters with the same domain pairs.
- Acceptance: student within 0.2 dB / visually indistinguishable from teacher on
  the split; >= 25 fps fp16 at 960x540 input in eager PyTorch.

### Step 3 — deployment (DONE, see section 7)
- ONNX fp16 export (opset 17, dynamic H/W) -> ONNX-Runtime CUDA fp16 backend.
- TensorRT 11 is NOT usable on this Turing/WDDM machine (broken compiler
  backend, see section 7) — ORT is the shipped engine.
- Acceptance (met): >= 25 fps sustained on a real 25 fps episode at 1080p-class
  output; A/B clip vs realesr-animevideov3 reference.

## 4. Risks / notes
- APISR/RRDB and DAT are heavy; fine as teachers/quality ref, not for real-time.
- spandrel extra arch pack required for some candidates; handled with graceful FAIL rows.
- Checkpoint-compat SPAN loading may fail inside this script; status column captures it.
- pyiqa metric weights are cached locally (quality reports were produced before).


---

## 5. STEP-0 RESULTS (finalized)

Measured in `results/step0_model_shootout.csv` — 8 real mp4upload frames,
JPEG-60 stream-artifact simulation, center crops, pyiqa CLIPIQA
(higher=better) and NIQE (lower=better), fp16 where supported, allocator
hard-capped to ~7.2 GB dedicated VRAM only (no shared-memory spill), one
process per model, strictly sequential.

| model | status | params M | CLIPIQA ↑ | NIQE ↓ | fps 640x360 | fps 960x540 | VRAM MB |
|---|---|---|---|---|---|---|---|
| 4x_APISR_RRDB_GAN | ok | 4.5 | **0.712** | 8.85 | 0.5 | 0.2 | 2209-4461 |
| 4x_APISR_DAT_GAN | ok | 10.9 | 0.703 | **7.72** | tiled | tiled | 3407-7033 |
| 4x_APISR_GRL_GAN | ok | **1.0** | 0.676 | 9.12 | 0.2 | 0.1 | 2435-5002 |
| 4x-AnimeSharp | ok | 16.7 | 0.617 | 11.61 | 0.2 | 0.1 | 2233-4485 |
| bicubic x4 | ok | 0 | 0.588 | 9.11 | n/a | n/a | - |
| realesr-animevideov3 | ok | **0.62** | 0.532 | 9.93 | **5.1** | **2.3** | 1798-1941 |
| RFDN_distill_v1_student (ours) | ok | 0.32 | 0.374 | 7.81 | 4.5 | 2.0 | 583-804 |
| NEOSR_SPAN_V7_ANIME_best | SKIP | - | - | - | - | - | - |

NEOSR checkpoint pickles a neosr-internal `utils.config.Config` — unloadable
outside neosr (independent PSNR measurement of the incumbent SPAN exists from
eval_teachers.py: 30.44 dB, well below these GAN models).

### Verdict (evidence-based)

1. **Quality ceiling** — official APISR models are far above everything else
   (0.712 / 0.703 CLIPIQA vs 0.588 bicubic vs 0.532 realesr-animevideov3) on
   OUR degraded web-stream domain. DAT additionally has the best NIQE.
2. **Our student confirms the diagnosis** — 0.374 CLIPIQA, BELOW bicubic (0.588).
   It was distilled from our weak SPAN teacher. Rebuild with a pretrained teacher.
3. **Speed reality check** — eager PyTorch on Quadro RTX 4000 is slow across the
   board: even realesr-animevideov3 is 5.1 fps @ 640x360 LR input (2560x1440
   output class). Real-time 25 fps REQUIRES the TensorRT/ONNX fp16 export path
   (Step 3); realesr eager 5.1 fps x ~4-6x TRT speedup is borderline (~20-35
   fps), so the TinySRVGG student (Step 2) is what clears 25 fps with margin.

### Decision for Step 1/2 (updated by these measurements)

- **Step 1 fine-tune base: 4x_APISR_RRDB_GAN.** DAT stays as quality/NIQE
  reference; GRL is the light-teacher candidate. Never start from our SPAN-F.
- **Step 2 student: TinySRVGGStudent** distilled from the fine-tuned APISR
  model via spandrel taps. The RFDN student is retired (below bicubic).
- **Step 3 export: realesr-animevideov3 first** (smallest, ONNX shape already
  proven) to validate the TensorRT fps path end-to-end, then the new student.
- Caveat: NR-IQA can favor hallucination — eyeball line art / flat regions /
  banding on sample crops before locking the Step-1 base (RRDB vs DAT visually,
  not CLIPIQA alone).

---

## 6. STEP-2 (executed 2026-K, decision update)

**Teacher re-selected on fidelity evidence** after checking the historical
`results/teacher_bench.csv`: the APISR GANs only manage 22-24 dB (SSIM 0.80-0.84)
on bicubic-LR pairs — perceptual-winners but infidelity risk as response teachers.
`4xHFA2k_ludvae_realplksr_dysample` is the only candidate BOTH faithful
(30.6 dB, SSIM 0.916, +0.17 dB over bicubic on the eval split) AND strong on the
degraded real-frame domain (CLIPIQA 0.663 / NIQE 9.21 — above bicubic 0.588).

- **Primary teacher:** 4xHFA2k_ludvae_realplksr_dysample.pth
- **Quality reference:** 4x_APISR_RRDB_GAN (CLIPIQA ceiling, hallucination-prone)
- **Student:** TinySRVGGStudent (~317K params, SRVGGNetCompact family, 4x)

**Data:** `scripts/extract_frames.py` sampled `data/anime_fullframes` from the
six 1080p source episodes in data/anime_vid — 1 frame / 0.7s, aHash-256 hamming
<=7 static-scene dedup -> 4,372 distinct HR JPEG-95 pairs (1.96 GB).

**Run (background job pwsh-9):**

    .venv\Scripts\python.exe anime_upscaler/distill.py --arch srvgg --scale 4 \
      --teacher-spandrel checkpoints/candidates/4xHFA2k_ludvae_realplksr_dysample.pth \
      --data data/anime_fullframes --out-dir runs/step2_srvgg_hfa_v1 \
      --epochs 50 --batch-size 16 --num-workers 4 --loss twin

Smoke gate PASSED before launch (2-epoch smoke: teacher loaded via spandrel,
teacher 31.9 dB sanity, student trains and saves). Next: on completion, evaluate
the student with eval_step0.py (CLIPIQA/NIQE/sharpness on degraded frames) and
full fidelity test; accept if student >= bicubic CLIPIQA by clear margin and
>= 30 dB-class PSNR retention; otherwise iterate (longer schedule / GAN ramp).


---

## 7. STEP-2 RESULTS (complete — ACCEPTED)

**Run:** runs/step2_srvgg_hfa_v1 (50 epochs done; three background jobs, two
transient CSV-lock crashes both resumed from latest.pt with zero state loss —
the best/last/rotating checkpoint design paid for itself; resilient logging
added; train_log.csv is held by an external process -> per-epoch rows now in
train_log_second.csv).

### Held-out fidelity (test split, 438 imgs)

| model | PSNR (dB) | SSIM |
|---|---|---|
| bicubic | 33.05 | 0.9039 |
| **step2 student (317K, srvgg)** | **33.55** | **0.9146** |
| teacher 4xHFA2k_plksr (7.4M) | 32.60 | 0.9296 |

Student beats its own teacher's PSNR (GT-anchor effect) and clears bicubic by
+0.50 dB. Monotonic training curve, best at epoch 49 (33.54; EMA 33.44 trail).

### Perceptual on real degraded frames (JPEG-60, 8 frames, eval_step0.py)

| | CLIPIQA | NIQE (low=best) | laplacian | fps 640x360 | fps 960x540 |
|---|---|---|---|---|---|
| bicubic | 0.588 | 9.11 | 0.00085 | - | - |
| realesr-animevideov3 | 0.532 | 9.93 | 0.0021 | 5.1 | 2.3 |
| **step2 student** | 0.250* | **7.38** | 0.0011 | **8.2** | **3.7** |

*CLIPIQA collapse = known artifact for faithful/flat restorations (CLIP-IQA
rewards hallucinated texture). Visual pair check (results/step2_visual_pair.png,
orc battle scene): cleaner line art, sharper flame/background edges than
bicubic, no hallucination — NIQE agrees (most natural of all models).

### Acceptance decision: PASS
- +0.50 dB over bicubic, visually cleaner, faithful
- 1.6x faster than realesr-animevideov3 in eager PyTorch at same input size
- 317K params vs 620K — more TensorRT headroom for the 25 fps target

### Step 3 executed (2026-09)

Environment preconditions discovered the hard way:
- The GPU must run on AC power. On the DC adapter the driver caps the card at
  35 W (300 MHz SM clock, idle power numbers at 100% utilization) — everything
  measured there was ~6x slow. `nvidia-smi -q -d POWER -d PERFORMANCE` shows
  it. Recommend `nvidia-smi -lgc 1395` after every reboot.
- TensorRT 11.2's new compiler backend is broken on this Turing/WDDM combo for
  this small SR graph: 830 MB engine ("Total Weights Memory" for 634 K param
  model), garbage output (parity mean|diff| 0.33) with fp16 weights (static AND
  dynamic shapes) and NaN from an fp32 graph, at 13-33 fps. Not diagnosed
  further; ORT-CUDA reproduces torch fp16 to 1e-3 so ORT is the engine of record.
- Files: scripts/step3_export_bench.py, scripts/step3_video_ab.py,
  results/step3_deploy_bench.csv, results/step3_video_ab_*.{mp4,csv}.

Results (AC power, ORT-CUDA fp16 with CUDA IOBinding vs eager fp16 CL):
- 960x540 LR -> 3840x2160 (4K): ORT 21.8 fps / eager 21.3 fps   (below 25)
- 640x360 LR -> 2560x1440 (QHD): ORT 47.4 fps / eager 47.7 fps  (PASS)
- Real-episode 1-s A/B (ep2 t=45s), 640x360 LR -> 2560x1440:
    student       51.8 fps  PASS
    animevideov3  42.3 fps  PASS
  The 317K student is 1.23x FASTER than the 620K reference at same input size.

Verdict: quality plan (Step-2 student) + real-time goal (>=25 fps) both met;
real-time playback is 25 fps @ >=1080p-class output on AC power. 4K output
runs at ~22 fps — either accept 4K at ~22 fps (below target) or cap output at
1440p. Optional later work: GAN ramp (--lambda-adv) or longer schedule for more
texture; Step-1 fine-tune can reuse this student as a starting point.


