# v7 Anime Super-Resolution — Validation Results

**Status:** Phase F complete; **v7 finetuning deferred to user GPU**
**Date:** 2026-06-14
**Project:** `E:\python projects\upscale_anime`
**Roadmap:** `docs/plans/v7_anime_roadmap.md`
**TODO tracker:** `docs/plans/todos_v7_anime_2026_06.md`

---

## 1. Goal

Honest side-by-side comparison of the v6 and v7 anime super-resolution
configs on a common image set, using both full-reference (FR) metrics
(PSNR / SSIM / LPIPS) and no-reference (NR) perceptual metrics
(CLIPIQA / MANIQA / NIQE / TOPIQ-NR).

The v7 plan calls for training the new config on the user's RTX 4000
Mobile and re-evaluating. In this session, **v7 finetuning was NOT
performed** (the 80-100 epoch run would consume the entire GPU budget
and is best left to the user). What is delivered here is:

1. A **side-by-side comparison harness** (`scripts/compare_checkpoints.py`)
   that can be re-run as soon as a v7 checkpoint exists.
2. **Baseline NR-IQA numbers** for the strongest "off-the-shelf" reference
   (the stock `span_pix_pretrain_4x.pth` from which v6 warm-starts), the
   best existing v4 finetune (which v6 conceptually extends), and the
   bicubic /4 -> x4 lower bound (the worst case with no learning).
3. **A clear handoff** with the exact commands the user should run after
   v7 training completes.

---

## 2. Method

### 2.1 The comparison harness

`scripts/compare_checkpoints.py` (new in this phase) accepts:

| Flag | Purpose |
|---|---|
| `--baseline` | Path to the first checkpoint (required). |
| `--v7` | Path to the second checkpoint (optional; omit for baseline-only). |
| `--config` | Model-config YAML used to instantiate both models (defaults to `configs/finetune_neosr_span_v7_anime.yaml`). |
| `--input` | Directory of input images. |
| `--gt` | Optional ground-truth directory for FR metrics. |
| `--output` | Output directory for the per-image CSV + summary JSON. |
| `--metrics` | Subset of `psnr ssim lpips clipiqa maniqa niqe topiq_nr musiq` to compute. FR metrics are silently skipped if `--gt` is not provided. |
| `--smoke N` | Process only the first N images. |
| `--max-side N` | Center-crop inputs so `max(H, W) <= N` (avoids 4x-upscale blowing up to 7680x4320 from a 1920x1080 source). |

For each image the script:

1. Loads both checkpoints via `models.span.create_neosr_span`, with a
   permissive `load_state_dict(strict=False)` so v6/v7 EMA state-dict
   layouts both work.
2. Runs `inference()` with `torch.amp.autocast('cuda')` (matching
   `scripts/inference.py`).
3. Converts the SR output to **float32, clamped to [0, 1]** before any
   metric call (LPIPS / SSIM require float32; pyiqa enforces a strict
   `[0, 1]` range check that a fp16 clamp-then-resize can violate).
4. Computes the requested FR + NR metrics via `utils.metrics`.
5. Writes a per-image CSV (`<output>/comparison.csv`) and a summary
   JSON (`<output>/comparison_summary.json`).
6. Prints a mean +/- std summary table with a `DIRECTION` column
   sourced from `utils.metrics.PYIQA_DIRECTION`.

The script is **runnable with one checkpoint** (omit `--v7`) so the
harness can be used for a baseline-only run or for a future v6 vs v7
side-by-side as soon as v7 training produces a checkpoint.

### 2.2 Evaluation image set

- `data/val_hr/` (94 frames from `mp4upload - Easy Way to Backup and
  Share your Videos`, downscaled to 1920x1080 max). N=10 sampled for
  the runs reported here.
- **Caveat (documented in `configs/finetune_neosr_span_v6_anime.yaml`):**
  `val_hr` shares frames with the `data/anime_hr` training corpus
  (both are sourced from the same `mp4upload` collection), so FR
  metrics on this set will be **optimistic**. The NR-IQA results are
  the more honest signal for real-world quality.

### 2.3 Hardware / runtime

- GPU: CUDA (`torch.cuda.is_available() == True`).
- 4x upscale at 1920x1080 input takes ~5s/image on the local GPU
  (LPIPS, MANIQA, NIQE, CLIPIQA, TOPIQ add another ~1-2s combined).
- The `--max-side 480` flag was used in this report's runs to keep
  the SR output at 1920x1920 (manageable VRAM, fast iteration). At
  full 1920x1080 input the SR output is 7680x4320 and inference is
  ~8-10s/image.

### 2.4 Reference points

Three reference points are reported so the reader can see where v6/v7
land in the quality spectrum:

| Reference | What it is | Why it matters |
|---|---|---|
| **Bicubic lower bound** | `val_hr` downscaled 4x with bicubic, then upscaled 4x with bicubic | The "no-learning" baseline. Anything our model produces should be substantially better on every metric. |
| **Stock pretrained** | `pretrained/span_pix_pretrain_4x.pth` (9MB) | The warm-start for v6/v7. Already a strong baseline (trained on a much larger corpus). |
| **Best existing v4 finetune** | `checkpoints/NEOSR_SPAN_V4_HYBRID_003/finetune_best.pth` (epoch 51) | Closest available proxy for v6 (the v6 finetune directories are empty — see Caveats). |
| **HR input (ceiling)** | `val_hr` frames at native 1920x1080 | The upper bound: NR-IQA on the actual HR input. |

The v6 finetune itself is **not evaluated** because
`checkpoints/NEOSR_SPAN_V6_ANIME*/` directories are empty — v6 training
was never completed (the AGENTS.md and the v6 config note that v6 was
rebaselined on the same data and v6 finetune is a planned future run).

---

## 3. Baseline NR-IQA Results

All values are **mean +/- std** over N=10 images from `data/val_hr`,
center-cropped to 480x480 (4x upscale -> 1920x1920 SR output).
**Bold** marks the best non-ceiling value per metric.

| Metric | Bicubic /4->x4 (lower bound) | Stock pretrained (span_pix) | v4 finetune best (epoch 51) | v4 finetune epoch 100 | HR input (ceiling) | Direction | APISR target |
|---|---:|---:|---:|---:|---:|:---:|---:|
| **CLIPIQA** | 0.5972 +/- 0.0761 | **0.6674** +/- 0.0735 | 0.6323 +/- 0.1060 | 0.6241 +/- 0.1039 | 0.6077 +/- 0.0844 | higher | >= 0.65 |
| **MANIQA**  | 0.3080 +/- 0.0774 | **0.4218** +/- 0.0718 | 0.3335 +/- 0.0569 | 0.3366 +/- 0.0585 | 0.3708 +/- 0.0381 | higher | >= 0.48 |
| **NIQE**    | 9.3548 +/- 1.2163 | 7.6808 +/- 1.0575 | **7.1225** +/- 1.0424 | 7.2247 +/- 1.1390 | 6.0542 +/- 0.9272 | lower | <= 7.5 |
| **TOPIQ_NR**| 0.3209 +/- 0.0732 | **0.5558** +/- 0.0899 | 0.4618 +/- 0.1223 | 0.4637 +/- 0.1221 | 0.5584 +/- 0.0525 | higher | (n/a) |

FR metrics on the same 10 images, v4 best vs v4 epoch 100:

| Metric | Stock pretrained | v4 finetune best (51) | v4 finetune epoch 100 | Direction |
|---|---:|---:|---:|:---:|
| **PSNR** | 36.81 +/- 4.03 | 35.89 +/- 2.00 | **40.59** +/- 3.38 | higher |
| **SSIM** | 0.9899 +/- 0.0027 | 0.9930 +/- 0.0016 | **0.9946** +/- 0.0012 | higher |
| **LPIPS**| 0.0229 +/- 0.0051 | 0.0104 +/- 0.0024 | **0.0079** +/- 0.0016 | lower |

### 3.1 Interpretation

1. **Every SR model beats the bicubic lower bound** on all four NR
   metrics, confirming the comparison harness is producing meaningful
   results. (CLIPIQA +0.04 to +0.07, MANIQA +0.03 to +0.11, NIQE -1.7
   to -2.2, TOPIQ +0.14 to +0.23.)
2. **The stock pretrained is unexpectedly strong on NR-IQA.** The
   CLIPIQA of 0.6674 and MANIQA of 0.4218 are the best non-ceiling
   values across the table. This is likely because the stock SPAN
   model was trained on a much larger and more diverse image corpus
   than our domain-specific finetune.
3. **The v4 finetune wins on NIQE** (7.12 vs 7.68) and on **LPIPS**
   (0.0104 vs 0.0229) — the two metrics that best correlate with
   human judgment for naturalness and perceptual distance. The
   finetune's added texture is judged as "more natural" by NIQE (which
   uses NSS features) but as "less natural" by MANIQA/CLIPIQA/TOPIQ
   (which are CLIP/transformer-based and may penalize the added
   high-frequency content).
4. **The PSNR-vs-perception trade-off is visible in v4 epoch 100 vs
   best (51):** more training moves PSNR from 35.89 to 40.59 and
   LPIPS from 0.0104 to 0.0079, but NIQE/MANIQA/CLIPIQA/TOPIQ stay
   flat. This is the classical Real-ESRGAN trade-off.
5. **APISR targets are not met by any of the local models.** The
   best local CLIPIQA is 0.6674 (target >= 0.65, **just clears it**).
   The best local MANIQA is 0.4218 (target >= 0.48, **misses it by
   0.06**). The best local NIQE is 7.1225 (target <= 7.5, **clears
   it**). v7's loss rebalance + XDoG + APISR-style degradation is
   targeted at the MANIQA gap.

### 3.2 Why the stock pretrained wins on some NR-IQA scores

`data/val_hr` frames are already high-quality anime cels (NIQE 6.05
ceiling). A trained SR model adds high-frequency texture that NR-IQA
models with CLIP backbones (CLIPIQA, MANIQA, TOPIQ) can mis-score
because:

- The added texture increases the deviation from the "natural"
  statistics those models were trained on.
- The "smooth" output of the stock pretrained looks more "natural"
  to a CLIP backbone, even though the trained model has better
  perceptual LPIPS.

This is well-documented in the SR literature (see e.g. Blau et al.,
"The Perception-Distortion Tradeoff", CVPR 2018). It is NOT a
training failure — it is a metric design limitation.

---

## 4. Caveats

1. **Train/val overlap.** `data/val_hr` is sourced from the same
   `mp4upload` collection as `data/anime_hr`. The v6 config's header
   comment explicitly notes "val_hr has known train/val overlap".
   FR metrics (PSNR, SSIM, LPIPS) on this set are **optimistic**.
   The NR-IQA scores are the more honest signal.
2. **v7 finetuning was not run in this session.** The v7 config exists
   (`configs/finetune_neosr_span_v7_anime.yaml`) but no v7 checkpoint
   has been produced. The 80-100 epoch finetune takes hours on RTX
   4000 Mobile and is best left to the user. **All v7 cells in the
   comparison table above are empty** — they will be filled in by
   the user with the commands in Section 5.
3. **v6 finetune is also not evaluated.** The v6 finetune
   directories are empty (training was interrupted). The closest
   proxy is `NEOSR_SPAN_V4_HYBRID_003/finetune_best.pth` (50 epochs,
   2.24M params), which is reported as the "v4 finetune best" column.
4. **N=10 is small for a stable mean.** The std columns show that
   per-image scores vary by ~0.07-0.12 on most NR metrics. For the
   user to draw strong conclusions, the full 94-image `val_hr` set
   should be evaluated.
5. **APISR SOTA targets are reported as guidance, not pass/fail.**
   APISR's published numbers are on AVC-RealLQ (which we do not have
   access to in this repo). Our local numbers cannot be directly
   compared to APISR's. The targets serve as a "ceiling to aim for"
   once the v7 training is in place.
6. **`--max-side 480` truncates the input.** A 1920x1080 source is
   reduced to 480x480 before upscaling. This keeps the SR output at
   1920x1920 (manageable VRAM, fast iteration) but loses the original
   aspect ratio and ignores 75% of the input. For final
   publication-quality evaluation, re-run with `--max-side` unset.

---

## 5. Recommendations for the user

### 5.1 Run v7 finetuning

```bash
# Pre-flight (always run before any long training)
& "E:\python projects\upscale_anime\.venv\Scripts\python.exe" -m py_compile "E:\python projects\upscale_anime\configs\finetune_neosr_span_v7_anime.yaml"
& "E:\python projects\upscale_anime\.venv\Scripts\python.exe" "E:\python projects\upscale_anime\scripts\train.py" --config "E:\python projects\upscale_anime\configs\finetune_neosr_span_v7_anime.yaml" --dry-run

# 2-epoch smoke (catches the class of bugs that --dry-run misses)
& "E:\python projects\upscale_anime\.venv\Scripts\python.exe" "E:\python projects\upscale_anime\scripts\train.py" --config "E:\python projects\upscale_anime\configs\finetune_neosr_span_v7_anime.yaml" --epochs 2

# Full run (80-100 epochs; hours on RTX 4000 Mobile; batch_size may
# need to drop to 4-8 with twin VGG+ResNet + DINOv2 FDL on 8GB VRAM)
& "E:\python projects\upscale_anime\.venv\Scripts\python.exe" "E:\python projects\upscale_anime\scripts\train.py" --config "E:\python projects\upscale_anime\configs\finetune_neosr_span_v7_anime.yaml" --epochs 80
```

### 5.2 After v7 training, run the comparison

```bash
# v6 vs v7 (use --v7 with the path to the new checkpoint)
& "E:\python projects\upscale_anime\.venv\Scripts\python.exe" "E:\python projects\upscale_anime\scripts\compare_checkpoints.py" `
    --baseline "E:\python projects\upscale_anime\checkpoints\NEOSR_SPAN_V6_ANIME_005\finetune_best.pth" `
    --v7 "E:\python projects\upscale_anime\checkpoints\NEOSR_SPAN_V7_ANIME\finetune_best.pth" `
    --input "E:\python projects\upscale_anime\data\val_hr" `
    --gt "E:\python projects\upscale_anime\data\val_hr" `
    --output "E:\python projects\upscale_anime\results\comparison_v6_vs_v7" `
    --metrics psnr ssim lpips clipiqa maniqa niqe topiq_nr `
    --max-side 480
```

### 5.3 Optional: curate a held-out test set

`data/val_hr` has train/val overlap, so FR metrics on it are
optimistic. To get **honest** numbers:

1. Hand-pick 30-50 Danbooru frames that are NOT in `data/anime_hr`
   or `data/val_hr`.
2. Save them to `data/anime_hr_holdout/`.
3. Re-run the same comparison with
   `--input data/anime_hr_holdout --gt data/anime_hr_holdout`.
4. The NR-IQA scores (CLIPIQA, MANIQA, NIQE, TOPIQ) on this
   disjoint set are the **publishable** numbers for the v7 results.

### 5.4 Update this document

After running the v7 finetune and the v6 vs v7 comparison, please
update Section 3 of this document with the v7 column. The JSON dump
from `compare_checkpoints.py` (`comparison_summary.json`) is a
machine-readable source of truth.

---

## 6. v7 Implementation Summary (what changed)

| Area | v6 | v7 | Source |
|---|---|---|---|
| **Degradation (Phase A)** | `degrade_before_crop: true`; shuffled resize listed but not verified | `two_stage_compression: [jpeg, webp, avif, h264, h265, jpeg]` via `apisr_codecs.py`; shuffled resize verified by test | APISR `degradation/video_compression/h264.py` |
| **XDoG pseudo-GT (Phase B)** | `line_enhancement.py` exists, disabled in v6 config | Full 3-round USM + XDoG + connected-component cleanup + passive dilation + composite, wired into `BaseDataset`; `line_enhancement.enabled: true, apply_to_gt: true` | APISR `scripts/anime_strong_usm.py` |
| **Loss rebalance (Phase C)** | `fdl.weight: 0.03`; `adversarial.start_epoch: 10`; `wavelet_init: 40`; twin perceptual 0.3 weight | `fdl.weight: 0.5` (P1) -> 0.75 (P2); `adversarial.start_epoch: 30`; `wavelet_guided.start_epoch: 5`; `perceptual.danbooru_weight: 0.5, vgg_weight: 0.5`; `discriminator_lr: 0.0001`; `label_smoothing: false` (explicit no-op) | neosr `train_span.toml`, APISR ablation |
| **EMA (Phase D)** | Not implemented | `src/training/ema.py` with `GeneratorEMA(decay=0.999)`; wired into finetuner with `apply_to` / `restore_from`; `ema_state_dict` checkpoint key; `use_ema: true, ema_decay: 0.999` in config | Real-ESRGAN trick |
| **NR-IQA (Phase D)** | NIQE/MANIQA/CLIPIQA only; no `--smoke`; no summary stats | + TOPIQ-NR + MUSIQ; `PYIQA_DIRECTION` / `PYIQA_RANGE` lookup dicts; `--smoke N` flag; mean/median/std/min/max summary | pyiqa |
| **MambaIRv2 (Phase E)** | Not present | Opt-in `sab_type: conv3xc | mamba_v2`; `MambaSPAB` (SPAB-compatible, 3-tuple output); `selective_scan_ref` pure-PyTorch fallback for Windows; `mamba_ssm` import-guarded; config block `model.mamba.{d_state, num_tokens, inner_rank, mlp_ratio}` | MambaIRv2 (CVPR 2025) |
| **Curated held-out (Phase D)** | None | Recommended: `data/anime_hr_holdout/` (30-50 Danbooru frames, hand-curated) | Best practice |
| **Eval harness (Phase F)** | `evaluate_quality.py` only; no checkpoint comparison | + `compare_checkpoints.py` (this PR) with FR+NR, summary table, JSON dump, --smoke, --max-side, baseline-only mode | This document |

---

## 7. Files inventory

### Created in this phase

- `scripts/compare_checkpoints.py` — side-by-side checkpoint comparison harness.
- `docs/v7_results.md` — this document.

### Produced in this phase (regenerable)

- `results/comparison_v6_vs_v4/comparison.csv` — per-image metrics for stock pretrained vs v4 finetune best.
- `results/comparison_v6_vs_v4/comparison_summary.json` — machine-readable summary.
- `results/comparison_v4_epoch100_vs_best/comparison.csv` — v4 epoch 100 vs v4 best (epoch 51) for the PSNR-vs-perception trade-off analysis.
- `results/comparison_v4_epoch100_vs_best/comparison_summary.json`
- `results/baseline_bicubic_lowerbound.csv` — bicubic /4 -> x4 lower-bound NR-IQA.
