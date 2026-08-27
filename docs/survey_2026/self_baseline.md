# Self Baseline — Anime Super-Resolution (2026-06)

**Project:** `E:\python projects\upscale_anime`
**Current "best" config:** `configs/finetune_neosr_span_v7_anime.yaml`
**Generated:** 2026-06-20
**Status:** v7 finetuning deferred to user GPU; v4 best + stock pretrained are our measured baselines.

---

## 1. Architecture

| Aspect | Value | Source |
|---|---|---|
| Backbone | SPAN (Swift Parameter-free Attention Network) — `NeosrSPAN` | `src/models/span/neosr_span.py` |
| Feature channels | 48 | `model.feature_channels: 48` |
| Upscale | 4x | `model.upscale: 4` |
| Bias | True | `model.bias: true` |
| Normalization | False (no `mean`/`img_range` subtraction) | `model.norm: false` |
| Attention block | `SPAB` (default) or `MambaSPAB` (opt-in `sab_type: mamba_v2`) | `model.sab_type: conv3xc` |
| Conv3XC | 1x1 sk + 1x1→3x3→1x1 conv path, fused into 3x3 eval_conv at inference | `Conv3XC.forward` |
| SPAB | 3x Conv3XC + SiLU + sigmoid attention | `src/models/span/neosr_span.py:130` |
| PixelShuffle upsampler | 4x | `nn.PixelShuffle(4)` |
| Param count | ~2.24M | measured at load |
| Warm-start | `pretrained/span_pix_pretrain_4x.pth` (Phhofm official pix-loss pretrain, 9MB) | `model.pretrained_path` |

Optional MambaIRv2 attention block (`model.sab_type: mamba_v2`):
- `MambaSPAB` wraps `MambaIRv2.ASSM` blocks.
- ~20K params per block vs ~285K for SPAB.
- Pure-PyTorch fallback for Windows (no Triton/mamba_ssm).
- Config: `mamba.{d_state: 16, num_tokens: 64, inner_rank: 32, mlp_ratio: 2.0}`.

---

## 2. Loss Stack (v7, two-phase)

| Loss | Phase 1 weight | Phase 2 weight | Type / Notes |
|---|---:|---:|---|
| Pixel (Charbonnier) | 1.0 | 0.5 | `losses.pixel_loss.CharbonnierLoss` |
| Perceptual (Twin VGG+ResNet) | 0.3 | 0.5 | `losses.twin_perceptual_loss.TwinPerceptualLoss` (delta=0.1, danbooru=0.5, vgg=0.5) |
| Line-art preservation | 1.0 | 1.0 | `losses.anime_losses.LineArtPreservationLoss` (multi-scale edge) |
| Flat-region preservation | 0.05 | 0.05 | `losses.anime_losses.FlatRegionPreservationLoss` |
| Frequency (DCT) | 0.05 | 0.15 | `losses.frequency_aware_loss.FrequencyAwareLoss` (block_size=8, hf_weight=1.5) |
| FDL (DINOv2 + sliced Wasserstein) | 0.5 | 0.5→0.75 (P2 ramp) | `losses.fdl_loss.FDLLoss` (num_proj=24, dino_variant=small, compute_every=4) |
| Wavelet-guided GAN | start_epoch=5, weight=0.5 | same (continuous) | `losses.wavelet_guided_loss.WaveletGuidedLoss` (hh_weight=2.0) |
| Adversarial (Relativistic GAN) | 0.0 (start_epoch=30) | 0.005 → 0.01 ramp | `losses.adversarial_loss.RelativisticGANLoss` (disc_lr=1e-4, no label smoothing) |
| Color consistency | disabled | disabled | `losses.anime_losses.ColorConsistencyLoss` |
| Gradient / temporal | disabled | disabled | `gradient_loss.py`, `temporal_loss.py` |

**Loss-accumulator caveat:** all `loss_dict` entries must be `torch.Tensor` or carry a `_cached` suffix + skip in the accumulator (per AGENTS.md "Loss accumulator tensor guard").

---

## 3. Training Schedule (v7)

| Stage | Epochs | Crop | Batch | LR | Notes |
|---|---:|---:|---:|---|---|
| Warmup | 0-5 | 128 | 64 | warmup cosine to 5e-5 | `warmup_epochs: 5` |
| Progressive crop | 1-15 | 128 | 64 | cosine | `progressive_crop.stages[0]` |
| Progressive crop | 16-35 | 192 | 64 | cosine | `progressive_crop.stages[1]` |
| Progressive crop | 36-80 | 256 | 64 | cosine | `progressive_crop.stages[2]` |
| Phase 1 (structure) | 1-40 | (above) | 64 | 5e-5 | No GAN, low frequency, low perceptual |
| Phase 2 (texture) | 41-80 | (above) | 64 | 5e-5 → 1e-7 | GAN ramps in starting at epoch 30 |
| Gradient accumulation | — | — | 64 effective | — | `gradient_accumulation_steps: 64` |
| EMA | — | — | — | — | `use_ema: true, ema_decay: 0.999` |
| Optimizer | AdamW | — | — | 5e-5 | base.yaml default |
| Scheduler | CosineAnnealingLR | — | — | — | with min_lr=1e-7 |
| Total epochs | 80 | — | — | — | `training.finetune.epochs: 80` |
| Early stopping | LPIPS, patience=15 | — | — | — | `early_stopping` |

---

## 4. Degradation Pipeline (v7)

| Stage | Settings | Source |
|---|---|---|
| Stage 1 compression | jpeg, webp | `data.degradation.compression_stage1` |
| Stage 2 compression | avif, h264, h265, jpeg (prob=0.8) | in-process via `src/data/compression_modules.py` (PyAV + pillow-heif) |
| Blur | sigma [0.1, 3.0], prob=0.8 | `degradation.blur_sigma` |
| Noise | sigma [0, 30], prob=0.6 | `degradation.noise_sigma` |
| JPEG | quality [50,60,70,80,90], prob=0.6 | `degradation.jpeg_quality` |
| Second-order | prob=0.4 | re-applies stage 2 |
| Mode | `anime_heavy` (APISR-tuned) | `degradation.mode: anime_heavy` |
| Shuffled resize | True | `degradation.shuffled_resize: true` |
| Degrade-before-crop | True | `degradation.degrade_before_crop: true` |
| XDoG pseudo-GT | USM 3-rounds + XDoG (sigma=0.6, k=2.5, gamma=0.97, eps=-15, phi=1e9) + connected-component cleanup + passive dilation | `data.preprocessing.line_enhancement.pseudo_gt_mode: apisr` |

---

## 5. Hardware & Cost (AGENTS.md + v7_results.md)

| Metric | Value |
|---|---|
| GPU | RTX 4000 Mobile, 8GB VRAM |
| CPU fallback | Yes (for tests) |
| Per-iter time | ~225ms at crop=128, batch=4 (v7 measured) |
| Full 80-epoch run | ~10-15 hours |
| 2-epoch smoke | ~3-5 minutes |
| BF16 autocast | Yes (Ampere Tensor Core compatible) |
| Gradient checkpointing | Off (we have VRAM headroom) |

---

## 6. Data

| Dataset | Path | Notes |
|---|---|---|
| Anime faces | `data/anime_faces` | train |
| Anime val | `data/val_hr` | 94 frames from mp4upload, **train/val overlap caveat** |
| Anime HR v6 | `data/anime_hr_v6` | deduplicated corpus |
| Anime video frames | `data/anime_video_frames` | training data |
| Held-out | `data/anime_hr_holdout/` | **planned, not yet created** (Phase G.3) |

---

## 7. Measured Baseline Scores (N=10, val_hr, 480x480 center-crop)

From `docs/v7_results.md` Section 3, mean +/- std:

| Metric | Bicubic /4->x4 | Stock pretrained (span_pix) | v4 finetune best (ep51) | v4 finetune ep100 | HR ceiling | APISR target |
|---|---:|---:|---:|---:|---:|---:|
| CLIPIQA  | 0.597 +/- 0.076 | **0.667** +/- 0.073 | 0.632 +/- 0.106 | 0.624 +/- 0.104 | 0.608 +/- 0.084 | >= 0.65 |
| MANIQA   | 0.308 +/- 0.077 | **0.422** +/- 0.072 | 0.334 +/- 0.057 | 0.337 +/- 0.059 | 0.371 +/- 0.038 | >= 0.48 |
| NIQE     | 9.355 +/- 1.216 | 7.681 +/- 1.058 | **7.123** +/- 1.042 | 7.225 +/- 1.139 | 6.054 +/- 0.927 | <= 7.5 |
| TOPIQ_NR | 0.321 +/- 0.073 | **0.556** +/- 0.090 | 0.462 +/- 0.122 | 0.464 +/- 0.122 | 0.558 +/- 0.053 | (n/a) |
| PSNR | — | 36.81 +/- 4.03 | 35.89 +/- 2.00 | **40.59** +/- 3.38 | — | higher |
| SSIM | — | 0.9899 +/- 0.0027 | 0.9930 +/- 0.0016 | **0.9946** +/- 0.0012 | — | higher |
| LPIPS | — | 0.0229 +/- 0.0051 | 0.0104 +/- 0.0024 | **0.0079** +/- 0.0016 | — | lower |

**Gaps vs APISR targets:**
- CLIPIQA: stock clears (0.667 >= 0.65).
- MANIQA: ALL miss (0.422 best, target 0.48 — gap 0.058).
- NIQE: v4 best clears (7.123 <= 7.5).
- LPIPS: v4 best clears (0.0104 <= 0.10).

**The MANIQA gap is the obvious target for v7+ work.**

---

## 8. What we DON'T have (pre-survey)

Based on a read of the codebase (not yet a survey), candidates the v7 stack does not include:

- **Transformer backbones**: HAT, DRCT, DAT2, RGT, ATD, SAFMN, SwinIR-Light (we have SwinIR-x4 as a teacher only)
- **MoE / dynamic routing**: SeemoRe, MoSR
- **Dynamic upsampler**: Dysample (we use fixed PixelShuffle)
- **RealPLKSR-style degradation**: pixel-unshuffle shuffle, layer-norm-in-architecture
- **DINOv3 perceptual loss**: we have DINOv2 FDL
- **Diffusion SR**: SUPIR, DiffBIR, PASD (out-of-scope for training)
- **GAN inversion**: pSp, e4e (narrow, anime face only)
- **Test-time augmentation**: 8x flip/rot ensemble at inference
- **Model souping**: weight averaging across checkpoints
- **Hard-example mining**: perceptual-difficulty curriculum
- **SAM / Lookahead optimizers**
- **MAE pretraining for SR**

These are the survey targets for Phase 1-3.

---

## 9. Files of interest for survey

| File | Purpose |
|---|---|
| `configs/finetune_neosr_span_v7_anime.yaml` | canonical v7 config (272 lines) |
| `configs/base.yaml` | default values |
| `src/models/span/neosr_span.py` | `NeosrSPAN` (359 lines) |
| `src/models/span/mambair_v2.py` | `MambaSPAB`, `selective_scan_ref`, `ASSM` |
| `src/models/span/parameter_free_attention.py` | SwiftParameterFreeAttentionBlock (alt) |
| `src/models/span/span_model.py` | original SPAN/SPANTiny/SPANF (alternative backbone) |
| `src/training/neosr_finetuner.py` | 1768 lines — main trainer |
| `src/training/ema.py` | GeneratorEMA |
| `src/data/degradation_pipeline.py` | `DegradationPipeline` (presets: bicubic/light/medium/heavy/anime/anime_heavy) |
| `src/data/compression_modules.py` | in-process codecs (PyAV, pillow-heif, imageio_ffmpeg fallback) |
| `src/data/line_enhancement.py` | USM, XDoG, pseudo-GT |
| `src/data/preprocessing_manager.py` | orchestrator (5 modes) |
| `src/losses/*.py` | 15 loss modules |
| `src/utils/metrics.py` | PSNR/SSIM/LPIPS/NIQE/MANIQA/CLIPIQA/TOPIQ/MUSIQ + `PYIQA_DIRECTION`/`PYIQA_RANGE` |
| `scripts/compare_checkpoints.py` | side-by-side eval harness |
| `docs/v7_results.md` | baseline measurements |
| `docs/plans/v7_anime_roadmap.md` | v7 build plan |
| `docs/APISR_DEGRADATION_SPEC.md` | pre-v7 APISR research spec |

