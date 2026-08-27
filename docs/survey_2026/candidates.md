# Candidates — Anime Super-Resolution (2026-06)

**Generated:** 2026-06-20
**Scope:** Top candidates from `docs/survey_2026/sources.md` worth deeper investigation. Each row is a candidate for adoption in v8+.
**Scoring legend:**
- **Effort**: S = <1 day, M = 1-5 days, L = >5 days of integration work.
- **Risk**: low / med / high.
- **Expected NR-IQA Δ on N=10 val_hr vs v7 baseline (qualitative)**: range based on published results.

---

## A. Architecture Candidates (NOT in v7)

| ID | Name | Source | Type | Effort | Risk | Expected Δ CLIPIQA / MANIQA / NIQE / LPIPS | Why relevant |
|---|---|---|---|---|---|---|---|
| A1 | **MambaIRv2 ASSM** as `sab_type: mamba_v2` | MambaIRv2 (CVPR 2025) | State-space attention | S | low | +0.01 / +0.02 / -0.1 / -0.001 | **Already opt-in in v7 — just flip the flag.** Beats SRFormer 0.35 dB with 9.3% fewer params. |
| A2 | **MambaIR (v1)** as additional `sab_type: mamba_v1` | MambaIR (ECCV 2024) | State-space attention | S | low | +0.005 / +0.01 / -0.05 / 0 | Drop-in for Model B. Beats SwinIR 0.45 dB. Simpler than v2. |
| A3 | **SPAN_fast** (32-ch, Mish, kaiming) | Phhofm `2xBHI_small_span_fast_pretrain` | SPAN variant | M | low | 0 / -0.01 / +0.05 / 0 | 2x inference speedup. NTIRE 2025 efficient-SR. Train 2x then 4x chain. |
| A4 | **SAFMN** as teacher candidate | SAFMN (ICCV 2023) | Lightweight ViT | M | med | n/a (teacher) / n/a / n/a / n/a | 3x smaller than IMDN. Train as a teacher for MTKD distillation. |
| A5 | **DRCT-S** as teacher candidate | DRCT (CVPRW 2024 NTIRE winner) | Dense-Residual Transformer | M | med | n/a (teacher) | Won NTIRE 2024 SR x4. Strong 4x PSNR ceiling. |
| A6 | **DAT2 / DAT** backbone swap | Phhofm `4xBHI_dat2_otf_nn` (2024) | Dense Aggregation Transformer | L | high | +0.02 / -0.01 / +0.2 / +0.001 | Heavy at 4x. May fit at crop=64 on 8GB. Already anime-tuned by Phhofm. |
| A7 | **HAT-Lite** as teacher candidate | HAT (CVPR 2023 ext.) | Hybrid Attention Transformer | L | high | n/a (teacher) | PSNR ceiling. HAT-L too heavy at 4x; HAT-Lite may fit. |
| A8 | **RealPLKSR + Dysample** backbone swap | Phhofm `2xPublic_realplksr_dysample_layernorm_real_nn` | PLKSR + dynamic upsampler | L | med | +0.005 / +0.005 / 0 / -0.001 | Newer arch line Phhofm is moving to. Dysample upsampler is the easy lift (see T2). |
| A9 | **MoSR / MoESR** as teacher candidate | MoSR (umzi2 2024) | Mixture-of-experts | L | med | n/a (teacher) | Newer SISR arch. |
| A10 | **AuraSR-v2** for AI-gen-style anime | fal/AuraSR-v2 (HF) | GigaGAN | M | high | varies | 0.6B params. Best for SD/AI-gen images. For hand-drawn anime: marginal — not Phhofm's anime-tuned. |

## B. Loss Function Candidates

| ID | Name | Source | Type | Effort | Risk | Expected Δ MANIQA / LPIPS / NIQE | Why relevant |
|---|---|---|---|---|---|---|---|
| B1 | **Patch-NCE / CUT** | Park et al. ECCV 2020 | Contrastive patch loss | M | low | +0.01 MANIQA / -0.002 LPIPS / 0 | Mutual info between LR/HR patches. Lightweight (no VGG/DINO). 5th perceptual term. |
| B2 | **Projected-GAN discriminator** (frozen VGG/DINO features in D) | Sauer et al. ICCV 2021 | Adversarial | M | med | +0.02 MANIQA / -0.003 LPIPS / -0.1 NIQE | D uses frozen DINOv2 features. Anime-aligned discriminator should reduce mode collapse on flat-color regions. |
| B3 | **DINOv3 perceptual loss** (replace DINOv2 in FDL) | DINOv3 (Meta 2024) | Feature loss | M | low | +0.01 MANIQA / -0.001 LPIPS / -0.05 NIQE | DINOv3 has stronger semantic features. Drop-in for FDL. Verify VRAM headroom first. |
| B4 | **VQ codebook feature loss** (anime-specific dictionary) | VQFR (ECCV 2022) | Codebook loss | L | med | +0.005 MANIQA / 0 / -0.05 NIQE | Anime has repeated tokens (eyes, line strokes). Could learn anime-specific VQ codebook. |
| B5 | **Total-Variation regularizer** | (classical) | Smoothness prior | S | low | 0 / 0 / -0.05 NIQE | Cheap regularizer. May smooth over-artifacts. |
| B6 | **Sobel/Laplacian edge loss** | (classical) | Edge preservation | S | low | 0 / 0 / 0 | Already in `gradient_loss.py` — verify it's wired up. |
| B7 | **LAB-space color consistency** | (custom) | Color | S | low | 0 / 0 / -0.1 NIQE | Replace our RGB color loss with LAB. May help MANIQA. |
| B8 | **CLIP-based semantic consistency loss** | SeeSR / SD-derived | Semantic | L | high | +0.01 MANIQA / 0 / 0 | Heavy (CLIP forward). Last resort for perceptual ceiling. |

## C. Training Technique Candidates

| ID | Name | Source | Type | Effort | Risk | Expected Δ | Why relevant |
|---|---|---|---|---|---|---|---|
| C1 | **Test-Time Augmentation (TTA)** — 8x flip/rot ensemble | EDSR / wtp | Inference | S | low | +0.1-0.3 dB PSNR / -5% LPIPS / +0.01 MANIQA | Free inference improvement. Implement in `inference.py` today. |
| C2 | **Model Souping** — weight average across v6/v7 bests | Wortsman et al. ICML 2022 | Inference | S | low | +0.005 / -0.001 / -0.05 | We have 7 finetune configs (v1-v7). Average `finetune_best.pth` weights. Cheap. |
| C3 | **VQD-SR learned degradation codebook** | VQD-SR 2023 | Degradation | M | med | +0.02 MANIQA / -0.002 LPIPS / -0.2 NIQE | Replace hand-crafted `DegradationPipeline` `anime_heavy` with learned codebook from real anime LQ. |
| C4 | **RealDGen-style learned degrader** | RealDGen (ICLR 2025) | Degradation | L | high | +0.01 MANIQA / -0.001 LPIPS / -0.1 NIQE | Diffusion-based realistic LR generation. Unsupervised. Replaces our synthetic pipeline. |
| C5 | **SAM optimizer** | Foret et al. ICLR 2021 | Optimizer | S | med | +0.1 dB PSNR / 0 / 0 | Sharpness-Aware Minimization. 2x compute per step. Useful for convergence phase. |
| C6 | **Lion optimizer** | Chen et al. 2023 | Optimizer | S | low | 0 / 0 / 0 | Single-state optimizer. Frees ~30% optimizer memory. Drop-in for AdamW. |
| C7 | **Lookahead optimizer wrapper** | Zhang et al. NeurIPS 2019 | Optimizer | S | low | +0.05 dB PSNR / 0 / 0 | Stabilizes training. Wrapper around AdamW. |
| C8 | **MAE pretraining for SR** | He et al. CVPR 2022 | Pretrain | L | med | +0.1 dB / +0.005 MANIQA / 0 | Masked autoencoder pretrain on anime HR before finetune. Heavy compute. |
| C9 | **Hard-example mining curriculum** | (custom) | Curriculum | M | med | +0.005 MANIQA / -0.001 LPIPS / -0.05 NIQE | Sample harder examples more often (e.g., low CLIPIQA from val set). |
| C10 | **3-phase GAN scheduler** | (custom) | Scheduler | S | low | +0.005 MANIQA / 0 / -0.1 NIQE | Add a Phase 3 for GAN-only fine-tuning at end. Stops GAN drift. |
| C11 | **OS-RealPLKSR degradation preset** | Phhofm (post-2024) | Degradation preset | S | low | +0.005 / 0 / -0.1 | Drop-in preset for our `DegradationPipeline`. Strict-IQA-filtered training pairs. |
| C12 | **AdaBound / RAdam** | (various) | Optimizer | S | low | 0 / 0 / 0 | Adaptive bound / rectification. Often no gain vs well-tuned AdamW. |
| C13 | **EMA of discriminator (D-EMA)** | (custom) | EMA | S | low | 0 / 0 / -0.05 NIQE | EMA already on G (v7). Add to D for stability. |
| C14 | **Mixed-precision BF16** | (PyTorch native) | Precision | S | low | 0 / 0 / 0 | **Already done in v7 Tier-1 perf PR.** |
| C15 | **Gradient accumulation re-tune** | (custom) | Tuning | S | low | 0 / 0 / 0 | Try grad_accum=32 vs 64 with batch=4. May speed up effective epoch time. |

## D. Warm-start / Pretrain Swap Candidates (drop-in replacement for `span_pix_pretrain_4x.pth`)

| ID | Name | Source | Effort | Risk | Expected Δ | Why relevant |
|---|---|---|---|---|---|---|
| D1 | `2xBHI_small_span_fast_pretrain` | Phhofm (G4) | S | low | 0 / +0.005 / 0 / 0 | 2x faster inference. Need 2x->2x chain (we'd need a 2x inference path). |
| D2 | `4xBHI_dat2_otf_nn` | Phhofm (G2) | L | high | +0.005 / -0.005 / -0.1 / -0.001 | Anime-tuned DAT2 OTF. Cold-start for v8 DAT2 arch. |
| D3 | `4xDRCT-mssim-pretrains` (drct-s) | Phhofm (G10) | L | med | +0.01 / +0.005 / -0.1 / -0.002 | drct-s may fit 8GB. Cold-start for v8 DRCT arch. |
| D4 | `4xmssim_mosr_pretrain` | Phhofm (G11) | L | med | varies | Photography focus, but mssim loss = same loss family as our warm-start. |
| D5 | `2xPublic_realplksr_dysample_layernorm_real_nn` | Phhofm (G12) | L | med | +0.005 / 0 / 0 / -0.001 | Newer RealPLKSR lineage. |

## E. Quick-Win Inference / API Improvements (no training)

| ID | Name | Source | Effort | Risk | Expected Δ | Why relevant |
|---|---|---|---|---|---|---|
| E1 | **TTA 8x flip/rot ensemble** | EDSR-T (CVPRW 2017) | S | low | +0.1-0.3 dB / +0.005 / -5% LPIPS | Free; integrate into `scripts/inference.py` and `scripts/compare_checkpoints.py`. |
| E2 | **Model soup of v6 + v7 best** | Wortsman et al. 2022 | S | low | +0.005 / -0.001 / -0.05 | Average `finetune_best.pth` from V6_ANIME_005 + V7_ANIME. |
| E3 | **Curated held-out test set** | AGENTS.md V7 plan | S | low | n/a (eval only) | Required for publishable NR-IQA. 30-50 Danbooru frames. |
| E4 | **Per-image metric dashboard** | (custom) | S | low | n/a (eval only) | Extend `compare_checkpoints.py` with per-image histograms. |

## F. Diffusion-Based Candidates (out-of-scope for training, in-scope for inference post-process)

| ID | Name | Source | Effort | Risk | Expected Δ | Why relevant |
|---|---|---|---|---|---|---|
| F1 | **DiffBIR post-process** | XPixel ICCV 2023 | M | med | +0.02 MANIQA / -0.005 LPIPS / +0.1 NIQE | Apply DiffBIR as a refinement step on v7 SR output. Slow but quality ceiling. |
| F2 | **OSEDiff one-step** | NeurIPS 2024 | M | med | +0.01 MANIQA / -0.003 LPIPS / -0.1 NIQE | One-step diffusion. More practical on 8GB than multi-step. |
| F3 | **SUPIR post-process** | SUPIR (CVPR 2024) | L | high | +0.03 MANIQA / -0.01 LPIPS / +0.2 NIQE | SOTA diffusion SR. 12s/image on RTX 4090. |

---

## Top 8 candidates by ROI (Ranked)

Based on integration effort vs expected NR-IQA gain:

1. **C1 + E1: TTA 8x flip/rot ensemble at inference** (S-effort, free +0.1-0.3 dB)
2. **E2: Model soup of v6 + v7 best checkpoints** (S-effort, free +0.005 MANIQA)
3. **A1: Flip v7 `sab_type: mamba_v2`** (S-effort, CVPR 2025 SOTA Mamba)
4. **B3: Swap DINOv2 -> DINOv3 in FDL loss** (M-effort, +0.01 MANIQA)
5. **C3: VQD-SR learned degradation codebook** (M-effort, +0.02 MANIQA)
6. **D1: Warm-start swap to `2xBHI_small_span_fast_pretrain`** (M-effort, 2x inference speed)
7. **B1: Patch-NCE contrastive loss** (M-effort, +0.01 MANIQA)
8. **C10: 3-phase GAN scheduler** (S-effort, +0.005 MANIQA)

---

## Top 5 by absolute quality ceiling (regardless of effort)

1. **A7: HAT-Lite** (L-effort, PSNR ceiling)
2. **A5: DRCT-S** (L-effort, NTIRE 2024 winner)
3. **F2: OSEDiff one-step post-process** (M-effort, +0.01 MANIQA)
4. **A6: DAT2** (L-effort, Phhofm anime-tuned)
5. **F3: SUPIR post-process** (L-effort, +0.03 MANIQA)

---

## Total candidates surveyed

- 10 architecture (A1-A10)
- 8 loss (B1-B8)
- 15 training technique (C1-C15)
- 5 warm-start swap (D1-D5)
- 4 inference / API (E1-E4)
- 3 diffusion post-process (F1-F3)
- **Total: 45 candidates**

