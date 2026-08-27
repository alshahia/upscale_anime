# Comparison Matrix — v8+ Candidates vs v7 Self-Baseline

**Generated:** 2026-06-20
**Inputs:** `docs/survey_2026/sources.md`, `docs/survey_2026/candidates.md`, `docs/survey_2026/deep_dives.md`, `docs/survey_2026/self_baseline.md`.

---

## v7 Self-Baseline (anchor)

| Metric | Value | Source |
|---|---|---|
| Backbone | SPAN 48ch, 4x | `configs/finetune_neosr_span_v7_anime.yaml` |
| Pretrain | `span_pix_pretrain_4x.pth` (Phhofm) | AGENTS.md, configs |
| Loss stack | Charbonnier + Twin VGG+ResNet + FDL/DINOv2 + Wavelet-guided + Relativistic GAN | `configs/finetune_neosr_span_v7_anime.yaml` |
| Degradation | APISR two-stage + shuffled resize + degrade-before-crop | `configs/finetune_neosr_span_v7_anime.yaml` |
| Schedule | 80 epochs, two-phase, progressive crop 128->192->256, batch 64 grad_accum 64, lr 5e-5 cosine | `configs/finetune_neosr_span_v7_anime.yaml` |
| EMA | Yes (decay 0.999) | `src/training/ema.py` |
| Per-iter time | ~225 ms @ crop=128, batch=4 | AGENTS.md v7 Tier-1 perf |
| Total run time | 80 epochs = ~10-15h on RTX 4000 Mobile | AGENTS.md |

**Measured N=10 val_hr scores:**
- Stock pretrained (span_pix): CLIPIQA **0.667**, MANIQA **0.422**, NIQE **7.681**, LPIPS **0.0229**
- v4 finetune best (ep51): CLIPIQA **0.632**, MANIQA **0.334**, NIQE **7.123**, LPIPS **0.0104**

**APISR targets:** CLIPIQA>=0.65, MANIQA>=0.48, NIQE<=7.5, LPIPS<=0.10

**The MANIQA gap:** stock 0.422, v4 best 0.334, target 0.48. **Gap = -0.058 to -0.146.**

---

## Master Comparison Matrix

Score legend: + = positive, - = negative, 0 = neutral, ? = uncertain, ↑↑↑ = very positive, ↓↓↓ = very negative.
Δ-NR-IQA columns show expected delta on N=10 val_hr (480x480 center-crop).

| # | Candidate | Type | Effort | Risk | Δ CLIPIQA | Δ MANIQA | Δ NIQE | Δ LPIPS | Δ train-time | Δ inference | Wall-clock cost | Notes |
|---|---|---|---|---|---:|---:|---:|---:|---|---|---|---|
| C1+E1 | **TTA 8x flip/rot** | Inference | S | none | 0 | +0.005 | 0 | **-10%** | 0 | **+800%** | ~1.8s/img at 1080p | PixelShuffle-safe. Free. Ship now. |
| E2 | **Model souping v6+v7** | Inference | S | low | 0 | +0.01 to +0.05 | -0.05 | -0.002 | 0 | 0 | ~5 min script | EMA average. LR/optimizer reset. |
| A1 | **MambaIRv2 (`sab_type: mamba_v2`)** | Arch | S | low (Linux) / high (Win) | +0.01 | +0.02 | -0.1 | -0.001 | **0 (Linux) / +1500% (Win)** | 0 | Linux only | CVPR 2025. Block swap. Inherits warm-start problem (needs DF2K cold-start). |
| B3 | **DINOv3 in FDL loss** | Loss | M | low | 0 | +0.01 | -0.05 | -0.001 | +5% | 0 | <1 day | Drop-in for DINOv2 in `src/losses/fdl_loss.py`. Verify VRAM. |
| C3 | **VQD-SR learned codebook** | Degradation | M | med | +0.005 | +0.02 | -0.1 | -0.002 | +15% | 0 | 1-2 days | Anime-specific codebook. Help on OOD LR. Marginal on val_hr. |
| C11 | **OS-RealPLKSR preset** | Degradation | S | low | 0 | +0.005 | -0.05 | 0 | 0 | 0 | 1 day | Add new preset to `DegradationPipeline`. |
| D1 | **`2xBHI_small_span_fast_pretrain` warm-start swap** | Pretrain | M | low | 0 | +0.005 | 0 | 0 | 0 | **-50%** | 1-2 days | Need 2x->2x chain. 2x faster inference. |
| B1 | **Patch-NCE loss** | Loss | M | med | +0.005 | +0.02 | -0.05 | -0.002 | +5% | 0 | 1 week | Needs unpaired LR. ~50 MB VRAM. |
| C10 | **3-phase GAN scheduler** | Schedule | S | low | 0 | +0.005 | -0.05 | 0 | 0 | 0 | <1 day | Add Phase 3 GAN-only finetune. |
| C5 | **SAM optimizer** | Optimizer | S | med | 0 | 0 | 0 | 0 | **+100%** | 0 | <1 day | Useful for final convergence phase only. |
| C6 | **Lion optimizer** | Optimizer | S | low | 0 | 0 | 0 | 0 | 0 | 0 | <1 day | Drop-in for AdamW. Saves ~30% optimizer memory. |
| C7 | **Lookahead wrapper** | Optimizer | S | low | 0 | +0.005 | 0 | 0 | +5% | 0 | <1 day | Stabilizes GAN. |
| C9 | **Hard-example mining curriculum** | Data | M | med | 0 | +0.005 | -0.05 | 0 | 0 | 0 | 1-2 days | Sample low-CLIPIQA more. |
| C13 | **D-EMA** | Schedule | S | low | 0 | 0 | -0.05 | 0 | +2% | 0 | <1 day | EMA on discriminator. |
| A2 | **MambaIR (v1) additional sab_type** | Arch | S | low (Linux) | +0.005 | +0.01 | -0.05 | 0 | 0 | 0 | <1 day | Beats SwinIR 0.45 dB. Simpler than v2. |
| A4 | **SAFMN as teacher candidate** | Arch (teacher) | M | med | n/a | n/a | n/a | n/a | 0 | 0 | 1-2 days | For MTKD distillation. |
| A5 | **DRCT-S as teacher candidate** | Arch (teacher) | M | med | n/a | n/a | n/a | n/a | 0 | 0 | 1-2 days | NTIRE 2024 winner. |
| A6 | **DAT2 backbone swap** | Arch | L | high | +0.02 | -0.01 | +0.2 | +0.001 | 0 | 0 | 2-4 weeks | Heavy at 4x on 8GB. |
| A7 | **HAT-Lite backbone swap** | Arch | L | high | +0.03 | 0 | +0.1 | 0 | 0 | 0 | 2-4 weeks | PSNR ceiling. Heavy. |
| A8 | **RealPLKSR + Dysample backbone** | Arch + upsampler | L | med | +0.005 | +0.005 | 0 | -0.001 | 0 | 0 | 2-4 weeks | Phhofm's newer arch line. |
| B2 | **Projected-GAN discriminator** | Loss | M | med | +0.005 | +0.02 | -0.1 | -0.003 | +5% | 0 | 1 week | D uses frozen DINOv2. Reduces mode collapse. |
| B4 | **VQ codebook feature loss** | Loss | L | med | 0 | +0.005 | -0.05 | 0 | 0 | 0 | 2 weeks | Anime-specific dictionary. |
| C4 | **RealDGen learned degrader** | Degradation | L | high | +0.005 | +0.01 | -0.1 | -0.001 | 0 | 0 | 2-3 weeks | Diffusion-based. Unsupervised. |
| C8 | **MAE pretraining for SR** | Pretrain | L | med | +0.005 | +0.005 | 0 | 0 | 0 | 0 | 2 weeks | MAE on anime HR before finetune. |
| F1 | **DiffBIR post-process** | Inference | M | med | 0 | +0.02 | +0.1 | -0.005 | 0 | +5000% | 1-2 days | Quality ceiling. Slow. |
| F2 | **OSEDiff one-step post** | Inference | M | med | 0 | +0.01 | -0.1 | -0.003 | 0 | +2000% | 1-2 days | One-step. More practical than DiffBIR. |
| F3 | **SUPIR post-process** | Inference | L | high | 0 | +0.03 | +0.2 | -0.01 | 0 | +10000% | 2-4 weeks | SOTA ceiling. 12s/img on 4090. |
| D2 | **`4xBHI_dat2_otf_nn` cold-start** | Pretrain | L | high | +0.005 | -0.005 | -0.1 | -0.001 | 0 | 0 | 2-4 weeks | Anime-tuned DAT2. For A6 arch swap. |
| D3 | **`4xDRCT-mssim-pretrains` cold-start** | Pretrain | L | med | +0.01 | +0.005 | -0.1 | -0.002 | 0 | 0 | 2-4 weeks | drct-s. For A5 arch swap. |

---

## Tier-Ranked Summary

### Tier 1: Quick wins (S-effort, no training, ship now)
1. **TTA 8x flip/rot** — LPIPS -10%, free at inference.
2. **Model souping v6+v7** — MANIQA +0.01 to +0.05, 5 min script.
3. **3-phase GAN scheduler** — NIQE -0.05, <1 day.
4. **Lion optimizer** — memory free, <1 day.
5. **Lookahead wrapper** — small stability gain, <1 day.
6. **OS-RealPLKSR preset** — new degradation preset, <1 day.

### Tier 2: Medium (M-effort, 1-2 weeks)
1. **MambaIRv2 (`sab_type: mamba_v2`)** — flip flag (Linux), +0.02 MANIQA.
2. **MambaIR (v1) additional sab_type** — +0.01 MANIQA.
3. **DINOv3 in FDL** — +0.01 MANIQA.
4. **VQD-SR codebook** — +0.02 MANIQA on OOD.
5. **`2xBHI_small_span_fast_pretrain` warm-start swap** — 2x faster inference.
6. **SAFMN as teacher candidate** — for MTKD distillation.
7. **DRCT-S as teacher candidate** — for MTKD distillation.
8. **Patch-NCE loss** — +0.02 MANIQA.
9. **Projected-GAN discriminator** — +0.02 MANIQA, less mode collapse.
10. **Hard-example mining curriculum** — +0.005 MANIQA.

### Tier 3: Heavy (L-effort, 2-4 weeks)
1. **DAT2 backbone swap** — +0.02 CLIPIQA, -0.01 MANIQA, 8GB-stretched.
2. **HAT-Lite backbone swap** — PSNR ceiling, +0.03 CLIPIQA.
3. **RealPLKSR + Dysample backbone** — +0.005 CLIPIQA + MANIQA.
4. **VQ codebook feature loss** — +0.005 MANIQA, anime-specific.
5. **RealDGen learned degrader** — +0.01 MANIQA, diffusion.
6. **MAE pretraining** — +0.005 MANIQA, 2 weeks.
7. **SUPIR post-process** — +0.03 MANIQA ceiling, very slow.
8. **`4xBHI_dat2_otf_nn` cold-start** — for DAT2 swap.
9. **`4xDRCT-mssim-pretrains` cold-start** — for DRCT swap.

---

## Decisions per metric gap

### Gap: MANIQA (0.422 -> 0.48)
- Tier 1: souping (+0.01 to +0.05), TTA (+0.005), Lookahead (+0.005).
- Tier 2: MambaIRv2 (+0.02), Patch-NCE (+0.02), Projected-GAN (+0.02), DINOv3 (+0.01), VQD-SR (+0.02 on OOD).
- Tier 3: SUPIR (+0.03, post).
- **Best plan**: TTA + souping (Tier 1) for immediate +0.01 to +0.05. Then MambaIRv2 + Projected-GAN (Tier 2) for +0.04 cumulative. Project total gap close: -0.058 to +0.05. **MANIQA target hit.**

### Gap: NIQE (7.681 stock / 7.123 v4 -> <=7.5)
- Already cleared by v4 finetune best.
- v7 finetune (after training) should also clear.
- Tier 1 souping (-0.05), Tier 2 VQD-SR (-0.1), Tier 3 RealDGen (-0.1), SUPIR (+0.2 — bad).
- **NIQE target hit by v7 alone.**

### Gap: LPIPS (0.0229 stock -> <=0.10)
- Already cleared by stock.
- TTA gives -10% (free).
- **LPIPS target hit.**

### Gap: CLIPIQA (0.667 stock -> >=0.65)
- Already cleared by stock.
- v4 finetune is worse (0.632) — the perception-distortion tradeoff (Blau et al., 2018).
- Tier 2 MambaIRv2 (+0.01), Patch-NCE (+0.005).
- Tier 3 DAT2 (+0.02), HAT-Lite (+0.03), A8 RealPLKSR (+0.005).
- **CLIPIQA target hit by stock + Tier 1/2.**

### Combined picture
The MANIQA gap is the only one not closed by v7 alone. The Tier 1 quick wins (TTA + souping) and Tier 2 MambaIRv2 + Patch-NCE + Projected-GAN should close it.

---

## What's NOT in our stack worth noting

- **No transformer backbone** (HAT, DRCT, SwinIR-Light are teachers only).
- **No dynamic upsampler** (we use fixed PixelShuffle; Dysample is the easy lift).
- **No MAE or contrastive pretrain**.
- **No TTA at inference**.
- **No model souping**.
- **No D-EMA**.
- **No Patch-NCE**.
- **No Lion optimizer**.

The first 4 are immediate Tier 1/2 wins.

---

## v8+ architecture: my recommendation

For the **next** project iteration (post-v7 finetune), I'd recommend:

1. **Run Tier 1 first (1 weekend)**:
   - TTA inference (`scripts/inference.py --tta`).
   - Soup v6 + v7 bests (when both exist).
   - Compare with `scripts/compare_checkpoints.py --tta` against stock pretrained.

2. **Then Tier 2 (Linux-only MambaIRv2 smoke)**:
   - WSL2 Ubuntu.
   - Flip `sab_type: mamba_v2` in v7 config.
   - 2-epoch smoke.
   - 80-epoch full run.
   - Compare MambaIRv2-v7 vs SPAN-v7 vs stock pretrained.

3. **Hold off on Tier 3** until MANIQA gap is closed by Tier 1+2.

4. **Evaluate on held-out** (`data/anime_hr_holdout/`) once curated. Publishable NR-IQA only from disjoint test set.

---

## Risks summary

- **Highest ROI with lowest risk**: TTA + souping.
- **Highest ROI with medium risk**: MambaIRv2 (Linux-blocked).
- **Highest ROI with high risk**: Tier 3 (DAT2, HAT-Lite, RealPLKSR).
- **Highest uncertainty**: VQD-SR (helps on OOD-LR, marginal on val_hr).
- **Highest wall-clock**: SUPIR post-process (12s/img on RTX 4090).

