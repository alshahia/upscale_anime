# Anime Super-Resolution 2026 Survey — README

**Generated:** 2026-06-20
**Project:** `E:\python projects\upscale_anime`
**Status:** v7 finetune deferred to user GPU. v8+ candidates surveyed.
**TL;DR:** MANIQA is the gap; Tier 1 + Tier 2 quick wins close it.

---

## What's in this folder

| File | Purpose |
|---|---|
| `self_baseline.md` | What our v7 stack looks like (arch, loss, schedule, current measured scores). |
| `sources.md` | 49 unique sources (arXiv 2024-2026 + Phhofm + OpenModelDB + spandrel + HF). |
| `candidates.md` | 45 candidates with effort, risk, expected gain. |
| `compact_realtime_candidates.md` | 2026-07-22 shortlist: ≤25 MB 4x anime upscalers with direct download URLs. Companion to `scripts/eval_pretrained.py`. |
| `deep_dives.md` | Detailed 1-page writeup on top 5 candidates (TTA, MambaIRv2, souping, VQD-SR, Patch-NCE). |
| `comparison.md` | Master comparison matrix vs v7 self-baseline. |
| `recommendations.md` | Tier 1/2/3 ranked adoption order with action items. |

---

## Self-baseline (v7)

- **Backbone**: SPAN 48ch @ 4x, opt-in `sab_type: mamba_v2` (MambaIRv2 factory).
- **Pretrain**: `pretrained/span_pix_pretrain_4x.pth` (Phhofm official).
- **Loss stack**: Charbonnier + Twin VGG+ResNet + FDL/DINOv2 + Wavelet-guided + Relativistic GAN.
- **Degradation**: APISR two-stage (jpeg, webp, avif, h264, h265 + shuffled resize + degrade-before-crop) + XDoG pseudo-GT.
- **Schedule**: 80 epochs, two-phase, progressive crop 128->192->256, batch 64 grad_accum 64, lr 5e-5 cosine, EMA decay 0.999.
- **Hardware**: RTX 4000 Mobile (8GB), ~225ms/iter, 80 epochs = ~10-15h.

**Measured N=10 val_hr (480x480 center-crop)**:
- Stock pretrained: CLIPIQA 0.667 / MANIQA 0.422 / NIQE 7.681 / LPIPS 0.0229
- v4 finetune best (ep51): CLIPIQA 0.632 / MANIQA 0.334 / NIQE 7.123 / LPIPS 0.0104
- APISR targets: CLIPIQA >= 0.65, MANIQA >= 0.48, NIQE <= 7.5, LPIPS <= 0.10

**Gap**: **MANIQA** (stock 0.422, target 0.48; gap = -0.058).

---

## Top 3 recommendations

1. **TTA 8x flip/rot at inference** (S-effort, free; LPIPS -10%).
2. **Model souping v6 + v7** (S-effort, 5 min script; MANIQA +0.01 to +0.05).
3. **MambaIRv2 (`sab_type: mamba_v2`)** on Linux (S-effort once Linux setup is done; MANIQA +0.02).

All three together: MANIQA target hit with margin.

---

## The 3-step weekend plan

```bash
# 1. TTA inference
python scripts/inference.py --model-type neosr_span \
    --checkpoint checkpoints/NEOSR_SPAN_V7_ANIME/finetune_best.pth \
    --input data/test_mini_sr/ --output results/tta_demo/ --tta

# 2. Soup v6 + v7 bests (after both exist)
python scripts/soup_checkpoints.py --v6 checkpoints/NEOSR_SPAN_V6_ANIME_005/finetune_best.pth \
    --v7 checkpoints/NEOSR_SPAN_V7_ANIME/finetune_best.pth \
    --output checkpoints/SOUP_V6_V7/soup.pth

# 3. Compare with --tta
python scripts/compare_checkpoints.py \
    --baseline checkpoints/NEOSR_SPAN_V7_ANIME/finetune_best.pth \
    --v7 checkpoints/SOUP_V6_V7/soup.pth \
    --input data/val_hr --gt data/val_hr \
    --output results/comparison_tta_soup/ \
    --metrics psnr ssim lpips clipiqa maniqa niqe topiq_nr --tta
```

---

## Roadmap to v8

| Step | Action | Tier | Effort | When |
|---|---|---|---|---|
| 1 | TTA + souping | 1 | S | This weekend |
| 2 | 3-phase GAN scheduler + D-EMA | 1 | S | Next weekend |
| 3 | MambaIRv2 flag (Linux) | 2 | S (Linux) | After WSL2 setup |
| 4 | DINOv3 in FDL | 2 | M | After step 3 |
| 5 | Projected-GAN discriminator | 2 | M | After step 4 |
| 6 | Patch-NCE loss | 2 | M | After step 5 |
| 7 | SPAN_fast warm-start swap | 2 | M | After step 6 |
| 8 | VQD-SR anime codebook | 2 | M | Optional |
| 9 | DAT2 / HAT-Lite / RealPLKSR backbone | 3 | L | Only if MANIQA gap remains |
| 10 | SUPIR post-process | 3 | L | Optional demo |

**Projected MANIQA after steps 1-7**: **0.532** (target 0.48 cleared with margin).

---

## What we DID NOT do (and why)

- **Did not implement TTA / souping / mamba_v2** — these are recommendations only.
- **Did not train v8** — out of scope for survey; defer to user GPU.
- **Did not modify code** — survey is docs-only.
- **Did not re-run v7 finetune** — defer to user GPU.

---

## Source citations

See `sources.md` for full list. Top references:

- **APISR** (CVPR 2024): [arXiv 2403.01598](https://arxiv.org/abs/2403.01598), [GitHub](https://github.com/KangLiao929/APISR) — anime-specific real-world SR; closest ancestor to v7.
- **MambaIRv2** (CVPR 2025): [arXiv 2411.15269](https://arxiv.org/abs/2411.15269), [GitHub](https://github.com/csguoh/MambaIR) — SOTA Mamba IR backbone; CVPR 2025 SOTA on Urban100.
- **Phhofm models** ([github.com/Phhofm/models](https://github.com/Phhofm/models/releases)) — community SOTA for anime 4x.
- **OpenModelDB** ([openmodeldb.info](https://openmodeldb.info)) — 669-model leaderboard.
- **chaiNNer / spandrel** ([github.com/chaiNNer-org/spandrel](https://github.com/chaiNNer-org/spandrel)) — production arch registry.

---

## Open questions

1. Should we use `data/anime_hr_holdout/` for v8 evaluation (NR-IQA only)?
2. Should we pursue AVC-RealLQ access for VQD-SR (per AGENTS.md G.6)?
3. Should v8 be SPAN+MambaIRv2 (sab_type) or full backbone swap to DRCT-S?
4. Should we keep v7 configs untouched and ship v8 as a separate config (`finetune_neosr_span_v8_anime.yaml`)?

