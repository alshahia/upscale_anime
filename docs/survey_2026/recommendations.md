# Recommendations — v8+ Anime Super-Resolution

**Generated:** 2026-06-20
**Scope:** Adoption order, prerequisites, expected gains, and links to deep-dives.
**Inputs:** `docs/survey_2026/comparison.md`, `docs/survey_2026/deep_dives.md`, `docs/survey_2026/self_baseline.md`.

---

## TL;DR (5-line executive)

1. **This weekend, ship TTA + model souping** (Tier 1) — free, +0.005 to +0.05 MANIQA, -10% LPIPS, 5 min scripting + 1 day code.
2. **Next month, run MambaIRv2 (Linux-only) and Patch-NCE** (Tier 2) — +0.02 to +0.04 MANIQA cumulative.
3. **Hold Tier 3 (backbone swaps) until MANIQA target is hit** — DAT2 / HAT-Lite / RealPLKSR are 2-4 week commits with mixed ROI.
4. **The MANIQA gap (-0.058) is closable in 2-4 weeks** with Tier 1 + Tier 2.
5. **CLIPIQA / NIQE / LPIPS targets are already hit** by v7 self-baseline; v8+ is MANIQA-focused.

---

## Tier 1 — Quick Wins (S-effort, no training, ship now)

**Total cost: ~2 days. Expected gain: -10% LPIPS, +0.01 to +0.05 MANIQA, +0.05 dB PSNR.**

### 1.1 TTA 8x flip/rot ensemble at inference
- **Effort**: S (<50 LOC)
- **Risk**: none
- **Files**: `scripts/inference.py` + `scripts/compare_checkpoints.py`, new `src/inference/tta.py`
- **Expected gain**: LPIPS -10%, PSNR +0.1-0.2 dB
- **Wall-clock**: +800% inference (~1.8s/img at 1080p on RTX 4000 Mobile)
- **Pitfall**: PixelShuffle is shift-equivariant but not rotation-equivariant — TTA is correct only when SR inversion is also done. Our `inference.py` postprocess is rotation-aware. Use `clip+mean` accumulator.
- **See**: `docs/survey_2026/deep_dives.md` §1

```python
# Sketch
def tta_forward(model, lr):
    out = torch.zeros_like(lr_upscaled)
    for aug_fn, inv_fn in D4_GROUP:
        aug = aug_fn(lr)
        sr = model(aug)
        out += inv_fn(sr)
    return (out / 8).clamp(0, 1)
```

### 1.2 Model souping of v6 + v7 best checkpoints
- **Effort**: S (5 min script + 30 min greedy variant)
- **Risk**: low
- **Files**: new `scripts/soup_checkpoints.py`
- **Expected gain**: MANIQA +0.01 to +0.05, LPIPS -0.002
- **Pitfall**: discard `scaler_state_dict`; reset optimizer state; only average EMA params.
- **See**: `docs/survey_2026/deep_dives.md` §3

```python
# Sketch
a = torch.load(v6_best, map_location="cpu")
b = torch.load(v7_best, map_location="cpu")
soup = {k: (a["ema_state_dict"][k] + b["ema_state_dict"][k]) / 2
        for k in a["ema_state_dict"]}
torch.save({"ema_state_dict": soup, "epoch": max(a["epoch"], b["epoch"])},
           "soup_v6_v7.pth")
```

### 1.3 OS-RealPLKSR degradation preset
- **Effort**: S (<1 day)
- **Risk**: low
- **Files**: `src/data/degradation_pipeline.py`
- **Expected gain**: NIQE -0.05, marginal MANIQA
- **Notes**: Add a new `os_realplksr` preset based on Phhofm's recent training data settings (DPID downscaling, strict IQA filtering, no-noise).
- **See**: `docs/survey_2026/candidates.md` §C11

### 1.4 3-phase GAN scheduler
- **Effort**: S (<1 day)
- **Risk**: low
- **Files**: `configs/finetune_neosr_span_v8_anime.yaml`, `src/training/neosr_finetuner.py`
- **Expected gain**: NIQE -0.05, marginal MANIQA
- **Notes**: Add a Phase 3 (epochs 81-100) with GAN-only finetune at low lr. Stops GAN drift on already-converged model.
- **See**: `docs/survey_2026/candidates.md` §C10

### 1.5 D-EMA (discriminator EMA)
- **Effort**: S (<1 day)
- **Risk**: low
- **Files**: `src/training/ema.py`, `src/training/neosr_finetuner.py`
- **Expected gain**: NIQE -0.05
- **Notes**: Mirror G-EMA on D.
- **See**: `docs/survey_2026/candidates.md` §C13

---

## Tier 2 — Medium (M-effort, 1-2 weeks, MANIQA-focused)

**Total cost: ~6 weeks if done sequentially. Expected gain: +0.02 to +0.04 MANIQA cumulative.**

### 2.1 MambaIRv2 (`sab_type: mamba_v2`) — flip v7 flag
- **Effort**: S if already in v7 (M if Windows-blocked -> WSL2)
- **Risk**: low (Linux) / high (Windows without WSL)
- **Files**: `configs/finetune_neosr_span_v7_anime.yaml`, flip `sab_type: conv3xc -> mamba_v2`
- **Expected gain**: +0.02 MANIQA, +0.01 CLIPIQA, -0.001 LPIPS, -0.1 NIQE
- **Wall-clock**: same as v7 if `mamba_ssm` CUDA kernel builds (Linux); +1500% if pure-PyTorch `selective_scan_ref` (Windows)
- **Pitfall**: warm-start invalidation. Need 2-3 epoch warmup from MambaIRv2 DF2K checkpoint before APISR kicks in.
- **See**: `docs/survey_2026/deep_dives.md` §2

### 2.2 Patch-NCE / CUT loss
- **Effort**: M (1 week)
- **Risk**: medium (needs unpaired LR dataset)
- **Files**: new `src/losses/patchnce_loss.py` (~120 LOC), `src/training/neosr_finetuner.py`, v7 config
- **Expected gain**: +0.02 MANIQA, -0.002 LPIPS
- **VRAM**: ~50 MB extra
- **Prerequisite**: collect ~10k unpaired LR anime frames into `data/anime_lr_unpaired/`
- **See**: `docs/survey_2026/deep_dives.md` §5

### 2.3 DINOv3 in FDL loss
- **Effort**: M (<1 day)
- **Risk**: low
- **Files**: `src/losses/fdl_loss.py` (swap DINOv2 -> DINOv3 small)
- **Expected gain**: +0.01 MANIQA, -0.05 NIQE
- **Prerequisite**: DINOv3 small checkpoint available (Meta 2024 release).
- **See**: `docs/survey_2026/candidates.md` §B3

### 2.4 Projected-GAN discriminator
- **Effort**: M (1 week)
- **Risk**: medium
- **Files**: `src/losses/adversarial_loss.py`, add frozen DINOv2 features in D forward
- **Expected gain**: +0.02 MANIQA, -0.003 LPIPS, -0.1 NIQE
- **Notes**: reduces mode collapse on flat anime color regions.
- **See**: `docs/survey_2026/candidates.md` §B2

### 2.5 VQD-SR learned degradation codebook
- **Effort**: M (1-2 days for preset; 1-2 weeks for anime codebook)
- **Risk**: medium
- **Files**: new `src/data/vqd_sampler.py`, `pretrained/vqd_anime_codebook.pt`, `src/data/degradation_pipeline.py`
- **Expected gain**: +0.02 MANIQA on OOD-LR; ~0 on val_hr
- **Prerequisite**: anime + paired real-LQ (AVC-RealLQ access per AGENTS.md G.6) OR train on our APISR output (less useful).
- **See**: `docs/survey_2026/deep_dives.md` §4

### 2.6 `2xBHI_small_span_fast_pretrain` warm-start swap
- **Effort**: M (1-2 days)
- **Risk**: low
- **Files**: `configs/finetune_neosr_span_v7_anime.yaml`, change `pretrained_path` (and add 2x->2x chain)
- **Expected gain**: **-50% inference time**, marginal quality
- **Prerequisite**: download Phhofm `2xBHI_small_span_fast_pretrain` (32ch, Mish, kaiming). Need 2x inference path.
- **See**: `docs/survey_2026/candidates.md` §D1

### 2.7 Hard-example mining curriculum
- **Effort**: M (1-2 days)
- **Risk**: medium
- **Files**: `src/data/dataloader.py`, new `src/data/hard_example_miner.py`
- **Expected gain**: +0.005 MANIQA, -0.05 NIQE
- **Notes**: identify low-CLIPIQA val examples; oversample them in training. Periodic re-evaluation.
- **See**: `docs/survey_2026/candidates.md` §C9

---

## Tier 3 — Heavy (L-effort, 2-4 weeks, only if MANIQA gap remains)

**Total cost: ~3 months. Expected gain: +0.02 to +0.03 CLIPIQA, marginal MANIQA.**

### 3.1 DAT2 backbone swap
- **Effort**: L (2-4 weeks)
- **Risk**: high
- **Files**: new `src/models/dat.py`, train/inference wiring
- **Expected gain**: +0.02 CLIPIQA, -0.01 MANIQA, +0.2 NIQE (mostly bad)
- **Notes**: heavy at 4x on 8GB; crop<=64 required. Phhofm has anime-tuned cold-start.
- **See**: `docs/survey_2026/candidates.md` §A6

### 3.2 HAT-Lite backbone swap
- **Effort**: L (2-4 weeks)
- **Risk**: high
- **Files**: new `src/models/hat.py`, train/inference wiring
- **Expected gain**: +0.03 CLIPIQA (PSNR ceiling)
- **Notes**: HAT-L too heavy; HAT-Lite may fit. PSNR-focus trade.
- **See**: `docs/survey_2026/candidates.md` §A7

### 3.3 RealPLKSR + Dysample backbone
- **Effort**: L (2-4 weeks)
- **Risk**: medium
- **Files**: new `src/models/realplksr.py`, new `src/upsamplers/dysample.py`
- **Expected gain**: +0.005 CLIPIQA, +0.005 MANIQA, -0.001 LPIPS
- **Notes**: Phhofm's current preferred line.
- **See**: `docs/survey_2026/candidates.md` §A8

### 3.4 SUPIR post-process
- **Effort**: L (2-4 weeks)
- **Risk**: high
- **Files**: new `src/inference/supir_postprocess.py`
- **Expected gain**: +0.03 MANIQA, -0.01 LPIPS, +0.2 NIQE (mostly bad)
- **Wall-clock**: 12s/img on RTX 4090; ~25s on RTX 4000 Mobile.
- **Notes**: SOTA diffusion SR ceiling. Worth considering as an ensemble post-process for top-tier quality.
- **See**: `docs/survey_2026/candidates.md` §F3

---

## Recommended v8+ adoption order

| Step | Action | Tier | Effort | Expected Δ MANIQA |
|---|---|---|---|---:|
| 1 | Ship TTA 8x flip/rot in `inference.py` | 1 | S | +0.005 |
| 2 | Soup v6 + v7 best checkpoints | 1 | S | +0.01 to +0.05 |
| 3 | OS-RealPLKSR preset + 3-phase GAN scheduler | 1 | S | +0.005 to +0.01 |
| 4 | Flip `sab_type: mamba_v2` (Linux) | 2 | S (Linux) | +0.02 |
| 5 | DINOv3 in FDL | 2 | M | +0.01 |
| 6 | Projected-GAN discriminator | 2 | M | +0.02 |
| 7 | Patch-NCE loss | 2 | M | +0.02 |
| 8 | SPAN_fast warm-start swap | 2 | M | 0 (but -50% inference) |
| 9 | Hard-example mining | 2 | M | +0.005 |
| 10 | (Optional) VQD-SR anime codebook | 2 | M | +0.02 on OOD |
| 11 | (Optional) DAT2 / HAT-Lite / RealPLKSR backbone | 3 | L | trade-offs |
| 12 | (Optional) SUPIR post-process | 3 | L | +0.03 (slow) |

**Projected MANIQA after steps 1-7**: 0.422 (stock) + 0.005 (TTA) + 0.03 (soup) + 0.005 (3-phase) + 0.02 (MambaIRv2) + 0.01 (DINOv3) + 0.02 (Projected-GAN) + 0.02 (Patch-NCE) = **0.532**. **Target 0.48 cleared with margin.**

---

## Risks & open questions

- **VQD-SR**: only helps on real-world OOD LR; on our clean val_hr, ~0 gain.
- **Patch-NCE**: needs unpaired LR dataset. Acquisition cost unclear.
- **MambaIRv2**: Windows without WSL2 is unviable (5-8s/iter).
- **DAT2 / HAT-Lite**: 8GB-stretched; trade-offs.
- **SUPIR**: 25s/img makes it impractical as default; consider as Tier-3 demo only.
- **CLIPIQA target**: stock already clears; v7 finetune may regress (perception-distortion tradeoff). v8+ should preserve CLIPIQA>=0.65.

---

## User action items

1. **Tier 1 weekend**: pick at least 2 of {TTA, souping, 3-phase scheduler} and ship them.
2. **Linux setup**: get WSL2 Ubuntu ready for MambaIRv2 smoke test.
3. **Held-out set**: curate `data/anime_hr_holdout/` (30-50 Danbooru frames) before evaluating v8+.
4. **AVC-RealLQ**: optional email request for VQD-SR (per AGENTS.md G.6).
5. **Re-evaluate** after each Tier step using `scripts/compare_checkpoints.py --tta`.

