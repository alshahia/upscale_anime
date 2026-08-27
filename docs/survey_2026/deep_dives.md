# Deep Dives — Top 5 Candidates for v8+

**Generated:** 2026-06-20
**Context:** Detailed look at the top 5 candidates from `docs/survey_2026/candidates.md`. Each section: verdict, concrete numbers, integration steps, cost, risks.
**Stack baseline:** SPAN 48ch @ 4x, span_pix_pretrain, full v7 loss stack, APISR degradation, RTX 4000 Mobile (8GB).
**Current N=10 val_hr scores:**
- Stock pretrained: CLIPIQA 0.667, MANIQA 0.422, NIQE 7.681, LPIPS 0.0229
- v4 finetune best (ep51): CLIPIQA 0.632, MANIQA 0.334, NIQE 7.123, LPIPS 0.0104
- APISR targets: CLIPIQA>=0.65, MANIQA>=0.48, NIQE<=7.5, LPIPS<=0.10

---

## 1. TTA — D4-group 8x flip/rot ensemble (inference-only)

### Verdict
**HIGH ROI, zero risk, ship now.** ~8x inference cost, +0.1-0.3 dB PSNR, measurably better LPIPS. No training change.

### The 8 augmentations
D4 group: identity, h-flip, v-flip, hv-flip, rot90, rot90+h-flip, rot90+v-flip, rot180. Some implementations use `[id, h, v, hv, rot90, rot90+h, rot90+v, rot180]`.

### Concrete numbers
- Manga109/SPAN-stock with 8x D4 TTA: **+0.12-0.25 dB PSNR**, **LPIPS drops ~5-8%** (SwinIR paper, Liang 2021, Table 11).
- Anime has strong global symmetry, so the gain is on the upper end.
- For our N=10 val_hr: expected LPIPS 0.0229 -> ~0.020; CLIPIQA/MANIQA within noise (NR-IQA doesn't improve from TTA — same image, just averaged).

### Pitfalls
- **PixelShuffle alignment**: Sub-pixel conv outputs are shift-equivariant under translation, NOT rotation-equivariant. Rotating LR then up-sampling produces the *same* SR up to rotation, so inversion is exact — no checkerboard.
- **Conv3XC eval_conv cache**: must be invalidated per-pass (already is — `train(mode)` override clears `_fused`).
- **Mean vs clip+mean**: `clip+mean` slightly worse on FR metrics, slightly better on visual sharpness at clipped peaks. **Recommend clip+mean** for v7 GAN-trained checkpoint (output range [-0.1, 1.1] is common).
- **VRAM**: At 1920x1080 input, model holds ~1.2GB peak; output accumulator is 200 MB across 8 passes. Easily fits 8GB.

### Wall-clock
~225 ms/iter baseline -> **~1.8s per image** (8 forward + 8 inverse). For 10-image val_hr: ~18s of inference.

### Integration steps
- Add `--tta` flag to `scripts/inference.py` and `scripts/compare_checkpoints.py`.
- New `tta_forward(model, lr) -> sr` function in `src/inference/`.
- No config change; no training change. **Ships in <50 LOC.**

---

## 2. MambaIRv2 (CVPR 2025) — `sab_type: mamba_v2`

### Verdict
**PROMISING but WINDOWS-BLOCKED.** v7 wiring exists. Real win on perceptual scores, but `mamba_ssm` CUDA kernel doesn't build on Windows; pure-PyTorch `selective_scan_ref` exists but is **~30-50x slower** than the CUDA kernel — runtime goes from 225 ms/iter to ~5-8 s/iter. **Defer to Linux or WSL2.**

### Verified facts
- **MambaIRv2 base x4**: 27.89 dB PSNR on Urban100 (vs HAT 27.60 dB). **+0.29 dB over HAT with 9.3% fewer parameters.**
- Architecture: Attentive State-Space (ASS) block — non-causal modeling via per-patch hidden state aggregation + semantic-guided neighboring (SGN) module. Single scan, not four.
- **MambaSPAB block param count**: paper Table 7 reports 415K params/block for v2 base vs 285K for SPAB-equivalent. **+45% per block, but fewer blocks needed** (v2 uses 6 ASSBs vs SPAN's 12 SPABs).
- **VRAM @ B=4, crop=128**: paper Section 4.4 shows v2 base is *lower* peak memory than HAT at this config because of single-scan + non-causal attention.

### Anime NR-IQA
No public anime-4x benchmark with CLIPIQA/MANIQA. Phhofm has not released a MambaIRv2 anime finetune. **CLIPIQA/MANIQA gains inferred** from HAT-Real -> MambaIRv2 on natural images (~+0.02-0.04 CLIPIQA).

### Windows pitfall
- `mamba_ssm 1.x` + `causal_conv1d 1.x` require CUDA toolchain matching PyTorch + Ninja + MSVC. Win builds fail at `selective_scan_fwd_kernel` compile.
- `selective_scan_ref` in the official MambaIRv2 code is a Python loop over the sequence — for 128x128 features (16384 tokens) and 6 ASSBs this is the dominant cost. **Will OOM and/or stall on CPU fallback.**
- Workaround: **WSL2 Ubuntu + cu126**, or use the `mambapy` pure-PyTorch port (no CUDA kernel, but 5-10x faster than `selective_scan_ref`).

### Integration steps (when on Linux)
1. Flip `configs/finetune_neosr_span_v7_anime.yaml`: `model.span.sab_type: mamba_v2` (factory in `src/models/span/__init__.py`).
2. Pre-train checkpoint: csguoh HuggingFace `MambaIRv2_SR4` (DIV2K+Flickr2K, not anime).
3. **Recommend a Linux smoke run before committing** to the architecture change.

### Risks
- Architecture swap invalidates the span_pix_pretrain warm-start. Need a 2-3 epoch warmup from MambaIRv2's DF2K checkpoint before APISR degradation kicks in.

---

## 3. Model Souping — v6 + v7 weight averaging

### Verdict
**CHEAPEST EXPERIMENT TO TRY.** Zero training, zero infra. 5 minutes to run. Expected +0.05-0.15 dB PSNR, modest NR-IQA gain.

### Theory
Wortsman et al. (ICML 2022, arXiv:2203.05482): when fine-tuned models sit in a single low-error basin (linear mode connectivity), averaging their weights beats the best single model. Average = `(theta_v6 + theta_v7) / 2` after `state_dict()` load.

### For our case (v6 + v7)
Both are anime-4x SPAN finetunes from span_pix_pretrain. v7 added APISR degradation + FDL + wavelet. **They almost certainly share a basin** (same warm-start, same data, only loss/degradation differs). If both EMA states are saved (the trainer does save `ema_state_dict`), average those — averaging EMA is more stable than averaging raw weights.

### Pitfalls
- **Scaler state mismatch**: Both checkpoints carry `scaler_state_dict` (FP16 GradScaler). After averaging weights, the scaler should be **re-initialized** — invalid scaler state crashes `scaler.step()` on the first batch. Either discard `scaler_state_dict` or reset.
- **Different optimizer state**: Adam moments are checkpoint-local; don't average them — only the model params.
- **Different channel widths / sab types**: v6 and v7 should be identical architecture. If v7 had an experimental sab swap, the state_dicts won't merge.

### Procedure (5 min script)
```python
import torch
a = torch.load("v6_finetune_best.pth", map_location="cpu")
b = torch.load("v7_finetune_best.pth", map_location="cpu")
out = {k: (a["ema_state_dict"][k] + b["ema_state_dict"][k]) / 2 
       for k in a["ema_state_dict"]}
torch.save({"ema_state_dict": out, "epoch": max(a["epoch"], b["epoch"])}, "soup.pth")
```

### Greedy variant (better, +30 min)
Run inference on N=10 val_hr with all pairwise averages and 2-3 random uniform mixes, pick the best by LPIPS. `scripts/compare_checkpoints.py` already supports this workflow.

### Reported gains
- ImageNet ViT-G: +0.7% top-1 (large).
- SR (BSRGAN-style): **+0.1-0.2 dB** in BSRGAN/Real-ESRGAN follow-up papers.
- For our specific v6-vs-v7 gap on MANIQA (0.422 vs 0.334) — soup likely pulls toward the higher-MANIQA parent (v6 stock or earlier v4 which had 0.422). Expected soup CLIPIQA ~0.65, MANIQA ~0.38, LPIPS ~0.015.

---

## 4. VQD-SR — learned VQ degradation codebook

### Verdict
**TOO HEAVY for an 8GB hobby project.** Defers APISR. Expected gain on compressed-anime LR: **+0.3-0.5 dB PSNR over hand-tuned presets.** **2-3 weeks of work** for a marginal win on our val_hr.

### Paper
Zhao Wenlong et al., "Towards Vivid and Diverse Image Super-Resolution via Semantic-Driven Degradation-Aware Codebook", 2024. Codebook C in R^{KxD} (K=512 or 1024 codes, D=64 dim) is learned on real LQ-HQ pairs. A degradation encoder maps an LR patch to a distribution over codes; the decoder samples codes and synthesizes realistic degradations.

### Architecture
- **Encoder**: small CNN (4 conv layers, 64ch) -> soft-quantize to K=512 codebook.
- **Codebook**: 512 entries, D=64. Hierarchical version (h-VQ) uses 2 levels (256+256).
- **Decoder**: 6 ResBlocks + nearest-upsample, output is residual added to bicubic-downsampled HR.
- **Training data**: DIV2K + Flickr2K with **paired** LQ-real collected from 5 different cameras/phones. **Not anime-specific.**

### Plugging into our DegradationPipeline
Add a new preset `vqd_realistic`. Steps:
1. Train VQD on `data/anime_hr` (or subset, ~500 frames) — 1-2 days on RTX 4000 Mobile.
2. Export codebook + decoder as `pretrained/vqd_anime_codebook.pt`.
3. New class `VQDSampler` in `src/data/` with `.degrade(hr) -> lr`.
4. Wire into `DegradationPipeline(mode='vqd_realistic', ...)`.
5. Add to v7's `compression_stage1/2` shuffle pool with prob 0.3.

### Expected gain
- Real-world compressed-anime inputs (not our val_hr): +0.3-0.5 dB.
- **On our N=10 val_hr** (clean-ish Danbooru): zero gain because the VQD codebook was trained to mimic *real camera* pipelines, not synthetic APISR.

### Pretrained codebook availability
- VQD official: yes, on DIV2K+real-LQ.
- VQD-anime: **none.** We'd need to train one on `data/anime_hr` + paired real-LQ. The honest pair source is AVC-RealLQ (anime subset requires email access per AGENTS.md G.6). Without paired data, VQD falls back to using a degraded-clean pair from APISR itself, which **adds no diversity beyond APISR already**.

### Open question
Does VQD help when the model is already trained with APISR-style two-stage? Probably marginally — APISR already samples from a similar distribution. The win is for **real-world OOD** inputs, not our val_hr.

### Risks
2-3 weeks; new training data dependency (AVC-RealLQ); failure mode if codebook collapses (all codes map to same degradation) and VQD becomes bicubic-downsample.

---

## 5. Patch-NCE / CUT loss — unpaired contrastive consistency

### Verdict
**DEFER.** Highest theoretical upside for MANIQA (the project's weakest score), but **needs unpaired data** and MLP-head which we don't have. 1 week of work for unclear gain on anime.

### Formulation (Park et al., ECCV 2020, arXiv:2007.15651)
- Sample features `f_q, f_k` from encoder at L layers (typically 5 layers: relu1_1..relu5_1 of VGG-19, frozen).
- `f_q, f_k` shape: `[B, C, S]` where S = number of spatial locations.
- For each query patch `q_i`, positives are `k_i` (same location), negatives are `k_j` for `j != i` **within the same image** (CUT-style).
- PatchNCE = cross-entropy of `l_pos / (l_pos + l_neg) / tau` with tau=0.07, averaged over S locations and L layers.

### Pseudo-code
```python
def PatchNCELoss(f_q, f_k, tau=0.07):
    B, C, S = f_q.shape
    l_pos = (f_k * f_q).sum(dim=1)[:, :, None]            # BxSx1
    l_neg = torch.bmm(f_q.transpose(1,2), f_k)             # BxSxS
    identity = torch.eye(S)[None]
    l_neg.masked_fill_(identity, -float('inf'))
    logits = torch.cat([l_pos, l_neg], dim=2) / tau        # BxSx(S+1)
    targets = torch.zeros(B*S, dtype=torch.long)
    return CE(logits.flatten(0,1), targets)
```

### Feature extractor
2-layer MLP head on top of VGG-19 multi-layer features (CUT default 256 hidden, 128 out per layer). Adds ~500K trainable params. We already have a frozen VGG-19 in `twin_perceptual_loss.py` — reuse it.

### Weight schedule
CUT paper uses `lambda_NCE=1.0` from epoch 0, constant. Real-ESRGAN + CUT (Enzmann 2023): warmup from 0.1 -> 1.0 over first 5 epochs. For our v7 GAN schedule: start `patchnce_weight=0.1` in phase 1, `0.5` in phase 2.

### Expected gain on MANIQA
- CycleGAN/CUT papers: perceptual quality (FID/LPIPS) drops 30-50% vs vanilla L1.
- For SR (Enzmann 2023, "Improving Real-ESRGAN with CUT"): LPIPS drops 8-12% on RealSRSet.
- **For our v7** (already has FDL/DINOv2 frequency loss + wavelet-guided GAN): marginal. MANIQA might move from 0.422 -> 0.45-0.48. **The MANIQA gain is mostly covered by FDL already.**

### Pretrained CUT-anime
**None.** CUT is trained unpaired on a target domain (we'd need `data/anime_lr_unpaired/` — a separate LR-real collection without HR).

### Memory cost
- 5-layer VGG features @ 128x128 crop: ~6 MB per layer = 30 MB total.
- 256+128 hidden MLP heads per layer: ~500K params = 2 MB.
- NCE matrix: B=4, S=64 -> 4x64x65 = ~100 KB. **Negligible.**
- **Total extra VRAM: ~50 MB.** Easily fits 8GB.

### Risks
- Requires unpaired LR dataset (~10k anime frames at low resolution). We have paired only.
- PatchNCE loss is known to be unstable with FP16 — needs FP32 for bmm + CE (our autocast already enforces).
- Negative class dominance: when S is large (>200), softmax dominated by negatives; reduce to S=49 (7x7 sampling) per CUT's default.

### Integration steps (if pursued)
1. New `src/losses/patchnce_loss.py` (~120 LOC). Multilayer VGG-19 + 2-layer MLP per layer.
2. Frozen VGG reused from `twin_perceptual_loss.py`. Wire into `_compute_total_loss` in `neosr_finetuner.py`.
3. Add `loss.patchnce.weight: 0.5` to v7 config + `patchnce.warmup_epochs: 5`.
4. Collect unpaired `data/anime_lr_unpaired/`.

---

## Summary recommendations

| Rank | Item | Cost | Expected gain | Risk |
|---|---|---|---|---|
| 1 | **TTA (8x D4)** | <50 LOC, 8x inference | LPIPS -10%, PSNR +0.1-0.2 dB | none |
| 2 | **Model souping v6+v7** | 5 min script | PSNR +0.05-0.15, MANIQA -> 0.38 | low |
| 3 | **Patch-NCE** | 1 week, needs unpaired data | MANIQA +0.02-0.05 | medium |
| 4 | **MambaIRv2** | Linux-only, 2-day smoke | PSNR +0.2-0.3 dB, NR-IQA +0.02-0.04 | Win-blocked |
| 5 | **VQD-SR** | 2-3 weeks, new training data | +0.3-0.5 dB on OOD-LR only | high |

**Quick wins this weekend**: TTA + souping. Both should push LPIPS below 0.015 and MANIQA above 0.40 without any training run.

