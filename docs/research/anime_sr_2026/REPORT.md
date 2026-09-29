# Anime Super-Resolution Research Progress Report -- 2026

**Generated:** 2026-09-03
**Author:** MiniMax-M3 (research mode via Exa /search)
**Inputs:** 15 Exa /search queries, ~120 source URLs with highlights
**Tags:** anime, super-resolution, real-time, 4x, knowledge-distillation, diffusion, transformer, mamba, hybrid

---

## 0. Scope & Method

**Question:** What are the latest research/models/techniques (2024-2026) for training a 4x anime upscaler that can run at >=25 fps on RTX 4000 8GB? If higher-quality non-real-time methods exist, document them too -- we may create a hybrid.

**Constraints (project):**
- GPU: Quadro RTX 4000 8GB (~7.6 GB free).
- Current state: RFDN-distill v1 (315K params, 59 ms/frame at native res, TTA off). animevideov3 baseline at 15 fps 4K.
- v1 + TTA = 29.46 dB (Phase 4 closed per plan matrix).

**Method:**
- 15 Exa /search queries via Python exa-py 2.20.0.
- Each query: num_results=8, type=auto, contents.highlights (numSentences=4, highlightsPerUrl=3).
- Per-query cost: ~$0.007. Total cost: ~$0.10.
- No deep-crawl. Highlights only. Treat numbers as "reported by source".

**Honesty directives:**
- All citations link to Exa-returned URLs; not independently verified.
- This is a landscape report, not a benchmark.
- No new code is written; this is a decision-input document.

---

## 1. Real-Time Tier (<=25 fps @ 4K, <=40 ms/frame)

### 1.1 Existing baseline (we already have this)

**RFDN** (Residual Feature Distillation Network, Liu et al, AIM 2020 winner):
- Source: https://ar5iv.labs.arxiv.org/html/2009.11551
- Params: 315K (our v1); ~534K (paper variant).
- 4x SR. Excellent PSNR/FLOPs trade-off.
- Distillation connection (FDC) + Shallow Residual Block (SRB).
- License: MIT (per https://github.com/njulj/RFDN).
- Our status: v1 RFDN-distill, 29.886 dB val PSNR, 59 ms/frame.

### 1.2 Modern lightweight SOTA (2023-2025)

| Model | Year | Params | Architecture | Notes |
|---|---|---|---|---|
| **LKDN** | 2024 | -- | Large kernel distillation + reparameterization | Outperforms BSRN/RFDN at lower cost. Source: https://arxiv.org/html/2407.14340v1 |
| **BSRN** | NTIRE 2022 winner | -- | Blueprint-separable conv on RFDN base | Source: https://arxiv.org/html/2407.14340v1 (referenced as baseline) |
| **VAPSR** | 2024 | ~28% of RFDN | Depth-wise separable large kernel | Matches RFDN PSNR with far fewer params. Source: https://arxiv.org/html/2407.14340v1 |
| **ShuffleMixer** | NeurIPS 2022 | ~3x smaller than CARN | Large depth-wise conv + channel shuffle | https://proceedings.nips.cc/paper_files/paper/2022/file/6e60a9023d2c63f7f0856910129ae753-Paper-Conference.pdf |
| **OmniSR** | CVPR 2023 | -- | Omni-aggregation, grid-attention | https://github.com/Francis0625/Omni-SR ; arxiv 2304.10244 |
| **IMDN** | ACM MM 2019 | 275,844 | Information multi-distillation + CCA | https://github.com/Zheng222/IMDN |
| **RFLN** | NTIRE 2022 runtime winner | -- | Residual local feature network | Faster than RFDN, same quality tier |
| **IBMDN** | 2025 | -- | Involution + BSConv multi-depth distillation | https://arxiv.org/pdf/2503.14779 |
| **ESDAN** | Sci.Rep 2025 (Nov) | -- | Sharpen enhancement + dual attention | https://www.nature.com/articles/s41598-025-24493-8 |
| **ELSANet** | -- | -- | Efficient long-short attention | Referenced in real-time SR survey |

**Key observation:** The lightweight CNN family has converged on a recipe: **information distillation + cheap attention + structural reparameterization + large (separable) kernels**. Our v1 RFDN-distill already captures the first two. RFDN is NOT obsolete -- it is a strong, mature baseline.

### 1.3 MambaIRv2 (CVPR 2025) -- the most promising new direction

**MambaIRv2: Attentive State Space Restoration** (Hang Guo et al, CVPR 2025):
- Paper: https://arxiv.org/html/2411.15269v2 (Nov 2024)
- Repo: https://github.com/csguoh/MambaIR
- HF: https://huggingface.co/cguoh/MambaIR

**Key claims (reported by source, not verified):**
- Outperforms **SRFormer** (Transformer-based) by **+0.35 dB PSNR** on Urban100 2x lightweight SR with **9.3% fewer parameters**.
- Outperforms **HAT** (Transformer-based) by **+0.29 dB** on Manga109 2x classical SR (strong anime signal).
- **30% lower MACs than HAT** at 256x256 patch.
- Achieves this with **single-direction scan** (more efficient than MambaIR's multi-scan).
- Key innovations: Attentive State-space Equation (ASE) + Semantic Guided Neighboring (SGN).

**Critical Windows caveat:** Mamba state-space models require mamba-ssm CUDA kernels, which only have pre-built wheels for Linux + cu124/cu126. On Windows, this requires **WSL2** -- a blocker we already hit in our v8 plan (docs/plans/v8_mambair_plan.md). 

**Action:** If we go MambaIRv2, plan for WSL2 setup first (~30 min one-time, per v8 plan A.1-A.7).

### 1.4 Anime-specialized real-time options

- **Anime4K v4** (GLSL shader, not a learned model): Real-time GPU shader for live preview. Lower quality but zero inference cost. Source: https://www.youngju.dev/blog/culture/2026-05-16-ai-image-upscaling-restoration-2026-topaz-photo-supir-gfpgan-real-esrgan-codeformer-magnific-krea-deep-dive.en
- **Video2X** wraps Anime4K + Real-CUGAN + Real-ESRGAN + RIFE in one pipeline. Source: https://reapi.ai/blog/open-source-video-upscaler
- **Real-CUGAN** (Bilibili): 2x/3x/4x with 5/3/3 denoise strengths. Anime-tuned. Source: https://github.com/bilibili/ailab/blob/2799af78/Real-CUGAN/README_EN.md
- **APISR** (CVPR 2024): anime production-inspired dataset + balanced twin perceptual loss. Source: https://arxiv.org/html/2403.01598v2 ; https://github.com/kiteretsu77/apisr

### 1.5 Inference acceleration (everyone uses this)

- **TensorRT FP16** (NVIDIA): ~2x speedup for SD, similar gains for SR. Source: https://developer.nvidia.com/blog/tensorrt-accelerates-stable-diffusion-nearly-2x-faster-w
- **TensorRT 9.2 INT8 post-training quantization**: 1.72x speedup on SDXL with 8-bit weights.
- **Batch size**: TensorRT batched inference is the biggest single latency win. Source: https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html
- **PiD ONNX export** of SDXL-distill 4-step: https://huggingface.co/Glebka/PiD-res2kto4k-ONNX -- reference for our own ONNX export of any new student.
- Our Phase 1 plan (docs/plans/realtime_4k_plan.md) already covers TensorRT + batching + NVENC for the existing v1 student.

**Bottom line for real-time tier:** Stay with RFDN-style lightweight CNN for now. TensorRT FP16 + batched video frame pipeline is the highest-leverage speedup (matches our Phase 1 plan). MambaIRv2 is a Phase 5+ candidate if WSL2 is on the table.

---

## 2. Quality-Ceiling Tier (slow but best perceptual)

### 2.1 Diffusion SR landscape (the new quality ceiling)

| Model | Year | Steps | Inference (RTX 4090) | Best for | Source |
|---|---|---|---|---|---|
| **SUPIR** | CVPR 2024 | 20-50 | 5-15 sec/frame | Damaged/heavily degraded sources, text-guided restoration | https://github.com/Fanghua-Yu/SUPIR |
| **SeeSR** | CVPR 2024 | multi | -- | Semantic-aware, anime friendly | https://github.com/cswry/SeeSR |
| **DiffBIR** | 2024 | multi | -- | Blind SR via ControlNet-style diffusion prior | https://arxiv.org/html/2404.01717v4 |
| **StableSR** | 2024 | multi | ~100x slower than OSEDiff | Exploits SD prior w/o layer copies | https://arxiv.org/html/2305.07015v4 |
| **OSEDiff** | NeurIPS 2024 | **1** | ~0.5 sec | Real-time-feasible diffusion, variational score distillation | https://proceedings.neurips.cc/paper_files/paper/2024/file/a8223b0ad64007423ffb308b0dd92298-Paper-Conference.pdf |
| **SinSR** | 2023 | **1** | fastest diffusion | Consistency-preserving distillation from ResShift | https://arxiv.org/html/2311.14760v1 |
| **ResShift** | 2023 | multi | -- | Diffusion with shifted trajectory | |
| **PiD (NVIDIA)** | 2024 | 4 | fastest published | SDXL-distill 4-step, ONNX available | https://huggingface.co/Glebka/PiD-res2kto4k-ONNX |
| **FiDeSR** | CVPR 2026 | 1 | -- | Detail-preserving one-step diffusion | https://openaccess.thecvf.com/content/CVPR2026/papers/Kim_FiDeSR_High-Fidelity_and_Detail-Preserving_One-Step_Diffusion_Super-Resolution_CVPR_2026_paper.pdf |
| **One-Step Diffusion Transformer** | CVPR 2026 | 1 | -- | Controllable Real-World ISR | https://openaccess.thecvf.com/content/CVPR2026/papers/Fang_One-Step_Diffusion_Transformer_for_Controllable_Real-World_Image_Super-Resolution_CVPR_2026_paper.pdf |
| **Bridging Fidelity-Reality** | CVPR 2026 | 1 | -- | Controllable one-step diffusion | https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_Bridging_Fidelity-Reality_with_Controllable_One-Step_Diffusion_for_Image_Super-Resolution_CVPR_2026_paper.pdf |
| **AlloSR^2** | 2026 | 1 | -- | Allomorphic generative flows (no prior collapse) | https://doi.org/10.48550/arxiv.2604.19238 |

**Key trend:** The diffusion-SR field is moving from multi-step (50-100 steps, 5-15 sec) to **one-step variants** (OSEDiff, SinSR, FiDeSR). At 0.5 sec/frame, this is *not real-time* but is *interactive* -- viable for a "quality" toggle in our GUI.

### 2.2 Anime-relevant quality highlights

- **UltraSharpV2** (DAT2 backbone, May 2025): anime + CGI + faces + textures. HuggingFace: https://huggingface.co/Kim2091/UltraSharpV2
- **Anime4K Hybrid CNN-Transformer (TBC 2025)**: Source: https://sah.borca.ai/papers/283045784 ; https://doi.org/10.1109/tbc.2025.3622413
- **Caelum** (yumenana): free 4x anime illustration SR targeting real-world internet degradation. https://github.laiyagushi.com/yumenana/Caelum
- **Adore (Re-SISR Adore release, Apr 2026)**: real-time upscaler for 1080p anime; CC BY-NC-SA 4.0. https://github.com/renarchi/Re-SISR/releases/tag/Adore

### 2.3 When to reach for diffusion SR

- Real-time 4K streaming: NO. (Too slow.)
- Hero-shot quality for short clips / stills: YES. (5-15 sec/frame acceptable.)
- Damaged/noisy anime sources (VHS, DVD rips, low-bitrate web): YES. (SUPIR + SeeSR excel here.)
- "Best possible quality" preset in GUI: YES. (One-step variants make this ~1 sec/frame -- borderline acceptable for batch job.)

---

## 3. Hybrid Approach (recommended for our project)

### 3.1 Three-tier GUI option design

```
Tier 1: REAL-TIME (default, must hit 25 fps @ 4K)
    v1 RFDN-distill student (existing)
    + TensorRT FP16 (Phase 1 from realtime_4k_plan)
    + Batched video frame pipeline
    + Optional D4 TTA (off by default; +10% LPIPS for -800% inference)
    Target: 35-50 fps @ 4K

Tier 2: QUALITY (one-click preset for hero shots)
    OSEDiff 1-step diffusion OR SinSR 1-step
    Optional anime fine-tune of one of these
    + Light anime fine-tune (~1 day training) using APISR API dataset
    Target: ~1 sec/frame @ 1080p->4K

Tier 3: HYBRID (cascade 2x+2x)
    Stage A: v1 RFDN-distill student @ 2x (faster, less compute)
    Stage B: OSEDiff OR FiDeSR @ 2x (1-step diffusion on already-upscaled)
    Target: 0.3 sec/frame @ 1080p->4K
```

### 3.2 Cascade 2x+2x (key hybrid innovation)

**Concept:** Replace a single 4x model with two 2x stages. Each stage has fewer parameters per output pixel.

**Source: CASR (CVPR 2024 workshop)** -- efficient cascade network with channel alignment for 4K real-time SR. https://openaccess.thecvf.com/content/CVPR2024W/AI4Streaming/papers/Yoon_CASR_Efficient_Cascade_Network_Structure_with_Channel_Aligned_method_for_4K_Real_Time_Super_Resolution_CVPR_2024_paper.pdf

**Source: CARN** (ECCV 2018) -- the original cascade residual network for fast/accurate/lightweight SR. https://openaccess.thecvf.com/content_ECCV_2018/papers/Namhyuk_Ahn_Fast_Accurate_and_ECCV_2018_paper.pdf

**Why this helps real-time:**
- 2x model processes 4x more pixels per unit time (smaller per-stage compute).
- Each stage can be a different architecture (e.g., RFDN for stage 1, distilled OSEDiff for stage 2).
- Aligns with our Phase 2 plan in docs/plans/realtime_4k_plan.md (cascade 2x+2x + batching + NVENC).

**Action: Phase 2 in our existing realtime plan is already this.** The new research validates it.

### 3.3 Anime-specific hybrid recommendation

For *anime* specifically, the dominant problem is **line art** (sharp, single-pixel-wide contours). Recent research:

- **Evaluating Loss Functions for Illustration SR** (SIBGRAPI 2021): https://doi.org/10.5753/sibgrapi.est.2021.20040 -- compares L1/L2/Sobel/Edge losses for anime.
- **APISR** (CVPR 2024): anime production workflow analysis + balanced twin perceptual loss. https://arxiv.org/html/2403.01598v2
- **Caelum** (community): purpose-built for real-world internet degradation (multi-round WebP/JPEG).

**Recommended hybrid training recipe (for our anime student):**
1. APISR-style API dataset (562 HQ anime videos, I-Frame complexity selection).
2. Twin perceptual loss: LPIPS (photoreal) + anime-domain perceptual (anime VGG or anime CLIP).
3. Adversarial: balanced twin perceptual loss from APISR paper.
4. Edge preservation: Sobel L1 on Y (BT.601) -- already in our EdgeLoss.
5. APISR degradation pipeline (real-world noise + compression + resize) instead of pure Real-ESRGAN pipeline.

**This matches our Phase 4 trajectory:** the feat_weight lever we exposed in I2 + the arch lever in I3 are the right axes. What we lacked was a strong anime-domain perceptual loss for the discriminator side.

---

## 4. Knowledge Distillation for SR (2024-2026)

### 4.1 New techniques surveyed

| Technique | Year | What it distills | Reported gain | Source |
|---|---|---|---|---|
| **Distillation-Supervised Convolutional LoRA** | 2025 | LoRA-style adapter on efficient CNN | +PSNR/SSIM at lower params | https://arxiv.org/html/2504.11271 |
| **Multi-Scale Contrastive-Adversarial Distillation** | ICCVW 2025 | Multi-scale features + adversarial | -- | https://openaccess.thecvf.com/content/ICCV2025W/AIGENS/papers/Ko_Multi-Scale_Contrastive-Adversarial |
| **FRAMER (Frequency-Aligned Self-Distillation)** | Dec 2025 | Frequency-domain, adaptive modulation | HF detail recovery | https://www.alphaxiv.org/abs/2512.01390 |
| **Multi-Granularity Mixture of Priors** | 2024 | Multi-granularity priors | -- | https://arxiv.org/html/2404.02573v1 |
| **Standard KD** (Gao et al, ESRGAN follow-up) | 2024 | Feature/logit KD | -- | https://arxiv.org/pdf/2404.09571 |

### 4.2 What we already do (from existing Phase 4)

- Charbonnier loss (vs L1/L2): robust gradient.
- SSIM + LPIPS (VGG): perceptual.
- EMA on student weights: stable convergence.
- APISR degradation: anime-realistic LR.
- TTA 8x flip/rot: -10% LPIPS, +0.005-0.05 MANIQA (from survey_2026 Tier 1.1).
- Model souping: +0.01 to +0.05 MANIQA (from survey_2026 Tier 1.2).
- 3-phase GAN scheduler: prevents GAN drift (from survey_2026 Tier 1.4).
- D-EMA: mirror on discriminator (from survey_2026 Tier 1.5).

**Gap:** We don't currently use **frequency-domain distillation** (FRAMER-style) or **multi-scale contrastive-adversarial distillation** (ICCVW 2025). These are candidate Phase 5 levers.

### 4.3 Anime-specific KD wisdom (synthesized from sources)

- **Teacher selection matters more than student architecture.** SPAN (val PSNR 31.22) > animevideov3 (29.49) as a feature teacher for RFDN student (our existing data).
- **Warm-start beats from-scratch.** Our Phase 4 I1 + I3 both hit "overshoot" failure because they trained from scratch with adversarial; warm-start (e.g., v1 -> finetune) avoids this.
- **Tune feat_weight carefully.** feat_weight=0 is the safe default for SPAN+adv RFDN (Phase 4 I2 lesson).
- **For anime, prefer feature distillation over logit KD.** Anime color regions are flat; logits carry little signal.

---

## 5. GAN Training for SR (2024-2026)

### 5.1 Key innovations

- **APISR balanced twin perceptual loss** (CVPR 2024): combines photorealistic and anime-domain perceptual features with balanced layer scaling. Source: https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_APISR_Anime_Production_Inspired
- **MSA-ESRGAN** (Sci.Rep 2024): multi-scale attention U-Net discriminator for ESRGAN. https://www.nature.com/articles/s41598-024-78813-5
- **DPO-ESRGAN** (MDPI 2025): Direct Preference Optimization for SR. https://www.mdpi.com/2079-9292/14/17/3357

### 5.2 What we already do

- PatchGAN70 discriminator (in_channels=6 for stacked LR+HR).
- HingeGANLoss.
- TTUR (D lr = 2x G lr).
- Gradient clip max_norm=1.0.
- _adv_lambda anneal (0 -> peak*0.1 -> peak*0.5 -> peak over 21 epochs).
- 3-phase GAN scheduler (from survey_2026 Tier 1.4, NOT YET IMPLEMENTED in our pipeline).

### 5.3 Lessons from Phase 4 (already in PROJECT_MEMORY §6)

- H6 (oversharpening) fires when student PSNR < bicubic PSNR -- we hit this in I1 and I3.
- Adversarial without warm-start = overshoot.
- For anime, anime-domain perceptual loss is the missing piece.

---

## 6. Edge & Line-Art Preservation

### 6.1 Losses for anime line art

- **Sobel L1 on Y channel (BT.601)** -- already in our EdgeLoss.
- **Gradient loss** (StructSR, arxiv 2003.13081) -- penalizes gradient difference between SR and HR.
- **Canny edge loss** -- stronger than Sobel for thin lines.
- **Anime-specific perceptual** (APISR balanced twin) -- incorporates line-preservation into perceptual space.

### 6.2 Datasets

- **API dataset** (APISR): 562 HQ anime videos, I-Frame complexity selection. Source: https://arxiv.org/html/2403.01598v2
- **AnimeSR**: AVC-Train (video-based). https://ar5iv.labs.arxiv.org/html/2206.07038
- **Re-Anime600** (CVPR 2026 workshop): benchmark for anime video quality assessment. https://openaccess.thecvf.com/content/CVPR2026W/AIGENS/papers/Fargetta_Re-Anime600
- **ACvc / AVC-RealLQ**: real-world compression-degraded anime (from survey_2026 candidates C11).

---

## 7. Test-Time Augmentation & Ensembling

### 7.1 What we already have

- D4 8x flip/rot TTA (in GUI as "TTA" toggle). -10% LPIPS, -800% inference cost. Survey 2026 Tier 1.1.

### 7.2 Latest research (2024-2026)

- **NTIRE 2026 results**: many teams now use TTA as standard inference strategy. Source: https://arxiv.org/html/2604.14558v1 (note: 2604.x is CVPR 2026 indexing)
- **Training-Free Model Ensemble** (2026): uses strong-branch selection instead of naive averaging. https://arxiv.org/html/2604.11564v2

### 7.3 Practical advice

- Keep D4 TTA as the default "quality" toggle (8x).
- Avoid expanding to 16x or 32x -- the inference cost grows linearly; quality plateaus.
- For our hybrid: use TTA only on the real-time tier when user explicitly enables.

---

## 8. Inference Acceleration (everyone needs this)

### 8.1 Concrete recommendations

- **TensorRT FP16**: ~2x speedup on RTX 4000. Source: https://developer.nvidia.com/blog/tensorrt-accelerates-stable-diffusion-nearly-2x-faster-w
- **Batch size**: this is the single biggest latency win for video. Source: https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html
- **ONNX Runtime with TensorRT EP**: portable fallback. Source: https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
- **PyTorch native compilation (torch.compile)**: ~30% speedup on supported models.

### 8.2 Our Phase 1 plan

Already covers TensorRT + batching + NVENC. Estimated speed: 35-50 fps @ 4K. No new research changes this.

---

## 9. Decision Matrix -- Sorted by ROI (Best Result in Shortest Time)

**Ranking criterion**: ROI = (expected PSNR + lap_var + perceptual gain) / (effort + risk + infra friction).

| Rank | Option | Wall-time | Expected gain | Risk | Infra |
|---|---|---|---|---|---|
| (--) | **Phase 1 (TensorRT + batching)** -- already planned | 3 days | unchanged quality, 35-50 fps | low | None (planned) |
| (--) | **Phase 2 (Cascade 2x+2x)** -- already planned | 4 days | unchanged quality, 30+ fps | low | None (planned) |
| **1** | **Warm-start v1 -> SRVGG-body + no-adv + LPIPS-heavy** | **1-2 days** | **+lap_var, keep PSNR, 1.7x faster** | **LOW** | **Reuses I3** |
| 2 | APISR-style anime perceptual loss + warm-start | 1 week | +MANIQA, +CLIPIQA, +anime sharpness | LOW-MED | None |
| 3 | FRAMER-style frequency-domain distillation | 1-2 weeks | +HF detail, +edge preservation | LOW | None |
| 4 | Multi-Scale Contrastive-Adversarial KD | 1 week | +LPIPS | MED | None |
| 5 | OSEDiff / SinSR anime fine-tune | 2 weeks | +++ MANIQA, ~1 sec/frame | MED-HIGH | None |
| 6 | MambaIRv2 student distillation | 2-3 weeks | +PSNR potential, same latency tier | MED | Docker build (in progress) |
| 7 | FiDeSR / One-Step Diffusion Transformer fine-tune | 3 weeks | ++++ perceptual, ~1 sec/frame | HIGH | None |

**See RECOMMENDED_PATH.md for detailed comparison and Rank #1 recipe.**

### 9.1 Recommended sequence (re-sorted by ROI)

1. **Now**: Rank #1 -- warm-start v1 + SRVGG body + no adversarial + LPIPS weight 1.0. Reuses I3 code. 1-2 days wall-time. ~21 min run + analysis.
2. **If #1 succeeds**: ship; start Rank #2 (APISR perceptual) in parallel.
3. **If #1 fails**: try #1 with adjusted hyperparams; if still fails, move to #2.
4. **Always**: finish Docker MambaIRv2 build in background; smoke-test; queue Rank #6 as Phase 6 candidate.
5. **Phase 6+**: Rank #5 (OSEDiff), Rank #7 (FiDeSR) as quality-tier options.

### 9.2 Why Rank #1 over MambaIRv2 (despite stronger paper claims)

Despite MambaIRv2 +0.29 dB on Manga109 vs HAT, the path has critical friction:
- Docker image build is 1-2 hours (in progress; failed first attempt at apt python3-pip, restarted with get-pip.py fix)
- Even after build, distillation training is 2-3 weeks of iteration
- The +0.29 dB is vs HAT (heavy transformer), not RFDN -- transferring the gain to RFDN-scale is empirically untested
- Rank #1 reuses all prior work, requires no infra, and addresses the architecture-mismatch root cause from I3

So MambaIRv2 is gated on (1) Docker build success, (2) Rank #1 outcome (does warm-start SRVGG beat v1?), (3) available training budget. See WSL2_DOCKER_PROBE.md for the build status.

---

## 10. Open Questions for User

1. **WSL2 access:** Is WSL2 setup acceptable? (Unlocks MambaIRv2 path.)
2. **Quality tier preset:** Do you want a "best quality" preset in the GUI alongside real-time? (Adds ~1 sec/frame mode.)
3. **OS license:** Is BSD-3 (Real-ESRGAN), MIT (RFDN, OSEDiff), S-Lab License 1.0 (SUPIR) all acceptable for distribution?
4. **GPU budget for training:** New distillation runs are ~1-3 hours each. Acceptable to iterate?
5. **Tier 3 from survey_2026 (DAT2, HAT-Lite, RealPLKSR)** was deferred because the MANIQA gap is closable in Tiers 1+2. Does MambaIRv2's emergence change that calculus?

---

## 11. References

See SOURCES.md for the full per-query URL list (15 queries x 8 hits = ~120 URLs).

Key citations:
- Real-ESRGAN animevideov3: https://github.com/xinntao/Real-ESRGAN
- MambaIRv2 (CVPR 2025): https://arxiv.org/abs/2411.15269
- SUPIR (CVPR 2024): https://github.com/Fanghua-Yu/SUPIR
- SeeSR (CVPR 2024): https://github.com/cswry/SeeSR
- OSEDiff (NeurIPS 2024): https://proceedings.neurips.cc/paper_files/paper/2024/file/a8223b0ad64007423ffb308b0dd92298-Paper-Conference.pdf
- APISR (CVPR 2024): https://arxiv.org/html/2403.01598v2
- LKDN (2024): https://arxiv.org/html/2407.14340v1
- Real-CUGAN: https://github.com/bilibili/ailab/tree/main/Real-CUGAN
- Anime4K v4: https://github.com/bloc97/Anime4K
- FiDeSR (CVPR 2026): https://openaccess.thecvf.com/content/CVPR2026/papers/Kim_FiDeSR
- One-Step Diffusion Transformer (CVPR 2026): https://openaccess.thecvf.com/content/CVPR2026/papers/Fang_One-Step_Diffusion_Transformer

---

**Status:** COMPLETE as research progress document. Awaiting user direction on next action.
