# Sources — Anime Super-Resolution Survey (2024-2026)

**Generated:** 2026-06-20
**Project:** `E:\python projects\upscale_anime`
**Scope:** Anime 4x SR on RTX 4000 Mobile (8GB VRAM). Sources surveyed: arXiv 2024-2026 + Phhofm releases + OpenModelDB + chaiNNer/spandrel + Hugging Face + research repos.
**Self-baseline:** `docs/survey_2026/self_baseline.md`.

---

## A. Anime-Specific SR Papers

| # | Title | Authors | Year/Venue | arXiv / Link | 1-line summary |
|---|---|---|---|---|---|
| A1 | **APISR** | Boyang Wang, Fengyu Yang, Xihang Yu, Chao Zhang, Hanbin Zhao | CVPR 2024 | [arXiv 2403.01598](https://arxiv.org/abs/2403.01598) / [GitHub](https://github.com/KangLiao929/APISR) | Anime-specific real-world SR. API dataset + prediction-oriented compression + XDoG/USM pseudo-GT + balanced twin perceptual loss. **Already integrated in v7.** |
| A2 | **AnimeSR** | Yanze Wu, Xintao Wang, Gen Li, Ying Shan | NeurIPS 2022 | [arXiv 2206.07038](https://arxiv.org/abs/2206.07038) / [GitHub](https://github.com/TencentARC/AnimeSR) | Video-based anime VSR with learnable degradation + AVC dataset. Tangential — on-the-fly synthetic degrader already mirrors premise. |
| A3 | **VQD-SR** | Zixi Tuo, Huan Yang, Jianlong Fu, Yujie Dun, Xueming Qian | 2023 | [arXiv 2303.09826](https://arxiv.org/abs/2303.09826) / [GitHub](https://github.com/researchmm/VQD-SR) | Multi-scale VQ degradation codebook learned from real animation data. Could replace hand-crafted `DegradationPipeline` `anime_heavy` preset. |
| A4 | **Face-to-Cartoon Incremental SR via KD** | Trinetra Devkatte, Shiv Ram Dubey, Satish Kumar Singh, Abdenour Hadid | 2024 | [arXiv 2401.15366](https://arxiv.org/abs/2401.15366) | Incremental KD: CelebA -> iCartoonFace. Inspiration for our MTKD/FAKD distillation to avoid catastrophic forgetting. |
| A5 | **Anime Imaging Enlargement CNN** | Tanakit Intaniyom et al. | 2021 | [arXiv 2110.02321](https://arxiv.org/abs/2110.02321) | Modified SRCNN baseline. Historical reference only. |

## B. General SOTA SR — Transformer / Mamba / Diffusion / Distillation

### Transformer-based

| # | Title | Authors | Year/Venue | Link | 1-line summary |
|---|---|---|---|---|---|
| B1 | **HAT** | Xiangyu Chen et al. | CVPR 2023 (ext. 2025) | [arXiv 2309.05239](https://arxiv.org/abs/2309.05239) / [GitHub](https://github.com/XPixelGroup/HAT) | Hybrid Attention Transformer. Channel + window + overlapping cross-attention. PSNR ceiling but heavy; HAT-Light may train on 8GB. |
| B2 | **DRCT** | Chih-Chung Hsu et al. | CVPRW 2024 (NTIRE winner) | [arXiv 2404.00722](https://arxiv.org/abs/2404.00722) / [GitHub](https://github.com/ming053l/DRCT) | Dense-residual-connected Transformer. Won NTIRE 2024 SR x4. Strong teacher candidate. |
| B3 | **DAT++** | Zhuofan Xia et al. | 2023 | [arXiv 2309.01430](https://arxiv.org/abs/2309.01430) | Deformable attention. Worth considering as SPAN replacement. |
| B4 | **SAFMN** | Long Sun et al. | ICCV 2023 | [arXiv 2302.13800](https://arxiv.org/abs/2302.13800) / [GitHub](https://github.com/sunny2109/SAFMN) | Spatially-Adaptive Feature Modulation. Very lightweight ViT. Fits 8GB. Good teacher candidate. |
| B5 | **DAT2** | Zheng Chen et al. | 2023+ | [GitHub](https://github.com/zhengchen1999/dat2) | Dense Aggregation Transformer v2. Phhofm `4xBHI_dat2_otf_nn` uses this. |

### Mamba-based

| # | Title | Authors | Year/Venue | Link | 1-line summary |
|---|---|---|---|---|---|
| B6 | **MambaIR** | Hang Guo et al. | ECCV 2024 | [arXiv 2402.15648](https://arxiv.org/abs/2402.15648) / [GitHub](https://github.com/csguoh/MambaIR) | First Mamba IR backbone. Beats SwinIR by 0.45 dB. Drop-in for Model B Mamba-PAN. |
| B7 | **MambaIRv2** | Hang Guo et al. | CVPR 2025 | [arXiv 2411.15269](https://arxiv.org/abs/2411.15269) / [GitHub](https://github.com/csguoh/MambaIR) | Attentive state-space restoration. Beats SRFormer by 0.35 dB with 9.3% fewer params. **Already opt-in in v7 (`sab_type: mamba_v2`).** |
| B8 | **Q-MambaIR** | Yujie Chen et al. | 2025/2026 | [arXiv 2503.21970](https://arxiv.org/abs/2503.21970) | INT2-4 quantization. Informative for future INT8 deployment. |

### Diffusion-based

| # | Title | Authors | Year/Venue | Link | 1-line summary |
|---|---|---|---|---|---|
| B9 | **DiffBIR** | Xinqi Lin et al. (XPixel) | ICCV 2023 | [arXiv 2308.15070](https://arxiv.org/abs/2308.15070) / [GitHub](https://github.com/XPixelGroup/DiffBIR) | Restoration + IRControlNet. Too slow standalone; region-adaptive guidance idea transferable. |
| B10 | **PASD** | Tao Yang et al. | ECCV 2024 | [arXiv 2308.14469](https://arxiv.org/abs/2308.14469) / [GitHub](https://github.com/yangxy/PASD) | Pixel-aware cross-attention in SD UNet. |
| B11 | **SeeSR** | Rongyuan Wu et al. | CVPR 2024 | [arXiv 2311.16518](https://arxiv.org/abs/2311.16518) / [GitHub](https://github.com/cswry/SeeSR) | Degradation-aware prompts + LQ-in-noise init. |
| B12 | **FaithDiff** | Junyang Chen et al. | 2024 | [arXiv 2411.18824](https://arxiv.org/abs/2411.18824) / [Project](https://jychen9811.github.io/FaithDiff_page/) | Unfreezes SD; fidelity upper bound. Not realistic on 8GB. |
| B13 | **StableSR** | Jianyi Wang et al. | IJCV 2024 | [arXiv 2305.07015](https://arxiv.org/abs/2305.07015) / [GitHub](https://github.com/IceClear/StableSR) | Time-aware encoder wrapping SD prior. Heavy. |
| B14 | **OSEDiff** | Rongyuan Wu et al. | NeurIPS 2024 | [arXiv 2406.08177](https://arxiv.org/abs/2406.08177) / [GitHub](https://github.com/cswry/OSEDiff) | One-step effective diffusion. Practical on 8GB. |
| B15 | **AdcSR** | Bin Chen et al. | CVPR 2025 | [arXiv 2411.13383](https://arxiv.org/abs/2411.13383) / [GitHub](https://github.com/Guaishou74851/AdcSR) | 73% time + 78% compute reduction vs OSEDiff. Diffusion-GAN hybrid. |
| B16 | **TinySR** | Linwei Dong et al. | 2025/2026 | [arXiv 2508.17434](https://arxiv.org/abs/2508.17434) | 5.68× speedup, 83% param reduction via inter-block activation + VAE compression. |
| B17 | **RealDGen** | Long Peng et al. | ICLR 2025 | [arXiv 2406.07255](https://arxiv.org/abs/2406.07255) | Content-degradation decoupled diffusion for realistic LR generation from unpaired data. Alternative to hand-crafted degrader. |

## C. Perceptual Losses

| # | Title | Authors | Year/Venue | Link | 1-line summary |
|---|---|---|---|---|---|
| C1 | **DISTS** | Keyan Ding, Kede Ma | T-IP 2021 | (IEEE T-IP) | VGG/AlexNet feature distance with SSIM-like structure. **Already in finetuner.** |
| C2 | **FDL** (in APISR) | Wang et al. | CVPR 2024 | APISR | DINOv2 + sliced Wasserstein in frequency domain. **Already in v7.** |
| C3 | **Patch-NCE / CUT** | Park et al. | ECCV 2020 | (ECCV 2020) | Mutual info via NCE between patches. Lightweight (no VGG/DINO). |
| C4 | **Relativistic GAN / Projected GAN** | Jolicoeur-Martineau / Sauer et al. | 2019 / ICCV 2021 | (older) | Relativistic D + Projected GAN (frozen VGG/DINO features in D). |

## D. Degradation Pipelines

| # | Title | Authors | Year/Venue | Link | 1-line summary |
|---|---|---|---|---|---|
| D1 | **Real-ESRGAN** | Xintao Wang et al. | 2021 | [arXiv 2107.10833](https://arxiv.org/abs/2107.10833) / [GitHub](https://github.com/xinntao/Real-ESRGAN) | High-order degradation. **Already basis of v7 pipeline.** |
| D2 | **VQFR** | Yuchao Gu et al. | ECCV 2022 (Oral) | [arXiv 2205.06803](https://arxiv.org/abs/2205.06803) / [Project](https://ycgu.site/projects/vqfr) | VQ codebook for face features. Anime has repeated tokens (eyes, strokes). |
| D3 | **RestoreFormer++** | Zhouxia Wang et al. | TPAMI 2023 | [arXiv 2308.07228](https://arxiv.org/abs/2308.07228) | Cross-attention with reconstruction priors. |
| D4 | **CodeFormer++** | Reddem et al. | 2025 | [arXiv 2510.04410](https://arxiv.org/abs/2510.04410) | Identity-preserving BFR. Restoration vs generation balance. |

## E. Optimizers / Training Tricks

| # | Title | Authors | Year/Venue | Link | 1-line summary |
|---|---|---|---|---|---|
| E1 | **SAM** | Foret et al. | ICLR 2021 | [arXiv 2010.01412](https://arxiv.org/abs/2010.01412) | Sharpness-Aware Minimization. 2x compute per step. |
| E2 | **Lion** | Chen et al. (Google) | 2023 | [arXiv 2302.06675](https://arxiv.org/abs/2302.06675) | Sign-momentum optimizer. Single state vs AdamW's 2. Frees ~30% optimizer memory. |
| E3 | **MAE** | He et al. | CVPR 2022 | [arXiv 2111.06377](https://arxiv.org/abs/2111.06377) / [GitHub](https://github.com/facebookresearch/mae) | MAE self-supervised pretraining. |
| E4 | **Lookahead** | Zhang et al. | NeurIPS 2019 | [arXiv 1907.08610](https://arxiv.org/abs/1907.08610) | k steps forward, 1 step back. Wrapper around AdamW. |

## F. Inference-Time Tricks

| # | Title | Authors | Year/Venue | Link | 1-line summary |
|---|---|---|---|---|---|
| F1 | **TTA in EDSR** | Lim et al. | CVPRW 2017 | (older) | 8x flip/rotation average at inference. +0.1-0.3 dB PSNR / -5% LPIPS common. |
| F2 | **Model Souping** | Wortsman et al. | ICML 2022 | [arXiv 2203.05482](https://arxiv.org/abs/2203.05482) | Average weights of finetuned variants. Free inference improvement. |
| F3 | **Consistency Models** | Song et al. | ICML 2023 | [arXiv 2303.01469](https://arxiv.org/abs/2303.01469) / [GitHub](https://github.com/openai/consistency_models) | Single-step generative. Reference for any future 1-step anime diffusion. |
| F4 | **LCMSR** | Xiaohui Sun et al. | 2025 | [arXiv 2503.19505](https://arxiv.org/abs/2503.19505) | Latent consistency model for SR. |

## G. Practitioner / Community (Phhofm)

| # | Name | Arch | Scale | Iter | Date | License | Notes |
|---|---|---|---|---|---|---|---|
| G1 | `4xBHI_small_hat-l` | HAT-L | 4x | 150k | 2026-02-04 | CC-BY-4.0 | Sharp / FDL variants. Heavy. |
| G2 | `4xBHI_dat2_otf_nn` | DAT2 | 4x | 220k | 2024-12-27 | CC-BY-4.0 | Handles JPEG+resize, OTF no-noise. |
| G3 | `2xBHI_small_span_pretrain` | SPAN | 2x | 100k | 2025-05-19 | CC-BY-4.0 | L1+MS-SIM. Direct lineage match. |
| G4 | `2xBHI_small_span_fast_pretrain` | SPAN_fast (32-ch, Mish, kaiming) | 2x | 100k | 2025-05-19 | CC-BY-4.0 | 2x faster than vanilla SPAN. NTIRE 2025 efficient-SR. **Strong candidate for 8GB-friendly warm-start swap.** |
| G5 | `4xNomosUni_span_multijpg` | SPAN | 4x | 57k | 2023-12-09 | CC-BY-4.0 | **Our current lineage.** Pretrained from `spanx4_ch48.pth`. |
| G6 | `2xNomosUni_span_multijpg` | SPAN | 2x | 37k | 2023-12-13 | CC-BY-4.0 | 2x chain. |
| G7 | `2xNomosUni_esrgan_multijpg` | RRDBNet | 2x | 110k | 2023-12-20 | CC-BY-4.0 | Strong 2x chain candidate. |
| G8 | `4xNomosUniDAT_otf` | DAT | 4x | 150k | 2023-09-23 | CC-BY-4.0 | Anime+photo robust. Heavy. |
| G9 | `2xNomosUni_compact_multijpg_ldl` | SRVGGNetCompact | 2x | 218k | 2024-01-11 | CC-BY-4.0 | Cheapest 2x chain. |
| G10 | `4xDRCT-mssim-pretrains` (drct-s/drct/drct-l) | DRCT | 4x | 75-108k | 2024-04-28 | CC-BY-0.4 | drct-s may fit 8GB at low crop. |
| G11 | `4xmssim_mosr_pretrain` | MoSR (umzi2) | 4x | 420k | 2024-08-25 | CC-BY-0.4 | Photography focus. |
| G12 | `2xPublic_realplksr_dysample_layernorm_real_nn` | RealPLKSR + Dysample + LayerNorm | 2x | 120k | 2025 | Apache-2.0 | Newer arch line. |
| G13 | `2xParagonSR_Nano_gan` | ParagonSR Nano | 2x | n/a | 2025-11-14 | CC-BY-4.0 | Phhofm's new network (R3GAN, ConvNeXt perceptual, clip-contrastive). **Not anime.** |

## H. OpenModelDB (669 models)

| # | Name | Arch | Scale | Author | Notes |
|---|---|---|---|---|---|
| H1 | `2x-Adore` | Real-CUGAN | 2x | renarchi | Realtime 1080p anime. Fast. |
| H2 | `StarSample V2` (HQ/Lite/NS) | HAT-L / ESRGAN / SPAN-S | 1x, 2x | `.derpy.` | Most-curated cartoon SR. **Lite=SPAN-S matches our stack.** |
| H3 | `Archiver Medium/Soft/Rough/RGB/AntiLines` | ESRGAN | 1x | Loganavter | Film-grain / scratch removal pre-step. |
| H4 | `NES-Composite-2-RGB` | Compact / OmniSR | 1x | pokepress | Reference for OmniSR. |

## I. chaiNNer / spandrel (v0.4.2, Feb 2026)

| # | Architecture | License | Status |
|---|---|---|---|
| I1 | SPAN | MIT | Stable, in core pkg. **Same arch as our stack.** |
| I2 | RealPLKSR + PLKSR | MIT | Stable. |
| I3 | HAT / HAT-L | Apache-2.0 | Stable. Heavy. |
| I4 | DRCT (S / base / L) | Apache-2.0 | Stable. |
| I5 | DAT / DAT-2 | Apache-2.0 | Stable. |
| I6 | Real-CUGAN (CUNet) | Apache-2.0 | Stable. |
| I7 | MoSR / MoESR | MIT | Stable. |
| I8 | AuraSR / AuraSR-v2 | cc / Apache-2.0 | Stable. fal.ai 0.6B params. |
| I9 | MambaSR / Mamba-PAN | varies | Experimental in `spandrel_extra_arches`. |

## J. Hugging Face

| # | Model | Author | DLs/mo | Likes | Arch | Date | License | Notes |
|---|---|---|---|---|---|---|---|---|
| J1 | `fal/AuraSR-v2` | fal | 577 | 333 | GigaGAN | 2024-07-30 | Apache-2.0 | 0.6B params, 4x. Fast on RTX 4000. AI-gen focus. |
| J2 | `fal/AuraSR (v1)` | fal | 167 | 307 | GigaGAN | 2024-07 | cc | Older v1. |
| J3 | `Kiteretsu77/APISR` (HikariDawn) | Kiteretsu77 | (Space) | 141 | GRL custom (anime-tuned) | 2024 | GPL-3.0 | **Single most relevant anime SR research model.** 2x/4x releases with APISR-style degradation. CLIPIQA/MANIQA/NIQE/TOPIQ benchmarks. |

## K. Major Anime SR Research / Competitions

| # | Project | Org | Year | Arch | Notes |
|---|---|---|---|---|---|
| K1 | **AnimeSR** | Tencent ARC | NeurIPS 2022 | ESRGAN + compression-degrade | First anime SR w/ AVC dataset; AnimeSR_v2 better naturalness. |
| K2 | **APISR** | Kiteretsu77 / UMich | CVPR 2024 | GRL + IC9600 dataset + XDoG/USM | **Most architecturally aligned with v7.** |
| K3 | **Real-CUGAN** | Bilibili AI Lab | 2020-2022 | CUNet | Million-scale anime patches; 2x/3x/4x; 1.5GB VRAM min. |

---

## Summary Stats

- **Total sources surveyed:** 49 unique items (5 anime + 17 general + 4 loss + 4 degradation + 4 optimizer + 4 inference + 13 community).
- **Already in our stack:** APISR (A1), DISTS (C1), FDL (C2), Relativistic GAN (C4), Real-ESRGAN degradation (D1), EMA (Real-ESRGAN trick), MambaIRv2 opt-in (B7), GAN training (Real-ESRGAN-derived).
- **Not in our stack — high relevance:**
  - Architecture: HAT-Light, DRCT-S, SAFMN, MoSR, MambaIR (v1), RealPLKSR+Dysample.
  - Loss: Patch-NCE, Projected-GAN discriminator.
  - Degradation: VQD-SR (learned codebook).
  - Optimizer: SAM, Lion, Lookahead.
  - Inference: TTA, Model souping.
  - Warm-start: `2xBHI_small_span_fast_pretrain` (G4) — drop-in replacement.

## Unfilled Gaps (webfetch limits)

1. CLIPIQA/MANIQA/NIQE for individual Phhofm releases — release notes have only slow.pics visuals + PSNR vs Urban100 / BHI100.
2. Exact file sizes / param counts for Phhofm assets — 2GB GitHub limit is the practical max.
3. Kaggle anime SR competitions 2024-2026 — none surfaced. NTIRE 2025 Efficient SR is closest academic SOTA competition.
4. Full HF model grid for "anime super resolution" — second-pass search with `image-to-image` pipeline_tag filter needed.

