# Compact Real-Time Anime Upscaler Candidates (2026-07-22)

**Generated:** 2026-07-22 (research session for pre-train sanity-check + alternative-model hunt)
**Scope:** Permissively-licensed 4x anime / illustration upscalers similar in size to SPAN-tiny (≤ ~25 MB / ≤ ~10 M params). Candidates must be open-source, downloadable as a single .pth / .safetensors, and realistically run real-time or semi-real-time on a single consumer GPU (RTX 4000 8GB target).
**Context:** Triggered after deleting `pretrained/span_nomosuni_4x.pth` (warm-start taint, `upsampler.0.bias mean ~0.43`). User wanted alternatives comparable in size to the kept SPAN pretrains.

**Sources surveyed (parallel research agents):**

- HuggingFace model search (`?search=anime upscaler`, `?search=4x anime`, etc.)
- Phhofm GitHub releases + OpenModelDB (`?q=anime`, `?q=compact`)
- xinntao/Real-ESRGAN releases + Kiteretsu77/APISR + TencentARC/AnimeSR
- 2024-2026 lightweight SR papers (ECBSR, MambaIRv2, SAFMN, KAIR, BasicSR)

---

## Ranked Picks (≤ 25 MB, real-time capable)

| # | Model | Size | License | Params | Scale | Source URL | Anime-tuned? | Speed claim |
|---|---|---:|---|---:|---:|---|---|---|
| 1 | `4xLSDIRCompactv2` (Phhofm) | **1.2 MB** | CC BY 4.0 | SRVGGNetCompact (~few hundred K) | 4x | https://github.com/Phhofm/models/releases/download/4xLSDIRCompact2/4xLSDIRCompactv2.pth | No (DIV2K-style) | "fast" -- likely highest throughput in class |
| 2 | `realesr-animevideov3` (xinntao) | **2.5 MB** | BSD-3 | SRVGGNetCompact | 1/2/3/4x | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth | **Yes** (animevideo) | Real-time on consumer GPU; used by AnimeSR authors |
| 3 | `RealESRGANv2-animevideo-xsx2` (xinntao) | **2.4 MB** | BSD-3 | SRVGGNetCompact | x2 | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/RealESRGANv2-animevideo-xsx2.pth | **Yes** (animevideo x2) | Embedded in mpv-AnimeJaNai; real-time |
| 4 | `eranet_N12_pretrain_325k` (NevermindNilas) | **2.7 MB** | MIT | reparameterized CNN (~116K fused) | 4x | https://github.com/NevermindNilas/eranet/releases/download/v1.1.0/eranet_N12_pretrain_325k.pth | No (clean bicubic) | **69.8 FPS @ 1080p→4K RTX 3090 TensorRT-FP16** -- fastest measured |
| 5 | `4x_APISR_GRL_GAN_generator` (Kiteretsu77) | **6.5 MB** | GPL-3.0 | GRL-small (1.03 M) | 4x | https://github.com/Kiteretsu77/APISR/releases/download/v0.1.0/4x_APISR_GRL_GAN_generator.pth | **Yes** (CVPR 2024 anime SOTA) | Lightweight APISR variant; ~10x faster than DAT |
| 6 | `lightweight-real-ESRGAN-anime` (xiongjie) | **10.3 MB** | MIT | RRDB 4-block / 32-ch | 4x | https://huggingface.co/xiongjie/lightweight-real-ESRGAN-anime/resolve/main/RealESRGAN_x4plus_anime_4B32F.pth | **Yes** (illustration) | ~30 ms @ 256→1024 GPU (model card) |
| 7 | `RealESRGAN_x4plus_anime_6B` (xinntao) | **17.9 MB** | BSD-3 | RRDB 6-block (~12 M) | 4x | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth | **Yes** (canonical anime) | ~30 ms @ 720p PyTorch-FP16 on RTX 4000 |
| 8 | `4x_APISR_RRDB_GAN_generator` (Kiteretsu77) | **17.9 MB** | GPL-3.0 | RRDB 6-block (4.47 M) | 4x | https://github.com/Kiteretsu77/APISR/releases/download/v0.2.0/4x_APISR_RRDB_GAN_generator.pth | **Yes** (APISR) | Beats Real-ESRGAN anime on perceptual metrics |
| 9 | `2xBHI_small_span_fast_pretrain` (Phhofm) | **~25 MB** | CC BY 4.0 | SPAN-fast variant (~5 M) | 2x→4x chain | https://github.com/Phhofm/models/releases (search `2xBHI_small_span_fast_pretrain`) | Anime-tuned downstream | ~50% fewer FLOPs vs vanilla SPAN |

## Skip / Caveats

- **ECBSR-m4c8** (Apache-2.0): smallest in class at ~1-3 MB but **no released pretrained .pth**; only configs + training code. Would require a training run before use.
- **Real-CUGAN 4x anime**: ~26 MB, MIT; canonical anime alternative. **Only ONNX release**; PyTorch port is community-maintained (state-dict shape not documented for direct load into ESRGAN-style loaders).
- **AnimeJaNai V3** (mpv bundle): real-time, ONNX-only inside mpv-AnimeJaNai distribution; PyTorch weights are third-party mirrors (unverified licensing).
- **MambaIRv2**: near-real-time, but **Linux-only** (mamba-ssm build hassle on Windows). Already covered by `docs/survey_2026/candidates.md` A1 as opt-in for v7 via `sab_type: mamba_v2`.
- **AnimeSR_v2**: 67 MB, Apache-2.0; NeurIPS 2022. Over size budget.
- **APISR 4x_DAT_GAN_generator**: 87 MB, GPL-3.0; over size budget.
- **Adore 2x** (renarchi, 2026): 5.4 MB, **CC-BY-NC-SA-4.0** -- explicitly real-time anime but **commercial-incompatible** license.

## Ponytail Recommendations

- **Pick #2 realesr-animevideov3** for the smallest anime-tuned option (BSD-3, 2.5 MB, used by AnimeSR authors).
- **Pick #4 eranet** for the fastest measured FPS (MIT, 2.7 MB, 69.8 FPS @ 4K).
- **Pick #1 4xLSDIRCompactv2** for the absolute smallest (CC BY 4.0, 1.2 MB) -- not anime-specific, but smallest in class and pairs well with anime-domain finetune.

## Integration Notes (for follow-up)

All `realesrgan-*` / `RealESRGAN_*` weights are RRDB-shaped -- state-dict is **drop-in** for the existing `neosr_span` / `RealESRGANModel` loaders in this repo after confirming the `params` key shape. `realesr-animevideov3` uses the SRVGGNetCompact (not RRDB) variant; needs a separate loader. `eranet` is a fully custom arch -- needs a thin adapter (the repo exposes `ERANet(C, N)` with `.eval()` fusing branches into 3x3 convs).

`APISR_*` weights load with `torch.load(weights_only=True)` and are reshape-compatible with the project's existing ESRGAN loaders if the GRL/RRDB generator class is registered.

## Verification Tool

`scripts/eval_pretrained.py` (created 2026-07-22) runs the same forward-pass sanity check on any candidate:

```bash
.venv\Scripts\python.exe scripts\eval_pretrained.py --checkpoint pretrained/<file>.pth --input data/test_mini_sr/2.png
```

Outputs: bias stats, output stats, SR PNG at `results/eval_pretrained/`, and verdict (OK / BROKEN). Already validated against the two kept SPAN pretrains (`span_pix_pretrain_4x.pth` OK, deleted `span_nomosuni_4x.pth` BROKEN with bias_mean=0.4324).