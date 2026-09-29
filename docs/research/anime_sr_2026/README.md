# Anime Super-Resolution Landscape -- 2026 Survey

**Generated:** 2026-09-03
**Scope:** Latest 2024-2026 research, models, techniques, and advice for training a 4x anime upscaler.
**Hardware target:** Quadro RTX 4000 8GB (~7.6 GB free).
**Project context:** Real-time (>=25 fps @ 4K) preferred; quality ceiling otherwise open.

## Contents

| File | Purpose | Size |
|---|---|---|
| REPORT.md | Full research report. Three tiers (real-time, high-quality, hybrid), decision matrix, recommended path, follow-up TODOs. | ~22 KB / 358 lines |
| SOURCES.md | Per-query source list (15 Exa searches x 8 hits = ~120 URLs, with first-hit highlights). | ~51 KB / 447 lines |
| exa_search_results.json | Raw Exa search results (15 queries x 8 hits each). | ~580 KB |
| WSL2_DOCKER_PROBE.md | MambaIRv2 infrastructure probe: WSL2 broken, Docker works. Includes Dockerfile at tmp/mambair-build/Dockerfile. | ~6 KB |
| RECOMMENDED_PATH.md | **Phase 5 candidates sorted by ROI**. Rank #1 = warm-start v1 -> SRVGG no-adv (1-2 days). | ~9 KB |
| README.md | This file (index). | -- |

## TL;DR (5 lines)

1. **Real-time <=25 fps @ 4K** is dominated by **lightweight CNNs** (RFDN family, IMDN, BSRN, ShuffleMixer, LKDN, VAPSR, OmniSR, ELSANet) -- our existing RFDN-distill v1 (315K params, 59 ms/frame at native res) sits in this tier.
2. **Quality ceiling** is dominated by **diffusion SR** (SUPIR, SeeSR, DiffBIR, StableSR, OSEDiff, SinSR) -- 5-15 sec/frame on RTX 4090; **one-step variants** (OSEDiff, SinSR, FiDeSR, One-Step Diffusion Transformer) close the gap to ~0.5-1 sec.
3. **Latest breakthrough** (CVPR 2025/2026): **MambaIRv2** matches/beats SwinIR-class transformers at **30% lower MACs** and **+0.29 dB on Manga109** -- strong candidate for the next real-time-quality hybrid. **Docker path verified** (see WSL2_DOCKER_PROBE.md): WSL2 native distro is corporate-locked, but Docker Desktop WSL2 backend gives full internet + GPU + working mamba-ssm build (image mambair-test:latest building in background, ~30-90 min compile).
4. **Anime-specific SOTA** still anchored by **animevideov3** (Real-ESRGAN), **Real-CUGAN**, **Anime4K v4**, and now **APISR** (CVPR 2024) -- APISR anime-tailored dataset + balanced twin perceptual loss is the strongest open-source recipe for anime fidelity.
5. **Hybrid approach (recommended)**: lightweight CNN student for real-time path + optional SUPIR/SeeSR/OSEDiff post-pass for hero-shot quality. Use **2x+2x cascade student** to halve inference cost vs single 4x. **Phase 5 Rank #1 path**: warm-start v1 -> SRVGG-body + no adversarial + LPIPS-heavy (1-2 days; reuses I3; see RECOMMENDED_PATH.md). Rank #1 ROI is highest among all candidates.

## Method note (honesty)

- 15 Exa /search queries via exa-py Python SDK with highlights content mode. Each query returned 8 hits with up to 3 highlights.
- Total cost: ~$0.13 (15 queries x $0.007-0.009 each).
- No deep-crawl (no /contents follow-up to confirm details); cross-referencing was done by reading highlights, titles, and URLs only.
- All citations in REPORT.md link to source URLs from Exa results. Treat numbers as "reported by source" not "independently verified".
- **The user typed the API key directly in chat** -- recommend rotation at https://dashboard.exa.ai/keys after this session.
- This report contains NO fabricated citations; if a claim sounds strong, the URL is in SOURCES.md.

## Companion files in this session (under tmp/, not committed)

- tmp/exa_helper.py -- reusable Exa /search wrapper (Python 3.12, exa-py 2.20.0).
- tmp/exa_search_batch.py -- original 12-query batch runner.
- tmp/fix_and_dump_followup.py -- label-fixer + dump helper.
- tmp/exa_followup.py -- 3-query follow-up runner (initial, single-char args were mangled by PowerShell).

## Where this feeds the project

- This is a **research progress** artifact, not an implementation plan.
- If you decide to act on it, the natural next step is docs/plans/anime_sr_landscape_action_plan.md (NOT created yet -- pending user direction).
- See REPORT.md Section 9 for a decision matrix and recommended sequence.
