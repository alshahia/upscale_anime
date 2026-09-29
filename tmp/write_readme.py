import os
os.makedirs(r"E:\python projects\upscale_anime\docs\research\anime_sr_2026", exist_ok=True)

readme = r"""# Anime Super-Resolution Landscape -- 2026 Survey

**Generated:** 2026-09-03
**Scope:** Latest 2024-2026 research, models, techniques, and advice for training a 4x anime upscaler.
**Hardware target:** Quadro RTX 4000 8GB (~7.6 GB free).
**Project context:** Real-time (>=25 fps @ 4K) preferred; quality ceiling otherwise open.

## Contents

| File | Purpose |
|---|---|
| REPORT.md | Full research report. Three tiers (real-time, high-quality, hybrid), decision matrix, recommended path, follow-up TODOs. |
| SOURCES.md | Per-query source list (15 Exa searches, ~140 source URLs, organized by category). |
| exa_search_results.json | Raw Exa search results (15 queries x 8 hits = ~120 results with highlights). |

## TL;DR (5 lines)

1. **Real-time <=25 fps @ 4K** is dominated by **lightweight CNNs** (RFDN family, IMDN, BSRN, ShuffleMixer, LKDN, VAPSR, OmniSR, ELSANet) -- our existing RFDN-distill v1 (315K params, 59 ms/frame at native res) sits in this tier.
2. **Quality ceiling** is dominated by **diffusion SR** (SUPIR, SeeSR, DiffBIR, StableSR, OSEDiff, SinSR) -- 5-15 sec/frame on RTX 4090; **one-step variants** (OSEDiff, SinSR, FiDeSR, One-Step Diffusion Transformer) close the gap to ~0.5-1 sec.
3. **Latest breakthrough** (CVPR 2026): **MambaIRv2** matches/beats SwinIR-class transformers at **30% lower MACs** -- strong candidate for the next real-time-quality hybrid if the Windows+WSL2 mamba-ssm wheel works.
4. **Anime-specific SOTA** still anchored by **animevideov3** (Real-ESRGAN), **Real-CUGAN**, **Anime4K v4**, and now **APISR** (CVPR 2024) -- APISR's anime-tailored dataset + balanced twin perceptual loss is the strongest open-source recipe for anime fidelity.
5. **Hybrid approach (recommended)**: lightweight CNN student for real-time path + optional SUPIR/SeeSR/OSEDiff post-pass for hero-shot quality. Use **2x+2x cascade student** to halve inference cost vs single 4x.

## Method note (honesty)

- 15 Exa /search queries via exa-py Python SDK with highlights content mode. Each query returned 8 hits with up to 3 highlights.
- Total cost: ~$0.10 (15 queries x $0.007 each).
- No deep-crawl (no /contents follow-up to confirm details); cross-referencing was done by reading highlights, titles, and URLs only.
- All citations in REPORT.md link to source URLs from Exa results. Treat numbers as "reported by source" not "independently verified".
- The user typed the API key directly in chat -- recommend rotation at https://dashboard.exa.ai/keys after this session.

## Companion files in this session

- tmp/exa_helper.py -- reusable Exa /search wrapper (Python 3.12, exa-py 2.20.0).
- tmp/exa_search_batch.py -- original 12-query batch runner.
- tmp/exa_followup.py -- 3-query follow-up runner.
- tmp/fix_and_dump_followup.py -- label-fixer + dump.
- tmp/exa_results_all.json -- earlier agent.runs artifact (kept for completeness; agent.runs is more expensive than /search).

## Where this feeds the project

- This is a **research progress** artifact, not an implementation plan.
- If you decide to act on it, the natural next step is docs/plans/anime_sr_landscape_action_plan.md (NOT created yet -- pending user direction).
"""

with open(r"E:\python projects\upscale_anime\docs\research\anime_sr_2026\README.md", "w", encoding="utf-8") as f:
    f.write(readme)
print("README.md written, size:", len(readme), "bytes")
