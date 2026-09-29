import os

path = r"E:\\python projects\\upscale_anime\\docs\\PROJECT_MEMORY.md"
with open(path, encoding="utf-8") as f:
    content = f.read()

# Use unicode section symbol (same as in file)
old_top = "- **2026-09-03**: Phase 5 research landscape survey committed. 15 Exa /search queries (~0.13 USD) covering SOTA, real-time lightweight, Mamba, diffusion, knowledge distillation, cascade, datasets, inference, GAN, edge, TTA, perceptual metrics, hybrid, anime fine-tune, 2025-2026 publications. Decision matrix at REPORT.md section 9 lists 9 candidate Phase 5 experiments. Docs at docs/research/anime_sr_2026/ (REPORT.md 22.7 KB / 358 lines, SOURCES.md 51 KB / 447 lines, exa_search_results.json 581 KB raw). No code changes. Security: Exa API key was typed directly in chat -- recommend rotation at https://dashboard.exa.ai/keys."

new_top = """- **2026-09-03**: Phase 5 candidates sorted by ROI; Rank #1 = warm-start v1 -> SRVGG no-adv. New doc docs/research/anime_sr_2026/RECOMMENDED_PATH.md (~9 KB) with full ranking + Rank #1 recipe. REPORT.md section 9 decision matrix re-sorted; WSL2_DOCKER_PROBE.md Alternative section re-sorted; PROJECT_MEMORY section 4 candidates re-numbered 5#1..5#7. Rank #1: warm-start v1 RFDN -> SRVGG body + no adversarial + LPIPS weight 1.0 (1-2 days, reuses I3, expected PSNR >= 29.0 + lap_var 40-60 + latency 61 ms/frame). Rank #6 MambaIRv2 gated on Docker build completion (~30-60 min remaining).
- **2026-09-03**: Phase 5 research landscape survey committed. 15 Exa /search queries (~0.13 USD) covering SOTA, real-time lightweight, Mamba, diffusion, knowledge distillation, cascade, datasets, inference, GAN, edge, TTA, perceptual metrics, hybrid, anime fine-tune, 2025-2026 publications. Decision matrix at REPORT.md section 9 lists 9 candidate Phase 5 experiments. Docs at docs/research/anime_sr_2026/ (REPORT.md 22.7 KB / 358 lines, SOURCES.md 51 KB / 447 lines, exa_search_results.json 581 KB raw). No code changes. Security: Exa API key was typed directly in chat -- recommend rotation at https://dashboard.exa.ai/keys."""

if old_top in content:
    content = content.replace(old_top, new_top, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("PROJECT_MEMORY §9 updated")
else:
    print("section 9 anchor not found")