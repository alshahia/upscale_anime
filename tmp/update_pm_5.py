import os

path = r"E:\\python projects\\upscale_anime\\docs\\PROJECT_MEMORY.md"
with open(path, encoding="utf-8") as f:
    content = f.read()

# Find by unicode section symbol
old_marker = "### Phase 5 candidates (NOT approved yet"
if old_marker in content:
    idx = content.index(old_marker)
    # Find end of section (next ### or ---)
    end_idx = len(content)
    for i in range(idx + 1, len(content) - 4):
        if content[i:i+4] == "\n###" or content[i:i+4] == "\n---":
            end_idx = i
            break
    old_block = content[idx:end_idx]
    new_block = """### Phase 5 candidates -- SORTED BY ROI (best quality gain / wall-time)

Ranking criterion: ROI = (expected PSNR + lap_var + perceptual gain) / (effort + risk + infra friction). Full analysis in docs/research/anime_sr_2026/RECOMMENDED_PATH.md.

- [ ] **5#1 (NEW 2026-09-03): Warm-start v1 -> SRVGG-body + no-adv + LPIPS-heavy** (S, 1-2 days) -- reuses I3 code; expected PSNR >= 29.0 + lap_var 40-60 + latency 61 ms/frame (1.7x faster than v1). HIGHEST confidence of all candidates.
- [ ] **5#2: APISR-style anime perceptual loss** (M, 1 week) -- add anime-domain perceptual features to our recipe; expected +0.02-0.05 MANIQA. PREREQUISITE: confirm APISR balanced twin perceptual loss licensing.
- [ ] **5#3: FRAMER-style frequency-domain distillation** (M, 1-2 weeks) -- frequency-aligned self-distillation; +HF detail. Low risk.
- [ ] **5#4: Multi-Scale Contrastive-Adversarial KD** (M, 1 week) -- ICCVW 2025 method; +LPIPS.
- [ ] **5#5: OSEDiff / SinSR anime fine-tune** (L, 2 weeks) -- 1-step diffusion as quality-tier preset; +++ MANIQA. Heavy compute.
- [ ] **5#6: MambaIRv2 student distillation** (L, 2-3 weeks) -- Docker build path viable (per WSL2_DOCKER_PROBE.md; mambair-test image building in background). +PSNR potential vs RFDN at same latency tier. Gated on Docker build completion (~30-60 min remaining).
- [ ] **5#7: FiDeSR / One-Step Diffusion Transformer anime fine-tune** (L, 3 weeks) -- CVPR 2026 SOTA; ++++ perceptual. Quality tier only.

"""
    content = content.replace(old_block, new_block, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("PROJECT_MEMORY §4 reordered")
else:
    print("marker not found")