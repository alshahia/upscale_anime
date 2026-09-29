import os

probe_path = r"E:\\python projects\\upscale_anime\\docs\\research\\anime_sr_2026\\WSL2_DOCKER_PROBE.md"
with open(probe_path, encoding="utf-8") as f:
    probe_content = f.read()

old_alt_marker = "## Alternative: skip MambaIRv2 entirely"
if old_alt_marker in probe_content:
    idx = probe_content.index(old_alt_marker)
    # Find next H2 section
    next_h2_idx = len(probe_content)
    for i in range(idx + 1, len(probe_content) - 2):
        if probe_content[i:i+3] == "\n## ":
            next_h2_idx = i + 1
            break
    old_alt = probe_content[idx:next_h2_idx]
    new_alt = """## Alternative / Phase 5 alternatives -- sorted by ROI

For the broader Phase 5 set (not just MambaIRv2), sorted by best quality gain per unit wall-time:

| Rank | Option | Wall-time | Expected gain | Risk |
|---|---|---|---|---|
| **1** | **Warm-start v1 -> SRVGG-body + no-adv + LPIPS-heavy** | **1-2 days** | **+lap_var, keep PSNR, 1.7x faster** | **LOW** |
| 2 | APISR-style anime perceptual loss + warm-start (5A) | 1 week | +MANIQA, +CLIPIQA, +anime sharpness | LOW-MED |
| 3 | FRAMER-style frequency-domain distillation (5B) | 1-2 weeks | +HF detail, +edge preservation | LOW |
| 4 | Multi-Scale Contrastive-Adversarial KD (5E) | 1 week | +LPIPS | MED |
| 5 | OSEDiff / SinSR anime fine-tune (5C) | 2 weeks | +++ MANIQA, ~1 sec/frame | MED-HIGH |
| 6 | MambaIRv2 student distillation | 2-3 weeks (after Docker build) | +PSNR potential | MED |
| 7 | FiDeSR / One-Step Diffusion Transformer fine-tune (5F) | 3 weeks | ++++ perceptual, ~1 sec/frame | HIGH |

**See RECOMMENDED_PATH.md for full analysis and Rank #1 recipe (warm-start v1 -> SRVGG no-adv).**

MambaIRv2 specific notes:
- +0.29 dB on Manga109 vs HAT (heavy transformer, not RFDN-scale)
- 30% lower MACs vs HAT at 256x256 (still higher than RFDN)
- Architectural diversity (state-space vs conv vs transformer)
- Whether that translates to a +PSNR RFDN-scale student is empirically untested

"""
    probe_content = probe_content.replace(old_alt, new_alt, 1)
    with open(probe_path, "w", encoding="utf-8") as f:
        f.write(probe_content)
    print("WSL2_DOCKER_PROBE.md Alternative section updated")
else:
    print("WSL2_DOCKER_PROBE.md Alternative anchor not found")