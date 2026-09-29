import os

report_path = r"E:\\python projects\\upscale_anime\\docs\\research\\anime_sr_2026\\REPORT.md"
with open(report_path, encoding="utf-8") as f:
    content = f.read()

old_section9 = (
    "## 9. Decision Matrix -- What to Try Next\n\n"
    "| Option | Effort | Risk | Real-time? | Quality vs animevideov3 | Recommendation |\n"
    "|---|---|---|---|---|---|\n"
    "| **A. Phase 1 (TensorRT + batching) -- we already planned this** | S (3 days) | low | YES (35-50 fps) | unchanged | **Ship now** |\n"
    "| **B. Phase 2 (Cascade 2x+2x + batching)** | M (4 days) | low | YES (30+ fps) | unchanged or + | **Ship after A** |\n"
    "| **C. APISR-style anime perceptual loss in our recipe** | M (1 week) | med | unchanged | +MANIQA, +CLIPIQA | **Phase 5 candidate** |\n"
    "| **D. OSEDiff / SinSR fine-tune on API dataset** | L (2 weeks) | med-high | NO (~1 sec/frame) | +++ MANIQA | **Phase 5 quality-tier** |\n"
    "| **E. MambaIRv2 student distillation** | L (2-3 weeks) | med | YES (likely) | + vs RFDN | **Phase 5 candidate IF WSL2 ready** |\n"
    "| **F. Frequency-domain distillation (FRAMER-style)** | M (1-2 weeks) | low | unchanged | + on edges/HF | **Phase 5 candidate** |\n"
    "| **G. FiDeSR / One-Step Diffusion Transformer fine-tune** | L (3 weeks) | high | NO | ++++ | **Phase 6 candidate** |\n"
    "| **H. Multi-Scale Contrastive-Adversarial KD** | M (1 week) | med | unchanged | + LPIPS | **Phase 5 candidate** |\n"
    "| **I. Direct Preference Optimization (DPO-ESRGAN)** | L (3 weeks) | high | unchanged | ++ perceptual | **Hold for Phase 6** |\n\n"
    "### 9.1 Recommended sequence (synthesized from this research)\n\n"
    "1. **Now:** Phase 1 from realtime_4k_plan (TensorRT + batching + NVENC) -- 3 days, hits 35-50 fps.\n"
    "2. **Next:** Phase 2 from realtime_4k_plan (cascade 2x+2x) -- 4 days, hits 30+ fps with same quality.\n"
    "3. **Phase 5 (proposed):** APISR-style anime perceptual loss + warm-start from v1 + frequency-domain distillation. Expected: +0.02 to +0.05 MANIQA, -10% LPIPS, no speed regression. ~1 week.\n"
    "4. **Phase 6 (proposed, optional):** OSEDiff fine-tune on API dataset for quality-tier preset. ~1 sec/frame. ~2 weeks.\n"
    "5. **Phase 7 (proposed, hold):** MambaIRv2 distillation, requires WSL2. ~3 weeks."
)

new_section9 = (
    "## 9. Decision Matrix -- Sorted by ROI (Best Result in Shortest Time)\n\n"
    "**Ranking criterion**: ROI = (expected PSNR + lap_var + perceptual gain) / (effort + risk + infra friction).\n\n"
    "| Rank | Option | Wall-time | Expected gain | Risk | Infra |\n"
    "|---|---|---|---|---|---|\n"
    "| (--) | **Phase 1 (TensorRT + batching)** -- already planned | 3 days | unchanged quality, 35-50 fps | low | None (planned) |\n"
    "| (--) | **Phase 2 (Cascade 2x+2x)** -- already planned | 4 days | unchanged quality, 30+ fps | low | None (planned) |\n"
    "| **1** | **Warm-start v1 -> SRVGG-body + no-adv + LPIPS-heavy** | **1-2 days** | **+lap_var, keep PSNR, 1.7x faster** | **LOW** | **Reuses I3** |\n"
    "| 2 | APISR-style anime perceptual loss + warm-start | 1 week | +MANIQA, +CLIPIQA, +anime sharpness | LOW-MED | None |\n"
    "| 3 | FRAMER-style frequency-domain distillation | 1-2 weeks | +HF detail, +edge preservation | LOW | None |\n"
    "| 4 | Multi-Scale Contrastive-Adversarial KD | 1 week | +LPIPS | MED | None |\n"
    "| 5 | OSEDiff / SinSR anime fine-tune | 2 weeks | +++ MANIQA, ~1 sec/frame | MED-HIGH | None |\n"
    "| 6 | MambaIRv2 student distillation | 2-3 weeks | +PSNR potential, same latency tier | MED | Docker build (in progress) |\n"
    "| 7 | FiDeSR / One-Step Diffusion Transformer fine-tune | 3 weeks | ++++ perceptual, ~1 sec/frame | HIGH | None |\n\n"
    "**See RECOMMENDED_PATH.md for detailed comparison and Rank #1 recipe.**\n\n"
    "### 9.1 Recommended sequence (re-sorted by ROI)\n\n"
    "1. **Now**: Rank #1 -- warm-start v1 + SRVGG body + no adversarial + LPIPS weight 1.0. Reuses I3 code. 1-2 days wall-time. ~21 min run + analysis.\n"
    "2. **If #1 succeeds**: ship; start Rank #2 (APISR perceptual) in parallel.\n"
    "3. **If #1 fails**: try #1 with adjusted hyperparams; if still fails, move to #2.\n"
    "4. **Always**: finish Docker MambaIRv2 build in background; smoke-test; queue Rank #6 as Phase 6 candidate.\n"
    "5. **Phase 6+**: Rank #5 (OSEDiff), Rank #7 (FiDeSR) as quality-tier options.\n\n"
    "### 9.2 Why Rank #1 over MambaIRv2 (despite stronger paper claims)\n\n"
    "Despite MambaIRv2 +0.29 dB on Manga109 vs HAT, the path has critical friction:\n"
    "- Docker image build is 1-2 hours (in progress; failed first attempt at apt python3-pip, restarted with get-pip.py fix)\n"
    "- Even after build, distillation training is 2-3 weeks of iteration\n"
    "- The +0.29 dB is vs HAT (heavy transformer), not RFDN -- transferring the gain to RFDN-scale is empirically untested\n"
    "- Rank #1 reuses all prior work, requires no infra, and addresses the architecture-mismatch root cause from I3\n\n"
    "So MambaIRv2 is gated on (1) Docker build success, (2) Rank #1 outcome (does warm-start SRVGG beat v1?), (3) available training budget. See WSL2_DOCKER_PROBE.md for the build status."
)

old_str = "".join(old_section9)
new_str = "".join(new_section9)
if old_str in content:
    content = content.replace(old_str, new_str, 1)
    print("REPORT.md section 9 updated")
else:
    print("REPORT.md section 9 anchor NOT FOUND")
    # debug
    if "Decision Matrix -- What to Try Next" in content:
        idx = content.index("Decision Matrix -- What to Try Next")
        print("Found marker at idx", idx)
        print("ctx:", content[idx:idx+200])

with open(report_path, "w", encoding="utf-8") as f:
    f.write(content)