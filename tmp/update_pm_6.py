import os

path = r"E:\\python projects\\upscale_anime\\docs\\PROJECT_MEMORY.md"
with open(path, encoding="utf-8") as f:
    content = f.read()

# Anchor: the WSL2 + Docker probe entry we just added
old_marker = "- Action: Once build completes, run a smoke test inside the container (mamba_ssm imports + 1 epoch of MambaIRv2 inference). If smoke passes, MambaIRv2 is unblocked for Phase 5D."

new_marker_addition = """

### 2026-09-03 -- Phase 5 ranked by ROI; Rank #1 = warm-start v1 -> SRVGG no-adv
- Decision: Phase 5 candidates re-sorted by ROI (best quality gain per wall-time). Detailed ranking at docs/research/anime_sr_2026/RECOMMENDED_PATH.md.
- Rank #1 (NEW): Warm-start v1 RFDN -> finetune with SRVGG body + no adversarial + LPIPS weight 1.0. Wall-time: 1-2 days. Reuses I3 code. Expected: PSNR >= 29.0 + lap_var 40-60 + latency 61 ms/frame (1.7x faster than v1).
- Rank #1 is HIGHEST confidence because: (1) reuses all Phase 4 infrastructure, (2) Phase 4 I3 already proved SRVGG body unlocks animevideov3 response signal (in-batch lap_var 3236 at epoch 1 with adv=0), (3) removing --lambda-adv + warm-starting should solve the H6 overshoot that killed I3.
- Rank #2-5 are pure-Python alternatives (no Docker needed). Rank #6 (MambaIRv2) is gated on Docker build completing (~30-60 min remaining). Rank #7 is highest-effort diffusion fine-tune.
- Reversal cost: none -- sorted list is documentation. No code changed.
- Action: pending user direction to pick from rank 1-7. Default recommendation: start Rank #1 as the highest-ROI bet."""

if old_marker in content:
    content = content.replace(old_marker, old_marker + new_marker_addition, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("PROJECT_MEMORY §6 updated with Rank #1 entry")
else:
    print("anchor not found")