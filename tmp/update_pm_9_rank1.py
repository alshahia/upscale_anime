"""Add a §9 update history entry for the Rank #1 launch prep (2026-09-03).

Appends a new bullet to the end of the existing §9 history section (the last
non-empty line ends with 'Per Q6, full I3 run preserved...').
"""
PATH = r"E:\\python projects\\upscale_anime\\docs\\PROJECT_MEMORY.md"
with open(PATH, "r", encoding="utf-8", newline="") as f:
    content = f.read()

NL = "\r\n" if "\r\n" in content else "\n"

# Append a new bullet at the very end of §9
new_bullet = (
    "- **2026-09-03**: Phase 5 Rank #1 LAUNCH PREP. Code added (+~90 LOC total): "
    "`--warm-start-mode {strict, partial}` arg in `anime_upscaler/distill.py` with RFDN -> SRVGG layer "
    "mapping (head -> body.0, upsampler.0 -> body.{last}); wrapper script "
    "`scripts/run_rank1_warmstart.ps1`; 2 new tests in `tests/test_tiny_srvgg.py` (partial warm-start "
    "mapping + forward sanity). Stale `python scripts/train.py` recipe in `docs/research/anime_sr_2026/"
    "RECOMMENDED_PATH.md` replaced with the verified `python anime_upscaler/distill.py` invocation. "
    "Smoke test (2 ep, batch 4, 78 s wall): val ep1 PSNR 26.97 dB / lap_var 3708.7; test PSNR 27.69 dB "
    "(vs bicubic 29.58). **CAVEAT discovered**: SRVGG body has an inherent sharpness bias that persists "
    "even with warm-start + no adversarial -- lap_var >3700 at epoch 1 (no adv involved). The Rank #1 "
    "hypothesis may be incomplete; full 40-ep run needed. 30 tests pass (was 28). Docker build pwsh-2 "
    "still in background for Rank #6 (MambaIRv2). New decision log entry in §6."
    + NL
)

with open(PATH, "w", encoding="utf-8", newline="") as f:
    f.write(content.rstrip(NL) + NL + NL + new_bullet)

print(f"OK: appended {len(new_bullet)} chars")
