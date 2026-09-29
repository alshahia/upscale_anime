"""Insert a new decision log entry into PROJECT_MEMORY.md after the existing
'Phase 5 ranked by ROI' entry (line 360) and before the §7 divider (line 362).
Preserves CRLF line endings.
"""
import sys

PATH = r"E:\\python projects\\upscale_anime\\docs\\PROJECT_MEMORY.md"
with open(PATH, "r", encoding="utf-8", newline="") as f:
    content = f.read()

NL = "\r\n" if "\r\n" in content else "\n"

# Anchor: the existing Phase 5 ranked entry ends with line 360
# 'Action: pending user direction to pick from rank 1-7. Default recommendation:'
# 'start Rank #1 as the highest-ROI bet.' followed by blank line, then '---'.
anchor = (
    "- Action: pending user direction to pick from rank 1-7. Default recommendation: "
    "start Rank #1 as the highest-ROI bet." + NL + NL + "---" + NL + NL
)

if anchor not in content:
    print("FATAL: anchor not found", file=sys.stderr)
    sys.exit(1)

new_entry = (
    "### 2026-09-03 -- Phase 5 Rank #1 LAUNCH PREP (warm-start v1 -> SRVGG no-adv)" + NL + NL
    + "- Decision: prepare infrastructure for Rank #1 (warm-start v1 RFDN -> TinySRVGGStudent, no adversarial). User chose to keep Docker build (Rank #6 MambaIRv2) running in background and prepare Rank #1 in parallel." + NL
    + "- Method: added --warm-start-mode {strict, partial} flag to distill.py with RFDN -> SRVGG layer mapping; fixed stale CLI recipe in RECOMMENDED_PATH.md (scripts/train.py -> anime_upscaler/distill.py); added wrapper script scripts/run_rank1_warmstart.ps1; added 2 new tests in tests/test_tiny_srvgg.py for partial warm-start." + NL
    + "- Code changes (+~90 LOC total):" + NL
    + "  - anime_upscaler/distill.py: argparse --warm-start-mode {strict, partial} (default strict for backward compat); partial path maps RFDN head -> SRVGG body.0 + RFDN upsampler.0 -> SRVGG body.{last}; logs which keys were mapped vs ignored." + NL
    + "  - tests/test_tiny_srvgg.py: 2 new tests (test_partial_warmstart_maps_rfdn_head_to_srvgg_body0 + test_partial_warmstart_forward_works_after_mapping). Total tests: 30 (was 28)." + NL
    + "  - scripts/run_rank1_warmstart.ps1 (NEW): wrapper for the full 40-epoch run. Flags: -Smoke, -Epochs, -BatchSize, -Lr, -OutDir, -V1Ckpt, -Python, -SkipWarmStart. Smoke goes to a separate subdir so it doesn't clobber the prod out-dir." + NL
    + "  - docs/research/anime_sr_2026/RECOMMENDED_PATH.md: stale `python scripts/train.py` recipe replaced with verified CLI invocation. Caveat about SRVGG body sharpness bias added (see below)." + NL
    + "- Recipe (verified working, smoke 2 ep on RTX 4000, ~80 s wall):" + NL
    + "  .venv\\\\Scripts\\\\python.exe anime_upscaler\\\\distill.py --resume pretrained\\\\RFDN_distill_v1_4x_student.pth --fresh-epoch --warm-start-mode partial --teacher animevideov3 --arch srvgg --lambda-adv 0 --feat-weight 1.0 --shortcut-anneal off --epochs 40 --batch-size 16 --lr 5e-5 --out-dir runs\\\\distill_v3_4x_srvgg_warmstart" + NL
    + "  Or via wrapper: `powershell -ExecutionPolicy Bypass -File scripts\\\\run_rank1_warmstart.ps1`" + NL
    + "- Smoke results (2 epochs, batch 4, 51 + 27 s = 78 s):" + NL
    + "  - val PSNR ep 1 = 26.97 dB (vs bicubic 28.55, teacher 29.13); lap_var 3708.7" + NL
    + "  - val PSNR ep 2 = 27.01 dB; lap_var 3706.9" + NL
    + "  - test PSNR = 27.69 dB (vs bicubic 29.58, teacher 29.12)" + NL
    + "- **CAVEAT DISCOVERED (2026-09-03)**: SRVGG body has an inherent sharpness bias that PERSISTS even with warm-start + no adversarial. lap_var >3700 at epoch 1 (no adversarial involved) -- the architecture itself overshoots. This is WORSE than I3's epoch-1 lap_var 3236 (which had adv=0.001 ramp). The Rank #1 hypothesis ('warm-start + no-adv solves H6') may be INCOMPLETE -- the SRVGG body alone triggers overshoot. Full 40-epoch run needed to confirm whether PSNR recovers to >=29.0 or stays below bicubic." + NL
    + "- If PSNR fails to recover, fallback options to try (in order):" + NL
    + "  - (a) Add a code change to push LPIPS weight higher (Phase 3 spec currently hardcodes 0.5*L1 + 1.0*LPIPS for SRVGG)." + NL
    + "  - (b) Try --warm-start-mode partial with v3 ckpt (warmer start, more conservative)." + NL
    + "  - (c) Drop the warm-start idea entirely and rely on --lambda-adv 0 + LPIPS-heavy alone." + NL
    + "  - (d) Move to Rank #2 (APISR-style perceptual) which doesn't carry the SRVGG-bias baggage." + NL
    + "- Reversal cost: low -- new code is opt-in (--warm-start-mode partial); wrapper is independent; new tests are additive." + NL
    + "- Action: Run the full 40-epoch Rank #1 launch when user approves. Default expectation: PSNR 29.0-29.5 dB + lap_var 40-60 + 61 ms/frame; if PSNR stays below bicubic at epoch 20, halt and apply fallback (a)." + NL + NL
)

new_content = content.replace(anchor, anchor.replace("---" + NL + NL, new_entry + "---" + NL + NL, 1), 1)
# Verify
if new_content == content:
    # Maybe the anchor didn't end with --- -- fall back: insert BEFORE the §7 divider
    print("Direct replace failed; falling back to anchor-by-section")
    div = "## 7. Halt conditions / guardrails"
    if div not in content:
        print("FATAL: §7 divider not found", file=sys.stderr)
        sys.exit(1)
    # Insert entry BEFORE the §7 divider
    insert_pos = content.find(div)
    new_content = content[:insert_pos] + new_entry + content[insert_pos:]

with open(PATH, "w", encoding="utf-8", newline="") as f:
    f.write(new_content)

print(f"OK: {len(content)} -> {len(new_content)} (delta {len(new_content) - len(content)})")
