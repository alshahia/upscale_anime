import re
import sys

PATH = r"E:\\python projects\\upscale_anime\\docs\\research\\anime_sr_2026\\RECOMMENDED_PATH.md"
with open(PATH, "r", encoding="utf-8", newline="") as f:
    content = f.read()

NL = "\r\n"

# Use a much simpler match: find the block from '### Expected recipe' through the
# last ``` of the bash code block, by counting lines.
start_marker = "### Expected recipe"
s_idx = content.find(start_marker)
if s_idx == -1:
    print("No start marker"); sys.exit(1)

# Find the ``` that closes the bash block
# The bash block opens with ```bash and closes with ``` -- need to find the second ```
first_fence = content.find("```", s_idx)
print(f"First fence at {first_fence}")
# Second ``` closes the bash block
second_fence = content.find("```", first_fence + 3)
print(f"Second fence at {second_fence}")

# Find end of line after second fence
end_idx = content.find(NL, second_fence) + len(NL)
print(f"End at {end_idx}")

old_block = content[s_idx:end_idx]
print(f"Old block len: {len(old_block)}")
print(f"Old block first 50: {repr(old_block[:50])}")
print(f"Old block last 50: {repr(old_block[-50:])}")

# Now build new block in LF and substitute
new_block = (
    "### Expected recipe\n\n"
    "**Entry point:** `python anime_upscaler/distill.py` (NOT `scripts/train.py` -- that does not exist).\n"
    "**CLI mapping notes (verified against `distill.py --help` on 2026-09-03):**\n"
    "- `--warm-start-from` does not exist -> use `--resume <path>` + `--fresh-epoch` (warm-start semantics built-in).\n"
    "- `--lambda-lpips 1.0` and `--lambda-pixel 1.0` do not exist -> LPIPS weight is fixed at 1.0 and pixel L1 weight at 0.5 in the Phase 3 SRVGG recipe (see `distill.py:583-587`).\n"
    "- `--warm-start-mode partial` is required because v1 (RFDN, 315K) has different layer names than TinySRVGGStudent (317K). Without `partial`, `--resume` raises RuntimeError on key mismatch. The new mode copies the 4 matching conv weights (head + upsampler.0) and randomises the middle 12-conv stack.\n"
    "\n"
    "```bash\n"
    "# From repo root, with venv active:\n"
    ".venv\\\\Scripts\\\\python.exe anime_upscaler\\\\distill.py ^\n"
    "    --resume pretrained\\\\RFDN_distill_v1_4x_student.pth ^\n"
    "    --fresh-epoch ^\n"
    "    --warm-start-mode partial ^\n"
    "    --teacher animevideov3 ^\n"
    "    --arch srvgg ^\n"
    "    --lambda-adv 0 ^\n"
    "    --feat-weight 1.0 ^\n"
    "    --shortcut-anneal off ^\n"
    "    --epochs 40 ^\n"
    "    --batch-size 16 ^\n"
    "    --lr 5e-5 ^\n"
    "    --out-dir runs\\\\distill_v3_4x_srvgg_warmstart\n"
    "```\n\n"
    "For convenience, a wrapper script that applies these flags + halt guards is\n"
    "provided at `scripts/run_rank1_warmstart.ps1`.\n\n"
    "### Smoke test (2026-09-03, 2 epochs, batch 4)\n\n"
    "Verified that the cross-arch partial warm-start infrastructure works:\n"
    "- 4 keys mapped: `body.0.weight/bias` <- RFDN `head.weight/bias`; `body.26.weight/bias` <- RFDN `upsampler.0.weight/bias`.\n"
    "- Middle 12-conv stack + PReLUs stay at random init.\n"
    "- Val epoch 1: PSNR 26.97 dB (vs bicubic 28.55, teacher 29.13); **lap_var 3708.7** (vs I3 epoch-1 lap_var 3236 with from-scratch+adv=0.001).\n"
    "- Val epoch 2: PSNR 27.01 dB; lap_var 3706.9.\n"
    "- Test PSNR 27.69 dB (vs bicubic 29.58) -- **PSNR < bicubic already at epoch 2**.\n"
    "- **Caveat discovered**: SRVGG body has an inherent sharpness bias that persists even with warm-start + no-adv. lap_var >3700 at epoch 1 (no adversarial involved) means the architecture itself overshoots. The full 40-epoch run is needed to confirm whether PSNR recovers to >=29.0 or stays below bicubic (H6 oversharpening halt)."
).replace("\n", NL)

new_content = content[:s_idx] + new_block + content[end_idx:]
with open(PATH, "w", encoding="utf-8", newline="") as f:
    f.write(new_content)
print(f"OK: {len(content)} -> {len(new_content)} (delta {len(new_content) - len(content)})")
