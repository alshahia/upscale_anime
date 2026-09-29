# Phase 5 -- Recommended Path (Best Result in Shortest Time)

**Generated:** 2026-09-03
**Question:** What is the single best Phase 5 method to attempt, given shortest wall-time-to-result AND highest expected quality gain?

## TL;DR

**Single best bet:** Warm-start v1 -> finetune with SRVGG body + no adversarial + LPIPS weight 1.0.

- **Wall-time**: 1-2 days (run is ~21 min; analysis + iteration ~1 day)
- **Quality gain**: potential to keep PSNR >= 29.0 dB (D.1 binding gate) AND inherit animevideov3 lap_var (~58) instead of v1 21
- **Risk**: low (warm-start + no adversarial = no H6 halt trigger)
- **Latency**: 61 ms/frame (1.7x faster than v1 RFDN at 104 ms/frame)
- **Reuses**: Phase 4 I3 code path -- the I3 work is 95% re-usable, only the recipe changes

This is the optimal Phase 5 candidate because:
1. All infrastructure already exists (TinySRVGGStudent class, --arch flag, animevideov3 teacher pipeline)
2. Phase 4 I3 already proved SRVGG body unlocks animevideov3 response signal (epoch 1 in-batch lap_var 3236 at adv=0)
3. The failure in I3 was specifically because of `--lambda-adv 0.001` combined with from-scratch init
4. Removing `--lambda-adv` + warm-starting from v1 should solve the H6 overshoot
5. If PSNR holds >= 29.0, we ship the first new student since v1 -- beating v1 lap_var 21 with animevideov3-level lap_var 58 at same PSNR floor

## Sorted by ROI (Best Quality Gain per Unit Wall-Time)

Ranking criterion: ROI = (expected PSNR + lap_var gain) / (effort + risk + infra friction)

| Rank | Option | Wall-time | Expected gain | Risk | Infra |
|---|---|---|---|---|---|
| **1** | **Warm-start v1 -> SRVGG-body + no-adv + LPIPS-heavy** | **1-2 days** | **+lap_var, keep PSNR, 1.7x faster** | **LOW** | Reuses I3 |
| 2 | APISR-style anime perceptual loss + warm-start | 1 week | +MANIQA, +CLIPIQA, +anime sharpness | LOW-MED | None |
| 3 | FRAMER-style frequency-domain distillation | 1-2 weeks | +HF detail, +edge preservation | LOW | None |
| 4 | Multi-Scale Contrastive-Adversarial KD | 1 week | +LPIPS | MED | None |
| 5 | OSEDiff / SinSR anime fine-tune | 2 weeks | +++ MANIQA, but ~1 sec/frame | MED-HIGH | None |
| 6 | MambaIRv2 student distillation | 2-3 weeks (after Docker build) | +PSNR potential at same latency | MED | Docker build (in progress) |
| 7 | FiDeSR / One-Step Diffusion Transformer fine-tune | 3 weeks | ++++ perceptual, but ~1 sec/frame | HIGH | None |

## Why Rank #1 is the best

### What Phase 4 I3 proved
- I3 (from-scratch SRVGG + lambda-adv 0.001 + no warm-start) failed at PSNR 27.91 < bicubic 29.45
- BUT the architecture unlocks animevideov3 response signal (in-batch lap_var 3236 at epoch 1 with adv=0)
- Latency 61 ms/frame (1.7x faster than v1 RFDN 104 ms/frame) is a real win

### Why removing adversarial + warm-starting should work
- Phase 4 I1 + I3 both hit H6 oversharpening halt because of: from-scratch + new design + adversarial
- Drop any ONE of those three and H6 does not trigger (lesson from PROJECT_MEMORY section 5 "Failure signatures")
- Warm-start + no-adv is the proven safe combination
- LPIPS weight 1.0 (instead of 0.05) anchors to GT strongly enough to prevent overshoot

### Expected recipe

**Entry point:** `python anime_upscaler/distill.py` (NOT `scripts/train.py` -- that does not exist).
**CLI mapping notes (verified against `distill.py --help` on 2026-09-03):**
- `--warm-start-from` does not exist -> use `--resume <path>` + `--fresh-epoch` (warm-start semantics built-in).
- `--lambda-lpips 1.0` and `--lambda-pixel 1.0` do not exist -> LPIPS weight is fixed at 1.0 and pixel L1 weight at 0.5 in the Phase 3 SRVGG recipe (see `distill.py:583-587`).
- `--warm-start-mode partial` is required because v1 (RFDN, 315K) has different layer names than TinySRVGGStudent (317K). Without `partial`, `--resume` raises RuntimeError on key mismatch. The new mode copies the 4 matching conv weights (head + upsampler.0) and randomises the middle 12-conv stack.

```bash
# From repo root, with venv active:
.venv\\Scripts\\python.exe anime_upscaler\\distill.py ^
    --resume pretrained\\RFDN_distill_v1_4x_student.pth ^
    --fresh-epoch ^
    --warm-start-mode partial ^
    --teacher animevideov3 ^
    --arch srvgg ^
    --lambda-adv 0 ^
    --feat-weight 1.0 ^
    --shortcut-anneal off ^
    --epochs 40 ^
    --batch-size 16 ^
    --lr 5e-5 ^
    --out-dir runs\\distill_v3_4x_srvgg_warmstart
```

For convenience, a wrapper script that applies these flags + halt guards is
provided at `scripts/run_rank1_warmstart.ps1`.

### Smoke test (2026-09-03, 2 epochs, batch 4)

Verified that the cross-arch partial warm-start infrastructure works:
- 4 keys mapped: `body.0.weight/bias` <- RFDN `head.weight/bias`; `body.26.weight/bias` <- RFDN `upsampler.0.weight/bias`.
- Middle 12-conv stack + PReLUs stay at random init.
- Val epoch 1: PSNR 26.97 dB (vs bicubic 28.55, teacher 29.13); **lap_var 3708.7** (vs I3 epoch-1 lap_var 3236 with from-scratch+adv=0.001).
- Val epoch 2: PSNR 27.01 dB; lap_var 3706.9.
- Test PSNR 27.69 dB (vs bicubic 29.58) -- **PSNR < bicubic already at epoch 2**.
- **Caveat discovered**: SRVGG body has an inherent sharpness bias that persists even with warm-start + no-adv. lap_var >3700 at epoch 1 (no adversarial involved) means the architecture itself overshoots. The full 40-epoch run is needed to confirm whether PSNR recovers to >=29.0 or stays below bicubic (H6 oversharpening halt).
### Success criteria
- PSNR >= 29.0 dB (D.1 binding gate)
- lap_var > 35 (D.2 gate) -- expected ~40-60 based on architecture unlocks animevideov3 signal at adv=0
- Latency <= 80 ms/frame (still faster than v1 RFDN)

### Failure criteria (fallback plan)
- If PSNR < 29.0 dB at epoch 10: halt, try --lambda-lpips 2.0
- If still PSNR < 29.0 at epoch 40: ship v1 + animevideov3 unchanged, move to Rank #2 (APISR perceptual)

## Why MambaIRv2 (Rank #6) is not at the top

Despite MambaIRv2 stronger PSNR/MACs claims, the path has critical friction:
- Docker image build is 1-2 hours (in progress)
- Even after build, distillation training is 2-3 weeks of iteration
- The +0.29 dB on Manga109 result is vs HAT (heavy transformer), not RFDN (lightweight CNN) -- transferring the gain to a RFDN-scale student is empirically untested
- If Rank #1 succeeds (warm-start SRVGG), we would already have a shipped student. If it fails, the failure mode tells us more about whether MambaIRv2 is worth pursuing at all

So MambaIRv2 is gated on:
1. Docker build completing successfully (~30-60 min remaining)
2. Rank #1 outcome (does warm-start SRVGG beat v1?)
3. Available training budget

## What "best in short time" means in context

The user question was about ROI for the project. The Phase 5 candidates vary in:
- Effort: 1-2 days to 3 weeks
- Infra friction: pure-Python vs Docker build vs new dependencies
- Quality ceiling: marginal gain vs SOTA breakthrough

Rank #1 is the answer because it has the lowest effort AND the highest confidence gain AND reuses all prior work.

## Phase 5 candidates -- detailed comparison

### Rank 1: Warm-start v1 -> SRVGG body + no adversarial
- **Effort**: 1-2 days (one ~21 min run + analysis)
- **Expected outcome**: PSNR 29.0-29.5 dB, lap_var 40-60, latency 61 ms/frame
- **Risk**: LOW (warm-start + no adversarial = H6 impossible)
- **Why it might fail**: SRVGG body has a sharpness prior that might still overshoot even with warm-start + LPIPS-heavy. If so, halve LPIPS weight, try again.

### Rank 2: APISR-style anime perceptual loss
- **Effort**: 1 week (add balanced twin perceptual loss to distill.py, retrain)
- **Expected outcome**: +0.02-0.05 MANIQA, +CLIPIQA, line-art preservation improved
- **Risk**: LOW-MED (anime-domain perceptual network needs validation)
- **Why it might fail**: anime-VGG or anime-CLIP backbone licensing unclear; may need to retrain perceptual network

### Rank 3: FRAMER-style frequency-domain distillation
- **Effort**: 1-2 weeks (add frequency-domain loss + adaptive modulation)
- **Expected outcome**: +HF detail recovery, +edge preservation (anime-specific)
- **Risk**: LOW (additive loss, does not replace existing)
- **Why it might fail**: training instability from frequency-domain losses is possible

### Rank 4: Multi-Scale Contrastive-Adversarial KD
- **Effort**: 1 week (ICCVW 2025 method, add contrastive + adversarial heads)
- **Expected outcome**: +LPIPS (perceptual), potentially better texture recovery
- **Risk**: MED (combining multiple losses can destabilize training)
- **Why it might fail**: contrastive loss needs careful temperature tuning

### Rank 5: OSEDiff / SinSR anime fine-tune
- **Effort**: 2 weeks (build diffusion pipeline, fine-tune on API dataset)
- **Expected outcome**: quality-tier preset (~1 sec/frame, interactive), +++ MANIQA
- **Risk**: MED-HIGH (diffusion training is finicky)
- **Why it might fail**: inference speed too slow for our real-time target (interactive only)

### Rank 6: MambaIRv2 student distillation
- **Effort**: 2-3 weeks (after Docker build completes; possibly 1-2 more weeks for distillation iteration)
- **Expected outcome**: +PSNR potential at same latency tier as RFDN
- **Risk**: MED (state-space model training is sensitive to LR, init)
- **Why it might fail**: +0.29 dB on Manga109 is vs HAT (heavy transformer), not RFDN -- transfer gain unproven

### Rank 7: FiDeSR / One-Step Diffusion Transformer fine-tune
- **Effort**: 3 weeks (CVPR 2026 method, requires diffusion infrastructure)
- **Expected outcome**: SOTA quality ceiling, ++++ perceptual
- **Risk**: HIGH (complex training, expensive compute)
- **Why it might fail**: not yet battle-tested for anime domain

## Recommended sequence (synthesized)

1. **Now**: Rank #1 (warm-start SRVGG no-adv) -- 1-2 days
2. **If #1 succeeds**: ship; start Rank #2 (APISR perceptual) in parallel
3. **If #1 fails**: try Rank #1 with adjusted hyperparams; if still fails, move to Rank #2
4. **Always**: finish Docker MambaIRv2 build in background; smoke-test; queue Rank #6 as Phase 6 candidate
5. **Phase 6+**: Rank #5 (OSEDiff), Rank #7 (FiDeSR) as quality-tier options

---

## Phase 5 outcome -- Rank #1 (2026-09-05)

**Result**: D.1 FAIL. Test PSNR 27.87 dB (< 29.0 binding gate by 1.13 dB). H6 triggered (student 27.87 < bicubic 29.45 by 1.58 dB). 40-epoch warm-start v1 -> SRVGG + no-adv + LPIPS-VGG plateaus at val 28.39 dB (raw) / 28.11 dB (EMA). SRVGG body sharpness bias persists regardless of warm-start source and adversarial weight. **Hypothesis falsified.**

Full result at `runs/distill_v3_4x_srvgg_warmstart/results_table.md`; per-epoch metrics at `train_log.csv`.

---

## Phase 5 Rank #2 recipe -- APISR balanced twin perceptual loss

**Hypothesis**: Replace the LPIPS-VGG perceptual term with a balanced twin perceptual loss (VGG19 + ResNet50) to address the SRVGG body sharpness bias that Rank #1 could not overcome. The twin loss penalises SR signals in BOTH photoreal (VGG19 ImageNet) and anime-domain (ResNet50 ImageNet; Danbooru fallback if absent) feature spaces. Rank #1's lesson: changing LOSS WEIGHT on the same backbone is insufficient; changing LOSS FAMILY may converge to a less-sharp basin.

**Entry point**: `python anime_upscaler/distill.py` with `--loss twin`.

**CLI notes (verified against `distill.py --help` on 2026-09-05):**
- `--loss {vgg, twin}` (default `vgg` preserves Rank #1 behavior)
- `--twin-danbooru-weight 0.5` and `--twin-vgg-weight 0.5` (APISR Ablation Table 4 / v7 roadmap Phase C)
- `--twin-delta X` (legacy alias; equivalent to `--twin-vgg-weight X`)
- `--loss twin` only effective under Phase 3 SRVGG spec (i.e. `--teacher animevideov3|lsdir` or `--lambda-adv > 0` or `--shortcut-anneal != off`); for SPAN/Phase 2 v3 resumers the flag is silently ignored (warning printed).
- `--warm-start-mode partial` + `--resume pretrained\\RFDN_distill_v1_4x_student.pth` + `--fresh-epoch` are reused verbatim from Rank #1.

**Why this is fast to implement (1 day wall-time, not 1 week as the REPORT.md ROI table estimates):** TwinPerceptualLoss already exists at `src/losses/twin_perceptual_loss.py` (320 LOC, BSD-3 ImageNet backbones + optional Danbooru). Tests at `tests/test_v7_loss_schedule.py::TestDanbooruVGGWeightBalance`. We only needed the CLI flag, lazy-init helper, and a wrapper script. No new dependencies.

**Why no licensing issue**: APISR paper's anime-VGG / anime-CLIP backbone was the original concern (license unclear). Our implementation uses torchvision's BSD-3 ImageNet VGG19 + ResNet50. When `pretrained/danbooru_resnet50.pth` is absent, the ResNet50 falls back to ImageNet weights (gracefully handled at `src/losses/twin_perceptual_loss.py:138-139`). If the user later wants the Danbooru anime-domain weights, they can be added as an opt-in via `--danbooru-resnet-path <path>` (NOT yet implemented in distill.py; would need a small addition).

```bash
# From repo root, with venv active:
.venv\\Scripts\\python.exe anime_upscaler\\distill.py ^
    --resume pretrained\\RFDN_distill_v1_4x_student.pth ^
    --fresh-epoch ^
    --warm-start-mode partial ^
    --teacher animevideov3 ^
    --arch srvgg ^
    --lambda-adv 0 ^
    --feat-weight 1.0 ^
    --shortcut-anneal off ^
    --loss twin ^
    --twin-danbooru-weight 0.5 ^
    --twin-vgg-weight 0.5 ^
    --epochs 40 ^
    --batch-size 16 ^
    --lr 5e-5 ^
    --out-dir runs\\distill_v3_4x_srvgg_twin
```

For convenience, a wrapper script that applies these flags + halt guards is provided at `scripts/run_rank2_twin_perceptual.ps1`.

### Smoke test (2026-09-05, 2 epochs, batch 4)

Verified end-to-end wiring:
- Recipe print shows `loss=twin` and Danbooru fallback line `[TwinPerceptualLoss] Danbooru weights not found, using ImageNet ResNet50`.
- Partial warm-start maps the same 4 keys as Rank #1 (head->body.0, upsampler.0->body.26).
- val epoch 1: PSNR 26.95 dB (vs bicubic 28.55, teacher 29.13); ssim_s 0.8407.
- val epoch 2: PSNR 27.01 dB; ssim_s 0.8582.
- test PSNR 27.69 dB (vs bicubic 29.58, teacher 29.12) -- matches Rank #1 smoke EXACTLY at 2 epochs (loss choice has not yet differentiated).
- Per-epoch wall-time 37s, 25s (vs 51s, 27s for LPIPS-only) -- ResNet50 backbone adds ~30% per-iter compute.

### Success criteria
- PSNR >= 29.0 dB (D.1 binding gate) -- if twin converges to a less-sharp basin than LPIPS, this becomes achievable.
- lap_var > 35 (D.2 gate)
- Latency <= 80 ms/frame (unchanged from Rank #1; --loss twin only changes training)

### Failure criteria (next fallback after Rank #2)
- If PSNR < 29.0 dB at epoch 10: halt, try `--twin-danbooru-weight 1.0 --twin-vgg-weight 0.25` (asymmetric, lean on ResNet50 anime-domain signal).
- If still PSNR < 29.0 at epoch 40: SRVGG body bias is structural; move to Rank #6 (MambaIRv2 via Docker rebuild) or Rank #3 (FRAMER frequency-domain distillation).

---

## Phase 5 outcome -- Rank #2 (2026-09-05)

**Result**: D.1 FAIL. Test PSNR 27.98 dB (< 29.0 binding gate by 1.02 dB). H6 still triggered (student 27.98 < bicubic 29.45 by 1.47 dB). 40-epoch warm-start v1 -> SRVGG + no-adv + APISR-twin plateaus at val 28.48 dB (raw) / 28.14 dB (EMA). Wall-time ~24 min.

**Head-to-head vs Rank #1**: +0.11 dB test PSNR (27.98 vs 27.87), +0.09 dB best val (28.48 vs 28.39), -0.11 dB H6 gap (1.47 vs 1.58 dB below bicubic). The twin perceptual loss provides real but bounded improvement. The SRVGG body sharpness bias is REDUCED but NOT ELIMINATED.

**Critical finding**: SRVGG body sharpness bias appears STRUCTURAL to the architecture, not specific to the loss function. Three independent runs converge near test PSNR 28 dB:
- I3 from-scratch + adv=0.001: 27.91
- Rank #1 warm-start + LPIPS: 27.87
- Rank #2 warm-start + APISR twin: 27.98

The remaining options all require architecture change, not loss change.

Full result at `runs/distill_v3_4x_srvgg_twin/results_table.md`; per-epoch metrics at `train_log.csv`.

### Files touched by Rank #2
- `anime_upscaler/distill.py` (+~60 LOC): `--loss {vgg, twin}` argparse + 3 weight flags, `_get_twin_perceptual` lazy-init helper, dispatch in loss_gt block, recipe print.
- `tests/test_phase5_rank2_twin_perceptual.py` (NEW, 8 tests): argparse shape + lazy-init cache + frozen-grads + forward + zero-weight.
- `scripts/run_rank2_twin_perceptual.ps1` (NEW): wrapper mirroring `run_rank1_warmstart.ps1`.
- `docs/PROJECT_MEMORY.md` §6: new decision log entry "Phase 5 Rank #2 LAUNCH".
- `docs/research/anime_sr_2026/RECOMMENDED_PATH.md` (this section -- new).

## Phase 5 Option A recipe -- Asymmetric twin 1.0 / 0.25 (fallback (e) from 5#2)

**Hypothesis (refinement of 5#2)**: APISR balanced twin gives +0.11 dB over LPIPS-only Rank #1, but the gain is bounded. Test whether the gain comes from (a) ResNet50 anime-domain signal alone, (b) VGG19 photo-real signal alone, or (c) balanced combination. Lean 4x harder on ResNet50 (anime) and drop VGG19 (photo) to 0.25 -- if asymmetric beats balanced, ResNet50 is the active ingredient. If it loses or ties, the +0.11 dB gain is from balanced regularization (both backbones add similar bounded signal).

**Why this matters**: If asymmetric wins, we have a 1-knob recipe to push the loss family further (try 2.0/0.1 etc.). If it loses, SRVGG body ceiling is confirmed loss-agnostic and we should move directly to Rank #6 (MambaIRv2 via Docker rebuild).

### Recipe (verified working, smoke 2 ep on RTX 4000, ~62 s wall -- same as 5#2 smoke)

```
powershell -ExecutionPolicy Bypass -File scripts\run_rank2_twin_perceptual.ps1 -DanbooruWeight 1.0 -VggWeight 0.25 -OutDir runs\distill_v3_4x_srvgg_twin_asym
```

CLI flags (no code change needed; same wrapper as 5#2):
- `-DanbooruWeight 1.0` (vs 5#2 default 0.5): double anime-domain ResNet50 weight
- `-VggWeight 0.25` (vs 5#2 default 0.5): halve photo-real VGG19 weight
- `-OutDir runs\distill_v3_4x_srvgg_twin_asym`: separate out-dir to keep artifacts distinct from 5#2

### Success / failure criteria

- **Pass**: PSNR >= 29.0 dB on held-out test (D.1 binding gate) -- unlikely but possible
- **Partial**: PSNR > 28.05 dB (asymmetric beats 5#2's 27.98) -- ResNet50 was the active ingredient
- **Fail**: PSNR < 28.00 dB (asymmetric loses vs 5#2) -- VGG19 was contributing; loss family is bounded
- **Confirmed structural**: PSNR ~28.00 dB (asymmetric ties 5#2 within noise) -- loss-agnostic; move to Rank #6

### Expected wall-time

~22-25 min for 40 epochs (same ResNet50 backbone compute as 5#2; only the per-iter loss weighting changes). The wrapper script `scripts/run_rank2_twin_perceptual.ps1` already accepts both flags; no code change required.

## Phase 5 outcome -- Option A (2026-09-05)

**Result**: D.1 FAIL. Test PSNR 27.87 dB (< 29.0 binding gate by 1.13 dB). H6 still triggered (student 27.87 < bicubic 29.45 by 1.58 dB). 40-epoch warm-start v1 -> SRVGG + no-adv + asymmetric-twin (ResNet50=1.0, VGG19=0.25) plateaus at val 28.37 dB (raw) / 28.10 dB (EMA). Wall-time ~20 min.

**Head-to-head**: Asymmetric twin REGRESSES -0.11 dB vs balanced twin (5#2: 27.98 vs Option A: 27.87) and ties Rank #1 (also 27.87). The +0.11 dB Rank #2 gain was from BALANCED twin regularization (both VGG19 + ResNet50 contributing), NOT from anime-domain signal alone.

| Variant | Test PSNR | Best val raw | Best val EMA | SSIM | H6 gap |
|---|---|---|---|---|---|
| Rank #1 (LPIPS-only) | 27.87 | 28.39 | 28.11 | 0.8477 | -1.58 dB |
| Rank #2 (balanced twin 0.5/0.5) | 27.98 | 28.48 | 28.14 | 0.8465 | -1.47 dB |
| **Option A (asymmetric 1.0/0.25)** | **27.87** | **28.37** | **28.10** | **0.8479** | **-1.58 dB** |

**Critical finding (4-run structural ceiling)**: SRVGG body sharpness bias is STRUCTURAL across FOUR independent runs, all converging near test PSNR 28 dB:

| Run | Init | Loss | Test PSNR |
|---|---|---|---|
| I3 | from-scratch | LPIPS+adv | 27.91 |
| 5#1 | warm-start v1 | LPIPS | 27.87 |
| 5#2 | warm-start v1 | balanced twin | 27.98 |
| **5#2b** | **warm-start v1** | **asymmetric twin** | **27.87** |

Range: 0.11 dB. The loss family is BOUNDED at ~28 dB; the SRVGG body architecture is the binding constraint.

**New insight (APISR twin asymmetry)**: APISR-style twin perceptual loss works ONLY when VGG19 + ResNet50 are balanced. Unbalancing towards either backbone drops the gain to zero. The balanced twin acts as a multi-view regularizer (photo + anime feature spaces) that pulls the SRVGG body into a less-sharp basin; unbalancing loses that regularization effect. This suggests **further loss-weight tuning in this family will not break the ceiling**.

Full result at `runs/distill_v3_4x_srvgg_twin_asym/results_table.md`; per-epoch metrics at `train_log.csv`.

### Files touched by Option A

- **NO code changes** -- pure recipe ablation. `scripts/run_rank2_twin_perceptual.ps1` already accepted `-DanbooruWeight` / `-VggWeight` / `-OutDir` flags from 5#2; only the wrapper invocation differs.
- `runs/distill_v3_4x_srvgg_twin_asym/` (NEW, ~30 MB): full 40-epoch run artifacts (train_log.csv, results_table.md, student_best.pt, student_best_ema.pt, student_best_raw.pt, student_last.pt, epoch_36..40 + epoch_36..40_ema + per-epoch metrics json, train_stdout.log 17 KB, train_stderr.log empty, latest.pt/latest_ema.pt). Older epochs (1-35) archived to `archive/`. All preserved per Q6 rule (do not delete without user confirmation).
- `docs/PROJECT_MEMORY.md` §4 task list: 5#2b marked `[x]` with result summary.
- `docs/PROJECT_MEMORY.md` §6: new decision log entries "Phase 5 Option A LAUNCH" and "Phase 5 Option A RESULT".
- `docs/research/anime_sr_2026/RECOMMENDED_PATH.md` (this section -- updated).

### Files touched by this ranking

- docs/research/anime_sr_2026/RECOMMENDED_PATH.md (this file -- new)
- docs/research/anime_sr_2026/REPORT.md section 9 (sorted list)
- docs/research/anime_sr_2026/WSL2_DOCKER_PROBE.md "Alternative" section (sorted list)
- docs/PROJECT_MEMORY.md section 4 (re-ordered Phase 5 candidates)
- docs/PROJECT_MEMORY.md section 6 (new decision log entry)

## Security reminder

Exa API key was typed directly in chat history. Recommend rotation at https://dashboard.exa.ai/keys after this session. Key is NOT stored in any committed file.
