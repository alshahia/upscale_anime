# v3 distillation — research and plan

Companion folder for the v3 distillation recipe to lift the v1 RFDN student above the bicubic baseline.

## Contents

| File | Purpose |
|---|---|
| `REPORT.md` | Full research report (15 KB). Ranking of 13 techniques, top-5 picks, pitfalls, augmentation suggestions, v3 recipe sketch, sources. Web access was off — citations are paraphrased from the in-repo `docs/survey_2026/` survey. |
| `TODO.md` | Runnable phase-by-phase checklist (small). Companion to `.claude/plans/distill_v3_recipe.md`. |

## Where the canonical plan lives

The full implementation plan is at `.claude/plans/distill_v3_recipe.md`. Read that first.

## Where the resumption note lives

Cross-session memory at `.windsurf/memory/distill_v3_plan.md`.

## TL;DR

- v1 student = **29.32 dB test PSNR < bicubic 29.45** — distillation recipe is the ceiling.
- Top 5 fixes: loss rebalance, Charbonnier+SSIM+LPIPS, EMA, APISR degradation, TTA 8×.
- Combined expected gain: +0.4 to +0.9 dB → student ~29.7–30.2 dB (50–80% retention of teacher gain).
- Architecture (RFDN, 315K params) stays. Teacher (SPAN-V7) stays.