# Phase 3 v3 Student Result (2026-09-02)

> **Status: NOT PROMOTED.** v3 does not meet the D.2 quality gate.
> Shipping the RealESRGAN-animevideov3 baseline in the GUI (Phase E) instead.

## TL;DR

- Trained a 30-epoch Phase 3 v3 student (warm-start from v1, animevideov3 teacher,
  adversarial + edge loss, no shortcut anneal).
- Training converged: PSNR 29.89 -> 29.99 dB, no NaN, no collapse.
- **D.2 gate FAIL**: lap_var on full-frame 3416x1920 = **22.3** (threshold 35).
  Visual review: v3 hair/eye crops are nearly indistinguishable from v1.
  Animevideov3 baseline lap_var=58.1, with clearly crisper hair strands.
- **Decision**: do not promote v3; ship animevideov3 baseline as the user-facing
  high-quality option (Phase E). v3 ckpt preserved in archive for ablation.

## What worked

- Per-epoch checkpoint rotation works as designed (Q6 requirement met).
- Adversarial loss warm-up (ep 1-5 lambda_adv=0) prevented D-collapse on
  warm-start -- without this, the very first attempt at C.1 mode-collapsed
  immediately.
- `--fresh-epoch` flag (added during this session) lets warm-start ignore
  the v1 ckpt's epoch counter.
- Phase 3 recipe (L_pix=0.5, L_perc=1.0, L_distill=0.5, L_grad=0.1, L_adv ramp)
  trains cleanly with adversarial on at LR=5e-5.

## What did not work

- **Bicubic shortcut anneal on warm-start causes mode collapse.** Two attempts:
  - `--shortcut-anneal 1to0` (5 epochs, handoff spec): PSNR 29.92 -> 5.41 over
    5 epochs. Halted at C.3.
  - `--shortcut-anneal 1to0slow` (15 epochs, new mode I added): PSNR 29.93 ->
    7.48 over 14 epochs. Halted at C.3.
  - Root cause: the v1 student was trained with shortcut_weight=1.0 throughout,
    producing a delta-from-bicubic. The Phase 3 anneal forces the student to
    also learn the bicubic component itself, which requires scaling the delta
    up by 4x in 5 epochs. With LR=5e-5 the student cannot keep up.

- **Final config that worked (`--shortcut-anneal off`)**: PSNR stable at ~29.99
  dB but lap_var barely above v1 (22.3 vs 21.0). The adversarial loss alone,
  with shortcut_weight frozen at 1.0, cannot push the student past v1's local
  minimum. The student's residual structure is fundamentally limited.

- **Training lap_var was misleading.** I logged val_lap_var=187 in the per-epoch
  metrics.json, computed on small 192x192 val crops. The full-frame harness
  lap_var=22.3 is the ground truth. Lesson: report lap_var at the resolution
  the user actually sees, not at training crop size.

## Final metrics

| Model             | PSNR (dB) | lap_var (full) | grad_mag | dt_ms (4x) |
|-------------------|----------:|---------------:|---------:|-----------:|
| bicubic           | 29.45     | 11.8           | 5.82     | 22         |
| v1_student_4x     | 29.89*    | 21.0           | 6.85     | 602        |
| v2 cascade 2x2    | --        | 30.8           | 6.74     | 602        |
| **RealESRGAN_v3** | 29.04     | **58.1**       | 6.36     | 269        |
| RealESRGAN_LSDIR  | --        | 132.6          | 6.70     | 263        |
| v3_student_4x     | **29.99** | 22.3           | 7.10     | 235        |

*v1 PSNR is val_psnr from the v1 ckpt (29.89); full-frame test may differ slightly.

Decision matrix:
- v3 vs v1: marginal improvement (lap_var +1.3, grad_mag +3.6%). Not meaningful.
- v3 vs animevideov3: animevideov3 is 2.6x sharper AND 235ms (fastest 4x model
  in the comparison). Animevideov3 is the clear winner.
- v3 advantage: marginally faster than v1 (235ms vs 602ms) and same PSNR.
  But animevideov3 (269ms) is in the same range, so this is not decisive.

## Artifacts

- Code: anime_upscaler/adv_losses/, anime_upscaler/{teacher.py,student.py,distill.py},
  scripts/compare_students_vs_pretrained.py, scripts/train_v3_smoke.py
  (commit `21151e7`).
- Training: runs/distill_v3_4x_v3/ (gitignored). Per-epoch ckpts in archive/.
- Eval: results/quality_compare_v3_2026_08/ (gitignored).
- v3 ckpt (un-promoted): runs/distill_v3_4x_v3_epoch18_ema_unpromoted.pth
  (kept for ablation; per Q6 rule "do not delete until user accepts").

## Lessons learned (for the AGENTS.md update)

1. **Bicubic shortcut anneal is hostile to warm-start.** v1-style students trained
   with shortcut=1.0 cannot be retrained with shortcut<1.0 without mode collapse.
   Either: (a) train from scratch with the anneal, or (b) keep shortcut=1.0 and
   rely on adversarial + edge loss alone.
2. **`val_lap_var` is misleading at training crop size.** Always evaluate at the
   resolution the user sees (full 4x frame for video, native res for image).
3. **Per-epoch checkpoint rotation works** but the alphabetical-sort bug shipped
   in `21151e7`; the fixed rotation is in the same commit's follow-up. The 30
   epoch run was archived incorrectly (only ep 8-9 on main disk). For long runs
   the rotation may want a higher cap than 5.
4. **Animevideov3 IS the v1 replacement** for shipping quality. The student is
   fundamentally limited by its residual structure; the SRVGG architecture learns
   detail more naturally.

## Next step

Phase E: ship RealESRGAN-animevideov3 in the GUI model picker so users can
choose it today. v3 student effort is parked until a non-residual architecture
is available (e.g., the "Variant E: SRVGG body" option in the plan).
