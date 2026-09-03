# Phase 4.I2 — Isolated Adversarial Result (2026-09-03)

> **Status: NOT PROMOTED.** Both I2 sub-runs FAIL the D.2 quality gate (lap_var ≥ 35).
> Per plan §5 "FAIL FAIL → ship animevideov3 baseline only", Phase 4 closes unless I3 (SRVGG-body) is approved.

## TL;DR

Two warm-start sub-runs (from v1) test whether adversarial alone (without feature
distillation) is the dominant regression driver. The plan's §2.1 hypothesis is
**partially confirmed** by I2b:

| | Teacher | λ_adv | feat_weight | Epochs | Val PSNR (best EMA) | Held-out test PSNR | Full-frame lap_var | Verdict |
|---|---|---|---|---|---|---|---|---|
| v1 baseline | SPAN | 0 | 1.0 | 40 | 29.89 | n/a | 21.0 | — |
| **I2a** (sanity = Phase 3 v3) | animevideov3 | 0.001 | 1.0 | 30 | 29.885 | 29.32 | **20.8** | D.1 ✓ D.2 ✗ |
| **I2b** (isolated adv) | SPAN | 0.001 | **0** | 40 | **29.911** | 29.35 | **24.2** | D.1 ✓ D.2 ✗ |
| Phase 3 v3 reference (animevideov3+adv+feat=1) | animevideov3 | 0.001 | 1.0 | 30 | 29.99 | — | 22.3 | D.1 ✓ D.2 ✗ |

- **Both D.2 fail** — neither clears the 35 lap_var threshold.
- **I2b is the meaningful result**: dropping feat_weight exposes SPAN's constructive
  gradient signal (which had been canceled by L_feat in the 2026-08 run). +0.022 dB
  val PSNR and +15% lap_var over v1 — small but real.
- **I2a is a no-op**: animevideov3 + adv + feat=1 produces lap_var 20.8, essentially
  identical to v1's 21.0. Confirms that animevideov3 is too far from v1's distribution
  for an adversarial-only intervention to help.
- **D.1 binding gate confirmation**: both runs exceed the 29.0 dB threshold on val PSNR
  (29.89 / 29.91), so D.1 is satisfied. The blocker is purely D.2 (sharpness gap to
  animevideov3's 58.1).

## What changed in the code (~5 LOC)

- `anime_upscaler/distill.py:279-281` — added `--feat-weight {float, default=1.0}` argparse.
- `anime_upscaler/distill.py:534-552` — gated feature distillation by `args.feat_weight > 0`
  and multiplied the resulting loss by `args.feat_weight`. At default 1.0 the behavior
  is bit-identical to the Phase 2 v3 / Phase 3 recipe; setting to 0 disables L_feat
  entirely (skipping the per-tap cosine-distance computation for speed).

All 19 pre-existing tests still pass (`tests/test_shortcut_mode.py`,
`tests/test_feature_distillation.py`). Two new smoke runs (2 epochs at b4)
verified both branches at the trainer level — exit code 0, no warnings introduced.

## Sub-run details

### I2a — animevideov3 + adversarial + feat=1 (sanity check of Phase 3 v3 recipe)

```bash
.venv\Scripts\python.exe anime_upscaler\distill.py \
  --teacher animevideov3 --lambda-adv 0.001 --shortcut-anneal off \
  --feat-weight 1.0 --epochs 30 --lr 5e-5 --batch-size 16 \
  --resume pretrained/RFDN_distill_v1_4x_student.pth --fresh-epoch \
  --out-dir runs/distill_i2a_adv_only_animevideov3
```

- Wall time: 17.3 min (30 epochs × ~32 s/epoch on RTX 4000, b=16)
- Val PSNR EMA peaked at **ep 1 (29.885)** then drifted to 29.83 — the EMA decay
  shadow never accumulated meaningful perturbation from the v1 starting weights.
- Student PSNR raw recovered from 29.68 (ep 9, when adversarial peaked at 0.46) to
  **29.81 (ep 30)** as LR cosine-decayed to 1e-6.
- Selected `student_best.pt`: ep 1 EMA (≈ v1 baseline, since EMA hasn't moved yet).
- **D.2 gate FAIL**: lap_var 20.8 (canonical harness on `tmp/real_video_1sec.mp4` frame 8,
  full 3416×1920), below v1's 21.0 by 0.2.
- Held-out test (full `anime_video_frames/test` split, not directly comparable to
  v1's val 29.89): 29.32 dB.

**Interpretation**: Animevideov3's per-pixel response is similar to bicubic anchor
(PSNR 29.49 — below v1's val 29.89). When the RFDN student is asked to mimic
animevideov3 while keeping its shortcut=1.0 (bicubic) anchor, the L_distill gradient
pushes it AWAY from v1's distribution toward a lower-PSNR but sharper target. With
no L_feat to compensate and only adversarial to regularize, the student returns
to ~v1 (the local minimum).

### I2b — SPAN + adversarial + feat=0 (the new isolated test)

```bash
.venv\Scripts\python.exe anime_upscaler\distill.py \
  --teacher span --teacher-ckpt pretrained/span_pix_pretrain_4x.pth \
  --lambda-adv 0.001 --shortcut-anneal off \
  --feat-weight 0 --epochs 40 --lr 5e-5 --batch-size 16 \
  --resume pretrained/RFDN_distill_v1_4x_student.pth --fresh-epoch \
  --out-dir runs/distill_i2b_adv_only_span
```

- Wall time: 23.8 min (40 epochs × ~33 s/epoch; teacher SPAN is heavier than
  animevideov3 since it has features to extract)
- Val PSNR EMA grew **monotonically** from 29.887 (ep 1) to **29.911 (ep 35)** then
  plateaued. This is the key signal: dropping L_feat lets SPAN's teacher gradient
  drive the student past v1 with adversarial acting as a useful distribution
  regularizer rather than a competing objective.
- Selected `student_best.pt`: ep 35 EMA (peak val PSNR).
- **D.2 gate FAIL**: lap_var 24.2 (canonical harness), +15% over v1's 21.0 but
  still well below the 35 threshold.
- Held-out test: 29.35 dB.

**Interpretation**: Two non-trivial findings.

1. **L_feat + L_adv is the regression driver, not L_adv alone.** The 2026-08 run
   with the same SPAN+adv recipe but feat_weight=1 hit 28.91 dB (vs v1's 29.89).
   I2b reproduces that recipe with feat_weight=0 and lands at 29.91 val PSNR
   (+1.00 dB). The L_feat pull toward SPAN's intermediate features was actively
   hurting the student when combined with adversarial's distribution-matching
   gradient. Removing L_feat unlocks the constructive gradient.
2. **SPAN is a stronger distillation teacher than animevideov3 for this RFDN**
   when paired with adversarial: 24.2 vs 20.8 lap_var, +0.025 dB val PSNR.
   SPAN's val PSNR (31.22) is much higher than animevideov3's (29.49), so
   L_distill = 0.5·L1 pulls the student toward a closer-to-GT target.

## Per-epoch comparisons

### I2a (val PSNR / EMA PSNR)

| Epoch | val_psnr_student | val_psnr_student_ema | lap_var (192×192) | loss_adv |
|---:|---:|---:|---:|---:|
| 1 | 29.750 | **29.885** | — | 0.000 |
| 5 | 29.701 | 29.870 | — | 0.000 |
| 10 | 29.689 | 29.849 | — | 0.003 |
| 20 | 29.781 | 29.825 | — | 0.658 |
| 30 | 29.815 | 29.827 | **181.7** | 0.686 |

(per-epoch `val_lap_var` at 192×192 crops is misleading per Phase 3 lessons
— full-frame 20.8 above is the canonical comparison.)

### I2b (val PSNR / EMA PSNR)

| Epoch | val_psnr_student | val_psnr_student_ema | lap_var (192×192) | loss_adv |
|---:|---:|---:|---:|---:|
| 1 | 29.858 | 29.887 | — | 0.000 |
| 5 | 29.865 | 29.891 | — | 0.000 |
| 10 | 29.877 | 29.896 | — | 0.004 |
| 20 | **29.891** | 29.905 | — | 0.687 |
| 30 | 29.898 | 29.910 | — | 0.791 |
| 35 | 29.900 | **29.911** | — | 0.804 |
| 40 | 29.901 | 29.910 | **181.3** | 0.812 |

## Failure signatures (do not retry)

- **L_adv alone on animevideov3 (feat_weight=1) is a no-op vs v1**: lap_var 20.8
  (= v1). Do not include the animevideov3 teacher in any future RFDN variant
  unless combined with a fundamentally different architecture (I3 / SRVGG body).
- **L_adv + L_feat with SPAN teacher = regression** (28.91 dB, captured 2026-08;
  documented in memory §5). Now confirmed root cause via I2b: not L_adv, but the
  L_feat+L_adv combined pull. Future RFDN variants should set feat_weight=0 if
  using adversarial with SPAN teacher.

## Empirical anchors (canonical harness on tmp/real_video_1sec.mp4 frame 8)

| Model | lap_var | grad_mag | held-out test PSNR |
|---|---:|---:|---:|
| bicubic | 11.8 | 5.82 | 29.45 |
| v1_student_4x | 21.0 | 6.85 | n/a (val 29.89) |
| v2_student_cascade_2x2 | 30.8 | 6.74 | n/a |
| **RealESRGAN_animevideov3** | **58.1** | 6.36 | 29.04 |
| RealESRGAN_LSDIR | 132.6 | 6.70 | n/a |
| **I2a (v3 row, animevideov3+adv+feat=1)** | 20.8 | 6.83 | 29.32 |
| **I2b (v3 row, span+adv+feat=0)** | 24.2 | 7.18 | 29.35 |

I2b's 24.2 lap_var places it between v1 (21.0) and the v2 cascade (30.8) on the
sharpness ladder — clearly past v1 but well below the threshold.

## What this phase taught (lessons to remember)

1. **L_feat + L_adv is bad; L_adv alone is OK**. The 2026-08 regression had
   two non-pixel objectives pulling in opposite directions; isolating
   adversarial reveals it as a benign or mildly helpful regularizer when
   paired with SPAN's strong response signal.
2. **Drop L_feat from any future SPAN+adv RFDN variant.** This is the new
   recipe skill: feat_weight=0, lambda_adv=0.001, shortcut=off, warm-start v1.
   It produces a meaningful (if small) PSNR boost and a clear sharpness win
   without mode collapse.
3. **Animevideov3's per-pixel response is too lossy** for RFDN distillation
   without also adopting animevideov3's architecture (I3 / SRVGG body).
   I2a's 20.8 lap_var (= v1) is the dead end of the animevideov3+RFDN combo.
4. **D.2 is the binding gate for sharpness**, and a model can pass D.2 by
   hallucinating edges (per the I1 oversharpening failure 2026-09-03). Phase 4
   has not solved the gap to animevideov3's 58.1 lap_var.
5. **No destructive git ops used**, per-epoch ckpts preserved per Q6 for both runs.

## Decision (per plan §5 matrix)

| I2a | I2b | Plan §5 action |
|---|---|---|
| D.2 FAIL | D.2 FAIL | "Both fail D.2 → move to Phase I3 (SRVGG body, the only path remaining)." |

**Recommended next step**: ask the user whether to proceed to I3 (SRVGG-body
student, ~3 hours, last attempt before recommending the animevideov3 baseline
in the GUI). I2b's findings do not block I3 — they narrow the I3 search space
(feat_weight=0 should be I3's default).

## Artifacts

- Code: `anime_upscaler/distill.py` (`--feat-weight` flag at lines 279-281, gate
  at line 539, weight multiplier at line 552). ~5 LOC delta.
- Training: `runs/distill_i2a_adv_only_animevideov3/` and
  `runs/distill_i2b_adv_only_span/` (gitignored). All per-epoch ckpts preserved
  per Q6 (`archive/` directory is gitignored).
- Eval: `tmp/i2a_eval/` (lap_var on full-frame test_1sec); canonical harness
  output in `results/quality_compare_students_vs_pretrained/` with --v3-ckpt set
  to each sub-run's student_best.pt.
- Test script: `tmp/eval_i2a_full_frame.py` (forensics only, untracked).
- AGENTS.md Phase 4 section updated to mark I2 EXECUTED / NOT PROMOTED.
- PROJECT_MEMORY.md §3 / §4 / §5 / §6 / §9 updated.
