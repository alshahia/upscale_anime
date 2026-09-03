# Phase 4 I1 — Nearest-Residual RFDN — Result

**Branch:** `feature/phase-1-realtime-4k`
**Date:** 2026-09-03
**Plan:** `docs/plans/student_phase4_nearest_adv_srvgg_plan.md` §1
**Run dir:** `runs/distill_i1_nearest_residual/` (preserved)
**Result status:** **NOT PROMOTED** — D.2 PASS but D.1 (PSNR) FAIL

---

## TL;DR

The "switch residual to nearest" hypothesis is **half-confirmed**:

| Hypothesis component | Verdict | Evidence |
|---|---|---|
| Nearest residual → sharper output (lap_var ↑) | **CONFIRMED** | lap_var 109.4 vs v1's 21.0 (**5.2× sharper**) |
| Nearest residual → better or comparable PSNR | **REFUTED** | PSNR 27.97 vs v1's 29.89 (**-1.92 dB regression**) |
| Nearest residual → student matches/exceeds bicubic | **REFUTED** | Student 27.97 < bicubic 29.45 (over-sharpened output) |

I1 produced the **sharpest student output we've ever trained** (5× v1's lap_var) but at the cost of pixel accuracy. The student overshoots into halos / texture hallucination — a classic adversarial-on-from-scratch failure mode.

---

## 1. What was run

```bash
.venv\Scripts\python.exe anime_upscaler\distill.py \
  --teacher animevideov3 --epochs 40 --batch-size 16 --lr 5e-5 \
  --lambda-adv 0.001 --shortcut-anneal off --shortcut-mode nearest \
  --out-dir runs/distill_i1_nearest_residual
```

- From-scratch (no `--resume`).
- Animevideov3 teacher (621K SRVGG, frozen).
- Phase 3 recipe: response distillation (0.5·L1) + pixel GT (0.5·L1 + LPIPS) + edge loss (0.1·EdgeLoss) + adversarial (peak 0.001, ramp 0→peak over 20 ep).
- Only change vs Phase 3 v3 was `shortcut_mode='nearest'` (one arg + 1 LOC in `student.py`).
- 40 epochs, ~22 min wall time on Quadro RTX 4000 8 GB.
- No mode collapse. Final EMA PSNR 27.34 dB (>25 dB halt threshold).

### 1.1 Code changes (committed)

| File | LOC | Purpose |
|---|---|---|
| `anime_upscaler/student.py` | +24 | Add `shortcut_mode` arg, validation, `set_shortcut_mode()`, conditional `align_corners` |
| `anime_upscaler/distill.py` | +5 | Add `--shortcut-mode` argparse, plumb through build + recipe log |
| `tests/test_shortcut_mode.py` | +49 (new) | 4 smoke tests (default, nearest, runtime flip, invalid) |
| **Total** | **+78 LOC, ~30 net** | Backward-compatible (default `shortcut_mode="bicubic"`) |

---

## 2. Empirical results

### 2.1 Distill-internal metrics (val/test, 192×192 crops, anime_video_frames)

| Metric | Final (ep 40 raw) | Final (ep 40 EMA) | Plan expectation |
|---|---|---|---|
| Val PSNR (student) | 28.45 dB | 27.34 dB | 29.5–30.0 |
| Val PSNR (bicubic) | 30.05 dB | — | — |
| Test PSNR (student) | **27.97 dB** | — | ≥ 29.0 (criterion) |
| Test PSNR (bicubic) | 29.45 dB | — | — |
| Test PSNR (animevideov3 teacher) | 29.04 dB | — | — |
| Test SSIM | 0.8335 | — | 0.85–0.88 |
| Test student retention of teacher gain | **nan%** | — | non-negative (criterion) |

### 2.2 Full-frame metrics on `tmp/real_video_1sec.mp4` frame 8 (canonical harness frame)

Computed with `tmp/eval_i1_correct_shortcut.py` using the correct `shortcut_mode='nearest'` (the canonical `compare_students_vs_pretrained.py` would have given wrong numbers because the vendored `apps/.../archs.py` hardcodes bicubic).

| Model | lap_var (full frame 3416×1920) | vs memory anchor |
|---|---|---|
| bicubic | 11.8 | matches (11.8) ✓ |
| **I1 (nearest shortcut, correct)** | **109.4** | vs v1: 21.0 (**5.2×**) |
| I1 (bicubic shortcut, eval-time risk) | 127.5 | would invalidate any comparison done via vendored `archs.py` |
| v1 RFDN (memory) | 21.0 | — |
| RealESRGAN AnimeVideo v3 (memory) | 58.1 | I1 is 1.9× animevideov3 |

### 2.3 Inference latency (Quadro RTX 4000, 854×480 LR → 3416×1920 SR, n=30 with 20 warmup)

| Model | ms/frame | vs 200 ms budget |
|---|---|---|
| I1 nearest (correct shortcut) | 86.5 | PASS |
| I1 bicubic (mismatch) | 93.4 | PASS |
| v1 (memory) | 104.4 | PASS |

Slight improvement over v1 — `torch 2.12` vs the older runtime used for the v1 measurement.

---

## 3. Success / failure criteria (per plan §1.5 / §1.6)

| Criterion | Target | Result | Status |
|---|---|---|---|
| **D.2 gate (lap_var ≥ 35)** | PASS | 109.4 | **PASS** |
| **D.1 gate (PSNR ≥ 29.0)** | PASS | 27.97 | **FAIL** |
| PSNR non-regression (≥ v1 29.89) | PASS | 27.97 | **FAIL** |
| No mode collapse (final EMA > 25 dB) | PASS | 27.34 | PASS |
| Inference latency ≤ 200 ms/frame | PASS | 86.5 | PASS |
| Wall time ~17 min | ~17 min | ~22 min | OK (close) |

**Net decision per plan §1.6: document, do NOT retry.**

The plan's failure criterion "Final lap_var < 25" did NOT fire — the residual-type hypothesis is correct (lap_var 5× v1). But the negative PSNR outcome is a different failure mode the plan did not enumerate.

---

## 4. Root-cause analysis

The PSNR regression has three plausible contributors. They are not fully isolated — Phase 4.I2 is the planned ablation to disambiguate.

### 4.1 Oversharpening (primary suspect)

The adversarial loss + nearest-residual shortcut create a feedback loop:
- Nearest shortcut provides a sharp, pixel-replicate baseline in the SR output.
- Discriminator rewards "looks like anime", which prefers high local variance.
- The body learns to push residuals that amplify high-frequency content beyond what HR actually contains.

Result: high lap_var (good for sharpness), low PSNR (output diverges from HR).

This is the same failure mode Phase 3 observed (memory §5: "adversarial alone → marginal only"), just compounded by the sharper anchor.

### 4.2 From-scratch + new-residual (secondary suspect)

The plan's risk note (§1.7) flagged this. We have **no warm-start comparison** to confirm or refute it. A warm-start v1 + nearest-only ablation would isolate this — but per plan §1.6 we are not retrying I1.

### 4.3 Shortcut mismatch in vendored archs.py (operational risk)

`apps/anime_upscaler_gui/anime_upscaler_gui/archs.py` line 591 has a hardcoded `mode="bicubic"` for inference. **Any** future I1-style ckpt loaded through `archs.build(kind="rfdn_student", ...)` would be evaluated with the wrong shortcut.

This is a deployment-time correctness issue, not the cause of the training-time PSNR regression. Documented here for Phase 4.I3 implementation: the SRVGG-body student (or any future nearest-shortcut variant) must update the vendored RFDN to honor the shortcut_mode.

---

## 5. Lessons for Phase 4.I2 and I3

1. **Adversarial + from-scratch + new-shortcut = oversharpening**. Any future I2/I3 variant should NOT combine all three. Pick one new thing per run.
2. **D.1 (PSNR ≥ 29.0) is the binding constraint** for shipping, not D.2 (lap_var). A model can trivially maximize lap_var while tanking PSNR — sharpness ≠ quality.
3. **Vendored `archs.py` needs shortcut_mode parity** before any new student variant can ship to the GUI. Either port `shortcut_mode` into `apps/.../archs.py::RFDN`, or add a new `kind="rfdn_student_nearest"`.
4. **Warm-start v1 + nearest-residual** is the next experiment to isolate (4.2). Phase 4.I2 covers adversarial-isolation; the warm-start-nearest ablation belongs in Phase 5 (post-I1/I2/I3) or as a Phase 4.I2 variant.
5. **Per-epoch ckpts preserved per Q6 rule** — `runs/distill_i1_nearest_residual/` retained with all 40 epochs in `archive/` for future forensics.

---

## 6. Artifacts

| Path | Purpose |
|---|---|
| `runs/distill_i1_nearest_residual/student_best.pt` | Best raw-state ckpt (PSNR 28.45 val) — preserved un-promoted |
| `runs/distill_i1_nearest_residual/student_best_raw.pt` | Same model, alternate filename |
| `runs/distill_i1_nearest_residual/student_best_ema.pt` | Best EMA ckpt — preserved |
| `runs/distill_i1_nearest_residual/student_last.pt` | Final epoch (40) raw state |
| `runs/distill_i1_nearest_residual/train_log.csv` | Per-epoch metrics (40 rows) |
| `runs/distill_i1_nearest_residual/epoch_*.json` + `epoch_*.pt` | Per-epoch state + metrics (last 5 un-archived; older in `archive/`) |
| `runs/distill_i1_nearest_residual/results_table.md` + `results_table.csv` | Final test-set metrics |
| `runs/distill_i1_nearest_residual.log` | Full stdout/stderr log (Tee-Object) |
| `tmp/i1_correct_eval/` | Full-frame eval outputs (bicubic, i1_nearest, i1_bicubic_mismatch) + metrics.csv |
| `tmp/eval_i1_correct_shortcut.py` | One-off full-frame eval with correct shortcut mode |
| `tmp/eval_i1_latency.py` | One-off latency measurement (warmup + sync) |

---

## 7. Decision

**Phase 4.I1: NOT PROMOTED.** Per plan §1.6, do not retry with different hyperparameters.

**Next action** (per user's earlier "Option I1 then Option I2 then Option I3" plan): proceed to Phase 4.I2 only after user reconfirms given this negative I1 result. The I2 hypothesis (adversarial-isolated ablation) is **more valuable now** than originally planned — it will isolate whether adversarial is the dominant regression driver (most likely per §4.1).

**Status of Phase 4 plan:**
- I1: COMPLETE, NOT PROMOTED.
- I2: pending user re-approval (mixed I1 result changes the cost/benefit).
- I3: pending.

