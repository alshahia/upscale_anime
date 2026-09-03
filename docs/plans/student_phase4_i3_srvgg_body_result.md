# Phase 4 I3 — SRVGG-Body Student — Result

**Branch:** `feature/phase-1-realtime-4k`
**Date:** 2026-09-03
**Plan:** `docs/plans/student_phase4_nearest_adv_srvgg_plan.md` §3
**Run dir:** `runs/distill_i3_srvgg_body/` (preserved)
**Result status:** **NOT PROMOTED** — D.2 PASS, D.1 (PSNR) FAIL, H6 oversharpening triggered.

---

## TL;DR

The "copy animevideov3's architecture" hypothesis is **half-confirmed**:

| Hypothesis component | Verdict | Evidence |
|---|---|---|
| SRVGG body → matches animevideov3's response signal | **CONFIRMED (and then some)** | full-frame lap_var **306.48** vs animevideov3's 58.1 (**5.3× sharper**) |
| SRVGG body → pixel accuracy comparable to v1 | **REFUTED** | Test PSNR **27.91** vs v1 29.89 (**−1.98 dB regression**) |
| SRVGG body → student ≥ bicubic on PSNR | **REFUTED (H6 halt)** | Student 27.91 < bicubic 29.45 (oversharpened) |
| SRVGG body → inference latency budget | **CONFIRMED** | 61.0 ms/frame (≤ 200 ms PASS); **~1.7× faster than v1** (104.4 ms) |

The architecture correctly unlocks animevideov3's high-frequency response signal, but the body is **over-anchored on sharpness** from epoch 1 — in-batch lap_var already at 3236 at epoch 1 (with adv=0). The adversarial term then slowly *reduced* sharpness (3236 → 2579 across 40 ep) but never brought pixel accuracy back to v1's range.

This is **I1's failure mode with a new architecture** instead of a new shortcut: from-scratch + new-arch + adversarial together overshoot the same way they did with from-scratch + nearest-shortcut + adversarial in I1.

---

## 1. What was run

```bash
.venv\Scripts\python.exe anime_upscaler\distill.py \
  --arch srvgg --teacher animevideov3 --epochs 40 --batch-size 16 --lr 5e-5 \
  --lambda-adv 0.001 --shortcut-anneal off --shortcut-mode nearest \
  --out-dir runs/distill_i3_srvgg_body
```

- From-scratch (no `--resume`). New arch `TinySRVGGStudent` (52 ch × 12 convs + PReLU + PixelShuffle + nearest residual; **317,300 params**, matches v1's 315K budget).
- Animevideov3 teacher (621K SRVGG, frozen). The architecture inductive bias now matches the teacher exactly (same conv stack design), so we expected the student to be able to *closely* match the teacher's response signal.
- Phase 3 recipe: response distillation (0.5·L1) + pixel GT (0.5·L1 + LPIPS) + edge loss (0.1·EdgeLoss) + adversarial (peak 0.001, ramp 0→peak over 20 ep).
- `--feat-weight 1.0` is the default but is irrelevant here — `s_feats=[]` for SRVGG by construction.
- `--shortcut-mode nearest` is also irrelevant — SRVGG always uses nearest residual.
- 40 epochs, ~21 min wall time on Quadro RTX 4000 8 GB.
- No mode collapse. Final raw PSNR 28.44 (val) → 27.91 (held-out test). Final EMA PSNR 28.10 (val).

### 1.1 Code changes (committed)

| File | LOC | Purpose |
|---|---|---|
| `anime_upscaler/student.py` | +99 | New `TinySRVGGStudent` class (matches `SRVGGNetCompact` in `teacher.py`); no return_features; no shortcut anneal |
| `anime_upscaler/distill.py` | +10 | `--arch {rfdn,srvgg}` argparse, arch-aware student build + forward dispatch (s_feats=[] for srvgg), recipe log line |
| `apps/anime_upscaler_gui/anime_upscaler_gui/archs.py` | +75 | Vendored `TinySRVGGStudent`, updated vendored RFDN to honor `shortcut_mode`, `build(kind='srvgg_student')` dispatch + `build(kind='rfdn_student')` auto-sniff shortcut_mode from ckpt args |
| `tests/test_tiny_srvgg.py` | +125 (new) | 9 smoke tests (params, shape scale4/scale2, return_features, residual presence, set_shortcut_weight no-op, set_shortcut_mode validate, vendored-vs-canonical parity, build() round-trip) |
| **Total** | **+309 LOC** | Backward-compatible (default `--arch rfdn` preserves Phase 2 v3 / Phase 3 / Phase 4 I1+I2 behavior) |

### 1.2 PREREQUISITE: vendored `archs.py` shortcut_mode parity

PROJECT_MEMORY §8 line 299 flagged `apps/.../archs.py:591` as **STALE**: the vendored RFDN hardcoded `mode="bicubic"`. **I3 fixes this** — the vendored RFDN now accepts a `shortcut_mode` arg (default bicubic for v1 back-compat), and `build(kind='rfdn_student')` auto-sniffs the mode from the ckpt's saved `args.shortcut_mode`. Any future Phase 4 / Phase 5 nearest-shortcut RFDN ckpt will now load correctly via the GUI.

---

## 2. Empirical results

### 2.1 Distill-internal metrics (val/test, 192×192 crops, anime_video_frames)

| Metric | Final (ep 40 raw) | Final (ep 40 EMA) | Plan expectation |
|---|---|---|---|
| Val PSNR (student) | **28.44 dB** | **28.10 dB** | 29.0–29.5 (optimistic) / 27.0–28.5 (pessimistic) |
| Val PSNR (bicubic) | 30.05 dB | — | — |
| **Test PSNR (student)** | **27.91 dB** | — | **≥ 28.0 (criterion)** |
| Test PSNR (bicubic) | 29.45 dB | — | — |
| Test PSNR (animevideov3 teacher) | 29.04 dB | — | — |
| Test SSIM | 0.8508 | — | — |
| Test student retention of teacher gain | nan% (negative) | — | non-negative (criterion) |

### 2.2 Full-frame metrics on `tmp/real_video_1sec.mp4` frame 8 (canonical harness frame)

| Model | lap_var (full frame 3416×1920) | vs memory anchor |
|---|---|---|
| bicubic | 11.79 | matches (11.8) ✓ |
| **I3 (TinySRVGGStudent, EMA-best)** | **306.48** | vs v1: 21.0 (**14.6×**); vs animevideov3: 58.1 (**5.3×**) |
| I3 (raw-best, control) | 196.68 | vs v1: 21.0 (9.4×); vs animevideov3: 58.1 (3.4×) |
| v1 RFDN (memory) | 21.0 | — |
| RealESRGAN AnimeVideo v3 (memory) | 58.1 | I3 is **5.3× sharper** |

### 2.3 Per-epoch training trajectory (40 epochs, distilled from `runs/distill_i3_srvgg_body/archive/` + `epoch_*_metrics.json`)

| Epoch | Val PSNR (raw) | Val PSNR (EMA) | In-batch lap_var | λ_adv |
|---|---|---|---|---|
| 1 | 26.74 | 25.72 | **3236.78** | 0.0000 |
| 10 | 28.14 | 26.68 | 3126.85 | 0.0001 |
| 20 | 28.30 | 27.45 | 2883.77 | 0.0005 |
| 30 | 28.41 | 27.87 | 2647.52 | 0.0010 |
| 40 | 28.44 | **28.10** | 2579.91 | 0.0010 |

Key observation: **PSNR monotonically improves throughout 40 epochs; never plateaus**. The model would likely continue improving with more epochs. Adversarial weight slowly REDUCED lap_var (3236 → 2580), but never enough to bring pixel accuracy back to v1's range.

### 2.4 Inference latency (Quadro RTX 4000, 854×480 LR → 3416×1920 SR, n=30 with 5 warmup)

| Model | ms/frame | vs 200 ms budget |
|---|---|---|
| **I3 SRVGG** (EMA-best) | **61.0** | PASS (**3.3× under budget**) |
| I3 SRVGG (raw-best, cold cache) | 427.8 | inconclusive (anomaly; warm cache gives 61) |
| v1 RFDN (memory) | 104.4 | PASS |
| animevideov3 SRVGG (memory) | 98.7 | PASS |

**SRVGG is the fastest architecture tested.** The reduced per-layer cost (12 plain 3×3 convs vs 6 RFDBlocks with 5 internal convs each + 1×1 attention) plus the lack of pixel-attention gates makes I3 ~1.7× faster than v1 RFDN. This is the one unambiguous win of the architecture change.

---

## 3. Success / failure criteria (per plan §3.6 / §3.7)

| Criterion | Target | Result | Status |
|---|---|---|---|
| **D.2 gate (lap_var ≥ 35)** | PASS | 306.48 | **PASS (8.7× margin)** |
| **D.1 gate (PSNR ≥ 29.0, NEW 2026-09-03 binding)** | PASS | 27.91 | **FAIL** (−1.09 dB) |
| **Plan criterion (PSNR ≥ 28.0)** | PASS | 27.91 | **FAIL** (−0.09 dB; just barely) |
| Student ≥ bicubic (H6 oversharpening halt) | PASS | 27.91 < 29.45 | **FAIL (H6 TRIGGERED)** |
| Inference latency ≤ 200 ms/frame | PASS | 61.0 | **PASS (3.3× under)** |
| Wall time ~17 min | ~17 min | ~21 min | OK (slower by 24%) |
| No mode collapse (final EMA > 25 dB) | PASS | 28.10 | PASS |

**Net decision per plan §3.7: document, do NOT retry.** All three of D.1 / plan PSNR / H6 fail simultaneously. Plan §5 matrix: `FAIL FAIL FAIL → ship animevideov3 baseline only (already shipped)`. Phase 4 closed.

---

## 4. Root-cause analysis

### 4.1 Architecture-unlocked response signal (the win)

The architecture change **worked as a channel**: the student's output distribution moved sharply toward animevideov3's. In-batch lap_var at epoch 1 (adv=0!) was already 3236, an order of magnitude above v1's typical batch-lap_var of ~350 (estimated from memory). The SRVGG body — when distilled from an SRVGG teacher — naturally produces high-frequency responses even before the adversarial term kicks in.

This is **the first time any variant of our student has reached animevideov3's sharpness regime**, let alone surpassed it. The path I2b identified ("animevideov3+RFDN is too lossy for adversarial-only intervention to help; SRVGG body needed") was correct: SRVGG body makes the response signal usable.

### 4.2 Oversharpening from epoch 1 (the loss)

The same response-signal unlock that produced 5× sharpness also produced 1.98 dB PSNR regression. Looking at the trajectory:

- **Epoch 1** (adv=0): val PSNR 26.74, in-batch lap_var **3236**. The student is already very sharp and very inaccurate — pure from-scratch + SRVGG-body bias without the GT anchor having a chance to pull it back.
- **Epochs 2-20** (adv=0 → 0.0005): PSNR climbs to 28.30 (+1.56 dB), lap_var drops to 2884 (−11%). The GT anchor (L_pix + LPIPS) is doing the heavy lifting, slightly trading sharpness for accuracy.
- **Epochs 21-40** (adv=0.001): PSNR climbs to 28.44 (+0.14 dB), lap_var drops to 2580 (−10%). Adversarial adds a small additional gradient but doesn't change the trade-off direction.

The architecture's prior is **sharp-by-default**. To get pixel accuracy back, we'd need either:
- A much stronger GT anchor (e.g., LPIPS weight 5×, or 2× the L1 weight), OR
- No adversarial (adv=0; adversarial rewards sharpness even when L_pix fights it), OR
- Warm-start from a pixel-accurate teacher (e.g., the v1 RFDN as a starting point).

### 4.3 Why this is "I1 with a new arch instead of a new shortcut"

I1 was from-scratch + nearest-residual + adv = 3 new things. I3 was from-scratch + new-arch + adv = also 3 new things. Same failure mode signature:

| | I1 (RFDN + nearest) | I3 (SRVGG + nearest) |
|---|---|---|
| From-scratch | yes | yes |
| New thing introduced | residual type (bicubic→nearest) | architecture (RFDN→SRVGG) |
| Adv | 0.001 | 0.001 |
| Test PSNR vs v1 | −1.92 dB | **−1.98 dB** |
| Test PSNR vs bicubic | −1.48 dB | **−1.54 dB** |
| Halt signature | H6 (student < bicubic) | H6 (student < bicubic) |
| Lap_var (full-frame) | 109.4 (5× v1) | **306.5 (15× v1)** |

**Generalization of the 2026-09-03 lesson**: "Pick one new thing per run; don't combine from-scratch + a new design choice + adversarial." I1 was the same lesson with a different new thing.

### 4.4 What we'd need to make I3 work (Phase 5 candidate, not Phase 4)

The architecture is correct. Three orthogonal fixes that would need to be tested:

1. **Warm-start v1 → finetune with SRVGG-distill** (single new thing: arch swap, no from-scratch). Risk: capacity mismatch (RFDN has 5 channels of attention features that SRVGG can't consume).
2. **Drop adversarial, push LPIPS weight to 1.0** (kill the sharpness amplifier, lean on perceptual anchor). Risk: PSNR-only training has known underfitting on anime texture.
3. **Lower `--lambda-adv 0.0001`** (10× lower than I3's 0.001). Risk: too small to matter.

None of these were in plan §3 scope. Documenting for Phase 5 reference.

---

## 5. Lessons for Phase 5 (post-Phase 4)

1. **SRVGG body unlocks animevideov3's response signal but at the cost of pixel accuracy**. The architecture is the right inductive bias; the loss recipe is the problem. Future SRVGG-student variants must NOT combine from-scratch + adversarial.
2. **SRVGG is the fastest architecture tested** (61 ms/frame at 854×480). If a future student variant hits PSNR ≥ 29.0 on SRVGG body, the latency win (1.7× over v1) is on the table for free.
3. **In-batch lap_var at epoch 1 is a leading indicator of oversharpening**. If epoch-1 in-batch lap_var > 5× v1's typical, the run will overshoot regardless of adversarial schedule. Halve λ_adv or warm-start.
4. **Vendored `archs.py` shortcut_mode parity is now in place** (PREREQUISITE met). Future nearest-shortcut RFDN ckpts will load correctly via the GUI; GUI dispatch for `srvgg_student` is ready but not in `registry.py` (would only be added on promotion).
5. **Phase 4 is closed**. Three sequential ablations all failed the binding D.1 gate. Plan §5 matrix "FAIL FAIL FAIL → ship animevideov3 baseline only" governs. The shipping recommendation is **unchanged from Phase 3.E**: v1 RFDN (real-time, 21.0 lap_var) and animevideov3 SRVGG (quality, 58.1 lap_var).

---

## 6. Artifacts

| Path | Purpose |
|---|---|
| `runs/distill_i3_srvgg_body/student_best.pt` | Best raw-state ckpt (val PSNR 28.44) — preserved un-promoted |
| `runs/distill_i3_srvgg_body/student_best_raw.pt` | Same model, alternate filename |
| `runs/distill_i3_srvgg_body/student_best_ema.pt` | Best EMA ckpt (val EMA PSNR 28.10) — preserved |
| `runs/distill_i3_srvgg_body/student_last.pt` | Final epoch (40) raw state |
| `runs/distill_i3_srvgg_body/train_log.csv` | Per-epoch metrics (40 rows) |
| `runs/distill_i3_srvgg_body/epoch_*.json` + `epoch_*.pt` | Per-epoch state + metrics (last 5 un-archived; older in `archive/`) |
| `runs/distill_i3_srvgg_body/results_table.md` + `results_table.csv` | Final test-set metrics |
| `runs/distill_i3_srvgg_body.log` | Full stdout/stderr log (Tee-Object) |
| `tmp/eval_i3_full_frame.py` | One-off full-frame eval with TinySRVGGStudent arch dispatch |
| `tmp/i3_eval/` | Full-frame eval outputs (i3_srvgg_ema + i3_srvgg_raw) + metrics.csv |
| `tmp/summarize_i3.py` | One-off per-epoch metric summary |

---

## 7. Decision

**Phase 4.I3: NOT PROMOTED.** Per plan §3.7, do not retry with different hyperparameters.

**Status of Phase 4 plan (final):**
- I1: COMPLETE, NOT PROMOTED (D.1 FAIL — oversharpening).
- I2a / I2b: COMPLETE, BOTH NOT PROMOTED (D.2 FAIL).
- I3: COMPLETE, NOT PROMOTED (D.1 FAIL + H6 oversharpening).

**Plan §5 decision matrix**: `FAIL FAIL FAIL → ship animevideov3 baseline only`. Animevideov3 (pre-trained, xinntao, BSD-3) remains the canonical "quality" GUI option. v1 RFDN remains the canonical "real-time" option.

**Total Phase 4 wall-time**: ~80 minutes of training (I1 22 min + I2a 17 min + I2b 24 min + I3 21 min).
**Total Phase 4 LOC**: ~+320 (5 new flags / dispatch points, 1 new student class, vendored GUI parity, 4 + 9 new tests).
**Net new model shipped**: none. v1 + animevideov3 remain the production options.

---

## 8. Cross-references

- Plan: `docs/plans/student_phase4_nearest_adv_srvgg_plan.md` §3
- I1 result: `docs/plans/student_phase4_i1_nearest_residual_result.md`
- I2 result: `docs/plans/student_phase4_i2_isolated_adv_result.md`
- Memory: `docs/PROJECT_MEMORY.md` §3 Phase 4 history, §4 pending, §5 anchors, §6 decisions, §7 halt conditions
- AGENTS.md Phase 4 section: lines ~1010–1108 (will be updated to "post-I3" after this commit)
- Halt conditions: §7 H6 (oversharpening) is the binding signal here
