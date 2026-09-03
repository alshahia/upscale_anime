# Phase 4 Plan — Architecture & Recipe Variants for Student v3+

**Branch:** `feature/phase-1-realtime-4k`
**HEAD:** `ad14781` (Phase 3.F lessons learned) → target `+3` new commits
**Author:** anime_upscaler agent (after user approval)
**Goal:** Three sequential experiments to break the RFDN+anneal ceiling established in Phase 3. Each is gated by an explicit success criterion; failures are documented, not retried.

---

## 0. Context (read first)

### What Phase 3 found
- v1 student (RFDN, 315K, bicubic residual) trained via SPAN distillation: **PSNR 29.89 dB, lap_var 21.0**.
- v3 student (RFDN, warm-start v1, teacher=animevideov3, adv=0.001, shortcut=off): **PSNR 29.99 dB, lap_var 22.3** — **D.2 gate FAIL** (threshold ≥ 35). Not shipped.
- RealESRGAN AnimeVideo v3 (pre-trained, 621K SRVGG, nearest residual): **PSNR 29.04 dB, lap_var 58.1** — winner.
- The RFDN+anneal combination mode-collapses in 5–15 epochs (Phase 3 v3 runs 1 and 2).
- The RFDN+adversarial-only combination (without anneal) plateaus at lap_var ≈ 22 — no real change from v1.

### What Phase 3 did NOT try
- **I1**: RFDN with NEAREST residual (single-line change to `anime_upscaler/student.py` line 101: `mode="bicubic"` → `mode="nearest"`).
- **I2**: Adversarial without feature distillation (add a `--feat-weight` flag, set to 0, teacher=SPAN with adversarial).
- **I3**: SRVGG-body student at our 315K budget (replace RFDN body with 12 stacked 3×3 convs + PReLU + PixelShuffle head + nearest residual). Mirrors animevideov3's inductive biases.

### Empirical anchors (use these for expectations)
| Model | Params | Residual type | PSNR (test) | lap_var (full) |
|---|---|---|---|---|
| bicubic | 0 | none | 29.45 | 11.8 |
| v1 RFDN | 315K | bicubic | 29.89 | 21.0 |
| v3 RFDN (adv, warm-start) | 315K | bicubic | 29.99 | 22.3 |
| animevideov3 SRVGG | 621K | nearest | 29.04 | **58.1** |
| SPAN (our ckpt) | 2.237M | bicubic | 30.44 (test, retrained) | — |

### Hard rules (apply to all phases)
1. **No destructive git ops** (`--force`, hard reset, branch delete).
2. **Per-epoch ckpts preserved** (Q6 rule: do not delete without user confirmation).
3. **All training writes to `runs/` (gitignored)** — never commit `.pth` files.
4. **Numeric-sort per-epoch rotation** (already fixed in `d62b524`).
5. **Halt conditions** (from Phase 3 handoff C.3, reproduced verbatim):
   - Val PSNR < 20 dB → mode collapse → halt.
   - Val PSNR < 26 dB after 10 epochs → slow convergence → halt.
   - EMA PSNR not improving for 10 consecutive epochs → plateau → halt.
6. **D.2 promotion gate** (from handoff): ship only if **full-frame lap_var ≥ 35** on the harness. PSNR is informational, not gating.

---

## 1. Phase I1 — Nearest-Residual RFDN

### 1.1 Hypothesis

The RFDN+SPAN student uses `mode="bicubic"` for the residual shortcut (see `anime_upscaler/student.py` line 101). The bicubic shortcut is *itself* a low-pass filter — the residual the student has to learn is `hr − bicubic(lr)`, which is high-frequency detail minus the bicubic-smoothed baseline. The model fights its own baseline.

Animevideov3 uses **nearest** residual: `sr = body(lr) + nearest_upsample(lr)`. Nearest is sharper than bicubic, so the residual carries more of the structure and the body has less work to do. Animevideov3's lap_var of 58.1 vs our v1's 21.0 may be largely explained by this residual-type choice, not the block architecture.

If true, switching our RFDN student's residual to nearest should bring lap_var significantly closer to animevideov3, at the same parameter count and inference speed.

### 1.2 Code changes

**`anime_upscaler/student.py`:**
- Add `shortcut_mode: str = "bicubic"` arg to `RFDN.__init__` (default unchanged for backward compatibility).
- Replace line 100–101 `mode="bicubic"` with `mode=self.shortcut_mode`.
- Add `set_shortcut_mode(mode: str)` setter (mirrors `set_shortcut_weight` at line 111).

**`anime_upscaler/distill.py`:**
- Add arg line ~277: `ap.add_argument("--shortcut-mode", choices=["bicubic", "nearest"], default="bicubic", help="residual shortcut interpolation mode (Phase 4 I1).")`.
- At line ~368: `student = RFDN(scale=args.scale, shortcut_mode=args.shortcut_mode).to(device)`.

**`tests/`:** add `test_shortcut_mode.py` smoke (5 LOC): `forward` produces same shape with both modes.

**Touched lines:** `student.py:64, 100-101, 117`; `distill.py:277, 368`. Net delta: ~10 LOC.

### 1.3 Command

```bash
.venv\Scripts\python.exe anime_upscaler\distill.py \
  --teacher animevideov3 --epochs 40 --batch-size 16 --lr 5e-5 \
  --lambda-adv 0.001 --shortcut-anneal off --shortcut-mode nearest \
  --out-dir runs/distill_i1_nearest_residual
```

### 1.4 Expected outcome

| Metric | Expected | Reasoning |
|---|---|---|
| Val PSNR (ep 40) | 29.5 – 30.0 dB | Same range as v1/v3; residual type shouldn't change PSNR much |
| **Full-frame lap_var** | **35 – 50** | If nearest residual is the missing piece, sharpness jumps significantly |
| Test SSIM | 0.85 – 0.88 | Matches v1 baseline |
| Best epoch | 25 – 40 | Same as v1 (no collapse) |
| Wall time | ~17 min | Matches v1 (no arch change) |

### 1.5 Success criteria (commit + ship)
1. **D.2 PASS**: full-frame lap_var ≥ 35 on the harness.
2. PSNR ≥ 29.0 dB (don't regress on accuracy).
3. No mode collapse (final EMA PSNR > 25 dB).

### 1.6 Failure criteria (document, do NOT retry)
- Val PSNR < 26 dB after 10 epochs → halt, document, move to Phase I2.
- Final lap_var < 25 (worse than v1) → the residual-type hypothesis is wrong; move to I2.

### 1.7 Risk + mitigation
- **Risk**: From-scratch with new residual type + adversarial destabilizes (similar to Phase 3 v3 mode-collapse pattern).
- **Mitigation**: NO `--shortcut-anneal` (don't combine two new things). LR=5e-5 (proven safe). Watch val PSNR every epoch; halt on collapse.
- **Risk**: Nearest residual looks pixelated at low res, model can't fill the gaps.
- **Mitigation**: Animevideov3 already proves this works at 621K; we should also work at 315K.

### 1.8 Documentation
- New doc: `docs/plans/student_phase4_i1_nearest_residual_result.md` (~80 lines).
- Update `AGENTS.md` "Phase 4 v3 attempt" → "Phase 4 I1 nearest-residual attempt".
- Commit message: `Phase 4.I1: nearest-residual RFDN — D.2 gate result`.

---

## 2. Phase I2 — Isolated Adversarial

### 2.1 Hypothesis

The Phase 3 v3 warm-start with `--teacher span --lambda-adv 0.001` regressed (PSNR 28.91 vs v1's 29.89). In that run, **two non-pixel losses were active simultaneously**:
- **L_adv** (adversarial, pulls toward "looks natural" distribution).
- **L_feat** (feature distillation, pulls toward SPAN's intermediate features — weight 0.5, magnitude 1.14).

These two objectives pull in *different* directions, neither aligned with pixel GT. The student ends up with neither sharp output nor accurate pixels.

**Sub-hypotheses:**
- **I2a**: Adversarial alone (animevideov3 teacher, no feature distillation) → mild PSNR improvement, lap_var small improvement. Already partially tested in Phase 3 v3 (29.99 dB, lap_var 22.3). Confirms adversarial isn't itself the problem.
- **I2b**: Adversarial + response-only distillation (span teacher with `feat_weight=0`) → isolates adversarial's effect when paired with a teacher that has features. **This is the new test.**

If I2b matches v1 (29.89 dB) or better, the regression in the recent warm-start was due to the **L_feat + L_adv combo**, not L_adv alone. Future runs can keep adversarial and drop feature distillation.

If I2b still regresses, adversarial itself destabilizes this student. Future runs should drop adversarial.

### 2.2 Code changes

**`anime_upscaler/distill.py`:**
- Add arg line ~280: `ap.add_argument("--feat-weight", type=float, default=1.0, help="feature distillation weight (Phase 4 I2). 0 disables L_feat entirely.")`.
- Modify line 528–542: change `if (not use_phase3) or not is_real_esr_teacher:` to `if ((not use_phase3) or not is_real_esr_teacher) and args.feat_weight > 0:`.
- Modify line 542: `loss_feat = args.feat_weight * (sum(feat_terms) / max(len(feat_terms), 1))`.

**Touched lines:** `distill.py:280, 529, 542`. Net delta: ~5 LOC.

### 2.3 Commands

**I2a** (re-confirm known marginal case, sanity check):
```bash
.venv\Scripts\python.exe anime_upscaler\distill.py \
  --teacher animevideov3 --lambda-adv 0.001 --shortcut-anneal off \
  --feat-weight 1.0 --epochs 30 --lr 5e-5 --batch-size 16 \
  --resume pretrained/RFDN_distill_v1_4x_student.pth --fresh-epoch \
  --out-dir runs/distill_i2a_adv_only_animevideov3
```

**I2b** (the new isolated test):
```bash
.venv\Scripts\python.exe anime_upscaler\distill.py \
  --teacher span --lambda-adv 0.001 --shortcut-anneal off \
  --feat-weight 0 --epochs 40 --lr 5e-5 --batch-size 16 \
  --resume pretrained/RFDN_distill_v1_4x_student.pth --fresh-epoch \
  --out-dir runs/distill_i2b_adv_only_span
```

### 2.4 Expected outcomes

| Metric | I2a | I2b |
|---|---|---|
| Val PSNR | 29.7 – 30.1 | 29.5 – 30.2 |
| **Full-frame lap_var** | 22 – 25 (small bump) | 22 – 30 (depends on SPAN's sharpness signal) |
| Feature distillation | 0 (auto, no taps) | 0 (forced) |
| Wall time | ~13 min | ~17 min |

### 2.5 Success criteria (ship I2's model)
1. **D.2 PASS**: full-frame lap_var ≥ 35.
2. PSNR ≥ 29.0 dB.
3. Beats the corresponding `feat_weight=1` baseline (i.e., I2b > recent span+adv regression at 28.91).

### 2.6 Failure criteria (document, do NOT retry)
- I2a or I2b regresses below 28.5 dB → adversarial destabilizes this student even without feature distillation. Drop adversarial from future plans.
- Both fail D.2 → move to Phase I3 (the only path remaining).

### 2.7 Risk + mitigation
- **Risk**: I2b's regression is unrelated to feature distillation (some other interaction). Hard to know without ablation.
- **Mitigation**: I2a is the sanity check. If I2a matches Phase 3 v3 (29.99) but I2b regresses, we know SPAN-specific interaction.
- **Risk**: `--resume` doesn't load D or adapters correctly (Phase 3 had bugs here).
- **Mitigation**: `--fresh-epoch` resets schedule; re-init D from scratch.

### 2.8 Documentation
- New doc: `docs/plans/student_phase4_i2_isolated_adv_result.md` (~80 lines).
- Update `AGENTS.md` with Phase 4 I2 lesson (either "adversarial OK without L_feat" or "adversarial destabilizes student").
- Commit message: `Phase 4.I2: isolated adversarial — feat-distill ablation`.

---

## 3. Phase I3 — SRVGG-Body Student

### 3.1 Hypothesis

RFDN's design (feature distillation + pixel attention) was made for **photo SR**, not anime. Animevideov3 (a plain SRVGG stack) outperforms our RFDN despite being smaller in some respects (621K vs 315K). The block design — not just the residual type — matters.

A SRVGG-body student at our 315K budget would:
- Use the **same inductive bias** as animevideov3 (12 stacked 3×3 convs + PReLU).
- Have the **same residual type** as animevideov3 (nearest).
- Distill directly from animevideov3 (no feature mismatch).
- Run at ~similar speed to RFDN (no FIM overhead, but more convs).

This is the architecturally correct fix — copy what works.

### 3.2 Code changes

**`anime_upscaler/student.py`:**
- New class `TinySRVGGStudent` (~50 LOC):
  - `__init__(scale=4, num_ch=42, num_blocks=12)` — 42 channels × 12 convs ≈ 315K params (verify with `student.num_params() < 600_000`).
  - `head = Conv2d(3, num_ch, 3, padding=1)`.
  - `body = Sequential(Conv2d(num_ch, num_ch, 3, padding=1) for _ in range(num_blocks))`.
  - `act = PReLU(num_ch)` (one shared across blocks, matches SRVGGNetCompact).
  - `upsample = Sequential(Conv2d(num_ch, 3*scale*scale, 3, padding=1), PixelShuffle(scale))`.
  - `forward`: `out = upsample(act(body(head(x)))) + nearest_interpolate(x, scale_factor=self.scale)`.
  - **No `return_features`** (SRVGG has no meaningful intermediate features for distillation; mirrors animevideov3's setup).
  - **No `shortcut_weight`** (single block design; no need for anneal).

**`anime_upscaler/distill.py`:**
- Add arg line ~281: `ap.add_argument("--arch", choices=["rfdn", "srvgg"], default="rfdn", help="student architecture (Phase 4 I3).")`.
- Modify line 368: build student based on `args.arch`:
  ```python
  if args.arch == "srvgg":
      student = TinySRVGGStudent(scale=args.scale).to(device)
  else:
      student = RFDN(scale=args.scale, shortcut_mode=args.shortcut_mode).to(device)
  ```
- Modify line 369–372: `tap_chans = None` and `adapters = None` when `args.arch == "srvgg"` (no taps → no feature distillation even with `feat_weight=1`).
- Modify line 504–507 (SRVGG path): already correctly returns `s_feats = []` when `is_real_esr_teacher` is True; same logic applies to `args.arch == "srvgg" + teacher=animevideov3`. No code change needed; logic already covers this case.

**`apps/anime_upscaler_gui/anime_upscaler_gui/archs.py`:**
- Add `TinySRVGGStudent` import + dispatch (mirrors existing `RFDN` path). Net delta: ~10 LOC.

**`tests/`:** add `test_tiny_srvgg.py` smoke:
- Forward pass: `out = srvgg(torch.randn(1, 3, 48, 48))`, assert shape `(1, 3, 192, 192)`.
- Param count: assert `< 600_000`.
- Latency: `time` 30 forward passes at `270x480` LR, assert < 200 ms/frame on RTX 4000.

**Touched lines:** `student.py:new section`, `distill.py:281, 368, 369-372`, `archs.py`. Net delta: ~80 LOC + new test file.

### 3.3 Smoke test (BEFORE training)

```bash
.venv\Scripts\python.exe -c "from anime_upscaler.student import TinySRVGGStudent; import torch; s = TinySRVGGStudent(); print(s.num_params()); print(s(torch.randn(1, 3, 48, 48)).shape)"
```

Expected: `315xxx` (params) and `torch.Size([1, 3, 192, 192])` (output shape).

### 3.4 Command

```bash
.venv\Scripts\python.exe anime_upscaler\distill.py \
  --arch srvgg --teacher animevideov3 --epochs 40 --batch-size 16 --lr 5e-5 \
  --lambda-adv 0.001 --shortcut-anneal off --shortcut-mode nearest \
  --out-dir runs/distill_i3_srvgg_body
```

### 3.5 Expected outcomes

| Metric | Optimistic | Pessimistic | Reasoning |
|---|---|---|---|
| Val PSNR | 29.0 – 29.5 | 27.0 – 28.5 | SRVGG has less efficient gradient flow than RFDN at small budgets |
| **Full-frame lap_var** | **40 – 55** | 25 – 35 | If architecture is the missing piece, sharpness jumps significantly |
| Inference latency | ~80 ms | ~150 ms | SRVGG is faster per conv but has more convs than RFDN blocks |
| Best epoch | 30 – 40 | n/a (collapse) | If it converges, late epoch |
| Wall time | ~17 min | n/a | Similar to v1 |

### 3.6 Success criteria (commit + ship)
1. **D.2 PASS**: full-frame lap_var ≥ 35.
2. PSNR ≥ 28.0 dB (don't regress catastrophically).
3. Inference latency ≤ 200 ms/frame at 270×480 LR (RTX 4000) — don't break real-time.

### 3.7 Failure criteria (document, do NOT retry)
- Val PSNR < 26 dB after 10 epochs → halt, mode collapse.
- Val PSNR < 27.5 dB at end → architecture didn't help; document and STOP this branch.
- Inference latency > 250 ms → breaks real-time budget; abandon.

### 3.8 Risk + mitigation
- **Risk**: New architecture, untested training dynamics. Could mode-collapse at LR=5e-5.
- **Mitigation**: Smoke test BEFORE training (verifies forward pass + param count). If collapse: try LR=1e-4 (smaller steps, slower but safer).
- **Risk**: SRVGG needs more epochs to converge than RFDN (12 convs vs 6 RFDBlocks).
- **Mitigation**: 40 epochs is generous; cosine LR decay handles late-stage refinement.
- **Risk**: `--arch srvgg` flag accidentally accepted but `--shortcut-mode` not honored.
- **Mitigation**: `TinySRVGGStudent` has no `shortcut_weight`; `--shortcut-mode` arg is silently ignored (acceptable; nearest is the only sensible default for SRVGG).

### 3.9 Documentation
- New doc: `docs/plans/student_phase4_i3_srvgg_body_result.md` (~100 lines, includes ablation table).
- Update `apps/anime_upscaler_gui/anime_upscaler_gui/registry.py` if shipping: add `TinySRVGGStudent` preset.
- Update `AGENTS.md` "Phase 4 I3 attempt" section.
- Commit message: `Phase 4.I3: SRVGG-body student — D.2 gate result`.

---

## 4. Promotion gate (applies after each phase)

After each phase, run the harness on the best epoch:

```bash
.venv\Scripts\python.exe scripts\compare_students_vs_pretrained.py \
  --v3-ckpt runs/<phase>_*/student_best_ema.pt \
  --output results/quality_compare_<phase>_*/
```

**Promotion rule:** Ship the new checkpoint as a GUI preset **iff** full-frame lap_var ≥ 35 AND PSNR ≥ baseline AND inference latency ≤ 200 ms/frame.

If the gate fails: keep the `runs/` directory for forensics (Q6), document the failure, move to next phase.

---

## 5. Decision matrix (what to ship after Phase 4)

| I1 | I2 | I3 | Action |
|---|---|---|---|
| PASS | FAIL | FAIL | Ship I1; I2/I3 documented as failed ablations |
| FAIL | PASS | FAIL | Ship I2's best (animevideov3 teacher + adv + no feat) |
| FAIL | FAIL | PASS | Ship I3 |
| PASS | PASS | FAIL | Ship I1 (cheapest); I2 documented |
| PASS | FAIL | PASS | Ship I3 if lap_var higher than I1; else I1 |
| FAIL | PASS | PASS | Ship I3 if lap_var higher than I2; else I2 |
| PASS | PASS | PASS | Ship I3 (most novel); document all three |
| FAIL | FAIL | FAIL | **STOP**: ship animevideov3 baseline only (already in registry) |

The handoff already shipped animevideov3 as the high-quality GUI option. If all three phases fail, the recommendation is unchanged from Phase 3.E: **users who want quality → animevideov3; users who want real-time → v1**.

---

## 6. Rollback

If all three phases fail and we need to revert code changes:

```bash
git checkout ad14781 -- anime_upscaler/student.py anime_upscaler/distill.py apps/anime_upscaler_gui/anime_upscaler_gui/archs.py
git clean -fd tests/test_shortcut_mode.py tests/test_tiny_srvgg.py
```

Per-epoch ckpts in `runs/` are preserved (Q6 rule).

---

## 7. Halt conditions (apply to all phases)

These are the **same** halt conditions from the Phase 3 handoff C.3, reproduced for clarity:

- **H1 (collapse)**: Val PSNR < 20 dB at any point → mode collapse → halt.
- **H2 (slow)**: Val PSNR < 26 dB after 10 epochs → slow convergence → halt.
- **H3 (plateau)**: EMA PSNR not improving for 10 consecutive epochs → halt.
- **H4 (lap regression)**: Full-frame lap_var at ep 40 < baseline (v1 = 21.0) → regression → halt.
- **H5 (D2 fail)**: Final lap_var < 35 → D.2 gate fail → don't ship, document.

---

## 8. Documentation deliverables

| Doc | When | Lines |
|---|---|---|
| `docs/plans/student_phase4_nearest_adv_srvgg_plan.md` (this file) | BEFORE any code | 250 |
| `docs/plans/student_phase4_i1_nearest_residual_result.md` | After I1 | 80 |
| `docs/plans/student_phase4_i2_isolated_adv_result.md` | After I2 | 80 |
| `docs/plans/student_phase4_i3_srvgg_body_result.md` | After I3 | 100 |
| `AGENTS.md` updates | After each phase | +30 per phase |

---

## 9. Commit history (target)

```
ad14781  Phase 3.F: AGENTS.md lessons learned + v3 result doc     ← HEAD
xxxxxxx  Phase 4 plan: I1 nearest-residual + I2 isolated adv + I3 SRVGG body  ← this doc
xxxxxxx  Phase 4.I1: nearest-residual RFDN (commit after run)
xxxxxxx  Phase 4.I2: isolated adversarial (commit after run)
xxxxxxx  Phase 4.I3: SRVGG-body student (commit after run)
```

---

## 10. Estimated timeline

| Phase | Code | Train | Eval | Doc | Total |
|---|---|---|---|---|---|
| I1 | 15 min | 17 min | 10 min | 15 min | ~1 hour |
| I2 (2 sub-runs) | 10 min | 30 min | 15 min | 20 min | ~1.5 hours |
| I3 | 90 min | 17 min | 30 min | 30 min | ~3 hours |
| **Total** | | | | | **~6 hours** |

If all three succeed and the GUI registry is updated, add 30 min for the registry entry + smoke test.

---

## 11. Approval gate

**STOP HERE.** Do not begin code changes until the user has reviewed this plan and explicitly approved Phase I1 (and the sequence I1 → I2 → I3).

The plan is reversible (Section 6). All runs are written to `runs/` (gitignored). No destructive operations are planned.
