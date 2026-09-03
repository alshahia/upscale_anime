# PROJECT_MEMORY.md — Live state of upscale_anime

> **The canonical "where are we?" for any session. Read this first.**
> **Update it whenever a decision is made, a phase completes, or state changes.**
> Pointed to by `AGENTS.md` "Memory protocol" section.

---

## 0. Memory protocol (for any agent)

### When to READ
- Start of every session (before any other AGENTS.md section).
- Before making non-trivial changes — confirm assumptions still hold.
- Before declaring a task complete — verify cross-references are intact.

### When to UPDATE
- A decision is made or reversed → add to "Decision log".
- A phase begins or completes → update "Phase history".
- Tasks move forward or get added/dropped → update "Pending tasks".
- A new finding (e.g. lap_var number) → update "Key empirical anchors".
- A new critical rule learned → update "Halt conditions / guardrails".

### How to UPDATE
- Append to the relevant section (do NOT rewrite history).
- Add a one-line entry with date stamp (YYYY-MM-DD).
- Keep entries short; cross-reference longer docs when needed.
- Never delete prior decisions — supersede them with a new entry that explains why.
- After updating, commit with message: `memory: <one-line summary of change>`.

### When NOT to use this file
- Long-form code documentation → `docs/plans/` or `AGENTS.md`.
- Bench tables / measurements → `docs/v7_results.md`, result JSONs.
- Per-phase detail → dedicated result doc (e.g. `student_v3_result_2026_08.md`).

---

## 1. Project snapshot

Anime super-resolution project. Two production paths:
- **RFDN Distill v1** (315K params) — real-time (~3 fps on RTX 4000), shipped, PSNR 29.89 / lap_var 21.0.
- **RealESRGAN AnimeVideo v3** (621K params, pre-trained from xinntao) — quality option, PSNR 29.04 / lap_var 58.1, already in GUI registry.

Goal: train a student that beats v1 and approaches animevideov3 quality. Phase 3 attempts (adversarial v3 RFDN) failed D.2 quality gate (lap_var 22.3 vs threshold 35). Phase 4.I1 (nearest-residual RFDN) executed 2026-09-03 — D.2 PASS but D.1 PSNR FAIL (oversharpened, 27.97 dB). Phase 4.I2 (isolated adversarial) executed same day — both sub-runs D.2 FAIL (I2a 20.8, I2b 24.2). I1 + I2 NOT PROMOTED. Per plan §5 "FAIL FAIL → ship animevideov3 baseline only"; I3 is optional last attempt.

Working directory: `E:\python projects\upscale_anime`
Branch: `feature/phase-1-realtime-4k`
HEAD: see §2 below.

---

## 2. Branch and HEAD

| Commit | Message |
|---|---|
| `7ac0665` | Phase 4.I2: feat-weight flag — isolated adversarial ablation |
| `19f6fa0` | docs: reflect Phase 4.I1 outcome in AGENTS.md, plan §11, PROJECT_MEMORY.md |
| `924995f` | memory: I1 result + decision log + empirical anchors updated |
| `bd72dd3` | Phase 4.I1: nearest-residual RFDN — D.2 PASS, D.1 FAIL, NOT PROMOTED |
| `2cb458c` | memory: self-record creation in §2 HEAD table and §9 update history |
| `4434609` | memory: add docs/PROJECT_MEMORY.md (canonical session-starting state file) and AGENTS.md pointer |
| `624fb11` | AGENTS.md: add Phase 4 plan section (I1+I2+I3, awaiting approval) |
| `26ac928` | Phase 4 plan: I1 nearest-residual + I2 isolated adv + I3 SRVGG body |
| `ad14781` | Phase 3.F: AGENTS.md lessons learned + v3 result doc |

Last updated: 2026-09-03 (after Phase 4.I1 execution).

---

## 3. Phase history

### Phase 1 (Real-time 4K) — SHIPPED (2026-08-31)
- TensorRT engine for RFDN student (2.37x speedup over PyTorch FP16).
- NVENC hardware encoder (26.5% CPU vs 42.5% libx264).
- Batching deferred (RFDN too small; b4 is slower than b1).
- **End-to-end fps**: 18.80 mean, 20.52 peak at 854x480 to 3416x1920.
- Goal: >=25 fps — **PARTIAL** (~75% of target).
- Full report: `docs/rfdn_realtime_report.md` Section 10.

### Phase 2 v3 student — SHIPPED (predecessor of v3)
- SPAN-teacher distillation, RFDN 315K, 40 epochs LR=5e-5.
- PSNR 29.89, lap_var 21.0.
- This is the current `RFDN_distill_v1_4x_student.pth`.

### Phase 3 v3 (adversarial) — NOT PROMOTED (2026-09-02)
- RFDN warm-start + animevideov3 teacher + PatchGAN70 + HingeGANLoss + edge loss.
- 30 epochs, ~17 min on RTX 4000.
- Result: PSNR 29.99 (marginal +0.1 dB over v1), **lap_var 22.3** (D.2 gate FAILED, threshold 35).
- Two warm-start + shortcut-anneal runs mode-collapsed (PSNR 29.93 to 5.41 in 5 ep).
- **Lesson**: shortcut anneal is hostile to warm-start. Adversarial alone plateau.
- Ckpt preserved un-promoted: `runs/distill_v3_4x_v3_epoch18_ema_unpromoted.pth`.
- Full doc: `docs/plans/student_v3_result_2026_08.md`.

### Phase 4 — I1 + I2 EXECUTED, BOTH NOT PROMOTED (2026-09-03)
- Three sequential experiments (I1 → I2 → I3) to break the RFDN+anneal ceiling.
- **I1**: nearest-residual RFDN — **COMPLETE**, **NOT PROMOTED**.
  - Hypothesis "nearest residual → sharper" confirmed: full-frame lap_var **109.4** (5.2× v1).
  - But PSNR regressed: **27.97 dB** vs v1 29.89 (−1.92 dB), student < bicubic (−1.48 dB).
  - D.2 gate PASS (lap_var ≥ 35). D.1 gate FAIL (PSNR ≥ 29.0). Sharpness ≠ quality.
  - Likely root cause: adversarial + from-scratch + new-shortcut = oversharpening.
  - Run preserved: `runs/distill_i1_nearest_residual/` (per Q6 rule).
  - Full doc: `docs/plans/student_phase4_i1_nearest_residual_result.md` (9056 bytes).
- **I2**: isolated adversarial — **COMPLETE**, **BOTH sub-runs NOT PROMOTED** (D.2 FAIL).
  - Two sub-runs warm-started from v1: I2a animevideov3 + adv + feat=1 (30 ep); I2b SPAN + adv + **feat_weight=0** (40 ep).
  - **I2a (sanity check, identical to Phase 3 v3 recipe)**: val PSNR EMA 29.885 (= v1); full-frame lap_var **20.8** (= v1). Animevideov3+adv+feat=1 is a no-op over v1.
  - **I2b (the new isolation)**: val PSNR EMA **29.911** (+0.022 dB over v1); full-frame lap_var **24.2** (+15% over v1). SPAN+adv+feat=0 yields a small but real improvement but still below 35 threshold.
  - **Lesson**: SPAN is a stronger distillation teacher than animevideov3 for RFDN. `feat_weight=0` is the safe default for any future SPAN+adv RFDN variant. Animevideov3's per-pixel response is too lossy for adversarial-only intervention to help; SRVGG body (I3) is needed.
  - Runs preserved: `runs/distill_i2a_adv_only_animevideov3/`, `runs/distill_i2b_adv_only_span/` (per Q6 rule).
  - Full doc: `docs/plans/student_phase4_i2_isolated_adv_result.md` (10196 bytes).
- **I3**: pending user decision. Per plan §5 "FAIL FAIL → ship animevideov3 baseline only"; I3 is optional.
- Full plan: `docs/plans/student_phase4_nearest_adv_srvgg_plan.md` (391 lines).
- Code added (Phase 4 I1): `--shortcut-mode {bicubic,nearest}` arg, `RFDN(shortcut_mode=...)` constructor, `set_shortcut_mode()` method, conditional `align_corners` for nearest mode, `tests/test_shortcut_mode.py` (4 tests, all passing).
- Code added (Phase 4 I2): `--feat-weight {float, default=1.0}` arg, gate at line 539 (feature distillation block conditioned on `args.feat_weight > 0`), weight multiplier at line 552. All 19 pre-existing tests still pass. Default = 1.0 (backward-compatible with Phase 2 v3 / Phase 3 recipe).

---

## 4. Pending tasks

### Blocked on user re-approval (matrix "FAIL FAIL → ship animevideov3 baseline only")

- [x] **Phase 4.I1** — Nearest-residual RFDN. **COMPLETE, NOT PROMOTED** (D.1 FAIL).
  Docs: `docs/plans/student_phase4_i1_nearest_residual_result.md`. Per Q6, run preserved.
- [x] **Phase 4.I2** — Isolated adversarial. **COMPLETE, NOT PROMOTED** (both D.2 FAIL).
  Code: ~5 LOC change (--feat-weight flag + gate + multiplier), backward-compat default=1.0.
  Two sub-runs: I2a (animevideov3 + adv, val PSNR EMA 29.885 = v1, lap_var 20.8),
  I2b (span + adv + feat=0, val PSNR EMA 29.911 = +0.022 over v1, lap_var 24.2).
  Per Q6, both runs preserved under `runs/distill_i2a_*/` and `runs/distill_i2b_*/`.
  Docs: `docs/plans/student_phase4_i2_isolated_adv_result.md`.
- [ ] **Phase 4.I3** — SRVGG-body student (optional; user decision pending).
  Code change: ~80 LOC (new `TinySRVGGStudent` + `--arch` flag + `archs.py` vendored-RFDN
  update so it honors `shortcut_mode` + smoke test). Recommend `--feat-weight 0` baseline
  per I2 lesson. Success: D.2 PASS, PSNR >= 28.0, inference <= 200 ms/frame.
  Output dir: `runs/distill_i3_srvgg_body/`.

### Always

- [ ] Update this file whenever state changes.

---

## 5. Key empirical anchors

### Anime video test (854x480 to 3416x1920, 18 frames, ~1 s clip)

| Model | PSNR (test dB) | lap_var (full frame) | SSIM | fps | infer ms |
|---|---|---|---|---|---|
| bicubic | 29.45 | 11.8 | 0.8780 | — | 22 |
| v1 RFDN (315K, ours) | **29.89** | 21.0 | 0.87 | 3.07 | 104.4 |
| v3 RFDN (315K, failed) | 29.99 | 22.3 | — | 3.22 | 102.4 |
| **RealESRGAN AnimeVideo v3** (621K, xinntao) | 29.04 | **58.1** | 0.8863 | 3.21 | 98.7 |
| SPAN checkpoint (2.2M, ours) | 30.44 (test retrain) | — | 0.9218 | — | — |
| **Phase 4.I1 RFDN (315K, nearest)** | 27.97 (regressed) | **109.4** (5× v1, oversharpened) | 0.8335 | — | 86.5 | D.2 PASS, D.1 FAIL, **NOT PROMOTED** |
| **Phase 4.I2a (RFDN 315K, animevideov3+adv+feat=1)** | 29.32 (test) | **20.8** (= v1) | — | — | 95.8 | D.2 FAIL (= v1, no improvement) |
| **Phase 4.I2b (RFDN 315K, span+adv+feat=0)** | 29.35 (test) | **24.2** (+15% over v1) | — | — | 97.2 | D.2 FAIL (+15%, still below 35) |

> Important: the **27.97** for I1 comes from distill.py's `evaluate()` on the held-out `test` split of `anime_video_frames` (192×192 crops). The PSNR for v1/v3/animevideov3 in this table is the **clip** mean from the canonical harness. Same data domain, slightly different test pipeline — close enough to flag the regression but not directly comparable. The **lap_var 109.4** was measured with `tmp/eval_i1_correct_shortcut.py` on the same `tmp/real_video_1sec.mp4` frame 8 used by the canonical harness, so it IS directly comparable to v1's 21.0 / animevideov3's 58.1.

### Architecture comparison

| Model | Params | Residual type | Block design |
|---|---|---|---|
| RFDN v1 (ours) | 315K | **bicubic** | FIM + PixelAttention |
| SRVGG animevideov3 | 621K | **nearest** | Plain 3x3 convs + PReLU |
| SPAN ckpt (ours) | 2.237M | **bicubic** | Swift Parameter-free Attention |

**Key insight (2026-09-03)**: SPAN checkpoint is ALSO bicubic-residual — bicubic isn't the sole reason for v1's smoothness. Animevideov3's nearest residual + anime-specific training data explains most of the gap.

### Failure signatures (so we don't retry dead paths)

- From-scratch + shortcut-anneal + LR >= 2e-4 → mode collapse (PSNR drops to 5 dB in 5 ep).
- Warm-start v1 + shortcut-anneal → mode collapse (PSNR 29.92 to 5.41).
- Warm-start v1 + span + adv + feat=1 → regression (PSNR 28.91 < v1's 29.89).
- Adversarial alone (animevideov3, no feat) → marginal only (PSNR 29.99, lap_var 22.3).
- **NEW 2026-09-03**: From-scratch + nearest-residual + adv=0.001 → oversharpened
  (PSNR 27.97 < bicubic 29.45; lap_var 109.4, **5× v1 but 1.9× animevideov3**).
  Adversarial + from-scratch + new shortcut together = oversharpening.
  Pick one new thing per run; do NOT combine all three.

---

## 6. Decision log

### 2026-09-03 — Phase 4 plan committed but NOT executing
- Decision: Wait for user approval before modifying code.
- Reason: Phase 3 already attempted 4 variations and all plateaued at lap_var 22-23. The user requested a plan, not unilateral execution.
- Reversal cost: low (plan is reversible via git checkout).

### 2026-09-03 — Phase 4.I1 executed (user approved after seeing the plan)
- Decision: Run `--shortcut-mode nearest` from-scratch per plan §1.3.
- Reason: User said "go to Phase I1".
- Result: D.2 PASS (lap_var 109.4), D.1 FAIL (PSNR 27.97). NOT PROMOTED.
- Reversal cost: low. Code change is backward-compatible (default = bicubic).

### 2026-09-03 — Sharpness ≠ quality (lesson from I1)
- Decision: Treat PSNR ≥ 29.0 dB as the binding constraint for any future student,
  not lap_var ≥ 35.
- Reason: I1 lap_var 109.4 (5× v1) was a great sharpness number but PSNR dropped
  −1.92 dB below v1. A model can trivially maximize lap_var by hallucinating edges.
- Action: All future Phase 4/5 promotions require BOTH D.1 AND D.2 (D.1 binds).

### 2026-09-03 — Animevideov3 is shipped as quality option (carry-over from Phase 3.E)
- Decision: RealESRGAN AnimeVideo v3 (pre-trained, xinntao) ships in GUI as quality option.
- Reason: Cannot be matched by our RFDN student at available compute (~17 min/run).
- Registry: `apps/anime_upscaler_gui/anime_upscaler_gui/registry.py::PRESET_CATALOG`.

### 2026-09-02 — RFDN+anneal ceiling confirmed (carry-over from Phase 3 lessons)
- Decision: Do NOT warm-start RFDN with shortcut-anneal.
- Reason: Three separate runs mode-collapsed (PSNR drops to 5-7 dB).
- Rule: warm-starting → `--shortcut-anneal off`; from-scratch → `--shortcut-anneal 1to0` OK.

### 2026-09-02 — Adversarial+feature-distillation is a bad combo
- Decision: Avoid combining L_adv with L_feat without isolation.
- Reason: span+adv+feat=1 regressed (28.91 dB); the two objectives pull in opposite directions.
- Action: Phase 4.I2 will isolate via `--feat-weight` flag.

### 2026-09-03 — Phase 4.I2 executed (user approved after seeing plan + I1 result)
- Decision: Run both sub-runs per plan §2.3: I2a animevideov3+adv+feat=1 (30 ep, sanity),
  I2b SPAN+adv+feat=0 (40 ep, the new isolation).
- Reason: User said "Proceed to I2 (isolated adversarial)". The plan §2.1 hypothesis was
  that L_feat+L_adv was the regression driver; isolating them reveals which is responsible.
- Result: I2a D.2 FAIL (lap_var 20.8 = v1, no improvement); I2b D.2 FAIL but PARTIAL WIN:
  val PSNR EMA 29.911 = +0.022 dB over v1, full-frame lap_var 24.2 = +15% over v1. Both NOT
  PROMOTED per matrix "FAIL FAIL → ship animevideov3 baseline only".
- Reversal cost: low. Code change is backward-compatible (`--feat-weight` defaults to 1.0,
  matches Phase 2 v3 / Phase 3 recipe). All 19 pre-existing tests pass.

### 2026-09-03 — feat_weight=0 unlocks SPAN+adv gradient (lesson from I2b)
- Decision: For any future SPAN+adv RFDN variant, default `--feat-weight 0` (disable L_feat).
- Reason: I2b reproduces the 2026-08 span+adv recipe but drops L_feat; goes from
  28.91 dB regression (span+adv+feat=1) to +0.022 dB over v1 (span+adv+feat=0). The
  L_feat L2-normalized cosine pull on SPAN's intermediate features was actively hurting
  the student when combined with adversarial's distribution-matching gradient.
- Action: Use `--feat-weight 0` in any future variant. Animevideov3+adv path remains a
  no-op (I2a = v1); the student needs SRVGG body (I3) to make animevideov3's response
  signal useful at this parameter scale.

### 2026-09-03 — Animevideov3 is the GRAIN OF TRUTH for shipping (re-affirmed post-I2)
- Decision: Animevideov3 baseline remains the canonical "quality" GUI option.
- Reason: Cannot be matched by RFDN student at available compute. After 4 distinct
  attempts (Phase 3 v3 + Phase 4 I1/I2a/I2b), the best RFDN variant (I2b) still lands at
  29.91 dB val PSNR / 24.2 lap_var — far short of animevideov3's 58.1 lap_var. The
  remaining 2.4× sharpness gap is the SRVGG-body contribution (I3 still pending if
  user wants to attempt it).
- Registry: `apps/anime_upscaler_gui/anime_upscaler_gui/registry.py::PRESET_CATALOG`.

---

## 7. Halt conditions / guardrails

### Training halt conditions (apply to any new training run)

- **H1 (collapse)**: Val PSNR < 20 dB at any epoch → mode collapse → halt.
- **H2 (slow)**: Val PSNR < 26 dB after 10 epochs → slow convergence → halt.
- **H3 (plateau)**: EMA PSNR not improving for 10 consecutive epochs → halt.
- **H4 (lap regression)**: Full-frame lap_var < v1 baseline (21.0) → regression → halt.
- **H5 (D2 fail)**: Final lap_var < 35 → D.2 gate fail → don't ship.
- **H6 (oversharpening, NEW 2026-09-03)**: Student test PSNR < bicubic test PSNR → adversarial
  is producing hallucinated edges (high lap_var, low PSNR); halt, document, isolate adversarial.

### Promotion gate (D.1 AND D.2; D.1 binds, NEW 2026-09-03)
- Ship new checkpoint iff full-frame lap_var >= 35 AND PSNR >= baseline AND inference latency <= 200 ms/frame.
- **D.1 (PSNR) is now the binding constraint** (2026-09-03 lesson from I1). A model can trivially
  max lap_var by hallucinating edges; PSNR >= 29.0 dB AND not regressing below v1 is the harder gate.
- Per-epoch ckpts preserved per Q6 rule (do not delete without user confirmation).

### Code guardrails
- No destructive git ops (`--force`, hard reset, branch delete).
- All training writes to `runs/` (gitignored) — never commit `.pth` files.
- Per-epoch rotation must sort by **numeric** epoch (`int(p.stem.split('_')[1])`), not alphabetic (Phase 3 bug `21151e7`).
- LR scheduler step must come AFTER optimizer step (Phase 3 warning).

---

## 8. Cross-references

### Plans and handoffs
- `docs/plans/student_phase4_nearest_adv_srvgg_plan.md` — Phase 4 plan (391 lines).
- `docs/plans/student_v3_result_2026_08.md` — Phase 3 v3 result + lessons.
- `docs/plans/student_adversarial_plan.md` — Phase 3 handoff (~482 lines, 19 sections).
- `docs/plans/student_adversarial_handoff_2026_08.md` — Phase 3 handoff doc (538 lines).
- `docs/rfdn_realtime_report.md` — Phase 1 E2E verification.
- `docs/plans/realtime_4k_plan.md` — Phase 1 master plan.

### Code anchors (use these to find the change site)
- `anime_upscaler/student.py:64-65` — `RFDN.__init__` signature (now includes `shortcut_mode`).
- `anime_upscaler/student.py:75-82` — `RFDN.shortcut_mode` storage + validation.
- `anime_upscaler/student.py:108-118` — forward: conditional `align_corners` for `nearest`/`area` modes.

- `anime_upscaler/student.py:126-132` — `set_shortcut_weight()` (unchanged from Phase 3).
- `anime_upscaler/student.py:134-144` — `set_shortcut_mode()` runtime hook.
- `anime_upscaler/distill.py:276-278` — `--shortcut-mode {bicubic,nearest}` arg (Phase 4 I1).
- `anime_upscaler/distill.py:279-281` — `--feat-weight {float, default=1.0}` arg (Phase 4 I2).
- `anime_upscaler/distill.py:371` — `RFDN(scale=args.scale, shortcut_mode=args.shortcut_mode)` build call.
- `anime_upscaler/distill.py:539` — feature distillation gate (`args.feat_weight > 0`).
- `anime_upscaler/distill.py:552` — feature distillation weight multiplier (`args.feat_weight * `).
- `anime_upscaler/distill.py:638-639` — `_shortcut_weight` anneal call (unchanged from Phase 3).
- `anime_upscaler/distill.py:384-385` — recipe log line now includes `shortcut_mode`.
- `tmp/eval_i2a_full_frame.py` — one-off full-frame lap_var eval for I2a/I2b (untracked).
- `apps/anime_upscaler_gui/anime_upscaler_gui/archs.py:591` — **STALE**: vendored RFDN's hardcoded `mode="bicubic"` shortcut. Must be updated before any nearest-shortcut ckpt is shipped via the GUI (Phase 4.I3 prerequisite).
- `apps/anime_upscaler_gui/anime_upscaler_gui/registry.py::PRESET_CATALOG` — GUI preset catalog.
- `tmp/eval_i1_correct_shortcut.py` — one-off full-frame eval with correct shortcut mode (forensics only, untracked).
- `tests/test_shortcut_mode.py` — 4 smoke tests covering default/nearest/runtime-flip/invalid.

### Eval / harness
- `scripts/compare_students_vs_pretrained.py` — quality harness with `--v3-ckpt` flag.
- `tmp/test_animevideov3_real_video.py` — real-video test for animevideov3.
- `tmp/test_v3_student_real_video.py` — real-video test for v3 student.
- `tmp/test_v1_student_real_video.py` — real-video test for v1 student.
- Test clip: `tmp/real_video_1sec.mp4` (18 frames, 854x480).

---

## 9. Update history

- **2026-09-03**: Initial creation. Phase 4 plan committed; awaiting approval.
- **2026-09-03**: Memory file created (`docs/PROJECT_MEMORY.md`, 224 lines) + AGENTS.md "Memory protocol — READ FIRST" section added at top of file. Commit `4434609`.
- **2026-09-03**: Self-record creation in §2 HEAD table and §9 update history. Commit `2cb458c`.
- **2026-09-03**: Phase 4.I1 executed. Code added (default backward-compat:
  `shortcut_mode="bicubic"`): `--shortcut-mode` arg, `RFDN(shortcut_mode=...)`,
  `set_shortcut_mode()`, conditional `align_corners` for nearest, 4 new tests
  (`tests/test_shortcut_mode.py`, all passing). 40-epoch run completed
  (~22 min on RTX 4000). **D.2 PASS (lap_var 109.4, 5× v1), D.1 FAIL
  (PSNR 27.97 < v1 29.89 and < bicubic 29.45). NOT PROMOTED.** Result doc:
  `docs/plans/student_phase4_i1_nearest_residual_result.md` (9056 bytes).
  Lesson: sharpness ≠ quality; adversarial + from-scratch + new shortcut =
  oversharpening. Per-epoch ckpts preserved per Q6. Commits `bd72dd3`, `924995f`, `7a7a624`.
- **2026-09-03**: Documentation pass after I1. **§7 new halt condition H6** (oversharpening),
  **§7 promotion gate** now binds on D.1 (PSNR), not just D.2 (lap_var). **§8 code anchors**
  refreshed for new line numbers in `student.py` (shortcut_mode lives at lines 64-65, 75-82,
  108-118, 134-144). AGENTS.md Phase 4 section updated to reflect I1 EXECUTED/NOT
  PROMOTED + decision matrix updated + status snapshot refreshed (HEAD now `7a7a624`).
  Plan file §11 "Approval gate" replaced with execution log. Commit pending below.
- **2026-09-03**: Phase 4.I2 executed (~40 min total: 17 + 24 = ~41 min). Code added to
  `distill.py` (~5 LOC): `--feat-weight {float, default=1.0}` argparse; feature distillation
  block gated by `args.feat_weight > 0` at line 539; weight multiplier at line 552. Two
  warm-start sub-runs from v1: I2a animevideov3+adv+feat=1 (30 ep, 17.3 min, val PSNR EMA
  29.885 = v1, full-frame lap_var **20.8** = v1, held-out test 29.32); I2b SPAN+adv+feat=0
  (40 ep, 23.8 min, val PSNR EMA **29.911** +0.022 over v1, full-frame lap_var **24.2** +15%
  over v1, held-out test 29.35). Both D.2 FAIL, both NOT PROMOTED per matrix. **Three new
  decisions logged** in §6: I2 executed, `feat_weight=0` is the safe default for any future
  SPAN+adv RFDN variant, animevideov3 baseline reaffirmed as the shipping quality option.
  **AGENTS.md** Phase 4 section updated to "post-I2": status header, I2 subsection
  EXECUTED+NOT PROMOTED, decision matrix current state "FAIL FAIL → ship animevideov3
  baseline only", status snapshot. New halt condition remains H6 (oversharpening from I1).
  New result doc: `docs/plans/student_phase4_i2_isolated_adv_result.md` (10196 bytes).
  All 19 pre-existing tests still pass (`test_shortcut_mode` + `test_feature_distillation`).
  Commits: `<this-commit>` (docs pass), `7ac0665` (I2 code + test).
