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

Goal: train a student that beats v1 and approaches animevideov3 quality. Phase 3 attempts (adversarial v3 RFDN) failed D.2 quality gate (lap_var 22.3 vs threshold 35). Phase 4 plan is committed but not yet executed.

Working directory: `E:\python projects\upscale_anime`
Branch: `feature/phase-1-realtime-4k`
HEAD: see §2 below.

---

## 2. Branch and HEAD

| Commit | Message |
|---|---|
| `4434609` | memory: add docs/PROJECT_MEMORY.md (canonical session-starting state file) and AGENTS.md pointer |
| `624fb11` | AGENTS.md: add Phase 4 plan section (I1+I2+I3, awaiting approval) |
| `26ac928` | Phase 4 plan: I1 nearest-residual + I2 isolated adv + I3 SRVGG body |
| `ad14781` | Phase 3.F: AGENTS.md lessons learned + v3 result doc |
| `d62b524` | Phase 3.B/C/D: warm-start v3 trained, D.2 gate fails, ship animevideov3 baseline instead |
| `21151e7` | Phase 3.A: PatchGAN + edge loss + RealESRTeacher + per-epoch ckpt rotation |
| `cccc0b5` | Phase 3 handoff: structured resume doc for adversarial student work |

Last updated: 2026-09-03 (memory file created).

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

### Phase 4 — PLANNED, NOT STARTED (2026-09-03)
- Three sequential experiments (I1 → I2 → I3) to break the RFDN+anneal ceiling.
- I1: nearest-residual RFDN (~1 hour).
- I2: isolated adversarial with --feat-weight flag (~1.5 hours).
- I3: SRVGG-body student architecture (~3 hours).
- Full plan: `docs/plans/student_phase4_nearest_adv_srvgg_plan.md` (391 lines).
- **AWAITING USER APPROVAL.** No code modified yet.

---

## 4. Pending tasks

### Blocked on user approval

- [ ] **Phase 4.I1** — Nearest-residual RFDN. Code change: ~10 LOC (`student.py` + `distill.py`). Train 40 ep from scratch (~17 min). Success: D.2 PASS (lap_var >= 35), PSNR >= 29.0. Output dir: `runs/distill_i1_nearest_residual/`.
- [ ] **Phase 4.I2** — Isolated adversarial. Code change: ~5 LOC (`--feat-weight` flag). Two sub-runs: I2a (animevideov3 + adv), I2b (span + adv + **feat_weight=0**). Success: I2b >= 29.5 dB. Output dirs: `runs/distill_i2a_*/`, `runs/distill_i2b_*/`.
- [ ] **Phase 4.I3** — SRVGG-body student. Code change: ~80 LOC (new `TinySRVGGStudent` + `--arch` flag + `archs.py` wire-up). Smoke test first. Success: D.2 PASS, PSNR >= 28.0, inference <= 200 ms/frame. Output dir: `runs/distill_i3_srvgg_body/`.
- [ ] Per-phase documentation: result docs (~80-100 lines each) + AGENTS.md updates.

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

---

## 6. Decision log

### 2026-09-03 — Phase 4 plan committed but NOT executing
- Decision: Wait for user approval before modifying code.
- Reason: Phase 3 already attempted 4 variations and all plateaued at lap_var 22-23. The user requested a plan, not unilateral execution.
- Reversal cost: low (plan is reversible via git checkout).

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

---

## 7. Halt conditions / guardrails

### Training halt conditions (apply to any new training run)

- **H1 (collapse)**: Val PSNR < 20 dB at any epoch → mode collapse → halt.
- **H2 (slow)**: Val PSNR < 26 dB after 10 epochs → slow convergence → halt.
- **H3 (plateau)**: EMA PSNR not improving for 10 consecutive epochs → halt.
- **H4 (lap regression)**: Full-frame lap_var < v1 baseline (21.0) → regression → halt.
- **H5 (D2 fail)**: Final lap_var < 35 → D.2 gate fail → don't ship.

### Promotion gate (D.2)
- Ship new checkpoint iff full-frame lap_var >= 35 AND PSNR >= baseline AND inference latency <= 200 ms/frame.
- PSNR is informational, not gating.
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
- `anime_upscaler/student.py:100-101` — bicubic residual shortcut (change target for Phase 4.I1).
- `anime_upscaler/student.py:64` — RFDN.__init__ args.
- `anime_upscaler/student.py:117` — set_shortcut_weight method.
- `anime_upscaler/distill.py:368` — `RFDN(scale=args.scale)` build call.
- `anime_upscaler/distill.py:528-542` — feature distillation loss application.
- `anime_upscaler/distill.py:638-639` — `_shortcut_weight` anneal call.
- `apps/anime_upscaler_gui/anime_upscaler_gui/archs.py` — arch dispatch (wire-up site for Phase 4.I3).
- `apps/anime_upscaler_gui/anime_upscaler_gui/registry.py::PRESET_CATALOG` — GUI preset catalog.

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
