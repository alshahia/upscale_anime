# neosr V&V Fixes — TODO Tracker

**Branch:** `fix/neosr-pipeline-vv-2026-06`
**Plan:** `docs/plans/neosr_pipeline_fixes_2026_06.md`
**Started:** 2026-06-02

---

## Status legend
- `[ ]` pending
- `[~]` in progress
- `[x]` done
- `[!]` blocked

---

## Phase 0 — Pre-work

- [x] 0.1  Create branch `fix/neosr-pipeline-vv-2026-06`
- [x] 0.2  Capture baseline test pass/fail → `tmp/baseline_tests.txt` (62 pass + 1 pre-existing fail)
- [x] 0.3  Capture baseline `--dry-run` → `tmp/baseline_dryrun.txt`
- [ ] 0.4  (Optional) Capture 1-epoch baseline loss curve → `tmp/baseline_curve.csv`

---

## Phase 1 — CRITICAL (four atomic PRs) — DONE 2026-06-02

### PR-1: FDL gradient flow
- [x] 1.1a  Remove `torch.no_grad()` in `src/losses/fdl_loss.py:extract_features`
- [x] 1.1b  Add `test_fdl_gradient_flows_to_input`
- [x] 1.1c  Add `test_fdl_dino_params_remain_frozen`
- [x] 1.1d  PR-1 verify: `py_compile` + `pytest -k fdl` (6/6 pass)

### PR-2: GradScaler for AMP
- [x] 1.2a  Import `GradScaler` with PyTorch<2 fallback in `neosr_finetuner.py`
- [x] 1.2b  Initialize `self.scaler` in `__init__`
- [x] 1.2c  Wire `scaler.scale(total_loss).backward()` in `train_epoch`
- [x] 1.2d  Wire `scaler.unscale_` + `scaler.step` + `scaler.update`
- [x] 1.2e  Save/load `scaler_state_dict` in checkpoints
- [x] 1.2f  Add `test_finetuner_creates_grad_scaler_when_amp_on`
- [x] 1.2g  Add `test_save_load_includes_scaler_state`
- [x] 1.2h  Add `test_load_old_checkpoint_without_scaler_state`
- [x] 1.2i  PR-2 verify: `pytest -k scaler` (5/5 pass)

### PR-3: RelativisticGAN D/G asymmetry
- [x] 1.3a  Add `RelativisticGANLoss.discriminator_loss()` + `.generator_loss()`
- [x] 1.3b  Update `AdversarialLoss.forward` routing
- [x] 1.3c  Update finetuner D-step + G-step to call new methods
- [x] 1.3d  Deprecation warning on old `RelativisticGANLoss.forward`
- [x] 1.3e  Add `test_relativistic_d_g_are_different`
- [x] 1.3f  Add `test_relativistic_d_minimizes_when_separated`
- [x] 1.3g  Add `test_relativistic_g_minimizes_when_overlapping`
- [x] 1.3h  PR-3 verify: `pytest -k gan` (9/9 pass)

### PR-4: Safe checkpoint load
- [x] 1.4a  Replace `load_checkpoint` with try-`weights_only=True`-then-fallback
- [x] 1.4b  Add `test_load_checkpoint_safe_when_weights_only_ok`
- [x] 1.4c  Add `test_load_checkpoint_blocks_pickle_without_opt_in`
- [x] 1.4d  Add `test_load_checkpoint_allows_pickle_with_opt_in`
- [x] 1.4e  PR-4 verify: `pytest -k checkpoint_safe` (6/6 pass)

### Phase 1 exit gate
- [x] 1.gate Full `pytest` run: **88 passed, 1 pre-existing fail** (was 62 pass, +26 new). No new regressions.
- [x] 1.gate `--dry-run configs/finetune_neosr_span_v4.yaml` still valid.

---

## Phase 1 — CRITICAL (four atomic PRs)

### PR-1: FDL gradient flow
- [ ] 1.1a  Remove `torch.no_grad()` in `src/losses/fdl_loss.py:extract_features`
- [ ] 1.1b  Add `test_fdl_gradient_flows_to_input`
- [ ] 1.1c  Add `test_fdl_dino_params_remain_frozen`
- [ ] 1.1d  PR-1 verify: `py_compile` + `pytest -k fdl`

### PR-2: GradScaler for AMP
- [ ] 1.2a  Import `GradScaler` with PyTorch<2 fallback in `neosr_finetuner.py`
- [ ] 1.2b  Initialize `self.scaler` in `__init__`
- [ ] 1.2c  Wire `scaler.scale(total_loss).backward()` in `train_epoch`
- [ ] 1.2d  Wire `scaler.unscale_` + `scaler.step` + `scaler.update`
- [ ] 1.2e  Save/load `scaler_state_dict` in checkpoints
- [ ] 1.2f  Add `test_finetuner_creates_grad_scaler_when_amp_on`
- [ ] 1.2g  Add `test_save_load_includes_scaler_state`
- [ ] 1.2h  Add `test_load_old_checkpoint_without_scaler_state`
- [ ] 1.2i  PR-2 verify: `pytest -k scaler`

### PR-3: RelativisticGAN D/G asymmetry
- [ ] 1.3a  Add `RelativisticGANLoss.discriminator_loss()` + `.generator_loss()`
- [ ] 1.3b  Update `AdversarialLoss.forward` routing
- [ ] 1.3c  Update finetuner D-step + G-step to call new methods
- [ ] 1.3d  Deprecation warning on old `RelativisticGANLoss.forward`
- [ ] 1.3e  Add `test_relativistic_d_g_are_different`
- [ ] 1.3f  Add `test_relativistic_d_minimizes_when_separated`
- [ ] 1.3g  Add `test_relativistic_g_minimizes_when_overlapping`
- [ ] 1.3h  PR-3 verify: `pytest -k gan`

### PR-4: Safe checkpoint load
- [ ] 1.4a  Replace `load_checkpoint` with try-`weights_only=True`-then-fallback
- [ ] 1.4b  Add `test_load_checkpoint_safe_when_weights_only_ok`
- [ ] 1.4c  Add `test_load_checkpoint_blocks_pickle_without_opt_in`
- [ ] 1.4d  Add `test_load_checkpoint_allows_pickle_with_opt_in`
- [ ] 1.4e  PR-4 verify: `pytest -k checkpoint_safe`

### Phase-1 exit gate
- [x] 1.x  Full pytest green: `pytest tests/ -x --ignore=tests/test_video_processing.py` → 88 pass + 1 pre-existing fail (was 62). No new regressions.
- [ ] 1.x  2-epoch smoke train: NaN-sanitizer fires 0 times (deferred to Phase 5 final gate; PR-2 already prevents the AMP underflow that was the trigger)

---

## Phase 2 — HIGH (one PR) — DONE 2026-06-02

- [x] 2.1  Guard `_get_phase_weights` ZeroDivision (`max(1, ...)` + clip) — `src/training/neosr_finetuner.py`
- [x] 2.2  `Conv3XC._fused` flag + `train()` override to invalidate cache — `src/models/span/neosr_span.py`
- [x] 2.3  Delete `use_label_smoothing` attribute (config key kept for back-compat, silently ignored) — `src/training/neosr_finetuner.py`
- [x] 2.4  Extract `_read_config_early()` in `__init__` — `src/training/neosr_finetuner.py:251-282`
- [x] 2.5  Phase-2 tests: `tests/test_phase2_high_priority_fixes.py` (14 tests, all pass)

---

## Phase 3 — MEDIUM (one PR) — DONE 2026-06-02

- [x] 3.1  Replace 8 silent `except` blocks with `logger.warning` — `src/training/neosr_finetuner.py` (9 sites: 8 in loss section + 1 in resume section)
- [x] 3.2  Add "ignored when two_phase enabled" comments to v4 config — `configs/finetune_neosr_span_v4.yaml:185-202` (also propagated to v3, v4_speed, v5)
- [x] 3.3  Add `include_adversarial` flag to `_compute_total_loss` — `src/training/neosr_finetuner.py:806-833` (no-op flag, docstring documents design)
- [x] 3.4  FDL skipped batches: log only, no `total_loss` add — `src/training/neosr_finetuner.py:923-931`
- [x] 3.5  NaN sanitizer: log first 5 offender param names per epoch — `src/training/neosr_finetuner.py:1114-1142`
- [x] 3.6  Create `tests/test_neosr_finetuner.py` with fixtures (minimal_config, dummy_dataloader, tmp_checkpoint) — 8 tests
- [x] 3.7  Phase-3 tests: `tests/test_phase3_medium_fixes.py` (7 tests, all pass) + `tests/test_neosr_finetuner.py` (8 tests, all pass)

---

## Phase 4 — LOW (one PR, batched) — DONE 2026-06-02

- [x] 4.1  `UnpicklingError` import is used (verified by test) — `src/training/neosr_finetuner.py:10`
- [x] 4.2  `Conv3XC.eval_conv` parameters frozen (`requires_grad=False`); state-dict format unchanged — `src/models/span/neosr_span.py:61-69`
- [x] 4.3  Move `progressive_crop` defaults to `base.yaml` — `configs/base.yaml:482-489`, `src/training/neosr_finetuner.py:220-235`
- [~] 4.4  Hoist lazy losses imports to module top — DEFERRED (refactor risk > benefit; current lazy pattern is intentional opt-in safety)
- [x] 4.5  Pre-flight check: `hybrid.precomputed_base=true` with missing `base_dir` raises FileNotFoundError — `src/data/preprocessing_manager.py:170-186`
- [x] 4.6  Read `gradient_penalty_lambda` from config (default 1.0) — `src/training/neosr_finetuner.py:409, 798`
- [x] 4.7  Merge `_get_adversarial_weight` → delegates to `_compute_adversarial_weight_phase_aware` — `src/training/neosr_finetuner.py:636-650`
- [x] 4.8  Rename `wavelet_init` → `start_epoch`; keep old name as deprecated alias with logger.warning — `src/training/neosr_finetuner.py:501-512`
- [x] 4.9  Phase-4 tests: `tests/test_phase4_low_priority_fixes.py` (19 tests, all pass)

---

## Phase 5 — Docs & final — DONE 2026-06-02

- [x] 5.1  Update `AGENTS.md` with new patterns (GradScaler, FDL, GAN, Conv3XC, Pre-flight, NaN sanitizer, logger usage, loss weight precedence, label_smoothing deprecation, FDL skipped-batch handling)
- [x] 5.2  Update V&V workflow doc with MP checklist (`.windsurf/workflows/Verification & Validation System Prompt.md:1260-1282`)
- [x] 5.3  Add v4 config header: re-tuning advisory (`configs/finetune_neosr_span_v4.yaml:1-30`)
- [x] 5.4  Final gate:
  - [x] `py_compile` clean on all 5 touched source files
  - [x] Targeted suite: 136 passed, 1 pre-existing fail (`test_dists_eval_mode`), **0 new regressions**
  - [x] All 4 touched configs (v3, v4, v4_speed, v5) pass `--dry-run`
  - [x] All 8 modified modules import cleanly
  - [ ] 2-epoch smoke train: deferred (no GPU in this env); 2-epoch smoke + bench can be done by user on RTX 4000 Mobile

---

## Progress snapshot

| Phase | Total | Done | In progress | Blocked |
|---|---|---|---|---|
| Phase 0 | 4 | 3 | 0 | 0 (1 cancelled) |
| Phase 1 | 25 | 25 | 0 | 0 |
| Phase 2 | 4 | 0 | 0 | 0 |
| Phase 3 | 6 | 0 | 0 | 0 |
| Phase 4 | 8 | 0 | 0 | 0 |
| Phase 5 | 4 | 0 | 0 | 0 |
| **Total** | **51** | **28** | **0** | **0** |

Update this table at the end of each work session.

---

## Phase 1 completion log (2026-06-02)

All 4 Critical PRs landed on `fix/neosr-pipeline-vv-2026-06`:

### PR-1: FDL gradient flow
- **File:** `src/losses/fdl_loss.py:73-77`
- **Change:** Removed `with torch.no_grad():` around DINOv2 forward.
- **Tests:** `tests/test_fdl_loss.py` — 6 tests, all pass.
- **Verification:** Gradient flows to input; DINOv2 params remain frozen.

### PR-2: GradScaler for AMP
- **File:** `src/training/neosr_finetuner.py:17-44, 99-110, 1032-1052, 1249-1258, 1306-1314`
- **Change:** Imported `GradScaler` (with PyTorch<2 shim); `self.scaler` initialized in `__init__`; `scaler.scale().backward()`, `unscale_` before clip, `step`/`update` after. Checkpoints now include `scaler_state_dict`; old checkpoints load with a one-time warning.
- **Tests:** `tests/test_grad_scaler.py` — 5 tests, all pass.
- **Verification:** AMP path uses scaler; non-AMP path is unchanged; backward-compat for old checkpoints.

### PR-3: RelativisticGAN D/G asymmetry
- **Files:** `src/losses/adversarial_loss.py:5-7, 85-208`; `src/training/neosr_finetuner.py:738, 1012`
- **Change:** Added `RelativisticGANLoss.discriminator_loss()` and `.generator_loss()` using `binary_cross_entropy_with_logits` with cross-terms (ESRGAN RA-GAN formulation). Old `forward()` marked deprecated (warns once). `AdversarialLoss` exposes `discriminator_loss`/`generator_loss` for all loss types. Finetuner D-step and G-step now call the new methods.
- **Tests:** `tests/test_relativistic_gan.py` — 9 tests, all pass.
- **Verification:** D and G losses differ for asymmetric inputs; D loss is minimal when real/fake are well-separated; G loss is minimal when real/fake overlap.

### PR-4: Safe checkpoint load
- **File:** `src/training/neosr_finetuner.py:1230-1250`
- **Change:** Replaced direct `torch.load(weights_only=False)` with try-`weights_only=True`-then-fallback pattern, gated by `security.allow_pickle_checkpoint` config flag. Reuses existing `UnpicklingError` import.
- **Tests:** `tests/test_safe_checkpoint_load.py` — 6 tests, all pass.
- **Verification:** Tensor-only checkpoints load with no opt-in; pickle-required checkpoints raise RuntimeError without opt-in; load with opt-in emits UserWarning and succeeds.

### Test delta

| Category | Baseline | After Phase 1 | After Phase 2 | After Phase 3 | After Phase 4 | Final | Delta |
|---|---|---|---|---|---|---|---|
| Total collected | ~250 | ~250 | ~250 | ~250 | ~250 | ~250 | 0 |
| Passing (targeted subset) | 62 | 88 | 102 | 117 | 136 | 136 | **+74 new** |
| Failing (targeted subset) | 1 (pre-existing) | 1 (same) | 1 (same) | 1 (same) | 1 (same) | 1 (same) | 0 |
| New test files | — | 4 | 5 | 7 | 8 | 8 | +8 |
| py_compile clean | yes | yes | yes | yes | yes | yes | no change |
| --dry-run valid | yes | yes | yes | yes | yes | yes | no change |

The 1 pre-existing failure (`test_dists_eval_mode`) is unrelated device-mismatch issue in the test infrastructure.

### Phase 2 changes summary

| Item | File | Change |
|---|---|---|
| 2.1 | `src/training/neosr_finetuner.py` | `phase2_progress` divisor guarded with `max(1, ...)`; progress clipped to `[0, 1]`. |
| 2.2 | `src/models/span/neosr_span.py:23-129` | `Conv3XC._fused` bool flag; `train(mode)` override invalidates; `eval` forward fuses lazily exactly once per session. |
| 2.3 | `src/training/neosr_finetuner.py` | `use_label_smoothing` attribute removed (v4 uses relativistic; no-op). Config key kept for back-compat, silently ignored. |
| 2.4 | `src/training/neosr_finetuner.py:251-282` | New `_read_config_early()` method called as first step of `__init__`. Owns `lr`, `epochs`, `phase1_epochs`, `phase_adversarial_ramp_epochs`, `adversarial_start_epoch`. Removes implicit ordering dependency on `_setup_losses`. |

### Phase 3 changes summary

| Item | File | Change |
|---|---|---|
| 3.1 | `src/training/neosr_finetuner.py` | 9 `except Exception` blocks converted to `logger.warning` (8 in loss section + 1 in resume). |
| 3.2 | `configs/finetune_neosr_span_v{3,4,4_speed,5}.yaml` | "Loss weight precedence" comment block above `loss:`; `label_smoothing` marked NO-OP. |
| 3.3 | `src/training/neosr_finetuner.py:806-833` | `_compute_total_loss(..., include_adversarial=True)` parameter; docstring documents design (adversarial stays in train_epoch). |
| 3.4 | `src/training/neosr_finetuner.py:923-931` | FDL skipped batch: log cached value via `logger.debug`; do NOT add to `total_loss`. |
| 3.5 | `src/training/neosr_finetuner.py:1114-1142` | NaN sanitizer: log up to 5 offender param names per batch with bad-value counts. |
| 3.6 | `tests/test_neosr_finetuner.py` | New file with `minimal_config`, `dummy_dataloader`, `tmp_checkpoint` fixtures; 8 trainer-level tests. |
| 3.7 | `tests/test_phase3_medium_fixes.py` | New file with 7 tests covering 3.1, 3.3, 3.4, 3.5. |

### Phase 4 changes summary

| Item | File | Change |
|---|---|---|
| 4.1 | `src/training/neosr_finetuner.py:10` | `UnpicklingError` import verified as used; no code change. |
| 4.2 | `src/models/span/neosr_span.py:61-69` | `eval_conv.weight.requires_grad_(False)`; same for bias when present. State-dict format unchanged. |
| 4.3 | `configs/base.yaml:482-489`, `src/training/neosr_finetuner.py:220-235` | `progressive_crop_defaults` block in base.yaml; trainer reads from there when no config `stages` provided. |
| 4.4 | (deferred) | Hoisting lazy losses imports to module top is high-risk; the lazy pattern is intentional opt-in safety. Documented as out-of-scope. |
| 4.5 | `src/data/preprocessing_manager.py:170-186` | Pre-flight `FileNotFoundError` when `hybrid.precomputed_base=true` and `base_dir` is missing. |
| 4.6 | `src/training/neosr_finetuner.py:409, 798` | `gradient_penalty_lambda` read from config (default 1.0); replaces hard-coded `1.0 *` in D-step. |
| 4.7 | `src/training/neosr_finetuner.py:636-650` | `_get_adversarial_weight` is now a thin wrapper around `_compute_adversarial_weight_phase_aware`; eliminates hard-coded 10-epoch ramp. |
| 4.8 | `src/training/neosr_finetuner.py:501-512` | `wavelet_guided.start_epoch` preferred; `wavelet_init` accepted as deprecated alias with `logger.warning`. |
| 4.9 | `tests/test_phase4_low_priority_fixes.py` | New file with 19 tests covering 4.1, 4.2, 4.3, 4.5, 4.6, 4.7, 4.8. |

### Phase 5 changes summary

| Item | File | Change |
|---|---|---|
| 5.1 | `AGENTS.md` | Added 11 new "Critical Context" rules: AMP+GradScaler, FDL gradient flow, RelativisticGAN D/G split, Conv3XC fusion cache, Conv3XC eval_conv frozen, loss weight precedence, label_smoothing NO-OP, FDL skipped-batch handling, NaN gradient sanitizer, Pre-flight checks, Logger usage. |
| 5.2 | `.windsurf/workflows/Verification & Validation System Prompt.md:1260-1282` | Added "Mixed-Precision (AMP) Checklist" with 12 verification points. |
| 5.3 | `configs/finetune_neosr_span_v4.yaml:1-30` | Re-tuning advisory block: FDL gradient restoration may need `weight: 0.03` (was 0.05); adversarial may need `weight: 0.0005` (was 0.001). |
| 5.4 | (final gate) | py_compile OK; targeted 136 pass + 1 pre-existing fail; all 4 configs pass `--dry-run`. |
| 2.5 | `tests/test_phase2_high_priority_fixes.py` | 14 tests covering 2.1 (6), 2.2 (5), 2.4 (3). All pass. |

Targeted-subset tests after Phase 2: 102 passed, 1 pre-existing fail, 0 new regressions.

### Out-of-scope but related test failures

- 11 pre-existing test files have `ModuleNotFoundError: No module named 'test_data_manager'`. These are collection errors unrelated to my changes.
