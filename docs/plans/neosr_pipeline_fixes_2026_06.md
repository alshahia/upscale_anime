# neosr Pipeline V&V Fixes — Implementation Plan

**Date:** 2026-06-02
**Branch:** `fix/neosr-pipeline-vv-2026-06`
**Source review:** `docs/reviews/neosr_vv_report_2026_06.md` (23 issues: 4 Critical, 4 High, 6 Medium, 9 Low)
**Workflow:** `.windsurf/workflows/Verification & Validation System Prompt.md`

---

## Locked-in decisions

| # | Decision | Rationale |
|---|---|---|
| Q1 | Four atomic PRs for Critical fixes | Easier to bisect, smaller diffs, focused reviews |
| Q2 | Add D/G methods to RelativisticGANLoss, deprecate `forward()` | Backward-compatible |
| Q3 | Remove `label_smoothing` config + attribute | v4 uses relativistic GAN; label smoothing is N/A |
| Q4 | Document-only for weight precedence | Zero code change, no breakage |
| Q5 | Discriminator stays in FP32, no scaler | Single scaler for generator + content losses |
| Q6 | `Conv3XC.eval_conv` → `nn.Parameter(requires_grad=False)` | State-dict format unchanged |
| Q7 | Update both AGENTS.md and V&V workflow doc | Future agents benefit from new patterns |

---

## Files to be touched

| File | Issues | Risk |
|---|---|---|
| `src/training/neosr_finetuner.py` | #2, #4, #5, #7, #8, #9, #11, #12, #13, #15, #16, #21, #22 | High (1435-line core) |
| `src/losses/fdl_loss.py` | #1 | Low |
| `src/losses/adversarial_loss.py` | #3 | Medium |
| `src/models/span/neosr_span.py` | #6, #17 | Medium |
| `src/losses/wavelet_guided_loss.py` | #23 | Low |
| `configs/finetune_neosr_span_v4.yaml` | #7, #10, #20, #23 | Low |
| `configs/base.yaml` | #18 | Low |
| `src/data/preprocessing_manager.py` | #20 | Low |
| **NEW** `tests/test_neosr_finetuner.py` | #14 + regressions | New file |
| `AGENTS.md` | New patterns documented | Low |
| `.windsurf/workflows/Verification & Validation System Prompt.md` | MP checklist | Low |

---

## Phase 0 — Pre-work (no code changes)

1. Create branch `fix/neosr-pipeline-vv-2026-06` (DONE).
2. Capture baseline test pass/fail: `pytest tests/test_two_phase_training.py tests/test_perceptual_loss.py tests/test_new_losses.py tests/test_critical_fixes.py -v` → `tmp/baseline_tests.txt`.
3. Capture baseline dry-run: `python scripts/train.py --config configs/finetune_neosr_span_v4.yaml --dry-run` → `tmp/baseline_dryrun.txt`.
4. (Optional) Capture 1-epoch loss curve → `tmp/baseline_curve.csv`.

---

## Phase 1 — CRITICAL fixes (four atomic PRs)

### PR-1: FDL gradient flow
- **File:** `src/losses/fdl_loss.py`
- **Change:** Remove `torch.no_grad():` wrapper around `self.feature_extractor.get_intermediate_layers(...)` in `extract_features` (lines 73-75). DINOv2 params are already frozen via `param.requires_grad = False` (lines 51-52).
- **Tests:** `test_fdl_gradient_flows_to_input`, `test_fdl_dino_params_remain_frozen`.
- **Verify:** `pytest tests/test_neosr_finetuner.py -k fdl -v`

### PR-2: GradScaler for AMP
- **File:** `src/training/neosr_finetuner.py`
- **Changes:** Import `GradScaler` with PyTorch<2 fallback; `self.scaler = GradScaler('cuda') if use_amp else None`; wrap backward; unscale-before-clip; `scaler.step/update`; save/load `scaler_state_dict`. Discriminator path untouched.
- **Tests:** `test_finetuner_creates_grad_scaler_when_amp_on`, `test_finetuner_no_grad_scaler_when_amp_off`, `test_save_load_includes_scaler_state`, `test_load_old_checkpoint_without_scaler_state`.
- **Verify:** `pytest tests/test_neosr_finetuner.py -k scaler -v`

### PR-3: RelativisticGAN D/G asymmetry
- **File:** `src/losses/adversarial_loss.py` + `src/training/neosr_finetuner.py`
- **Change:** Add `RelativisticGANLoss.discriminator_loss()` and `.generator_loss()` using `binary_cross_entropy_with_logits` with cross-terms. Mark old `.forward()` as deprecated (warn once). Update finetuner D-step and G-step to call new methods.
- **Tests:** `test_relativistic_d_g_are_different`, `test_relativistic_d_minimizes_when_separated`, `test_relativistic_g_minimizes_when_overlapping`, `test_vanilla_gan_unchanged`.
- **Verify:** `pytest tests/test_neosr_finetuner.py -k gan -v`

### PR-4: Safe checkpoint load
- **File:** `src/training/neosr_finetuner.py:load_checkpoint`
- **Change:** Replace direct `torch.load(weights_only=False)` with try-`weights_only=True`-then-fallback, gated by `security.allow_pickle_checkpoint` config flag. Reuse `UnpicklingError` from existing import.
- **Tests:** `test_load_checkpoint_safe_when_weights_only_ok`, `test_load_checkpoint_blocks_pickle_without_opt_in`, `test_load_checkpoint_allows_pickle_with_opt_in`.
- **Verify:** `pytest tests/test_neosr_finetuner.py -k checkpoint_safe -v`

### Phase-1 exit gate
- All four task PRs merged on the branch.
- Full test suite green: `pytest tests/ -x --ignore=tests/test_video_processing.py`.
- 2-epoch smoke train on tiny dataset completes without NaN-sanitizer firing.

---

## Phase 2 — HIGH (one PR)

- 2.1 `max(1, epochs - phase1_epochs)` guard in `_get_phase_weights`; clip `phase2_progress` to [0,1].
- 2.2 `Conv3XC._fused` flag + `train()` override to invalidate cache.
- 2.3 Delete `self.use_label_smoothing` attribute and config key.
- 2.4 `_read_config_early()` extracted in `__init__` for `adversarial_start_epoch`, `phase1_epochs`, etc.

---

## Phase 3 — MEDIUM (one PR)

- 3.1 Replace 8 `except Exception as e: print(...)` blocks with `logger.warning(...)`.
- 3.2 Add "ignored when two_phase_training.enabled" comments to v4 config.
- 3.3 `_compute_total_loss(..., include_adversarial=False)`; `validate()` passes `False`.
- 3.4 FDL skipped batches: log cached value only, do NOT add to `total_loss`.
- 3.5 NaN sanitizer: log first 5 offender param names per epoch.
- 3.6 New `tests/test_neosr_finetuner.py` with `minimal_config`, `dummy_dataloader`, `tmp_checkpoint` fixtures.

---

## Phase 4 — LOW (one PR, batched)

- 4.1 Use or remove `from pickle import UnpicklingError`.
- 4.2 `Conv3XC.eval_conv` → `nn.Parameter(requires_grad=False)`.
- 4.3 Move `progressive_crop` defaults to `base.yaml`.
- 4.4 Hoist lazy loss imports to top of file.
- 4.5 `PreprocessingManager` pre-flight: raise `FileNotFoundError` if `hybrid.precomputed_base: true` and `base_dir` missing.
- 4.6 Read `gradient_penalty_lambda` from config (default 1.0).
- 4.7 Merge `_get_adversarial_weight` + `_compute_adversarial_weight_phase_aware`.
- 4.8 Rename `wavelet_init` → `start_epoch`; keep old name as deprecated alias.

---

## Phase 5 — Docs & final

- 5.1 AGENTS.md new subsections: GradScaler required, FDL needs gradient flow, RelativisticGAN has D/G methods, Conv3XC cache invalidation rule.
- 5.2 V&V workflow doc: new "Mixed-Precision Training Checklist" section.
- 5.3 v4 config header comment: re-tuning advisory after FDL+GAN real-gradients fix.
- 5.4 Final gate: `py_compile` clean, full pytest, `--dry-run` OK, 2-epoch smoke (FDL decreases, PSNR increases, sanitizer 0 fires), `profile_gpu.py` shows ≥20% eval speedup.

---

## Master TODO (40 items)

See `docs/plans/todos_neosr_fixes_2026_06.md`.

---

## Risk matrix per PR

| PR | Backward-compat risk | Migration | Training regression risk |
|---|---|---|---|
| PR-1 FDL | Low (only affects backward graph; identical numerics) | None | Low (FDL now actually trains — desired) |
| PR-2 Scaler | Medium (old ckpts lack scaler state) | Graceful warn + init fresh | Low (NaN sanitizer stays as defense) |
| PR-3 GAN | Low (forward() still works) | Deprecation warn | Medium (D/G may need re-balancing) |
| PR-4 Checkpoint | Low (default allow_pickle=False) | Opt-in flag | None (load-only path) |

---

## Backward-compatibility statement

- All existing checkpoints (pre-fix) continue to load.
- All existing configs (including `finetune_neosr_span_v4.yaml`) continue to work without changes.
- New behavior is opt-in or naturally backward-compatible.
- `use_label_smoothing` is the only removed config key (it was a no-op).

---

## Re-tuning advisory

After PR-1 + PR-3 land, FDL and adversarial losses start contributing real gradients. The v4 config's loss weights may produce different dynamics on first run. Recommended starting point: halve `fdl.weight` and `loss.adversarial.weight` on first run, then re-tune.
