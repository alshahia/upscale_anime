# Preprocessing Pipeline Fix — Task List (Path A)

Source plan: `PREPROCESSING_PIPELINE_FIX_PLAN.md`  
Started: 2026-06-02  
Status: **IN PROGRESS**

## Quick status legend
- [ ] = pending
- [~] = in progress
- [x] = complete
- [!] = blocked

## Phase 1 — Unblock Documented Workflows (≈ 1.5 hours)  ✅ COMPLETE

- [x] **T01** — Fix CLI flag dashes in `scripts/precompute_pairs.py:88-91`
  - Added `dest=` aliases so `--estimate-only`/`--available-space` (dashes) work alongside underscore variants.
  - **Verify:** `python scripts/precompute_pairs.py --estimate-only --all_datasets --crop_size 32` prints report. ✅

- [x] **T02** — Convert float→uint8 in `_process_quality_adaptive`
  - **File:** `src/data/preprocessing_manager.py:421-449`
  - Converted `hr_np` to uint8 before `analyze_image` call.
  - **Verify:** `PreprocessingManager({'mode': 'quality_adaptive'}).process(torch.rand(1,3,256,256))` returns tensor. ✅

- [x] **T03** — Branch pre-flight error message by mode
  - **File:** `src/data/preprocessing_manager.py:182-220`
  - Different message for `mode='precomputed'` vs `mode='hybrid'`.
  - **Verify:** `precomputed` mode with no dir gives a clear, mode-correct error. ✅
  - Added regression test: `test_precomputed_mode_raises_when_enabled_and_missing`.

- [x] **T04** — Add `enabled` flag to `precomputed:` config; loosen pre-flight
  - **File:** `src/data/preprocessing_manager.py:160-220`
  - Default `pc_config.get('enabled', True)`; for `hybrid` mode honor `hybrid.precomputed_base` only.
  - **Verify:** `test_modes` test passes (now uses `precomputed.enabled: false` opt-out). ✅

- [x] **T05** — Add 6 missing exports to `data/__init__.py`
  - **File:** `src/data/__init__.py:1-180`
  - Exported `PreprocessingManager`, `OnTheFlyProcessor`, `PrecomputedDataset`, `DatasetSampler`, `create_sampler_from_config`, `LineEnhancer`, `estimate_dataset_storage`, `estimate_multiple_datasets`, `check_against_available_space`, `print_storage_estimate`, `create_default_dataset_configs`.
  - **Verify:** `from data import PreprocessingManager, ...` succeeds. ✅

- [x] **T06** — Phase 1 verification gate ✅
  - `pytest tests/test_preprocessing.py -v` → **16/16 pass** (added 1 regression test).
  - `precompute_pairs.py --estimate-only --all_datasets` runs. ✅
  - Smoke: `PreprocessingManager({'mode':'quality_adaptive'}).process(...)` returns tensor. ✅
  - Full unit test suite: 378 pass, 14 pre-existing failures (test env issues, unrelated).

## Phase 2 — Implement All 5 Modes (≈ 1-2 days)

- [ ] **T09a** — Create `DegradationPipeline`; relocate logic
  - **New file:** `src/data/degradation_pipeline.py`
  - Relocate `BaseDataset._apply_degradation_full_image`, `_apply_two_stage_degradation`, `_apply_shuffled_degradation`, `_apply_blur`, `_apply_noise`, `_apply_jpeg` into a single `DegradationPipeline` class.
  - **Verify:** `pytest tests/ -v` → no regressions; 2-epoch smoke passes.

- [ ] **T09b** — Refactor `PreprocessingManager` to use `DegradationPipeline`
  - **File:** `src/data/preprocessing_manager.py:287-292, 421-461`
  - `_process_on_the_fly` → use `DegradationPipeline`.
  - `_process_quality_adaptive` → analyze + tier + `DegradationPipeline`.
  - `_process_gpu_degradation` → use `self.gpu_modules` with `torch.Generator` (not `np.random`).
  - **Verify:** Each mode returns different LR outputs (degradation actually applied).

- [ ] **T09c** — Wire all 5 modes into `BaseDataset.__getitem__`
  - **File:** `src/data/base.py:740-784`
  - Replace narrow `if pm_mode == 'hybrid'` check with full dispatch.
  - **Verify:** Setting each mode in v4 config + dry-run training loop succeeds.

- [ ] **T09d** — Add per-mode tests
  - **File:** `tests/test_preprocessing.py`
  - New `TestPreprocessingManagerModes` class.
  - **Verify:** Tests pass.

- [ ] **T10** — Phase 2 verification gate
  - `pytest tests/ -v` → all pass.
  - Smoke: each of 5 modes works on 64x64 dummy tensor.
  - CLI: `precompute_pairs.py --hr_dir data/anime_video_frames --estimate-only --crop_size 32` succeeds.

## Phase 3 — Config & Default Cleanup (≈ 1 hour)

- [ ] **T11** — Fix `create_default_dataset_configs()` paths
  - **File:** `src/data/storage_estimator.py:320-355`
  - `data/anime_hr_1` → `data/anime_hr`; `data/val_hr_1` → `data/val_hr 1`.
  - **Verify:** `precompute_pairs.py --estimate-only --all_datasets` lists all real datasets.

- [ ] **T12** — Fix v4 `dataset_sampling` schema
  - **File:** `configs/finetune_neosr_span_v4.yaml:76-78`
  - Convert `1` → `1.0` (int → float).
  - **Verify:** Loaded YAML shows floats.

- [ ] **T13** — Phase 3 verification gate
  - `precompute_pairs.py --estimate-only --all_datasets --available-space 30` reports all real datasets.

## Phase 4 — Architecture & Integration (≈ 1 day)

- [ ] **T14** — Add `seed` parameter to `storage_estimator` + `--seed` CLI flag
  - **Files:** `src/data/storage_estimator.py:63-104`, `scripts/precompute_pairs.py`
  - Use `np.random.default_rng(seed)`.
  - **Verify:** Two calls with same seed return identical output.

- [ ] **T15** — Add `preprocessing_manager` + `sample_ratio` + `line_enhancement` to `VideoDataset`
  - **File:** `src/data/video_dataset.py:__init__`
  - Add the 6 missing parameters; pass to underlying frame dataset.
  - **Verify:** `VideoDataset(...).process(...)` works.

- [ ] **T16** — Normalize PIL import style repo-wide
  - **File:** `src/data/preprocessing_manager.py:203, 279, 306, 367`
  - Replace `import PIL.Image` with `from PIL import Image`.
  - **Verify:** `grep -r "import PIL\b" src/` → 0 matches.

- [ ] **T17** — Log silent exception in `_load_meta`
  - **File:** `src/data/precomputed_dataset.py:51-58`
  - Replace bare `except Exception: self.meta = {}` with logged version.
  - **Verify:** Corrupt `meta.yaml` produces warning, not silent failure.

- [ ] **T18** — Phase 4 verification gate
  - `pytest tests/ -v` → no regressions.
  - `grep -r "import PIL\b" src/` → 0 matches.

## Phase 5 — Tests, Docs, Final Verification (≈ 0.5 day)

- [ ] **T19** — Add regression test for all 5 modes
  - **File:** `tests/test_preprocessing.py`
  - **Verify:** Tests pass.

- [ ] **T20** — Add subprocess test for `--estimate-only --all_datasets`
  - **File:** `tests/test_preprocessing.py`
  - **Verify:** Subprocess exits 0.

- [ ] **T21** — Add determinism test for `storage_estimator`
  - **File:** `tests/test_preprocessing.py`
  - **Verify:** Same seed → same result.

- [ ] **T22** — Update AGENTS.md
  - Document CLI flag style (dashes preferred).
  - Update "Training Modes" section.

- [ ] **T23** — Update config headers
  - **Files:** `configs/base.yaml`, `configs/finetune_neosr_span_v4.yaml`, `configs/finetune_neosr_span_v5*.yaml`
  - Add header comment block summarizing 5 supported modes.

- [ ] **T24** — 2-epoch smoke test
  - `python scripts/train.py --config configs/finetune_neosr_span_v4.yaml --epochs 2`
  - **Verify:** No crash, loss decreases.

- [ ] **T25** — Final verification gate
  - [ ] `pytest tests/ -v` → 100% pass
  - [ ] `precompute_pairs.py --estimate-only --all_datasets` succeeds
  - [ ] `precompute_pairs.py --hr_dir data/anime_hr --estimate-only --available-space 5` succeeds
  - [ ] `from data import PreprocessingManager, PrecomputedDataset, DatasetSampler, LineEnhancer, estimate_dataset_storage` succeeds
  - [ ] All 5 `PreprocessingManager.MODES` round-trip a dummy tensor
  - [ ] `grep -r "import PIL\b" src/` → 0 matches
  - [ ] 2-epoch smoke passes
  - [ ] AGENTS.md + config headers updated
  - [ ] No dead code (OnTheFlyProcessor removed or wired)
  - [ ] No silent `except Exception` (Issue #15 fixed)
