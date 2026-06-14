# Preprocessing Pipeline Fix Plan — Path A (Implement All 5 Modes)

## V&V Review Source

Review based on `.windsurf/workflows/Verification & Validation System Prompt.md`  
16 issues found, classified: 5 Critical, 5 High, 4 Medium, 2 Low.

## Decision: Path A

Per user direction, all 5 modes (`on_the_fly`, `precomputed`, `gpu_degradation`, `hybrid`, `quality_adaptive`) will be **fully implemented and wired** into the training pipeline. A new shared `DegradationPipeline` module will be the single source of truth for the degradation logic, used by both `BaseDataset` and `PreprocessingManager`.

## Total Effort: 4-6 person-days

## Strategy

1. **Phase 1** (~1.5h): Unblock documented workflows, fix CI-red test.
2. **Phase 2** (~1-2d): Implement all 5 modes properly via shared `DegradationPipeline`.
3. **Phase 3** (~1h): Config & default path cleanup.
4. **Phase 4** (~1d): Reproducibility, VideoDataset parity, style cleanup.
5. **Phase 5** (~0.5d): Tests, docs, final verification.

---

## Phase 1 — Unblock Documented Workflows (≈ 1.5 hours)

> **Goal:** Make CI green and the documented commands in AGENTS.md work.
> **Checkpoint:** `pytest tests/test_preprocessing.py -v` → 15/15 pass; CLI documented commands run.

### T01 — Fix CLI flag dashes
- **File:** `scripts/precompute_pairs.py:88-91`
- Add `dest=` aliases so both `--estimate-only`/`--estimate_only` and `--available-space`/`--available_space` work.
- **Verify:** `python scripts/precompute_pairs.py --estimate-only --all_datasets --crop_size 128` succeeds.

### T02 — Convert float→uint8 in `_process_quality_adaptive`
- **File:** `src/data/preprocessing_manager.py:421-449`
- After `np.transpose`, convert to uint8 before calling `analyze_image`:
  ```python
  hr_uint8 = (hr_np * 255.0).clip(0, 255).astype(np.uint8)
  quality_score = self.quality_analyzer.analyze_image(hr_uint8)
  ```
- **Verify:** `PreprocessingManager({'mode': 'quality_adaptive'}).process(torch.rand(1,3,256,256))` returns tensor.

### T03 — Branch pre-flight error message by mode
- **File:** `src/data/preprocessing_manager.py:182-194`
- Branch message on `self.mode` so `mode='precomputed'` mentions `precomputed.base_dir`, not `hybrid.precomputed_base`.
- **Verify:** `precomputed` mode with no dir gives a clear, mode-correct error.

### T04 — Add `enabled` flag to `precomputed:` config; loosen pre-flight
- **File:** `src/data/preprocessing_manager.py:160-194`
- Add `pc_config.get('enabled', True)` check; default `False` for `mode='hybrid'` (backward compat with current behavior where `hybrid.precomputed_base` controls it).
- **Verify:** `test_modes` test passes; constructing any mode never raises spuriously.

### T05 — Add 6 missing exports to `data/__init__.py`
- **File:** `src/data/__init__.py:1-132`
- Add to all 3 import-fallback blocks + `__all__`:
  - `PreprocessingManager`, `OnTheFlyProcessor`
  - `PrecomputedDataset`
  - `DatasetSampler`, `create_sampler_from_config`
  - `LineEnhancer`
  - `estimate_dataset_storage`, `estimate_multiple_datasets`, `check_against_available_space`, `print_storage_estimate`, `create_default_dataset_configs`
- **Verify:** `python -c "from data import PreprocessingManager, PrecomputedDataset, DatasetSampler, LineEnhancer, estimate_dataset_storage"` succeeds.

### T06 — Phase 1 verification gate
- `pytest tests/test_preprocessing.py -v` → 15/15 pass.
- `python scripts/precompute_pairs.py --estimate-only --all_datasets` → clean output.
- Smoke: `python -c "from data.preprocessing_manager import PreprocessingManager; m = PreprocessingManager({'mode':'quality_adaptive','scale':4}); import torch; m.process(torch.rand(1,3,64,64))"` → tensor output.

---

## Phase 2 — Implement All 5 Modes (≈ 1-2 days)

> **Goal:** All 5 modes actually run end-to-end. Single `DegradationPipeline` class is the source of truth.
> **Checkpoint:** Every mode in `PreprocessingManager.MODES` produces correct LR output; `BaseDataset` invokes the manager for every mode.

### T09a — Create `DegradationPipeline`; relocate logic
- **New file:** `src/data/degradation_pipeline.py`
- **Source:** `BaseDataset._apply_degradation_full_image` (line 613), `_apply_two_stage_degradation` (line 574), `_apply_shuffled_degradation` (line 480ish), `_apply_blur`, `_apply_noise`, `_apply_jpeg`.
- **Design:**
  ```python
  class DegradationPipeline:
      """Single source of truth for image degradation. Reusable as a callable.

      Modes:
        - 'bicubic'        : pure bicubic downsample
        - 'light'/'medium'/'heavy'/'anime'/'anime_heavy' : preset configs
        - 'two_stage'      : APISR-style two-stage compression
        - 'shuffled'       : randomize blur/noise/resize/compression order
        - 'degrade_first'  : degrade full image, then crop
      """
      def __init__(self, mode: str = 'anime_heavy', cfg: Optional[Dict] = None,
                   scale: int = 4, two_stage: bool = False,
                   compression_stage1=None, compression_stage2=None,
                   rng: Optional[np.random.Generator] = None):
          ...

      def __call__(self, img: np.ndarray) -> np.ndarray:
          """Apply degradation; return LR image (uint8 RGB)."""
          ...
  ```
- **Refactor `BaseDataset`:** Replace `_apply_degradation_full_image`, `_apply_two_stage_degradation`, `_apply_shuffled_degradation` with calls to `DegradationPipeline`. Keep behavior identical (no regression).
- **Verify:** `pytest tests/ -v` → no regressions. Run 2-epoch smoke.

### T09b — Refactor `PreprocessingManager` to use `DegradationPipeline`
- **File:** `src/data/preprocessing_manager.py:287-292, 421-461`
- Replace `_process_on_the_fly` body with a call to `DegradationPipeline(mode=..., cfg=...)`.
- Implement `_process_quality_adaptive` properly:
  - Analyze image (already fixed in T02 to be uint8).
  - Determine tier (light/medium/heavy) using `quality_analyzer.get_degradation_tier(quality_score)`.
  - Look up tier-specific cfg from `quality_adaptive` config (`light_degradation`, `medium_degradation`, `heavy_degradation`).
  - Build `DegradationPipeline` with that cfg and apply.
- Implement `_process_gpu_degradation` properly:
  - Use `self.gpu_modules` (already initialized in `_init_gpu_degradation_handler`).
  - Apply each enabled module with its probability.
  - Use `torch.Generator` for reproducibility (per AGENTS.md gotcha about `np.random` in `nn.Module.forward`).
- **Verify:** Each of `on_the_fly`, `gpu_degradation`, `quality_adaptive` modes returns different LR outputs for the same HR (i.e., degradation actually applied, not just bicubic).

### T09c — Wire all 5 modes into `BaseDataset.__getitem__`
- **File:** `src/data/base.py:740-784`
- Replace the current narrow `if pm_mode == 'hybrid' and pm_available` check with a full dispatch:
  ```python
  if self.preprocessing_manager is not None:
      pm = self.preprocessing_manager
      if pm.mode == 'precomputed':
          lr_tensor, hr_tensor = pm.process(idx=idx)  # returns (lr, hr)
      elif pm.mode in ('on_the_fly', 'gpu_degradation', 'quality_adaptive'):
          lr_tensor = pm.process(hr_tensor)
          hr_tensor = torch.from_numpy(hr.transpose(2,0,1)).float() / 255.0
      elif pm.mode == 'hybrid':
          hr, lr = self._apply_hybrid_preprocessing(hr, idx)
  else:
      hr, lr = self._apply_standard_degradation(hr)
  ```
- For `precomputed` mode, `BaseDataset.__getitem__` should skip the HR-load from disk and just return what the manager gives.
- **Verify:** Setting each mode in v4 config + dry-run training loop succeeds.

### T09d — Add per-mode tests
- **File:** `tests/test_preprocessing.py`
- New test class `TestPreprocessingManagerModes`:
  - `test_on_the_fly_degrades` — assert LR != bicubic of HR (degradation actually applied).
  - `test_gpu_degradation_runs` — assert no exception + LR shape correct.
  - `test_quality_adaptive_selects_tier` — synthesize low-quality vs high-quality HR; assert tier differs.
  - `test_hybrid_runs` — already covered; expand to assert runtime augmentation runs.
  - `test_precomputed_returns_pair` — create temp dir, write 1 PNG to `lr/` and `hr/`; assert `process(idx=0)` returns both.

### T10 — Phase 2 verification gate
- `pytest tests/ -v` → all pass.
- Smoke: each of 5 modes works on a 64x64 dummy tensor.
- `python scripts/precompute_pairs.py --hr_dir data/anime_video_frames --estimate-only --crop_size 32` → succeeds.

---

## Phase 3 — Config & Default Cleanup (≈ 1 hour)

> **Goal:** Default configs and CLI defaults point to directories that actually exist.
> **Checkpoint:** `precompute_pairs.py --all_datasets` finds all real datasets.

### T11 — Fix `create_default_dataset_configs()` paths
- **File:** `src/data/storage_estimator.py:320-355`
- `data/anime_hr_1` → `data/anime_hr`; `data/val_hr_1` → `data/val_hr 1` (with space). Add `data/anime_video_frames 2` if it should be included.
- **Verify:** `precompute_pairs.py --estimate-only --all_datasets` lists all real datasets.

### T12 — Fix v4 `dataset_sampling` schema
- **File:** `configs/finetune_neosr_span_v4.yaml:76-78`
- Convert `1` → `1.0` (int → float to match `base.yaml` schema).
- **Verify:** `python -c "import yaml; print(yaml.safe_load(open('configs/finetune_neosr_span_v4.yaml'))['data']['dataset_sampling'])"` shows floats.

### T13 — Phase 3 verification gate
- `precompute_pairs.py --estimate-only --all_datasets --available-space 30` → report lists all real datasets.

---

## Phase 4 — Architecture & Integration (≈ 1 day)

> **Goal:** Reproducibility, VideoDataset parity, style cleanup, fix silent errors.

### T14 — Add `seed` parameter to `storage_estimator`
- **File:** `src/data/storage_estimator.py:63-104`
- Add `seed: int = 42` parameter; use `rng = np.random.default_rng(seed)` and `rng.choice(...)` instead of `np.random.choice`.
- Add `--seed` CLI flag to `precompute_pairs.py`.
- **Verify:** Two calls with same `seed=42` return identical output; `np.random.seed(...)` no longer required.

### T15 — Add `preprocessing_manager` + `sample_ratio` + `line_enhancement` to `VideoDataset`
- **File:** `src/data/video_dataset.py:__init__`
- Add the 6 missing parameters; pass them to the underlying frame dataset.
- **Verify:** `VideoDataset(video_dir='data/anime_vid', preprocessing_manager=pm).__getitem__(0)` uses the manager.

### T16 — Normalize PIL import style repo-wide
- **File:** `src/data/preprocessing_manager.py:203, 279, 306, 367` (uses `import PIL.Image`)
- Replace with `from PIL import Image`; replace `PIL.Image.open` with `Image.open`.
- Apply to any other files with the same style.
- **Verify:** `grep -r "import PIL" src/` returns 0 matches.

### T17 — Log silent exception in `_load_meta`
- **File:** `src/data/precomputed_dataset.py:51-58`
- Replace bare `except Exception: self.meta = {}` with logged version.
- **Verify:** Pass a corrupt `meta.yaml` and confirm log message appears.

### T18 — Phase 4 verification gate
- `pytest tests/ -v` → no regressions.
- `grep -r "import PIL\b" src/` → 0 matches.
- Smoke: corrupt `meta.yaml` produces warning, not silent failure.

---

## Phase 5 — Tests, Docs, Final Verification (≈ 0.5 day)

> **Goal:** Lock the fixes in with tests and update docs.

### T19 — Add regression test for all 5 modes
- **File:** `tests/test_preprocessing.py`
- Test that constructs `PreprocessingManager` for every mode and calls `process(...)` on a dummy tensor. Asserts no exception, output shape is `(1, 3, 16, 16)`.

### T20 — Add subprocess test for `--estimate-only --all_datasets` CLI
- **File:** `tests/test_preprocessing.py`
- Subprocess test: `subprocess.run(['python', 'scripts/precompute_pairs.py', '--estimate-only', '--all_datasets', '--crop_size', '32'], check=True)` exits 0.

### T21 — Add determinism test for `storage_estimator`
- **File:** `tests/test_preprocessing.py`
- Two calls with same `seed=42` return identical results.

### T22 — Update AGENTS.md
- **File:** `AGENTS.md`
- Document CLI flag style (dashes preferred).
- Update "Training Modes" section to reflect 5 fully-supported modes.
- Note: Path A implementation: `DegradationPipeline` is the single source of truth.

### T23 — Update config headers
- **Files:** `configs/base.yaml`, `configs/finetune_neosr_span_v4.yaml`, `configs/finetune_neosr_span_v5*.yaml`
- Add header comment block summarizing the 5 supported modes and their config keys.

### T24 — 2-epoch smoke test
- Run: `python scripts/train.py --config configs/finetune_neosr_span_v4.yaml --epochs 2`
- **Verify:** No crash, loss decreases.

### T25 — Final verification gate (pre-completion checklist)
- [ ] `pytest tests/ -v` → 100% pass
- [ ] `python scripts/precompute_pairs.py --estimate-only --all_datasets` → succeeds
- [ ] `python scripts/precompute_pairs.py --hr_dir data/anime_hr --estimate-only --available-space 5` → succeeds
- [ ] `python -c "from data import PreprocessingManager, PrecomputedDataset, DatasetSampler, LineEnhancer, estimate_dataset_storage"` → succeeds
- [ ] All 5 `PreprocessingManager.MODES` round-trip a dummy tensor
- [ ] `grep -r "import PIL\b" src/` → 0 matches
- [ ] 2-epoch smoke passes
- [ ] AGENTS.md + config headers updated
- [ ] No dead code (OnTheFlyProcessor removed or wired)
- [ ] No silent `except Exception` (Issue #15 fixed)

---

## Dependencies Graph

```
T01, T02, T05 ──► T06 ──┐
T03, T04 ──► T06       │
                       ├──► T09a ──► T09b ──► T09c ──► T09d ──► T10 ──┐
T06 ──► T11, T12 ──► T13                                              ├──► T18 ──► T19-T25
                       (parallel)                                    │
T13 ──► T14, T15, T16, T17 ──► T18                                   │
                                                                      │
```

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| `DegradationPipeline` refactor changes training behavior | Run 2-epoch smoke after T09a; full pytest after T10 |
| `VideoDataset` change breaks video tests | Run `tests/test_video_processing.py` after T15 |
| `np.random` change in `storage_estimator` alters estimates | Snapshot baseline before T14; compare within 1% |
| New `__init__.py` exports cause circular imports | Verify with `python -c "import data"` after T05 |
| Removing dead `OnTheFlyProcessor` breaks an importer | `grep -r OnTheFlyProcessor src/ scripts/ configs/ tests/` first |

---

## What This Plan Does NOT Touch

- `src/training/*` (losses, optimizers, trainer) — orthogonal
- `src/models/*` (model architectures) — orthogonal
- `src/inference/*` — orthogonal
- GPU-side inference & quality metric tooling
- Pre-existing tests for the v4 finetune (unchanged behavior preserved)
