# AGENTS.md — upscale_anime

## Project

PyTorch anime super-resolution with progressive ensemble distillation and neosr SPAN finetuning. Model paths:
- **Model A**: SPAN-Tiny + MTKD/FAKD distillation from EDSR/RCAN/SwinIR teachers
- **Model B**: Mamba-PAN with 4-direction scanning (requires `mamba-ssm`)
- **Finetune**: neosr SPAN with multi-loss (perceptual, GAN, frequency, FDL, wavelet-guided) training via `src/training/neosr_finetuner.py`

## Setup

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -r requirements.txt
```
# alwayes serach for local venv ( if exists) use it for all commands ( if not exist use the global python)

No `pyproject.toml`, `setup.py`, or lockfile. Dependencies managed via `requirements.txt` only.

## Key Commands

```bash
# Always use --config (required arg)
python scripts/train.py --config configs/model_a_ntire.yaml
python scripts/train.py --config configs/model_a_ntire.yaml --dry-run   # validate config only
python scripts/train.py --config configs/auto_stage_pipeline.yaml       # full automated pipeline

# Optimized finetune (v4 - APISR-style degradation, twin perceptual, FDL, wavelet-guided)
python scripts/train.py --config configs/finetune_neosr_span_v4.yaml
python scripts/train.py --config configs/finetune_neosr_span_v4.yaml --dry-run

# Download teacher models (auto-downloaded by default in Stage 1)
python scripts/download_models.py --model all --scale 4

# Run tests
pytest tests/

# Lint / format (from requirements.txt)
black .
flake8 .

# Inference
python scripts/inference.py --model-type neosr_span --checkpoint checkpoints/finetune_best.pth --input data/test/ --output results/
python scripts/inference.py --model-type neosr_span --checkpoint checkpoints/finetune_best.pth --input lr.jpg --output sr.png --gt ground_truth.png --metrics

# Quality evaluation
python scripts/evaluate_quality.py --input results/ --output quality_report.csv

# Dataset analysis
python scripts/analyze_dataset.py --dir data/anime_hr
```

Note: `.agentrc` references `ruff`, `bandit`, `mypy` — these are **not** in requirements.txt. Use `black` + `flake8`.

## Architecture

```
src/                  # All library code (bare imports: from models..., from training...)
  models/             # span/, mamba_pan/, teachers/ (edsr, rcan, swinir)
  training/           # orchestrator, trainers, callbacks, stage_controller
  distillation/       # mtkd/ (aggregation), fakd/ (affinity loss)
  losses/             # pixel, wavelet, perceptual, anime-specific
  data/               # dataloader, augmentation, video_extraction, anime_degradation
  inference/          # engine
  utils/              # config, metrics, pretrained_models, nan_handler
scripts/              # All entry points (CLI scripts)
configs/              # YAML configs; base.yaml is the root
tests/                # pytest tests (no conftest.py, no pytest.ini)
```

**Import convention**: scripts add `src` to `sys.path` then use bare imports (`from models.span...`). Never use `src.` prefix. Match existing files.

**Test coverage**: ~25-30% of `src/` modules have tests. No `conftest.py`, no `pytest.ini`. Many modules (orchestrator, ensemble_trainer, losses, utils) are untested. Don't assume test coverage is comprehensive.
- **Trainer test stub pattern**: For tests of `NeosrSPANFinetuner` that don't need the full `__init__` (which loads data, builds discriminator, etc.), bypass `__init__` with `NeosrSPANFinetuner.__new__(cls)` and inject only the attributes the test needs. Canonical example: `tests/test_grad_scaler.py::TestGradScalerInitialization`. Reuse this pattern; do NOT call full `__init__` in unit tests.

## Training Modes

`model_a` | `model_b` | `both_parallel` | `both_sequential` | `ensemble` | `full` | `auto_stage` | `finetune`

## Critical Gotchas

- **Quality Metrics**: Use LPIPS (not just PSNR/SSIM) to evaluate perceptual quality. LPIPS correlates with human judgment; lower is better. PSNR >25dB and SSIM >0.85 are good targets. See `src/utils/metrics.py` for available metrics.
- **No-Reference Evaluation**: For real-world images without GT, use `scripts/evaluate_quality.py` with NIQE/MANIQA/CLIPIQA. CLIPIQA performs best for anime content (used by APISR/CVPR 2024).
- **Two-Phase Training**: The finetune mode uses two-phase training by default (Phase 1: epochs 1-30 for structure, Phase 2: epochs 31-50 for texture). This balances pixel accuracy with perceptual quality.
- **Perceptual Loss Types**: DISTS for texture preservation, Dual (VGG+ResNet) for structural features. Use `type: dists` or `type: dual` in config. Both are supported.
- **Line Art Edge Detection**: Multi-scale edge detection (`edge_method: multi_scale`) captures both thick character outlines and thin detail lines. Use this for anime content.
- **Progressive Crop Training**: During finetune, you can use progressive crop to gradually increase patch size for better feature learning at different scales. Set `progressive_crop.enabled: true` with stages defining crop sizes and epoch counts. Example: epochs 1-15 use crop=128, 16-50 use crop=256, 51-80 use crop=512. Crop changes are logged but dataloader restart is needed for full effect. See `configs/finetune_neosr_span_v3.yaml` for config schema.
- **Stage 1 NaN losses**: Knowledge Aggregation can produce NaN/Inf with small batches or multiple teachers. Use `use_simple_aggregation: true` or single-teacher mode. See `src/utils/nan_handler.py`.
- **Discriminator in GAN training**: Discriminator and DISTS must run in FP32 under mixed precision. See "Mixed precision FP32 override" below.
- **Defensive config access**: Use `config.get('key') or {}` not `config.get('key', {})` — values can be explicitly `None`.
- **PyTorch 2.x API**: Use `torch.amp.autocast('cuda', ...)` and `GradScaler('cuda')`, not the deprecated `torch.cuda.amp` variants.
- **Mixed precision FP32 override**: DISTS perceptual loss, frequency loss, and discriminator MUST run in FP32 under autocast. Use `torch.amp.autocast('cuda', enabled=False)` with `.float()` inputs. FP16 causes silent NaN in these operations. See `src/training/neosr_finetuner.py`.
- **AMP + GradScaler wiring**: The finetuner instantiates `self.scaler = GradScaler('cuda')` when AMP is enabled. `scaler.scale(loss).backward()`, `scaler.unscale_(optimizer)` before `clip_grad_norm_`, and `scaler.step(optimizer); scaler.update()` after. The scaler state is included in checkpoints via `scaler_state_dict`; old checkpoints load with a one-time warning. See `src/training/neosr_finetuner.py:99-110, 1032-1052, 1249-1258, 1306-1314`.
- **FDL gradient flow**: The FDL (Frequency Distribution Loss) is wrapped in a `torch.amp.autocast('cuda', enabled=False)` block to run in FP32, and `torch.no_grad()` is INTENTIONALLY absent so gradients flow back to the generator. DINOv2 parameters are frozen separately at `fdl_loss.py:51-52` so only the input grads propagate. See `src/losses/fdl_loss.py:73-77` and `tests/test_fdl_loss.py`.
- **RelativisticGAN D/G split**: `RelativisticGANLoss` exposes separate `discriminator_loss(real, fake)` and `generator_loss(real, fake)` methods (ESRGAN RA-GAN formulation with cross-terms). The old `forward(...)` is deprecated. The finetuner calls the new methods in its D-step and G-step respectively. `AdversarialLoss` wraps both modes (relativistic, vanilla). See `src/losses/adversarial_loss.py:5-7, 85-208` and `tests/test_relativistic_gan.py`.
- **Conv3XC fusion cache**: The `Conv3XC` block fuses its 1x1 + 3x3 + 1x1 paths into a single 3x3 `eval_conv` at the first eval forward. The result is cached in the `_fused` flag (not a Parameter/buffer — does not appear in state_dict). The `train(mode)` override invalidates the cache so train→eval switches re-fuse with current weights. See `src/models/span/neosr_span.py:23-129` and `tests/test_phase2_high_priority_fixes.py::TestConv3XCFusionCaching`.
- **Conv3XC.eval_conv parameters are frozen**: `eval_conv.weight` and `eval_conv.bias` are set to `requires_grad=False` at construction time. They are populated by `update_params()` which writes to `.data` (no gradient tracking) and must never be touched by the optimizer. State-dict format is unchanged (still stores `eval_conv.weight` and `eval_conv.bias` keys).
- **Loss weight precedence**: When `two_phase_training.enabled: true`, the per-phase `phase1.<component>_weight` and `phase2.<component>_weight` values TAKE PRECEDENCE over `loss.<component>.weight`. The trainer linearly interpolates between phase1 and phase2 weights during phase 2. The `loss.*.weight` values act as a FALLBACK for single-phase training. See `configs/finetune_neosr_span_v4.yaml:185-202` for the precedence rule comment block.
- **`loss.adversarial.label_smoothing` is a NO-OP**: v4 uses `RelativisticGANLoss` (binary cross-entropy with cross-terms), which is not compatible with binary target smoothing. The key is silently ignored for backward compat; a config comment in v4 marks it.
- **FDL skipped-batch handling**: When `fdl.compute_every > 1`, batches where `batch_idx % compute_every != 0` log a `logger.debug` entry with the cached value and add NOTHING to `total_loss`. The cached value has `requires_grad=False` so this is a no-op for gradients; not adding it to `total_loss` keeps the displayed loss honest. See `src/training/neosr_finetuner.py:923-931`.
- **Loss accumulator tensor guard**: The loss-accumulator loops in `train_epoch` and `validate` iterate `for key, value in loss_dict.items(): value.item()`. Adding non-tensor values (Python floats, ints, strs) to `loss_dict` will crash with `AttributeError: 'float' object has no attribute 'item'`. When adding scalar/cache/log values, use a separate field name (e.g. `fdl_cached`, not `fdl`) AND make the accumulator skip them: `if not isinstance(value, torch.Tensor): continue`. See `src/training/neosr_finetuner.py:1169-1175, 1225-1233` and `tests/test_phase3_medium_fixes.py::TestLossAccumulatorToleratesFloat`.
- **NaN gradient sanitizer**: The trainer scans all `model.named_parameters()` after backward, replaces NaN/Inf gradients with zero, and logs up to 5 offender param names per batch (with their bad-value counts) via `logger.warning`. This makes debugging much faster than the previous count-only message.
- **Pre-flight checks**: `PreprocessingManager` raises `FileNotFoundError` at init time when `hybrid.precomputed_base: true` and the configured `base_dir` is missing. This catches typos and forgotten `precompute_pairs.py` runs at startup. See `src/data/preprocessing_manager.py:170-186` and `tests/test_phase4_low_priority_fixes.py::TestPreprocessingManagerPreflight`.
- **Windows terminal**: Avoid unicode in print statements (`✓`, `✗`, emojis). Use ASCII (`[OK]`, `[ERROR]`). cp1252 encoding breaks on chars above U+00FF.
- **Platform-dependent features**: Before enabling features like `torch.compile()`, `triton`, or hardware-accelerated codecs, verify availability on the current platform. Windows lacks Triton support; use feature flags with safe defaults (`false` on Windows, `true` on Linux).
- **Python indentation**: When inserting code into existing Python files, read at least 10 lines of surrounding context to match the exact indentation level (tabs vs spaces, depth). After editing, verify with `python -m py_compile <file>`.
- **Optional deps**: `tensorboard`, `mamba-ssm`, `matplotlib` use try/except import with availability flags. Gracefully degrade.
- **Device placement**: Loss modules with pre-trained extractors (VGG, DISTS) need `.to(device)` — buffers (mean/std) must follow.
- **VRAM target**: 8GB (RTX 4000 Mobile). Keep batch sizes modest.
- **Checkpoint compatibility**: Multiple `checkpoint_compatible_model*.py` variants exist in `src/models/span/`. When loading fails, analyze checkpoint keys first, then pick the matching loader. Don't guess — inspect with `torch.load(path).keys()`.
- **Pre-flight before training**: Before running long training jobs, verify: config is valid YAML (`--dry-run`), data paths exist, no unicode in print statements, GPU is available (`torch.cuda.is_available()`).
- **Multi-dataset tensor shapes**: When combining datasets, crop sizes must match for batching. Check `src/data/dataloader.py` for standardization logic.
- **torch.load security**: Always use `weights_only=True` with fallback to `weights_only=False`. Pattern: `try: torch.load(path, weights_only=True) except (UnpicklingError, RuntimeError): torch.load(path, weights_only=False)`. The fallback is gated by `config.security.allow_pickle_checkpoint: true`; without the opt-in, the load fails. See `src/training/neosr_finetuner.py:1230-1250` and `tests/test_safe_checkpoint_load.py`.
- **Config schema consistency**: All configs must match `base.yaml` schema. Duplicate top-level keys (e.g., `training:` appearing 3x) silently merge unpredictably. Non-standard keys (`optimizer.type`, `datasets.train`) won't work with the standard config loader. Validate with `--dry-run`.
- **Silent exception handling**: Avoid bare `except Exception:` without logging. The codebase has 18+ unlogged exception handlers. Always log the exception: `except Exception as e: logger.warning(f"...: {e}")`.
- **np.random reproducibility**: `np.random` calls in `nn.Module.forward()` (e.g., `src/data/anime_degradation.py`) break `torch.manual_seed()` reproducibility. Use `torch.rand()` or `torch.randint()` with a `torch.Generator` instead.
- **New dependencies**: LPIPS and pyiqa are required for quality evaluation. Install with: `pip install lpips pyiqa`
- **Logger usage**: `src/training/neosr_finetuner.py` defines `logger = logging.getLogger(__name__)` at module level. Use `logger.warning(...)` for recoverable errors (loss loading failures, NaN sanitization, deprecated config keys) and `logger.debug(...)` for verbose diagnostics (FDL skipped batches). Reserve `print(...)` for user-facing progress lines (loss weights, epoch summaries).
- **V4 enhancements**: `configs/finetune_neosr_span_v4.yaml` includes APISR-style degradation (WebP/AVIF/H.264 compression, shuffled resize), twin perceptual loss (VGG19+ResNet50), FDL loss (DINOv2), wavelet-guided GAN, and XDoG line enhancement. Target scores: NIQE <5.0, MANIQA >0.5, CLIPIQA >0.6.

## Pre-Flight Checklist (run before every training)

Run these 6 checks before any training run. Each takes < 30 seconds and catches a class of bugs that py_compile + pytest + --dry-run miss.

1. **GPU + VRAM**: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_properties(0).total_memory/1e9)"`. Training runs on GPU; CPU fallback is for testing only.

2. **Config paths**: When adding a CLI flag that calls `config.set('a.b.c', value)`, the path MUST match the trainer's `self.config.get('a.b.c')` path. `grep -n "self.config.get.*<key>"` to confirm. Path mismatches cause silent-fallback bugs that only surface at runtime.

3. **Batch size vs VRAM**: With v4 (twin VGG+ResNet + FDL DINOv2 + discriminator) at scale=4, crop=128, expect ~0.4GB/sample. On 8GB GPU: batch_size <= 14. On 16GB: <= 28. If `training.finetune.batch_size` is too high, override with `--batch-size N`.

4. **Loss dict types**: Any new key in `loss_dict` must be a `torch.Tensor` (or guarded against in the accumulator loop). See "Loss accumulator tensor guard" above.

5. **Checkpoint I/O**: If resuming from a checkpoint saved before 2026-06-02 (Phase 1 PR-4), add `--allow-pickle`. The first save under the new code path will be a clean tensor-only checkpoint.

6. **2-epoch smoke**: Before committing or handing off any change to a config, loss function, or training-loop branch, run: `python scripts/train.py --config <cfg> --epochs 2`. Crashes that look-good-in-tests only show up here. This is REQUIRED, not optional. ~2-5 min.

## V4 Quality Improvements

The v4 config (`finetune_neosr_span_v4.yaml`) incorporates techniques from APISR (CVPR 2024) and state-of-the-art SR research:

### Degradation Model
- **Two-stage compression**: Stage 1 (JPEG+WebP before resize), Stage 2 (AVIF+H.264 after resize)
- **Shuffled resize**: Random order of degradation operations (blur, noise, resize, compression)
- **Degrade-before-crop**: Apply degradation on full image before cropping (APISR approach)

### Loss Functions
- **Twin Perceptual Loss**: VGG19 (ImageNet) + ResNet50 with balanced scaling (`L_per = L_ResNet + delta * L_VGG`)
- **FDL Loss**: Frequency Distribution Loss with DINOv2 features and sliced Wasserstein distance
- **Wavelet-Guided Loss**: Haar wavelet decomposition for GAN stability (HH subband weighted 2x)
- **XDoG Line Enhancement**: Pseudo-GT with sharpened hand-drawn lines

### GAN Training
- **Higher adversarial weight**: 0.001 (Phase 2), max 0.005 (vs 0.0001/0.0005 in v3)
- **Higher discriminator LR**: 0.0001 (vs 0.000005 in v3)
- **Longer training**: 150 epochs (vs 80 in v3), Phase 1 extended to 50 epochs

## V7 Anime SOTA Roadmap (2026-06)

The v7 config (`finetune_neosr_span_v7_anime.yaml`) is the project's
current "best" anime SR config. It synthesizes 6 research-driven phases
documented in `docs/plans/v7_anime_roadmap.md`. As of 2026-06-14, the
**config and code are merged, but no v7 finetune has been trained** (the
80-100 epoch run is best left to the user's GPU).

### v7 quality targets (APISR-aligned)

| Metric | Target | Direction | Source |
|---|---:|:---:|---|
| CLIPIQA | >= 0.65 | higher | APISR (CVPR 2024) Table 1 |
| MANIQA  | >= 0.48 | higher | APISR Table 1 |
| NIQE    | <= 7.5  | lower  | APISR Table 1 |
| LPIPS   | <= 0.10 | lower  | "perceptual good" empirical |
| TOPIQ-NR| >= 0.50 | higher | local convention |

These are guidance, not pass/fail. The local `val_hr` is sourced from
the same `mp4upload` collection as `data/anime_hr`, so FR metrics on
`val_hr` are **optimistic** by ~1-3 dB PSNR. For publishable numbers,
evaluate on `data/anime_hr_holdout/` (curated disjoint set; see below).

### Honest held-out test set: `data/anime_hr_holdout/`

`val_hr` has known train/val overlap (both are sourced from the same
`mp4upload` collection). For honest v6/v7 comparison:

1. Hand-pick 30-50 Danbooru frames that are NOT in `data/anime_hr`
   or `data/val_hr`. Verify by file-hash dedup.
2. Save them to `data/anime_hr_holdout/`.
3. Re-run the comparison with
   `--input data/anime_hr_holdout --gt data/anime_hr_holdout`.
4. The NR-IQA scores (CLIPIQA, MANIQA, NIQE, TOPIQ) on this disjoint
   set are the **publishable** numbers.

### Phase F eval harness

`scripts/compare_checkpoints.py` (added 2026-06-14) is the side-by-side
checkpoint comparison harness. It:

- Loads two (or one) checkpoints via the same model-config YAML.
- Computes FR metrics (PSNR/SSIM/LPIPS) when `--gt` is provided.
- Computes NR metrics (CLIPIQA/MANIQA/NIQE/TOPIQ/MUSIQ) regardless.
- Writes per-image CSV + summary JSON, prints a mean +/- std table.
- Supports `--smoke N` (fast CI) and `--max-side N` (avoid 4x-upscale
  blowing up to 7680x4320 from a 1920x1080 source).
- Is runnable with one checkpoint (omit `--v7`) for baseline-only mode.

Baseline NR-IQA numbers (N=10, 480x480 center-crop) are reported in
`docs/v7_results.md` Section 3. The best local scores are: stock
pretrained CLIPIQA=0.6674, MANIQA=0.4218, NIQE=7.6808, TOPIQ=0.5558.

### User action items (after v7 training)

```bash
# 1. Train v7 (80-100 epochs; hours on RTX 4000 Mobile)
python scripts/train.py --config configs/finetune_neosr_span_v7_anime.yaml --epochs 80

# 2. Compare v6 vs v7
python scripts/compare_checkpoints.py --baseline checkpoints/NEOSR_SPAN_V6_ANIME/finetune_best.pth --v7 checkpoints/NEOSR_SPAN_V7_ANIME/finetune_best.pth --input data/val_hr --gt data/val_hr --output results/comparison_v6_vs_v7/ --metrics psnr ssim lpips clipiqa maniqa niqe topiq_nr

# 3. (Optional) Re-run on the disjoint held-out set
python scripts/compare_checkpoints.py --baseline checkpoints/NEOSR_SPAN_V6_ANIME/finetune_best.pth --v7 checkpoints/NEOSR_SPAN_V7_ANIME/finetune_best.pth --input data/anime_hr_holdout --gt data/anime_hr_holdout --output results/comparison_v6_vs_v7_holdout/ --metrics psnr ssim lpips clipiqa maniqa niqe topiq_nr

# 4. Update docs/v7_results.md Section 3 with the v7 column.
```

## V7 Post-Build Follow-Ons (Phase G)

The v7 build (`d0f7f23` on `fix/neosr-pipeline-vv-2026-06`) is **complete in code and tests** but the long finetune run is deferred to user GPU. Four follow-on items were identified after commit. See `docs/plans/todos_v7_anime_2026_06.md` for full list.

### Status (2026-06-14)

| Item | Status | Owner | Priority |
|---|---|---|---|
| G.1 Run v7 finetune (2-epoch smoke + 80-epoch full) | pending | user GPU | high |
| G.2 Add `apisr` named preset to degradation config | active (sub-agent) | sub-agent | medium |
| G.3 Curate `data/anime_hr_holdout/` (30-50 Danbooru frames) | pending | user | medium |
| G.4 Fix 15 pre-existing test collection errors | active (sub-agent) | sub-agent | medium |
| G.5 End-to-end MambaIRv2 integration test | active (sub-agent) | sub-agent | medium |
| G.6 Optional: request AVC-RealLQ access | pending | user (email) | low |
| G.7 Optional: MambaIRv2 GPU benchmark | pending | user GPU | low |

### v7 baseline measurements (already in `docs/v7_results.md` Section 3)

| Metric | Direction | Stock pretrained | v4 finetune best | HR ceiling | APISR target |
|---|:---:|---:|---:|---:|---:|
| CLIPIQA  | up   | **0.667** | 0.632 | 0.608 | >= 0.65 |
| MANIQA   | up   | **0.422** | 0.334 | 0.371 | >= 0.48 |
| NIQE     | down | 7.681     | **7.123** | 6.054 | <= 7.5  |
| TOPIQ_NR | up   | **0.556** | 0.462 | 0.558 | (n/a)   |

**MANIQA is the gap.** All other targets are already cleared by stock pretrained or v4 finetune. After v7 finetune, watch MANIQA most closely.

## V7 Smoke Testing

Before kicking off the 80-epoch v7 finetune on real GPU time, run the 2-epoch smoke variant end-to-end. It exercises every v7 feature (APISR two-stage degradation, XDoG/USM pseudo-GT, EMA, twin perceptual, FDL=0.5, MambaIRv2-ready `sab_type` factory) at minimal compute.

**Config**: `configs/finetune_neosr_span_v7_anime_smoke.yaml` — inherits from the real v7 config and overrides only the size knobs: `epochs=2`, `batch_size=4`, `crop_size=32`, `num_workers=2`, `run_name=NEOSR_SPAN_V7_ANIME_SMOKE` (so it cannot clobber the real v7 run). ~1.6 GB VRAM, 4-10 min on RTX 4000 Mobile.

**CI runner**: `scripts/run_v7_smoke.py` — runs the 11 v7-relevant test files in a single pytest invocation, skipping `@pytest.mark.slow` tests and the known-flaky `test_topiq_higher_for_cleaner_image`. ~55 s on CPU. Exits 0 on full pass, 1 on any failure.

**Meta-test**: `tests/test_v7_smoke_runner.py` — subprocess-invokes the smoke runner and asserts rc=0. Validates the CI pipeline itself.

```bash
# Validate the smoke config (no GPU needed)
python scripts/train.py --config configs/finetune_neosr_span_v7_anime_smoke.yaml --dry-run

# Run the 2-epoch smoke on your GPU
python scripts/train.py --config configs/finetune_neosr_span_v7_anime_smoke.yaml

# Run the CPU-friendly test subset (CI)
python scripts/run_v7_smoke.py
```

## Checkpoint Management

The training system includes automatic checkpoint management to prevent overwriting previous runs.

### Run Naming

| Method | Description |
|--------|-------------|
| **Auto (Hybrid)** | `run_{timestamp}_{model_name}_{suffix}` e.g., `run_20240115_143052_neosr_span_progCrop` |
| **Custom** | Use `--run-name "my_exp"` in CLI or `output.run_name` in config |

### Duplicate Handling

When a run name already exists, the system can:
- **`auto_increment`**: Automatically append `_001`, `_002`, etc.
- **`ask`**: Prompt user to choose (CLI mode)

Config: `output.duplicate_handling: auto_increment` or `ask`

### CLI Options

```bash
# Auto-generated name
python scripts/train.py --config configs/finetune_neosr_span_v3.yaml

# Custom name
python scripts/train.py --config configs/finetune_neosr_span_v3.yaml --run-name "exp_progCrop_v1"

# Override via config
# In config: output.run_name: "my_exp"
```

### Tracker CSV

All runs are logged to `checkpoints/run_tracker.csv`:

| Field | Description |
|-------|-------------|
| run_id | Unique ID (run_001, run_002, ...) |
| run_name | Directory name |
| config_file | Config used |
| timestamp | Start time |
| status | started/completed/interrupted |
| best_loss | Best validation loss |
| final_epoch | Epoch reached |
| checkpoint_dir | Full path |

### Config Value Restoration on Resume

When resuming from a checkpoint, training-critical config values that affect training behavior (not just weights) should be restored from the checkpoint:

- `phase1_epochs` / `phase2_epochs` for two-phase training
- `adversarial_start_epoch` for GAN training
- `progressive_crop` stages and current crop size

This ensures resumed training continues with the correct phase/loss schedule, even if the current config has different values than when training started.

**Example**: If checkpoint was saved at epoch 37 with `phase1_epochs=40`, resuming should use `phase1_epochs=40` (from checkpoint), not the current config's value.

### Testing Checkpoint Save/Load

When adding new training features that affect loss computation or phase determination, add tests that:

1. Save a checkpoint with modified config values
2. Load the checkpoint in a fresh instance with different config values
3. Verify the loaded values match the checkpoint, not the current config

**Example**: `test_phase_determination_with_restored_phase1_epochs()` in `tests/test_two_phase_training.py`

## Quality Evaluation

For super-resolution quality assessment, use these metrics:

| Metric | Type | Target | Direction |
|--------|------|--------|-----------|
| PSNR | Full-ref | >25 dB | higher is better |
| SSIM | Full-ref | >0.85 | higher is better |
| LPIPS | Full-ref | <0.25 | lower is better |
| NIQE | No-ref | <5.0 | lower is better |
| MANIQA | No-ref | >0.7 | higher is better |
| CLIPIQA | No-ref | >0.7 | higher is better |
| TOPIQ-NR | No-ref | >0.5 | higher is better |
| MUSIQ | No-ref | (n/a) | higher is better |

Command-line evaluation:
```bash
# With ground truth
python scripts/inference.py --gt ground_truth.png --metrics

# Batch no-reference evaluation
python scripts/evaluate_quality.py --input results/ --metrics niqe maniqa clipiqa topiq_nr --output report.csv

# Side-by-side checkpoint comparison (e.g. v6 vs v7 after v7 training)
python scripts/compare_checkpoints.py --baseline checkpoints/v6_best.pth --v7 checkpoints/v7_best.pth --input data/val_hr --gt data/val_hr --output results/comparison_v6_vs_v7/ --metrics psnr ssim lpips clipiqa maniqa niqe topiq_nr
```

## Data Flow

1. Place HR images in `data/` or videos in `data/anime_vid/`
2. Video frames extracted via `scripts/process_videos.py` (pre or on-demand mode)
3. RealESRGAN-style degradation generates LR pairs on-the-fly (configurable presets: light/medium/heavy/anime)
4. Teacher models auto-download to `pretrained/` if missing

## Preprocessing Pipeline (NEW - v5)

The project includes a unified preprocessing system with 5 configurable modes:

### Preprocessing Modes

| Mode | Description | Speed | Quality |
|------|-------------|-------|---------|
| `on_the_fly` | CPU-based degradation during training | Slow | High variance |
| `precomputed` | Load pre-generated LR/HR pairs from disk | Fastest | Fixed |
| `gpu_degradation` | GPU-accelerated degradation (anime modules) | Fast | Medium |
| `hybrid` | Precomputed base + runtime augmentation | Medium-Fast | High |
| `quality_adaptive` | Per-image quality-based degradation | Medium | Highest |

**Default mode**: `hybrid` (best speed/quality balance)

### Configuration

```yaml
data:
  preprocessing:
    mode: "hybrid"  # Select mode
    
    # Storage settings
    storage:
      available_space_gb: 10
      crop_size: 128
      format: "png"
    
    # Dataset sampling (values must be float in [0.0, 1.0])
    dataset_sampling:
      anime_video_frames: 0.3  # 30% of this dataset
      anime_hr: 0.5            # 50% of this dataset

    # Hybrid mode (default)
    hybrid:
      precomputed_base: true
      base_dir: "data/precomputed_4x"
      runtime_augmentation:
        enabled: true
        blur_prob: 0.3
        noise_sigma: [0, 15]
```

### Storage Estimation

Before precomputing, estimate storage requirements:

```bash
# Estimate all datasets (dashes, not underscores)
python scripts/precompute_pairs.py --estimate-only --all_datasets --crop-size 128 --available-space 10

# Estimate single dataset (deterministic via --seed)
python scripts/precompute_pairs.py --estimate-only --hr-dir data/anime_hr --crop-size 128 --seed 42
```

### Precomputation

```bash
# Precompute with size check first
python scripts/precompute_pairs.py --estimate-only --all_datasets --crop-size 128 --available-space 10

# Then precompute (adjust ratios if needed)
python scripts/precompute_pairs.py --all-datasets --crop-size 128 --format jpeg --workers 8
```

### New Components

| Component | File | Purpose |
|-----------|------|---------|
| `PreprocessingManager` | `src/data/preprocessing_manager.py` | Unified orchestrator for all 5 modes |
| `DegradationPipeline` | `src/data/degradation_pipeline.py` | Single source of truth for image degradation (presets, two-stage, shuffled) |
| `OnTheFlyProcessor` | `src/data/preprocessing_manager.py` | CPU on-the-fly degradation (re-exported for backward compat) |
| `PrecomputedDataset` | `src/data/precomputed_dataset.py` | Loads pre-generated LR/HR pairs |
| `DatasetSampler` | `src/data/dataset_sampler.py` | Global/per-dataset sampling control |
| `LineEnhancer` | `src/data/line_enhancement.py` | XDoG line enhancement (parity with `BaseDataset`) |
| `storage_estimator` | `src/data/storage_estimator.py` | Worst-case storage estimation |
| `precompute_pairs.py` | `scripts/precompute_pairs.py` | Pregeneration script with resume |

### DegradationPipeline (single source of truth)

All 5 modes use the same underlying `DegradationPipeline` class (`src/data/degradation_pipeline.py`).
This guarantees that `PreprocessingManager._process_on_the_fly`, `BaseDataset._apply_shuffled_degradation`,
and `_process_quality_adaptive` produce the same degradation behavior given the same config.

**Presets:**

| Preset | Use case |
|--------|----------|
| `bicubic` | Pure bicubic downsample (no degradation) |
| `light` | Mild blur + noise + JPEG |
| `medium` | Moderate blur + noise + JPEG |
| `heavy` | Strong blur + noise + JPEG |
| `anime` | Anime-tuned (banding, color quantization) |
| `anime_heavy` | APISR-style anime heavy (default) |

**Determinism:** pass `seed=N` to `DegradationPipeline(..., seed=N)` or via config
`preprocessing.on_the_fly.seed: N` for reproducible degradations across runs.

**Modes that share the pipeline:**
- `on_the_fly`     -> `DegradationPipeline(mode=cfg.on_the_fly.mode, ...)`
- `quality_adaptive` -> `DegradationPipeline(mode=tier_preset, ...)` (per-image tier selected by `QualityAnalyzer`)
- `gpu_degradation` -> GPU modules (`AnimeBlur`, `ColorQuantization`, `BandingArtifact`, etc.)
- `hybrid` / `precomputed` -> precomputed pairs (no per-batch degradation)
- `BaseDataset._apply_standard_degradation` -> legacy path (delegated to DegradationPipeline in v5)

### Pre-flight checks

`PreprocessingManager.__init__` raises `FileNotFoundError` at startup if:
- `mode='precomputed'` AND `precomputed.enabled: true` (default) AND `precomputed.base_dir` is missing
- `mode='hybrid'` AND `hybrid.precomputed_base: true` AND `hybrid.base_dir` is missing

To skip the check (e.g. for tests, smoke runs), set `precomputed.enabled: false` or
`hybrid.precomputed_base: false`. The error message names the missing path and the
`precompute_pairs.py` command to populate it.

## GPU Utilization Optimization

Training was bottlenecked by loss computation (51% of batch time), not data loading. Profiling revealed:

| Config | Total (ms) | Loss (ms) | Forward (ms) | Backward (ms) | Speedup |
|--------|------------|-----------|--------------|---------------|---------|
| v4 (Twin VGG+ResNet50) | 505 | 259 | 76 | 162 | baseline |
| v4_speed (VGG + no checkpointing) | 339 | 139 | 106 | 86 | **1.49x** |

### Key Optimizations Applied
- **Perceptual loss**: Twin VGG19+ResNet50 (202ms) -> VGG single layer (89ms) = 56% faster
- **FDL compute_every**: 2 -> 4 (saves 50% of DINOv2 forward passes)
- **Gradient checkpointing**: disabled (trades VRAM for speed; we have headroom at 2.9GB/8.6GB)
- **VGGPerceptualLoss bug fix**: Added `.features` accessor (was crashing silently)

### Profiling Tools
- `scripts/profile_gpu.py` — measures data load, forward, loss, backward, optimizer step times
- `scripts/profile_loss_breakdown.py` — measures time per individual loss component

### Loss Component Timing (v4 original)
| Component | Time (ms) | % of Loss | Fix Applied |
|-----------|-----------|-----------|-------------|
| Perceptual (Twin) | 202 | 49.6% | Switched to VGG single layer |
| FDL (DINOv2) | 112 | 27.6% | compute_every: 4 |
| Discriminator | 77 | 18.8% | Required for GAN |
| Frequency | 10 | 2.6% | Keep (fast) |
| Others | <5 | <2% | Keep |

## Training Performance

For long v7 runs on 8GB GPUs the loss stack (twin VGG+ResNet perceptual + FDL/DINOv2 + wavelet-guided + discriminator) can dominate per-iter time. Three perf levers are available without touching v7 core code.

### Profiling data (Quadro RTX 4000, batch=4, crop=64, full v7 loss stack)

Profiled with `python scripts/profile_gpu.py --config configs/finetune_neosr_span_v7_anime.yaml --batches 3 --device cuda`:

| Phase               | Avg (ms) | % of Total |
|---------------------|---------:|-----------:|
| Data Loading        |      0.3 |       0.2% |
| Forward Pass        |     32.3 |      14.3% |
| Loss Computation    |    102.7 |      45.6% |
| Backward Pass       |     81.8 |      36.4% |
| Optimizer Step      |      7.9 |       3.5% |
| **Total per batch** |  **225.0** |  **100%** |

So at the **v7 config defaults (batch=4)** the trainer runs at **~225ms/iter**, or **~0.5s/iter with gradient accumulation overhead**. An 80-epoch run on RTX 4000 Mobile = ~10-15 hours. The 2-epoch smoke = ~3-5 minutes.

The previous 10.89s/iter that triggered the perf investigation was caused by `data.batch_size: 128` in the smoke config: the DataLoader loaded 128 images per batch but `training.finetune.batch_size: 4` only fed 4 to the GPU. The other 124 images per step were wasted dataloader + x265 codec work. This was fixed in commit `0867e4c` by setting `batch_size: 4` consistently across all three config paths (`data.batch_size`, `training.batch_size`, `training.finetune.batch_size`).

If you see iter times >1s/iter after this fix, check:
1. `git diff HEAD~1 configs/finetune_neosr_span_v7_anime.yaml` -- make sure `data.batch_size` is 4 not 128
2. `git diff HEAD~1 configs/finetune_neosr_span_v7_anime_smoke.yaml` -- same check
3. Run `python scripts/profile_loss_breakdown.py --config configs/finetune_neosr_span_v7_anime.yaml --batches 3 --device cuda` to see per-loss-component timing

### CLI perf flags (added 2026-06-14)

```bash
python scripts/train.py --config configs/finetune_neosr_span_v7_anime.yaml \
  --batch-size 8 --crop-size 64 --grad-accum 1 \
  --no-fdl --no-wavelet
```

| Flag | Effect | Config path overridden |
|---|---|---|
| `--batch-size N` | sets `training.batch_size`, `data.batch_size`, `training.finetune.batch_size` | trainer: `self.finetune_cfg.get('batch_size', 4)` |
| `--crop-size N` | sets `data.crop_size`, `data.preprocessing.storage.crop_size` | `BaseDataset` (data) + `storage_estimator` |
| `--grad-accum N` | sets `training.finetune.gradient_accumulation_steps` | trainer: `self.gradient_accumulation` |
| `--no-fdl` | sets `training.finetune.loss.fdl.enabled=false` (skip DINOv2 forward) | trainer: `self.use_fdl` |
| `--no-wavelet` | sets `training.finetune.loss.wavelet_guided.enabled=false` | trainer: `self.use_wavelet_guided` |

Each override prints a `[CLI override] key: old -> new` line. The paths were verified against `src/training/neosr_finetuner.py` to avoid silent-fallback bugs. The original v7 config is not modified.

### Fast preset: `configs/finetune_neosr_span_v7_anime_fast.yaml`

Inherits the full v7 config but disables the three dominant loss components (FDL, ResNet50 perceptual twin, wavelet-guided), drops batch to 8 to fit on 8GB, and reduces `num_workers` to 4. **Expected: 3-5x speedup** (rough: 10.89s/iter -> 2-3s/iter on RTX 4000 Mobile at crop=64) at the cost of **~5-10% perceptual quality loss** on NR-IQA scores (CLIPIQA, MANIQA, NIQE). The full v7 config is still recommended for the final 80-epoch run.

```bash
# Recommended fast-iteration command
python scripts/train.py --config configs/finetune_neosr_span_v7_anime_fast.yaml --epochs 80
```

### Conditional FDL warmup (`loss.fdl.conditional`)

Opt-in via config; default off (no behavior change):

```yaml
loss:
  fdl:
    enabled: true
    conditional:
      enabled: true
      warmup_steps: 100   # skip DINOv2 forward for the first 100 global steps
```

The DINOv2 forward pass is skipped entirely while `global_step < warmup_steps`; the FDL contribution to `total_loss` is zero during that window. Use this to defer the ~28% perceptual-time cost until the model has learned basic pixel-level structure.

## Pre-Completion Checklist

Before marking a multi-step task as complete:
- [ ] All new modules/classes are imported and wired into existing factories/pipelines
- [ ] All new config options have been traced: config → code path → execution
- [ ] Platform-dependent features have been verified for the current OS (Windows/Linux/macOS)
- [ ] `py_compile` or equivalent syntax check passes on all modified Python files
- [ ] `--dry-run` or equivalent validation passes for config changes
- [ ] No dead code: every new file is imported by at least one existing module

## Git-ignored

`data/`, `checkpoints/`, `logs/`, `results/`, `pretrained/`, `*.pth`, `*.pt`, `*.ckpt`, `.venv/`

## Dead Code Patterns

The project accumulates dead code through iterative development. Before adding new variants:
- **Checkpoint models**: `src/models/span/` had 7 `checkpoint_compatible_model*.py` files (6 were dead). Keep only the one actually imported.
- **Checkpoint loaders**: `src/utils/` had 3 loaders (2 never imported). Consolidate before adding new ones.
- **Config files**: `configs/` accumulates debug/working variants with non-standard schemas. Delete broken configs, don't fix them.
- **Summary .md files**: Implementation summaries in root become stale quickly. Prefer updating AGENTS.md.

Rule of thumb: if a file isn't imported by `src/` or referenced by a config, it's likely dead.

## Existing Agent Instructions

- `.windsurf/rules/base.md` — extensive style/debugging/convention rules (v1.8)
- `.windsurf/.agentrc` — references quality gate (partially outdated)
- `.windsurf/workflows/` — many workflow prompt files (mostly generic, not repo-specific)
