# AGENTS.md — upscale_anime

## Memory protocol — READ FIRST

**At the start of every session**, before reading any other section of this file:

1. Read [`docs/PROJECT_MEMORY.md`](docs/PROJECT_MEMORY.md) — the canonical "where are we?" file. It has the current branch/HEAD, pending tasks, empirical anchors (PSNR/lap_var numbers), decision log, and halt conditions.
2. Confirm the branch and HEAD match what you intend to work on.
3. Cross-reference any task you pick up against the "Pending tasks" section.

**During work**, update `PROJECT_MEMORY.md` whenever:
- A decision is made or reversed → append to "Decision log" with date stamp.
- A phase begins or completes → update "Phase history".
- Tasks move forward → update "Pending tasks" (mark done / add new).
- A new finding (PSNR, lap_var, fps, etc.) → update "Key empirical anchors".
- A new critical rule → update "Halt conditions / guardrails".

After updating, commit with: `memory: <one-line summary>`.

Do not use `PROJECT_MEMORY.md` for long-form docs, code documentation, or large tables — those belong in `docs/plans/`, `AGENTS.md`, or `docs/v7_results.md`. The memory file is for **state**, not reference material.

---

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

## Pretrained Checkpoints

The `pretrained/` folder contains 3 SPAN checkpoints with **very different** behavior. Loading the wrong one as a warm-start silently produces flat-blue outputs that training cannot recover from.

| File | Status | Use for |
|---|---|---|
| `pretrained/span_nomosuni_4x.pth` | **BROKEN** -- flat-blue output (`upsampler.0.bias mean ~0.43`). Missing upstream `spanx4_ch48.pth` warm-start that the official `4xNomosUni_span_multijpg` lineage requires. | **DO NOT USE** as `pretrained_path` |
| `pretrained/span_pix_pretrain_4x.pth` | **OK** -- Phhofm official pix-loss pretrain (the actual warm-start of `4xNomosUni_span_multijpg`). Produces real images (output std ~0.24). | canonical warm-start for all SPAN finetune configs |
| `pretrained/span_mssim_pretrain_4x.pth` | **OK** -- Phhofm official mssim-loss pretrain. | alternative (different pretrain loss objective) |

### Warm-start taint signature

A checkpoint warm-started from `span_nomosuni_4x.pth` has `model.upsampler.0.bias` with `mean ~0.43, std ~0.02` (vs `mean ~0.06, std ~0.007` for clean pretrains). Training does NOT recover from this -- V4_HYBRID_003 ran 66 epochs and the bias stayed at 0.4331. Forward-pass output is dominated by this bias and becomes flat.

### Audit

Run `python scripts/audit_pretrain_ckpts.py` to scan all checkpoints. It writes `results/audit/taint_report.csv` and per-checkpoint SR PNGs to `results/audit/`. Taint verdict is `TAINTED` if `bias_mean in [0.35, 0.45]` AND `bias_std < 0.05`.

### History (2026-06-20)

50 tainted checkpoints across 5 locations were deleted:
- `checkpoints/NEOSR_SPAN_V3_FINE_TUNE_001/` (2 .pth)
- `checkpoints/NEOSR_SPAN_V4_HYBRID_{002,003,004}/` (46 .pth combined)
- `checkpoints/finetune_epoch_{5,10}.pth` (root-level, 2 .pth)

All 7 finetune configs (`finetune_neosr_span.yaml`, `_v2.yaml`, `_v3.yaml`, `_v4.yaml`, `_v4_speed.yaml`, `_v5.yaml`, `_v5_corrected.yaml`) were patched from `span_nomosuni_4x.pth` to `span_pix_pretrain_4x.pth`. v6 and v7 already used `span_pix_pretrain_4x.pth` (no change).

To re-introduce `span_nomosuni_4x.pth` for any reason, first re-download from https://github.com/Phhofm/models/releases/tag/4xNomosUni_span_multijpg and verify `upsampler.0.bias mean < 0.1`.

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

## V7 Tier 1 Performance Speedups (2026-06-20)

Branch: `perf/v7-tier1-speedups`. Targets the **loss (45.6%) + backward (36.4%) = 82% of iter time** identified by `scripts/profile_loss_breakdown.py`. Five pure refactors, no model architecture changes. Expected wall-clock savings: **30-50ms per batch (~15-22%)** on the 225ms v7 baseline at crop=128, batch=4 on RTX 4000 Mobile.

### Phase 1: `torch.no_grad()` on frozen-backbone target forwards

Every loss with a frozen backbone (VGG19, ResNet50, DINOv2, edge detectors) was running the **target-side forward with autograd enabled**. The target is constant w.r.t. the generator, so the autograd graph through it is pure waste.

| File | Change | Est. savings |
|---|---|---|
| `src/losses/fdl_loss.py:108-110` | Wrap `target_feat = self.extract_features(target_norm)` in `no_grad` (DINOv2 frozen) | 10-20 ms |
| `src/losses/twin_perceptual_loss.py:244-268` | New `forward` extracts target VGG+ResNet under `no_grad`, then pred; keeps `compute_vgg_loss`/`compute_resnet_loss` public for backward compat | 10-20 ms |
| `src/losses/wavelet_guided_loss.py:89-90` | Wrap `target_w = self._haar_wavelet_dec(target)` in `no_grad` | 1-2 ms |
| `src/training/neosr_finetuner.py` G-step | Wrap `disc_real = self.discriminator(hr.float())` in `no_grad` (relativistic cross-term only flows through `disc_fake`) | 2-5 ms |

### Phase 2: Cache D-step disc outputs for G-step reuse

`src/training/neosr_finetuner.py:_train_discriminator` now returns `_disc_real_tensor` and `_disc_fake_tensor` (detached). The G-step reuses the cached `disc_real` under `no_grad` instead of re-running the discriminator on HR. Also dropped the redundant `sr.detach().clone()` (3 MB allocation per batch) in favor of `sr.detach()`. Saves 5-10 ms per adversarial step.

### Phase 3: Precompute DCT basis; fuse Haar wavelet into grouped conv

- **DCT basis** (`src/losses/frequency_aware_loss.py`): The old code rebuilt the 64-element DCT-II basis on every forward via 64 individual `torch.cos()` calls (each a GPU kernel launch + CPU sync). New: precomputed once in `__init__` using `math.cos` on CPU, registered as a buffer. Saves 1-3 ms per forward.
- **Haar wavelet** (`src/losses/wavelet_guided_loss.py`): The old code used 6 separate `F.conv2d` calls (4 subband × row+col cascade), which actually applied the conv **twice** (cascade) producing `H/4` subbands — a two-level decomposition, not the standard single-level. New: single grouped `F.conv2d` with kernel `[4, 2, 2]` (outer product of avg/diff H × W × 0.25) at stride 2, producing `H/2` subbands. **Bit-exact** equivalence to the correct single-level decomposition (verified: max abs diff = 0.0 vs canonical reference). Saves 1-2 ms per forward.

### Phase 4: BF16 autocast (instead of FP16)

`src/training/neosr_finetuner.py:1132` changed `autocast('cuda')` (default FP16) to `autocast('cuda', dtype=torch.bfloat16)`. On Ampere (RTX 30/40-series including RTX 4000 Mobile) BF16 has the same Tensor Core throughput as FP16, FP32's exponent range (no underflow/overflow), and **does not require GradScaler** (scaler is a no-op when BF16 or AMP disabled).

- Existing `GradScaler('cuda', enabled=self.use_amp)` kept for FP16 fallback compatibility.
- `enabled=False` FP32 overrides on DISTS, discriminator, wavelet decomposition, and FDL DINOv2 are preserved — numerically sensitive ops continue to run in FP32 regardless of autocast dtype.

**This is the only change that requires GPU validation.** Cannot verify on CPU. User must run 2-epoch smoke test + `scripts/profile_loss_breakdown.py` to confirm BF16 stability on RTX 4000 Mobile.

### Phase 5: `fast_validation` flag in `_compute_total_loss`

`src/training/neosr_finetuner.py:_compute_total_loss` gained a `fast_validation: bool = False` parameter. When `True`, the heavy frozen-backbone losses (FDL/DINOv2, frequency/DCT, wavelet-guided) are skipped and only the cheap ones (pixel, perceptual, line_art, color, flat) are computed. `validate()` calls it with `fast_validation=True`. Cuts validation wall-clock ~5-10x.

Validation loss values are still logged (just cheaper subset). Checkpoint selection (PSNR/SSIM/LPIPS) is unaffected.

### Verified

- `pytest tests/test_fdl_loss.py tests/test_relativistic_gan.py tests/test_phase2_high_priority_fixes.py tests/test_phase3_medium_fixes.py tests/test_two_phase_training.py tests/test_safe_checkpoint_load.py tests/test_phase4_low_priority_fixes.py tests/test_grad_scaler.py` → **90/90 passing**.
- `pytest tests/test_perceptual_loss.py tests/test_new_losses.py tests/test_xdog_pseudo_gt.py tests/test_neosr_finetuner.py tests/test_v7_loss_schedule.py tests/test_critical_fixes.py tests/test_phase1_critical_fixes.py tests/test_phase3_medium_priority_fixes.py tests/test_phase5_additional_fixes.py` → **99 passed, 3 pre-existing failures unrelated to changes** (DISTS CPU/CUDA mismatch, `torch.load` local-class pickle, `ReduceLROnPlateau(verbose=)` removed in newer PyTorch). Confirmed via `git stash` that all 3 fail on `HEAD` too.
- `pytest tests/test_v7_tier1_phase5_fast_validation.py` → **13/13 passing** (Phase 5 dedicated regression test: signature, behavior, disabled-flag handling, `validate()` wiring).
- `scripts/train.py --config configs/finetune_neosr_span_v7_anime.yaml --dry-run` → **valid**.
- `py_compile src/training/neosr_finetuner.py src/losses/*.py tests/test_v7_tier1_phase5_fast_validation.py` → clean.

### User action items (after this commit)

1. **GPU smoke test (REQUIRED):**
   ```bash
   python scripts/train.py --config configs/finetune_neosr_span_v7_anime_smoke.yaml
   ```
   Verify no NaN, losses sane, no autocast errors. Then run the full v7 finetune.

2. **Profile to confirm savings:**
   ```bash
   python scripts/profile_loss_breakdown.py --config configs/finetune_neosr_span_v7_anime.yaml --batches 20 --device cuda
   ```
   Compare against the pre-Tier-1 baseline (Loss 45.6% / Backward 36.4%).

3. **NOT committed yet** per AGENTS.md convention. Review `git diff` then commit:
   ```bash
   git diff
   git add src/losses/fdl_loss.py src/losses/twin_perceptual_loss.py src/losses/wavelet_guided_loss.py src/losses/frequency_aware_loss.py src/training/neosr_finetuner.py
   git commit -m "v7 tier 1: no_grad frozen backbones, disc cache, DCT precomp, Haar fusion, BF16, fast_validation"
   ```

## V8+ Survey (2026-06-20)

A complete 2024-2026 survey of anime super-resolution candidates is in `docs/survey_2026/`:
- `README.md` — executive summary + 3-step weekend plan.
- `self_baseline.md` — what v7 looks like.
- `sources.md` — 49 sources (arXiv + Phhofm + OpenModelDB + spandrel + HF).
- `candidates.md` — 45 candidates scored by effort/risk/expected gain.
- `deep_dives.md` — 1-page writeups for top 5.
- `comparison.md` — master matrix vs v7 baseline.
- `recommendations.md` — Tier 1/2/3 ranked adoption order.

### The MANIQA gap

Our v7 self-baseline N=10 val_hr scores vs APISR targets:

| Metric | Stock pretrained | v4 finetune best | APISR target | Gap |
|---|---:|---:|---:|---:|
| CLIPIQA | **0.667** | 0.632 | >= 0.65 | cleared |
| MANIQA  | **0.422** | 0.334 | >= 0.48 | **-0.058** |
| NIQE    | 7.681 | **7.123** | <= 7.5 | cleared (v4) |
| LPIPS   | 0.0229 | **0.0104** | <= 0.10 | cleared |

**MANIQA is the only target not cleared by our self-baseline.**

### Tier 1 quick wins (S-effort, ship this weekend, expected Δ MANIQA +0.01 to +0.06)

1. **TTA 8x flip/rot ensemble** at inference (`scripts/inference.py --tta`): LPIPS -10%, free.
2. **Model souping** v6 + v7 `finetune_best.pth` EMA params: MANIQA +0.01 to +0.05.
3. **OS-RealPLKSR preset** in `DegradationPipeline`: NIQE -0.05.
4. **3-phase GAN scheduler**: NIQE -0.05.
5. **D-EMA** (mirror G-EMA on D): NIQE -0.05.

### Tier 2 medium wins (M-effort, 1-2 weeks, expected Δ MANIQA +0.02 to +0.04)

1. **MambaIRv2** (`sab_type: mamba_v2`, Linux-only): MANIQA +0.02.
2. **DINOv3 in FDL** (`src/losses/fdl_loss.py`): MANIQA +0.01.
3. **Projected-GAN discriminator**: MANIQA +0.02, less mode collapse.
4. **Patch-NCE loss**: MANIQA +0.02 (needs unpaired LR).
5. **`2xBHI_small_span_fast_pretrain`** warm-start swap: -50% inference.
6. **Hard-example mining curriculum**: MANIQA +0.005.
7. **VQD-SR learned codebook**: MANIQA +0.02 on OOD-LR only.

### Tier 3 heavy (L-effort, 2-4 weeks, only if MANIQA gap remains)

- DAT2 backbone swap: CLIPIQA +0.02, MANIQA -0.01.
- HAT-Lite: CLIPIQA +0.03 (PSNR ceiling).
- RealPLKSR + Dysample: CLIPIQA +0.005, MANIQA +0.005.
- SUPIR post-process: MANIQA +0.03 (slow, 25s/img).

### Projected MANIQA after Tier 1+2: 0.532 (target 0.48 cleared with margin)

See `docs/survey_2026/recommendations.md` for the full action plan with code snippets, expected wall-clock, and integration steps.

## V8 Tier 1 Shipped (2026-06-20)

Two Tier-1 items from the V8+ survey are now in the working tree (untracked on branch `perf/v7-tier1-speedups`). Code + tests are green; only docs and the long GPU smoke remain.

### TTA — D4 8x flip/rot ensemble

- **`src/inference/tta.py`** — `D4_AUGMENTATIONS` (8 (aug, inv) pairs) and `tta_forward(model, lr, use_clip=True)`. The forward runs the model 8 times on D4-augmented inputs, applies the inverse aug to each prediction, optionally clamps to `[0, 1]`, and returns the mean in the original dtype. `use_clip=True` is the default (matches the `clamp(0, 1)` behavior used everywhere else for SR output).
- **`scripts/inference.py --tta`** — `argparse` flag, prints `[TTA] Enabled: 8x D4 flip/rot ensemble (8x inference cost)`, applied in both single-image and directory-batch paths.
- **`scripts/compare_checkpoints.py --tta`** — same flag, applied inside `run_inference_tta(...)` helper.
- **`tests/test_tta.py`** — 9 tests: D4 closure under inversion (3), `tta_forward` averaging/clipping/dtype/manual-mean equivalence (5), CUDA smoke (1, skipped on CPU-only).

### Model souping

- **`scripts/soup_checkpoints.py`** — `soup_checkpoints(input_paths, output_path, use_ema=True)` averages EMA state-dicts across N checkpoints (Wortsman et al., ICML 2022). Preference order: `ema_state_dict` > `model_state_dict` > `params` > `state_dict` > raw. Output drops `scaler_state_dict` and `optimizer_state_dict` (per-checkpoint state; user re-inits on resume) and stamps `soup_inputs` / `soup_sources` / `soup_n` / `soup_timestamp` metadata. CLI: `--inputs a.pth b.pth [...] --output soup.pth [--no-ema]`.
- **`tests/test_soup_checkpoints.py`** — 8 tests: 2-ckpt EMA, EMA-fallback to model_state_dict, key-mismatch ValueError, single-ckpt ValueError, 3-ckpt average, dtype preservation, `load_state_dict` EMA preference, raw-state_dict fallback.

### Verification (per `docs/plans/todos_v8_mambair_2026_06.md` D.1–D.3)

```
pytest tests/test_tta.py tests/test_soup_checkpoints.py  -> 17 passed
py_compile src/inference/tta.py scripts/soup_checkpoints.py scripts/inference.py scripts/compare_checkpoints.py  -> clean
```

### Quick-start

```bash
# TTA inference on a single image
python scripts/inference.py --model-type neosr_span \
    --checkpoint checkpoints/NEOSR_SPAN_V7_ANIME/finetune_best.pth \
    --input lr.jpg --output sr_tta.png --tta

# Side-by-side with TTA enabled
python scripts/compare_checkpoints.py \
    --baseline checkpoints/NEOSR_SPAN_V6_ANIME/finetune_best.pth \
    --v7      checkpoints/NEOSR_SPAN_V7_ANIME/finetune_best.pth \
    --input data/val_hr --gt data/val_hr \
    --output results/comparison_v6_vs_v7/ \
    --metrics psnr ssim lpips clipiqa maniqa niqe topiq_nr --tta

# Soup v6 + v7 finetune_best
python scripts/soup_checkpoints.py \
    --inputs checkpoints/NEOSR_SPAN_V6_ANIME/finetune_best.pth \
            checkpoints/NEOSR_SPAN_V7_ANIME/finetune_best.pth \
    --output  checkpoints/SOUP_V6_V7/soup.pth
```

### NOT committed

Per AGENTS.md convention, the V8 Tier 1 files are in the working tree but uncommitted. User reviews `git diff` and commits when ready.

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

### Codec backends (v7 prediction-oriented compression)

The v7 config uses `compression_stage2: [avif, h264, h265, jpeg]` to
synthesize real-world compression artifacts. The previous
implementation spawned an `ffmpeg` subprocess for every video-codec
frame; that path is spawn-bound and slow on Windows. v7 now uses
**in-process** codec libraries as the primary path:

| Codec | Backend | Speed (RTX 4000, 64x64) | Source |
|---|---|---:|---|
| JPEG | OpenCV `cv2.imencode` (in-process) | <1 ms | always available |
| WebP | PIL `.save(format='WEBP')` (in-process) | ~5 ms | always available |
| AVIF  | `pillow_heif.register_heif_opener()` + PIL `.save(format='AVIF')` (in-process) | ~7-19 ms | `avif >= 1.4` |
| h264  | PyAV `av.Codec('libx264', 'r')` + `av.open(BytesIO, 'w', format='mp4')` (in-process) | ~9-14 ms | `av >= 17.0` |
| h265  | PyAV `av.Codec('libx265', 'r')` (in-process; HEVC intra is intrinsically slow) | ~46-56 ms | `av >= 17.0` |
| h264/h265 fallback | `imageio_ffmpeg.write_frames` (subprocess) | 10-100 ms | `imageio-ffmpeg` |

**Backend priority order** (set in `src/data/compression_modules.py`):

1. PyAV (`av` package) for h264 + h265 -- in-process, no spawn.
2. pillow-heif (`pillow_heif` package) for AVIF -- in-process.
3. `imageio_ffmpeg` subprocess -- only used if both PyAV and pillow-heif
   are unavailable. This is the legacy path; it works but is slower and
   produces extra tempfile I/O.
4. JPEG fallback -- last resort. The compression module never raises
   on a missing backend; it logs a warning and degrades to JPEG so
   training continues.

At `CompressionPipeline.__init__` the trainer log gets one
`[CompressionPipeline] codecs=... | backends: av=... pillow_heif=...
imageio_ffmpeg=...` line that greps as the single source of truth
for "which in-process codec is in use right now". Each codec class
also emits a one-time `[AVIFCompression] backend=...` /
`[PyAVVideoCompression] backend=...` line on first use.

**If you see codec step taking >100 ms/frame**, the codec has fallen
back to the ffmpeg subprocess path -- check the `[Compression] backend=...`
line. Common causes:

- `av` or `pillow_heif` not installed. The init log line will show
  `av=False` or `pillow_heif=False`.
- The `av` package was installed but `libx264` / `libx265` is missing
  in the bundled libav build. Check with
  `python -c "import av; print('libx264' in av.codecs_available)"`.
- A custom build of libav that lacks H.264/H.265. The trainer
  will still train (JPEG fallback) but the prediction-oriented
  compression will be weaker.

Empirical benchmark on Quadro RTX 4000 (this session, 2026-06-14,
`scripts_temp/test_codec_speed.py`):

| Codec   | 32x32 | 64x64 | 128x128 |
|---------|------:|------:|--------:|
| AVIF    |  7.04 |  6.74 |   18.87 |
| h264    | 10.94 | 10.90 |   13.65 |
| h265    | 45.89 | 47.55 |   56.43 |

For comparison, the previous `imageio_ffmpeg` subprocess path
measured 0.01-0.07 s/frame *plus* per-call subprocess spawn
(50-200 ms cold). Net: the codec step in v7 training is now
~5-10x faster wall-clock.

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

## `apps/anime_upscaler_gui/` — UI/UX Overhaul (Phases 0–14 done)

The Tkinter desktop app has its own **14-phase UI/UX overhaul** in
progress. If you touch this app in a fresh conversation, read the
**resume point first**:

- **`apps/anime_upscaler_gui/docs/RESUME.md`** — state of play, test infra
  (the conftest.py shared-root pattern is critical), next steps, and
  quick commands. Start here.
- **`apps/anime_upscaler_gui/docs/plans/ui_ux_overhaul.md`** — the master
  plan with full per-phase detail.
- **`apps/anime_upscaler_gui/docs/plans/ui_ux_overhaul_checklist.md`** —
  per-phase task list with current status (11/14 done).
- **`apps/anime_upscaler_gui/docs/audit/ui_ux_audit.md`** — the initial
  audit that informed the plan.
- **`apps/anime_upscaler_gui/docs/onboarding.md`**, **`docs/themes.md`**, and
  **`docs/dev/widgets.md`** — Phase 12 user and contributor docs.

Quick facts: Tk GUI at `apps/anime_upscaler_gui/anime_upscaler_gui/`,
~1200 LOC, pytest `179/179` passing. Standalone venv at
`apps/anime_upscaler_gui/.venv/`. Run tests from the **repo root** with
`apps/anime_upscaler_gui/.venv/Scripts/python.exe -m pytest apps/anime_upscaler_gui/tests/ -q`.

**Last updated 2026-08-31**: Phase 1 (Real-time 4K) shipped. See "Phase 1 (Real-time 4K) Shipped" entry below + `docs/rfdn_realtime_report.md` Section 10.

**Last updated 2026-07-26**: Phases 0–10 complete (token sweep, data layer,
widget extraction, tabbed layout, settings/persistence, theme system,
threading fixes, interaction improvements, empty/error states, telemetry,
accessibility), Phase 12 (Documentation), and Phase 14 (i18n EN + AR —
minimal viable: two Python dicts in `messages.py`, `Ctrl+Shift+L`
toggle, `side_for()` RTL mirroring helper). Phase 11 (Visual polish /
Phosphor icons) and Phase 13 (Splash) remain — both blocked by missing
icon assets.

## Phase 1 (Real-time 4K) Shipped (2026-08-31)

Phase 1 of the `docs/plans/realtime_4k_plan.md` plan (B + C + NVENC) is in the working tree. Phase 1.B (TensorRT engine) and Phase 1.N (NVENC encoder) are production-ready; Phase 1.C (batching) is code-complete but the batched-TensorRT path has a known stream-sync bug (deferred — see docs).

### Shipped

- **Phase 1.B — TensorRT engine for RFDN student**
  - Per-shape per-batch disk-persisted engine cache (`%APPDATA%/anime_upscaler_gui/cache/trt/`, 1 GB LRU cap). Engines built on first run, reused thereafter.
  - TRT 11.2.1.2 backend via `_TrtBackend` with multi-stream `wait_stream` synchronization.
  - 2.37× inference speedup over PyTorch FP16 (Phase 1.B bench).
  - Files: `trt_engine.py` (new, 10 KB), `pipeline.py` (extended), `settings.py` + `widgets/settings_panel.py` (TRT checkbox).
- **Phase 1.N — NVENC hardware encoder**
  - `h264_nvenc -preset p1 -rc constqp -qp 18` path with auto-fallback to libx264 when NVENC unavailable.
  - 26.5% CPU at 4K encode vs 42.5% for libx264 (Phase 1.N measurement).
  - Files: `ffmpeg.py` (extended, 88 → 158 lines), `settings.py`, `settings_panel.py` (NVENC checkbox + preset combobox), `pipeline.py` (`_RunJob` threaded), `app.py` (`_enqueue_next` populates fields).
- **Phase 1.C — Batching** (code-complete, batched-TensorRT path DEFERRED)
  - `_process_video_batch` + `_to_tensor_batch` + `_tensor_to_bgr_batch` helpers; `_PinnedPool` extended for 4D (N, 3, H, W) input + (N, H, W, 3) output buffers.
  - PyTorch batched path works (1.11–1.16× speedup); TensorRT batched path needs stream-sync fix (per-call `torch.cuda.synchronize` works but kills async perf; CUDA graphs are the proper fix).
  - E2E measurement shows b4 is SLOWER than b1 on RTX 4000 for this model — batching deferred in practice; the per-frame path (b=1) is the production path.
  - Phase 1.C also exposed two latent bugs in the existing batched call site, fixed: `writer = None` default + pre-bound ref lists.

### Measured end-to-end fps (10.2 s 854x480 → 3416x1920)

| Config | fps mean | fps peak | infer ms |
|---|---:|---:|---:|
| **trt_b1_nvenc (production)** | **18.80** | **20.52** | 23.81 |
| trt_b4_nvenc | 16.82 | 19.11 | 104.00 |
| pt_b1_nvenc | 10.02 | 11.33 | 83.44 |
| pt_b4_nvenc | 9.01 | 10.77 | 262.87 |
| pt_b1_libx (baseline) | 10.18 | 10.86 | 74.72 |

### Goal assessment

- Inference ≥ 25 fps @ 4K: **PASS** (31.4 fps @ 4K, Phase 1.B bench).
- End-to-end ≥ 25 fps @ 4K: **PARTIAL** (18.80 / 20.52 fps — ~75% of target; 13 ms/frame short, requires CUDA graphs or NVDEC, Phase 2 territory).
- TRT ≥ 2× speedup: **PASS** (1.87× end-to-end, 2.37× inference only).
- NVENC CPU < 30%: **PASS** (26.5% vs 42.5%).
- Engine cache survives restart: **PASS** (rebuild on missing, reuse on present).
- Batching ≥ 3×: **DEFERRED** (RFDN too small; b4 is 11% slower than b1).

### Full report

See `docs/rfdn_realtime_report.md` Section 10 for the full Phase 1 E2E verification with all measurements, files-changed inventory, and Phase 2 candidates to close the remaining gap to 25 fps end-to-end.

## Phase 3 v3 student attempt (2026-09-02) — not promoted

Attempted to train an adversarial v3 student (RFDN distilled from RealESRGAN-animevideov3 with PatchGAN70 + HingeGANLoss + Sobel edge loss). See `docs/plans/student_v3_result_2026_08.md` for the full writeup. Summary:

- v3 trained cleanly (30 epochs, ~17 min on RTX 4000, no NaN, no collapse) but D.2 quality gate failed: full-frame lap_var=22.3 (threshold 35) vs v1's 21.0. Marginal improvement only.
- Animevideov3 baseline (lap_var=58.1) is now the user-facing high-quality option in the GUI; v3 student is parked.
- Code from Phase 3 is committed and works (Phase A smokes pass; can be re-purposed for a non-residual architecture in a future attempt).

### Lessons learned (read these before any future distillation work)

1. **Bicubic shortcut anneal is hostile to warm-start.** The RFDN student trained with shortcut_weight=1.0 (v1, v2) cannot be re-trained with shortcut<1.0 without mode collapse. Two attempts mode-collapsed:
   - `--shortcut-anneal 1to0` (5 epochs): PSNR 29.92 → 5.41 over 5 epochs.
   - `--shortcut-anneal 1to0slow` (15 epochs): PSNR 29.93 → 7.48 over 14 epochs.
   Root cause: v1 student learned a delta-from-bicubic. Forcing it to also produce the full SR requires scaling the delta by 4x in 5 epochs, which LR=5e-5 cannot do.
   **Decision rule**: if warm-starting, leave `--shortcut-anneal off` (or omit) and rely on adversarial + edge loss alone. If training from scratch, `--shortcut-anneal 1to0` (5 epochs) is OK (verified in B.3 smoke).

2. **`val_lap_var` at training crop size is misleading.** I logged val_lap_var=187 in the per-epoch metrics.json, computed on small 192x192 val crops. The full-frame harness lap_var=22.3 is the ground truth. Always evaluate at the resolution the user actually sees (full 4x frame for video, native res for image). The training set's high-frequency aliasing artifacts inflate Laplacian variance on small crops.

3. **Per-epoch checkpoint rotation works** but the alphabetical-sort bug shipped in commit `21151e7`. Always sort by **numeric** epoch (see `distill.py::end_of_epoch` rotation block):
   ```python
   def _epoch_num(p):
       return int(p.stem.split("_")[1])  # NOT alphabetical sort
   ```
   Otherwise ep 9 wins over ep 30 because "9_ema" > "30" in string comparison, leaving only the last ~2 epochs on disk.

4. **Animevideov3 IS the v1 replacement** for shipping quality. The RFDN student's residual structure (v1 with shortcut_weight=1.0) is fundamentally limited; the SRVGG architecture learns detail more naturally. The Phase 3 v3 plan's hoped-for gain didn't materialize with the available training compute (~17 min/run).

5. **The per-epoch rotation metric JSON files were archived alongside the .pt files**, which made finding the best epoch awkward. If you re-derive metrics from the archive, glob across both `runs/.../epoch_*_metrics.json` AND `runs/.../archive/epoch_*_metrics.json`.

### Phase 3 artifacts in tree

- `anime_upscaler/adv_losses/__init__.py`, `adversarial.py` (PatchGAN70 + HingeGANLoss), `edge_loss.py` (Sobel L1 on Y)
- `anime_upscaler/teacher.py` — added `SRVGGNetCompact`, `RealESRTeacher`, `TEACHERS` registry
- `anime_upscaler/student.py` — added `RFDN.shortcut_weight` + `set_shortcut_weight(w)`
- `anime_upscaler/distill.py` — added `--teacher`, `--lambda-adv`, `--shortcut-anneal {off,1to0,1to0slow}`, `--no-ema`, `--fresh-epoch`; per-epoch rotation with numeric sort; Phase 3 recipe + Phase 2 v3 recipe preserved
- `scripts/train_v3_smoke.py` — 2-epoch harness
- `scripts/compare_students_vs_pretrained.py` — `--v3-ckpt` appends v3 row to MODELS
- v3 ckpt (un-promoted): `runs/distill_v3_4x_v3_epoch18_ema_unpromoted.pth`

## Phase 4 plan: I1 nearest-residual + I2 isolated adv + I3 SRVGG body (2026-09-03)

Three sequential experiments to break the RFDN+anneal ceiling established in Phase 3. Full plan in `docs/plans/student_phase4_nearest_adv_srvgg_plan.md` (391 lines, committed `26ac928`).

**Status (2026-09-03, post-I3 — PHASE 4 CLOSED)**: I1 + I2 + I3 ALL EXECUTED, ALL **NOT PROMOTED**.
- I1 result doc: `docs/plans/student_phase4_i1_nearest_residual_result.md` (9056 bytes).
- I2 result doc: `docs/plans/student_phase4_i2_isolated_adv_result.md` (10196 bytes).
- I3 result doc: `docs/plans/student_phase4_i3_srvgg_body_result.md` (~13 KB).
- I1 code: commit `bd72dd3`. I2 code: `7ac0665`. I3 code: `<this-commit>`.
- Tests: `tests/test_shortcut_mode.py` (4) + `tests/test_feature_distillation.py` (15) + `tests/test_tiny_srvgg.py` (9) — all 28 pass post-I3.
- Memory file `docs/PROJECT_MEMORY.md` §3 / §4 / §5 / §6 / §8 reflects the I1 + I2 + I3 outcomes.

### Phase 4.I1 — Nearest-residual RFDN (cheap, ~1 hour) — **EXECUTED 2026-09-03, NOT PROMOTED**

- **Hypothesis**: RFDN+bicubic's lap_var ceiling comes from the residual *type*, not the block. Animevideov3 uses `mode='nearest'`, ours uses `mode='bicubic'`.
- **Change**: Add `--shortcut-mode {bicubic,nearest}` to `distill.py`; `RFDN.shortcut_mode` honors it. ~10 LOC + `set_shortcut_mode()` runtime hook.
- **Command**: `distill.py --teacher animevideov3 --shortcut-mode nearest --lambda-adv 0.001 --shortcut-anneal off --epochs 40 --lr 5e-5`
- **Success criterion**: D.2 PASS (full-frame lap_var >= 35), PSNR >= 29.0.
- **Result**: D.2 PASS (lap_var **109.4**, 5.2× v1 — hypothesis CONFIRMED), but D.1 FAIL (PSNR **27.97** vs v1 29.89, student < bicubic 29.45). **NOT PROMOTED.**
- **Root cause**: adversarial + from-scratch + new-shortcut together oversharpen.
- **Output dir**: `runs/distill_i1_nearest_residual/` (preserved per Q6 rule, all 40 epochs).
- **Full doc**: `docs/plans/student_phase4_i1_nearest_residual_result.md`.
- **Lesson** ("sharpness ≠ quality"): D.1 (PSNR) is the binding constraint, not D.2. See `docs/PROJECT_MEMORY.md` §6 decision log.

### Phase 4.I2 — Isolated adversarial (medium, ~1.5 hours) — **EXECUTED 2026-09-03, NOT PROMOTED**

- **Hypothesis**: Recent span+adv regression (PSNR 28.91 vs v1's 29.89) was caused by L_feat + L_adv pulling in opposite directions. Isolating them reveals which is the real culprit.
- **Change**: Add `--feat-weight` flag to `distill.py`; gate feature loss at line 539, weight multiplier at line 552. ~5 LOC. Default = 1.0 (backward-compat with Phase 2 v3 / Phase 3 recipe).
- **Runs**: I2a (animevideov3 + adv + feat=1.0, 30 ep = Phase 3 v3 sanity check) and I2b (SPAN + adv + **feat_weight=0**, 40 ep = the new isolation).
- **Success criterion**: I2b >= 29.5 dB val PSNR (beats feat=1 baseline at 28.91).
- **Result**: Both D.2 FAIL, but I2b is the meaningful partial win.
  - **I2a**: val PSNR EMA 29.885 (= v1), full-frame lap_var **20.8** (= v1). Animevideov3+adv+feat=1 is a no-op over v1.
  - **I2b**: val PSNR EMA **29.911** (+0.022 dB over v1), full-frame lap_var **24.2** (+15% over v1's 21.0). SPAN+adv+feat=0 yields a small but real improvement.
  - I2a proves animevideov3+RFDN is a dead end; I2b confirms L_feat was the regression driver in the 2026-08 span+adv run, not L_adv.
- **Output dirs**: `runs/distill_i2a_adv_only_animevideov3/` (17.3 min), `runs/distill_i2b_adv_only_span/` (23.8 min). Both gitignored per Q6.
- **Full doc**: `docs/plans/student_phase4_i2_isolated_adv_result.md` (10196 bytes).
- **Lesson**: `feat_weight=0` is the safe default for any future SPAN+adv RFDN variant. Animevideov3 is too far from v1's distribution for adversarial-only intervention to help; SRVGG body (I3) is needed to make animevideov3's response signal useful.

### Phase 4.I3 — SRVGG-body student (heavy, ~3 hours) — **EXECUTED 2026-09-03, NOT PROMOTED**

- **Hypothesis**: Animevideov3's plain SRVGG stack is just better for anime than RFDN's FIM+PA design. Copy the architecture at our 315K budget.
- **Change**: New `TinySRVGGStudent` class in `student.py` (12 convs × 52 channels, PReLU, PixelShuffle, nearest residual, 317,300 params — matches v1's 315K). Add `--arch {rfdn,srvgg}` to `distill.py`. Wire into `archs.py` (vendored copy + auto-sniff shortcut_mode for rfdn_student). ~+320 LOC total + 9 smoke tests.
- **PREREQUISITE**: PROJECT_MEMORY §8 line 299 flagged `archs.py:591` as STALE; I3 fixes it (vendored RFDN now honors shortcut_mode; auto-sniffed from ckpt args by build()).
- **Smoke first**: `python -c "from anime_upscaler.student import TinySRVGGStudent; ..."` → params 317,300 (<600K) and shape `(1, 3, 192, 192)`. PASS.
- **Command**: `distill.py --arch srvgg --teacher animevideov3 --shortcut-mode nearest --lambda-adv 0.001 --epochs 40 --lr 5e-5` (full command in result doc §1).
- **Success criterion**: D.2 PASS (lap_var >= 35), PSNR >= 28.0, inference <= 200 ms/frame.
- **Result**: **D.2 PASS (lap_var 306.48, 5.3× animevideov3, 14.6× v1)**, **D.1 FAIL (PSNR 27.91 < 29.0)** + **H6 oversharpening halt triggered** (student 27.91 < bicubic 29.45), latency **61 ms/frame** (1.7× faster than v1's 104 ms; PASS with huge margin). NOT PROMOTED.
- **Architecture unlock confirmed**: SRVGG body matches animevideov3's response signal exactly — in-batch lap_var at epoch 1 (adv=0) was already 3236 vs v1's typical ~350 (~9× sharper). The body has a sharpness prior.
- **Overshoot signature**: same as I1 (from-scratch + new-design + adv = oversharpened). Generalization: pick one new thing per run.
- **Output dir**: `runs/distill_i3_srvgg_body/` (preserved per Q6, all 40 epochs).
- **Full doc**: `docs/plans/student_phase4_i3_srvgg_body_result.md`.
- **Lesson**: SRVGG body is correct architecture but **at this compute budget (~17-22 min/run) it cannot be made PSNR-accurate enough to ship**. Phase 5 candidates: warm-start v1 → finetune with SRVGG-distill, OR SRVGG + no-adv + LPIPS-weight=1.0 (push GT anchor instead of adversarial).

### Halt conditions (apply to all phases)

- Val PSNR < 20 dB → mode collapse → halt.
- Val PSNR < 26 dB after 10 epochs → slow → halt.
- EMA PSNR not improving for 10 consecutive epochs → plateau → halt.
- Final lap_var < 35 → D.2 gate fail → don't ship, document.
- **H6 (oversharpening, NEW 2026-09-03)**: Student PSNR < bicubic PSNR on the held-out test split → adversarial is producing hallucinated edges; stop, document, isolate adversarial (Phase 4.I2 path).

### Decision matrix (what to ship after Phase 4)

| I1 | I2 | I3 | Ship |
|---|---|---|---|
| PASS | FAIL | FAIL | I1 |
| FAIL | PASS | FAIL | I2's best |
| FAIL | FAIL | PASS | I3 |
| PASS | PASS | FAIL | I1 (cheapest) |
| PASS | FAIL | PASS | I3 if lap_var higher than I1; else I1 |
| FAIL | PASS | PASS | I3 if lap_var higher than I2; else I2 |
| PASS | PASS | PASS | I3 (most novel) |
| FAIL | FAIL | FAIL | animevideov3 baseline only (already shipped) |

**Current state (2026-09-03, post-I3 — PHASE 4 CLOSED)**: I1 row = D.1 FAIL, I2 row = D.2 FAIL (both sub-runs), I3 row = D.1 FAIL + H6 triggered. Per matrix "FAIL FAIL FAIL → ship animevideov3 baseline only". **No new student shipped**. Production GUI options remain v1 RFDN (real-time, 21.0 lap_var) and animevideov3 SRVGG (quality, 58.1 lap_var). Phase 5 candidates are documented in the I3 result doc §4.4 but are NOT planned.

### Rollback (if all fail)

```bash
git checkout ad14781 -- anime_upscaler/student.py anime_upscaler/distill.py apps/anime_upscaler_gui/anime_upscaler_gui/archs.py
git clean -fd tests/test_shortcut_mode.py tests/test_tiny_srvgg.py
```

Per-epoch ckpts in `runs/` are preserved (Q6 rule).

### Key files to edit (when approved)

| File | Phase | Net LOC |
|---|---|---|
| `anime_upscaler/student.py` | I1, I3 | +60 |
| `anime_upscaler/distill.py` | I1, I2, I3 | +15 |
| `apps/anime_upscaler_gui/anime_upscaler_gui/archs.py` | I3 | +10 |
| `tests/test_shortcut_mode.py` | I1 | +10 |
| `tests/test_tiny_srvgg.py` | I3 | +30 |

### Status snapshot (2026-09-03, post-I3 — Phase 4 closed)

- **Branch**: `feature/phase-1-realtime-4k`
- **HEAD**: post-I3 (commits below, after I2's `7d4e491`)
- **Phase 4 I1 commits**: `bd72dd3` (code + result doc + tests), `924995f` (memory update), `7a7a624` (memory self-record).
- **Phase 4 I2 commits**: `7ac0665` (code + tests), `8929287` (memory update), `7d4e491` (memory self-record).
- **Phase 4 I3 commits**: `7783669` (TinySRVGGStudent + --arch flag + vendored GUI copy + result doc + 9 tests), `c82767e` (memory update), `bcf5354` (memory self-record).
- **Phase 4 I1 outcome**: D.2 PASS (lap_var 109.4), D.1 FAIL (PSNR 27.97). NOT PROMOTED.
- **Phase 4 I2 outcome**: I2a D.2 FAIL (lap_var 20.8 = v1), I2b D.2 FAIL (lap_var 24.2, +15% over v1 but below 35). I2b is the marginal winner (val PSNR +0.022 dB over v1) confirming L_feat was the regression driver in the 2026-08 span+adv run.
- **Phase 4 I3 outcome**: D.2 PASS (lap_var 306.48, 5.3× animevideov3, 14.6× v1), D.1 FAIL (PSNR 27.91), H6 oversharpening halt triggered (student < bicubic). Latency win: 61 ms/frame (1.7× faster than v1). NOT PROMOTED.
- **Per matrix "FAIL FAIL FAIL → ship animevideov3 baseline only"**: **Phase 4 closed**. v1 RFDN + animevideov3 SRVGG remain the production GUI options. No new model shipped.
- **Total Phase 4 wall-time**: ~80 min of training (I1 22 + I2a 17 + I2b 24 + I3 21).
- **Total Phase 4 LOC**: ~+320 (5 new flags / dispatch points, 1 new student class, vendored GUI parity, 4 + 9 new tests).


Rule of thumb: if a file isn't imported by `src/` or referenced by a config, it's likely dead.

## Existing Agent Instructions

- `.windsurf/rules/base.md` — extensive style/debugging/convention rules (v1.8)
- `.windsurf/.agentrc` — references quality gate (partially outdated)
- `.windsurf/workflows/` — many workflow prompt files (mostly generic, not repo-specific)
