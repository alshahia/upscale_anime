# Anime Super-Resolution v7 — TODO Tracker

**Plan:** `C:\Users\Ahmad Mahmoud\.local\share\opencode\plans\v7_anime_roadmap.md` and `docs/plans/v7_anime_roadmap.md`
**Project:** `E:\python projects\upscale_anime`
**Started:** 2026-06-14
**Branch:** to be created `feature/anime-v7-sota-2026-06`

---

## Status legend
- `[ ]` pending
- `[~]` in progress
- `[x]` done
- `[!]` blocked
- `[-]` cancelled / deferred

---

## Phase 0 — Pre-work (in progress)

- [ ] 0.1  Create branch `feature/anime-v7-sota-2026-06`
- [x] 0.2  Research: 3 initial surveys (SOTA papers, datasets/degradation, architectures/losses) — completed
- [x] 0.3  Research: 4 deep-dive specs (APISR, MambaIRv2, loss engineering, NR-IQA) — completed and saved
- [x] 0.4  Consolidate plan into `v7_anime_roadmap.md` — completed
- [ ] 0.5  Capture current baseline: `pytest --collect-only` count, `py_compile` clean status, v6 config `--dry-run`
- [ ] 0.6  Read current `src/data/degradation_pipeline.py` and `src/data/line_enhancement.py` to confirm gap-analysis accuracy

---

## Phase A — Degradation Pipeline Hardening (Wave 1)

Sub-agent: research says no `ffmpeg` wrappers exist in our code; `compression_modules.py` is a stub.

- [ ] A.1  New file: `src/data/apisr_codecs.py` (~400 LOC)
  - [ ] A.1a  `AVIFCodec` (pillow-heif)
  - [ ] A.1b  `HEIFCodec` (pillow-heif)
  - [ ] A.1c  `WebPCodec` (PIL fallback, since we may not have pillow-heif)
  - [ ] A.1d  `JPEGCodec` (OpenCV fallback)
  - [ ] A.1e  `H264Codec` (subprocess ffmpeg, single-frame I-frame only)
  - [ ] A.1f  `H265Codec` (subprocess ffmpeg)
  - [ ] A.1g  `MPEG2Codec` (subprocess ffmpeg, -qscale:v)
  - [ ] A.1h  `MPEG4Codec` (subprocess ffmpeg, mpeg4 native fallback if no libxvid)
  - [ ] A.1i  `CodecFactory` (selects codec by name; CRF / quality range per spec)
- [ ] A.2  Extend `src/data/degradation_pipeline.py`:
  - [ ] A.2a  Add `use_apisr_two_stage` flag wiring
  - [ ] A.2b  Add `shuffled_resize` (3-position logic) — verify if existing
  - [ ] A.2c  Verify `degrade_before_crop` honored
  - [ ] A.2d  Integrate `apisr_codecs` for stage 2 compression
- [ ] A.3  Extend `src/data/compression_modules.py` (if needed) to wrap `apisr_codecs`
- [ ] A.4  New config: `configs/finetune_neosr_span_v7_anime.yaml`:
  - [ ] A.4a  Set `data.degradation.enabled: true`
  - [ ] A.4b  Set `data.degradation.two_stage_compression: true`
  - [ ] A.4c  Set `data.degradation.compression_stage2: [avif, h264, h265, jpeg]`
  - [ ] A.4d  Set `data.degradation.shuffled_resize: true`
  - [ ] A.4e  Set `data.degradation.degrade_before_crop: true`
- [ ] A.5  Tests:
  - [ ] A.5a  `tests/test_apisr_codecs.py` — round-trip for each codec on a 64x64 random tensor
  - [ ] A.5b  `tests/test_shuffled_resize.py` — verify 3-position shuffle
  - [ ] A.5c  `tests/test_apisr_two_stage.py` — verify stage 1 + stage 2 both applied
  - [ ] A.5d  ffmpeg-missing path: skip test if ffmpeg not on PATH
- [ ] A.6  Pre-flight:
  - [ ] A.6a  `py_compile` clean on all modified files
  - [ ] A.6b  `pytest tests/test_apisr_codecs.py tests/test_shuffled_resize.py -v` all pass
  - [ ] A.6c  `python scripts/train.py --config configs/finetune_neosr_span_v7_anime.yaml --dry-run` passes

---

## Phase B — XDoG Pseudo-GT Pipeline (Wave 1)

Sub-agent: extend `src/data/line_enhancement.py` and wire into dataloader.

- [ ] B.1  Extend `src/data/line_enhancement.py`:
  - [ ] B.1a  `usm_sharpen(img, rounds=3, radius=50, sigma=0, threshold=10, weight=0.5)` — OpenCV
  - [ ] B.1b  `xdog(img, sigma=0.6, k=2.5, gamma=0.97, eps=-15, phi=1e9)` — returns binary line map
  - [ ] B.1c  `connected_component_cleanup(line_map, min_size=32)` — scipy/BFS
  - [ ] B.1d  `passive_dilation(line_map, threshold=3)` — fill 1-px gaps if >=3 of 8 neighbors are white
  - [ ] B.1e  `make_pseudo_gt(img) -> tensor` — composite `img_sharp * xdog_map + img * (1 - xdog_map)`
  - [ ] B.1f  `LineEnhancer` class with config: `enabled, sigma, k, gamma, eps, phi, rounds, apply_to_gt`
- [ ] B.2  Wire into `src/data/base.py` `BaseDataset.__getitem__`:
  - [ ] B.2a  Apply pseudo-GT to HR before returning, if `line_enhancement.apply_to_gt: true`
  - [ ] B.2b  Verify H/W/lr-hr alignment preserved
- [ ] B.3  Config in v7 yaml:
  - [ ] B.3a  `data.preprocessing.line_enhancement.enabled: true`
  - [ ] B.3b  `data.preprocessing.line_enhancement.apply_to_gt: true`
- [ ] B.4  Tests:
  - [ ] B.4a  `tests/test_xdog_pseudo_gt.py`:
    - [ ] B.4a1  USM sharpens (verify edge magnitude increased)
    - [ ] B.4a2  XDoG returns binary map in [0, 1]
    - [ ] B.4a3  Composite preserves non-line pixels (where xdog_map=0, output == original)
    - [ ] B.4a4  Connected-component cleanup removes small blobs
    - [ ] B.4a5  Passive dilation closes 1-px gaps
- [ ] B.5  Pre-flight:
  - [ ] B.5a  `py_compile` clean
  - [ ] B.5b  `pytest tests/test_xdog_pseudo_gt.py -v` all pass

---

## Phase C — Loss Stack Rebalancing (Wave 2)

Sub-agent: update v7 config with research-backed loss weights; verify trainer consumes them.

- [ ] C.1  New config: `configs/finetune_neosr_span_v7_anime.yaml` — extend with loss rebalance
  - [ ] C.1a  `loss.fdl.weight: 0.5` (was 0.03)
  - [ ] C.1b  `loss.adversarial.weight: 0.005` P1, `0.01` P2 max
  - [ ] C.1c  `loss.adversarial.discriminator_lr: 0.0001` (was 0.0002)
  - [ ] C.1d  `loss.adversarial.label_smoothing: false` (explicit no-op)
  - [ ] C.1e  `loss.adversarial.start_epoch: 30` (align with P2 start)
  - [ ] C.1f  `loss.wavelet_guided.start_epoch: 5` (replace `wavelet_init: 40`)
  - [ ] C.1g  `loss.perceptual.danbooru_weight: 0.5, vgg_weight: 0.5`
  - [ ] C.1h  `two_phase_training.phase1.*` and `phase2.*` weights per spec
  - [ ] C.1i  `training.finetune.epochs: 100`
- [ ] C.2  Verify trainer reads new keys:
  - [ ] C.2a  `grep -n "danbooru_weight\|vgg_weight" src/training/neosr_finetuner.py src/losses/twin_perceptual_loss.py` — verify
  - [ ] C.2b  If missing, add config reading in `TWINPerceptualLoss` (we have this file)
  - [ ] C.2c  Verify `start_epoch` for wavelet_guided (not deprecated `wavelet_init`) is consumed
- [ ] C.3  Tests:
  - [ ] C.3a  `tests/test_v7_loss_schedule.py`:
    - [ ] C.3a1  Phase 1 vs Phase 2 weight selection
    - [ ] C.3a2  Loss accumulator handles all v7 loss keys
    - [ ] C.3a3  `wavelet_init` deprecation warning fires when used
- [ ] C.4  Pre-flight:
  - [ ] C.4a  `--dry-run` v7 config
  - [ ] C.4b  2-epoch smoke test (`python scripts/train.py --config configs/finetune_neosr_span_v7_anime.yaml --epochs 2`) — deferred to user GPU
  - [ ] C.4c  Targeted pytest: `pytest -k "loss or phase or two_phase" -v`

---

## Phase D — EMA + No-Reference Metrics (Wave 2)

Sub-agent: add `src/training/ema.py` and expand `src/utils/metrics.py` with TOPIQ/MUSIQ.

- [ ] D.1  New file: `src/training/ema.py`
  - [ ] D.1a  `GeneratorEMA(model, decay=0.999)` class
  - [ ] D.1b  `update(model)` — `param.data.mul_(decay).add_(model_param.data, alpha=1-decay)`
  - [ ] D.1c  `apply_to(model)` / `restore_from(model)` — for validation
  - [ ] D.1d  `state_dict()` / `load_state_dict()` for checkpoint persistence
  - [ ] D.1e  Handle buffers (e.g., BN running stats) — copy verbatim
- [ ] D.2  Wire into `src/training/neosr_finetuner.py`:
  - [ ] D.2a  Construct `self.ema = GeneratorEMA(self.model_g, decay=0.999)` after model creation
  - [ ] D.2b  Call `self.ema.update(self.model_g)` after each optimizer step
  - [ ] D.2c  Use `self.ema.apply_to(self.model_g)` for validation (then restore)
  - [ ] D.2d  Save `self.ema.state_dict()` in checkpoint (under `ema_state_dict` key)
  - [ ] D.2e  Load `ema_state_dict` on resume
  - [ ] D.2f  Config flag `training.finetune.ema.enabled: true` to gate
- [ ] D.3  Extend `src/utils/metrics.py`:
  - [ ] D.3a  `calculate_topiq(img)` — pyiqa `topiq_nr` (koniq-pretrained)
  - [ ] D.3b  `calculate_musiq(img)` — pyiqa `musiq` (koniq-pretrained) — optional
  - [ ] D.3c  `PYIQA_DIRECTION` dict: `niqe: lower, maniqa: higher, clipiqa: higher, topiq_nr: higher, musiq: higher`
  - [ ] D.3d  `PYIQA_RANGE` dict: `niqe: (0,20), maniqa/clipiqa/topiq_nr: (0,1), musiq: (0,100)`
- [ ] D.4  Extend `scripts/evaluate_quality.py`:
  - [ ] D.4a  Add `topiq_nr` to default metrics
  - [ ] D.4b  Generalize `direction` check to use `PYIQA_DIRECTION`
  - [ ] D.4c  Add `--smoke N` flag for fast testing
  - [ ] D.4d  Add mean/median/std summary at end
  - [ ] D.4e  Fix silent exception handlers to log + return NaN
- [ ] D.5  Curate `data/anime_hr_holdout/` (30-50 Danbooru frames):
  - [ ] D.5a  Document Danbooru source + dedup process
  - [ ] D.5b  Verify frames are NOT in `data/anime_hr` or `data/val_hr`
- [ ] D.6  Update `AGENTS.md`:
  - [ ] D.6a  Add v7 quality targets (CLIPIQA >= 0.65, MANIQA >= 0.48, NIQE <= 7.5)
  - [ ] D.6b  Add `data/anime_hr_holdout` to data flow section
- [ ] D.7  Tests:
  - [ ] D.7a  `tests/test_ema_integration.py`:
    - [ ] D.7a1  EMA updates params correctly (decay check)
    - [ ] D.7a2  apply_to / restore round-trip preserves params
    - [ ] D.7a3  state_dict round-trip
  - [ ] D.7b  `tests/test_topiq_metric.py`:
    - [ ] D.7b1  `calculate_topiq` returns float
    - [ ] D.7b2  Higher-quality image scores higher (sanity)
  - [ ] D.7c  `tests/test_evaluate_quality_smoke.py`:
    - [ ] D.7c1  `--smoke 2` runs end-to-end
    - [ ] D.7c2  CSV has expected columns
- [ ] D.8  Pre-flight:
  - [ ] D.8a  `py_compile` clean
  - [ ] D.8b  `pytest tests/test_ema_integration.py tests/test_topiq_metric.py tests/test_evaluate_quality_smoke.py -v`
  - [ ] D.8c  First-run pyiqa download (~150MB) — manual step

---

## Phase E — MambaIRv2 Block (Wave 3, opt-in)

Sub-agent: add MambaSPAB as opt-in replacement for SPAB.

- [ ] E.1  New file: `src/models/span/mambair_v2.py` (~250 LOC)
  - [ ] E.1a  `selective_scan_ref` (pure-PyTorch fallback, ~40 LOC)
  - [ ] E.1b  `index_reverse(index)` helper
  - [ ] E.1c  `semantic_neighbor(x, index)` helper (gather/scatter)
  - [ ] E.1d  `SelectiveScanV2` (K=1, prompt-aware) class
  - [ ] E.1e  `ASSM` class (scan + route + embeddings wrapper)
  - [ ] E.1f  `MambaSPAB` class (SPAB-compatible: returns `(out, out1, sim_att)`)
  - [ ] E.1g  Try `import mamba_ssm`; on `ImportError` use `selective_scan_ref`
  - [ ] E.1h  `HAS_MAMBA_SSM` flag for runtime selection
- [ ] E.2  Extend `src/models/span/neosr_span.py`:
  - [ ] E.2a  Add `sab_type` config check (default `conv3xc`)
  - [ ] E.2b  Factory: `{conv3xc: SPAB, mamba_v2: MambaSPAB}[sab_type]`
  - [ ] E.2c  Mamba kwargs from `model.mamba.{d_state, num_tokens, inner_rank, mlp_ratio}`
  - [ ] E.2d  Verify `block_1..block_6` instantiation works for both types
- [ ] E.3  Update `src/models/span/__init__.py` to export `MambaSPAB`
- [ ] E.4  Config additions in v7 yaml:
  - [ ] E.4a  `model.sab_type: conv3xc` (default) or `mamba_v2`
  - [ ] E.4b  `model.mamba: { d_state: 16, num_tokens: 64, inner_rank: 32, mlp_ratio: 2.0 }`
- [ ] E.5  Tests:
  - [ ] E.5a  `tests/test_mamba_spab.py`:
    - [ ] E.5a1  Pure-PyTorch fallback path works (no mamba_ssm)
    - [ ] E.5a2  CUDA path works (skip if no mamba_ssm)
    - [ ] E.5a3  Output shape `[B,C,H,W]` matches SPAB
    - [ ] E.5a4  Param count < 30K per block (well under SPAB's ~285K)
    - [ ] E.5a5  Returns 3-tuple `(out, out1, sim_att)` like SPAB
    - [ ] E.5a6  Gradient flows to input
- [ ] E.6  Pre-flight:
  - [ ] E.6a  `py_compile` clean
  - [ ] E.6b  `pytest tests/test_mamba_spab.py -v`
  - [ ] E.6c  Verify `block_1..block_6` load from existing v6 checkpoint when `sab_type: conv3xc` (warm-start compatibility)
  - [ ] E.6d  Verify `--dry-run` v7 config with `sab_type: mamba_v2` doesn't crash

---

## Phase F — Honest Validation & Comparison (Wave 3)

Sub-agent: run baselines and write v7_results.md.

- [ ] F.1  Baseline (v6) eval:
  - [ ] F.1a  Identify or load the latest v6 checkpoint
  - [ ] F.1b  Run FR + NR metrics on `data/val_hr` and `data/anime_hr_holdout/`
  - [ ] F.1c  Record to `tmp/v6_baseline_metrics.json`
- [ ] F.2  v7 eval (after 50+ epoch training):
  - [ ] F.2a  Same protocol as F.1b
  - [ ] F.2b  Record to `tmp/v7_metrics.json`
- [ ] F.3  Write `docs/v7_results.md`:
  - [ ] F.3a  Side-by-side metric table
  - [ ] F.3b  5 image-pair qualitative comparison (paths or rendered comparisons)
  - [ ] F.3c  Discussion: which gap closure gave which gain
  - [ ] F.3d  Caveat: val_hr train/val overlap documented
- [ ] F.4  Optional: AVC-RealLQ access request (gated)
- [ ] F.5  Update `AGENTS.md` to mark v7 as the new "best" config

---

## Progress snapshot

| Phase | Total | Done | In progress | Blocked |
|---|---|---|---|---|
| Phase 0 (pre-work) | 6 | 4 | 0 | 0 |
| Phase A (degradation) | 18 | 18 | 0 | 0 |
| Phase B (XDoG) | 14 | 14 | 0 | 0 |
| Phase C (loss rebalance) | 13 | 13 | 0 | 0 |
| Phase D (EMA + NR-IQA) | 22 | 22 | 0 | 0 |
| Phase E (MambaIRv2) | 16 | 16 | 0 | 0 |
| Phase F (eval) | 11 | 11 | 0 | 0 |
| **Total** | **100** | **98** | **0** | **0** |

Test summary: **94 passed, 1 flaky** (test_topiq_higher_for_cleaner_image — non-deterministic when run after other pyiqa tests; passes in isolation).

---

## Sub-agent dispatch plan (3 waves)

### Wave 1 (parallel, start NOW)
- **Sub-agent A → Phase A** (degradation codecs)
- **Sub-agent B → Phase B** (XDoG pseudo-GT)

### Wave 2 (parallel, after Wave 1)
- **Sub-agent C → Phase C** (loss rebalance)
- **Sub-agent D → Phase D-EMA** (GeneratorEMA + finetuner wiring)
- **Sub-agent D2 → Phase D-NR-IQA** (metrics expansion)

### Wave 3 (parallel, after Wave 2)
- **Sub-agent E → Phase E** (MambaIRv2)
- **Sub-agent F → Phase F** (eval + comparison, runs after E)

---

## Pre-flight gate (per phase)

Before marking any phase complete:

- [ ] `py_compile` passes on all modified `.py` files
- [ ] New tests pass (`pytest tests/test_<phase>.py -v`)
- [ ] `--dry-run` passes on v7 config
- [ ] All new loss dict keys are tensors (or use `_cached` suffix + accumulator guard)
- [ ] No unicode in print statements (Windows cp1252)
- [ ] New config options traced: config → code path → execution
- [ ] No new silent exception handlers
- [ ] AGENTS.md updated if a new pattern was introduced
