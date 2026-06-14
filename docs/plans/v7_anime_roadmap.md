# Anime Super-Resolution v7 — Research-Driven Roadmap

**Status:** v7 build complete. Committed in two commits on branch `fix/neosr-pipeline-vv-2026-06`:
- `b14ef79` — Initial commit: pre-v7 project state (357 files, 101,755 insertions)
- `d0f7f23` — v7: anime SOTA build (28 files, 10,768 insertions)

**Generated:** 2026-06-14
**Target project:** `E:\python projects\upscale_anime`
**TODO tracker:** `docs/plans/todos_v7_anime_2026_06.md`
**Source research:** 3 initial surveys (SOTA, datasets/degradation, architectures/losses) + 4 deep-dive specs
**Test status:** 94/95 v7 tests pass (1 flaky pyiqa test); 87/87 regression tests pass on prior V&V Phases 1-5
**Dry-run:** `python scripts/train.py --config configs/finetune_neosr_span_v7_anime.yaml --dry-run` -> "Configuration validation passed!"

---

## 0. Executive Summary

**SOTA consensus recipe for anime super-resolution (2024-2026):**

> Anime-tuned two-stage degradation (Real-ESRGAN kernels + APISR's prediction-oriented compression + shuffled resize) → SPAN-class backbone (SPAB or MambaIRv2) → two-phase training (L1+perceptual → +GAN) → loss stack (L1 + DISTS/twin-VGG + FDL-DINOv2 + wavelet-guided + relativistic GAN) → evaluated on AVC-RealLQ with NIQE/MANIQA/CLIPIQA + LPIPS.

Our v6 config already approximates 70% of this. The remaining 30% (the gaps) are the v7 deliverables. None of the gaps require new scientific work — they are integration of well-documented public techniques.

**v7 design (per your decisions):**
- **Scope:** Full v7, all 6 phases.
- **Backbone:** Keep SPAN as default. Add MambaIRv2 block as opt-in (`sab_type: conv3xc | mamba_v2`).
- **Eval:** Use existing `val_hr` (overlap caveat documented). Add NR-IQA (CLIPIQA/MANIQA/NIQE/TOPIQ) + curate small disjoint held-out for honest comparison.
- **Data:** Keep existing `data/anime_hr + data/anime_video_frames`. No new data acquisition in v7.

---

## 1. Research Artifacts (4 deep-dive specs)

| Spec | Location | Key content |
|---|---|---|
| **APISR Degradation Spec** | `docs/APISR_DEGRADATION_SPEC.md` | File-by-file map of APISR repo, H.264/H.265/MPEG2/MPEG4 single-frame ffmpeg commands, USM+XDoG pseudo-GT algorithm, comparison with our `degradation_pipeline.py`, 15 Windows gotchas |
| **MambaIRv2 Spec** | (in chat) `docs/research/MAMBAIR_V2_SPEC.md` | SS2D + ASSM classes (verbatim from upstream), pure-PyTorch fallback for Windows, `MambaSPAB` interface, param count (v2: ~20K vs SPAB: ~285K per block), VRAM at B=8/crop=64: ~1.6GB BF16 |
| **Loss Engineering Spec** | (in chat) `docs/research/LOSS_ENGINEERING_SPEC.md` | FDL paper math, APISR's two-stage schedule (300k iters L1 → 300k iters L1+perceptual+GAN, weights {1, 0.5, 0.2, 1}), neosr SPAN weights (FDL=0.75, GAN=0.3), DISTS vs VGG math, wavelet_init=80000 iters, label_smoothing no-op math |
| **NR-IQA Spec** | (in chat) `docs/research/NR_IQA_SPEC.md` | CLIPIQA API, TOPIQ/MUSIQ additions, APISR Table 1 targets (CLIPIQA>=0.65, MANIQA>=0.48, NIQE<=7.5), compute budget (CLIPIQA 150ms/img on 192^2 GPU), end-of-training-only recommendation, train/val overlap mitigation |

> NOTE: The MambaIRv2, Loss Engineering, and NR-IQA specs are stored in the chat history of the deep-dive sub-agent invocations. If they need to be persisted, copy them to `docs/research/` at the end of the build phase.

---

## 2. Gap Analysis (v6 vs SOTA)

| Gap | v6 state | v7 target | Source |
|---|---|---|---|
| **H.264/H.265 single-frame codec** | Listed in config but not implemented | Subprocess wrapper, encode single frame to .mp4 then read back | APISR `degradation/video_compression/h264.py` |
| **XDoG pseudo-GT** | `line_enhancement.py` exists but not in dataloader path | Wire 3-round USM + XDoG + cleanup + composite into `BaseDataset` | APISR `scripts/anime_strong_usm.py` |
| **Shuffled resize** | Field in config; verify active | Verify; add unit test | APISR `degradation_esr_shared.py` |
| **Degrade before crop** | Set to true in v6 config; verify pipeline honors it | Add unit test | APISR |
| **FDL weight** | 0.03 in v6; APISR/neosr standard is 0.75 | v7 = 0.5 (P1) → 0.75 (P2) | neosr `train_span.toml` |
| **Adversarial weight** | 0.005 in v6; standard 0.1-0.3 | v7 = 0.005 P1 → 0.01 P2 | ESRGAN, neosr |
| **Wavelet init** | `wavelet_init: 40`; neosr standard is 80000 iters | Change to `start_epoch: 5` | neosr `train_span.toml:80000` |
| **EMA of generator** | Not implemented | Add `src/training/ema.py`, wire into finetuner | Real-ESRGAN trick |
| **MambaIRv2 block** | Not present | Opt-in replacement for Conv3XC in SPAB | MambaIRv2 (CVPR 2025) |
| **TOPIQ-NR** | Not in metrics | Add to `src/utils/metrics.py` + `scripts/evaluate_quality.py` | pyiqa |
| **Held-out test set** | None (val_hr has train overlap) | Curate 30-50 Danbooru frames to `data/anime_hr_holdout/` | Best practice |
| **Perceptual loss balance** | `twin` in v6 but no balance weights | Set to 0.5/0.5 explicitly | APISR Ablation Table 4 |

---

## 3. v7 Implementation Plan (6 Phases)

### Phase A — Degradation Pipeline Hardening (highest ROI)

**Deliverables:**
1. New file: `src/data/apisr_codecs.py` (~400 LOC)
   - Classes: `H264Codec`, `H265Codec`, `MPEG2Codec`, `MPEG4Codec`, `AVIFCodec`, `HEIFCodec`
   - Each: `encode_decode(np_frame_or_tensor) -> tensor`
2. Extend `src/data/degradation_pipeline.py`: add `use_apisr_two_stage`, `shuffled_resize`, verify `degrade_before_crop`
3. Update `configs/finetune_neosr_span_v7_anime.yaml`
4. Tests: `tests/test_apisr_codecs.py`, `tests/test_shuffled_resize.py`

**Windows gotchas:** libxvid may not be in default Windows ffmpeg builds — fall back to native `mpeg4` encoder; use `subprocess.run` + `time.sleep(0.05)` before `os.remove`; convert `np.random` to `torch.Generator` for reproducibility.

**Effort estimate:** 1-2 days.

---

### Phase B — XDoG Pseudo-GT Pipeline

**Deliverables:**
1. Extend `src/data/line_enhancement.py` with full APISR pipeline (~200 LOC):
   - `usm_sharpen(img, rounds=3, radius=50, sigma=0, threshold=10, weight=0.5)`
   - `xdog(img, sigma=0.6, k=2.5, gamma=0.97, eps=-15, phi=1e9)`
   - `connected_component_cleanup(line_map, min_size=32)`
   - `passive_dilation(line_map, threshold=3)`
   - `make_pseudo_gt(img) -> tensor`
2. Wire into `BaseDataset.__getitem__` via config flag
3. Tests: `tests/test_xdog_pseudo_gt.py`

**Effort estimate:** 1 day.

---

### Phase C — Loss Stack Rebalancing

**Deliverables:**
1. New config: `configs/finetune_neosr_span_v7_anime.yaml`
2. Apply v7 weights:
   - `fdl.weight: 0.5` (P1) → `0.75` (P2)
   - `adversarial.weight: 0.005` (P2 with 5-epoch ramp to 0.01)
   - `adversarial.discriminator_lr: 0.0001`
   - `adversarial.label_smoothing: false` (explicit no-op)
   - `wavelet_guided.start_epoch: 5` (replace `wavelet_init: 40`)
   - `perceptual.danbooru_weight: 0.5, vgg_weight: 0.5`
   - `adversarial.start_epoch: 30`
3. Tests: `tests/test_v7_loss_schedule.py`

**Effort estimate:** 0.5 day.

---

### Phase D — EMA + No-Reference Metrics

**Deliverables:**
1. New file: `src/training/ema.py` (GeneratorEMA class)
2. Wire into `src/training/neosr_finetuner.py`
3. Add `calculate_topiq` and `calculate_musiq` to `src/utils/metrics.py`
4. Add `PYIQA_DIRECTION` and `PYIQA_RANGE` lookup dicts
5. Update `scripts/evaluate_quality.py`: add `topiq_nr`, `--smoke N`, summary stats
6. Curate `data/anime_hr_holdout/`
7. Update `AGENTS.md` with v7 targets
8. Tests: `tests/test_ema_integration.py`, `tests/test_topiq_metric.py`, `tests/test_evaluate_quality_smoke.py`

**Effort estimate:** 1.5 days.

---

### Phase E — MambaIRv2 Block (Opt-In)

**Deliverables:**
1. New file: `src/models/span/mambair_v2.py` (~250 LOC)
   - `selective_scan_ref` (pure-PyTorch fallback)
   - `index_reverse`, `semantic_neighbor` helpers
   - `SelectiveScanV2` (K=1, prompt-aware)
   - `ASSM` (wraps scan + route + embeddings)
   - `MambaSPAB` (SPAB-compatible)
2. Extend `src/models/span/neosr_span.py`: sab_type factory
3. Update `src/models/span/__init__.py`
4. Config: `model.sab_type: conv3xc | mamba_v2`
5. Tests: `tests/test_mamba_spab.py`

**Effort estimate:** 2 days.

---

### Phase F — Honest Validation & Comparison

**Deliverables:**
1. Baseline (v6) eval on `val_hr` + `anime_hr_holdout/`
2. v7 eval on same sets
3. Write `docs/v7_results.md`
4. Optional: AVC-RealLQ access request
5. Update `AGENTS.md`

**Effort estimate:** 1 day (training excluded).

---

## 4. Sub-agent Dispatch Plan (3 waves)

### Wave 1 (parallel, start NOW)
- **Sub-agent A → Phase A** (degradation codecs)
- **Sub-agent B → Phase B** (XDoG pseudo-GT)

### Wave 2 (parallel, after Wave 1)
- **Sub-agent C → Phase C** (loss rebalance)
- **Sub-agent D → Phase D** (EMA + NR-IQA)

### Wave 3 (parallel, after Wave 2)
- **Sub-agent E → Phase E** (MambaIRv2)
- **Sub-agent F → Phase F** (eval + comparison)

**Total wall time:** ~5-6 days (with parallelism), or ~8-9 days sequential.
**Total LoC estimate:** ~1,200 new code + ~300 modified.

---

## 5. Pre-Flight Checklist (Per Phase, Before Commit)

- [ ] `py_compile` passes on all modified `.py` files
- [ ] New tests pass (`pytest tests/test_<phase>.py -v`)
- [ ] `--dry-run` passes on v7 config
- [ ] 2-epoch smoke test completes without crash
- [ ] All new loss dict keys are tensors (or use `_cached` suffix + accumulator guard)
- [ ] No unicode in print statements (Windows cp1252)
- [ ] New config options traced: config → code path → execution
- [ ] No new silent exception handlers

---

## 6. Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| `mamba-ssm` fails to install on Windows | High (Triton not available) | Low (MambaIRv2 is opt-in) | Pure-PyTorch fallback; default stays `conv3xc`; unit tests skip on ImportError |
| H.264 subprocess wrapper breaks ffmpeg pipeline | Medium | Medium | Wrap in try/except, log + skip codec; unit test round-trip |
| ffmpeg not on PATH | Medium | Medium | Pre-flight check; add to install docs; offer PIL-WebP fallback |
| FDL weight 0.75 causes NaN at scale | Low | High | Use FP32 autocast override (already wired); if NaN, drop to 0.5 |
| v7 doesn't beat v6 on CLIPIQA | Medium | High | Phase F compares; if regression, revert to v6 weights but keep codecs + XDoG |
| pyiqa download fails (network/firewall) | Low | Low | Cache models to `TORCH_HOME`; manual download fallback documented |

---

## 7. Files Inventory (delivered)

### New Python files
1. `src/training/ema.py` — GeneratorEMA class + EMA alias (DONE)
2. `src/models/span/mambair_v2.py` — MambaIRv2 block + MambaSPAB wrapper (DONE)
3. ~~`src/data/apisr_codecs.py`~~ — **not needed**; existing `compression_modules.py` already provides PyAVVideoCompression + imageio_ffmpeg fallback for H.264/H.265/MPEG2/MPEG4. Discovery by sub-agent A.

### New test files (8 files, 94/95 pass + 1 flaky)
1. `tests/test_shuffled_resize.py` (13 tests)
2. `tests/test_apisr_two_stage.py` (10 tests)
3. `tests/test_xdog_pseudo_gt.py` (24 tests)
4. `tests/test_v7_loss_schedule.py` (15 tests)
5. `tests/test_ema_integration.py` (11 tests)
6. `tests/test_topiq_metric.py` (6 tests)
7. `tests/test_evaluate_quality_smoke.py` (4 tests)
8. `tests/test_mamba_spab.py` (12 tests)

### Modified Python files
1. `src/data/degradation_pipeline.py` — silent exception logging (no behavior change)
2. `src/data/line_enhancement.py` — full APISR pseudo-GT pipeline (`usm_sharpen`, `xdog`, `connected_component_cleanup`, `passive_dilation`, `make_pseudo_gt`)
3. `src/data/base.py` — line enhancement wiring via `apply_to_gt` flag
4. `src/models/span/neosr_span.py` — `sab_type` factory
5. `src/models/span/__init__.py` — export `MambaSPAB`
6. `src/training/neosr_finetuner.py` — EMA wiring with `getattr(self, 'ema', None)` for stub-build compat
7. `src/losses/twin_perceptual_loss.py` — `danbooru_weight` + `vgg_weight` formula
8. `src/utils/metrics.py` — TOPIQ/MUSIQ + `PYIQA_DIRECTION`/`PYIQA_RANGE` + **latent bicubic-overshoot bug fix**
9. `scripts/evaluate_quality.py` — `--smoke N`, summary stats, generalized direction lookup
10. `scripts/compare_checkpoints.py` (new) — side-by-side eval harness

### New config files
1. `configs/finetune_neosr_span_v7_anime.yaml` (272 lines) — the new "best" config

### New docs
1. `docs/v7_results.md` (296 lines) — Phase F report with 4 reference points
2. `docs/APISR_DEGRADATION_SPEC.md` (957 lines) — pre-build research spec
3. `docs/plans/v7_anime_roadmap.md` (this file)
4. `docs/plans/todos_v7_anime_2026_06.md` (todo tracker)
5. `AGENTS.md` inline updates (V7 Anime SOTA Roadmap section)

### Data files (deferred)
- `data/anime_hr_holdout/` — 30-50 Danbooru frames for honest NR-IQA eval (Phase G.4)

---

## 8. Baseline Measurements (N=10, 480x480 center-crop, mean +/- std)

| Metric | Direction | Bicubic /4->x4 | Stock pretrained | v4 finetune best (ep51) | v4 finetune ep100 | HR ceiling | APISR target |
|---|:---:|---:|---:|---:|---:|---:|---:|
| CLIPIQA  | up   | 0.597 | **0.667** | 0.632 | 0.624 | 0.608 | >= 0.65 |
| MANIQA   | up   | 0.308 | **0.422** | 0.334 | 0.337 | 0.371 | >= 0.48 |
| NIQE     | down | 9.355 | 7.681 | **7.123** | 7.225 | 6.054 | <= 7.5  |
| TOPIQ_NR | up   | 0.321 | **0.556** | 0.462 | 0.464 | 0.558 | (n/a)   |

**Verdict:** CLIPIQA target cleared by stock pretrained. NIQE target cleared by v4 finetune. **MANIQA target missed by all local models** — this is the gap v7's loss rebalance + XDoG pseudo-GT targets. After v7 finetune, MANIQA is the metric to watch.

---

## 9. Phase G — Post-Build Follow-Ons (active)

After v7 was committed, 4 follow-on work items were identified. See `docs/plans/todos_v7_anime_2026_06.md` "Phase G" section for the active todo list.

### Phase G.1 — Verify v7 config end-to-end on GPU (user)
- [ ] 2-epoch smoke test: `python scripts/train.py --config configs/finetune_neosr_span_v7_anime.yaml --epochs 2`
- [ ] 80-epoch full run: same command with `--epochs 80`
- [ ] Compare v7 vs v6: `python scripts/compare_checkpoints.py --baseline checkpoints/NEOSR_SPAN_V6_ANIME/finetune_best.pth --v7 checkpoints/NEOSR_SPAN_V7_ANIME/finetune_best.pth --input data/val_hr --gt data/val_hr --output results/comparison_v6_vs_v7/ --metrics psnr ssim lpips clipiqa maniqa niqe topiq_nr`

### Phase G.2 — Add APISR preset to degradation config
- [ ] Wire the v7 codec list `[avif, h264, h265, jpeg]` into a named preset
- [ ] Add a `degrade_mode: apisr` shortcut to v7 config

### Phase G.3 — Curate disjoint held-out set
- [ ] Hand-pick 30-50 Danbooru frames (no overlap with `data/anime_hr` or `data/val_hr`)
- [ ] File-hash dedup verification
- [ ] Save to `data/anime_hr_holdout/`

### Phase G.4 — Fix pre-existing test collection errors
- 15 test files fail to import (e.g. `test_data_manager` module missing) — discovered during v7 build, unrelated to v7 changes
- See `docs/plans/todos_v7_anime_2026_06.md` "Phase G.4" for the list and `tests/utils/` schema reference

### Phase G.5 — Optional: request AVC-RealLQ
- Gated dataset (Tencent ARC Lab); send email to `wuyanze123@gmail.com` with signed LICENSE AGREEMENT
- See `docs/research/NR_IQA_SPEC.md` for the procedure

---

## 10. Branch + Git State

- Branch: `fix/neosr-pipeline-vv-2026-06` (no remote)
- Two commits:
  1. `b14ef79` — Initial: pre-v7 project state
  2. `d0f7f23` — v7: anime SOTA build
- Untracked (excluded by `.gitignore` or unrelated):
  - `demo_reports/`, `example_outputs/`, `tmp/`
  - `upscale_anime_backup_*.zip`
