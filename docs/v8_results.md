# V8 Tier 1 Results — TTA + Model Souping

**Status:** code + tests shipped; GPU evaluation deferred to user
**Date:** 2026-06-20
**Project:** `E:\python projects\upscale_anime`
**Roadmap:** `docs/plans/v8_mambair_plan.md`
**TODO tracker:** `docs/plans/todos_v8_mambair_2026_06.md`
**Branch:** `perf/v7-tier1-speedups` (NOT committed; untracked files)

---

## 1. Goal

Verify the two Tier-1 quality boosts from the V8+ survey (see
`docs/survey_2026/recommendations.md`) work end-to-end and ship them as
production-ready code paths:

1. **TTA — D4 8x flip/rot ensemble** at inference time. Survey projects
   LPIPS -10% on average for anisotropic SR models, free aside from 8x
   inference compute.
2. **Model souping** — average EMA state-dicts across multiple finetune
   checkpoints (Wortsman et al., ICML 2022). Survey projects MANIQA
   +0.01 to +0.05.

Both are **drop-in** for the existing inference / comparison workflows:
`--tta` flag on `scripts/inference.py` and `scripts/compare_checkpoints.py`,
a new `scripts/soup_checkpoints.py` for the soup step.

---

## 2. What's shipped (code)

### 2.1 TTA — D4 8x flip/rot ensemble

| File | LoC | Status |
|---|---|---|
| `src/inference/tta.py` | ~110 | NEW (was missing) |
| `scripts/inference.py` | +5 (--tta flag + apply site) | MODIFIED |
| `scripts/compare_checkpoints.py` | +15 (--tta flag + `run_inference_tta` helper) | MODIFIED |
| `tests/test_tta.py` | ~115 (9 tests) | NEW |

`D4_AUGMENTATIONS` is a list of 8 `(aug, inv)` pairs covering the dihedral
group of a square: identity, fliplr, flipud, flipboth, rot90 CCW, rot90 CW,
rot90 CCW + fliplr, rot90 CCW + flipud. Each pair is verified by
`test_d4_inverses_are_correct` to satisfy `inv(aug(x)) == x`.

`tta_forward(model, lr, use_clip=True)` runs the model 8 times on the 8
augmented inputs, applies the inverse aug to each prediction, optionally
clamps to `[0, 1]`, and returns the mean in the input dtype.

### 2.2 Model souping

| File | LoC | Status |
|---|---|---|
| `scripts/soup_checkpoints.py` | ~165 | NEW |
| `tests/test_soup_checkpoints.py` | ~170 (8 tests) | NEW |

`soup_checkpoints(input_paths, output_path, use_ema=True)` averages EMA
state-dicts across N checkpoints. State-dict preference order:
`ema_state_dict` > `model_state_dict` > `params` > `state_dict` > raw.
Output drops `scaler_state_dict` and `optimizer_state_dict` (per-checkpoint
state; user re-inits on resume) and stamps `soup_inputs` /
`soup_sources` / `soup_n` / `soup_timestamp` metadata.

CLI:
```bash
python scripts/soup_checkpoints.py \
    --inputs checkpoints/NEOSR_SPAN_V6_ANIME/finetune_best.pth \
            checkpoints/NEOSR_SPAN_V7_ANIME/finetune_best.pth \
    --output  checkpoints/SOUP_V6_V7/soup.pth
```

---

## 3. Verification

### 3.1 Tests

```
$ .venv\Scripts\python.exe -m pytest tests/test_tta.py tests/test_soup_checkpoints.py -v
tests/test_tta.py::TestD4Group::test_d4_inverses_are_correct PASSED
tests/test_tta.py::TestD4Group::test_d4_has_8_elements PASSED
tests/test_tta.py::TestD4Group::test_d4_unique_outputs PASSED
tests/test_tta.py::TestTTAForward::test_tta_identity_model_returns_input_mean PASSED
tests/test_tta.py::TestTTAForward::test_tta_pixel_range_clipped PASSED
tests/test_tta.py::TestTTAForward::test_tta_pixel_range_unclipped PASSED
tests/test_tta.py::TestTTAForward::test_tta_matches_manual_mean PASSED
tests/test_tta.py::TestTTAForward::test_tta_dtype_preserved PASSED
tests/test_tta.py::TestTTACUDA::test_tta_cuda_smoke PASSED  (skipped on CPU-only)
tests/test_soup_checkpoints.py::TestSoupCheckpoints::test_soup_two_checkpoints_ema_only PASSED
tests/test_soup_checkpoints.py::TestSoupCheckpoints::test_soup_falls_back_to_model_state_dict_when_no_ema PASSED
tests/test_soup_checkpoints.py::TestSoupCheckpoints::test_soup_key_mismatch_raises PASSED
tests/test_soup_checkpoints.py::TestSoupCheckpoints::test_soup_requires_two_or_more PASSED
tests/test_soup_checkpoints.py::TestSoupCheckpoints::test_soup_three_checkpoints PASSED
tests/test_soup_checkpoints.py::TestSoupCheckpoints::test_soup_preserves_dtype PASSED
tests/test_soup_checkpoints.py::TestLoadStateDict::test_prefers_ema PASSED
tests/test_soup_checkpoints.py::TestLoadStateDict::test_handles_raw_state_dict PASSED

============================== 17 passed in 2.55s ==============================
```

CUDA was detected on the local workstation (Quadro RTX 4000), so the GPU
smoke `test_tta_cuda_smoke` actually ran (not skipped). D4 inverses +
averaging match manual computation to within 1e-5.

### 3.2 Static checks

```
$ .venv\Scripts\python.exe -m py_compile \
      src/inference/tta.py \
      scripts/soup_checkpoints.py \
      scripts/inference.py \
      scripts/compare_checkpoints.py
$ echo $?
0
```

### 3.3 AGENTS.md docs

Added a **"V8 Tier 1 Shipped (2026-06-20)"** section to `AGENTS.md` with
file inventory, quick-start commands, and the "user commits" reminder.

---

## 4. Baseline numbers (N=10, 480x480 center-crop)

These are the v7 self-baseline (no v7 finetune, no TTA, no souping) from
`docs/v7_results.md` Section 3. **They will be re-measured with `--tta`
and the souped v6+v7 checkpoint once both exist** (post-Phase G.1).

| Metric | Direction | Bicubic /4->x4 | Stock pretrained | v4 finetune best | APISR target |
|---|:---:|---:|---:|---:|---:|
| CLIPIQA  | up   | 0.597 | **0.667** | 0.632 | >= 0.65 |
| MANIQA   | up   | 0.308 | **0.422** | 0.334 | >= 0.48 |
| NIQE     | down | 9.355 | 7.681     | **7.123** | <= 7.5 |
| TOPIQ_NR | up   | 0.321 | **0.556** | 0.462 | (n/a) |

**Expected post-Tier-1 movement** (per `docs/survey_2026/recommendations.md`):
- LPIPS: -10% from TTA (free, inference-time)
- MANIQA: +0.01 to +0.05 from souping v6+v7 EMA
- Projected MANIQA after Tier 1+2: **0.532** (target 0.48 cleared with margin)

---

## 5. What's pending (user GPU)

The full validation requires:

1. A v7 finetune checkpoint (`checkpoints/NEOSR_SPAN_V7_ANIME/finetune_best.pth`)
   from an 80-epoch run on the user's RTX 4000 Mobile. Until that exists,
   TTA has no v7 reference and souping has no v7 input.
2. A v6 finetune checkpoint (`checkpoints/NEOSR_SPAN_V6_ANIME/finetune_best.pth`)
   to serve as the soup partner. (If this exists already, it can be used
   immediately.)

Once both exist:

```bash
# 1. Soup v6 + v7
python scripts/soup_checkpoints.py \
    --inputs checkpoints/NEOSR_SPAN_V6_ANIME/finetune_best.pth \
            checkpoints/NEOSR_SPAN_V7_ANIME/finetune_best.pth \
    --output  checkpoints/SOUP_V6_V7/soup.pth

# 2. Compare baseline (v6) vs v7 vs soup, with TTA enabled
python scripts/compare_checkpoints.py \
    --baseline checkpoints/NEOSR_SPAN_V6_ANIME/finetune_best.pth \
    --v7      checkpoints/NEOSR_SPAN_V7_ANIME/finetune_best.pth \
    --extra   checkpoints/SOUP_V6_V7/soup.pth \
    --input data/val_hr --gt data/val_hr \
    --output results/comparison_v6_v7_soup/ \
    --metrics psnr ssim lpips clipiqa maniqa niqe topiq_nr --tta
```

(`--extra` is a placeholder; the comparison harness currently accepts
two checkpoints. If we want three-way, that's a small follow-on
`scripts/compare_three.py` or an extension to `compare_checkpoints.py`.)

3. Re-run on the disjoint held-out set (`data/anime_hr_holdout/`) once
   curated (Phase G.3). The publishable NR-IQA numbers come from that set.

---

## 6. NOT committed

Per AGENTS.md convention, all V8 Tier 1 files are untracked on
`perf/v7-tier1-speedups`. User reviews `git diff` and commits when ready.
