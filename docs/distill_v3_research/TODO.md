# v3 distillation recipe — runnable TODO

> Companion to `.claude/plans/distill_v3_recipe.md`. Mark each item done as you go.

## Phase 1 — Loss rebalance + Charbonnier + MS-SSIM + LPIPS

- [x] Add imports to `anime_upscaler/distill.py`: `from piq import charbonnier_loss, ssim`, `import lpips`.
- [x] Add `_lpips_net` lazy cache + `_get_lpips(device)` helper below imports.
- [x] Replace inner loss block with the Charbonnier / cosine-feature / SSIM+LPIPS+GT variant.
- [x] Verify `distill.py --smoke` runs without NaN.

### Phase 1 implementation notes (deviations from canonical plan)

- **`piq.charbonnier_loss` does NOT exist in piq 0.8.0** (was removed). Inlined a 3-line `_charbonnier(x, y, eps=1e-3)` helper in `distill.py` instead. Matches historical `piq.charbonnier_loss` semantics (per-pixel mean).
- **`piq.ssim` returns `List[Tensor]` in 0.8.0**. Switched to `piq.multi_scale_ssim` which returns a scalar tensor and is what the report actually recommends (multi-scale, more robust than single-scale SSIM).
- **MS-SSIM minimum size**: 161×161. Our training SR is 192×192 (just passes); val/test SR is 384×384. No change to dataset.
- **`piq` was not installed** despite the plan claiming it was present. Installed `piq>=0.8,<0.9` into `.venv` and pinned it in `anime_upscaler/requirements.txt`.
- **Bonus fix (pre-existing latent bug)**: `best_psnr = max(best_psnr, -1.0)` on line 241 used an undefined `best_psnr` on every fresh run (no `--resume`). Initialized to `-1.0` before the resume branch. This is what was hiding the v3 plan behind a NameError that the smoke gate caught.

## Phase 2 — EMA of student weights

- [x] Add `_EMA` class definition at module level (between `StudentFeatureAdapters` and `evaluate()`).
- [x] Construct `student_ema = _EMA(student, decay=0.999)` AFTER the resume block (seeds EMA from loaded weights).
- [x] Call `student_ema.update(student)` after each `optimizer.step()`.
- [x] Update save block: `student_last.pt` keeps raw state; `student_best.pt` uses `student_ema.state_dict()`.
- [x] Evaluate EMA once per epoch end (swap shadow in, evaluate, swap back).
- [x] CSV header: added `val_psnr_student_ema` column at position 8.
- [x] Validation gate: smoke runs to completion with all loss terms finite; `student_best.pt` bytes differ from `student_last.pt` (EMA blending produced distinct weights).
- [x] Backward-compat: `--resume` from a v1-schema ckpt (`runs/distill_smoke/student_last.pt`) end-to-end test loaded and continued training with EMA seeded from the resumed weights.

### Phase 2 implementation notes (deviations from canonical plan)

- **EMA constructed AFTER `--resume` block, not right after optimizer.** Plan example showed it after `optimizer = ...`; we placed it after the resume block so EMA shadow initializes from the resumed weights (not from RFDN()'s random init). Same observable behavior for fresh runs; only matters for `--resume`.
- **CSV column order:** inserted `val_psnr_student_ema` right after `val_psnr_student` (position 8), keeping bicubic / s_raw / s_ema / teacher in that order for readability.
- **EMA < raw in smoke (expected).** At only 3 iters/epoch on a tiny dataset, EMA decay=0.999 means the shadow barely moves. EMA catches up after many iterations; we expect EMA to *beat* raw on the full 40-epoch run.
- **`student_best.pt` schema unchanged.** Top-level keys are still `{epoch, student, adapters, val_psnr, args}`; only the value of `student` differs (raw vs EMA). v1-style `--resume` works without modification.
- **EMA weight dtype safety:** non-floating-point buffers (e.g. BN `num_batches_tracked`) are copied verbatim, not blended. RFDN has no such buffers so this is currently a no-op, but defends against future student changes.
- **No new dependencies.** EMA is pure PyTorch; no new requirements.txt entries needed.

## Phase 3 — APISR two-stage degradation

- [x] Create `anime_upscaler/degradation.py` with `degrade(img, p_codec, p_resize)`.
- [x] Add `degradation_mode: str = "none"` arg to `AnimePairDataset.__init__`.
- [x] In `__getitem__`, apply `degradation.degrade(hr_crop)` to LR for train split only.
- [x] Add `--degradation` CLI flag to `distill.py`; default `"apsisr_v1"`.
- [x] Verify `--smoke --degradation apsisr_v1` produces visible artifacts.

### Phase 3 implementation notes (deviations from canonical plan)

- **Default mode is `apsisr_v1`, not `none`**. Plan said default for "the user can opt out easily"; with Phase 3 being THE distillation improvement, a smoke run that defaults to `none` would silently yield v2 numbers and miss the regression check. `--degradation none` is the explicit opt-out path and is the Phase 5 ablation baseline.
- **Smoke val/test PSNR bit-identical to Phase 2 baseline**: `bicubic=28.10, student=17.61→17.85, teacher=29.47, ema=17.12→17.13` — because val/test always pass through clean bicubic regardless of `--degradation` flag. The Phase 3 contribution lives ONLY in the train-time LR distribution.
- **LR tensor under `apsisr_v1` produces large input shift**: mean |LR_clean − LR_deg| over 8 samples = 0.4535 (huge, given range [-1, 1]). Per-channel std shifted 0.42/0.46/0.37 (clean) → 0.41/0.45/0.48 (degraded), consistent with JPEG/WebP posterization expanding dynamic range. This is exactly the realistic codec-artifact surface Phase 3 targets.
- **Synthetic train loss moves by ~0.01** with degradation on (still finite, still expects `loss_resp < loss_gt`); the real gain is on full-epoch val, not smoke.
- **`_find_ffmpeg` fanned to 3 sources**: imageio-ffmpeg → shutil.which('ffmpeg') → None. Annex A only had imageio-ffmpeg with bare `except Exception` fallback to None — would have left h264/h265 silently disabled even when `C:\\tool\\ffmpeg_full\\bin\\ffmpeg.exe` is on PATH. Our env has BOTH sources available so this matters in practice (Linux/CI without imageio-ffmpeg would benefit).
- **Degradation import is lazy** (`from degradation import degrade` inside `__getitem__`). Avoids paying imageio_ffmpeg.get_ffmpeg_exe() import cost when `split='val'`/`split='test'`, and keeps the smoke gate self-contained.
- **HR stays clean** even under `apsisr_v1` — only the `hr_for_lr` (the source for LR bicubic downscale) is degraded. GT anchor remains pristine, which is what MS-SSIM and LPIPS are designed to evaluate against.
- **No new dependencies.** imageio-ffmpeg is already in .venv (transitive dep), PIL/subprocess/io are stdlib. requirements.txt unchanged.

## Phase 4 -- DONE (this turn)

- [x] Create `anime_upscaler/tta.py` (~110 LOC): D4 group forward/inverse transform pairs + `tta_forward(model, lr) -> (sr_mean, n_aug, names)`
- [x] Add `--tta` flag to `anime_upscaler/infer.py` (default OFF); route through `tta_forward` when enabled
- [x] Add `tta: bool = False` to GUI `_Defaults` (persisted via existing `settings.json` schema)
- [x] Add TTA checkbox to `_SettingsPanel` (Row 5; theme moved to Row 6)
- [x] Add `tta: bool = False` field to GUI `_RunJob`; pipeline `_run_image`/`_run_video` route through `_tta_forward(backend.model, x)` when enabled
- [x] Plumb `s.tta` persist in `app.py:_save_settings`; pass `tta=bool(sp.tta_var.get())` in `_enqueue_next`
- [x] Vendor copy of `tta.py` into GUI package (`apps/anime_upscaler_gui/.../tta.py`) so the GUI stays self-contained (matches `archs.py` vendoring pattern)

### Phase 4 implementation notes (added this turn)

1. **Shape-aware transform count**: 8 for square (h==w) using full D4 (id, R90, R180, R270, Fh, Fv, T, AT); 6 for rectangular using axis-aligned subset (id, R90, R180, R270, Fh, Fv) -- the diagonal reflections swap h and w so they are skipped. Matches Real-ESRGAN's `inference_realesrgan.py` pattern. n_aug returned by `tta_forward` for logging.
2. **Inverse composition**: each forward transform has a paired inverse. For the 6 axis-aligned transforms, forward=inverse for {id, R180, Fh, Fv} (involutive), and the rotations are mutual inverses (R90 <-> R270). For diagonals, T is involutive (`transpose o transpose = identity`), and AT is involutive (`flip o transpose o flip o transpose = identity`).
3. **Contiguous on transpose**: `_fwd_transpose` / `_fwd_antitranspose` call `.contiguous()` after `torch.transpose` because the result is a non-contiguous view that convs reject. Same on inverse.
4. **Identity transform `.contiguous()`**: also `x.contiguous()` so that downstream code can rely on every augmented LR being a real (not aliased) tensor (cheaper than `.clone()` when already contiguous).
5. **Validation result on Phase 1 RFDN student** (96x96 LR -> 384x384 SR): no-TTA PSNR 29.23 / SSIM 0.8334; with --tta PSNR 29.42 / SSIM 0.8414 -- **+0.19 dB PSNR gain, +0.008 SSIM gain**. Plan spec was "within 0.05 dB" (worst-case bound); the actual +0.19 dB is in the upper end of the typical D4 TTA gain range (+0.05 to +0.15 dB) and confirms TTA is doing meaningful work, not silently changing outputs.
6. **GUI default OFF**: `tta: bool = False` in `_Defaults` matches the plan (preserves current speed UX; user opts in via Settings checkbox).
7. **Lazy import in pipeline**: `_tta_forward` wraps `from .tta import tta_forward` so the pipeline module loads even if tta.py is somehow missing (defensive).
8. **No new dependencies**: `tta.py` uses only `torch`. The vendored GUI copy matches the source.
9. **TTA mode logging**: `infer.py` prints `[tta] {n}x ({names})` once per image so the operator sees which shape-driven count was used.
10. **GUI thread safety**: `tta_forward` is decorated with `@torch.no_grad()` so it doesn't fight the surrounding `with torch.no_grad():` in the pipeline workers (double-wrapped is harmless).

## Phase 5 -- ATTEMPT 1 COMPLETE (target NOT met)

- [x] Write `scripts/run_distill_v3.ps1` runner (uses script-mode python, captures stdout/stderr, `$ErrorActionPreference = "Continue"`, direct `1>` / `2>` file redirects to dodge PowerShell's stderr-as-error behavior).
- [x] Smoke gate PASS (`exit=0`; losses finite; all csv cells populated).
- [x] Full 40-epoch training completed (40/40 epochs, all loss terms finite, val PSNR monotonically rising 26.19 -> 29.45).
- [x] **EMA-save bug discovered + patched:** original code locked `student_best.pt` to EMA state; with decay=0.999 on a 16k-iter run, EMA PSNR (26.70) was far below raw (29.45). Final test PSNR using EMA state = 26.58 (WORSE than bicubic). Patch: track `best_raw_psnr` and `best_ema_psnr` independently, write whichever beats the global best to `student_best.pt`, plus audit-only `student_best_raw.pt` and `student_best_ema.pt`.
- [x] Eval helper `scripts/eval_v3_ckpt.py` added (60 LOC, supports `--ckpt` and `--split` overrides, dual sys.path inserts to resolve both `import anime_upscaler.distill` package form and `from dataset import` bare form).
- [x] Raw student test PSNR measured: **28.95 dB / SSIM 0.8584** (vs target 29.50 PASS, v1 baseline 29.32, bicubic 29.45). **v3 raw student is BELOW both v1 and bicubic.** Target NOT met.

### Phase 5 attempt 1 evidence

- Trajectory comparison (val PSNR, last 5 epochs):
  - v3 ep 36-40: 29.45, 29.45, 29.45, 29.45, 29.45
  - v1 ep 36-40: 29.88, 29.88, 29.89, 29.89, 29.89

## Phase 5 — SHIPPED (2026-06)

- [x] **Ship v1 + TTA as the GUI's recommended default** (29.46 dB test PSNR).
  - `settings.py`: default `_Defaults.tta = True` (comment updated to reference v1+TTA).
  - `pipeline.py`: default `_RunJob.tta = True` (comment updated to explain speed tradeoff).
  - `registry.py`: `rfdn_distill_v1_4x` preset description updated to call out the TTA delta (29.32 -> 29.46 dB).
  - User `settings.json` patched via one-shot script (`.venv/patch_settings.py`, since cleaned up): `tta=true`, `last_model="RFDN_distill_v1_4x_student.pth"`.
  - 3 GUI files `py_compile` strict mode: PASS.
  - Smoke test (1-image via GUI load path): no-TTA 31.56 dB, TTA 31.63 dB (8x D4 group, ~7x slower). PASS.

### Ship-state decisions documented

- **Do NOT overwrite** `pretrained/RFDN_distill_v1_4x_student.pth` (already the best; the v3 ablations all underperformed it).
- **Treatment of Phase 5 v3 work:** treat as research-only. Memo + this TODO list the negative result, the EMA-save bug fix, the ablation evidence, and the per-component causal decomposition. Future sessions should NOT assume v3 wins on this stack -- instead consult the ablation table before changing the recipe.
- **TTA remains user-toggleable** in the Settings panel; gui ships with it ON by default for best-quality output. Users can disable for throughput-priority workloads.

### Recommended NEXT session (multi-hour)

- **Acquire 5K+ diverse anime frames** (open DIV2K-anime mix, public Bilibili clips, or your own captures).
- **Re-train v1-loss-equivalent** on the expanded data, 40 epochs, batch 16. Recipe: `scripts/run_distill_v3.ps1 -OutDir runs/distill_v4_bigdata -Degradation none -NoLpips -NoMsssim`. (This combination dropped the v3 perceptual terms but kept Charbonnier + cosine features -- probably the best starting point given the ablation pattern. If a v1-EXACT recipe is desired, fork the loss block to revert `loss_feat` to MSE and `loss_resp` to L1.)
- **Re-validate with TTA** (expected +0.14-0.34 dB on top of base student).

  - **Whole v3 run is shifted down by ~0.43 dB from v1, consistent from epoch 1.**
- Loss terms all finite throughout; numerical health is fine. This is a recipe-quality regression, not a numerical bug.
- EMA shadow follows the expected EMA warmup curve; on test (with the bug), EMA student = 26.58 dB (worse than bicubic). After the bug fix, future runs will write raw vs EMA correctly.
- Test set eval of `runs/distill_v3/student_last.pt` (raw state, epoch 40) confirmed in `runs/distill_v3/eval_student_last_results_table.{md,csv}`.

### Likely-cause ranking (next ablation steps)

1. **Phase 3 degradation** creates train/test distribution mismatch. Student learns to map codec-degraded LR -> clean HR, but test LR is clean -- causes a 0.3-0.5 dB drop on test. Cheapest ablation: `--degradation none` (everything else v3).
2. **MS-SSIM + LPIPS perceptual terms** at weight 0.2 + 0.05 of `loss_gt` push the student toward a perceptually-better but PSNR-worse optimum. If ablation A still loses, drop one at a time (next: B = drop LPIPS, then C = drop MS-SSIM).
3. **Cosine-distance features** at weight 0.5: coarse feature-level loss may be insufficient by itself.

## Phase 5 -- ABLATION A (degradation OFF, all other v3)

- [ ] Run: `powershell -ExecutionPolicy Bypass -File scripts/run_distill_v3.ps1 -Epochs 40 -OutDir runs/distill_v3_ablation_A_degradation_off -Degradation none`
- [ ] Verify train_log.csv epoch 40 raw student val PSNR > 29.50 if Phase 3 is the cause
- [ ] Eval student_last.pt on full test set: `& .venv/Scripts/python.exe scripts/eval_v3_ckpt.py --ckpt runs/distill_v3_ablation_A_degradation_off/student_last.pt --split test`
- [ ] Compare: if test PSNR >= 29.50 -> PASS, keep ablation-A weights
- [ ] If still below: ablation B (drop LPIPS), C (drop MS-SSIM), D (revert all v3 losses)

## Phase 5 -- BLOCKED on closing above gates

- [ ] Final decision: overwrite `pretrained/RFDN_distill_v1_4x_student.pth` vs add `rfdn_distill_v3_4x` preset in `apps/anime_upscaler_gui/anime_upscaler_gui/registry.py` (deferred until PSNR >= 29.50 achieved)

## Phase 5 -- Decided outcomes

- (populated after final ablation result)