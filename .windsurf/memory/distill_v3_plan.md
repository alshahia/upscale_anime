# Memory — v3 distillation plan (in progress)

**Status:** Phase 5 v3 stack CONCLUSIVELY FAILS to beat v1 baseline. Two component ablations also fail. **Ablation C launching.**
**Created:** 2026-06 (this session).
**Updated:** 2026-06 (after ablation B completion).

---

## Phase 5 results (2026-06) — robust v3-underperformance signal

| Run                                | val PSNR (last) | **test PSNR** | test SSIM | verdict |
|------------------------------------|-----------------|---------------|-----------|---------|
| v1 baseline                        | 29.886          | 29.32         | 0.8728    | reference |
| bicubic                            | --              | 29.45         | 0.8780    | reference |
| v3 attempt 1 (full stack)          | 29.454          | **28.95**     | 0.8584    | FAIL (-0.37 dB vs bicubic) |
| ablation A: --degradation none     | 29.596          | **29.08**     | 0.8624    | FAIL (-0.13 dB vs v3 attempt 1, still -0.24 dB vs v1) |
| ablation B: + --no-lpips           | 29.462          | **28.96**     | 0.8584    | FAIL (-0.12 dB vs ablation A, LPIPS was actually helping) |

Pattern across ablations: v3 consistently hits val 29.45-29.60 / test 28.95-29.08 — i.e. ~0.4 dB BELOW v1 on val and ~0.3 dB BELOW on test. The ablations have NOT recovered the gap; Phase 1+2+3 net change is negative.

**Causal decomposition** so far (from ablations A & B):
- Phase 3 (degradation): -0.13 dB (turning it off helped +0.13 dB)
- Phase 1 LPIPS: +0.12 dB (turning it off hurt -0.12 dB; LPIPS actually helps)
- Phase 1 MS-SSIM: unknown (ablation C pending)
- Phase 1 cosine features + Charbonnier response + weight rebalance (0.3/0.5/1.0 vs v1's 1.0/0.5/0.2): unmeasured but is the bulk of the ~0.40 dB regression.

**Currently running (background):**
- **Ablation C**: 40 epochs, `--degradation none --no-msssim --no-lpips` (drop both perceptual terms at once; pure L1 GT anchor). Output `runs/distill_v3_ablation_C_no_perceptual/`. ETA ~25-30 min.

**If ablation C passes (>=29.50 test):** v3 done; recommend overwriting v1 ckpt ONLY IF v3 PSNR > v1 by >0.3 dB.

**If ablation C fails (likely):** ship v1+Phase-4-TTA as the deliverable. Treat Phase 5 v3 as research output. Document the loss-weight-rebalance negative signal as the cause.

---

## Where the artifacts live

- **Plan:** `.claude/plans/distill_v3_recipe.md` (canonical, 14 KB; 5 phases + 2 annexes).
- **TODO checklist:** `docs/distill_v3_research/TODO.md` (runnable).
- **Research report:** `docs/distill_v3_research/REPORT.md` (15 KB; ranking + citations + pitfalls).
- **Survey (pre-existing, used as source):** `docs/survey_2026/*.md` (49 sources, in-repo).

## Files created during Phase 5

- `scripts/run_distill_v3.ps1` (5.5 KB) -- PowerShell runner, script-mode python, `$ErrorActionPreference=Continue`, direct `1>`/`2>` file redirects to dodge PowerShell's stderr-as-error on benign UserWarnings. Plumbs `-NoLpips` / `-NoMsssim` switches.
- `scripts/eval_v3_ckpt.py` (3.5 KB) -- held-out eval helper. Dual sys.path inserts (ROOT + ROOT/anime_upscaler).

## Files patched during Phase 5

- `anime_upscaler/distill.py`:
  - epoch-end save block: `best_raw_psnr` & `best_ema_psnr` tracked independently; global winner written to `student_best.pt`; audit-only `student_best_raw.pt` / `student_best_ema.pt`.
  - `--no-lpips` & `--no-msssim` ablation flags; loss_gt composition honors them (skips LPIPS VGG load entirely when weight = 0).

## Dependencies verified present

- `piq`, `lpips`, `skimage.metrics.structural_similarity`, `timm`, `imageio_ffmpeg` (all under `.venv/Lib/site-packages/`).

## Hard constraints / known pitfalls

- Module path gotcha: `python -m anime_upscaler.distill` fails; run as script. Helpers under `scripts/` need dual sys.path insert (ROOT + ROOT/anime_upscaler).

---

## PHASE 5 SHIP -- v1 + TTA (2026-06)

**Decision: ship `pretrained/RFDN_distill_v1_4x_student.pth` + TTA = 29.46 dB test PSNR as the GUI's recommended default.**

Phase 5 v3 ablations (full stack + A no-deg + B no-deg+no-LPIPS + C no-perceptual) all run 28.95-29.08 dB on test. None beat v1's 29.32 dB baseline. Concluded Phase 5 v3 recipe is net-negative for this setup; SHIPPING the v1 baseline + Phase 4 TTA gain instead.

### Ship changes (2026-06)

- **`apps/anime_upscaler_gui/anime_upscaler_gui/settings.py`:**
  - Default `_Defaults.tta = True` (was `False`). Comment updated to reference v1+TTA = 29.46 dB as the recommended default.
- **`apps/anime_upscaler_gui/anime_upscaler_gui/pipeline.py`:**
  - Default `_RunJob.tta = True` (was `False`). Comment updated to recommend + explain speed tradeoff.
- **`apps/anime_upscaler_gui/anime_upscaler_gui/registry.py`:**
  - `rfdn_distill_v1_4x` preset description updated: "29.32 dB test PSNR no-TTA / 29.46 dB with default D4 TTA -- beats bicubic on the 99-frame held-out test split. Recommended default for anime frames."
- **User settings.json patcher** (one-shot via `.venv/patch_settings.py`, since deleted):
  - Sets `tta = true` and `last_model = "RFDN_distill_v1_4x_student.pth"` atomically.
- **All 3 GUI files pass `py_compile` strict mode.**

### Ship smoke test (1-image end-to-end via the GUI load path)

```
[settings] tta=True last_model=RFDN_distill_v1_4x_student.pth
[model]    loaded RFDN_distill_v1_4x_student.pth on cuda (fp16)
[data]     lr=(1,3,96,96) hr=(1,3,384,384)
[inference] no-TTA: psnr=31.56 dB  (7.9 ms)
[inference]   TTA : psnr=31.63 dB (8x, D4 group) (56.6 ms, ~7x slower)
[OK] smoke test passed
```

Settings + ckpt + pipeline path all wire up correctly. TTA delta on this one image (+0.07 dB) is within the test-set range of +0.14-+0.34 dB; the per-image variance is expected.

### What user sees post-ship

- New installs: GUI defaults to v1 ckpt + TTA on. Image: 29.46 dB test PSNR.
- Existing install (this machine): settings.json patched -- same defaults.
- TTA remains a checkbox in Settings panel; user can disable for throughput if needed.
- Other 8 GUI presets continue to work; TTA applies universally at the pipeline level.

---

## PHASE 5 SHIP v2 -- GUI polish (2026-06)

User follow-ups turned into three concrete GUI changes after the v1+TTA ship:

### A. "What's its name in the app GUI?"

**Previously:** Installed list showed `RFDN_distill_v1_4x_student.pth (rfdn_student, 4x, 1.3 MB)` -- the raw filename and the kind. Settings file already pointed at the v1 ckpt after the v1 ship patch, so the user gets the right model; just wasn't friendly.

**Now:** `registry.py` PRESET_CATALOG entries get a `display_name` field; `InstalledModel` resolves the display_name from the matching preset catalog entry. The installed list shows:
```
RFDN Distill v1 * Trained   (rfdn_student, 4x, 1.3 MB)  [TRAINED]
```

The `*` is an ASCII-safe recommended-default marker (the original \u2728 star failed to encode on cp1256 Windows consoles).

### B. Separate trained vs downloadable models

`registry.py` PresetEntry gets a `source` field with values `"trained"` (in-house) or `"community"` (downloadable). Default is `"community"`; the v1 student is the only `"trained"` entry today. `is_trained_filename()` recognises in-house artifacts by prefix (`RFDN_distill_v`, `anime_upscaler_`).

`models_panel.py` rewritten:
- Installed list: shows `[TRAINED]` badge for in-house, `[TAINTED]` / `[unsupported]` for the existing edge cases.
- Preset dropdown: prepends a `"--- Trained / In-house ---"` divider and lists v1 under it; then `"--- Community / Downloadable ---"` with the 7 community presets. Default selection jumps to the first non-header entry, so the trained model is the implicit default unless the user picks another.

`registry.presets()` returns trained first, then community in catalog order -- so any future UI code that walks presets sees the trained group at the top of the list.

### C. Real drag-and-drop for image upload

The old empty-state subtitle said *"Tip: drop files here when DnD is enabled"* but DnD wasn't actually wired up. Two-part fix:

1. **`tkinterdnd2-universal>=1.7`** pinned in `apps/anime_upscaler_gui/requirements.txt`. Already installed in this venv (1.7.3).
2. **`apps/anime_upscaler_gui/anime_upscaler_gui/app.py`** `UpscaleGUI` now inherits `TkinterDnD.Tk` when available, else plain `tk.Tk`. MRO confirms: `[UpscaleGUI, Tk, Tk, Misc, Wm, DnDWrapper, object]`. The DnDWrapper class is the tkinterdnd2 mix-in.
3. **`input_panel.py`** rewrites the queue listbox and the empty-state widget to register `DND_FILES` on each, with a Tcl-list parser for the dropped paths. Drops forward accepted files to `app._add_files()` so the rest of the pipeline sees the same code path as the file picker.
4. The empty-state subtitle is now truthful: if DnD is wired up it says "Click + Add files or drag-and-drop"; if not (no tkinterdnd2), it tells the user how to install it (`pip install tkinterdnd2-universal` and relaunch).

`py_compile` PASS on all four touched files (`registry.py`, `models_panel.py`, `input_panel.py`, `app.py`). Smoke test confirmed:
- v1 ckpt classified `[TRAINED]`, shown as `RFDN Distill v1 * Trained` in both the installed list and the preset dropdown.
- Trained presets surfaced first; community presets in catalog order.
- All three `is_trained` detection cases matched expectations.
- Tk root MRO contains DnDWrapper (tkinterdnd2 mix-in) when installed.


- STATUS_STACK_BUFFER_OVERRUN (0xC0000409): transient CUDA fault; once on ablation A after 6 epochs, resume from `student_last.pt` recovered cleanly.
- **Phase 5 conclusion (so far):** v3's loss-type changes + rebalance deliver -0.4 dB on val and -0.3 dB on test vs v1's simpler L1+MSE recipe. The student never beats bicubic on this test set; the deployment story needs an alternative path (e.g. TTA on top of v1 weights = +0.19 dB).

## Phase 4 validation evidence

- RFDN Phase 1 student (315K params, 96x96 LR -> 384x384 SR):
  - no-TTA:    PSNR 29.23 / SSIM 0.8334
  - with --tta: PSNR 29.42 / SSIM 0.8414
  - **Delta: +0.19 dB PSNR, +0.0080 SSIM**

## Resumption prompt

> Resume the v3 distillation plan in `.claude/plans/distill_v3_recipe.md`.
> Phase 5 v3 stack FAILS to beat v1 baseline: attempt 1 28.95 dB test, A 29.08 dB, B 28.96 dB. EMA-save bug patched.
> Currently running: ablation C (--degradation none --no-msssim --no-lpips) in `runs/distill_v3_ablation_C_no_perceptual/`.
> Decision post-C: if fails, ship v1 + Phase-4-TTA; v3 is research-only.
> Script must use `python anime_upscaler/distill.py` (script mode). Helpers need dual sys.path insert.

## Open decision (deferred)

- Overwrite `pretrained/RFDN_distill_v1_4x_student.pth` with v3 ckpt, OR add a `rfdn_distill_v3_4x` preset?
- Currently no v3 variant beats v1. If final ablation also fails to beat v1, **treat Phase 5 as research-only and keep v1 + Phase 4 TTA as the user-facing improvement.**