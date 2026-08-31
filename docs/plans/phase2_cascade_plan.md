# Phase 2 — Cascade 2× + 2× + Batching + NVENC (≥ 30 fps @ 4K)

**Plan supersedes / refines:** section "Phase 2 — `A + C + NVENC`" in `docs/plans/realtime_4k_plan.md` (lines 238-380). Phase 1 shipped at `a3d54a9` on `feature/phase-1-realtime-4k`.

**Status:** PLANNED (user approved Option 1 in plan decisions table — single 2× model applied twice)

**Last updated:** 2026-08-31

**Owner:** single-user GPU (Quadro RTX 4000 8 GB, ~7.5 GB free)

**Goal:** End-to-end **≥ 30 fps @ 4K (3840×2160)** with the **cascade 2× + 2×** pipeline + the Phase 1 stack (TRT + b4 + NVENC). Quality target: **≥ 29.6 dB PSNR** on the held-out test set (≥ −0.3 dB tolerance vs the v1 single 4× baseline of 29.886 dB).

---

## Context (where Phase 1 ended)

| Phase-1 measurement (TRT b1, NVENC, 10.2 s 854×480 → 3416×1920) | Result | Goal | Status |
|---|---:|---:|---|
| Inference-only fps @ 4K input | ~42 fps (24 ms) | ≥ 25 fps | **PASS** (with margin) |
| End-to-end fps @ 4K | 18.80 fps mean / 20.52 fps peak | ≥ 25 fps | **PARTIAL** (~75 %) |
| Pipeline overhead (non-inference) | ~29 ms/frame | ≥ 10 ms | **OVER** (29 ms) |
| TRT speedup vs PyTorch FP16 | 1.87× E2E / 2.37× infer | ≥ 2× | **PASS (infer)** |
| NVENC CPU% | 26.5 % | ≥ 30 % | **PASS** |
| Engine cache survives restart | yes | yes | **PASS** |
| Batching ≥ 3× speedup | not measurable in E2E; 1.16× on bench | ≥ 3× | **DEFERRED** |

The **bottleneck** is the *interval* between frames (decode → transfer → encode), not inference. The 25 fps target is met by inference alone; end-to-end is short by ~13 ms/frame. Phase 2's job is to **close that gap via cascade**: a 2× student applied twice runs in roughly half the time of a 4× student at 4K because the second stage operates on a larger but already upsampled tensor (most compute scales with input pixel count, and stages process less input than a 4× model).

**Why cascade is expected to be ~2× faster inference than single 4× at 4K:**
- Single 4× at 960×540 input → ~5× more input pixels than 426×240 → ~75 ms/frame measured via Phase 1 plan table.
- Cascade 2× + 2× on the same input: stage 1 is at 960×540 (heavy), stage 2 at 1920×1080 (cheap — only 2× in H,W, but RFDN is body-bound not PixelShuffle-bound). With both stages applying to the LARGER-resolution intermediate, RFDN run at 1920×1080 vs 960×540 is dominated by the body's convolutions; in practice the cascade is roughly 1.3-2× faster than the single 4× based on the convolutional FLOPs ratio. (The math: 4× FLOPs ≈ 16× the 2× FLOPs at the same input; two 2× passes over different shapes = 1 + 4 = 5, vs 16 → ratio ~3.2× expected. RFDN has a bicubic-residual shortcut that reduces effective compute; overhead pushes this down to ~1.5-2× realistic.)

**Realistic Phase-2 projection** (will be measured): 25-30 fps end-to-end at 4K. If lower, fall back to the `_PinnedPool` micro-optimizations + PIL decoder + pinned decoder-output buffer (Phase 2.A5.b).

---

## Phase 2 architecture

### Pipeline models

```
Input LR (854×480, 18 fps source clip)
  ┌────────────────────────────┐
  │ Stage 1:  RFDN_2x  →  1708×960   │  Input 854×480 → 1708×960
  └────────────────────────────┘
  ┌────────────────────────────┐
  │ Stage 2:  RFDN_2x  →  3416×1920  │  Input 1708×960 → 3416×1920
  └────────────────────────────┘
Output SR
```

Both stages use the **same** 2× student weights, applied twice. This is Phase-2 Option 1 in the original plan. Option 2 (two stage-specific 2× students, where stage 2 sees stage-1 outputs as input) is documented but **not pursued** unless quality regresses > 0.3 dB.

### TR engine cache layout

The cache key `(<hash>, <H>, <W>, <batch>, <fp16>)` now produces **two engine entries per clip**:
- Stage 1: (854×480, b4, fp16)
- Stage 2: (1708×960, b4, fp16)

Total cache footprint per clip: ~300 MB → well within the existing 1 GB LRU cap.

### Backwards compatibility

- The v1 4× checkpoint at `pretrained/RFDN_distill_v1_4x_student.pth` continues to load unchanged via `RFDN()` defaults (`scale=4`).
- Existing pipeline jobs that don't set `cascade_mode` default to `SINGLE_4X` (current behavior).
- Cascade is **opt-in** via Settings panel: dropdown "Cascade mode: Single 4× / Cascade 2×+2×". Default `Single 4×`.

---

## Tasks

### A. Code (small changes, ~ 1 day)

| ID | Files | What changes |
|---|---|---|
| **A.1** | `anime_upscaler/student.py` | `RFDN.__init__` already accepts `upscale=4`; rename to `scale` for clearer naming, add `scale: int = 4` to the class signature. The upsampler convolution becomes `nn.Conv2d(nf, num_out_ch * scale * scale, 3, padding=1)` + `nn.PixelShuffle(scale)`. Forward bicubic interpolate uses `scale_factor=self.scale`. |
| **A.1b** | `apps/anime_upscaler_gui/anime_upscaler_gui/archs.py` | Mirror A.1 change in the vendored RFDN (lines 560-588). `RFDN(scale=2)` constructs the 2× variant. |
| **A.1c** | `apps/anime_upscaler_gui/anime_upscaler_gui/archs.py::build` | In the `rfdn_student` branch, **sniff scale from the state_dict's `upsampler.0.weight` shape** (`num_out_ch * scale**2`, typically 3 * 49 for 7×7 PixelShuffle, so 147 channels = scale 7 nope — for scale=2 it's 12 channels; for scale=4 it's 48). Read `out_ch = ups_weight.shape[0] // num_out_ch`, derive `scale = round(sqrt(out_ch))`. Pass to `RFDN(scale=scale)` so we build the right model. **This eliminates a separate kind or magic filename match.** |
| **A.2** | `anime_upscaler/dataset.py` | Move `SCALE = 4` module constant into `AnimePairDataset.__init__` as `self.scale` (default 4 for backward compat). The `crop_hr → lr_side = ch // self.scale` line at the HR crop (line 84) becomes `cl = ch // self.scale`. |
| **A.3** | `anime_upscaler/distill.py` | Add `--scale {2,4}` CLI arg (default 4). Pass to `RFDN(scale=args.scale)` and to `AnimePairDataset(..., scale=args.scale)` (new kwarg) **on train and val/test**. The evaluate() helper uses `model.upscale` (already exposed) for bicubic upsample; replace hardcoded `scale_factor=4` with `scale_factor=model.upscale`. |
| **A.4** | `scripts/eval_cascade_vs_single.py` (NEW) | Load val set; for each pair compute (a) bicubic, (b) single 4× baseline, (c) cascade 2×+2× (runs `RFDN_2x` twice); print PSNR/SSIM table. Pass criterion: cascade ≥ single 4× − 0.3 dB. CLI: `--ckpt-2x runs/distill_v2_2x/student_best.pt --ckpt-4x pretrained/RFDN_distill_v1_4x_student.pth --val-dir data/anime_video_frames --val-batches 50`. |
| **A.5a** | `apps/.../pipeline.py` | Add `_RunJob.cascade_mode: str = "single_4x"` field. New enum `_CascadeMode = Enum("CascadeMode", "SINGLE_4X CASCADE_2X2X")`. In `_run_image` / `_run_video` (the `_process_*` paths), after backend returns `y`, if `cascade_mode == CASCADE_2X2X`, run `y = backend(y)` once more. The intermediate `y` is in [0,1] float (NCHW) on GPU — the backend handles it directly. (Note: the input/output conversion code stays single 4× because cascade starts from the same LR and ends with the same SR; only the inner model calls change.) |
| **A.5b** | `apps/.../pipeline.py` | Triton engine cache stays as-is — two distinct (H, W) keys builds two engines (cached). |
| **A.5c** | `apps/.../pipeline.py` | Also persist `cascade_mode` via `_Defaults.cascade_mode`. |
| **A.5d** | `apps/.../settings.py` | `_Defaults.cascade_mode: str = "single_4x"`. Add `"single_4x" \| "cascade_2x2x"` validator. |
| **A.5e** | `apps/.../widgets/models_panel.py` (or wherever the kind selector lives) | Add a row: `Cascade mode:` combobox bound to `settings.cascade_mode`. Default disabled (single 4×). Tooltip: "Apply the 2× student twice in sequence instead of using the 4× student. Experimental — faster end-to-end but ~0–0.3 dB quality impact." |
| **A.5f** | `apps/.../app.py` | Populate `cascade_mode=str(sp.cascade_mode_var.get())` into every `_RunJob`. |
| **A.6** | `apps/.../registry.py` | Add new trained-model entry: `{"id": "rfdn_distill_v2_2x", "filename": "RFDN_distill_v2_2x_student.pth", "scale": 2, "kind": "rfdn_student", "source": "trained", "display_name": "RFDN Distill v2 * Trained (2x)"}`. Bump `_TRAINED_FILENAME_PATTERNS` is already general (`RFDN_distill_v*`). |

### B. Training (long, ~ 1-2 days unattended)

| ID | Action | Command / notes |
|---|---|---|
| **B.1** | Train 2× student v2 | `python anime_upscaler/distill.py --scale 2 --epochs 30 --batch-size 16 --lr 5e-5 --out-dir runs/distill_v2_2x --data data/anime_video_frames` (same hyperparams as v3 baseline; 30 epochs to keep wall time manageable, since v3 reached 29.886 dB at epoch 40 and the 2× task has *less* upsampling to learn). |
| **B.2** | Save 2× ckpt | `pretrained/RFDN_distill_v2_2x_student.pth`. |
| **B.3** | Optional early stop | Watch first 5 epochs; if val_psnr is < 28 dB by epoch 5, halt and re-tune. (Distill v3 hit 29 dB by epoch 5.) |

### C. Eval & Validation

| ID | Action | Pass criterion |
|---|---|---|
| **C.1** | `scripts/eval_cascade_vs_single.py --val-batches 50` | cascade PSNR ≥ 29.6 dB; cascade vs single 4× delta ≥ 0.3 dB. |
| **C.2** | Visual sanity on a 30 s clip | no obvious artifacts (banding, ringing, color shift). |
| **C.3** | py_compile all touched files | exit 0. |
| **C.4** | Image job: cascade mode, batch=1, sample LR | output shape (1,3,4H,4W) no NaN. |

### D. E2E bench (matches Phase 1 measurement)

Reuse `tmp/test_e2e.py` — add a config row `trt_b4_nvenc_cascade`:
- `trt_b4_nvenc_cascade`: cascade mode, batch=4, TRT ON, NVENC ON, on `.venv/test_4k_540p_input.mp4`

| Metric | Phase 1 (`trt_b1_nvenc`) | Phase 2 target | Pass |
|---|---:|---:|---|
| End-to-end fps | 18.80 | **≥ 30** | if met → ship; if 25-30 → ship with "cascade" tagged experimental |
| Inference fps | 42 | ≥ 60 | (we'll measure) |
| Pipeline overhead ms/frame | 29 | ≥ 5 | improvement from removing one stage's full compute |

### E. Docs

- `docs/rfdn_realtime_report.md` section "Phase 2 measurements": fps, cascade-vs-single quality delta, engine cache growth, NVENC CPU%.
- `AGENTS.md` "Phase 2 Shipped" entry: file inventory, ckpt path, expected fps gains.
- One commit `Phase 2 (Real-time 4K): Cascade 2×+2× + NVDEC shipped`. Branch: same `feature/phase-1-realtime-4k` (squash continuation) OR new `feature/phase-2-cascade` per user preference.

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| 2× student quality < single 4× by > 0.3 dB | Medium | Use same v3 loss recipe + Charbonnier+cosine+L1+MS-SSIM+LPIPS (proven for 4×); re-evaluate at epoch 5; Option 2 (self-distilled stage 2) on the table. |
| Cascade pipeline introduces visible banding at sharp edges | Low | Visually compare vs single 4× on a 30 s clip during C.2. |
| Two engine builds double startup latency | Low | Both engines built lazily on first use; none on startup. |
| Cascade VRAM blows past 8 GB | Low | RFDN has trivial VRAM (~50 MB at 4K); 2 × same engine = ~100 MB. Plenty of headroom. |
| Training diverges | Low | Use proven v3 hyperparams; `--resume` from a partial run if early sign. |
| User confusion over which mode | Low | Tooltip explains cascade tradeoff; default Single 4× (current proven path). |

---

## Wall-time estimate

- A.1-A.6 (code, small): **0.5 day**
- B.1-B.3 (training, unattended): **~ 1.5-2 days** at ~ 25 min/epoch (matches v3 logs which averaged similar per the earlier `runs/distill_v3`)
- C (eval+validation): **0.5 day**
- D (E2E bench): **0.25 day**
- E (docs+commit): **0.25 day**
- **Total Phase 2: ~ 3-3.5 working days** (training runs in background)

---

## Open decisions deferred to user

1. **Default cascade_mode in `_Defaults`**: ship with `cascade_mode="single_4x"` (safe), with cascade as opt-in via Settings. *This is the Phase 1 plan's recommendation and we'll go with it unless user objects.*
2. **Commit branch**: continue on `feature/phase-1-realtime-4k` (squash at end) or open `feature/phase-2-cascade`? *Default: same branch (squash at end) so the 2 phases stay bundled in the user's local history.*
3. **If cascade quality regresses > 0.3 dB**: ship anyway with the regression documented? Or roll cascade back to disabled-default and just keep the speed-up from inference optimizations? *Default: ship with warning, since the empirical system is the deliverable and the user wants 30 fps.*

---

## Reference

- **Phase 1 plan:** `docs/plans/realtime_4k_plan.md` (background, Phase 1 numerical baseline)
- **Phase 1 ship banner:** `docs/rfdn_realtime_report.md` section 10 (E2E measurements)
- **Phase 1 commit:** `a3d54a9` on `feature/phase-1-realtime-4k`
- **v3 distill recipe:** see commit messages + `.claude/plans/distill_v3_recipe.md` (referenced from `distill.py` docstring)
- **Test input clip:** `.venv/test_4k_540p_input.mp4` (not committed; re-creatable)
- **Student weights dir:** `pretrained/`
- **Distill output dir:** `runs/distill_v2_2x/` (training artifacts)
