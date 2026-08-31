# Real-time 4K (25 fps) Plan — TensorRT + Cascade + NVENC

**Status:** PLANNED (user approved both phases; Phase 1 = TensorRT + Batching + NVENC; Phase 2 = Cascade 2×+2× + Batching + NVENC)
**Last updated:** 2026-08-31
**Owner:** user GPU (Quadro RTX 4000 8 GB, single GPU, ~7.5 GB free)
**Goal:** End-to-end **≥ 25 fps @ 4K (3840×2160)** in the GUI, with the existing RFDN-distilled student, no quality regression (PSNR ≥ 29.89 dB − 0.05 dB FP16 tolerance).

---

## Context

The current student (`pretrained/RFDN_distill_v1_4x_student.pth`) is a 315,844-param RFDN distilled at width 52, 6 RFDB blocks, 4× native PixelShuffle, val PSNR **29.886 dB** at epoch 40.

**Measured current speed** (GUI worker thread + async reader + ffmpeg pipe encoder, 1-sec 854×480 clip, RTX 4000, FP16):

| Config | avg fps | per-frame infer |
|---|---:|---:|
| TTA ON, medium, full input | 2.38 | 372 ms |
| noTTA, medium, full input | 10.37 | 59 ms |
| noTTA, ultrafast, downscale_max_edge=480 | 18.28 | 19 ms |
| noTTA, ultrafast, downscale_max_edge=426 | 20.70 | 15 ms |

To hit 25 fps @ 4K output (3840×2160), the model takes 960×540 input → ~5× more input pixels than the 426×240 above → inference alone becomes **~75 ms/frame** → ~13 fps. End-to-end without encoder or I/O improvements stays below target.

**User's initial idea** (expand student, freeze most weights, add last layers) was analyzed and **rejected for speed**: adding parameters always adds latency. The student has plenty of capacity; the bottleneck is throughput.

**Approved approach** (two phases, both inclusive of NVENC encoder switch and batching):

1. **Phase 1 — TensorRT + Batching + NVENC**: ship in ~3 days, current 4× student, no retraining. Target: 35–50 fps @ 4K.
2. **Phase 2 — Cascade 2× + 2× + Batching + NVENC**: ship in ~4 days after Phase 1, train a new 2× student. Target: 30+ fps @ 4K with same quality.

---

## Phase 1 — `B + C + NVENC`: TensorRT + Batching + NVENC

### P1. Goal

End-to-end **≥ 25 fps @ 4K (3840×2160)** on RTX 4000 with the existing `RFDN_distill_v1_4x_student.pth`, no quality loss (PSNR ≥ current 29.89 dB within FP16 tolerance, typically 0.05 dB drift).

Expected landing zone: **35–50 fps @ 4K** in the GUI.

### P1. Sub-tasks

#### B1. Install TensorRT (or fallback to ONNX + trtexec)

- Try `pip install torch_tensorrt` (matches existing torch CUDA build).
- Fallback: install TensorRT wheel from NVIDIA, get `trtexec` on PATH.
- Fallback: pure ONNX + Python `tensorrt` runtime.
- Detect at app startup; abort gracefully if not present (keep PyTorch eager path as fallback).

**Files touched**: env only (`requirements.txt` or pip install).

#### B2. Engine builder module

New file: `apps/anime_upscaler_gui/anime_upscaler_gui/trt_engine.py`

Responsibilities:
- `_TrtEngineCache` class — module-level singleton.
- `get_engine(model, input_hw, batch_size, fp16=True) -> Callable` — returns `(x: Tensor) -> y: Tensor`.
- Per-(H, W, batch_size, fp16) cache entry.
- Engine build is lazy on first call to that shape.
- Falls back to `model.forward` if TensorRT is not importable OR engine build fails.
- Disk-persists engines under `%APPDATA%/anime_upscaler_gui/cache/trt/<hash>.engine` so restarts don't rebuild.

**Acceptance**:
- First call to a new shape returns within 60 s with a `building` log line.
- Subsequent calls <1 ms lookup, identical numerics (within FP16 epsilon).

#### B3. Pipeline integration

Edit `apps/anime_upscaler_gui/anime_upscaler_gui/pipeline.py`:

In `_run_image` and `_run_video`, replace
```python
with torch.inference_mode():
    y = model(x)
```
with
```python
y = self._trt_run(x)  # wrapper: picks TRT engine or falls back to model.forward
```

Where `_trt_run` is set per-instance from the engine cache (lazy init when TRT is available, else `lambda x: model(x)`).

The pinned-buffer flow (`pinned_in` / `pinned_out`) already in place stays untouched.

**Acceptance**:
- `py_compile pipeline.py` PASS.
- `py_compile trt_engine.py` PASS.
- Image job: output PSNR within 0.05 dB of eager (FP16 is the same numerics).
- Video job: no behavior change when TRT is disabled.

#### C1. Re-enable batching for video (currently forced to 1)

Edit `apps/anime_upscaler_gui/anime_upscaler_gui/app.py`:

In `_enqueue_next`, the current code forces `batch_size=1` for video (a workaround from earlier checkpoints). The proper fix is to make `_run_video` handle list-shape I/O — see C2.

#### C2. Pipeline batched video path

Edit `apps/anime_upscaler_gui/anime_upscaler_gui/pipeline.py`:

In `_run_video`, replace the `for frame in sync_reader` loop with a buffer-accumulator:

```python
buf: list[np.ndarray] = []
for frame in sync_reader:
    buf.append(frame)
    if len(buf) >= batch_size:
        yield from self._flush_batch(buf, ...)
        buf.clear()
if buf:
    yield from self._flush_batch(buf, ...)
```

`_flush_batch`:
- Allocates `pinned_in` of shape `(N, 3, H, W)` from pool keyed by `(H, W, N)`.
- Stacks input: `np.stack(buf, axis=0)` → uint8 → fills pinned.
- `_to_tensor(pinned_in=pinned_in, batch=True)`.
- Inference on `(N, 3, H, W)` tensor.
- Output pinned: `(N, H, W, 3)` uint8 → splits → writes N frames.

**Acceptance**:
- Batch=1: identical to current behavior.
- Batch=4: 4 frames written per inference call.
- Throughput scales: at 426×240 input, batch=4 should give ~3-4× fps vs batch=1.

#### N1. NVENC switch

Edit `apps/anime_upscaler_gui/anime_upscaler_gui/ffmpeg.py`:

Add to `OpenEncoderConfig`:
```python
use_nvenc: bool = False
nvenc_preset: str = "p1"   # p1=fastest, p4=balanced, p7=best
nvenc_qp: int = 18
```

In `open_encoder`, when `use_nvenc=True`, build args as:
```python
[
    "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps),
    "-i", "pipe:0",
    "-c:v", "h264_nvenc", "-preset", preset, "-rc", "constqp", "-qp", str(qp),
    "-pix_fmt", "yuv420p", "-r", str(fps),
    output_path,
]
```

When `use_nvenc=False`: keep current `libx264 -preset ultrafast`.

Add `_detect_nvenc_support() -> bool` that probes `ffmpeg -hide_banner -encoders` for `h264_nvenc` at app startup.

**Acceptance**:
- `h264_nvenc` works on RTX 4000: encode a 1-sec 4K clip, verify file size, decode back, sanity check.
- `libx264` fallback still works when NVENC absent (e.g., on CPU-only systems).

#### N2. GUI Settings panel

Edit `apps/anime_upscaler_gui/anime_upscaler_gui/widgets/settings_panel.py`:

Add to the `Video` group:
- Checkbox: `Use NVIDIA hardware encoder (NVENC)` (default: ON if detected, OFF otherwise).
- Optional dropdown: `NVENC preset` (p1, p2, p3, p4) (default p1).

Add to the `Performance` group:
- Checkbox: `Use TensorRT engine` (default: ON if available).
- Tooltip explaining first-run engine build may take 30-60 s per resolution.

Edit `apps/anime_upscaler_gui/anime_upscaler_gui/settings.py`:

Add keys:
```python
"video.use_nvenc": bool,
"video.nvenc_preset": str,
"perf.use_tensorrt": bool,
```

Add to defaults in `_defaults()`.

#### N3. Settings persistence & migration

Edit `apps/anime_upscaler_gui/anime_upscaler_gui/settings.py`:

- `_load_json` already has graceful defaults — just add the new keys.
- No version bump needed (additive).

### P1. Tests (acceptance gate)

| Test | Pass criteria |
|---|---|
| `py_compile` on all touched files | exit 0 |
| Import smoke: `UpscaleGUI` launches headless (`--no-splash`) | no traceback |
| Image job: small image, batch=1, TRT ON | PSNR drop < 0.05 dB vs eager |
| Image job: same, TRT OFF (fallback) | identical to before |
| Video job: `.venv/test_1sec.mp4`, batch=1, TRT ON | ≥ previous fps |
| Video job: same, batch=4, TRT ON | ≥ 3× the batch=1 fps |
| Video job: `.venv/test_4k_540p_input.mp4` (need to create), batch=4, TRT ON + NVENC | **≥ 25 fps** end-to-end |
| Encoder test: h264_nvenc 4K 25 fps, 5 sec clip | file created, `ffprobe` reports correct stream, CPU usage <30% |
| Engine cache: kill app mid-build, relaunch | second launch reuses partial cache; no corrupted engines |

### P1. Files modified / added

**New**:
- `apps/anime_upscaler_gui/anime_upscaler_gui/trt_engine.py`

**Modified**:
- `apps/anime_upscaler_gui/anime_upscaler_gui/pipeline.py` (B3, C2)
- `apps/anime_upscaler_gui/anime_upscaler_gui/ffmpeg.py` (N1)
- `apps/anime_upscaler_gui/anime_upscaler_gui/app.py` (C1)
- `apps/anime_upscaler_gui/anime_upscaler_gui/settings.py` (N3)
- `apps/anime_upscaler_gui/anime_upscaler_gui/widgets/settings_panel.py` (N2)

**Test artifacts** (kept, not committed):
- `.venv/test_4k_540p_input.mp4` — 10-sec 960×540 test input.
- `.venv/test_4k_out_trt_b4_nvenc.mp4` — Phase 1 end-to-end output.

### P1. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| `torch_tensorrt` install fails on Windows | Medium | Fallback to ONNX + Python `tensorrt` wheel; then to ONNX + `trtexec` CLI. |
| Engine build for dynamic shapes is slow | Low | Pre-bucket input sizes; build at startup for common resolutions. |
| NVENC not detected on system | Low | Auto-disable checkbox, default to `libx264`. |
| Batch > 1 in `_run_video` re-introduces the list-shape bug | Low | Add explicit shape assertion; unit test batch=4 on a 5-frame clip. |
| TRT FP16 numerics differ slightly from eager | Low | Tolerance 0.05 dB PSNR drop is acceptable; log warning if exceeded. |
| First-run engine build blocks the GUI | Medium | Run engine build in the worker thread; show "preparing engine for 960×540…" in status bar. |

### P1. Estimated wall time

- B (TRT install + module + integration + tests): **1.5–2 days**
- C (batched video path + tests): **0.5 day** (already partially done in earlier checkpoint)
- N (NVENC + settings + tests): **0.5 day**
- **Total Phase 1: ~3 working days**

---

## Phase 2 — `A + C + NVENC`: Cascade 2× + 2× + Batching + NVENC

### P2. Goal

Replace the single 4× student with a **cascade of two 2× stages** (one model applied twice, or two stage-specific models). Combined with batching + NVENC, push end-to-end **≥ 30 fps @ 4K** while **maintaining or slightly improving quality** (target: ≥ 29.7 dB PSNR, within 0.3 dB of current single 4× baseline).

### P2. Sub-tasks

#### A1. Add `scale=2` to RFDN

Edit `anime_upscaler/models/rfdn.py`:

Add `scale: int = 4` to `RFDN.__init__`. The upsampler becomes:
```python
n_out = 3 * (scale ** 2)
self.upsampler = nn.Sequential(
    nn.Conv2d(width, n_out, 3, 1, 1),
    nn.PixelShuffle(scale),
)
```

This lets one architecture produce 2× or 4× via different upsample factor. (Body, blocks, fuse, pa are scale-agnostic.)

Backwards compat: existing 4× checkpoints load unchanged (`scale` defaults to 4, weight shapes match).

**Files**: `anime_upscaler/models/rfdn.py`.

#### A2. Add `--scale` arg to distill.py

Edit `anime_upscaler/distill.py`:

Add `--scale {2,4}` arg (default 4 to preserve current behavior). Pass to student and teacher constructors.

**Files**: `anime_upscaler/distill.py`.

#### A3. Train a 2× student

**Decision point** — recommend Option 1 first:

**Option 1 — Single 2× model, applied twice:**
- Train `RFDN_2x_student.pth` (same architecture, ~310K params, scale=2).
- Same training data, same loss, same teacher.
- ~30 epochs, ~2 days on RTX 4000.
- Pipeline applies it twice: `lr → model(lr) → model(.)`.

**Option 2 — Two stage-specific 2× models:**
- Train `RFDN_2x_stage1.pth` (480p→960p specialist) and `RFDN_2x_stage2.pth` (1920p→3840p specialist).
- Stage 2 trained on stage-1 outputs (self-distillation chain) to recover cascade error.
- 2× training cost.
- ~4 days.

**Recommendation**: Option 1 first. Compare quality. Move to Option 2 only if cascade loses > 0.3 dB vs single 4×.

**Output**: `pretrained/RFDN_distill_v2_2x_student.pth`.

**Files**: `pretrained/RFDN_distill_v2_2x_student.pth` (new), training log under `runs/distill_v2_2x/`.

#### A4. Quality validation

Create `scripts/eval_cascade_vs_single.py`:
- Load validation set (paired LR/HR from `data/anime_video_frames/val` or wherever).
- Run single 4× model on each pair.
- Run cascade 2× + 2× on each pair.
- Compute PSNR, SSIM for both.
- Print table.
- **Pass criteria**: cascade ≥ single 4× − 0.3 dB (otherwise fall back to single 4× in pipeline).

**Files**: `scripts/eval_cascade_vs_single.py` (new).

#### A5. Cascade pipeline mode

Edit `apps/anime_upscaler_gui/anime_upscaler_gui/pipeline.py`:

Add a new mode `cascade_2x2x`:
```python
class CascadeMode(Enum):
    SINGLE_4X = auto()     # current behavior
    CASCADE_2X2X = auto()  # new
```

In `_run_image` / `_run_video`:
```python
if mode == CascadeMode.CASCADE_2X2X:
    y = model(x)               # 2× upscale
    y = model(y)               # 2× upscale again
else:
    y = model(x)               # 4× upscale
```

Apply TRT + pinned buffers in the same way — each stage reuses the same engine cache (input shape changes between stages, so TRT will build two engines per (batch, h, w) tuple).

**GUI toggle**: Add to Models tab: `Cascade mode` (Single 4× / Cascade 2× + 2×). Default: Single 4× (safe), set Cascade as opt-in.

**Files**:
- `apps/anime_upscaler_gui/anime_upscaler_gui/pipeline.py`
- `apps/anime_upscaler_gui/anime_upscaler_gui/app.py` (mode dropdown)
- `apps/anime_upscaler_gui/anime_upscaler_gui/settings.py` (persist mode)
- `apps/anime_upscaler_gui/anime_upscaler_gui/registry.py` (add new model entry, mark as `[CASCADE 2X+2X]`)

### P2. Tests (acceptance gate)

| Test | Pass criteria |
|---|---|
| 2× student loads, runs on sample 480p image | output shape 960×540, no NaN |
| Cascade eval: 50 paired samples | mean PSNR ≥ 29.6 dB |
| Cascade speed (TRT ON, batch=4, NVENC) at 4K | **≥ 30 fps** end-to-end |
| Cascade quality: visual on test_30s_TTA.mp4 | no obvious artifacts vs single 4× |
| Model registry: new 2× model appears with `[CASCADE]` tag | confirmed in dropdown |
| Backwards compat: existing 4× models still work | unchanged behavior |

### P2. Files modified / added

**New**:
- `pretrained/RFDN_distill_v2_2x_student.pth` (new weight)
- `scripts/eval_cascade_vs_single.py`
- `runs/distill_v2_2x/` (training artifacts)

**Modified**:
- `anime_upscaler/models/rfdn.py` (A1)
- `anime_upscaler/distill.py` (A2)
- `apps/anime_upscaler_gui/anime_upscaler_gui/pipeline.py` (A5)
- `apps/anime_upscaler_gui/anime_upscaler_gui/app.py` (A5)
- `apps/anime_upscaler_gui/anime_upscaler_gui/settings.py` (A5)
- `apps/anime_upscaler_gui/anime_upscaler_gui/registry.py` (A5)
- `apps/anime_upscaler_gui/anime_upscaler_gui/widgets/models_panel.py` (dropdown update if needed)

### P2. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Cascade quality < single 4× by > 0.3 dB | Medium | Eval first on small set; fall back to single 4× in pipeline if cascade loses too much. |
| 2× student training diverges | Low | Use same hyperparameters as v1 training (worked). Add early-stopping. |
| Cascade adds visible artifacts at sharp edges | Low | Train stage 2 with self-distilled outputs (Option 2). |
| TRT engine cache fills up with many cascade shapes | Low | Engines are keyed by full shape; ~10-20 engines total, ~20 MB each = 200-400 MB total. Acceptable. |
| User confusion: which mode to pick? | Low | Default to Single 4× (current proven path); Cascade marked as "experimental, faster" with measured numbers in tooltip. |

### P2. Estimated wall time

- A1+A2 (code change + distill.py arg): **0.5 day**
- A3 (train 2× student, Option 1): **~2 days** (training mostly idle)
- A4 (eval script + run): **0.5 day**
- A5 (cascade pipeline + GUI + tests): **1 day**
- **Total Phase 2: ~4 working days**

---

## Shared / cross-phase concerns

### Engine cache management

- Path: `%APPDATA%/anime_upscaler_gui/cache/trt/`.
- Filename: `engine_<model_hash>_<H>x<W>_bs<N>_<fp16|fp32>.engine`.
- On app startup, prune engines whose model_hash no longer matches a known model.
- Cap total cache size at 1 GB; LRU-evict oldest.

### First-run latency

- Engine build is 30-60 s per shape. Don't block the GUI.
- Worker thread emits a "Preparing inference engine for 960×540 (this happens once)…" log line.
- Status bar shows `Building engine… 32%` based on cache hit progress (since cache key set is small at startup).

### Fallback paths

- No TensorRT → eager PyTorch FP16 (current behavior). Toggle in Settings to disable explicitly.
- No NVENC → libx264 -preset ultrafast (current behavior). Toggle to disable explicitly.
- Cascade quality regression → Single 4× (current default).

### Documentation

Update `docs/rfdn_realtime_report.md` at end of each phase with measured fps numbers on `.venv/test_4k_540p_input.mp4`.

---

## Combined timeline

```
Week 1 (Phase 1):  B (TRT) + C (batching) + N (NVENC) → 25+ fps @ 4K
Week 2 (Phase 2):  A (cascade 2× + 2×) + C + N → 30+ fps @ 4K, similar quality
```

---

## Open decisions (user input received)

| # | Decision | Resolution |
|---|---|---|
| 1 | TensorRT install path | Try `torch_tensorrt` first; auto-fallback to ONNX + Python `tensorrt`. |
| 2 | GUI defaults for new toggles | TRT ON by default; NVENC ON by default; both auto-disable when unavailable. |
| 3 | Phase 2 Option 1 vs 2 | Option 1 first (one 2× model applied twice); Option 2 only if quality regresses > 0.3 dB. |
| 4 | Phase 1 acceptance threshold | Ship at 25 fps @ 4K (meets target); Phase 2 is the 30 fps push. |

---

## Key file paths (quick reference)

- Student weight: `pretrained/RFDN_distill_v1_4x_student.pth` (315,844 student + 10,176 adapters = 326,020 params)
- Current GUI: `apps/anime_upscaler_gui/anime_upscaler_gui/`
- Pipeline: `apps/anime_upscaler_gui/anime_upscaler_gui/pipeline.py`
- FFmpeg wrapper: `apps/anime_upscaler_gui/anime_upscaler_gui/ffmpeg.py`
- Settings: `apps/anime_upscaler_gui/anime_upscaler_gui/settings.py`
- Widgets: `apps/anime_upscaler_gui/anime_upscaler_gui/widgets/`
- Distill script: `anime_upscaler/distill.py`
- RFDN model: `anime_upscaler/models/rfdn.py`
- Test clip: `.venv/test_1sec.mp4` (854×480, 30 fps, 30 frames from source middle)
- Report: `docs/rfdn_realtime_report.md`

## Hardware spec

- GPU: Quadro RTX 4000, 8 GB (~7.5 GB free)
- FP16: supported, currently active
- NVENC: supported (h264_nvenc, hevc_nvenc)
- VRAM budget for engines: ~1 GB cap (configurable)
- No multi-GPU — single GPU design
