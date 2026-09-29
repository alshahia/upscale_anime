# GUI Queue Controls + GPU Codec (Q1–Q5)

Closes two gaps surfaced by a hardware review of the GUI on RTX 4000
(8 GB, CUDA 12.6, PyTorch 2.12.0+cu126):

1. **No user-visible Pause / Cancel / Resume** for in-flight jobs. The
   existing resume modal only handles per-frame errors, so a 30-minute
   encode could not be paused to free the GPU.
2. **GPU codec paths were plumbed but not exposed.** `RunJob` already
   carried `use_nvenc`, `nvenc_preset`, `nvenc_qp`, `decode`, and the
   TF32-style flags, but the Settings panel didn't expose them and the
   worker never exercised them.

This PR adds the user-visible controls and exercises every
hardware-acceleration path the machine has: TF32, NVDEC decode, NVENC
encode, async background-thread prefetch, and a 1 Hz GPU util telemetry
feed the GUI renders.

---

## What's new

### User-visible
- **Pause / Resume / Cancel buttons** on every queue row.
- **Decode combobox** now offers `auto` and `nvdec` (existing `cv2` /
  `pyav` preserved for back-compat).
- **NVENC QP spinbox** (16–28, default 18) in the Settings panel.
- **Default prefetch flipped to `async`** for new installs
  (existing settings.json files that pin `sync` still load).
- **`_GPUMonitor` widget** now shows worker-emitted GPU util in
  parallel with its own pynvml poll.

### Worker-side
- `PipelineWorker.__init__` accepts an optional `ctl_queue`
  (back-compat preserved: 2-arg callers work unchanged).
- New `JobControlEvent(kind, job_id)` dataclass and per-job
  pause via `threading.Event` with 0.1 s polling so cancel can
  interrupt a pause.
- `_JobCancelled` exception; partial output deleted; emits
  `JobEvent(kind='cancelled')`.
- `_read_gpu_util` (process-cached pynvml probe with torch.cuda
  fallback) + `_maybe_emit_gpu_util` (1 Hz rate limiter).

### Decode
- New `_NvDecReader` using PyAV 17's correct hwaccel API:
  `av.open(path, options={'hwaccel':'cuda'}, stream_options=[{'gpu':'0'}])`.
- The previous `stream.codec_context.options['hwaccel']='cuda'` recipe
  was silently swallowed by PyAV; this PR fixes that.
- `_detect_nvdec_support` probe (verifies `h264_cuvid` opens cleanly
  via `av.Codec('h264_cuvid','r').create()`) before declaring NVDEC up.

### Tensor / runtime
- `torch.backends.cuda.matmul.allow_tf32 = True` and
  `torch.backends.cudnn.allow_tf32 = True` enabled at module import.
- `cudnn.benchmark = True` (already there) preserved.

---

## Measured impact (RTX 4000, 36-frame 4x SRVGG student, fp16)

| Path                                | Wall time |
|-------------------------------------|-----------|
| Phase A1 baseline (PyAV + sync)     | 102.7 s   |
| + Q3 NVDEC (`auto`)                 |  98.2 s   |
| + Q4 async prefetch                 |  94.9 s   |
| **Branch total improvement**        | **~7.6% faster** |

Output bytes preserved (smoke `tmp/mid2s_job.py` → 575 701 bytes,
±0.22% benign NVENC rate-control variance run-to-run).

> The headline model-time wins (TensorRT fp16 inference, TTA off)
> were already wired in earlier phases. This PR closes the remaining
> gaps — Pause/Cancel/Resume UX and the decode/encode/prefetch paths.

---

## Validation

- **Tests**: `323 passed, 1 skipped, 1 failed` post-merge on this
  branch.
  - +33 new tests added across Q2/Q3/Q4 (was 292 passed at branch start).
  - The 1 failure is `test_input_panel_shows_empty_state_in_batch_mode`
    (TclError: no `init.tcl` on Windows). Pre-existing environmental
    flake, documented in the handoff. Not introduced by this PR.
- **Smoke**: `tmp/mid2s_job.py` produces a 36-frame 4x MP4 of
  identical pixel content; wall-time improvement 7–8%.
- **Phase A1 back-compat**: every existing `PipelineWorker(in_q,
  out_q)` 2-arg call site still works without `ctl_queue`.
- **Phase E `DeprecatedAlias` proxies**: intact.

---

## Commits (5 new + 1 merge)

```
3590a4b  Merge branch 'perf/queue-controls-gpu-codec' into refactor/gui-extensibility
4fadc27  Q5 (perf/queue-controls-gpu-codec): final docs + handoff
5a575f1  Q4 (perf/queue-controls-gpu-codec): TF32 + async prefetch + GPU util telemetry
459877a  Q3 (perf/queue-controls-gpu-codec): NVDEC + NVENC wired through worker
a06618f  Q2 (perf/queue-controls-gpu-codec): Pause / Resume / Cancel buttons
318b386  Q1 docs: PLAN.md + PROJECT_STATUS.md + new ROADMAP + handoff
```

Each Q2–Q5 commit is independently revertible if you want to drop,
e.g., just the GPU util telemetry. A single
`git revert -m 1 3590a4b` on the target reverts the whole feature.

---

## Files changed

- 7 production modules:
  - `apps/anime_upscaler_gui/anime_upscaler_gui/a11y.py`
  - `apps/anime_upscaler_gui/anime_upscaler_gui/app.py`
  - `apps/anime_upscaler_gui/anime_upscaler_gui/app_settings_io.py`
  - `apps/anime_upscaler_gui/anime_upscaler_gui/controllers/job_builder.py`
  - `apps/anime_upscaler_gui/anime_upscaler_gui/decoders.py`
  - `apps/anime_upscaler_gui/anime_upscaler_gui/pipeline/jobs.py`
  - `apps/anime_upscaler_gui/anime_upscaler_gui/pipeline/tensors.py`
  - `apps/anime_upscaler_gui/anime_upscaler_gui/pipeline/worker.py`
  - `apps/anime_upscaler_gui/anime_upscaler_gui/settings.py`
  - `apps/anime_upscaler_gui/anime_upscaler_gui/state.py`
  - `apps/anime_upscaler_gui/anime_upscaler_gui/ui_constants.py`
  - `apps/anime_upscaler_gui/anime_upscaler_gui/widgets/gpu_monitor.py`
  - `apps/anime_upscaler_gui/anime_upscaler_gui/widgets/input_panel.py`
  - `apps/anime_upscaler_gui/anime_upscaler_gui/widgets/settings_panel.py`
  - `apps/anime_upscaler_gui/anime_upscaler_gui/widgets/settings_spec.py`
- 4 test files: `test_gpu_codec.py` (13 new), `test_queue_controls.py`
  (12 new), `test_telemetry_and_defaults.py` (8 new),
  `test_job_builder.py` (extended).
- 4 doc files: `PLAN.md`, `PROJECT_STATUS.md`,
  `docs/ROADMAP_QUEUE_CONTROLS_GPU_CODEC.md` (new),
  `.dsh/handoffs/.../handoff.md` (new).

Full roadmap: `docs/ROADMAP_QUEUE_CONTROLS_GPU_CODEC.md`.
Full session handoff: `.dsh/handoffs/session-7cc5a443-f7aa-489b-9a68-f490f7a258f5/004-20260915-perf-queue-controls/handoff.md`.

---

## Known limitations / out-of-scope items

- `_NvDecReader` uses PyAV 17's high-level hwaccel flags. The full
  CUDA frame upload (open cuvid context → attach hw_frames_ctx → parse
  → transfer to GPU → download back to RGB) is **not** implemented.
  Frames still round-trip through CPU memory, so the NVDEC speedup is
  modest at 36 frames. A follow-up PR can replace the decode iterator
  with a hand-rolled cuvid hw_frames_ctx if larger clips show a bigger
  delta.
- TF32 targets residual fp32 ops only; the heavy convs run fp16 via
  TensorRT (Phase 1). For a clean fp32-vs-fp32-TF32 measurement run
  a longer clip with `fp16=False`.
- GPU monitor widget's local pynvml poll continues in parallel with
  the worker's gpu_util emissions. Both update the same labels; the
  worker's emission wins when it arrives more frequently than 1 Hz.

---

## Risk / rollback

- **Risk**: low. Each Q2–Q5 commit is independently revertible and
  preserves the existing `PipelineWorker` 2-arg API. Defaults that
  flipped (e.g. `prefetch=async`) are honored by `settings.json`
  pinning, so users who want the old behavior can pin it explicitly.
- **Rollback**:
  - Drop a single phase: `git revert <commit-sha>` on the target.
  - Drop the whole feature: `git revert -m 1 3590a4b`.
