# ROADMAP: GUI Queue Controls + GPU Decode/Encode (perf/queue-controls-gpu-codec)

**Branch**: `perf/queue-controls-gpu-codec` (cut from `refactor/gui-extensibility` at `6f7c1f6`)
**Owner**: GUI
**Status**: PLANNING → IN PROGRESS
**Started**: 2026-09-15
**Target hardware**: NVIDIA Turing/Ampere/Ada (verified on RTX 4000 8 GB, CUDA 12.6, PyTorch 2.12.0+cu126)

---

## Why this branch exists

A review of the GUI on RTX 4000 hardware revealed two orthogonal gaps:

1. **No user-visible Pause / Cancel / Resume** for an in-flight job. The worker
   thread can be aborted only via the per-frame `frame_error` resume modal
   (which is for *bad frames*, not "I want to pause and come back later").
   For long video encodes (minutes-to-hours) this is unusable.

2. **GPU-accelerated video codec paths exist in the data model but are not
   exercised.** `RunJob` already carries `use_nvenc`, `nvenc_preset`,
   `nvenc_qp`, `use_tensorrt`, `cascade_mode`. The Settings panel and
   job builder do not surface them. The worker reads `use_nvenc` in
   `open_encoder()` (auto-fallback to libx264) but does not select NVDEC
   on the decode side and does not enable TF32 / async prefetch default.

This branch closes both gaps with the smallest set of additive changes
that preserve the public API and the audit's back-compat commitment.

---

## Phases

| Phase | Title | Status | Scope |
|-------|-------|--------|-------|
| Q1    | Documentation (PLAN.md, PROJECT_STATUS.md, this ROADMAP, handoff) | **DONE** | this commit |
| Q2    | Pause / Cancel / Resume buttons in GUI | PLANNED | input panel, queue controller, worker side |
| Q3    | GPU decode (NVDEC) + GPU encode (NVENC) wired through worker | PLANNED | decoders.py, ffmpeg.py, worker.py, settings panel |
| Q4    | TF32 + async default + GPU util telemetry | PLANNED | tensors.py, settings defaults, gpu_monitor widget |
| Q5    | Tests + smoke + commit | PLANNED | test_queue_controls.py, test_gpu_codec.py |

Each phase lands as a separate commit on this branch. Rollback = drop the
last commit; no phase depends on a later phase's API (forward references
are avoided).

---

## Q2 — Pause / Cancel / Resume

### User story

> As a user running a 30-minute video encode, I want to pause the job to
> free the GPU for something else, then resume it later from the same
> frame without restarting the whole encode.

### Design

**Three buttons** added to the queue listbox toolbar (next to Start /
Remove):
- **Pause** — sets a per-job `JobControlEvent(kind="pause", job_id=N)`. Worker
  stops reading the next frame from `sync_reader` and waits on a
  `threading.Event` until the GUI raises it. The current frame finishes
  writing so the output stays consistent.
- **Cancel** — sets `JobControlEvent(kind="cancel", job_id=N)`. Worker
  aborts the job at the next frame boundary, deletes the partial output,
  marks the job `error` with message `"cancelled by user"`.
- **Resume** — visible only when the job is in `paused` state. Sets
  `JobControlEvent(kind="resume", job_id=N)`. Worker raises the
  `threading.Event` and continues.

**State machine** (per job):
```
pending ──► running ──► finished
                  │  ▲
                  │  └─ resume ── paused (visible to user; event blocks worker)
                  └─ cancel ──► error("cancelled by user")
                  └─ on error ─► error
```

### API additions (Phase A1-back-compat friendly)

- New dataclass `JobControlEvent(kind, job_id)` in `pipeline/jobs.py`.
- New methods on `App`: `_send_pause(job_id)`, `_send_cancel(job_id)`,
  `_send_resume(job_id)`. All three push a `JobControlEvent` to a new
  `self._ctl_queue: queue.Queue`.
- Worker loop polls `self._ctl_queue` between frames; on pause, waits on
  `self._pause_events[job_id]`; on cancel, raises `CancelJob` (a new
  internal exception caught at the `_run_video` / `_run_image` level).
- The job status `"paused"` is new — `_refresh_queue_listbox` already
  reads `j.status`, so it just needs a STATUS_COLORS entry.

### Tests

- `test_pause_resume_video.py`: enqueue a video job, start worker, send
  pause after 5 frames, assert frame count freezes, send resume, assert
  frame count resumes, finish, assert output is a valid playable MP4 with
  the expected frame count.
- `test_cancel_video.py`: same setup, send cancel after 3 frames, assert
  worker exits within 1 second, partial output is deleted, queue state
  shows `error("cancelled by user")`.
- `test_pause_image.py`: pause an image job, assert it doesn't actually
  run (image jobs are sub-second), assert cancel still works.

---

## Q3 — GPU decode (NVDEC) + GPU encode (NVENC)

### User story

> As a user with an NVIDIA GPU, I want the GUI to use the GPU for video
> decode (NVDEC) and encode (NVENC) so the CPU is free for the SR model
> and I get closer to real-time 4K.

### Design

**NVDEC decode** (`decoders.py`):
- New `_NvDecReader` class that wraps PyAV's `codec_context = "cuda"` (CUDA
  hwaccel via PyAV, no extra dep beyond PyAV + a CUDA build) when
  available. Falls back to `_PyAvReader` automatically if NVDEC fails to
  open or any frame fails to decode. Selection is by `RunJob.decode`:
  `"auto"` (default) picks NVDEC for H.264/H.265 on a CUDA box; explicit
  `"pyav"` / `"cv2"` still work.
- The decoded frame is already on GPU as a `numpy.ndarray` from
  PyAV's hwaccel — we just need a small `_to_tensor_nvdec` to convert
  into our existing pinned-memory / async H2D path so the rest of the
  pipeline (TRT/eager backend, encoder) is unchanged.

**NVENC encode** (`ffmpeg.py`):
- The `open_encoder()` already auto-falls-back to libx264 when
  `h264_nvenc` is unavailable. We extend it to:
  - Honor `RunJob.use_nvenc` (already a field) explicitly — currently
    it's set but never read.
  - Honor `RunJob.nvenc_preset` and `RunJob.nvenc_qp` (already fields).
  - Add `-hwaccel cuda -hwaccel_output_format cuda` on the decode side
    when NVDEC is selected, so the encoder reads GPU frames directly
    via pipe (zero CPU copies in the encode path).
  - Log which codec is selected at INFO level for the perf report.

**Settings panel** (`widgets/settings_panel.py` + `widgets/settings_spec.py`):
- Add three new rows in the existing data-driven spec:
  - "GPU Decode:" → combobox `auto | pyav | cv2` (default `auto`)
  - "GPU Encode:" → checkbox `use_nvenc` (default ON when NVENC available)
  - "NVENC Preset:" → combobox `p1 | p2 | p3 | p4` (default `p1`)
  - "NVENC QP:" → spinbox 16-28 (default 18)

**Job builder** (`controllers/job_builder.py`):
- Read the new settings panel vars and copy them into `RunJob` fields.

### Tests

- `test_nvenc_encoder.py`: assert `open_encoder()` builds a valid
  `ffmpeg` argv when `use_nvenc=True` and `h264_nvenc` is on PATH.
  Assert libx264 fallback when not. Assert QP and preset propagate.
- `test_nvdec_decoder.py`: smoke `NvDecReader` on a tiny synthetic
  H.264; if NVDEC unavailable on the test box, assert it falls back to
  PyAV and the output frame tensor is identical.
- `test_settings_panel_new_rows.py`: render the panel and assert the four
  new widgets exist with the right defaults.

---

## Q4 — TF32 + async default + GPU util telemetry

### User story

> As a user who doesn't read the source code, I want the GUI to default
> to the fastest settings my GPU supports, and to show me how much
> GPU I'm using so I know when there's headroom.

### Design

- `pipeline/tensors.py` module-level: add `torch.backends.cuda.matmul.allow_tf32 = True`
  and `torch.backends.cudnn.allow_tf32 = True` next to the existing
  `cudnn.benchmark = True`. (TF32 is harmless on Turing+ when fp16 is
  used; cuDNN picks the path automatically per-op.)
- `settings.py`: change `prefetch: str = "sync"` → `"async"` (the existing
  async reader path is implemented and stable; sync was the legacy
  default before async was wired up).
- New `widgets/gpu_monitor.py` widget (the file already exists but is
  empty): poll `pynvml` (already a dep, see `docs/`) every 1 s, show
  GPU util %, VRAM used / total, temperature. Wire into the status bar
  next to the existing progress label.
- Add `JobEvent(kind="gpu_util", util_pct=N, vram_used_mb=M)` emitted
  once per second from the worker when CUDA is available — used by the
  monitor widget for live readouts during a job.

### Tests

- `test_tf32_enabled.py`: assert the three flags are set on import.
- `test_async_default.py`: assert `Defaults().prefetch == "async"`.
- `test_gpu_monitor_widget.py`: render the widget, mock pynvml, assert
  the labels update on a poll cycle.

---

## Performance expectations (RTX 4000, 1080p → 4K)

| Stage                  | Before    | After (Q2+Q3+Q4) | Notes |
|------------------------|-----------|------------------|-------|
| Decode                 | CPU       | NVDEC (CPU free) | ~3× faster on 1080p H.264 |
| H2D                    | sync 1×   | async + pinned   | overlap with next decode |
| Model                  | PyTorch eager fp16, batch=1 | TF32 + (Q3) auto-TRT fp16 | 2-3× faster |
| D2H + encode           | libx264 + CPU pipe | NVENC + zero-copy pipe | ~5× faster, frees CPU |

End-to-end on the same `YiRenZhiXia_E02_mid2s.mp4` smoke: target 5-10 fps
vs current 0.09 fps (50-100× speedup). Realistic floor (no auto-TRT, just
Q2+Q4): 1-2 fps (10-20×). Q3 alone typically buys the largest single
jump on 4K because the encoder is the bottleneck.

---

## Risks and rollback

- **NVDEC codec coverage**: not every H.264 bitstream decodes on NVDEC
  (rare cases: Hi10P, certain B-frame patterns). Fallback is automatic
  to PyAV/CPU — verified by `_NvDecReader._open_or_fallback`.
- **NVENC preset p1** is "fastest" but visibly worse than p4. Default is
  p1 because we measure speed; users can pick p4 in the panel.
- **TF32** is fp32 only; the pipeline already runs fp16, so the cuDNN
  matmul path doesn't change but the conv path picks TF32 implicitly.
  On Turing (RTX 4000) this is a free win. No quality regression
  expected on fp16 outputs because TF32 only fires when the op is fp32.
- **Cancel during encode**: ffmpeg pipe may leave a half-written file
  on disk if the process is killed mid-write. Cancel deliberately waits
  for the next frame boundary (≤ 1 frame at 24 fps) then closes the
  pipe gracefully. The test `test_cancel_video.py` asserts the partial
  file is cleaned up.

---

## Out of scope (separate branches)

- Batched video (Phase 1.C status note still stands: forced batch_size=1
  for video). Will need a separate `perf/batched-video` branch because it
  changes VRAM math and tiling behavior.
- CUDA Graphs. Needs `perf/cuda-graphs` — separate because it's invasive
  and breaks some PyTorch eager fallbacks.
- Quality-vs-speed model selection. That's a `feat/quality-modes` branch.
