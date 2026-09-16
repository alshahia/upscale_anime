# Handoff: perf/queue-controls-gpu-codec — Q1-Q5 COMPLETE

**Branch**: `perf/queue-controls-gpu-codec`
**Commits on branch** (5 new since `refactor/gui-extensibility`):
  - `6f7c1f6`  Preserve pre-existing uncommitted changes
  - `318b386`  Q1 docs (PLAN.md + PROJECT_STATUS.md + ROADMAP + this handoff)
  - `a06618f`  Q2 Pause / Resume / Cancel buttons
  - `459877a`  Q3 NVDEC + NVENC wired through worker
  - `5a575f1`  Q4 TF32 + async prefetch + GPU util telemetry
  - `<Q5>`     Final doc sync + speedup measurements

## What was delivered

User-visible:
  * Three new toolbar buttons in the queue panel (Pause / Resume / Cancel)
    with status-aware enable/disable.
  * `Decode:` combobox gains `auto` and `nvdec` (existing `cv2` / `pyav`
    preserved for back-compat).
  * `NVENC QP:` spinbox (16-28, default 18) now reachable from the
    Settings panel.
  * `prefetch` defaults to `async` for new installs (settings.json files
    that pin `sync` still load).
  * `_GPUMonitor` widget shows worker-emitted GPU util in addition to its
    own pynvml poll.

Worker-side:
  * `PipelineWorker.__init__` accepts an optional `ctl_queue` (back-compat
    preserved: existing 2-arg callers work without it).
  * `JobControlEvent(kind, job_id)` dataclass in `pipeline.jobs`.
  * `_drain_ctl_for_job` re-queues events for other job_ids so a slow
    pause for job 99 doesn't delay cancel for job 42.
  * `_check_pause` blocks on `threading.Event` with 0.1s polling so cancel
    can interrupt a pause.
  * `_JobCancelled` exception; partial output deleted; emits
    `JobEvent(kind='cancelled')`.
  * `_read_gpu_util` (process-cached pynvml probe with torch.cuda fallback)
    + `_maybe_emit_gpu_util` (1 Hz rate limiter).
  * `JobEvent(kind='gpu_util')` consumed by `_GPUMonitor.apply_event`.

Decode:
  * `_NvDecReader` using PyAV 17's correct hwaccel API:
    `av.open(path, options={'hwaccel':'cuda'}, stream_options=[{'gpu':'0'}])`.
  * The previous `stream.codec_context.options = {'hwaccel':'cuda'}`
    recipe was silently swallowed by PyAV -- this branch fixes that.
  * Probe verifies `h264_cuvid` opens cleanly before declaring NVDEC up;
    when the probe fails (driver mismatch, hybrid-graphics laptops)
    NVDEC is reported as unavailable.

Tensor / runtime:
  * `torch.backends.cuda.matmul.allow_tf32 = True` and
    `torch.backends.cudnn.allow_tf32 = True` enabled at module import.
  * `cudnn.benchmark = True` (already there) preserved.

## Validation results

  * **pytest**: 325 passed, 13 warnings (was 292 / 13 at branch start;
    +33 new tests across Q2/Q3/Q4).
  * **smoke** (`tmp/mid2s_job.py`): produces `YiRenZhiXia_E02_mid2s_x4.mp4`
    of identical 575701-byte content.
  * **Wall time** (RTX 4000, 36-frame 4x clip, SRVGG student, fp16):
        Q1 baseline (PyAV + sync)        : 102.7 s
        + Q3 NVDEC (auto)                :  98.2 s
        + Q4 async prefetch              :  94.9 s
        = branch total improvement       : ~7.6% faster
  * **Phase A1 back-compat**: every existing `PipelineWorker(in_q, out_q)`
    call site still works without `ctl_queue`.
  * **Phase E `DeprecatedAlias` proxies**: intact. `_PipelineWorker`
    still monkey-patches through to the real class.

## Known limitations / out-of-scope items

  * `_NvDecReader` uses PyAV's high-level hwaccel flags; the full CUDA
    frame upload (open cuvid context -> attach hw_frames_ctx -> parse
    -> transfer to GPU -> download back to RGB) is not yet implemented.
    Frames still round-trip through CPU memory, which is why the NVDEC
    speedup is modest at 36 frames. A follow-up branch can replace the
    decode iterator with a hand-rolled cuvid hw_frames_ctx if larger
    clips show a bigger delta.
  * TF32 path targets residual fp32 ops only; the heavy convs run fp16
    via TensorRT. The fp32 speedup is therefore small on this clip.
    For a clean fp32 vs fp32-TF32 measurement, run a longer clip with
    `fp16=False`.
  * GPU monitor widget's local pynvml poll continues in parallel with
    the worker's gpu_util emissions. Both update the same labels; the
    worker's emission wins when it arrives more frequently than 1 Hz.

## How to reproduce

```powershell
# Tests
.venv\Scripts\python.exe -m pytest apps/anime_upscaler_gui/tests `
  --ignore=apps/anime_upscaler_gui/tests/test_video_end_to_end.py `
  --ignore=apps/anime_upscaler_gui/tests/test_assets.py `
  --basetemp=E:/Temp/pytest_qN

# Smoke
.venv\Scripts\python.exe tmp/mid2s_job.py

# Force-NVDEC smoke (separate harness, not in the canonical smoke):
.venv\Scripts\python.exe -c "
import sys, queue, pathlib
sys.path.insert(0, r'E:/python projects/upscale_anime/apps/anime_upscaler_gui')
sys.path.insert(0, r'E:/python projects/upscale_anime')
from apps.anime_upscaler_gui.anime_upscaler_gui import pipeline as P
src = pathlib.Path(r'C:/Users/Ahmad Mahmoud/Downloads/Video/[Anime3rb.com] Yi Ren Zhi Xia - 1 [480p].mp4')
out = pathlib.Path('E:/Temp/_q3_auto.mp4')
in_q, out_q = queue.Queue(), queue.Queue()
w = P.PipelineWorker(in_q, out_q); w.start()
in_q.put(P.RunJob(
    job_id=1, input_path=src, output_path=out, is_video=True,
    model_filename=r'E:/python projects/upscale_anime/pretrained/SRVGG_distill_v1_4x_student.pth',
    kind='srvgg_student', scale=4, outscale=4.0, fp16=True, device='cuda',
    batch_size=4, decode='auto', prefetch='async', pin_memory='auto',
    downscale_max_edge=576, gpu_guard_mode='warn', tile_size=256, tile_overlap=32,
    cut_start_seconds=712.03, cut_end_seconds=714.03,
    video_crf=18, video_preset='medium', tta=False,
    use_tensorrt=True, use_nvenc=True, nvenc_preset='p1', nvenc_qp=18))
while True:
    evt = out_q.get(timeout=600)
    if evt.kind in ('finished','error','fatal_error'): break
w.stop(); w.join(timeout=10)
print('done', evt.kind)
"
```

## Files changed in the branch (commits)

Q1 (`318b386`): PLAN.md, PROJECT_STATUS.md, this handoff, new
  docs/ROADMAP_QUEUE_CONTROLS_GPU_CODEC.md.
Q2 (`a06618f`): apps/anime_upscaler_gui/anime_upscaler_gui/state.py,
  ui_constants.py, a11y.py, pipeline/jobs.py, pipeline/worker.py,
  widgets/input_panel.py, app.py, tests/test_queue_controls.py.
Q3 (`459877a`): decoders.py, pipeline/worker.py, widgets/settings_panel.py,
  widgets/settings_spec.py, controllers/job_builder.py,
  app_settings_io.py, tests/test_gpu_codec.py, tests/test_job_builder.py.
Q4 (`5a575f1`): pipeline/tensors.py, pipeline/worker.py, settings.py,
  widgets/gpu_monitor.py, app.py, tests/test_telemetry_and_defaults.py.
Q5 (`<this>`): docs/ROADMAP_QUEUE_CONTROLS_GPU_CODEC.md,
  PROJECT_STATUS.md, this handoff.

## Next steps (if any)

  1. (Optional) follow-up branch: real cuvid hw_frames_ctx upload in
     `_NvDecReader` for the full GPU decode speedup.
  2. (Optional) end-user-visible "use_nvdec" toggle in the Settings panel
     beyond the Decode combobox (the Decode combobox already covers
     this; an explicit checkbox would be redundant).
  3. (No action) the recommended model uses fp16; TF32 contributes
     little for the heavy convs. The branch's headline win is async
     prefetch + NVDEC; bigger model-time wins require TensorRT, which
     was already wired in Phase 1.
