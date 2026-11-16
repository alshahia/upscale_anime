"""PipelineWorker: threaded driver that consumes RunJob objects and emits JobEvents.

Owns the per-job lifecycle:
  1. dispatch to image or video path
  2. backend selection (PyTorch / TensorRT via make_backend)
  3. decode frames
  4. infer (with optional cascade + TTA)
  5. encode + write
  6. emit progress / finished / error events

Reads its inputs from sibling submodules so this file stays focused on flow
control:
  * ``jobs.RunJob`` / ``jobs.JobEvent`` / ``jobs.SKIP_*`` -- data carriers
  * ``tensors.to_tensor`` / ``tensor_to_bgr`` / ``downscale_if_needed`` / ...
  * ``backends.make_backend`` / ``backends.cascade_count`` /
    ``backends.tta_forward``
  * ``encoder.encode_write`` -- ffmpeg pipe guard

Phase A2 of the GUI extensibility refactor moved these out of the monolithic
``pipeline.py`` (1034 LOC) into a focused submodule. The public API stays the
same: every name is re-exported from ``pipeline.__init__``.
"""
import os
import queue
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Any, List, Optional  # noqa: F401

import cv2
import torch
import torch.nn.functional as F

from ..arch_registry import build
from ..arch_registry import capabilities as _arch_caps
from ..decoders import (_AsyncReader, _Cv2Reader, _NvDecReader, _PyAvReader, _SkipFirstFrames,
                        load_image_rgb, save_image_rgb)
from ..ffmpeg import (extract_cut, ffmpeg_available, has_audio_stream, open_encoder,
                     probe_duration)

from .backends import _cascade_count, _make_backend, _tta_forward
from .encoder import _encode_write
from .jobs import (
    ABORT_JOB,
    RETRY_FRAME,
    SKIP_FRAME,
    SKIP_REST,
    JobControlEvent,
    JobEvent,
    RunJob,
)
from .tensors import (_PinnedPool, _downscale_if_needed, _resize_keep_ar,
                      _tensor_to_bgr, _tensor_to_bgr_batch, _to_tensor,
                      _to_tensor_batch)

_pinned = _PinnedPool()

# AsyncReader queue depth per prefetch mode (job.prefetch is "sync" |
# "async"). Kept in one place so the depth is tunable without touching
# the call site.
_PREFETCH_DEPTH = {"async": 8}


# Q2 (perf/queue-controls-gpu-codec): internal exception used to bail out
# of _run_video / _run_image / _process_video_batch when the user cancels.
# Caught at the run() level; we emit a clean "cancelled" event and the
# caller deletes the partial output file. Using a dedicated exception
# (vs. a magic return value) keeps the cancel path composable with the
# existing on_frame_error handler.
class _JobCancelled(Exception):
    """Raised internally by the worker when a cancel control arrives."""


def _same_file(input_path: Path, output_path: Path) -> bool:
    """True when both paths name the same filesystem entry (hard links too).

    Worker-side mirror of controllers.job_builder._same_file: the worker
    re-validates every job (batch queues, persisted/reloaded queues) before it
    opens an encoder, so it must catch hard links as well as spelling changes.
    """
    try:
        if input_path.exists() and output_path.exists():
            return os.path.samefile(input_path, output_path)
    except OSError:
        pass
    try:
        a = os.path.realpath(os.fspath(input_path))
        b = os.path.realpath(os.fspath(output_path))
    except (OSError, ValueError):
        a = os.path.abspath(os.fspath(input_path))
        b = os.path.abspath(os.fspath(output_path))
    return os.path.normcase(a) == os.path.normcase(b)


def _unlink_partial_output(out_path) -> None:
    """Best-effort delete of a partial output file.

    Shared by the cancel path (``_run_one``) and the recurrent error path so
    a job that never reached its "finished" event leaves no truncated file
    behind. The caller must have closed the encoder handle first: on Windows
    ffmpeg / cv2 keep the file open, and unlinking it early raises
    PermissionError.
    """
    try:
        if out_path.exists():
            out_path.unlink()
    except OSError:
        pass


def _handle_encode_result(worker, job, out_path, encode_rc, frames) -> bool:
    """Shared post-encode gate: True only when the output may be reported.

    Task 7: both video paths used to ignore the ffmpeg exit code (and an
    empty output file), so a failed encode could still be announced as
    "finished" and leave a truncated file behind. When this returns False
    the error event emitted below is the job single terminal event, so the
    caller must return without emitting "finished" or remuxing audio.
    """
    if encode_rc != 0:
        _unlink_partial_output(out_path)
        worker.emit(JobEvent(kind="error", job_id=job.job_id,
                             message=f"encoder failed (exit {encode_rc}); output discarded"))
        return False
    if frames == 0 or not out_path.exists() or out_path.stat().st_size == 0:
        _unlink_partial_output(out_path)
        worker.emit(JobEvent(kind="error", job_id=job.job_id,
                             message="encoder produced no usable output"))
        return False
    return True


def _mux_job_audio(out_path, input_path, *, start_seconds: float = 0.0,
                   duration_seconds: float | None = None) -> bool:
    """Remux the source audio into the encoded output.

    Returns False when the source has no audio (video-only output stays
    valid); the caller treats a False on an audio-bearing source as a warning
    it must show, never as silence shipped quietly.
    """
    from ..ffmpeg import has_audio_stream, mux_audio

    if not has_audio_stream(input_path):
        return False
    return mux_audio(out_path, input_path, out_path,
                     start_seconds=start_seconds,
                     duration_seconds=duration_seconds)


def _audio_mux_window(out_path, *, fps: float, start_frame: int,
                      n_frames: int) -> tuple[float, float | None]:
    """Return the audio window (start, duration) matching the encoded picture.

    Derived from the frames that actually reached out_path -- the frame the
    decoder skipped to and the finished file duration -- never from the raw
    cut fields. cut_end_seconds == 0 means "to EOF", so a start-only cut is a
    positive window; deriving it as cut_end - cut_start used to fall back to
    the FULL source audio from 0:00 and desync the export. The start is
    frame-exact (start_frame / fps) so the sound matches the first encoded
    frame, and the duration is probed from the finished file, falling back to
    the encoded frame count / fps when ffprobe cannot report one.
    """
    start_seconds = start_frame / fps if fps > 0 and start_frame > 0 else 0.0
    duration_seconds = probe_duration(out_path)
    if duration_seconds is None or duration_seconds <= 0:
        duration_seconds = (n_frames / fps) if fps > 0 and n_frames > 0 else None
    return start_seconds, duration_seconds


# Q4 (perf/queue-controls-gpu-codec): light-weight GPU util probe used
# by the worker to emit ``JobEvent(kind="gpu_util")`` once per second.
# We prefer pynvml (the canonical NVIDIA System Management Interface)
# because it gives util / VRAM / temperature with no CUDA runtime cost.
# Falls back to a torch.cuda.memory_allocated snapshot when pynvml is
# missing (common on Windows test boxes); util% is reported as 0 in that
# case because PyTorch itself has no CUDA utilisation counter.
_GPU_UTIL_PROBE: Optional[dict] = None  # lazy probe cache
_GPU_UTIL_MIN_INTERVAL_S = 1.0           # worker emits gpu_util at most 1Hz


def _read_gpu_util() -> Optional[dict]:
    """Return ``{util, mem_used_mb, mem_total_mb, temp_c}`` or None.

    Process-cached: the first call probes pynvml + opens the device
    handle; subsequent calls reuse the same handle until the process
    ends. Returns None when neither pynvml nor torch.cuda is available.
    """
    global _GPU_UTIL_PROBE
    if _GPU_UTIL_PROBE is not None:
        return _GPU_UTIL_PROBE
    probe: Optional[dict] = None
    try:
        import pynvml  # type: ignore
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
            probe = {
                "util": int(util),
                "mem_used_mb": int(mem.used) // (1024 * 1024),
                "mem_total_mb": int(mem.total) // (1024 * 1024),
                "temp_c": int(temp),
                "_handle": handle,
                "_pynvml": pynvml,
            }
        except Exception:
            probe = None
    except Exception:
        probe = None
    if probe is None:
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                free, total = _torch.cuda.mem_get_info()
                probe = {
                    "util": 0,
                    "mem_used_mb": (total - free) // (1024 * 1024),
                    "mem_total_mb": total // (1024 * 1024),
                    "temp_c": 0,
                }
        except Exception:
            probe = None
    _GPU_UTIL_PROBE = probe if probe is not None else {}
    return _GPU_UTIL_PROBE or None


def _maybe_emit_gpu_util(emit_fn, job_id: int, last_emit_ref: list) -> None:
    """Emit a gpu_util JobEvent at most once per second.

    ``last_emit_ref`` is a single-element list so the closure can mutate
    the caller's timestamp (Python 3 lacks the ``nonlocal`` workaround
    that's needed when state lives on the worker instance only).
    """
    now = time.perf_counter()
    if now - last_emit_ref[0] < _GPU_UTIL_MIN_INTERVAL_S:
        return
    info = _read_gpu_util()
    if not info:
        return
    emit_fn(JobEvent(
        kind="gpu_util",
        job_id=job_id,
        message=f"util={info['util']}% mem={info['mem_used_mb']}/"
                f"{info['mem_total_mb']}MB t={info['temp_c']}C",
        fps=float(info["util"]),
    ))
    last_emit_ref[0] = now


# Q2: sentinel events used to wake a paused worker between frames.
# Keyed by job_id so multiple jobs can have independent pause state.
# The worker creates the Event on first pause, clears it on resume,
# and pops it from the dict when the job ends (so a stale event can't
# accidentally wake the next job that happens to reuse the same id).

# ============================================================================ #
# Pipeline worker (threaded, drives one job at a time from the in_queue)
# ============================================================================ #
class PipelineWorker(threading.Thread):
    """Consumes RunJob objects from in_queue, emits JobEvent to out_queue.

    Public alias for the previous private name ``_PipelineWorker``.
    Existing scripts that imported ``_PipelineWorker`` keep working; that
    name is kept as a deprecated alias.

    Lifecycle:
        start()  -> thread begins polling in_queue
        put(job) -> enqueue a job (jobs run sequentially)
        stop()   -> enqueue a sentinel and join()
    """

    def __init__(self, in_queue: "queue.Queue", out_queue: "queue.Queue",
                 ctl_queue: Optional["queue.Queue"] = None):
        """``ctl_queue`` is the GUI->worker pause/cancel/resume channel.

        If omitted (legacy callers / tests), the worker synthesizes an
        internal queue so the same code path works whether or not the
        GUI wires the new control channel. This keeps the Phase A1
        back-compat commitment intact: existing tests that build a
        PipelineWorker with just (in_queue, out_queue) keep passing.
        """
        super().__init__(daemon=True)
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.ctl_queue = ctl_queue if ctl_queue is not None else queue.Queue()
        # job_id -> threading.Event. Lazily populated by _check_pause();
        # cleared on job completion so we don't leak events across jobs.
        self._pause_events: dict = {}
        self._shutdown = False

    # -- Q2 control-channel helpers -------------------------------------------
    def send_control(self, evt: JobControlEvent) -> None:
        """Public helper the GUI uses to push pause/cancel/resume.

        Equivalent to ``self.ctl_queue.put(evt)`` but exposed as a
        method so the GUI doesn't have to know which queue to use.
        """
        self.ctl_queue.put(evt)

    def _drain_ctl_for_job(self, job_id: int) -> None:
        """Pop and apply any pending control events for this job.

        Called between frames in _run_video and after model load in
        _run_image. Cancel is applied immediately (raises
        _JobCancelled). Pause/resume just adjust the per-job event --
        the actual blocking happens in _check_pause().

        Drains the whole queue (not just up to the first foreign event),
        applies the ones for `job_id`, and puts the rest back in their
        original order -- control events are FIFO per sender, and the
        worker only cares about its current job.
        """
        pending = []
        try:
            while True:
                pending.append(self.ctl_queue.get_nowait())
        except queue.Empty:
            pass
        cancel = False
        for evt in pending:
            if evt.job_id != job_id:
                # For a different (probably future) job: preserve order.
                self.ctl_queue.put(evt)
                continue
            if evt.kind == "cancel":
                # Raise after the loop so events queued behind this one
                # still get put back.
                cancel = True
                continue
            if evt.kind == "pause":
                # Make sure a paused event exists for this job.
                self._pause_events.setdefault(job_id, threading.Event())
                self.emit(JobEvent(kind="log", job_id=job_id,
                                    message="paused by user"))
            elif evt.kind == "resume":
                ev = self._pause_events.get(job_id)
                if ev is not None:
                    ev.set()
                self.emit(JobEvent(kind="log", job_id=job_id,
                                    message="resumed by user"))
        if cancel:
            raise _JobCancelled()

    def _check_pause(self, job_id: int) -> None:
        """Block the worker on the per-job pause event until resume.

        Called at every frame boundary in _run_video and after model
        load in _run_image. Polls the control queue while blocked so
        cancel can interrupt the pause. Returns once the event is set
        (resume arrived) or raises _JobCancelled if a cancel arrived
        during the pause.
        """
        ev = self._pause_events.get(job_id)
        if ev is None or ev.is_set():
            return
        self.emit(JobEvent(kind="log", job_id=job_id,
                            message="waiting on resume"))
        # Use a short timeout so we can poll the control queue for cancel.
        while not ev.is_set():
            # Drain any controls that arrived -- they might be a cancel.
            self._drain_ctl_for_job(job_id)
            ev.wait(timeout=0.1)

    def _finalize_job(self, job_id: int) -> None:
        """Clean up per-job pause state when the job ends (any reason).

        Ensures a stale pause event can't accidentally wake a future
        job that reuses the same id. Called from _run_one's outer
        try/finally.
        """
        self._pause_events.pop(job_id, None)

    def stop(self):
        self._shutdown = True
        self.in_queue.put(None)

    def emit(self, evt: JobEvent):
        self.out_queue.put(evt)

    def run(self):
        try:
            while not self._shutdown:
                job = self.in_queue.get()
                if job is None:
                    return
                try:
                    self._run_one(job)
                except Exception as e:
                    self.emit(JobEvent(kind="error", job_id=job.job_id,
                                        message=f"worker crashed: {e}",
                                        exc=e))
                    traceback.print_exc()
        except BaseException as e:
            # Worker thread itself died (e.g. out_queue.put crashed, in_queue
            # unusable). Surface a fatal_error so the GUI can show a banner
            # instead of silently going dark.
            self.emit(JobEvent(kind="fatal_error", job_id=0,
                                message=f"worker thread died: {e}",
                                exc=e))
            traceback.print_exc()

    # --- per-job dispatcher ---
    def _run_one(self, job: RunJob):
        if not job.input_path.exists():
            self.emit(JobEvent(kind="error", job_id=job.job_id,
                                message=f"input not found: {job.input_path}"))
            return
        # Task 8: an encoder opened on the input path truncates the user
        # source before a frame is read. Refuse before any file is touched
        # (the GUI validates too; this catches persisted/reloaded queues).
        # Task 8 review follow-up: resolve() equality misses hard links (two
        # names, one inode), so compare with os.path.samefile when both paths
        # exist and only fall back to the resolved compare for a destination
        # that does not exist yet.
        if _same_file(job.input_path, job.output_path):
            self.emit(JobEvent(kind="error", job_id=job.job_id,
                                message="output path equals input path; refusing to overwrite the source"))
            return
        try:
            if job.is_video:
                self._run_video(job)
            else:
                self._run_image(job)
        except _JobCancelled:
            # Clean up partial output. The encoder pipe is already closed
            # because _run_video/_run_image's `finally` block ran on the
            # way out. We just delete the file and emit a terminal event.
            _unlink_partial_output(job.output_path)
            self.emit(JobEvent(kind="cancelled", job_id=job.job_id,
                                message="cancelled by user"))
        finally:
            self._finalize_job(job.job_id)

    # --- image ---
    def _run_image(self, job: RunJob):
        self.emit(JobEvent(kind="started", job_id=job.job_id,
                            message=f"image: {job.input_path.name}"))
        # Q2: drain pending control events so cancel-during-load is
        # honoured. Image jobs are sub-second, so a pause request is
        # effectively a no-op (the work finishes before we get back to
        # _check_pause), but cancel still aborts cleanly.
        self._drain_ctl_for_job(job.job_id)
        device = torch.device(job.device if torch.cuda.is_available() else "cpu")
        ckpt_path = _find_ckpt(job)
        try:
            torch_model = build(job.kind, ckpt_path).to(device).eval()
            if job.fp16:
                torch_model = torch_model.half()
        except Exception as e:
            self.emit(JobEvent(kind="error", job_id=job.job_id,
                                message=f"load failed: {e}", exc=e))
            return
        backend, backend_name = _make_backend(torch_model, ckpt_path, job.kind, device, job.fp16, job.use_tensorrt, job.tta)

        # TTA honors the arch's Capability: recurrent kinds (animesr) take a
        # 5-D (B, N, C, H, W) input the D4 TTA wrapper does not produce.
        use_tta = job.tta and _arch_caps(job.kind).tta

        rgb = load_image_rgb(job.input_path)
        if rgb is None:
            self.emit(JobEvent(kind="error", job_id=job.job_id,
                                message=f"cv2 failed to read: {job.input_path}"))
            return
        rgb = _downscale_if_needed(rgb, job.downscale_max_edge)
        h0, w0 = rgb.shape[:2]

        # Pre-allocate pinned host buffer for the LR input (Phase 5 perf:
        # non_blocking + cached pinned memory -> ~7 ms saved vs the original
        # per-frame pageable->pinned staging). Only used when on CUDA.
        pinned_in = _pinned.get_input(h0, w0) if device.type == "cuda" else None
        x = _to_tensor(rgb, device, job.fp16, fp16_pin=False, pinned_in=pinned_in)
        pm = _arch_caps(job.kind).pad_multiple
        if pm:
            ph = (pm - h0 % pm) % pm
            pw = (pm - w0 % pm) % pm
            if ph or pw:
                x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        # Recurrent temporal models (animesr) need the 5-D
        # (b, n, c, h, w) contract the same way the video path does; a
        # still image has no temporal context, so the LR frame is
        # replicated ``recurrent_frames`` times and the centre output
        # frame is kept. Without this dispatch the 4-D input reaches
        # MSRSW.forward, which unpacks five sizes and fails.
        recurrent_frames = _arch_caps(job.kind).recurrent_frames
        try:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                # Cascade: 2x model applied twice -> 4x; 4x model once.
                n_cascade = (job.cascade_mode if job.cascade_mode is not None
                             else _cascade_count(backend.model))
                if recurrent_frames > 0:
                    y = self._recurrent_image_forward(
                        backend, x, n_cascade, recurrent_frames)
                else:
                    y = x
                    for _ in range(n_cascade):
                        if use_tta:
                            y = _tta_forward(backend.model, y)
                        else:
                            y = backend(y)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            self.emit(JobEvent(kind="error", job_id=job.job_id,
                                message=f"inference failed: {e}", exc=e))
            return
        om = _arch_caps(job.kind).out_multiple
        if om:
            y = y[..., :h0 * om, :w0 * om]
        if job.fp16:
            y = y.float()

        # Pre-allocate pinned host buffer for the SR output. The fast path in
        # _tensor_to_bgr casts to uint8 ON the GPU, then async-copies to the
        # pinned buffer -- ~10x faster than the default float32->cpu->*255 path.
        out_h_y, out_w_y = int(y.shape[-2]), int(y.shape[-1])
        pinned_out = _pinned.get_output(out_h_y, out_w_y) if device.type == "cuda" else None
        sr_bgr = _tensor_to_bgr(y, pinned_out=pinned_out)
        # The model may have native scale != user's outscale.
        # Resize the SR to exactly (h0*outscale, w0*outscale).
        target_h = int(round(h0 * job.outscale))
        target_w = int(round(w0 * job.outscale))
        sr_bgr = _resize_keep_ar(sr_bgr, target_h, target_w)
        if save_image_rgb(cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2RGB), job.output_path):
            self.emit(JobEvent(kind="finished", job_id=job.job_id,
                                message=f"-> {job.output_path.name} ({dt_ms:.1f} ms)",
                                infer_ms=dt_ms, progress=1.0))
        else:
            self.emit(JobEvent(kind="error", job_id=job.job_id,
                                message=f"save failed: {job.output_path}"))

    def _recurrent_image_forward(self, backend, x, n_cascade: int,
                                 recurrent_frames: int):
        """Still-image forward through a recurrent 5-D model.

        ``x`` is (1, 3, H, W). It is replicated to (1, n, 3, H, W) so the
        recurrent cell sees a full prev/cur/nxt context and the centre
        output frame is returned as (1, 3, H', W'). This is the same trick
        ``_OnnxBackend.__call__`` uses for still images on recurrent
        kinds; without it every AnimeSR image job feeds 4-D input into
        ``MSRSW.forward`` and dies unpacking ``b, n, c, h, w``.
        """
        n = max(1, int(recurrent_frames))
        # (1, 3, H, W) -> (1, n, 3, H, W); repeat (not expand) so the model
        # gets a real tensor rather than a stride-0 view.
        y = x.unsqueeze(1).repeat(1, n, 1, 1, 1)
        for _ in range(max(1, n_cascade)):
            y = backend(y)
        return y[:, n // 2]

    # --- video ---
    def _run_video(self, job: RunJob):
        self.emit(JobEvent(kind="started", job_id=job.job_id,
                            message=f"video: {job.input_path.name}"))
        # Q2: drain any pending control events that arrived while the
        # job was queued. If the user cancelled while we were loading,
        # bail out before we waste time on a 4x super-res.
        self._drain_ctl_for_job(job.job_id)
        device = torch.device(job.device if torch.cuda.is_available() else "cpu")
        ckpt_path = _find_ckpt(job)
        try:
            torch_model = build(job.kind, ckpt_path).to(device).eval()
            if job.fp16:
                torch_model = torch_model.half()
        except Exception as e:
            self.emit(JobEvent(kind="error", job_id=job.job_id,
                                message=f"load failed: {e}", exc=e))
            return
        backend, backend_name = _make_backend(torch_model, ckpt_path, job.kind, device, job.fp16, job.use_tensorrt, job.tta, batch_size=job.batch_size)

        # Recurrent temporal models (e.g. AnimeSR) need all frames at once
        # for prev/cur/nxt context. Dispatch to a dedicated 5D path.
        if _arch_caps(job.kind).recurrent_frames > 0:
            self._run_video_recurrent(job, backend, device)
            return

        if job.decode == "nvdec":
            # Q3: explicit NVDEC selection. _NvDecReader raises if the
            # probe said NVDEC is unavailable so the user gets a clean
            # error instead of a silent CPU fallback.
            sync = _NvDecReader(job.input_path)
        elif job.decode == "pyav":
            sync = _PyAvReader(job.input_path)
        elif job.decode == "auto":
            # Q3: prefer NVDEC when available; same dispatch as the
            # open_reader() factory.
            from ..decoders import _detect_nvdec_support, _HAS_PYAV
            if _detect_nvdec_support():
                sync = _NvDecReader(job.input_path)
            elif _HAS_PYAV:
                sync = _PyAvReader(job.input_path)
            else:
                sync = _Cv2Reader(job.input_path)
        else:
            sync = _Cv2Reader(job.input_path)
        fps, total, w, h = sync.fps, sync.total, sync.w, sync.h
        target_w = int(round(w * job.outscale))
        target_h = int(round(h * job.outscale))

        # Cut window: total range and start..end frame indices. total == 0 or
        # fps <= 0 means unknown metadata (PyAV cannot always report
        # stream.frames), so treat it as "run to EOF" instead of failing with
        # "cut window is empty".
        unknown_meta = (fps <= 0 or total <= 0)
        if fps > 0:
            start_frame = int(round(job.cut_start_seconds * fps))
        else:
            start_frame = 0
        if job.cut_end_seconds > 0 and fps > 0:
            end_frame = int(round(job.cut_end_seconds * fps))
            if total > 0:
                end_frame = min(end_frame, total)
        elif total > 0:
            end_frame = total
        else:
            end_frame = -1  # unbounded: run to EOF
        if end_frame >= 0 and end_frame <= start_frame:
            self.emit(JobEvent(kind="error", job_id=job.job_id,
                                message="cut window is empty (end <= start)"
                                        if end_frame == start_frame
                                        else f"cut range exceeds video (fps={fps:g}, frames={total})"))
            sync.release()
            return

        # Frame stride / limit (limit == 0 means unbounded: run to EOF)
        limit = max(end_frame - start_frame, 0) if end_frame > 0 else 0
        # Re-open the *selected* decoder and skip to start_frame. The old code
        # always re-opened with cv2, silently ignoring decode=pyav; use the
        # reader factory so the selected decode backend actually decodes.
        sync.release()
        if job.decode == "nvdec":
            new_sync_reader = _SkipFirstFrames(_NvDecReader(job.input_path), start_frame)
            async_inner = new_sync_reader
        elif job.decode == "pyav":
            new_sync_reader = _SkipFirstFrames(_PyAvReader(job.input_path), start_frame)
            async_inner = new_sync_reader
        elif job.decode == "auto":
            from ..decoders import _detect_nvdec_support, _HAS_PYAV
            if _detect_nvdec_support():
                new_sync_reader = _SkipFirstFrames(_NvDecReader(job.input_path), start_frame)
            elif _HAS_PYAV:
                new_sync_reader = _SkipFirstFrames(_PyAvReader(job.input_path), start_frame)
            else:
                new_sync_reader = _SkipFirstFrames(_Cv2Reader(job.input_path), start_frame)
            async_inner = new_sync_reader
        else:
            cap = cv2.VideoCapture(str(job.input_path))
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            new_sync_reader = _Cv2Reader.from_cap(
                cap, job.input_path, fps, total, w, h)
            async_inner = new_sync_reader
        if job.prefetch == "async":
            sync_reader = _AsyncReader(async_inner, prefetch=_PREFETCH_DEPTH.get(job.prefetch, 8))
            sync_reader.start(max_frames=limit)
        else:
            sync_reader = new_sync_reader

        out_path = job.output_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pipe = open_encoder(
            out_path, fps, target_w, target_h,
            crf=job.video_crf, preset=job.video_preset,
            use_nvenc=job.use_nvenc, nvenc_preset=job.nvenc_preset, nvenc_qp=job.nvenc_qp,
        )
        use_pipe = pipe is not None
        # When use_pipe=True the ffmpeg subprocess owns the pipe and `writer`
        # is never used; default to None so the batched path (Phase 1.C)
        # does not hit UnboundLocalError when it references `writer` later.
        writer = None
        if not use_pipe:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (target_w, target_h))
            if not writer.isOpened():
                self.emit(JobEvent(kind="error", job_id=job.job_id, message="no encoder (ffmpeg/cv2)"))
                sync_reader.release()
                return

        skip_remaining = False
        proc = 0
        encode_rc = 0
        # Task 9 follow-up: track a caught-stage failure so the finally can
        # discard the truncated output even when the encoder dies mid-stream.
        failed = False
        stage = "decode"
        t_wall = time.perf_counter()
        last_emit = t_wall
        # Batched path is incompatible with TTA (8 augmentations x N would
        # explode VRAM) and animesr (3-frame-center trick needs per-frame
        # handling). Fall back to per-frame for those.
        use_batch = (
            job.batch_size > 1
            and not job.tta
            and _arch_caps(job.kind).batch_video
            and device.type == "cuda"
        )
        try:
            for frame_idx_global, payload in _batched(sync_reader, job.batch_size, limit):
                if skip_remaining:
                    # Still consume to drain the reader cleanly
                    continue
                # Q2: pause / cancel check between frames. We block on
                # the per-job threading.Event if the user paused, or
                # raise _JobCancelled if the user cancelled. Both ops
                # happen at the frame boundary so the output MP4 stays
                # consistent (the previous frame's encode is already
                # flushed to the pipe).
                self._drain_ctl_for_job(job.job_id)
                self._check_pause(job.job_id)
                stage = "decode"
                # Normalize payload: single frame (np.ndarray H,W,3) for
                # batch_size=1; list of N frames for batch_size>1.
                if use_batch and isinstance(payload, list):
                    # Single-element list refs (mutable scalars via list wrapping).
                    # Pre-bind them so they are accessible after the call --
                    # Python does not bind a local for `proc_ref=[proc]` kwargs.
                    last_emit_ref = [t_wall]
                    proc_ref = [proc]
                    skip_remaining_ref = [skip_remaining]
                    stage = "encode"
                    self._process_video_batch(
                        job, backend, device, payload,
                        start_frame + frame_idx_global, limit,
                        pipe, use_pipe, writer,
                        target_h, target_w,
                        t_wall, last_emit_ref, proc_ref, skip_remaining_ref,
                    )
                    proc = proc_ref[0]
                    skip_remaining = skip_remaining_ref[0]
                    last_emit = last_emit_ref[0]
                    continue
                rgb = payload
                src_idx = start_frame + frame_idx_global
                # Downscale-before on per-frame basis
                rgb_in = _downscale_if_needed(rgb, job.downscale_max_edge)
                # Pre-allocate pinned host buffer for the LR input (Phase 5
                # perf: non_blocking + cached pinned memory). _PinnedPool
                # keys by (h, w, n) so this is a dict lookup after the first frame.
                h_in, w_in = rgb_in.shape[:2]
                pinned_in = _pinned.get_input(h_in, w_in, n=1) if device.type == "cuda" else None
                x = _to_tensor(rgb_in, device, job.fp16, fp16_pin=(job.prefetch == "async"),
                                pinned_in=pinned_in)
                try:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        # Cascade: 2x model applied twice -> 4x; 4x model once.
                        # TTA honors arch Capability (see use_tta in _run_image).
                        use_tta = job.tta and _arch_caps(job.kind).tta
                        n_cascade = (job.cascade_mode if job.cascade_mode is not None
                                     else _cascade_count(backend.model))
                        y = x
                        for _ in range(n_cascade):
                            if use_tta:
                                y = _tta_forward(backend.model, y)
                            else:
                                y = backend(y)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    if job.fp16:
                        y = y.float()
                except Exception as e:
                    # Per-frame failure: ask GUI how to proceed (image jobs don't do this).
                    self.emit(JobEvent(kind="frame_error", job_id=job.job_id,
                                        message=str(e), frame_idx=src_idx, exc=e))
                    if job.on_frame_error:
                        try:
                            action = job.on_frame_error(src_idx, str(e))
                        except Exception:
                            action = SKIP_FRAME
                    else:
                        action = SKIP_FRAME
                    if action == SKIP_FRAME:
                        continue
                    if action == SKIP_REST:
                        skip_remaining = True
                        continue
                    if action == ABORT_JOB:
                        # "Abort job" is a user cancel. Drain queued controls
                        # first (the GUI may re-queue a cancel behind the
                        # modal), then raise the cancellation type so _run_one
                        # emits a clean cancelled and unlinks the partial.
                        self._drain_ctl_for_job(job.job_id)
                        raise _JobCancelled()
                    if action == RETRY_FRAME:
                        continue
                    continue
                # Pre-allocate pinned host buffer for the SR output. The fast path
                # in _tensor_to_bgr does clamp+round+cast-to-uint8 ON the GPU
                # and async-copies to the pinned buffer -- ~10x faster than the
                # default float32->cpu->*255 path for large SR frames.
                out_h_y, out_w_y = int(y.shape[-2]), int(y.shape[-1])
                pinned_out = _pinned.get_output(out_h_y, out_w_y, n=1) if device.type == "cuda" else None
                sr_bgr = _tensor_to_bgr(y, pinned_out=pinned_out)
                sr_bgr = _resize_keep_ar(sr_bgr, target_h, target_w)
                stage = "encode"
                if use_pipe:
                    _encode_write(pipe, sr_bgr)
                else:
                    writer.write(sr_bgr)
                stage = "decode"
                proc += 1
                now = time.perf_counter()
                if now - last_emit > 0.5:
                    elapsed = now - t_wall
                    fps_a = proc / max(elapsed, 1e-6)
                    self.emit(JobEvent(kind="progress", job_id=job.job_id,
                                        progress=(proc / limit) if limit > 0 else -1.0, fps=fps_a,
                                        infer_ms=dt_ms, frame_idx=src_idx))
                    # Q4 (perf/queue-controls-gpu-codec): piggy-back a
                    # gpu_util emission on the existing 2 Hz progress
                    # tick so the GUI can show util/VRAM/temp without
                    # spinning up a second polling thread. The helper
                    # rate-limits itself to 1 Hz internally.
                    _maybe_emit_gpu_util(self.emit, job.job_id, [last_emit])
                    last_emit = now
        except _JobCancelled:
            # Cancel is owned by _run_one: it emits the terminal "cancelled"
            # and unlinks the partial output. Just release the encoder and
            # reader handles on the way out (the finally below does that).
            raise
        except Exception as e:
            # Task 9 follow-up (Task 7 review): _encode_write raises when the
            # encoder process dies mid-stream. Catch it here so the error is
            # terminal (no "finished" can follow) and the truncated output is
            # removed instead of escaping _run_one as a "worker crashed".
            self.emit(JobEvent(kind="error", job_id=job.job_id,
                                message=f"{stage} failed: {e}", exc=e))
            failed = True
        finally:
            # Task 7: capture the encoder exit code. A non-zero rc must become
            # an "error" event with no "finished" and the output discarded --
            # it used to be ignored and the job was reported DONE.
            try:
                if use_pipe:
                    try:
                        pipe.stdin.close()
                    except OSError:
                        pass
                    try:
                        encode_rc = pipe.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        # Encoder wedged (e.g. NVENC session hang): kill it so
                        # the worker thread doesn't block forever.
                        pipe.kill()
                        encode_rc = pipe.wait()
                else:
                    writer.release()
            finally:
                # Task 9 follow-up: always release the reader and drop a
                # truncated output, even when the pipe teardown raises. The
                # old finally skipped the reader release when close() failed.
                try:
                    sync_reader.release()
                finally:
                    if failed:
                        _unlink_partial_output(out_path)

        if failed:
            return

        # Task 7: a non-zero encoder exit (or an empty/unusable output) is a
        # failure, not a finished job -- check before remuxing or reporting.
        if not _handle_encode_result(self, job, out_path, encode_rc, proc):
            return

        # Task 9: a cancel that arrived while the encoder was winding down
        # must not be reported as a finished job -- honour it before the mux.
        self._drain_ctl_for_job(job.job_id)

        wall = time.perf_counter() - t_wall
        avg_fps = proc / max(wall, 1e-6)

        # Keep the sound: remux the source audio over the encoded picture.
        # The encoder was closed by the finally above, so out_path is final,
        # and the window is derived from the frames that were actually encoded
        # (frame-exact start + probed duration), not from the raw cut fields.
        audio_ok = True
        if use_pipe:
            mux_start, mux_duration = _audio_mux_window(
                out_path, fps=fps, start_frame=start_frame, n_frames=proc)
            audio_ok = _mux_job_audio(out_path, job.input_path,
                                      start_seconds=mux_start,
                                      duration_seconds=mux_duration)
        else:
            audio_ok = False  # cv2 fallback cannot mux audio

        # Task 9: likewise a cancel that arrived during the audio remux.
        self._drain_ctl_for_job(job.job_id)

        message = f"-> {out_path.name} ({proc} frames, {avg_fps:.2f} fps)"
        if not audio_ok and has_audio_stream(job.input_path):
            message += " [warning: audio not preserved]"

        self.emit(JobEvent(kind="finished", job_id=job.job_id,
                            message=message,
                            fps=avg_fps, progress=1.0))

        # If a cut window was applied AND ffmpeg is available, extract a cut preview.
        if (job.cut_end_seconds > job.cut_start_seconds > 0
                and use_pipe and ffmpeg_available()):
            cut_path = out_path.with_name(out_path.stem + "_cut.mp4")
            ok = extract_cut(out_path, cut_path,
                             0.0, job.cut_end_seconds - job.cut_start_seconds)
            if ok:
                self.emit(JobEvent(kind="log", job_id=job.job_id,
                                    message=f"cut: {cut_path.name}"))


    # --- recurrent video path (temporal models like AnimeSR) ---
    def _run_video_recurrent(self, job: RunJob, backend, device):
        """Video path for recurrent temporal models (e.g. AnimeSR).

        These models expect 5D input (batch, num_frames, channels, H, W) and
        use prev/cur/nxt frame context internally. The cut window is consumed
        in sliding windows of ``Capability.recurrent_frames`` frames
        (overlapping by one frame, so every output frame keeps its temporal
        neighbours); each window is forwarded and encoded as soon as it is
        produced. VRAM therefore stays bounded by the window size instead of
        growing with the clip length.

        Error model: "error" is terminal -- the event is emitted and the
        partial output file is unlinked, and no "finished" event follows.
        """
        self.emit(JobEvent(kind="started", job_id=job.job_id,
                            message=f"video (recurrent): {job.input_path.name}"))
        self._drain_ctl_for_job(job.job_id)

        # --- decoder setup ---
        if job.decode == "nvdec":
            sync = _NvDecReader(job.input_path)
        elif job.decode == "pyav":
            sync = _PyAvReader(job.input_path)
        elif job.decode == "auto":
            from ..decoders import _detect_nvdec_support, _HAS_PYAV
            if _detect_nvdec_support():
                sync = _NvDecReader(job.input_path)
            elif _HAS_PYAV:
                sync = _PyAvReader(job.input_path)
            else:
                sync = _Cv2Reader(job.input_path)
        else:
            sync = _Cv2Reader(job.input_path)
        fps, total, w, h = sync.fps, sync.total, sync.w, sync.h
        target_w = int(round(w * job.outscale))
        target_h = int(round(h * job.outscale))

        # Cut window
        if fps > 0:
            start_frame = int(round(job.cut_start_seconds * fps))
        else:
            start_frame = 0
        if job.cut_end_seconds > 0 and fps > 0:
            end_frame = int(round(job.cut_end_seconds * fps))
            if total > 0:
                end_frame = min(end_frame, total)
        elif total > 0:
            end_frame = total
        else:
            end_frame = -1
        if end_frame >= 0 and end_frame <= start_frame:
            self.emit(JobEvent(kind="error", job_id=job.job_id,
                                message="cut window is empty (end <= start)"
                                        if end_frame == start_frame
                                        else f"cut range exceeds video (fps={fps:g}, frames={total})"))
            sync.release()
            return
        limit = max(end_frame - start_frame, 0) if end_frame > 0 else 0

        # Re-open decoder and skip to start_frame
        sync.release()
        if job.decode == "nvdec":
            sync_reader = _SkipFirstFrames(_NvDecReader(job.input_path), start_frame)
        elif job.decode == "pyav":
            sync_reader = _SkipFirstFrames(_PyAvReader(job.input_path), start_frame)
        elif job.decode == "auto":
            from ..decoders import _detect_nvdec_support, _HAS_PYAV
            if _detect_nvdec_support():
                sync_reader = _SkipFirstFrames(_NvDecReader(job.input_path), start_frame)
            elif _HAS_PYAV:
                sync_reader = _SkipFirstFrames(_PyAvReader(job.input_path), start_frame)
            else:
                sync_reader = _SkipFirstFrames(_Cv2Reader(job.input_path), start_frame)
        else:
            cap = cv2.VideoCapture(str(job.input_path))
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            sync_reader = _Cv2Reader.from_cap(
                cap, job.input_path, fps, total, w, h)

        # --- encoder setup ---
        out_path = job.output_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # H2: window/stride of the sliding chunker. The window is the arch's
        # Capability.recurrent_frames (3 for AnimeSR) and the stride is one
        # less, so consecutive windows share a frame of temporal context
        # instead of restarting the recurrence on a hard cut. Every forward
        # therefore sees at most `window` frames: a 10 s 4K cut becomes ~80
        # chunks of ~150 MB instead of one 11.9 GB stack.
        window = max(1, int(_arch_caps(job.kind).recurrent_frames or 1))
        step = max(1, window - 1)
        n_cascade = (job.cascade_mode if job.cascade_mode is not None
                     else _cascade_count(backend.model))
        n_cascade = max(1, int(n_cascade))
        scale = getattr(backend.model, "scale",
                        getattr(backend.model, "upscale", 4))

        pipe = None
        writer = None
        use_pipe = False
        # B1: track whether this job actually created the output file so the
        # error path only unlinks what the encoder opened.
        started_writing = False
        failed = False
        encode_rc = 0
        stage = "decode"  # which step an unexpected exception came from
        try:
            pipe = open_encoder(
                out_path, fps, target_w, target_h,
                crf=job.video_crf, preset=job.video_preset,
                use_nvenc=job.use_nvenc, nvenc_preset=job.nvenc_preset, nvenc_qp=job.nvenc_qp,
            )
            use_pipe = pipe is not None
            started_writing = True
            if not use_pipe:
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (target_w, target_h))
                if not writer.isOpened():
                    self.emit(JobEvent(kind="error", job_id=job.job_id, message="no encoder (ffmpeg/cv2)"))
                    failed = True
                    return

            # --- chunked inference: decode a window, forward it, encode it ---
            t_wall = time.perf_counter()
            last_emit = t_wall
            n_frames = 0
            preflighted = False
            for chunk, skip in _recurrent_windows(sync_reader, window, step, limit):
                # H1: pause / cancel at chunk granularity -- the same
                # contract _run_video honors between frames.
                self._drain_ctl_for_job(job.job_id)
                self._check_pause(job.job_id)
                if not preflighted:
                    preflighted = True
                    f_h, f_w = chunk[0].shape[:2]
                    self._recurrent_vram_preflight(job, device, len(chunk), f_h, f_w, scale)
                stage = "recurrent inference"
                sr_bgrs, dt_ms = self._recurrent_chunk_forward(
                    job, backend, device, chunk, n_cascade, target_h, target_w)
                stage = "encode"
                # The sliding window re-yields the previous chunk's last
                # frame as its first frame; `skip` drops it so every output
                # frame is written exactly once, in order.
                new_bgrs = sr_bgrs[skip:]
                if limit > 0:
                    new_bgrs = new_bgrs[:max(0, limit - n_frames)]
                for sr_bgr in new_bgrs:
                    if use_pipe:
                        _encode_write(pipe, sr_bgr)
                    else:
                        writer.write(sr_bgr)
                n_frames += len(new_bgrs)
                now = time.perf_counter()
                if new_bgrs:
                    # H1: same payload keys _run_video emits so the status
                    # bar advances instead of freezing on a silent job.
                    self.emit(JobEvent(
                        kind="progress", job_id=job.job_id,
                        progress=(n_frames / limit) if limit > 0 else -1.0,
                        fps=n_frames / max(now - t_wall, 1e-6),
                        infer_ms=dt_ms,
                        frame_idx=start_frame + n_frames - 1))
                    # Q4: piggy-back a gpu_util emission on the progress tick.
                    _maybe_emit_gpu_util(self.emit, job.job_id, [last_emit])
                    last_emit = now
                stage = "decode"  # the next window pull is a decode step
            if n_frames == 0:
                self.emit(JobEvent(kind="error", job_id=job.job_id,
                                    message="no frames decoded"))
                failed = True
                return

            wall = time.perf_counter() - t_wall
            avg_fps = n_frames / max(wall, 1e-6)
        except _JobCancelled:
            # Cancel is owned by _run_one: it closes nothing here (the
            # finally below already released the encoder), unlinks the
            # partial output and emits the terminal "cancelled" event.
            raise
        except Exception as e:
            # B1: "error" is terminal. Emit it and do NOT fall through to
            # the "finished" emit -- a failed job used to be re-labelled
            # DONE by the GUI and left its truncated output on disk.
            self.emit(JobEvent(kind="error", job_id=job.job_id,
                               message=f"{stage} failed: {e}", exc=e))
            failed = True
        finally:
            # Happy-path cleanup (pipe close + 30s wait + kill,
            # writer.release, sync_reader.release). It also runs on the
            # error / cancel path so the encoder handle is gone before the
            # partial output file is unlinked (Windows keeps it locked).
            if writer is not None:
                writer.release()
            elif pipe is not None:
                try:
                    pipe.stdin.close()
                except OSError:
                    pass
                try:
                    encode_rc = pipe.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    # Encoder wedged (e.g. NVENC session hang): kill it so
                    # the worker thread doesn't block forever.
                    pipe.kill()
                    encode_rc = pipe.wait()
            sync_reader.release()
            if failed and started_writing:
                _unlink_partial_output(out_path)

        # The encoder is closed by the finally above, so out_path is final.
        # Remux the source audio before announcing success; a failed mux on an
        # audio-bearing source is surfaced in the message, never silent. The
        # window mirrors the encoded frames, never the raw cut fields (End=0
        # means "to EOF", so Start-only cuts are positive windows).
        if not failed:
            # Task 7: a non-zero encoder exit (or an empty/unusable output) is
            # a failure, not a finished job -- check before remuxing or
            # reporting; the output file is discarded by the helper.
            if not _handle_encode_result(self, job, out_path, encode_rc, n_frames):
                return
            # Task 9: do not report finished when a cancel is already pending.
            self._drain_ctl_for_job(job.job_id)
            audio_ok = True
            if use_pipe:
                mux_start, mux_duration = _audio_mux_window(
                    out_path, fps=fps, start_frame=start_frame, n_frames=n_frames)
                audio_ok = _mux_job_audio(out_path, job.input_path,
                                          start_seconds=mux_start,
                                          duration_seconds=mux_duration)
            else:
                audio_ok = False  # cv2 fallback cannot mux audio

            # Task 9: likewise a cancel that arrived during the remux.
            self._drain_ctl_for_job(job.job_id)

            message = f"-> {out_path.name} ({n_frames} frames, {avg_fps:.2f} fps)"
            if not audio_ok and has_audio_stream(job.input_path):
                message += " [warning: audio not preserved]"

            self.emit(JobEvent(kind="finished", job_id=job.job_id,
                                message=message, fps=avg_fps, progress=1.0))

    def _recurrent_chunk_forward(self, job, backend, device, frames_rgb,
                                 n_cascade: int, target_h: int, target_w: int):
        """Run one recurrent window through the backend.

        ``frames_rgb`` holds at most ``Capability.recurrent_frames`` frames,
        so the peak VRAM of the forward is bounded by the window size
        instead of the cut length. Returns ``(list of BGR uint8 frames,
        infer_ms)`` -- one entry per input frame, in order.
        """
        n = len(frames_rgb)
        rgbs_in = [_downscale_if_needed(r, job.downscale_max_edge) for r in frames_rgb]
        h_in, w_in = rgbs_in[0].shape[:2]
        pinned_in = _pinned.get_input(h_in, w_in, n=n) if device.type == "cuda" else None
        x = _to_tensor_batch(rgbs_in, device, job.fp16, pinned_in=pinned_in)
        # (n, 3, H, W) -> (1, n, 3, H, W): the recurrent contract.
        x = x.unsqueeze(0)

        # Pad to the arch's pad_multiple (animesr needs H, W % 4 == 0).
        pm = _arch_caps(job.kind).pad_multiple
        if pm > 0:
            ph = (pm - h_in % pm) % pm
            pw = (pm - w_in % pm) % pm
            if ph > 0 or pw > 0:
                x = F.pad(x, (0, pw, 0, ph), mode="reflect")

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            # Cascade: 2x model applied twice -> 4x; 4x model once.
            y = x
            for _ in range(n_cascade):
                y = backend(y)
            if job.fp16:
                y = y.float()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0

        # (1, n, 3, H*scale, W*scale) -> (n, 3, H*scale, W*scale)
        y = y.squeeze(0)
        out_h, out_w = int(y.shape[-2]), int(y.shape[-1])
        pinned_out = _pinned.get_output(out_h, out_w, n=n) if device.type == "cuda" else None
        sr_bgrs = _tensor_to_bgr_batch(y, pinned_out=pinned_out,
                                       target_h=target_h, target_w=target_w)
        return sr_bgrs, dt_ms

    def _recurrent_vram_preflight(self, job, device, n_frames: int,
                                  height: int, width: int, scale) -> None:
        """Warn when the first recurrent chunk is unlikely to fit in VRAM.

        Chunking already bounds the *peak* by the window size; this cheap
        pre-flight estimate turns the remaining risk into a visible log
        line so the user can shorten the cut or lower downscale_max_edge.
        vram.estimate() has no frame-count axis, so the recurrent path
        keeps its own local copy of that conservative math.
        """
        if device.type != "cuda" or job.gpu_guard_mode == "off":
            return
        try:
            free, _total = torch.cuda.mem_get_info()
        except Exception:
            return
        if free <= 0:
            return
        est = _recurrent_vram_estimate(n_frames, height, width, job.fp16, scale)
        if est > 0.6 * free:
            self.emit(JobEvent(
                kind="log", job_id=job.job_id,
                message=f"VRAM pre-flight: a {n_frames}-frame chunk at "
                        f"{height}x{width} needs ~{est / 1e9:.2f} GB of "
                        f"{free / 1e9:.2f} GB free VRAM; shorten the cut "
                        f"window or lower downscale_max_edge to stay in budget"))

    # --- batched video path (Phase 1.C) ---
    def _process_video_batch(
        self, job, backend, device, frames_rgb: "list",
        src_idx_start: int, limit: int,
        pipe, use_pipe: bool, writer,
        target_h: int, target_w: int,
        t_wall: float,
        last_emit_ref: "list",
        proc_ref: "list",
        skip_remaining_ref: "list",
    ):
        """Process N frames in one batched forward pass.

        Updates proc and skip_remaining in the supplied single-element lists
        (Python integers + bools are immutable; the ref-list pattern keeps
        the outer loop in sync without restructuring the whole _run_video).

        Error model: a single bad frame in the batch fails the whole batch;
        we emit frame_error with src_idx_start and treat as SKIP_REST (safe
        default -- could be enhanced with per-frame try/except inside the
        batch later, but for v1 of batching the simpler path is better).
        """
        n = len(frames_rgb)
        # Downscale-before per frame; assert all share the same shape so we
        # can build a single batched tensor.
        rgbs_in = [_downscale_if_needed(r, job.downscale_max_edge) for r in frames_rgb]
        h_in, w_in = rgbs_in[0].shape[:2]
        for r in rgbs_in:
            if r.shape[:2] != (h_in, w_in):
                # Shape mismatch (e.g. different resolution after downscale
                # unlikely but defensive). Fall back to per-frame processing.
                self._process_video_batch_fallback(
                    job, backend, device, rgbs_in,
                    src_idx_start, limit,
                    pipe, use_pipe, writer,
                    target_h, target_w,
                    t_wall, last_emit_ref, proc_ref, skip_remaining_ref,
                )
                return
        # Build batched input
        pinned_in = _pinned.get_input(h_in, w_in, n=n) if device.type == "cuda" else None
        try:
            x = _to_tensor_batch(rgbs_in, device, job.fp16, pinned_in=pinned_in)
        except Exception as e:
            self.emit(JobEvent(kind="frame_error", job_id=job.job_id,
                                message=f"batch stack failed: {e}",
                                frame_idx=src_idx_start, exc=e))
            skip_remaining_ref[0] = True
            return
        # Inference
        try:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                # TTA already excluded by use_batch gate. Cascade: 2x model
                # applied twice -> 4x; 4x model once (single shot).
                n_cascade = (job.cascade_mode if job.cascade_mode is not None
                             else _cascade_count(backend.model))
                y = x
                for _ in range(n_cascade):
                    y = backend(y)  # grows by `scale` per iteration
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if job.fp16:
                y = y.float()
        except Exception as e:
            self.emit(JobEvent(kind="frame_error", job_id=job.job_id,
                                message=str(e), frame_idx=src_idx_start, exc=e))
            if job.on_frame_error:
                try:
                    action = job.on_frame_error(src_idx_start, str(e))
                except Exception:
                    action = SKIP_REST
            else:
                action = SKIP_REST
            if action == ABORT_JOB:
                raise e
            skip_remaining_ref[0] = True
            return
        # Decode outputs and write each frame
        out_h_y, out_w_y = int(y.shape[-2]), int(y.shape[-1])
        pinned_out = _pinned.get_output(out_h_y, out_w_y, n=n) if device.type == "cuda" else None
        sr_bgrs = _tensor_to_bgr_batch(y, pinned_out=pinned_out,
                                        target_h=target_h, target_w=target_w)
        for i, sr_bgr in enumerate(sr_bgrs):
            if use_pipe:
                _encode_write(pipe, sr_bgr)
            else:
                writer.write(sr_bgr)
            proc_ref[0] += 1
            now = time.perf_counter()
            if now - last_emit_ref[0] > 0.5:
                elapsed = now - t_wall
                fps_a = proc_ref[0] / max(elapsed, 1e-6)
                self.emit(JobEvent(kind="progress", job_id=job.job_id,
                                    progress=(proc_ref[0] / limit) if limit > 0 else -1.0, fps=fps_a,
                                    infer_ms=dt_ms, frame_idx=src_idx_start + i))
                # Q4 (perf/queue-controls-gpu-codec): see site a; the
                # ref-list pattern is the same here -- last_emit_ref[0]
                # is mutated by the helper to remember the last gpu_util
                # emission time, which is independent from the 2 Hz
                # progress tick so a long pause won't suppress util.
                _maybe_emit_gpu_util(self.emit, job.job_id, last_emit_ref)
                last_emit_ref[0] = now

    def _process_video_batch_fallback(
        self, job, backend, device, rgbs_in, src_idx_start, limit,
        pipe, use_pipe, writer, target_h, target_w, t_wall,
        last_emit_ref, proc_ref, skip_remaining_ref,
    ):
        """Batched path fell back because of shape mismatch: process each
        frame individually using the batched backend (one frame each)."""
        # Simple fallback: run per-frame via _to_tensor + _tensor_to_bgr
        for i, rgb_in in enumerate(rgbs_in):
            h, w = rgb_in.shape[:2]
            pinned_in = _pinned.get_input(h, w, n=1) if device.type == "cuda" else None
            x = _to_tensor(rgb_in, device, job.fp16, fp16_pin=False, pinned_in=pinned_in)
            try:
                with torch.no_grad():
                    # Cascade: 2x model applied twice -> 4x; 4x model once.
                    n_cascade = (job.cascade_mode if job.cascade_mode is not None
                                 else _cascade_count(backend.model))
                    y = x
                    for _ in range(n_cascade):
                        y = backend(y)
                if job.fp16:
                    y = y.float()
            except Exception as e:
                self.emit(JobEvent(kind="frame_error", job_id=job.job_id,
                                    message=str(e),
                                    frame_idx=src_idx_start + i, exc=e))
                continue
            out_h_y, out_w_y = int(y.shape[-2]), int(y.shape[-1])
            pinned_out = _pinned.get_output(out_h_y, out_w_y, n=1) if device.type == "cuda" else None
            sr_bgr = _tensor_to_bgr(y, pinned_out=pinned_out)
            sr_bgr = _resize_keep_ar(sr_bgr, target_h, target_w)
            if use_pipe:
                _encode_write(pipe, sr_bgr)
            else:
                writer.write(sr_bgr)
            proc_ref[0] += 1

# ============================================================================ #
# Batching + helper
# ============================================================================ #
def _recurrent_windows(frames, window: int, step: int, limit: int = 0):
    """Yield ``(window_frames, skip)`` sliding windows over a frame stream.

    ``window`` is ``Capability.recurrent_frames`` (3 for AnimeSR) and
    ``step`` its stride (``window - 1``), so consecutive windows share one
    frame of temporal context. Each yielded list holds at most ``window``
    frames, which bounds host RAM (only two windows are ever resident) and
    VRAM (the model never holds the whole cut window at once) no matter how
    long the cut is.

    ``skip`` is how many leading frames of the window were already emitted
    by the previous window (the shared context frame); the caller drops
    them so every output frame is produced exactly once. ``limit`` caps the
    number of frames the caller is still allowed to emit; 0 means "run to
    EOF".
    """
    window = max(1, int(window))
    step = max(1, min(int(step), window))
    buf: List[Any] = []
    win_idx = 0      # window counter; window k starts at k * step
    produced = 0     # frames whose output has already been handed out
    for frame in frames:
        if limit > 0 and produced >= limit:
            break
        buf.append(frame)
        if len(buf) < window:
            continue
        start = win_idx * step
        skip = min(max(produced - start, 0), len(buf))
        yield buf, skip
        produced = start + window
        win_idx += 1
        buf = buf[step:]  # keep only the context frame the next window needs
    if buf and (limit <= 0 or produced < limit):
        # Trailing partial window (clip shorter than one window, or the
        # last frames that never filled one).
        start = win_idx * step
        skip = min(max(produced - start, 0), len(buf))
        if skip < len(buf):
            yield buf, skip


# H2: VRAM pre-flight for the recurrent path. This mirrors the conservative
# math in anime_upscaler_gui/vram.py (output tensor + 4x activations + ~50 MB
# model) multiplied by the number of frames the window holds resident at
# once. vram.estimate() has no frame-count axis, so the recurrent path keeps
# its own local copy instead of changing that shared helper's signature.
_RECURRENT_VRAM_MODEL_BYTES = 50 * 1024 * 1024
_RECURRENT_VRAM_ACT_FACTOR = 4


def _recurrent_vram_estimate(n_frames: int, height: int, width: int,
                             fp16: bool, scale: int = 4) -> int:
    """Cheap upper-bound VRAM estimate (bytes) for one recurrent chunk."""
    db = 2 if fp16 else 4
    try:
        scale = int(scale)
    except (TypeError, ValueError):
        scale = 4
    out_h = max(1, height * scale)
    out_w = max(1, width * scale)
    output_bytes = out_h * out_w * 3 * db * max(1, int(n_frames))
    return (output_bytes
            + output_bytes * _RECURRENT_VRAM_ACT_FACTOR
            + _RECURRENT_VRAM_MODEL_BYTES)


def _batched(it, batch_size: int, limit: int):
    """Yield (start_idx, frame) for batch_size=1 mode; returns single frames."""
    batch = []
    consumed = 0
    for frame in it:
        batch.append(frame)
        consumed += 1
        if len(batch) == batch_size:
            yield consumed - batch_size, batch[0] if batch_size == 1 else batch
            batch = []
        if limit and consumed >= limit:
            break
    if batch:
        yield consumed - len(batch), batch[0] if batch_size == 1 else batch


def _find_ckpt(job: RunJob) -> Path:
    """The pipeline doesn't know where pretrained/ lives; the GUI passes the absolute
    path as model_filename via a side channel. We reconstruct it here by stashing
    the directory on the first call. For MVP we look next to the package + the
    settings-declared pretrained_dir."""
    # The GUI is expected to write the absolute path into job.model_filename.
    return Path(job.model_filename)
