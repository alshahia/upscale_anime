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
from ..archs import capabilities as _arch_caps
from ..decoders import (_AsyncReader, _Cv2Reader, _NvDecReader, _PyAvReader, _SkipFirstFrames,
                        load_image_rgb, save_image_rgb)
from ..ffmpeg import extract_cut, ffmpeg_available, open_encoder

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
        try:
            if job.is_video:
                self._run_video(job)
            else:
                self._run_image(job)
        except _JobCancelled:
            # Clean up partial output. The encoder pipe is already closed
            # because _run_video/_run_image's `finally` block ran on the
            # way out. We just delete the file and emit a terminal event.
            out = job.output_path
            try:
                if out.exists():
                    out.unlink()
            except OSError:
                pass
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
        try:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                # Cascade: 2x model applied twice -> 4x; 4x model once.
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
                # Normalize payload: single frame (np.ndarray H,W,3) for
                # batch_size=1; list of N frames for batch_size>1.
                if use_batch and isinstance(payload, list):
                    # Single-element list refs (mutable scalars via list wrapping).
                    # Pre-bind them so they are accessible after the call --
                    # Python does not bind a local for `proc_ref=[proc]` kwargs.
                    last_emit_ref = [t_wall]
                    proc_ref = [proc]
                    skip_remaining_ref = [skip_remaining]
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
                        # A user cancel that arrived while the frame-error
                        # modal was up is re-queued by the GUI's wait loop;
                        # drain here so the job unwinds as a clean cancel
                        # instead of a worker crash.
                        self._drain_ctl_for_job(job.job_id)
                        raise e
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
                if use_pipe:
                    _encode_write(pipe, sr_bgr)
                else:
                    writer.write(sr_bgr)
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
        finally:
            if use_pipe:
                pipe.stdin.close()
                try:
                    pipe.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    # Encoder wedged (e.g. NVENC session hang): kill it so
                    # the worker thread doesn't block forever.
                    pipe.kill()
                    pipe.wait()
            else:
                writer.release()
            sync_reader.release()

        wall = time.perf_counter() - t_wall
        avg_fps = proc / max(wall, 1e-6)
        self.emit(JobEvent(kind="finished", job_id=job.job_id,
                            message=f"-> {out_path.name} ({proc} frames, {avg_fps:.2f} fps)",
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
