"""Inference pipeline: job dataclass, PyTorch + ONNX backends, worker thread.

Single module owns:
  - RunJob            : one input file with all settings needed to process it
  - JobEvent          : progress / error / frame-error event for the GUI queue
  - _ResumeEvent      : action to take when a video frame fails (skip/retry/abort)
  - _PyTorchBackend   : wraps a built model in a no_grad callable
  - _OnnxBackend      : optional onnxruntime backend (auto-export if .onnx missing)
  - PipelineWorker    : threading.Thread that processes jobs from a queue
  - _to_tensor / _tensor_to_bgr : shared pre/post helpers

Public API: RunJob, JobEvent, PipelineWorker (and SKIP_FRAME/SKIP_REST/
ABORT_JOB/RETRY_FRAME constants). The previous private underscore names
(_RunJob, _JobEvent, _PipelineWorker) are kept as deprecated aliases
for backwards compatibility with existing scripts.
"""
import logging
import os
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .archs import build, _save_sr
from .archs import capabilities as _arch_caps
from .decoders import (_AsyncReader, _Cv2Reader, _PyAvReader, _SkipFirstFrames,
                       load_image_rgb, save_image_rgb)
from .ffmpeg import open_encoder, extract_cut, ffmpeg_available
log = logging.getLogger(__name__)

# Phase 4: D4 TTA. Lazy import so the pipeline module is usable even when
# TTA isn't installed for some reason (it ships in the same package as the rest).
def _tta_forward(model, x):
    from .tta import tta_forward as _fwd
    return _fwd(model, x)

try:
    import onnx
    import onnxruntime as ort
    _HAS_ORT = True
except Exception:
    _HAS_ORT = False

try:
    from .trt_engine import _TrtBackend as _TrtBackend_t
    from .trt_engine import _HAS_TRT as _HAS_TRT
except Exception:
    _TrtBackend_t = None
    _HAS_TRT = False


# ============================================================================ #
# Cascade mode (Phase 2.A)
# ============================================================================ #
# A 2x student (`model.scale == 2`) applied twice yields the same final
# resolution as a single 4x model but with two smaller/cheaper model passes.
# This is the cascade 2x + 2x path: total upscale = 4x, but inference is
# roughly 1.5-2x faster end-to-end than a single 4x model at 4K input.
# The 2x and 4x code paths stay uniform: `_cascade_count()` decides how many
# times to invoke the backend, given the model that's loaded. Same `kind`
# field ('rfdn_student') -- the only thing that varies is the weights.
def _cascade_count(model) -> int:
    """How many times to invoke `model()` to reach a 4x output.

    * 4x model -> 1 call (current behavior).
    * 2x model -> 2 calls (cascade 2x+2x).
    * Other scales are passed through unchanged (best-effort).
    """
    s = getattr(model, "scale", getattr(model, "upscale", 4))
    try:
        s = int(s)
    except (TypeError, ValueError):
        s = 4
    if s >= 4:
        return 1
    # 4 / 2 = 2 cascades; 4 / 3 ~= 1 (3x model single-shot).
    n = max(1, round(4 / s))
    return int(n)


# ============================================================================ #
# Job + event dataclasses
# ============================================================================ #
@dataclass
class RunJob:
    """One unit of work. Created by the GUI when 'Start' is pressed.

    Public alias for the previous private name ``RunJob``. Existing
    scripts that imported ``RunJob`` keep working; that name is kept
    as a deprecated alias (see ``__init__.py`` for the warning policy).
    """
    job_id: int
    input_path: Path
    output_path: Path
    is_video: bool
    model_filename: str
    kind: str
    scale: int
    outscale: float
    fp16: bool
    device: str
    batch_size: int
    decode: str
    prefetch: str
    pin_memory: str
    downscale_max_edge: int
    gpu_guard_mode: str
    tile_size: int
    tile_overlap: int
    cut_start_seconds: float = 0.0
    cut_end_seconds: float = 0.0
    video_crf: int = 18
    video_preset: str = "medium"
    # Phase 5 ship: D4 TTA (default ON for the recommended model
    # rfdn_distill_v1_4x + TTA = 29.46 dB test PSNR). When True,
    # _run_image/_run_video route through tta_forward(backend.model, x)
    # instead of backend(x). 8x cost (square); 6x for rect. Disable in the
    # GUI Settings panel if peak throughput matters more than quality.
    tta: bool = True
    # Optional callback for video mid-frame failures: signature () -> str (resume action).
    # The GUI supplies a callable that blocks until the user picks one.
    on_frame_error: Optional[Callable[[int, str], str]] = None
    # Phase 1 (Real-time 4K): TensorRT engine. Default ON; auto-disabled
    # by the backend selector when TensorRT isn't importable, on CPU, when
    # fp16 is False, when TTA is requested (TTA bypasses the backend and
    # needs backend.model), or for the 'animesr' kind (3-frame-center trick).
    use_tensorrt: bool = True
    # Phase 1 (Real-time 4K): NVIDIA hardware encoder (NVENC). Default ON;
    # auto-fallback to libx264 in open_encoder() if h264_nvenc is not
    # available on this system.
    use_nvenc: bool = True
    nvenc_preset: str = "p1"   # p1 = fastest, p4 = balanced, p7 = best (we expose p1..p4)
    nvenc_qp: int = 18         # constant-QP target (lower = better quality)
    # Phase 2 (Real-time 4K): Cascade mode. None means "auto from model scale":
    # 2x model -> cascade 2x2x (apply twice); 4x model -> single shot. Set to
    # an explicit integer to force a specific cascade depth.
    cascade_mode: Optional[int] = None


@dataclass
class JobEvent:
    """Worker -> GUI signal.

    Public alias for the previous private name ``JobEvent``. Existing
    scripts that imported ``JobEvent`` keep working; that name is kept
    as a deprecated alias.
    """
    kind: str  # "progress" | "started" | "finished" | "error" | "frame_error" | "log" | "fatal_error"
    job_id: int = 0
    message: str = ""
    progress: float = 0.0      # 0..1
    fps: float = 0.0
    infer_ms: float = 0.0
    frame_idx: int = -1
    exc: Optional[BaseException] = None


# Resume actions for video mid-frame errors
SKIP_FRAME = "skip_frame"
SKIP_REST = "skip_rest"
ABORT_JOB = "abort_job"
RETRY_FRAME = "retry_frame"


# ============================================================================ #
# Tensor helpers
# ============================================================================ #
# ============================================================================ #
# Pinned-memory pool + cudnn autotune
# ============================================================================ #
# Per-shape pinned host buffers, reused across frames. Allocating pinned memory
# is expensive; caching by (h, w) means we pay it once per unique resolution.
# Combined with the GPU-side uint8 cast below this yields ~10x faster GPU->CPU
# transfers for large SR outputs (e.g. 3416x1920 fp16 tensor -> 19.6 MB uint8
# instead of 78 MB float32). See docs/rfdn_realtime_report.md for measurements.
class _PinnedPool:
    """Module-level cache of pinned host tensors, keyed by (H, W, N).

    Where N is the batch size:
      * N=1 -> (3, H, W) input / (H, W, 3) output (single-frame path)
      * N>1 -> (N, 3, H, W) input / (N, H, W, 3) output (batched path)
    Allocating pinned memory is expensive; caching by (h, w, n) means we
    pay it once per unique resolution.
    """

    def __init__(self):
        self._inputs: dict = {}    # (H, W, N) -> (N, 3, H, W) float32 pinned
        self._outputs: dict = {}   # (H, W, N) -> (N, H, W, 3) uint8 pinned

    def get_input(self, h: int, w: int, n: int = 1) -> torch.Tensor:
        key = (h, w, n)
        t = self._inputs.get(key)
        if t is None:
            t = torch.empty(n, 3, h, w, dtype=torch.float32, pin_memory=True)
            self._inputs[key] = t
        return t

    def get_output(self, h: int, w: int, n: int = 1) -> torch.Tensor:
        key = (h, w, n)
        t = self._outputs.get(key)
        if t is None:
            t = torch.empty(n, h, w, 3, dtype=torch.uint8, pin_memory=True)
            self._outputs[key] = t
        return t


# Let cuDNN autotune conv kernels -- helps RFDN/ERANet/SRVGG noticeably on
# fixed-shape video frames. Cheap to enable; benchmark cache survives the
# worker lifetime.
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass


_pinned = _PinnedPool()


def _to_tensor(rgb_uint8: "np.ndarray", device, half: bool, fp16_pin: bool = False,
               pinned_in: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Single-frame: (H, W, 3) uint8 -> (1, 3, H, W) float [0,1] on device.

    Fast path: when pinned_in is supplied AND we are on CUDA, copy the
    normalized HWC->CHW float32 array into pinned_in[0] (the batched pool
    returns (N, 3, H, W) buffers; this writes into slot 0) and transfer
    asynchronously (non_blocking=True). Skips a per-frame allocation +
    pageable->pinned staging copy.

    Slow path (original): allocate a fresh numpy float32 array and let
    PyTorch handle the transfer. Used when no pinned buffer is supplied or
    on CPU.
    """
    arr = np.ascontiguousarray(rgb_uint8.transpose(2, 0, 1)).astype(np.float32) / 255.0
    if pinned_in is not None and device.type == "cuda":
        # Pool always returns 4D (n, 3, H, W); for single-frame we use slot 0.
        if pinned_in.ndim == 4:
            np.copyto(pinned_in[0].numpy(), arr)
            t = pinned_in[:1].to(device, non_blocking=True)
        else:
            # Backwards compat: legacy 3D pinned_in
            np.copyto(pinned_in.numpy(), arr)
            t = pinned_in.unsqueeze(0).to(device, non_blocking=True)
        if half:
            t = t.half()
        return t
    t = torch.from_numpy(arr).unsqueeze(0)
    if device.type == "cuda" and fp16_pin:
        t = t.pin_memory().to(device, non_blocking=True)
    else:
        t = t.to(device)
    if half:
        t = t.half()
    return t


def _to_tensor_batch(rgbs_uint8: "list", device, half: bool,
                     pinned_in: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Batched: list of N (H, W, 3) uint8 -> (N, 3, H, W) float [0,1] on device.

    pinned_in is (N, 3, H, W); we copy each frame into its slot and
    async-copy the whole batch in one shot. Skips per-frame transfer
    overhead. Falls back to per-frame _to_tensor if pinned_in is None.
    """
    n = len(rgbs_uint8)
    if pinned_in is not None and device.type == "cuda":
        for i, rgb in enumerate(rgbs_uint8):
            arr = np.ascontiguousarray(rgb.transpose(2, 0, 1)).astype(np.float32) / 255.0
            np.copyto(pinned_in[i].numpy(), arr)
        t = pinned_in.to(device, non_blocking=True)
        if half:
            t = t.half()
        return t
    # CPU / slow path: stack fresh tensors.
    arrs = [np.ascontiguousarray(r.transpose(2, 0, 1)).astype(np.float32) / 255.0 for r in rgbs_uint8]
    t = torch.from_numpy(np.stack(arrs, axis=0))
    if device.type == "cuda" and fp16_pin:
        t = t.pin_memory().to(device, non_blocking=True)
    else:
        t = t.to(device)
    if half:
        t = t.half()
    return t


def _tensor_to_bgr(y: torch.Tensor, pinned_out: Optional[torch.Tensor] = None) -> "np.ndarray":
    """Single-frame: (1, 3, H, W) float [0,1] -> (H, W, 3) uint8 BGR.

    Fast path: when pinned_out is supplied AND y is on CUDA, do the
    clamp+round+cast-to-uint8 on the GPU (4x smaller transfer) and stream
    into pinned_out[0] (the batched pool returns (N, H, W, 3) buffers;
    this writes slot 0) async. Skips the slow default float32->cpu->*255
    path (~46 ms on a 3416x1920 tensor).

    Slow path (original): CPU-side clamp / .cpu() / *255 / astype.
    """
    if pinned_out is not None and y.is_cuda:
        y_u8 = (y.float().clamp(0, 1) * 255).round().to(torch.uint8)
        y_u8 = y_u8.squeeze(0).permute(1, 2, 0).contiguous()
        if pinned_out.ndim == 4:
            # (N, H, W, 3) -- write to slot 0
            pinned_out[0].copy_(y_u8, non_blocking=True)
            torch.cuda.synchronize()
            arr_o = pinned_out[0].numpy()
        else:
            pinned_out.copy_(y_u8, non_blocking=True)
            torch.cuda.synchronize()
            arr_o = pinned_out.numpy()
    else:
        y_cpu = y.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        arr_o = (y_cpu * 255).astype("uint8")
    return cv2.cvtColor(arr_o, cv2.COLOR_RGB2BGR)


def _tensor_to_bgr_batch(y: torch.Tensor, pinned_out: Optional[torch.Tensor] = None,
                          target_h: Optional[int] = None,
                          target_w: Optional[int] = None) -> "list":
    """Batched: (N, 3, H, W) float [0,1] -> list of N (H, W, 3) uint8 BGR.

    Fast path: when pinned_out is supplied AND y is on CUDA, cast to uint8
    on the GPU, async-copy the whole batch into pinned_out (N, H, W, 3)
    once, then numpy-views each slot. Avoids per-frame copies.

    Optional target_h/target_w: if both supplied, resize each output before
    converting to BGR (used to match a user-requested outscale != native).
    """
    n = y.shape[0]
    if pinned_out is not None and y.is_cuda:
        y_u8 = (y.float().clamp(0, 1) * 255).round().to(torch.uint8)
        # (N, 3, H, W) -> (N, H, W, 3)
        y_u8 = y_u8.permute(0, 2, 3, 1).contiguous()
        pinned_out.copy_(y_u8, non_blocking=True)
        torch.cuda.synchronize()
        out = []
        for i in range(n):
            arr_o = pinned_out[i].numpy()
            if target_h is not None and target_w is not None and arr_o.shape[0] != target_h:
                arr_o = cv2.resize(arr_o, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            out.append(cv2.cvtColor(arr_o, cv2.COLOR_RGB2BGR))
        return out
    # CPU / slow path
    y_cpu = y.clamp(0, 1).permute(0, 2, 3, 1).cpu().float().numpy()
    out = []
    for i in range(n):
        a = (y_cpu[i] * 255).astype("uint8")
        if target_h is not None and target_w is not None and a.shape[0] != target_h:
            a = cv2.resize(a, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        out.append(cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
    return out


def _resize_keep_ar(bgr, target_h: int, target_w: int):
    h, w = bgr.shape[:2]
    if (h, w) == (target_h, target_w):
        return bgr
    return cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)


def _downscale_if_needed(rgb: "np.ndarray", max_edge: int) -> "np.ndarray":
    """Cap the longer edge at max_edge pixels; return unchanged if smaller."""
    if max_edge <= 0:
        return rgb
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return rgb
    scale = max_edge / longest
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ============================================================================ #
# Backends
# ============================================================================ #
def _make_backend(model, ckpt_path, kind, device, fp16, use_tensorrt, tta=False,
                     batch_size: int = 1):
    """Pick TRT or PyTorch backend based on settings + environment.

    batch_size: required for TRT (engine is built for this exact batch size).
    Returns (backend, backend_name). Falls back to PyTorch when:
        * TensorRT not importable
        * torch is on CPU
        * fp16 is False (TRT engine is fp16-only here)
        * kind == 'animesr' (TRT doesn't model the 3-frame-center trick)
        * tta=True (TTA bypasses the backend and needs backend.model)
        * TRT backend construction fails for any reason
    """
    if (not tta) and use_tensorrt and _HAS_TRT and device.type == "cuda" and fp16 and _arch_caps(kind).tensorrt:
        try:
            cache_dir = Path(os.environ.get("APPDATA", str(Path.home()))) / "anime_upscaler_gui" / "cache" / "trt"
            cache_dir.mkdir(parents=True, exist_ok=True)
            backend = _TrtBackend_t(model, kind, device, fp16, cache_dir, batch_size=batch_size)
            log.info("backend: TensorRT (cache=%s, batch=%d)", cache_dir, batch_size)
            return backend, "TensorRT"
        except Exception as e:
            log.warning("trt backend init failed, falling back to PyTorch: %s", e)
    backend = _PyTorchBackend(ckpt_path, kind, device, fp16)
    return backend, "PyTorch"


# ============================================================================ #
class _PyTorchBackend:
    """Wraps a built model in a no-grad callable."""

    def __init__(self, ckpt_path, kind: str, device, fp16: bool):
        self.model = build(kind, ckpt_path).to(device).eval()
        if fp16:
            self.model = self.model.half()
        self.kind = kind
        self.device = device
        self.fp16 = fp16

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            n_recurrent = _arch_caps(self.kind).recurrent_frames
            if n_recurrent:
                # x is (B, 3, H, W); a recurrent model wants (B, N, 3, H, W);
                # the frame is repeated N times, output's center frame taken.
                x = x.unsqueeze(1).expand(-1, n_recurrent, -1, -1, -1).contiguous()
                y = self.model(x)
                y = y[:, y.shape[1] // 2]
            else:
                y = self.model(x)
        return y


class _OnnxBackend:
    """Optional onnxruntime-gpu backend; auto-exports the .onnx if missing."""

    def __init__(self, onnx_path: Path, kind: str, device, fp16: bool, model=None):
        if not _HAS_ORT:
            raise RuntimeError("onnxruntime not installed")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)
        self.kind = kind
        self.fp16 = fp16
        self.np_dtype = np.float16 if fp16 else np.float32
        self.device = device
        # Phase 2 (Real-time 4K): the cascade helper inspects backend.model.scale;
        # stash it here so ONNX behaves the same as PyTorch / TRT.
        self.model = model

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            inp = x.detach().cpu().numpy().astype(self.np_dtype)
        n_recurrent = _arch_caps(self.kind).recurrent_frames
        if n_recurrent:
            inp = np.repeat(inp[:, None], n_recurrent, axis=1)
        out = self.sess.run(None, {"input": inp})[0]
        if n_recurrent:
            out = out[:, out.shape[1] // 2]
        return torch.from_numpy(out).to(self.device)
F_ENC_DIED = "video encoder process died mid-stream: "


# --- encoder write guard --------------------------------------------------- #
# When the ffmpeg encoder dies mid-stream (driver failure, NVENC session loss),
# Python surfaces it as a bare OSError [Errno 22] from the dead pipe. Wrap it
# into a one-line, actionable error for the GUI queue instead of a raw dump.
def _encode_write(pipe, sr_bgr: "np.ndarray") -> None:
    try:
        pipe.stdin.write(sr_bgr.tobytes())
    except (OSError, ValueError) as e:
        _m = F_ENC_DIED + repr(e)
        raise RuntimeError(_m) from e


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

    def __init__(self, in_queue: "queue.Queue", out_queue: "queue.Queue"):
        super().__init__(daemon=True)
        self.in_queue = in_queue
        self.out_queue = out_queue
        self._shutdown = False

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
        if job.is_video:
            self._run_video(job)
        else:
            self._run_image(job)

    # --- image ---
    def _run_image(self, job: RunJob):
        self.emit(JobEvent(kind="started", job_id=job.job_id,
                            message=f"image: {job.input_path.name}"))
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
                        y, _n_aug, _names = _tta_forward(backend.model, y)
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

        if job.decode == "pyav":
            sync = _PyAvReader(job.input_path)
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
        if job.decode == "pyav":
            new_sync_reader = _SkipFirstFrames(_PyAvReader(job.input_path), start_frame)
            async_inner = new_sync_reader
        else:
            cap = cv2.VideoCapture(str(job.input_path))
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            new_sync_reader = _Cv2Reader.__new__(_Cv2Reader)
            new_sync_reader.path = str(job.input_path)
            new_sync_reader.cap = cap
            new_sync_reader.fps = fps
            new_sync_reader.total = total
            new_sync_reader.w = w
            new_sync_reader.h = h
            async_inner = new_sync_reader
        if job.prefetch == "async":
            sync_reader = _AsyncReader(async_inner, prefetch=8)
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
                                y, _n_aug, _names = _tta_forward(backend.model, y)
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
                    last_emit = now
        finally:
            if use_pipe:
                pipe.stdin.close()
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


def _find_ckpt(job: RunJob) -> Path:
    """The pipeline doesn't know where pretrained/ lives; the GUI passes the absolute
    path as model_filename via a side channel. We reconstruct it here by stashing
    the directory on the first call. For MVP we look next to the package + the
    settings-declared pretrained_dir."""
    # The GUI is expected to write the absolute path into job.model_filename.
    return Path(job.model_filename)


# ----------------------------------------------------------------------------
# Deprecated underscore aliases.
#
# Existing scripts (tmp/mid2s_job.py, tmp/srvgg_distill_*.py,
# scripts/step2_video_test.py, scripts/step2_visual_pair.py,
# scripts/step3_*.py) imported the private names directly. They keep
# working with these aliases. New code should import the public names.
# ----------------------------------------------------------------------------
_RunJob = RunJob  # noqa: F822 -- intentional backwards-compat alias
_JobEvent = JobEvent  # noqa: F822 -- intentional backwards-compat alias
_PipelineWorker = PipelineWorker  # noqa: F822 -- intentional backwards-compat alias