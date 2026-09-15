"""RunJob / JobEvent dataclasses + resume-action constants.

These are the data carriers that flow between the GUI and the worker:
  * ``RunJob``     -- everything the worker needs to process one input file.
  * ``JobEvent``   -- everything the worker emits back to the GUI per status
                       change (started / progress / finished / error / etc.).
  * SKIP_FRAME / SKIP_REST / ABORT_JOB / RETRY_FRAME -- what the GUI returns
    from ``RunJob.on_frame_error`` to tell the worker how to handle a bad
    video frame.

Phase A2 of the GUI extensibility refactor moved these out of the monolithic
``pipeline.py`` (1034 LOC) into a focused submodule. The public API stays the
same: every name is re-exported from ``pipeline.__init__``.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

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

