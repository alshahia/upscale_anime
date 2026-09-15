"""pipeline -- the inference engine for the GUI (Phase A2 sub-package).

Before Phase A2 this was a single 1034-line ``pipeline.py`` file. It is now a
sub-package with five focused modules so new models / new backends / new
post-processing steps slot in without editing a giant file:

  * ``jobs.py``     -- RunJob / JobEvent dataclasses + SKIP_* resume actions
  * ``tensors.py``  -- pinned-memory pool, HWC<->CHW, resize, downscale
  * ``backends.py`` -- PyTorch / ONNX / TensorRT backends, cascade, TTA
  * ``encoder.py``  -- ffmpeg pipe write guard
  * ``worker.py``   -- PipelineWorker threaded driver (lifecycle / decode /
                       infer / encode / emit)

The public API is preserved exactly: every name that used to live on the old
``pipeline`` module is re-exported here, so existing scripts and tests that
did ``from apps.anime_upscaler_gui.anime_upscaler_gui import pipeline as P``
continue to work unchanged.

See ``docs/gui_audit.md`` (Phase A2 section) for the full split rationale.
"""

# --- jobs --- (data carriers + resume actions)
from .jobs import (  # noqa: F401
    ABORT_JOB,
    RETRY_FRAME,
    SKIP_FRAME,
    SKIP_REST,
    JobEvent,
    RunJob,
)

# --- tensors --- (shape-only helpers + pinned-memory pool)
from .tensors import (  # noqa: F401
    _PinnedPool,
    _downscale_if_needed,
    _pinned,
    _resize_keep_ar,
    _tensor_to_bgr,
    _tensor_to_bgr_batch,
    _to_tensor,
    _to_tensor_batch,
)

# --- backends --- (inference engines + selection policy)
from .backends import (  # noqa: F401
    REGISTRY,
    BackendRegistry,
    BackendSelectionCtx,
    _HAS_ORT,
    _HAS_TRT,
    _OnnxBackend,
    _PyTorchBackend,
    _cascade_count,
    _make_backend,
    _tta_forward,
    register_backend,
    select_backend,
)

# --- encoder --- (ffmpeg pipe guard)
from .encoder import (  # noqa: F401
    F_ENC_DIED,
    _encode_write,
)

# --- worker --- (the threaded driver itself + helpers)
from .worker import (  # noqa: F401
    PipelineWorker,
    _batched,
    _find_ckpt,
)

# --- decoder re-exports ---
# Worker.py imports these from ..decoders; tests assert they remain reachable via
# ``pipeline._Foo`` for back-compat (see test_video_pipeline_fixes.py).
from ..decoders import (  # noqa: F401
    _AsyncReader,
    _Cv2Reader,
    _PyAvReader,
    _SkipFirstFrames,
)

# ----------------------------------------------------------------------------
# Deprecated underscore aliases (back-compat with existing scripts/tests).
# ----------------------------------------------------------------------------
# Existing scripts and tests imported the private names directly. They keep
# working with these aliases but the access emits a DeprecationWarning -- see
# apps/anime_upscaler_gui/anime_upscaler_gui/_deprecation.py for the
# single-source-of-truth removal timeline (currently 0.4.0). New code should
# import the public names directly.
from .._deprecation import make_alias as _make_deprecated_alias
_REMOVAL_VERSION = "0.4.0"
_RunJob = _make_deprecated_alias("RunJob", RunJob, _REMOVAL_VERSION)  # noqa: F822
_JobEvent = _make_deprecated_alias("JobEvent", JobEvent, _REMOVAL_VERSION)  # noqa: F822
_PipelineWorker = _make_deprecated_alias("PipelineWorker", PipelineWorker, _REMOVAL_VERSION)  # noqa: F822

__all__ = [
    # jobs
    "RunJob", "JobEvent", "SKIP_FRAME", "SKIP_REST", "ABORT_JOB", "RETRY_FRAME",
    # tensors
    "_PinnedPool", "_pinned", "_to_tensor", "_to_tensor_batch",
    "_tensor_to_bgr", "_tensor_to_bgr_batch", "_resize_keep_ar",
    "_downscale_if_needed",
    # backends (Phase B3 adds the registry)
    "BackendRegistry", "BackendSelectionCtx", "REGISTRY",
    "register_backend", "select_backend",
    "_HAS_ORT", "_HAS_TRT", "_PyTorchBackend", "_OnnxBackend",
    "_make_backend", "_cascade_count", "_tta_forward",
    # encoder
    "F_ENC_DIED", "_encode_write",
    # worker
    "PipelineWorker", "_batched", "_find_ckpt",
    # re-exported decoders
    "_AsyncReader", "_Cv2Reader", "_PyAvReader", "_SkipFirstFrames",
    # deprecated aliases
    "_RunJob", "_JobEvent", "_PipelineWorker",
]