"""Inference backends: PyTorch, ONNX, TensorRT (optional), cascade helper, TTA.

A "backend" is anything that exposes ``__call__(x: torch.Tensor) -> torch.Tensor``
and ``.model`` (the underlying torch.nn.Module so cascade/TTA can introspect).

Phase A2 of the GUI extensibility refactor moved these out of the monolithic
``pipeline.py`` (1034 LOC) into a focused submodule.

Phase B3 adds BackendRegistry + select_backend(): a strategy pattern that
replaces the old if/else chain in _make_backend. Adding a new backend is now
one ``@register_backend("name", predicate, factory)`` decorator. The default
registry holds TensorRT (highest priority) and PyTorch (always-last fallback),
which is exactly what the previous _make_backend silently did.

Public API (Phase A1 contract):
  select_backend(model, ckpt_path, kind, device, fp16, use_tensorrt, tta,
                 batch_size=1) -> (backend, name)
  BackendRegistry                          # one entry (name, predicate, factory)
  register_backend(name, predicate, factory) -> BackendRegistry
                                              # decorator: add to default REGISTRY
  REGISTRY  -> list[BackendRegistry]        # priority-ordered (index 0 = try first)

Backwards-compat (Phase A1 contract): _make_backend is preserved as an alias.
"""
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Tuple

import numpy as np
import torch

from .._deprecation import make_alias

from ..archs import build, _save_sr, capabilities as _arch_caps

log = logging.getLogger(__name__)


# Public: the registry entry type. A backend registers itself by name + a
# predicate (can-it-run?) + a factory (build-it-now). The registry is
# priority-ordered: index 0 is tried first.
@dataclass(frozen=True)
class BackendRegistry:
    """One entry in the backend registry.

    Attributes:
      name:     stable string identifier (e.g. "TensorRT", "PyTorch").
                Returned to the caller so logs / UI can surface the choice.
      predicate: callable(BackendSelectionCtx) -> bool.
                True means "I can run this job". Should be cheap and
                side-effect free (no I/O, no model loading); it answers
                "should we try this backend?" so the registry can iterate.
      factory:   callable(BackendSelectionCtx) -> backend_instance.
                Builds and returns the backend. May raise -- the registry
                treats raised exceptions as "this backend failed", falls
                through to the next entry, and logs at WARNING level.
    """
    name: str
    predicate: Callable[["BackendSelectionCtx"], bool]
    factory: Callable[["BackendSelectionCtx"], Any]


@dataclass(frozen=True)
class BackendSelectionCtx:
    """Read-only snapshot passed to predicates and factories.

    Same fields the old _make_backend took. Bundled so a custom backend can
    grow without changing the function signature.
    """
    model: Any         # the already-built torch.nn.Module (eval mode, fp16 applied)
    ckpt_path: str
    kind: str          # e.g. "rfdn_student", "animesr"
    device: Any        # torch.device
    fp16: bool
    use_tensorrt: bool
    tta: bool
    batch_size: int = 1


# The default registry. Higher-priority backends are listed first;
# the final entry (PyTorch) is the always-on fallback.
REGISTRY: List[BackendRegistry] = []


def register_backend(
    name: str,
    predicate: Callable[[BackendSelectionCtx], bool],
    factory: Callable[[BackendSelectionCtx], Any],
) -> BackendRegistry:
    """Decorator-free helper: register a backend on the default REGISTRY.

    Usage::

        register_backend("PyTorch",
            predicate=lambda c: True,
            factory=lambda c: _PyTorchBackend(c.ckpt_path, c.kind, c.device, c.fp16),
        )

    Returns the registered entry so it can also be stored by the caller.
    Tests typically build a temporary registry; production uses REGISTRY.
    """
    entry = BackendRegistry(name=name, predicate=predicate, factory=factory)
    REGISTRY.append(entry)
    return entry


def select_backend(
    model,
    ckpt_path,
    kind,
    device,
    fp16,
    use_tensorrt,
    tta=False,
    batch_size: int = 1,
    registry: List[BackendRegistry] = None,
) -> Tuple[Any, str]:
    """Pick the first backend whose predicate accepts this job.

    Walks ``registry`` (default: REGISTRY) in order; the first backend whose
    predicate returns True is built via factory(). If the factory raises,
    we log a WARNING and fall through to the next entry.

    Returns ``(backend, backend_name)``. Always returns SOMETHING because the
    default REGISTRY ends with a PyTorch backend whose predicate is "always True".
    If you pass a custom registry without a fallback, a NoBackendError is raised.
    """
    ctx = BackendSelectionCtx(
        model=model, ckpt_path=ckpt_path, kind=kind, device=device,
        fp16=fp16, use_tensorrt=use_tensorrt, tta=tta, batch_size=batch_size,
    )
    if registry is None:
        registry = REGISTRY
    last_exc: Exception = None
    for entry in registry:
        try:
            if not entry.predicate(ctx):
                continue
        except Exception as e:
            log.warning("backend %r predicate raised (%s); skipping", entry.name, e)
            continue
        try:
            backend = entry.factory(ctx)
            log.info("backend: %s (kind=%s, batch=%d)", entry.name, kind, batch_size)
            return backend, entry.name
        except Exception as e:
            log.warning("backend %r factory failed (%s); trying next", entry.name, e)
            last_exc = e
    if last_exc is not None:
        raise RuntimeError(
            "no backend in the registry succeeded; last error: %r" % last_exc
        ) from last_exc
    raise RuntimeError("no backend matched (registry=%r)" % ([e.name for e in registry],))


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
# Built-in backends: TensorRT (priority 1) and PyTorch (always-on fallback)
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

    def __call__(self, x):
        with torch.no_grad():
            return self.model(x)


class _OnnxBackend:
    """Optional onnxruntime-gpu backend; auto-exports the .onnx if missing."""

    def __init__(self, onnx_path, kind: str, device, fp16: bool, model=None):
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

    def __call__(self, x):
        with torch.no_grad():
            inp = x.detach().cpu().numpy().astype(self.np_dtype)
        n_recurrent = _arch_caps(self.kind).recurrent_frames
        if n_recurrent:
            inp = np.repeat(inp[:, None], n_recurrent, axis=1)
        out = self.sess.run(None, {"input": inp})[0]
        if n_recurrent:
            out = out[:, out.shape[1] // 2]
        return torch.from_numpy(out).to(self.device)


# --- TensorRT backend (registered as a priority-1 backend) ---
def _trt_predicate(ctx: BackendSelectionCtx) -> bool:
    """TRT needs: not TTA, use_tensorrt requested, lib available, CUDA device,
    fp16 (TRT engine is fp16-only here), and the arch says it supports TRT."""
    if ctx.tta:
        return False
    if not ctx.use_tensorrt:
        return False
    if not _HAS_TRT or _TrtBackend_t is None:
        return False
    if not (hasattr(ctx.device, "type") and ctx.device.type == "cuda"):
        return False
    if not ctx.fp16:
        return False
    if not _arch_caps(ctx.kind).tensorrt:
        return False
    return True


def _trt_factory(ctx: BackendSelectionCtx):
    """Build the TRT engine + return a _TrtBackend instance."""
    cache_dir = Path(os.environ.get("APPDATA", str(Path.home()))) / "anime_upscaler_gui" / "cache" / "trt"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return _TrtBackend_t(ctx.model, ctx.kind, ctx.device, ctx.fp16, cache_dir, batch_size=ctx.batch_size)


# --- PyTorch backend (always-on fallback, predicate=True) ---
def _pytorch_predicate(ctx: BackendSelectionCtx) -> bool:
    return True


def _pytorch_factory(ctx: BackendSelectionCtx):
    return _PyTorchBackend(ctx.ckpt_path, ctx.kind, ctx.device, ctx.fp16)


# Register the default registry. Order matters: TRT first (when available),
# PyTorch last (always-on fallback).
register_backend("TensorRT", _trt_predicate, _trt_factory)
register_backend("PyTorch", _pytorch_predicate, _pytorch_factory)


# ============================================================================ #
# Backwards-compat: the old function name.
# ============================================================================ #
_make_backend = make_alias(
    "select_backend", select_backend, removal_version="0.4.0",
)  # noqa: F822 -- preserved for Phase A1/B3 contract


__all__ = [
    # Public Phase B3 API:
    "BackendRegistry",
    "BackendSelectionCtx",
    "REGISTRY",
    "register_backend",
    "select_backend",
    # Underscore-aliased (Phase A1 contract):
    "_make_backend",
    "_PyTorchBackend",
    "_HAS_TRT",
    "_HAS_ORT",
    "_cascade_count",
    "_tta_forward",
    # PyTorch class for direct construction in tests (used by callers that
    # need a specific backend without going through the selector):
    "_TrtBackend_t",  # noqa: F822 -- alias for the optional TRT backend class
]
