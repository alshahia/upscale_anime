"""Inference backends: PyTorch, ONNX, TensorRT (optional), cascade helper, TTA.

A "backend" is anything that exposes ``__call__(x: torch.Tensor) -> torch.Tensor``
and ``.model`` (the underlying torch.nn.Module so cascade/TTA can introspect).

Backend selection policy lives in ``make_backend()``: it picks TensorRT when
available + fp16 + CUDA + no TTA + arch supports it; otherwise PyTorch.
ONNX is opt-in (constructed directly by callers; not selected by this module).

Phase A2 of the GUI extensibility refactor moved these out of the monolithic
``pipeline.py`` (1034 LOC) into a focused submodule. The public API stays the
same: every name is re-exported from ``pipeline.__init__``.
"""
import logging
import os
from pathlib import Path

import numpy as np
import torch

from ..archs import build, _save_sr, capabilities as _arch_caps

log = logging.getLogger(__name__)

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
