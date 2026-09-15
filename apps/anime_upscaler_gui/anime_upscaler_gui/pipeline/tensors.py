"""Tensor pre/post-processing: pinned-memory pool, HWC<->CHW, resize, downscale.

Pure numpy/torch helpers shared between image and video paths.
No backend or job knowledge here -- this module is "shape-only".

Phase A2 of the GUI extensibility refactor moved these out of the monolithic
``pipeline.py`` (1034 LOC) into a focused submodule. The public API stays the
same: every name is re-exported from ``pipeline.__init__``.
"""
from typing import Optional

import cv2
import numpy as np
import torch

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

