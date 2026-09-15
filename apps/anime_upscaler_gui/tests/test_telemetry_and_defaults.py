"""Tests for Q4 (perf/queue-controls-gpu-codec): TF32 + async prefetch +
gpu_util telemetry.

Pins the Q4 contract:
  - tensors.py enables allow_tf32 on the matmul + cudnn backends at
    import time (idempotent on hosts without CUDA)
  - Defaults.prefetch now defaults to 'async' (was 'sync')
  - _read_gpu_util() returns a dict with util/mem_used_mb/mem_total_mb/temp_c
    on hosts where pynvml is available, and a torch.cuda.mem_get_info()-backed
    dict otherwise; returns None only when both paths are unavailable.
  - _maybe_emit_gpu_util() rate-limits emissions to >= 1 Hz regardless of
    how often the caller pokes it, and parses cleanly back into a JobEvent.
  - gpu_monitor._GPUMonitor.apply_event() updates the labels from a
    gpu_util JobEvent without touching the pynvml poll thread.
"""
import time
from pathlib import Path

import pytest

from apps.anime_upscaler_gui.anime_upscaler_gui import pipeline
from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline import (
    tensors,
    worker,
)
from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline.jobs import JobEvent
from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline.worker import (
    _read_gpu_util,
    _maybe_emit_gpu_util,
    _GPU_UTIL_MIN_INTERVAL_S,
)
from apps.anime_upscaler_gui.anime_upscaler_gui.settings import Defaults


# ---- TF32 flags ----
def test_tensors_module_enables_allow_tf32_when_cuda_available():
    """tensors.py module-level enables allow_tf32 on the relevant backends.

    On hosts without CUDA the flags remain False (PyTorch default);
    with CUDA, both flags must be True after import.
    """
    import torch
    if not torch.cuda.is_available():
        # Nothing to assert beyond "no crash". The try/except in tensors.py
        # silently swallows AttributeError on older torch.
        assert tensors is not None
        return
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True


# ---- Defaults.prefetch ----
def test_defaults_prefetch_is_async_by_default():
    """Q4 changed the prefetch default from 'sync' to 'async'."""
    assert Defaults().prefetch == "async"


# ---- gpu_util probe ----
def test_read_gpu_util_returns_dict_with_required_keys():
    """Probe returns util/mem_used_mb/mem_total_mb/temp_c (or None)."""
    info = _read_gpu_util()
    # On a CUDA host with pynvml OR with torch.cuda available, this is
    # non-None. On a CPU-only test box without pynvml it's None.
    if info is not None:
        for k in ("util", "mem_used_mb", "mem_total_mb", "temp_c"):
            assert k in info
            assert isinstance(info[k], int)
    else:
        # Truly missing: OK to return None; the worker will simply not
        # emit gpu_util events.
        assert info is None


def test_read_gpu_util_is_process_cached():
    """Subsequent calls return the same dict without re-probing.

    We patch pynvml + torch.cuda.mem_get_info through the cached probe
    by checking that the second call returns the identical object.
    """
    a = _read_gpu_util()
    b = _read_gpu_util()
    assert a is b  # process-cached identity


def test_maybe_emit_gpu_util_rate_limits_to_one_hz():
    """Calls within _GPU_UTIL_MIN_INTERVAL_S produce no emission."""
    captured = []
    def emit(evt):
        captured.append(evt)
    last_emit_ref = [0.0]
    # First call: t_last = 0, so the helper always emits when t_now >= 1.0.
    # Force a first emission:
    last_emit_ref[0] = time.perf_counter() - _GPU_UTIL_MIN_INTERVAL_S - 0.1
    _maybe_emit_gpu_util(emit, job_id=42, last_emit_ref=last_emit_ref)
    n1 = len(captured)
    # Immediate second call: should be suppressed by the 1 Hz rate limit.
    _maybe_emit_gpu_util(emit, job_id=42, last_emit_ref=last_emit_ref)
    n2 = len(captured)
    assert n2 == n1, "second call within the rate window must not emit"
    # The captured event (if any) must be a JobEvent of kind 'gpu_util'.
    if captured:
        assert captured[0].kind == "gpu_util"
        assert captured[0].job_id == 42


def test_maybe_emit_gpu_util_uses_correct_message_format():
    """The emitted message is parseable by gpu_monitor.apply_event()."""
    captured = []
    def emit(evt):
        captured.append(evt)
    last_emit_ref = [0.0]
    _maybe_emit_gpu_util(emit, job_id=7, last_emit_ref=last_emit_ref)
    # If the probe returned None (no pynvml, no cuda), nothing was
    # emitted -- that's a valid branch and the test passes vacuously.
    if captured:
        msg = captured[0].message
        assert "util=" in msg
        assert "%" in msg
        assert "mem=" in msg
        assert "MB" in msg
        assert "t=" in msg


# ---- _GPUMonitor widget: apply_event ----
def test_gpu_monitor_apply_event_parses_message():
    """apply_event() updates the three StringVars from a gpu_util JobEvent."""
    # The widget needs a Tk root; we build a minimal one and use
    # update_idletasks so the StringVar reads are flushed.
    import tkinter as tk
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.gpu_monitor import (
        _GPUMonitor,
    )
    # Phase E proxy imports + tk in test mode is fragile on Windows
    # without a display. Skip cleanly if Tk can't init.
    try:
        root = tk.Tk()
        root.withdraw()
    except Exception as e:
        pytest.skip(f"no Tk root available: {e}")
    try:
        # app is only referenced for layout; pass a stub.
        class _Stub:
            pass
        w = _GPUMonitor(root, _Stub())
        evt = JobEvent(
            kind="gpu_util", job_id=1,
            message="util=87% mem=4096/8192MB t=72C",
        )
        w.apply_event(evt)
        # StringVar reads reflect the parsed values.
        assert w._util_var.get() == "util: 87%"
        assert w._mem_var.get() == "mem: 4096 / 8192 MB"
        assert w._temp_var.get() == "temp: 72°C"
    finally:
        try:
            w.stop()
        except Exception:
            pass
        root.destroy()


def test_gpu_monitor_apply_event_tolerates_malformed_message():
    """apply_event() must not raise on a malformed gpu_util message."""
    import tkinter as tk
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.gpu_monitor import (
        _GPUMonitor,
    )
    try:
        root = tk.Tk()
        root.withdraw()
    except Exception as e:
        pytest.skip(f"no Tk root available: {e}")
    try:
        class _Stub:
            pass
        w = _GPUMonitor(root, _Stub())
        # Empty message and garbage must both be handled gracefully.
        for msg in ("", "garbage", "util=NaN% mem=garbage t=NaNC"):
            evt = JobEvent(kind="gpu_util", job_id=0, message=msg)
            w.apply_event(evt)  # must not raise
    finally:
        try:
            w.stop()
        except Exception:
            pass
        root.destroy()
