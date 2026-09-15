"""Encoder write guard -- wraps ffmpeg pipe writes with one-line error reporting.

When the ffmpeg encoder process dies mid-stream (driver failure, NVENC
session loss), Python surfaces it as a bare OSError [Errno 22] from the dead
pipe. We wrap it into a single-line, actionable RuntimeError so the worker
emits a clean JobEvent instead of dumping a raw traceback to the GUI queue.

Phase A2 of the GUI extensibility refactor moved these out of the monolithic
``pipeline.py`` (1034 LOC) into a focused submodule. The public API stays the
same: every name is re-exported from ``pipeline.__init__``.
"""
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
