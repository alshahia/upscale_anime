"""Map low-level exceptions to user-friendly messages.

Future i18n: the `humanize` function should return a translated string
key when Phase 14 lands. For now it returns English text.
"""
from __future__ import annotations

import errno
from typing import Optional
from urllib.error import URLError


def humanize(exc: BaseException) -> str:
    """Return a user-readable string for `exc`."""
    # Network errors
    if isinstance(exc, URLError):
        reason = getattr(exc, "reason", exc)
        return f"Network error: {reason}"
    # Disk-full (ENOSPC = 28 on POSIX; on Windows it's WinError 112).
    if isinstance(exc, OSError):
        if exc.errno in (errno.ENOSPC, 28, 112):
            return "Out of disk space; free some space and try again."
        if exc.errno in (errno.EACCES, 13):
            return "Permission denied."
        if exc.errno in (errno.ENOENT, 2):
            return f"File not found: {exc.filename or exc}"
    # Torch / model errors
    name = type(exc).__name__
    if name in ("RuntimeError", "ValueError", "FileNotFoundError"):
        return f"{name}: {exc}"
    return f"{name}: {exc}"


# Sentinel for "skip showing the dialog" — return this from a download
# callback if you don't want any error UI (e.g. user-cancelled).
SKIP = "skip"
