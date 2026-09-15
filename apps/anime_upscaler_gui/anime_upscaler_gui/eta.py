"""Formatting helpers (durations, file sizes, speeds).

Lightweight, dependency-free. Phase 14 will route `format_eta` through
babel.dates for locale-aware formatting.
"""


def format_eta(seconds: float) -> str:
    """Format a duration in seconds as `H:MM:SS`, `MM:SS`, or `Ns`.

    Examples:
        format_eta(75)   -> "1:15"
        format_eta(3661) -> "1:01:01"
        format_eta(45.4) -> "45s"
    """
    if seconds is None or seconds < 0 or seconds != seconds:  # NaN guard
        return "—"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}:{s:02d}"
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def format_speed(mb_per_s: float) -> str:
    """Format a transfer speed as `N.N MB/s` or `N.N KB/s`."""
    if mb_per_s is None or mb_per_s < 0:
        return "—"
    if mb_per_s < 1.0:
        return f"{mb_per_s * 1024:.1f} KB/s"
    return f"{mb_per_s:.1f} MB/s"
