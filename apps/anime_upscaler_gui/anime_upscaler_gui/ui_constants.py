"""UI constants centralized so the theme is one place to edit."""
import sys

from .state import JobStatus


# ---- Layout (4-pt spacing scale) ------------------------------------------- #
PAD_X = 6
PAD_Y = 4
GROUP_PAD = 8

SPACE_XS = 4
SPACE_SM = 8
SPACE_MD = 12
SPACE_LG = 16
SPACE_XL = 24


# ---- Colors (light theme; swap to a darker set later if a theme module is added) ----
BG = "#f5f5f5"
SURFACE = "#ffffff"
SURFACE_ALT = "#f3f4f6"
FG = "#1a1a1a"
TEXT_MUTED = "#555555"
ACCENT = "#2b6cb0"
ACCENT_HOVER = "#2c5282"
ACCENT_PRESSED = "#1e3a8a"
ERROR = "#c53030"
OK = "#2f855a"
WARN = "#b7791f"
DISABLED = "#9ca3af"
BORDER = "#e5e7eb"
BORDER_STRONG = "#d1d5db"


# ---- Preview / listbox specifics ------------------------------------------ #
PREVIEW_BG = "#1f2937"
PREVIEW_LABEL = "#f9fafb"
LISTBOX_BG = "#ffffff"
LISTBOX_FG = "#111827"


# ---- Status background tints (matching foreground) ------------------------ #
STATUS_BG_PENDING = "#f3f4f6"
STATUS_BG_RUNNING = "#dbeafe"
STATUS_BG_DONE = "#d1fae5"
STATUS_BG_ERROR = "#fee2e2"
STATUS_BG_SKIPPED = "#e5e7eb"
STATUS_BG_CANCELLED = "#fef3c7"
# Q2 (perf/queue-controls-gpu-codec): paused jobs render with a calm
# indigo/blue-grey tint so users can tell "paused" apart from "running"
# at a glance without reading the status text.
STATUS_BG_PAUSED = "#e0e7ff"


# ---- Fonts (use platform defaults; explicit family avoids surprises on Windows) ----
FONT_FAMILY = "Segoe UI" if sys.platform == "win32" else ("Helvetica" if sys.platform == "darwin" else "DejaVu Sans")
FONT_FAMILY_MONO = "Consolas" if sys.platform == "win32" else ("Menlo" if sys.platform == "darwin" else "DejaVu Sans Mono")

# Typography scale
FONT_CAPTION = (FONT_FAMILY, 9)
FONT_BODY = (FONT_FAMILY, 10)
FONT_BASE = FONT_BODY
FONT_HEADING = (FONT_FAMILY, 11, "bold")
FONT_TITLE = (FONT_FAMILY, 14, "bold")
FONT_DISPLAY = (FONT_FAMILY, 20, "bold")
FONT_MONO = (FONT_FAMILY_MONO, 10)


# ---- Queue status colors (foreground) ------------------------------------- #
STATUS_COLORS = {
    JobStatus.PENDING: FG,
    JobStatus.RUNNING: ACCENT,
    JobStatus.DONE: OK,
    JobStatus.ERROR: ERROR,
    JobStatus.SKIPPED: DISABLED,
    JobStatus.CANCELLED: WARN,
    JobStatus.PAUSED: "#4338ca",  # indigo-700: distinguishable from running blue
}

STATUS_BG_COLORS = {
    JobStatus.PENDING: STATUS_BG_PENDING,
    JobStatus.RUNNING: STATUS_BG_RUNNING,
    JobStatus.DONE: STATUS_BG_DONE,
    JobStatus.ERROR: STATUS_BG_ERROR,
    JobStatus.SKIPPED: STATUS_BG_SKIPPED,
    JobStatus.CANCELLED: STATUS_BG_CANCELLED,
    JobStatus.PAUSED: STATUS_BG_PAUSED,
}
