"""Theme system: 3 themes (Light / Dark / High-contrast) sharing an Indigo Slate palette.

`apply(theme)` mutates `ui_constants` module attributes IN PLACE so existing
imports (`from .ui_constants import BG`) keep pointing at live values. Widgets
constructed AFTER `apply()` see the new colors; existing widgets must be
rebuilt (`app._build_ui()`) to pick up the change.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

from .state import JobStatus


# ============================================================================ #
# Theme dataclass
# ============================================================================ #
@dataclass(frozen=True)
class Theme:
    name: str

    # Surfaces
    bg: str
    surface: str
    surface_alt: str

    # Text
    text: str
    text_muted: str
    disabled: str

    # Borders
    border: str
    border_strong: str

    # Accent
    accent: str
    accent_hover: str
    accent_pressed: str

    # Status text colors
    pending: str
    running: str
    done: str
    error: str
    skipped: str
    cancelled: str

    # Status row tints (used by future listbox / treeview; not yet wired to widgets)
    bg_pending: str
    bg_running: str
    bg_done: str
    bg_error: str
    bg_skipped: str
    bg_cancelled: str

    # Preview / listbox
    preview_bg: str
    preview_label: str
    listbox_bg: str
    listbox_fg: str


# ============================================================================ #
# Three concrete themes
# ============================================================================ #
LIGHT = Theme(
    name="light",
    bg="#F8FAFC",
    surface="#FFFFFF",
    surface_alt="#F1F5F9",
    text="#0F172A",
    text_muted="#64748B",
    disabled="#94A3B8",
    border="#E2E8F0",
    border_strong="#CBD5E1",
    accent="#4F46E5",
    accent_hover="#4338CA",
    accent_pressed="#3730A3",
    pending="#64748B",
    running="#2563EB",
    done="#15803D",
    error="#B91C1C",
    skipped="#B45309",
    cancelled="#57534E",
    bg_pending="#F1F5F9",
    bg_running="#DBEAFE",
    bg_done="#DCFCE7",
    bg_error="#FEE2E2",
    bg_skipped="#FEF3C7",
    bg_cancelled="#E7E5E4",
    preview_bg="#1F2937",
    preview_label="#F9FAFB",
    listbox_bg="#FFFFFF",
    listbox_fg="#111827",
)

DARK = Theme(
    name="dark",
    bg="#0F172A",
    surface="#1E293B",
    surface_alt="#334155",
    text="#F1F5F9",
    text_muted="#94A3B8",
    disabled="#64748B",
    border="#334155",
    border_strong="#475569",
    accent="#818CF8",
    accent_hover="#A5B4FC",
    accent_pressed="#C7D2FE",
    pending="#94A3B8",
    running="#60A5FA",
    done="#4ADE80",
    error="#F87171",
    skipped="#FBBF24",
    cancelled="#A8A29E",
    bg_pending="#1E293B",
    bg_running="#1E3A8A",
    bg_done="#14532D",
    bg_error="#7F1D1D",
    bg_skipped="#78350F",
    bg_cancelled="#44403C",
    preview_bg="#020617",
    preview_label="#F1F5F9",
    listbox_bg="#1E293B",
    listbox_fg="#F1F5F9",
)

HIGH_CONTRAST = Theme(
    name="high_contrast",
    bg="#FFFFFF",
    surface="#FFFFFF",
    surface_alt="#F4F4F5",
    text="#000000",
    text_muted="#1F2937",
    disabled="#6B7280",
    border="#000000",
    border_strong="#000000",
    accent="#0000EE",
    accent_hover="#0000BB",
    accent_pressed="#000088",
    pending="#374151",
    running="#1D4ED8",
    done="#15803D",
    error="#B91C1C",
    skipped="#B45309",
    cancelled="#52525B",
    bg_pending="#F4F4F5",
    bg_running="#E0E7FF",
    bg_done="#DCFCE7",
    bg_error="#FEE2E2",
    bg_skipped="#FEF3C7",
    bg_cancelled="#E7E5E4",
    preview_bg="#000000",
    preview_label="#FFFFFF",
    listbox_bg="#FFFFFF",
    listbox_fg="#000000",
)

THEMES: Dict[str, Theme] = {"light": LIGHT, "dark": DARK, "high_contrast": HIGH_CONTRAST}


# ============================================================================ #
# Apply
# ============================================================================ #
# Map from dataclass field -> module attribute in `ui_constants`.
# Anything in here will be patched when `apply()` runs.
_FIELD_TO_CONST = {
    "bg": "BG",
    "surface": "SURFACE",
    "surface_alt": "SURFACE_ALT",
    "text": "FG",
    "text_muted": "TEXT_MUTED",
    "disabled": "DISABLED",
    "border": "BORDER",
    "border_strong": "BORDER_STRONG",
    "accent": "ACCENT",
    "accent_hover": "ACCENT_HOVER",
    "accent_pressed": "ACCENT_PRESSED",
    "error": "ERROR",
    "pending": "PENDING_FG",
    "running": "RUNNING_FG",
    "done": "OK",
    "skipped": "WARN",
    "cancelled": "CANCELLED_FG",
    "bg_pending": "STATUS_BG_PENDING",
    "bg_running": "STATUS_BG_RUNNING",
    "bg_done": "STATUS_BG_DONE",
    "bg_error": "STATUS_BG_ERROR",
    "bg_skipped": "STATUS_BG_SKIPPED",
    "bg_cancelled": "STATUS_BG_CANCELLED",
    "preview_bg": "PREVIEW_BG",
    "preview_label": "PREVIEW_LABEL",
    "listbox_bg": "LISTBOX_BG",
    "listbox_fg": "LISTBOX_FG",
}


def apply(theme: Theme) -> None:
    """Mutate `ui_constants` module attributes IN PLACE so the rest of the app
    sees the new palette without rebinding imports."""
    import apps.anime_upscaler_gui.anime_upscaler_gui.ui_constants as ui

    for fld, attr in _FIELD_TO_CONST.items():
        setattr(ui, attr, getattr(theme, fld))

    # Re-derive derived dicts (foreground colors per status). Mutate, don't
    # replace, so imported refs (e.g. `from .ui_constants import STATUS_COLORS`)
    # keep pointing at the same object.
    ui.STATUS_COLORS[JobStatus.PENDING] = theme.pending
    ui.STATUS_COLORS[JobStatus.RUNNING] = theme.running
    ui.STATUS_COLORS[JobStatus.DONE] = theme.done
    ui.STATUS_COLORS[JobStatus.ERROR] = theme.error
    ui.STATUS_COLORS[JobStatus.SKIPPED] = theme.skipped
    ui.STATUS_COLORS[JobStatus.CANCELLED] = theme.cancelled

    ui.STATUS_BG_COLORS[JobStatus.PENDING] = theme.bg_pending
    ui.STATUS_BG_COLORS[JobStatus.RUNNING] = theme.bg_running
    ui.STATUS_BG_COLORS[JobStatus.DONE] = theme.bg_done
    ui.STATUS_BG_COLORS[JobStatus.ERROR] = theme.bg_error
    ui.STATUS_BG_COLORS[JobStatus.SKIPPED] = theme.bg_skipped
    ui.STATUS_BG_COLORS[JobStatus.CANCELLED] = theme.bg_cancelled


def by_name(name: str) -> Theme:
    if name not in THEMES:
        raise KeyError(f"unknown theme: {name!r}; choose from {sorted(THEMES)}")
    return THEMES[name]


def available() -> List[str]:
    return sorted(THEMES.keys())


# ============================================================================ #
# WCAG contrast ratio
# ============================================================================ #
def _hex_to_rgb(s: str) -> tuple:
    s = s.lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def _relative_luminance(rgb: tuple) -> float:
    def _chan(c: int) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (_chan(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(fg: str, bg: str) -> float:
    """WCAG 2.x relative luminance ratio (1.0 to 21.0)."""
    l1 = _relative_luminance(_hex_to_rgb(fg))
    l2 = _relative_luminance(_hex_to_rgb(bg))
    lighter, darker = (l1, l2) if l1 >= l2 else (l2, l1)
    return (lighter + 0.05) / (darker + 0.05)


# Convenience for tests: list (field, fg, bg) pairs that must hit 4.5:1.
REQUIRED_CONTRAST_PAIRS: List[tuple] = [
    ("text", "text", "bg"),
    ("text", "text", "surface"),
    ("text_muted", "text_muted", "bg"),
    ("accent", "accent", "bg"),
    ("error", "error", "bg"),
    ("done", "done", "bg"),
    ("running", "running", "bg"),
    ("pending", "pending", "bg"),
    ("skipped", "skipped", "bg"),
    ("cancelled", "cancelled", "bg"),
]


def assert_wcag(theme: Theme, min_ratio: float = 4.5) -> List[tuple]:
    """Return a list of (label, fg, bg, ratio) for any pair that fails the
    minimum contrast threshold. Empty list = all pass."""
    data = asdict(theme)
    failures = []
    for label, fg_key, bg_key in REQUIRED_CONTRAST_PAIRS:
        fg = data[fg_key]
        bg = data[bg_key]
        r = contrast_ratio(fg, bg)
        if r < min_ratio:
            failures.append((label, fg, bg, round(r, 2)))
    return failures
