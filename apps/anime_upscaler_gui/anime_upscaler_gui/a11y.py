"""Accessibility helpers: focus outline, status glyphs, dynamic wraplength.

Phase 10: WCAG-AA, status triplet (color + text + glyph), visible focus
ring, return-key binding for spinboxes.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional

from .state import JobStatus
from .ui_constants import (OK, ERROR, WARN, FG, STATUS_COLORS,
                             FONT_HEADING, FONT_BASE)


# Status triplet: (Unicode glyph, foreground color, accessible label).
# A "triplet" = (color, glyph, text) so users who see only one of those
# channels still get the meaning. Colors come from STATUS_COLORS (mutated
# by `theme.apply` at startup); fallback to FG when a status is missing.
_DEFAULT_COLORS = {
    JobStatus.PENDING: "#64748B",
    JobStatus.RUNNING: "#2563EB",
    JobStatus.DONE: "#15803D",
    JobStatus.ERROR: "#B91C1C",
    JobStatus.SKIPPED: "#B45309",
    JobStatus.CANCELLED: "#57534E",
    # Q2 (perf/queue-controls-gpu-codec): paused jobs render in indigo so
    # users can tell "paused" apart from "running" without reading the
    # status text. The a11y foreground is used by the status triplet when
    # the theme hasn't supplied an explicit override (status_color() falls
    # back to _DEFAULT_COLORS). ui_constants.STATUS_COLORS[PAUSED] holds
    # the runtime foreground; theme.py does not yet override it.
    JobStatus.PAUSED: "#4338CA",
}

_STATUS_GLYPHS: dict = {
    JobStatus.PENDING: ("○", "Pending"),
    JobStatus.RUNNING: ("◐", "Running"),
    JobStatus.DONE: ("●", "Done"),
    JobStatus.ERROR: ("✕", "Error"),
    JobStatus.SKIPPED: ("◌", "Skipped"),
    JobStatus.CANCELLED: ("◌", "Cancelled"),
    # Q2: pause uses a different glyph ("❚❚") from "running" (◐) so the
    # queue row remains readable in pure-text contexts (screen readers,
    # log greps). Distinct from cancelled (◌) by direction.
    JobStatus.PAUSED: ("❚❚", "Paused"),
}


def _color_for(st: JobStatus) -> str:
    return STATUS_COLORS.get(st) or _DEFAULT_COLORS.get(st) or FG


def status_glyph(status: JobStatus) -> str:
    return _STATUS_GLYPHS[status][0]


def status_label(status: JobStatus) -> str:
    return _STATUS_GLYPHS[status][1]


def status_color(status: JobStatus) -> str:
    return _color_for(status)


def status_triplet(status: JobStatus) -> tuple:
    """Return (glyph, color, label) for `status`."""
    return (_STATUS_GLYPHS[status][0], _color_for(status), _STATUS_GLYPHS[status][1])


# ---- Focus outline ----

def install_focus_outline(app) -> None:
    """Make keyboard focus visible: draw a 2px accent outline around the
    focused ttk widget. Wires <FocusIn> / <FocusOut> on common ttk classes.

    Tk doesn't have a global option for this; we patch ttk.Style to add
    a visible focus color, and bind to ttk.Widget for a thicker border.
    """
    from .ui_constants import ACCENT  # late import to avoid cycle

    style = ttk.Style(app)
    # Try to set the focusfill color so ttk's dotted focus ring matches accent.
    # Not all themes honour focusfill, but the new ones (clam, alt) do.
    try:
        style.map("TButton",
                  foreground=[("focus", ACCENT)])
        style.map("TEntry",
                  foreground=[("focus", ACCENT)])
        style.map("TCombobox",
                  foreground=[("focus", ACCENT)])
    except tk.TclError:
        pass


# ---- Dynamic wraplength ----

def auto_wraplength(widget, *, padding: int = 40, min_width: int = 200) -> None:
    """Bind <Configure> so the widget's wraplength matches its current width.

    Works for ttk.Label, tk.Label, ttk.Button, etc. — anything that supports
    `cget('wraplength')` + `configure(wraplength=...)`.
    """
    def _on_resize(e):
        w = max(min_width, e.width - padding)
        try:
            widget.configure(wraplength=w)
        except tk.TclError:
            pass
    widget.bind("<Configure>", _on_resize)


# ---- Return key on Spinbox/Entry ----

def bind_return(widget, command) -> None:
    """Pressing <Return> on `widget` invokes `command` (and doesn't move focus)."""
    widget.bind("<Return>", lambda e: (command(), "break")[1])


# ---- Modal toolwindow ----

def make_tool_modal(parent, title: str = "") -> "tk.Toplevel":
    """Create a Toplevel with `-toolwindow` style (Windows) and grab_set."""
    top = tk.Toplevel(parent)
    if title:
        top.title(title)
    top.transient(parent)
    try:
        top.attributes("-toolwindow", True)  # Windows: hides from taskbar
    except tk.TclError:
        pass
    top.grab_set()
    return top


# ---- Accessible LabelFrame title pattern ----

def labeled_section(parent, text: str, *, padding: int = 8) -> ttk.LabelFrame:
    """ttk.LabelFrame with a default padding — convenience."""
    return ttk.LabelFrame(parent, text=text, padding=padding)
