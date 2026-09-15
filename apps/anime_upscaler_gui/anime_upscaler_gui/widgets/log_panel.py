"""Log panel: ScrolledText backed by a `logging.Handler`.

Other modules call `logging.getLogger("anime_upscaler_gui")` and the
panel picks up records via `_LogPanelHandler.emit`. Safe to use from
worker threads (we use `after(0, ...)` to hop to the GUI thread).
"""
from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk, scrolledtext

from ..ui_constants import FONT_MONO, GROUP_PAD


_LEVEL_TAG = {
    logging.DEBUG: "debug",
    logging.INFO: "info",
    logging.WARNING: "warning",
    logging.ERROR: "error",
    logging.CRITICAL: "critical",
}


class _LogPanelHandler(logging.Handler):
    """Logging handler that forwards records to a `_LogPanel` widget.

    Emits to the GUI thread via `widget.after(0, ...)`. The widget must
    outlive this handler; the app is responsible for `set_target(None)`
    on teardown.
    """

    def __init__(self, level: int = logging.INFO) -> None:
        super().__init__(level=level)
        self._target: "_LogPanel | None" = None

    def set_target(self, panel: "_LogPanel | None") -> None:
        self._target = panel

    def emit(self, record: logging.LogRecord) -> None:
        if self._target is None:
            return
        try:
            msg = self.format(record)
            level = record.levelno
            self._target.after(0, lambda m=msg, lvl=level: self._target._append(m, lvl))
        except Exception:
            # Never let logging break the app
            pass


class _LogPanel(ttk.LabelFrame):
    """Bottom-of-window ScrolledText that displays log records."""

    def __init__(self, parent, app, *, max_lines: int = 500):
        super().__init__(parent, text="Log", padding=GROUP_PAD)
        self.app = app
        self._max_lines = max_lines

        self.text = scrolledtext.ScrolledText(self, height=8, font=FONT_MONO,
                                              state="disabled", wrap="none")
        self.text.pack(fill="both", expand=True)

        # Tags for severity coloring (set by the app theme)
        self.text.tag_configure("debug", foreground="#94A3B8")
        self.text.tag_configure("info", foreground="#0F172A")
        self.text.tag_configure("warning", foreground="#B45309")
        self.text.tag_configure("error", foreground="#B91C1C")
        self.text.tag_configure("critical", foreground="#FFFFFF", background="#B91C1C")

    def _append(self, msg: str, level: int) -> None:
        try:
            self.text.config(state="normal")
            self.text.insert("end", msg + "\n", _LEVEL_TAG.get(level, "info"))
            # Trim to max_lines
            line_count = int(self.text.index("end-1c").split(".")[0])
            if line_count > self._max_lines:
                excess = line_count - self._max_lines
                self.text.delete("1.0", f"{excess + 1}.0")
            self.text.see("end")
            self.text.config(state="disabled")
        except tk.TclError:
            pass

    def clear(self) -> None:
        self.text.config(state="normal")
        self.text.delete("1.0", "end")
        self.text.config(state="disabled")
