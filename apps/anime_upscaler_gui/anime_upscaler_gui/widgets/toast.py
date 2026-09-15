"""Toast: a transient, non-blocking notification that auto-dismisses.

Use: `Toast(parent, "Settings saved").show(duration_ms=2500)` — appears
in the bottom-right of `parent` and fades out.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ..ui_constants import ACCENT, FG, BG, FONT_BASE, GROUP_PAD


class Toast(tk.Toplevel):
    """A borderless top-level that overlays its parent."""

    def __init__(self, parent, message: str, *, duration_ms: int = 2500):
        super().__init__(parent)
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self._duration_ms = duration_ms
        self._alpha_target = 1.0
        self._fade_step = 0.1
        self._fade_interval = 50

        body = ttk.Frame(self, padding=GROUP_PAD, relief="solid", borderwidth=1)
        body.pack(fill="both", expand=True)
        label = ttk.Label(body, text=message, font=FONT_BASE, foreground=FG, background=BG)
        label.pack()

        # Position bottom-right of parent
        self.update_idletasks()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        px = parent.winfo_rootx()
        py = parent.winfo_rooty()
        w = self.winfo_width()
        h = self.winfo_height()
        x = px + pw - w - 20
        y = py + ph - h - 20
        self.geometry(f"+{x}+{y}")

    def show(self) -> None:
        self.after(self._duration_ms, self._start_fade)
        # Bring forward
        self.lift()

    def _start_fade(self) -> None:
        # Try a smooth fade; if the platform doesn't support alpha, just destroy.
        try:
            current = self.attributes("-alpha")
        except tk.TclError:
            self.destroy()
            return
        if current <= 0.0:
            self.destroy()
            return
        try:
            self.attributes("-alpha", max(0.0, current - self._fade_step))
        except tk.TclError:
            self.destroy()
            return
        self.after(self._fade_interval, self._start_fade)
