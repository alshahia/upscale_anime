"""Status section: progress bar + status label.

Lives at the bottom of the window; always visible. Pipeline events route through
the orchestrator's _handle_event, which writes to status_var/progress_var on this
widget.
"""
import tkinter as tk
from tkinter import ttk

from ..a11y import auto_wraplength
from ..ui_constants import FG, FONT_MONO, GROUP_PAD, PAD_Y


class _StatusBar(ttk.LabelFrame):
    def __init__(self, parent, app):
        super().__init__(parent, text="Status", padding=GROUP_PAD)
        self.app = app

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress = ttk.Progressbar(self, variable=self.progress_var, maximum=100.0)
        self.progress.pack(fill="x", pady=(0, PAD_Y))
        self.status_var = tk.StringVar(value="Idle.")
        self.status_label = ttk.Label(self, textvariable=self.status_var,
                                      font=FONT_MONO, foreground=FG, justify="left")
        self.status_label.pack(fill="x")
        # Wrap to the label's real width instead of a fixed 900px, which is
        # wider than the 800px minimum window and clipped long messages.
        auto_wraplength(self.status_label, padding=24, min_width=200)
