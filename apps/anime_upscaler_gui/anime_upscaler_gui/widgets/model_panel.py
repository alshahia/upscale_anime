"""Model row: dropdown + re-scan button. Plain Frame (no LabelFrame) -- it sits
inside the 'Upscale' LabelFrame alongside Input / Settings / Output."""
import tkinter as tk
from tkinter import ttk

from ..ui_constants import FONT_BASE, FONT_MONO, PAD_X


class _ModelPanel(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        ttk.Label(self, text="Model:", font=FONT_BASE).pack(side="left")
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(self, textvariable=self.model_var,
                                            state="readonly", width=60, font=FONT_MONO)
        self.model_dropdown.pack(side="left", padx=(PAD_X, PAD_X), fill="x", expand=True)
        ttk.Button(self, text="Re-scan", command=app._refresh_model_dropdown).pack(side="right")
