"""Output section: where to save (system default / same folder / custom),
output folder picker, file suffix, batch folder name."""
import tkinter as tk
from tkinter import ttk

from ..ui_constants import FONT_MONO, GROUP_PAD, PAD_X, PAD_Y


class _OutputPanel(ttk.LabelFrame):
    def __init__(self, parent, app):
        super().__init__(parent, text="Output", padding=GROUP_PAD)
        s = app.settings.data
        self.app = app

        # Mode radio
        rmode = ttk.Frame(self)
        rmode.pack(fill="x")
        ttk.Label(rmode, text="Where:").pack(side="left")
        self.output_mode_var = tk.StringVar(value=s.output_mode)
        ttk.Radiobutton(rmode, text="System default", variable=self.output_mode_var,
                        value="system_default").pack(side="left", padx=(PAD_X, PAD_X))
        ttk.Radiobutton(rmode, text="Same folder as input", variable=self.output_mode_var,
                        value="same_folder").pack(side="left")
        ttk.Radiobutton(rmode, text="Custom", variable=self.output_mode_var,
                        value="custom").pack(side="left", padx=(PAD_X, 0))

        # Folder row
        r = ttk.Frame(self)
        r.pack(fill="x", pady=(PAD_Y, 0))
        ttk.Label(r, text="Folder:").pack(side="left")
        self.output_dir_var = tk.StringVar(value=s.output_dir)
        ttk.Entry(r, textvariable=self.output_dir_var, font=FONT_MONO).pack(side="left", padx=(PAD_X, PAD_X), fill="x", expand=True)
        ttk.Button(r, text="Browse...", command=app._pick_output_dir).pack(side="right")

        # Suffix + batch folder name
        r2 = ttk.Frame(self)
        r2.pack(fill="x", pady=(PAD_Y, 0))
        ttk.Label(r2, text="Suffix:").pack(side="left")
        self.suffix_var = tk.StringVar(value=s.single_file_suffix)
        ttk.Entry(r2, textvariable=self.suffix_var, width=20).pack(side="left", padx=(PAD_X, PAD_X))
        ttk.Label(r2, text="Batch folder name:").pack(side="left", padx=(PAD_X * 2, 0))
        self.batch_folder_var = tk.StringVar(value=s.batch_folder_name)
        ttk.Entry(r2, textvariable=self.batch_folder_var, width=20).pack(side="left", padx=(PAD_X, 0))
