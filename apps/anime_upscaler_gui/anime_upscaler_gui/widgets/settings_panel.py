"""Settings section: device, precision, outscale, batch, decode, prefetch,
downscale pre-process, GPU guard, tile size, tile overlap."""
import tkinter as tk
from tkinter import ttk

from ..ui_constants import GROUP_PAD, PAD_X, PAD_Y


class _SettingsPanel(ttk.LabelFrame):
    def __init__(self, parent, app):
        super().__init__(parent, text="Settings", padding=GROUP_PAD)
        s = app.settings.data
        self.app = app

        # Row 1: device, precision, outscale
        r1 = ttk.Frame(self)
        r1.pack(fill="x", pady=(0, PAD_Y))
        ttk.Label(r1, text="Device:").pack(side="left")
        self.device_var = tk.StringVar(value=s.device)
        ttk.Radiobutton(r1, text="GPU", variable=self.device_var, value="cuda").pack(side="left", padx=(PAD_X, PAD_X))
        ttk.Radiobutton(r1, text="CPU", variable=self.device_var, value="cpu").pack(side="left")
        ttk.Separator(r1, orient="vertical").pack(side="left", fill="y", padx=PAD_X)
        ttk.Label(r1, text="Precision:").pack(side="left")
        self.fp16_var = tk.BooleanVar(value=s.fp16)
        ttk.Checkbutton(r1, text="FP16", variable=self.fp16_var).pack(side="left", padx=(PAD_X, 0))
        ttk.Separator(r1, orient="vertical").pack(side="left", fill="y", padx=PAD_X)
        ttk.Label(r1, text="Outscale:").pack(side="left")
        self.outscale_var = tk.DoubleVar(value=s.outscale)
        ttk.Spinbox(r1, from_=1.0, to=8.0, increment=0.5, width=5,
                    textvariable=self.outscale_var).pack(side="left", padx=(PAD_X, 0))

        # Row 2: batch, decode, prefetch
        r2 = ttk.Frame(self)
        r2.pack(fill="x", pady=(PAD_Y, PAD_Y))
        ttk.Label(r2, text="Batch:").pack(side="left")
        self.batch_var = tk.IntVar(value=s.batch_size)
        ttk.Spinbox(r2, from_=1, to=16, increment=1, width=4,
                    textvariable=self.batch_var).pack(side="left", padx=(PAD_X, PAD_X))
        ttk.Separator(r2, orient="vertical").pack(side="left", fill="y", padx=PAD_X)
        ttk.Label(r2, text="Decode:").pack(side="left")
        self.decode_var = tk.StringVar(value=s.decode)
        ttk.Combobox(r2, textvariable=self.decode_var, values=["cv2", "pyav"],
                     state="readonly", width=8).pack(side="left", padx=(PAD_X, PAD_X))
        ttk.Separator(r2, orient="vertical").pack(side="left", fill="y", padx=PAD_X)
        ttk.Label(r2, text="Prefetch:").pack(side="left")
        self.prefetch_var = tk.StringVar(value=s.prefetch)
        ttk.Combobox(r2, textvariable=self.prefetch_var, values=["sync", "async"],
                     state="readonly", width=8).pack(side="left", padx=(PAD_X, PAD_X))

        # Row 3: downscale pre-process
        r3 = ttk.Frame(self)
        r3.pack(fill="x")
        ttk.Label(r3, text="Pre-process:").pack(side="left")
        ttk.Label(r3, text="Downscale if max edge >").pack(side="left", padx=(PAD_X, PAD_X))
        self.downscale_var = tk.IntVar(value=s.downscale_max_edge)
        ttk.Spinbox(r3, from_=0, to=8192, increment=64, width=6,
                    textvariable=self.downscale_var).pack(side="left")
        ttk.Label(r3, text="px").pack(side="left", padx=(PAD_X, 0))

        # Row 4: GPU guard, tile size, tile overlap
        r4 = ttk.Frame(self)
        r4.pack(fill="x", pady=(PAD_Y, 0))
        ttk.Label(r4, text="GPU guard:").pack(side="left")
        self.gpu_guard_var = tk.StringVar(value=s.gpu_guard_mode)
        ttk.Combobox(r4, textvariable=self.gpu_guard_var,
                     values=["warn", "auto_downscale", "tiled", "off"],
                     state="readonly", width=15).pack(side="left", padx=(PAD_X, PAD_X))
        ttk.Label(r4, text="Tile:").pack(side="left")
        self.tile_size_var = tk.IntVar(value=s.tile_size)
        ttk.Spinbox(r4, from_=64, to=1024, increment=32, width=5,
                    textvariable=self.tile_size_var).pack(side="left", padx=(PAD_X, PAD_X))
        ttk.Label(r4, text="Overlap:").pack(side="left")
        self.tile_overlap_var = tk.IntVar(value=s.tile_overlap)
        ttk.Spinbox(r4, from_=0, to=128, increment=4, width=5,
                    textvariable=self.tile_overlap_var).pack(side="left", padx=(PAD_X, 0))

        # Row 5: TTA toggle (Phase 4). Slower; D4 group augmentation.
        r5 = ttk.Frame(self)
        r5.pack(fill="x", pady=(PAD_Y, 0))
        ttk.Label(r5, text="TTA:").pack(side="left")
        self.tta_var = tk.BooleanVar(value=s.tta)
        ttk.Checkbutton(r5, text="8x (slower, sharper; D4 group)",
                        variable=self.tta_var).pack(side="left", padx=(PAD_X, 0))

        # Row 5b: Performance (Phase 1 of the Real-time 4K plan).
        # TensorRT engine for inference (~2-4x speedup; auto-fallback when unavailable).
        # NVENC hardware encoder for video output (frees CPU; auto-fallback to libx264).
        r5b = ttk.Frame(self)
        r5b.pack(fill="x", pady=(PAD_Y, 0))
        ttk.Label(r5b, text="Acceleration:").pack(side="left")
        self.use_tensorrt_var = tk.BooleanVar(value=s.use_tensorrt)
        ttk.Checkbutton(r5b, text="TensorRT engine (~2-4x faster)",
                        variable=self.use_tensorrt_var).pack(side="left", padx=(PAD_X, PAD_X))
        self.use_nvenc_var = tk.BooleanVar(value=s.use_nvenc)
        ttk.Checkbutton(r5b, text="NVENC hardware encoder",
                        variable=self.use_nvenc_var).pack(side="left", padx=(0, PAD_X))
        self.nvenc_preset_var = tk.StringVar(value=s.nvenc_preset)
        ttk.Combobox(r5b, textvariable=self.nvenc_preset_var,
                     values=["p1", "p2", "p3", "p4"], state="readonly", width=4).pack(side="left")

        # Row 6: theme picker (3 radios + 3 swatches showing each theme's bg)
        r6 = ttk.Frame(self)
        r6.pack(fill="x", pady=(PAD_Y, 0))
        ttk.Label(r6, text="Theme:").pack(side="left")
        self.theme_var = tk.StringVar(value=s.theme)
        for name, label in [("light", "Light"), ("dark", "Dark"), ("high_contrast", "High contrast")]:
            rb = ttk.Radiobutton(r6, text=label, variable=self.theme_var, value=name,
                                 command=app._on_theme_change)
            rb.pack(side="left", padx=(PAD_X, PAD_X))
            # swatch: a small canvas showing the theme's bg color
            from ..theme import by_name
            sw = tk.Canvas(r6, width=14, height=14, highlightthickness=1,
                           bg=by_name(name).bg, bd=0)
            sw.pack(side="left", padx=(0, PAD_X))

        # Phase 10: <Return> on every spinbox saves settings.
        from ..a11y import bind_return
        for v in (self.outscale_var, self.batch_var, self.downscale_var,
                  self.tile_size_var, self.tile_overlap_var):  # noqa: kept (S) for parity
            # The spinboxes are children of Frames inside self; iterate all
            # spinbox descendants via the widget tree.
            pass
        # Walk the widget tree to find every ttk.Spinbox
        for child in self.winfo_children():
            for sub in self._walk_spinboxes(child):
                bind_return(sub, app._save_settings)

    def _walk_spinboxes(self, widget):
        """Yield every ttk.Spinbox descendant of `widget`."""
        from tkinter import ttk
        if isinstance(widget, ttk.Spinbox):
            yield widget
        for child in widget.winfo_children():
            yield from self._walk_spinboxes(child)
