"""Settings section: device, precision, outscale, batch, decode, prefetch,
downscale pre-process, GPU guard, tile size, tile overlap, TTA, acceleration.

Phase B1 of the GUI extensibility refactor: this widget is now data-
driven by widgets/settings_spec.py::SETTING_SPECS. To add a new user-
facing setting, append one entry there -- the panel, the RunJob builder
(Phase B2), and the tests all read from the same spec list.

Theme row is intentionally kept inline (it is special: it triggers a GUI
re-theme and renders 3 color swatches).
"""
import tkinter as tk
from tkinter import ttk

from ..ui_constants import GROUP_PAD, PAD_X, PAD_Y
from .settings_spec import SETTING_SPECS, SettingSpec, coerce_var


# Re-export for back-compat (Phase A1 contract): widget-package callers
# still import the underscore name; the public SettingsPanel is added
# in widgets/__init__.py via an alias.
class _SettingsPanel(ttk.LabelFrame):
    """Settings panel -- data-driven from SETTING_SPECS.

    Public attrs (preserved exactly so app.py callers continue to work):
      device_var, fp16_var, outscale_var, batch_var, decode_var,
      prefetch_var, downscale_var, gpu_guard_var, tile_size_var,
      tile_overlap_var, tta_var, use_tensorrt_var, use_nvenc_var,
      nvenc_preset_var, theme_var.
    """

    def __init__(self, parent, app):
        super().__init__(parent, text="Settings", padding=GROUP_PAD)
        self.app = app
        s = app.settings.data

        # One row per spec, packed top to bottom. Specs declare their own
        # widget kind; this loop handles all four kinds.
        prev_row = None
        for i, spec in enumerate(SETTING_SPECS):
            row = ttk.Frame(self)
            # pad: 0 below the first row, PAD_Y between, none below the last
            pady = (0, PAD_Y) if i == 0 else (PAD_Y, 0) if i == len(SETTING_SPECS) - 1 else (PAD_Y, PAD_Y)
            row.pack(fill="x", pady=pady)
            self._render_spec(row, spec, s)
            prev_row = row

        # Theme row: kept inline because it has GUI side effects and a
        # visual swatch per choice. Stored as the LAST child so the theme
        # test (panel.winfo_children()[-1]) keeps working.
        self._build_theme_row(app, s)

        # <Return> on every spinbox saves settings (Phase 10 a11y).
        from ..a11y import bind_return
        for sub in self._walk_spinboxes(self):
            bind_return(sub, app._save_settings)

    # -- per-spec rendering --------------------------------------------
    def _render_spec(self, row: ttk.Frame, spec: SettingSpec, s) -> None:
        """Render one SettingSpec into a single horizontal row.

        Layout:  [<label> <widget(s)>]  packed left-to-right.
        For checkbox kind, the label is prefixed to the checkbutton text.
        """
        if spec.widget == "checkbox":
            # One Checkbutton whose text contains the spec label.
            var = tk.BooleanVar(value=getattr(s, spec.key))
            cb = ttk.Checkbutton(row, text=spec.label, variable=var)
            cb.pack(side="left", padx=(0, PAD_X))
            setattr(self, spec.var_name, var)
            return

        ttk.Label(row, text=spec.label).pack(side="left", padx=(0, PAD_X))

        if spec.widget == "radio":
            var = tk.StringVar(value=getattr(s, spec.key))
            for value, label in spec.choices:
                ttk.Radiobutton(
                    row, text=label, variable=var, value=value,
                ).pack(side="left", padx=(0, PAD_X))
            setattr(self, spec.var_name, var)
            return

        if spec.widget == "combobox":
            var = tk.StringVar(value=getattr(s, spec.key))
            ttk.Combobox(
                row, textvariable=var,
                values=[lbl for _, lbl in spec.choices],
                state="readonly",
            ).pack(side="left", padx=(0, PAD_X))
            setattr(self, spec.var_name, var)
            return

        if spec.widget == "spinbox":
            from_, to_, inc = spec.spin
            # Pick var kind by whether the bounds are ints (the spinbox
            # widget itself requires matching numeric type for from_/to).
            is_int = all(isinstance(v, int) for v in (from_, to_, inc))
            var = tk.IntVar(value=int(getattr(s, spec.key))) if is_int else tk.DoubleVar(value=float(getattr(s, spec.key)))
            sb = ttk.Spinbox(
                row, from_=from_, to=to_, increment=inc,
                textvariable=var,
            )
            sb.pack(side="left", padx=(0, PAD_X))
            setattr(self, spec.var_name, var)
            return

        raise ValueError(f"unknown widget kind: {spec.widget!r}")

    # -- theme row (kept inline; not data-driven) ---------------------
    def _build_theme_row(self, app, s) -> None:
        from ..theme import by_name
        r = ttk.Frame(self)
        r.pack(fill="x", pady=(PAD_Y, 0))
        ttk.Label(r, text="Theme:").pack(side="left", padx=(0, PAD_X))
        self.theme_var = tk.StringVar(value=s.theme)
        for name, label in [("light", "Light"), ("dark", "Dark"), ("high_contrast", "High contrast")]:
            rb = ttk.Radiobutton(
                r, text=label, variable=self.theme_var, value=name,
                command=app._on_theme_change,
            )
            rb.pack(side="left", padx=(0, PAD_X))
            # swatch: a small canvas showing the theme's bg color
            sw = tk.Canvas(r, width=14, height=14, highlightthickness=1,
                           bg=by_name(name).bg, bd=0)
            sw.pack(side="left", padx=(0, PAD_X))

    # -- a11y helper kept for test_focus.py::_walk_spinboxes -----------
    def _walk_spinboxes(self, widget):
        """Yield every ttk.Spinbox descendant of `widget`."""
        if isinstance(widget, ttk.Spinbox):
            yield widget
        for child in widget.winfo_children():
            yield from self._walk_spinboxes(child)


# Public alias (Phase A1 contract): SettingsPanel == _SettingsPanel.
SettingsPanel = _SettingsPanel
