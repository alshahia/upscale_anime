"""Models section: installed-models listbox, preset dropdown, custom URL row,
download progress bar + status.

Phase 5 additions:
  - Installed list shows the preset display_name (falls back to filename).
    In-house training artifacts get a [TRAINED] prefix so the user can
    spot their own trained models among downloaded community weights.
  - Preset dropdown sections Trained models (top) and Community models with
    a '---' header divider, so the user can tell what ships with the app vs
    what needs a network download.
"""
import tkinter as tk
from tkinter import ttk

from ..registry import is_supported_kind
from ..ui_constants import FONT_MONO, GROUP_PAD, PAD_X, PAD_Y, TEXT_MUTED
from .empty_state import _EmptyState


_TRAINED_HEADER = "--- Trained / In-house ---"
_COMMUNITY_HEADER = "--- Community / Downloadable ---"


class _ModelsPanel(ttk.LabelFrame):
    def __init__(self, parent, app):
        super().__init__(parent, text="Models", padding=GROUP_PAD)
        self.app = app

        # Installed list row
        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Label(top, text=f"Installed in {app.registry.pretrained_dir}",
                  font=FONT_MONO, foreground=TEXT_MUTED).pack(side="left")
        ttk.Button(top, text="Refresh", command=self.refresh_listbox).pack(side="right")

        # Empty state (shown when no installed models; hides the listbox).
        self._empty = _EmptyState(
            self,
            icon=u"\u2193",
            title="No models installed",
            subtitle="Pick a preset below and click Download, or import a local .pth file.",
            cta_label="Download first preset",
            cta_command=self._download_first_preset,
        )
        self.models_listbox = tk.Listbox(self, height=5, font=FONT_MONO)

        # Preset dropdown
        r = ttk.Frame(self)
        r.pack(fill="x", pady=(PAD_Y, 0))
        ttk.Label(r, text="Preset:").pack(side="left")
        self.preset_var = tk.StringVar()
        self.preset_dropdown = ttk.Combobox(r, textvariable=self.preset_var,
                                            state="readonly", width=50)
        self.preset_dropdown.pack(side="left", padx=(PAD_X, PAD_X), fill="x", expand=True)
        ttk.Button(r, text="Download preset",
                   command=app._download_preset).pack(side="left", padx=(0, PAD_X))

        # Custom URL
        r2 = ttk.Frame(self)
        r2.pack(fill="x", pady=(PAD_Y, 0))
        ttk.Label(r2, text="Custom URL:").pack(side="left")
        self.custom_url_var = tk.StringVar()
        ttk.Entry(r2, textvariable=self.custom_url_var, font=FONT_MONO).pack(side="left", padx=(PAD_X, PAD_X), fill="x", expand=True)
        ttk.Button(r2, text="Download", command=app._download_custom_url).pack(side="left")
        ttk.Button(r2, text="Import local...", command=app._import_local).pack(side="left", padx=(PAD_X, 0))

        # Progress + status
        self.dl_progress_var = tk.DoubleVar(value=0.0)
        self.dl_progress = ttk.Progressbar(self, variable=self.dl_progress_var, maximum=100.0)
        self.dl_progress.pack(fill="x", pady=(PAD_Y, 0))
        self.dl_status_var = tk.StringVar(value="")
        self.dl_speed_var = tk.StringVar(value="")
        self.dl_eta_var = tk.StringVar(value="")
        status_row = ttk.Frame(self)
        status_row.pack(fill="x")
        ttk.Label(status_row, textvariable=self.dl_status_var, font=FONT_MONO,
                  foreground=TEXT_MUTED).pack(side="left")
        ttk.Label(status_row, textvariable=self.dl_speed_var, font=FONT_MONO,
                  foreground=TEXT_MUTED).pack(side="left", padx=(PAD_X, 0))
        ttk.Label(status_row, textvariable=self.dl_eta_var, font=FONT_MONO,
                  foreground=TEXT_MUTED).pack(side="right")
        # Cancel button (hidden by default; shown by app._start_download)
        self.dl_cancel_btn = ttk.Button(self, text="Cancel", command=app._cancel_download)
        # Not packed until a download starts

        self.refresh_listbox()
        self.refresh_presets()

    def _download_first_preset(self):
        """Download the first selectable preset; called from the empty-state CTA."""
        # Skip the section divider rows.
        items = [v for v in self.preset_dropdown["values"] if not v.startswith("---")]
        if items:
            self.preset_var.set(items[0])
        self.app._download_preset()

    def refresh_listbox(self):
        self.app.registry.scan_installed()
        self.models_listbox.delete(0, "end")
        for m in self.app.registry._installed:
            tag = ""
            if m.tainted:
                tag = "  [TAINTED]"
            elif not m.supported:
                tag = "  [unsupported]"
            elif m.is_trained:
                # In-house trained artifact. Show the friendly display_name
                # first (e.g. "RFDN Distill v1"); falls back to filename if the
                # registry does not have a display_name for this file.
                tag = "  [TRAINED]"
            label = m.display_name or m.filename
            self.models_listbox.insert(
                "end",
                f"{label}  ({m.kind}, {m.scale}x, {m.size_mb:.1f} MB){tag}",
            )
        # Toggle empty state vs listbox based on install count
        if self.app.registry._installed:
            self._empty.pack_forget()
            self.models_listbox.pack(fill="x", pady=(PAD_Y, PAD_Y), before=self.preset_dropdown.master)
        else:
            self.models_listbox.pack_forget()
            self._empty.pack(fill="x", pady=(PAD_Y, PAD_Y), before=self.preset_dropdown.master)

    def refresh_presets(self):
        items = []
        trained = [p for p in self.app.registry.presets() if p.source == "trained"]
        community = [p for p in self.app.registry.presets() if p.source != "trained"]
        if trained:
            items.append(_TRAINED_HEADER)
            for p in trained:
                tag = "  [unsupported]" if not is_supported_kind(p.kind) else ""
                label = p.display_name or p.id
                items.append(f"{label}  ({p.kind}, {p.scale}x, {p.size_mb} MB, {p.license}){tag}")
        if community:
            items.append(_COMMUNITY_HEADER)
            for p in community:
                tag = "  [unsupported]" if not is_supported_kind(p.kind) else ""
                label = p.display_name or p.id
                items.append(f"{label}  ({p.kind}, {p.scale}x, {p.size_mb} MB, {p.license}){tag}")
        self.preset_dropdown["values"] = items
        # Default to the first selectable preset (trained, if available; else community).
        if items and not self.preset_var.get():
            for v in items:
                if not v.startswith("---"):
                    self.preset_var.set(v)
                    break
