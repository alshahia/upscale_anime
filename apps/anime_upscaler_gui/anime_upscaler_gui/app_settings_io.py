"""Settings persistence ceremony for the GUI (mixin extracted from app.py).

Owns: reading current form values into settings, theme change handling,
reset, open-folder, export/import, moving app data, and the close sequence.

Mixin contract: instances provide the collaborator attributes used here
(settings, settings_panel, output_panel, input_panel, paths, queue_ctrl,
registry, status_bar, models_panel, model_panel, _build_ui, _refresh_model_dropdown,
_refresh_queue_listbox, _confirm_destructive, _worker). All coordination still
happens through the orchestrator (UpscaleGUI).
"""
import os
import sys
from pathlib import Path
from tkinter import filedialog, messagebox

from .registry import _ModelRegistry
from .theme import apply as _apply_theme, by_name as _theme_by_name
from .errors import humanize as _humanize_error


class SettingsIOMixin:
    """Save/apply/reset/export/import settings + app-data relocation."""

    def _save_settings(self):
        s = self.settings.data
        sp = self.settings_panel
        op = self.output_panel
        s.device = sp.device_var.get()
        s.fp16 = bool(sp.fp16_var.get())
        s.outscale = float(sp.outscale_var.get())
        s.batch_size = int(sp.batch_var.get())
        s.decode = sp.decode_var.get()
        s.prefetch = sp.prefetch_var.get()
        s.downscale_max_edge = int(sp.downscale_var.get())
        s.output_mode = op.output_mode_var.get()
        s.output_dir = op.output_dir_var.get()
        s.single_file_suffix = op.suffix_var.get()
        s.batch_folder_name = op.batch_folder_var.get() or "anime_upscaler_gui"
        s.queue_mode = self.input_panel.queue_mode_var.get()
        s.theme = sp.theme_var.get()
        s.tta = bool(sp.tta_var.get())
        s.use_tensorrt = bool(sp.use_tensorrt_var.get())
        s.use_nvenc = bool(sp.use_nvenc_var.get())
        s.nvenc_preset = sp.nvenc_preset_var.get()
        s.nvenc_qp = int(sp.nvenc_qp_var.get())
        s.cascade_mode = getattr(s, "cascade_mode", None)  # None=auto-from-model-scale
        sel = self.model_panel.model_dropdown.get()
        if sel:
            s.last_model = sel.split()[0]
        self.settings.save()
        self.status_bar.status_var.set("Settings saved.")

    def _on_theme_change(self):
        """User picked a new theme in the SettingsPanel radio row.

        Apply the theme to the ui_constants module, then rebuild the UI so
        every widget reflects the new palette. Persist the choice so the
        next launch opens with the same theme.
        """
        name = self.settings_panel.theme_var.get()
        try:
            _apply_theme(_theme_by_name(name))
        except KeyError as e:
            messagebox.showerror("Unknown theme", str(e))
            return
        self.settings.data.theme = name
        self.settings.save()
        self._build_ui()
        self._refresh_model_dropdown()
        self.models_panel.refresh_listbox()

    def _reset_settings(self):
        if not self._confirm_destructive("Reset", "Reset all settings to defaults?"):
            return
        from .settings import _Defaults
        self.settings.data = _Defaults()
        self.settings.save()
        self._build_ui()  # simplest: rebuild the UI from defaults

    def _open_settings_folder(self):
        path = self.paths.app_dir
        path.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(str(path))  # noqa
            elif sys.platform == "darwin":
                os.system(f"open '{path}'")
            else:
                os.system(f"xdg-open '{path}'")
        except Exception as e:
            messagebox.showerror("Cannot open", str(e))

    def _export_settings(self):
        path = filedialog.asksaveasfilename(
            title="Export settings to...",
            defaultextension=".json",
            initialfile="anime_upscaler_settings.json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            self.settings.export(Path(path))
            self.status_bar.status_var.set(f"Settings exported to {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def _import_settings(self):
        path = filedialog.askopenfilename(
            title="Import settings from...",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
        )
        if not path:
            return
        if not messagebox.askyesno(
            "Import settings",
            f"Replace current settings with the contents of:\n{path}\n\n"
            f"Unknown fields will be ignored; missing fields keep their current values.",
        ):
            return
        try:
            self.settings.import_file(Path(path))
            self._build_ui()  # refresh widget values from new settings
            self.status_bar.status_var.set(f"Settings imported from {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Import failed", _humanize_error(e))

    def _move_app_data(self):
        new_dir = filedialog.askdirectory(title="Move app data to...",
                                          initialdir=str(self.paths.app_dir))
        if not new_dir:
            return
        new_dir = Path(new_dir)
        if new_dir.resolve() == self.paths.app_dir.resolve():
            return
        if not messagebox.askyesno(
            "Move app data",
            f"Move all app data to:\n{new_dir}\n\n"
            f"Files at the old location ({self.paths.app_dir}) will be removed.",
        ):
            return
        try:
            self.paths.move_to(new_dir)
        except Exception as e:
            messagebox.showerror("Move failed", str(e))
            return
        # Update settings to reflect the new location; re-create paths/registry.
        self.settings.data.app_dir = str(new_dir)
        self.settings.save()
        self.registry = _ModelRegistry(self.settings.pretrained_dir_path(),
                                       presets_path=self.paths.registry_path)
        self._refresh_model_dropdown()
        self.models_panel.refresh_listbox()
        self.status_bar.status_var.set(f"App data moved to: {new_dir}")

    def _on_close(self):
        try:
            self._save_settings()
        except Exception:
            pass
        try:
            self._worker.stop()
            self._worker.join(timeout=2)
        except Exception:
            pass
        self.destroy()
