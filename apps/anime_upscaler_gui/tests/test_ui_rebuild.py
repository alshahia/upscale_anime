"""M1 regression: _build_ui() must be idempotent.

_on_theme_change / _reset_settings / _import_settings / _toggle_locale all
rebuild the whole UI tree on an already-built window. _build_ui used to pack
a fresh outer frame without tearing the old one down, and since `pack` with
side=top appends, N rebuilds left N stacked copies of the UI. The outer
frame is now kept in `self._ui_frame` and destroyed before the next build.
"""
import queue

import tkinter as tk
from pathlib import Path
from tkinter import ttk

import pytest


@pytest.fixture
def app(shared_tk_root, tmp_path):
    """UpscaleGUI shell with enough surface for a real _build_ui()."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.app import UpscaleGUI
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import (
        _AppPaths, _Settings, _Defaults)
    from apps.anime_upscaler_gui.anime_upscaler_gui.registry import _ModelRegistry

    app_dir = tmp_path / "aug"
    app_dir.mkdir(parents=True, exist_ok=True)
    paths = _AppPaths(app_dir=app_dir)
    settings = _Settings(paths)
    settings.data = _Defaults()
    settings.data.pretrained_dir = str(tmp_path / "empty_pretrained")

    a = tk.Toplevel(shared_tk_root)
    for name in ("_build_ui", "_rebuild_ui", "_on_theme_change",
                 "_refresh_model_dropdown", "_make_scrollable",
                 "_build_preview_section", "_build_actions_section"):
        setattr(a, name, getattr(UpscaleGUI, name).__get__(a, type(a)))
    a.paths = paths
    a.settings = settings
    a.registry = _ModelRegistry(Path(tmp_path) / "empty_pretrained")
    a._jobs = []
    a._log_handler = None
    a._resume_boxes = {}
    a._ctl_queue = queue.Queue()
    for name in ("_add_files", "_clear_jobs", "_retry_selected", "_on_start",
                 "_remove_selected", "_refresh_queue_listbox",
                 "_pick_output_dir", "_download_preset", "_download_custom_url",
                 "_import_local", "_save_settings", "_cancel_download",
                 "_confirm_destructive", "_reset_settings", "_open_settings_folder",
                 "_export_settings", "_import_settings", "_move_app_data",
                 "_show_about", "_bind_shortcuts"):
        setattr(a, name, lambda *args, **kwargs: None)
    yield a
    # The handler is attached to the root logger by attach_panel(); point it
    # at nothing so it never forwards to this test's destroyed widgets.
    handler = a.__dict__.get("_log_handler")
    if handler is not None:
        try:
            handler.set_target(None)
        except Exception:
            pass
    monitor = a.__dict__.get("gpu_monitor")
    if monitor is not None:
        try:
            monitor.stop()
        except Exception:
            pass
    try:
        a.destroy()
    except tk.TclError:
        pass


def _ui_children(app):
    """Direct children of the window that the UI tree owns."""
    return [w for w in app.winfo_children()
            if isinstance(w, (tk.Frame, ttk.Frame))]


def test_build_ui_twice_leaves_one_tree(app):
    app._build_ui()
    first = app._ui_frame
    panels = app.input_panel, app.settings_panel, app.output_panel
    assert len(_ui_children(app)) == 1

    app._build_ui()  # the second rebuild

    # The previous tree is gone: same count, new outer frame, new panels.
    assert first.winfo_exists() == 0, "old outer frame was not destroyed"
    assert app._ui_frame is not first
    assert len(_ui_children(app)) == 1
    assert all(p.winfo_exists() == 0 for p in panels), (
        "panels from the first build were leaked"
    )
    assert app.input_panel is not panels[0]


def test_repeated_build_ui_does_not_stack(app):
    for _ in range(5):
        app._build_ui()
        assert len(_ui_children(app)) == 1


def test_on_theme_change_rebuilds_in_place(app):
    """The real-world trigger: theme changes rebuild the UI N times."""
    app._build_ui()
    frames = [app._ui_frame]
    log_panels = [app.log_panel]

    for theme in ("light", "dark", "high_contrast"):
        # What the SettingsPanel radio row hands to _on_theme_change.
        app.settings_panel.theme_var.set(theme)
        app._on_theme_change()
        frames.append(app._ui_frame)
        log_panels.append(app.log_panel)

    # Every build but the last was torn down.
    for f in frames[:-1]:
        assert f.winfo_exists() == 0, "a rebuild left the previous UI tree alive"
    assert frames[-1].winfo_exists() == 1
    assert len(set(frames)) == 4
    assert len(_ui_children(app)) == 1
    # The log handler was re-pointed at the fresh panel, not a dead one.
    assert app._log_handler._target is app.log_panel
    assert log_panels[-1] is not log_panels[0]
    assert all(p.winfo_exists() == 0 for p in log_panels[:-1]), (
        "log panels from earlier builds are still alive"
    )
