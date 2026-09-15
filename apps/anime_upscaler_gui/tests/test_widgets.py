"""Headless instantiation tests for the widget subpackage.

Each widget is instantiated with a real Tk() root, then we verify that the
expected public attributes exist. Tests use try/finally to ensure the widget
is destroyed even if an assertion fails.

No `_build_ui` is called on the orchestrator, so we only need a minimal
`app` shim that exposes the attributes each widget reads at construction
time: `settings`, `registry`, and the `_*` callbacks the widgets bind to.
"""
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import pytest


@pytest.fixture
def root(shared_tk_root):
    """Module-scoped Tk root via the session-shared root in conftest.py."""
    return shared_tk_root


@pytest.fixture
def fake_settings(tmp_path):
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _Settings, _AppPaths, _Defaults
    app_dir = tmp_path / "aug"
    app_dir.mkdir(parents=True, exist_ok=True)
    paths = _AppPaths(app_dir=app_dir)
    settings = _Settings(paths)
    settings.data = _Defaults()
    settings.data.pretrained_dir = str(tmp_path / "empty_pretrained")
    return settings


@pytest.fixture
def fake_app(fake_settings, root, tmp_path):
    """Bare-minimum orchestrator shim: settings + registry + stubbed methods."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.registry import _ModelRegistry
    app = type("FakeApp", (), {})()
    app.settings = fake_settings
    app.registry = _ModelRegistry(Path(tmp_path) / "empty_pretrained")
    # Methods invoked by widget buttons / traces -- no-op stubs.
    app._add_files = lambda: None
    app._clear_jobs = lambda: None
    app._retry_selected = lambda: None
    app._remove_selected = lambda: None
    app._refresh_queue_listbox = lambda: None
    app._refresh_model_dropdown = lambda: None
    app._pick_output_dir = lambda: None
    app._download_preset = lambda: None
    app._download_custom_url = lambda: None
    app._import_local = lambda: None
    app._on_theme_change = lambda: None
    app._cancel_download = lambda: None
    app._save_settings = lambda: None
    app._jobs = []
    return app


def test_input_panel_creates_widgets(fake_app, root):
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets import _InputPanel
    panel = _InputPanel(tk.Frame(root), fake_app)
    try:
        assert isinstance(panel.queue_mode_var, tk.StringVar)
        assert isinstance(panel.input_var, tk.StringVar)
        assert isinstance(panel.filter_var, tk.StringVar)
        assert isinstance(panel.input_label, ttk.Label)
        assert isinstance(panel.queue_listbox, tk.Listbox)
    finally:
        panel.destroy()


def test_model_panel_creates_widgets(fake_app, root):
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets import _ModelPanel
    panel = _ModelPanel(tk.Frame(root), fake_app)
    try:
        assert isinstance(panel.model_var, tk.StringVar)
        assert isinstance(panel.model_dropdown, ttk.Combobox)
    finally:
        panel.destroy()


def test_settings_panel_creates_widgets(fake_app, root):
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets import _SettingsPanel
    panel = _SettingsPanel(tk.Frame(root), fake_app)
    try:
        for attr in ("device_var", "fp16_var", "outscale_var", "batch_var",
                     "decode_var", "prefetch_var", "downscale_var",
                     "gpu_guard_var", "tile_size_var", "tile_overlap_var"):
            assert hasattr(panel, attr), f"missing {attr}"
    finally:
        panel.destroy()


def test_output_panel_creates_widgets(fake_app, root):
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets import _OutputPanel
    panel = _OutputPanel(tk.Frame(root), fake_app)
    try:
        for attr in ("output_mode_var", "output_dir_var", "suffix_var", "batch_folder_var"):
            assert hasattr(panel, attr), f"missing {attr}"
    finally:
        panel.destroy()


def test_models_panel_creates_widgets(fake_app, root):
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets import _ModelsPanel
    panel = _ModelsPanel(tk.Frame(root), fake_app)
    try:
        for attr in ("preset_var", "preset_dropdown", "custom_url_var",
                     "models_listbox", "dl_progress_var", "dl_progress", "dl_status_var"):
            assert hasattr(panel, attr), f"missing {attr}"
    finally:
        panel.destroy()


def test_status_bar_creates_widgets(fake_app, root):
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets import _StatusBar
    panel = _StatusBar(tk.Frame(root), fake_app)
    try:
        assert isinstance(panel.progress_var, tk.DoubleVar)
        assert isinstance(panel.status_var, tk.StringVar)
    finally:
        panel.destroy()


def test_models_panel_refresh_listbox_empty(fake_app, root):
    """refresh_listbox should not crash when no models are installed."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets import _ModelsPanel
    panel = _ModelsPanel(tk.Frame(root), fake_app)
    try:
        panel.refresh_listbox()
        assert panel.models_listbox.size() == 0
    finally:
        panel.destroy()


def test_models_panel_refresh_presets_empty(fake_app, root):
    """refresh_presets should populate dropdown from registry.presets()."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets import _ModelsPanel
    panel = _ModelsPanel(tk.Frame(root), fake_app)
    try:
        panel.refresh_presets()
        items = list(panel.preset_dropdown["values"])
        # Hardcoded catalog ships presets even on a fresh install.
        assert len(items) > 0
    finally:
        panel.destroy()


def _is_packed(w):
    """True if widget is currently packed. pack_info() raises TclError when unpacked."""
    try:
        w.pack_info()
        return True
    except tk.TclError:
        return False


def test_input_panel_mode_change_toggles_widgets(fake_app, root):
    """Switching Single <-> Batch should re-pack the input_label / queue_listbox."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets import _InputPanel
    # Pretend we have jobs so the empty-state doesn't kick in.
    fake_app._jobs = [1]
    panel = _InputPanel(tk.Frame(root), fake_app)
    try:
        assert panel.queue_mode_var.get() == "single"
        assert _is_packed(panel.input_label)  # packed in single mode
        assert not _is_packed(panel.queue_listbox)  # not packed
        panel.queue_mode_var.set("batch")
        panel._on_mode_change()
        assert _is_packed(panel.queue_listbox)
        assert not _is_packed(panel.input_label)
        panel.queue_mode_var.set("single")
        panel._on_mode_change()
        assert _is_packed(panel.input_label)
        assert not _is_packed(panel.queue_listbox)
    finally:
        panel.destroy()
