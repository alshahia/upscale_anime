"""Phase 7: _confirm_destructive helper + menu bar wiring."""
import tkinter as tk
from tkinter import ttk
from unittest.mock import patch

import pytest


@pytest.fixture
def app(shared_tk_root):
    """Shared UpscaleGUI shell."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.app import UpscaleGUI
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths, _Settings
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.status_bar import _StatusBar
    paths = _AppPaths()
    s = _Settings(paths)
    a = UpscaleGUI.__new__(UpscaleGUI)
    if not getattr(shared_tk_root, "_aug_tk_used", False):
        tk.Tk.__init__(a)
        a.title("test")
        shared_tk_root._aug_tk_used = True
    else:
        a = tk.Toplevel(shared_tk_root)
        # Bind the methods the tests call
        for name in ("_show_about", "_on_theme_change", "_confirm_destructive",
                     "_reset_settings", "_bind_shortcuts", "_save_settings",
                     "_move_app_data", "_export_settings", "_import_settings",
                     "_open_settings_folder"):
            setattr(a, name, getattr(UpscaleGUI, name).__get__(a, type(a)))
    a.paths = paths
    a.settings = s
    a._resume_boxes = {}
    a.status_bar = _StatusBar(a, a)
    yield a
    try:
        a.destroy()
    except tk.TclError:
        pass


def test_confirm_destructive_returns_true_on_yes(app):
    with patch("apps.anime_upscaler_gui.anime_upscaler_gui.app.messagebox.askyesno", return_value=True) as m:
        assert app._confirm_destructive("T", "msg") is True
        m.assert_called_once_with("T", "msg")


def test_confirm_destructive_returns_false_on_no(app):
    with patch("apps.anime_upscaler_gui.anime_upscaler_gui.app.messagebox.askyesno", return_value=False):
        assert app._confirm_destructive("T", "msg") is False


def test_reset_settings_uses_confirm_destructive(app):
    """Resetting should be gated by the helper, not a raw askyesno."""
    called = {}
    def _fake_confirm(title, message):
        called["title"] = title
        called["message"] = message
        return False  # user said no
    app._confirm_destructive = _fake_confirm
    # Reset should not do anything (user said no)
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _Defaults
    before = app.settings.data.outscale
    app._reset_settings()
    assert called.get("title") == "Reset"
    assert app.settings.data.outscale == before


def test_menubar_built_with_5_cascades(app):
    """_MenuBar produces 5 top-level menus: File, Edit, Run, Tools, Help."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.menu_bar import _MenuBar
    # Stub the app methods the menu references
    for name in ("_add_files", "_save_settings", "_export_settings", "_import_settings",
                 "_on_close", "_clear_jobs", "_retry_selected", "_remove_selected",
                 "_reset_settings", "_on_start", "_refresh_model_dropdown",
                 "_open_settings_folder", "_move_app_data", "_show_about"):
        setattr(app, name, lambda: None)
    bar = _MenuBar(app, app)
    # A tk.Menu's `index("end")` returns the last index, or None if empty.
    # Iterate via .keys() and inspect each entry's type/label.
    end = bar.index("end")
    if end is None:
        labels = []
    else:
        labels = []
        for i in range(end + 1):
            t = bar.type(i)
            if t == "cascade":
                labels.append(bar.entrycget(i, "label"))
    assert "File" in labels, labels
    assert "Edit" in labels, labels
    assert "Run" in labels, labels
    assert "Tools" in labels, labels
    assert "Help" in labels, labels


def test_bind_shortcuts_registers_sequences(app):
    """All shortcuts in _bind_shortcuts are bound on the root."""
    for name in ("_add_files", "_save_settings", "_on_start", "_remove_selected",
                 "_refresh_model_dropdown", "_on_close", "_reset_settings", "_toggle_locale"):
        setattr(app, name, lambda: None)
    app._bind_shortcuts()
    sequences = ["<Control-o>", "<Control-s>", "<Control-Return>", "<Delete>",
                 "<F5>", "<Control-q>", "<Control-r>", "<Control-Shift-L>"]
    for seq in sequences:
        assert app.bind(seq) != "", f"{seq} not bound"


def test_messages_look_up_returns_english():
    from apps.anime_upscaler_gui.anime_upscaler_gui.messages import _
    # Known key
    assert "Tainted" in _("tainted_checkpoint_title")
    # printf-style placeholders
    out = _("worker_died_long", message="boom")
    assert "boom" in out
    out2 = _("diskspace_full")
    assert "disk" in out2.lower()


def test_messages_unknown_key_passes_through():
    """Phase 14 will fall back to a default; for now unknown keys echo back."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.messages import _
    assert _("never_seen_key") == "never_seen_key"


def test_pick_input_was_removed():
    """Phase 7 cleanup: the dead `_pick_input` is gone."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.app import UpscaleGUI
    assert not hasattr(UpscaleGUI, "_pick_input")
    assert not hasattr(UpscaleGUI, "_clear_input")
