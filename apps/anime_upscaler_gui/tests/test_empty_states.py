"""Phase 8: empty/error states + download progress + settings corruption."""
import json
import os
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from unittest.mock import patch

import pytest


# ---- eta formatter ----

def test_eta_seconds():
    from apps.anime_upscaler_gui.anime_upscaler_gui.eta import format_eta
    assert format_eta(5) == "5s"
    assert format_eta(45) == "45s"
    assert format_eta(59) == "59s"


def test_eta_minutes_seconds():
    from apps.anime_upscaler_gui.anime_upscaler_gui.eta import format_eta
    assert format_eta(60) == "1:00"
    assert format_eta(75) == "1:15"
    assert format_eta(3599) == "59:59"


def test_eta_hours():
    from apps.anime_upscaler_gui.anime_upscaler_gui.eta import format_eta
    assert format_eta(3600) == "1:00:00"
    assert format_eta(3661) == "1:01:01"
    assert format_eta(7325) == "2:02:05"


def test_eta_handles_bad_input():
    from apps.anime_upscaler_gui.anime_upscaler_gui.eta import format_eta
    assert format_eta(-1) == "—"
    assert format_eta(None) == "—"
    # NaN: `float('nan') != float('nan')`
    assert format_eta(float("nan")) == "—"


def test_format_speed():
    from apps.anime_upscaler_gui.anime_upscaler_gui.eta import format_speed
    assert "MB/s" in format_speed(2.5)
    assert "KB/s" in format_speed(0.5)
    assert format_speed(-1) == "—"
    assert format_speed(None) == "—"


# ---- settings corruption flag ----

def test_settings_corrupt_flag_set_on_bad_json(tmp_path):
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths, _Settings
    d = tmp_path / "appdir"
    paths = _AppPaths(d)
    paths.settings_path.write_text("{not valid json")
    s = _Settings(paths)
    assert s.corrupt is True
    assert paths.settings_path.with_suffix(".bak").exists()


def test_settings_corrupt_flag_false_on_good_json(tmp_path):
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths, _Settings
    d = tmp_path / "appdir"
    paths = _AppPaths(d)
    paths.settings_path.write_text(json.dumps({"outscale": 3.0}))
    s = _Settings(paths)
    assert s.corrupt is False


def test_settings_corrupt_flag_false_on_no_file(tmp_path):
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths, _Settings
    d = tmp_path / "appdir"
    paths = _AppPaths(d)
    if paths.settings_path.exists():
        paths.settings_path.unlink()
    s = _Settings(paths)
    assert s.corrupt is False


# ---- empty state widget ----

@pytest.fixture
def root(shared_tk_root):
    return shared_tk_root


def test_empty_state_constructs(root):
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.empty_state import _EmptyState
    captured = []
    es = _EmptyState(root, icon="X", title="No files", subtitle="Add one",
                     cta_label="Add", cta_command=lambda: captured.append("clicked"))
    assert es.winfo_exists()


def test_empty_state_set_title_and_subtitle(root):
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.empty_state import _EmptyState
    es = _EmptyState(root, icon="!", title="orig", subtitle="orig sub")
    es.set_title("new")
    es.set_subtitle("new sub")
    children = es.winfo_children()
    labels = [c for c in children if isinstance(c, tk.Label)]
    titles = [c.cget("text") for c in labels if c.cget("text") in ("new", "new sub")]
    assert "new" in titles
    assert "new sub" in titles


def test_empty_state_cta_invokes_command(root):
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.empty_state import _EmptyState
    captured = []
    es = _EmptyState(root, icon="!", title="t", subtitle="s",
                     cta_label="Go", cta_command=lambda: captured.append(1))
    # Find the button and invoke it
    btns = [c for c in es.winfo_children() if isinstance(c, ttk.Button)]
    assert btns, "no CTA button"
    btns[0].invoke()
    assert captured == [1]


def test_empty_state_no_cta_when_label_missing(root):
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.empty_state import _EmptyState
    es = _EmptyState(root, icon="!", title="t", subtitle="s")
    btns = [c for c in es.winfo_children() if isinstance(c, ttk.Button)]
    assert btns == []


# ---- input panel empty state integration ----

def test_input_panel_shows_empty_state_in_batch_mode(tmp_path, root):
    from apps.anime_upscaler_gui.anime_upscaler_gui.app import UpscaleGUI
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths, _Settings
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.input_panel import _InputPanel
    paths = _AppPaths(tmp_path)
    s = _Settings(paths)
    app = UpscaleGUI.__new__(UpscaleGUI)
    # minimal stubs
    tk.Tk.__init__(app)
    app.title("t")
    app.paths = paths
    app.settings = s
    app._jobs = []
    app._add_files = lambda: None
    app._clear_jobs = lambda: None
    app._retry_selected = lambda: None
    app._remove_selected = lambda: None
    app._refresh_queue_listbox = lambda: None
    try:
        panel = _InputPanel(root, app)
        panel.queue_mode_var.set("batch")
        panel._on_mode_change()
        panel._refresh_empty()
        root.update_idletasks()
        from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.empty_state import _EmptyState
        empty_children = [c for c in panel.winfo_children() if isinstance(c, _EmptyState)]
        assert empty_children, "empty state not created"
        # `pack_info()` returns a dict if packed, raises TclError otherwise.
        try:
            empty_children[0].pack_info()
        except tk.TclError:
            raise AssertionError("empty state was not packed after refresh_empty")
        # And the listbox should NOT be packed (since jobs is empty).
        try:
            panel.queue_listbox.pack_info()
            raise AssertionError("listbox should be hidden when empty")
        except tk.TclError:
            pass  # expected — listbox is pack_forget'd
    finally:
        app.destroy()


# ---- pre-flight output folder check ----

def test_preflight_rejects_unwritable_output(tmp_path, root):
    """If the output dir can't be created, pre-flight returns False."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.app import UpscaleGUI
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths, _Settings
    from apps.anime_upscaler_gui.anime_upscaler_gui.state import Job
    paths = _AppPaths(tmp_path)
    s = _Settings(paths)
    app = UpscaleGUI.__new__(UpscaleGUI)
    tk.Tk.__init__(app)
    app.title("t")
    app.paths = paths
    app.settings = s
    app.settings_panel = type("SP", (), {"device_var": tk.StringVar(value="cpu")})()
    try:
        # An output path under a read-only directory on Windows can be created
        # via a non-existent parent whose ancestor is a file. Easiest: an
        # OSError-raising mkdir mock.
        from unittest.mock import MagicMock
        from apps.anime_upscaler_gui.anime_upscaler_gui.state import JobStatus
        bad = Job(id=1, input=tmp_path / "in.png",
                  output=tmp_path / "nonexistent_ancestor" / "out.png")
        with patch("pathlib.Path.mkdir", side_effect=OSError(13, "perm denied")):
            with patch("apps.anime_upscaler_gui.anime_upscaler_gui.app.messagebox.showerror") as m:
                ok = app._preflight_check([bad])
        assert ok is False
        assert m.called
    finally:
        app.destroy()
