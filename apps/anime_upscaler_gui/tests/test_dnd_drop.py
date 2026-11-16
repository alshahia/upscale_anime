"""M1 regression (PR-02): drag-and-drop must reach the real add-files path.

InputPanel._on_drop forwarded dropped paths as

    self.app._add_files(paths=accepted)

but UpscaleGUI._add_files took no arguments, so every drop raised TypeError
and no file was ever added. This drives the real _on_drop into the real
_add_files.
"""
import inspect
import types


def _drop(app, *paths):
    """Deliver a Tcl-list drop event; braces protect temp paths with spaces."""
    data = " ".join("{" + str(p) + "}" for p in paths)
    app.input_panel._on_drop(types.SimpleNamespace(data=data))


def test_drop_adds_jobs_via_add_files(gui_app, tmp_path):
    gui_app._build_ui()
    gui_app.input_panel.queue_mode_var.set("batch")
    a, b = tmp_path / "a.png", tmp_path / "b.png"

    _drop(gui_app, a, b)

    assert [j.input for j in gui_app._jobs] == [a, b]
    assert gui_app.input_panel.queue_listbox.size() == 2


def test_drop_ignores_unsupported_paths(gui_app, tmp_path):
    gui_app._build_ui()
    gui_app.input_panel.queue_mode_var.set("batch")

    _drop(gui_app, tmp_path / "notes.txt")

    assert gui_app._jobs == []


def test_add_files_signature_accepts_paths():
    """Pin the contract InputPanel relies on."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.app import UpscaleGUI

    assert "paths" in inspect.signature(UpscaleGUI._add_files).parameters
