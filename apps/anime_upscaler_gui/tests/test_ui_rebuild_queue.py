"""M1 regression (PR-03): rebuilding the UI must not drop the queue.

_on_theme_change / _toggle_locale / _reset_settings / _import_settings all
rebuilt the widget tree but only repopulated the model dropdown and the
models list. The queue listbox came back empty (the jobs survived in memory,
rows went to 0) and the live queue mode reverted to the last saved value. The
shared _rebuild_ui() helper now re-applies all of it.
"""
from apps.anime_upscaler_gui.anime_upscaler_gui.state import Job


def _seed(app, tmp_path, n=3):
    app._build_ui()
    app.input_panel.queue_mode_var.set("batch")
    for i in range(1, n + 1):
        app._jobs.append(Job(
            id=app._next_job_id(),
            input=tmp_path / ("img%d.png" % i),
            output=tmp_path / ("img%d_out.png" % i),
        ))
    app._refresh_queue_listbox()
    return list(app._jobs)


def test_rebuild_preserves_rows_and_mode(gui_app, tmp_path):
    jobs = _seed(gui_app, tmp_path)
    assert gui_app.input_panel.queue_listbox.size() == 3

    gui_app._rebuild_ui()

    assert gui_app.input_panel.queue_mode_var.get() == "batch"
    assert gui_app.input_panel.queue_listbox.size() == 3
    assert gui_app._jobs == jobs


def test_theme_change_preserves_queue(gui_app, tmp_path):
    _seed(gui_app, tmp_path)
    gui_app.settings_panel.theme_var.set("dark")

    gui_app._on_theme_change()

    assert gui_app.input_panel.queue_mode_var.get() == "batch"
    assert gui_app.input_panel.queue_listbox.size() == 3


def test_reset_repopulates_queue_from_defaults(gui_app, tmp_path):
    _seed(gui_app, tmp_path)
    gui_app._confirm_destructive = lambda *a, **k: True

    gui_app._reset_settings()

    # Reset lets defaults win for the mode, but the rows are still rebuilt:
    # the bug was rows going to 0, not the mode changing.
    assert gui_app.input_panel.queue_listbox.size() == 3
