"""M1 regression (PR-04): Pause/Cancel/Resume must act on the selected row.

_selected_job_id and _refresh_controls indexed self.app._jobs with the
listbox row index, but the listbox is filtered -- with filter "Running" and
row 0 selected, the controls targeted the first job overall instead of the
running one. Both now map through _visible_jobs().
"""
from apps.anime_upscaler_gui.anime_upscaler_gui.state import Job, JobStatus


def _seed(app, tmp_path):
    app._build_ui()
    app.input_panel.queue_mode_var.set("batch")
    done = Job(id=1, input=tmp_path / "a.png", output=tmp_path / "a_out.png",
               status=JobStatus.DONE)
    running = Job(id=2, input=tmp_path / "b.png", output=tmp_path / "b_out.png",
                  status=JobStatus.RUNNING)
    pending = Job(id=3, input=tmp_path / "c.png", output=tmp_path / "c_out.png",
                  status=JobStatus.PENDING)
    app._jobs.extend([done, running, pending])
    app._job_counter = 3
    app._refresh_queue_listbox()
    return done, running, pending


def _select_first(app):
    app.input_panel.queue_listbox.selection_clear(0, "end")
    app.input_panel.queue_listbox.selection_set(0)


def test_selected_id_follows_the_filter(gui_app, tmp_path):
    done, running, pending = _seed(gui_app, tmp_path)

    gui_app.input_panel.filter_var.set("Running")
    assert gui_app.input_panel.queue_listbox.size() == 1
    _select_first(gui_app)

    assert gui_app.input_panel._selected_job_id() == running.id
    assert gui_app.input_panel._selected_job_id() != done.id


def test_refresh_controls_targets_filtered_row(gui_app, tmp_path):
    _seed(gui_app, tmp_path)
    gui_app.input_panel.filter_var.set("Running")
    _select_first(gui_app)

    gui_app.input_panel._refresh_controls()

    # The selected job is RUNNING -> Pause and Cancel are live. Before the
    # fix the row mapped to the DONE first job, so both stayed disabled.
    assert str(gui_app.input_panel.pause_btn.cget("state")) == "normal"
    assert str(gui_app.input_panel.cancel_btn.cget("state")) == "normal"


def test_all_filter_selection_is_unaffected(gui_app, tmp_path):
    done, running, pending = _seed(gui_app, tmp_path)
    gui_app.input_panel.filter_var.set("All")
    assert gui_app.input_panel.queue_listbox.size() == 3

    gui_app.input_panel.queue_listbox.selection_clear(0, "end")
    gui_app.input_panel.queue_listbox.selection_set(2)

    assert gui_app.input_panel._selected_job_id() == pending.id
