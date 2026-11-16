"""Task 10: pause must block again after a resume; output widgets must be live.

The real-worker tests drive the real PipelineWorker thread and a real ffmpeg
encoder. Decode and model are fakes (no checkpoint needed); the frame stream is
rate-limited so pause/resume windows are observable end-to-end.
"""
from __future__ import annotations

import queue
import time

import numpy as np
import pytest

from apps.anime_upscaler_gui.anime_upscaler_gui import ffmpeg
from apps.anime_upscaler_gui.anime_upscaler_gui.app import UpscaleGUI
from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline import worker as worker_mod
from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline.jobs import (
    JobControlEvent,
    RunJob,
)
from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline.worker import PipelineWorker
from apps.anime_upscaler_gui.anime_upscaler_gui.state import Job, JobStatus


FRAME = 32
JOB_ID = 101
TERMINALS = ("finished", "error", "cancelled", "fatal_error")

pytestmark = pytest.mark.skipif(
    not ffmpeg.ffmpeg_available(),
    reason="real encoder required for the end-to-end pause tests",
)


@pytest.fixture(autouse=True)
def _restore_gpu_probe_cache():
    """Do not leak the CPU-faked GPU-probe result into the rest of the suite.

    The real-worker tests below force torch.cuda.is_available() to False; the
    worker then caches the "no GPU" probe result in worker._GPU_UTIL_PROBE for
    the whole pytest process, so later telemetry tests would read that empty
    cache instead of probing the real device. Restore the cache after each test.
    """
    yield
    worker_mod._GPU_UTIL_PROBE = None


# --------------------------------------------------------------------------- #
# 10.1 pause state machine (unit)
# --------------------------------------------------------------------------- #

def _worker_stub():
    w = object.__new__(PipelineWorker)
    w._pause_events = {}
    w.ctl_queue = queue.Queue()
    w.emit = lambda evt: None
    return w


def test_second_pause_clears_the_event():
    """Brief 10.1: pause -> resume -> pause must leave the event unset.

    The brief calls the control event JobControl; the real type is
    JobControlEvent (pipeline/jobs.py), used here.
    """
    w = _worker_stub()
    for kind in ("pause", "resume", "pause"):
        w.ctl_queue.put(JobControlEvent(job_id=7, kind=kind))
        w._drain_ctl_for_job(7)
    assert w._pause_events[7].is_set() is False


def test_resume_after_pause_sets_the_event():
    """Pin the half of the contract that already worked: resume wakes the job."""
    w = _worker_stub()
    w.ctl_queue.put(JobControlEvent(job_id=7, kind="pause"))
    w._drain_ctl_for_job(7)
    assert w._pause_events[7].is_set() is False
    w.ctl_queue.put(JobControlEvent(job_id=7, kind="resume"))
    w._drain_ctl_for_job(7)
    assert w._pause_events[7].is_set() is True


# --------------------------------------------------------------------------- #
# Real-worker harness: real PipelineWorker thread + real ffmpeg encoder.
# --------------------------------------------------------------------------- #

class _SlowReader:
    """Decoder stand-in that yields a fixed number of frames at a slow pace."""

    def __init__(self, n_frames):
        self._n = n_frames
        self.fps = 25.0
        self.total = n_frames
        self.w = self.h = FRAME
        self.release_calls = 0

    def __iter__(self):
        for _ in range(self._n):
            time.sleep(0.03)
            yield np.zeros((FRAME, FRAME, 3), dtype=np.uint8)

    def release(self):
        self.release_calls += 1


class _Model:
    """4x stand-in model; the pipeline only needs it to upsample tensors."""

    scale = 4

    def to(self, *_a, **_k):
        return self

    def eval(self):
        return self

    def half(self):
        return self

    def __call__(self, x):
        return x.repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2)


class _Backend:
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        return self.model(x)


def _install(monkeypatch, n_frames):
    """Fake decode/model; everything else (encoder, worker thread) is real."""
    reader = _SlowReader(n_frames)
    monkeypatch.setattr(worker_mod, "_PyAvReader", lambda p: reader)
    model = _Model()
    monkeypatch.setattr(worker_mod, "build", lambda kind, ckpt: model)
    monkeypatch.setattr(worker_mod, "_make_backend",
                        lambda *a, **k: (_Backend(model), "PyTorch"))
    monkeypatch.setattr(worker_mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(worker_mod, "_mux_job_audio", lambda *a, **k: True)
    return reader


def _job(tmp_path, job_id=JOB_ID, output_name="out.mp4"):
    inp = tmp_path / "input.mp4"
    inp.write_bytes(b"decoder is faked; only existence is checked")
    return RunJob(
        job_id=job_id,
        input_path=inp,
        output_path=tmp_path / output_name,
        is_video=True,
        model_filename="dummy.pth",
        kind="realsrgan",
        scale=4,
        outscale=4.0,
        fp16=False,
        device="cpu",
        batch_size=1,
        decode="pyav",
        prefetch="sync",
        pin_memory="off",
        downscale_max_edge=0,
        gpu_guard_mode="off",
        tile_size=0,
        tile_overlap=0,
        tta=False,
        use_tensorrt=False,
        use_nvenc=False,
        video_preset="ultrafast",
    )


class _Monitor:
    """Collect worker events and wait for predicates without busy-looping."""

    def __init__(self, out_queue):
        self.q = out_queue
        self.events = []
        self._cursor = 0

    def drain(self, seconds=0.0):
        end = time.monotonic() + seconds
        while True:
            remaining = end - time.monotonic()
            try:
                self.events.append(self.q.get(timeout=max(0.0, remaining)))
            except queue.Empty:
                return self.events

    def wait(self, pred, timeout=10.0):
        end = time.monotonic() + timeout
        while True:
            while self._cursor < len(self.events):
                evt = self.events[self._cursor]
                self._cursor += 1
                if pred(evt):
                    return evt
            remaining = end - time.monotonic()
            if remaining <= 0:
                return None
            try:
                self.events.append(self.q.get(timeout=min(0.05, remaining)))
            except queue.Empty:
                pass

    def terminal_kinds(self):
        return [e.kind for e in self.events if e.kind in TERMINALS]

    def max_progress_frame(self):
        frames = [e.frame_idx for e in self.events if e.kind == "progress"]
        return max(frames, default=-1)


def _start(monkeypatch, tmp_path, n_frames=60):
    _install(monkeypatch, n_frames)
    job = _job(tmp_path)
    in_q: "queue.Queue" = queue.Queue()
    out_q: "queue.Queue" = queue.Queue()
    ctl_q: "queue.Queue" = queue.Queue()
    worker = PipelineWorker(in_q, out_q, ctl_queue=ctl_q)
    worker.start()
    in_q.put(job)
    return worker, job, in_q, out_q, ctl_q


def _stop(worker, ctl_q):
    """Cancel-then-stop so a worker blocked in _check_pause() can exit."""
    try:
        ctl_q.put(JobControlEvent(kind="cancel", job_id=JOB_ID))
    except Exception:
        pass
    worker.stop()
    worker.join(timeout=15)


def _wait_running(monitor):
    assert monitor.wait(lambda e: e.kind == "started", 10) is not None, (
        "worker never started the job")
    assert monitor.wait(lambda e: e.kind == "progress", 10) is not None, (
        "no progress event before the pause")


def test_paused_job_stops_advancing_and_resume_completes(tmp_path, monkeypatch):
    """Start a real job, pause it, prove it stops, resume, prove it finishes."""
    worker, job, _in_q, out_q, ctl_q = _start(monkeypatch, tmp_path)
    mon = _Monitor(out_q)
    try:
        _wait_running(mon)
        ctl_q.put(JobControlEvent(kind="pause", job_id=JOB_ID))
        blocked = mon.wait(
            lambda e: e.kind == "log" and "waiting on resume" in e.message, 10)
        assert blocked is not None, "worker never blocked on the pause"
        mon.drain(0.3)
        snapshot = mon.max_progress_frame()
        mon.drain(0.6)
        assert mon.max_progress_frame() == snapshot, (
            "pipeline advanced while paused (frame %d -> %d)"
            % (snapshot, mon.max_progress_frame()))
        assert not mon.terminal_kinds(), (
            "terminal event while paused: %r" % mon.terminal_kinds())
        ctl_q.put(JobControlEvent(kind="resume", job_id=JOB_ID))
        fin = mon.wait(lambda e: e.kind in TERMINALS, 30)
        assert fin is not None and fin.kind == "finished", (
            "resume did not finish the job: %r" % mon.terminal_kinds())
        assert job.output_path.exists(), "finished job left no output"
    finally:
        _stop(worker, ctl_q)


def test_second_pause_after_resume_blocks_again(tmp_path, monkeypatch):
    """Task 10 regression: pause -> resume -> pause must block the pipeline again.

    Before the fix the pause branch only created the event (setdefault); the
    resume left it set, so the second pause was a no-op. The GUI showed the job
    as paused while the pipeline kept encoding to completion.
    """
    worker, _job_obj, _in_q, out_q, ctl_q = _start(monkeypatch, tmp_path)
    mon = _Monitor(out_q)
    try:
        _wait_running(mon)
        ctl_q.put(JobControlEvent(kind="pause", job_id=JOB_ID))
        first = mon.wait(
            lambda e: e.kind == "log" and "waiting on resume" in e.message, 10)
        assert first is not None, "first pause did not block"
        # Resume and immediately pause again (the GUI can do this in one click
        # sequence: Pause -> Resume -> Pause).
        ctl_q.put(JobControlEvent(kind="resume", job_id=JOB_ID))
        ctl_q.put(JobControlEvent(kind="pause", job_id=JOB_ID))
        mon.drain(0.6)
        snapshot = mon.max_progress_frame()
        terminals = mon.terminal_kinds()
        assert not terminals, (
            "second pause did not block: the job reached %r"
            % terminals)
        mon.drain(1.0)
        assert mon.max_progress_frame() == snapshot, (
            "pipeline advanced after the second pause (frame %d -> %d)"
            % (snapshot, mon.max_progress_frame()))
        assert not mon.terminal_kinds(), (
            "terminal event after the second pause: %r"
            % mon.terminal_kinds())
        ctl_q.put(JobControlEvent(kind="resume", job_id=JOB_ID))
        fin = mon.wait(lambda e: e.kind in TERMINALS, 30)
        assert fin is not None and fin.kind == "finished", (
            "final resume did not finish the job: %r" % mon.terminal_kinds())
    finally:
        _stop(worker, ctl_q)


def test_abort_while_paused_is_a_single_cancelled(tmp_path, monkeypatch):
    """Cancel while paused must unwind cleanly: one cancelled, no finished."""
    worker, job, in_q, out_q, ctl_q = _start(monkeypatch, tmp_path, n_frames=40)
    mon = _Monitor(out_q)
    try:
        _wait_running(mon)
        ctl_q.put(JobControlEvent(kind="pause", job_id=JOB_ID))
        assert mon.wait(
            lambda e: e.kind == "log" and "waiting on resume" in e.message,
            10) is not None, "pause did not block before the abort"
        ctl_q.put(JobControlEvent(kind="cancel", job_id=JOB_ID))
        term = mon.wait(lambda e: e.kind in TERMINALS, 20)
        assert term is not None and term.kind == "cancelled", (
            "abort while paused did not produce a cancelled terminal: %r"
            % mon.terminal_kinds())
        mon.drain(0.4)
        assert mon.terminal_kinds() == ["cancelled"], (
            "expected exactly one terminal (cancelled), got %r"
            % mon.terminal_kinds())
        assert not job.output_path.exists(), (
            "abort while paused left the partial output behind")
        # The worker must survive the paused abort and run the next job.
        job2 = _job(tmp_path, job_id=JOB_ID + 1, output_name="out2.mp4")
        in_q.put(job2)
        fin = mon.wait(
            lambda e: e.job_id == JOB_ID + 1 and e.kind in TERMINALS, 30)
        assert fin is not None and fin.kind == "finished", (
            "worker did not run the next job after a paused abort: %r"
            % [e.kind for e in mon.events if e.job_id == JOB_ID + 1])
        assert job2.output_path.exists()
    finally:
        _stop(worker, ctl_q)


# --------------------------------------------------------------------------- #
# 10.4 live output settings: widgets must not be re-read from stale settings
# --------------------------------------------------------------------------- #

def _bind_real_methods(app):
    """Bind the Task 10 hook (and friends) onto the conftest Toplevel stand-in.

    The hook does not exist before the fix; the behavioral assertions below
    then observe the stale settings.data directly (no bind is needed for that).
    """
    for name in (
        "_on_output_setting_changed",
        "_on_queue_selection_changed",
        "_rebuild_ui",
        "_find_job",
        "_send_pause",
        "_send_resume",
        "_remove_selected",
    ):
        try:
            setattr(app, name, getattr(UpscaleGUI, name).__get__(app, type(app)))
        except AttributeError:
            pass


def _click_row(app, row):
    """Select a queue row the way a user does, and deliver the Tk event.

    Task 10b: tests must not poke _refresh_controls directly -- a real user
    click produces <<ListboxSelect>>, and the app must react to that event.
    selection_set alone is not enough (the toolbar used to stay disabled).
    """
    lb = app.input_panel.queue_listbox
    lb.selection_clear(0, "end")
    lb.selection_set(row)
    lb.event_generate("<<ListboxSelect>>")


def _setup_batch_jobs(app, tmp_path):
    """Build the UI in batch mode with one RUNNING and one PENDING job.

    Returns the (running, pending) Job objects; rows are 0 and 1.
    """
    _bind_real_methods(app)
    app._build_ui()
    app.input_panel.queue_mode_var.set("batch")
    running = Job(id=JOB_ID, input=tmp_path / "in.mp4",
                  output=tmp_path / "out.mp4",
                  status=JobStatus.RUNNING, is_video=True)
    pending = Job(id=JOB_ID + 1, input=tmp_path / "next.mp4",
                  output=tmp_path / "next_out.mp4",
                  status=JobStatus.PENDING, is_video=True)
    # The listbox is submitted to the pack manager only when the panel sees
    # batch mode plus a non-empty queue (real user flow: add files last).
    app._jobs.extend([running, pending])
    app.input_panel._refresh_empty()
    app._refresh_queue_listbox()
    return running, pending


def test_output_setting_change_persists_to_settings_data(gui_app):
    """Editing an output widget must update settings.data immediately."""
    _bind_real_methods(gui_app)
    gui_app._build_ui()
    assert gui_app.settings.data.single_file_suffix == "_upscaled"
    gui_app.output_panel.suffix_var.set("_live")
    assert gui_app.settings.data.single_file_suffix == "_live", (
        "live suffix edit not persisted; settings.data is stale")
    gui_app.output_panel.output_mode_var.set("custom")
    assert gui_app.settings.data.output_mode == "custom"
    gui_app.output_panel.batch_folder_var.set("batch_live")
    assert gui_app.settings.data.batch_folder_name == "batch_live"
    gui_app.output_panel.output_dir_var.set("C:/tmp/live-out")
    assert gui_app.settings.data.output_dir == "C:/tmp/live-out"


def test_rebuild_keeps_live_output_settings(gui_app):
    """A rebuild (theme change) must not revert unsaved output widget edits.

    Before the fix the rebuilt panel re-read settings.data, which still held
    the last-saved suffix/mode, silently losing the live widget choice.
    """
    _bind_real_methods(gui_app)
    gui_app._build_ui()
    gui_app.output_panel.suffix_var.set("_live")
    gui_app.output_panel.output_mode_var.set("custom")
    gui_app._rebuild_ui()
    assert gui_app.output_panel.suffix_var.get() == "_live", (
        "rebuild reverted the live suffix to stale settings.data")
    assert gui_app.output_panel.output_mode_var.get() == "custom"


def test_start_path_uses_live_output_widgets(gui_app, tmp_path):
    """Pin the half that already worked: Start builds from the live widgets."""
    _bind_real_methods(gui_app)
    gui_app._build_ui()
    gui_app.output_panel.output_mode_var.set("custom")
    gui_app.output_panel.output_dir_var.set(str(tmp_path))
    gui_app.output_panel.suffix_var.set("_live")
    out = gui_app._compute_output_path(tmp_path / "frame.png")
    assert out == tmp_path / "frame_live.png"


def test_gui_pause_cycle_feeds_blocking_events(gui_app, tmp_path):
    """GUI Pause -> Resume -> Pause must end with a blocking worker event."""
    job, _pending = _setup_batch_jobs(gui_app, tmp_path)
    _click_row(gui_app, 0)
    assert not gui_app.input_panel.pause_btn.instate(["disabled"])
    assert gui_app.input_panel.resume_btn.instate(["disabled"])
    gui_app._send_pause(job.id)
    assert job.status == JobStatus.PAUSED
    # Task 10b: the post-pause refresh must leave the toolbar usable without
    # the test manually re-selecting the row.
    assert not gui_app.input_panel.resume_btn.instate(["disabled"]), (
        "pause refresh stranded the toolbar (Resume disabled)")
    assert gui_app.input_panel.pause_btn.instate(["disabled"])
    gui_app._send_resume(job.id)
    assert job.status == JobStatus.RUNNING
    gui_app._send_pause(job.id)
    assert job.status == JobStatus.PAUSED
    # Feed the three control events through the real drain path.
    stub = _worker_stub()
    while not gui_app._ctl_queue.empty():
        stub.ctl_queue.put(gui_app._ctl_queue.get_nowait())
    stub._drain_ctl_for_job(job.id)
    assert stub._pause_events[job.id].is_set() is False


# --------------------------------------------------------------------------- #
# 10b: toolbar usability -- real selection events + refresh must not strand
# --------------------------------------------------------------------------- #

def test_real_selection_enables_queue_controls(gui_app, tmp_path):
    """Brief 10b.2: selecting a row with the mouse enables the toolbar.

    RED (pre-fix): the listbox had no <<ListboxSelect>> binding, so a real
    click updated the Tk selection but Pause/Resume/Cancel stayed disabled.
    """
    _setup_batch_jobs(gui_app, tmp_path)
    panel = gui_app.input_panel
    assert panel.pause_btn.instate(["disabled"]), (
        "toolbar should start disabled with no row selected")
    _click_row(gui_app, 0)
    assert not panel.pause_btn.instate(["disabled"]), (
        "Pause stayed disabled after selecting a RUNNING job (no select binding)")
    assert not panel.cancel_btn.instate(["disabled"])
    assert panel.resume_btn.instate(["disabled"])
    # Selecting the PENDING row disables Pause again (no stale control state).
    _click_row(gui_app, 1)
    assert panel.pause_btn.instate(["disabled"])
    assert panel.resume_btn.instate(["disabled"])


def test_refresh_preserves_queue_selection(gui_app, tmp_path):
    """Brief 10b.1: the ~2 Hz queue refresh must keep the selected job.

    RED (pre-fix): _refresh_queue_listbox deletes and re-inserts every row,
    which cleared the Tk selection, so _selected_job_id() fell back to 0.
    """
    running, _pending = _setup_batch_jobs(gui_app, tmp_path)
    _click_row(gui_app, 0)
    assert gui_app.input_panel._selected_job_id() == running.id
    gui_app._refresh_queue_listbox()
    assert gui_app.input_panel.queue_listbox.curselection() == (0,), (
        "periodic refresh cleared the selection")
    assert gui_app.input_panel._selected_job_id() == running.id
    assert not gui_app.input_panel.pause_btn.instate(["disabled"])


def test_pause_does_not_strand_selection_or_controls(gui_app, tmp_path):
    """Brief 10b.3: pause -> refresh cycles must leave Resume/Cancel usable.

    RED (pre-fix): _send_pause flipped the status and refreshed the listbox,
    which dropped the selection -> Resume and Cancel both disabled -> the
    paused job was stranded with no way to resume or cancel it.
    """
    job, _pending = _setup_batch_jobs(gui_app, tmp_path)
    _click_row(gui_app, 0)
    gui_app._send_pause(job.id)
    assert job.status == JobStatus.PAUSED
    assert gui_app.input_panel.queue_listbox.curselection() == (0,), (
        "pause refresh dropped the selected row")
    assert gui_app.input_panel.resume_btn.instate(["disabled"]) is False, (
        "Resume disabled after pausing the selected job")
    assert gui_app.input_panel.cancel_btn.instate(["disabled"]) is False
    # The worker emits progress/status events while paused; those refresh the
    # listbox repeatedly and must not strand the job again.
    for _ in range(3):
        gui_app._refresh_queue_listbox()
    assert gui_app.input_panel._selected_job_id() == job.id
    assert gui_app.input_panel.resume_btn.instate(["disabled"]) is False
    gui_app._send_resume(job.id)
    assert job.status == JobStatus.RUNNING
    assert not gui_app.input_panel.pause_btn.instate(["disabled"])
    assert gui_app.input_panel.resume_btn.instate(["disabled"])


def test_refresh_selects_neighbor_when_selected_job_disappears(gui_app, tmp_path):
    """Brief 10b.1: removing the selected job must not leave a stale selection."""
    _running, pending = _setup_batch_jobs(gui_app, tmp_path)
    _click_row(gui_app, 0)
    gui_app._remove_selected()
    assert [j.id for j in gui_app._visible_jobs()] == [pending.id]
    selection = gui_app.input_panel.queue_listbox.curselection()
    assert selection, "selection was cleared instead of moving to a neighbor"
    assert gui_app.input_panel._selected_job_id() == pending.id, (
        "selection points at a job that no longer exists")

