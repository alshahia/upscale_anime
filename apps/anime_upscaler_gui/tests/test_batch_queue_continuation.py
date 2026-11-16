"""M1 regression (PR-01): a failed batch job must not be retried forever.

_enqueue_next used to treat JobStatus.ERROR as runnable, so the error
handler's continuation call (_handle_event "error" -> _enqueue_next(_jobs))
picked the same failed job again: the first bad file looped the queue and no
later job ever started. Auto-continuation now advances PENDING only, while an
explicit Start resets ERROR -> PENDING so retry still works.
"""
import queue
import types

import pytest

from apps.anime_upscaler_gui.anime_upscaler_gui import app as app_mod
from apps.anime_upscaler_gui.anime_upscaler_gui.app import UpscaleGUI
from apps.anime_upscaler_gui.anime_upscaler_gui.state import Job, JobStatus


class _FakeVar:
    def __init__(self, value=""):
        self._v = value

    def set(self, value):
        self._v = value

    def get(self):
        return self._v


def _fake_app():
    """Just enough app surface for the real _enqueue_next to run."""
    a = types.SimpleNamespace()
    a.settings_panel = object()
    a.settings = types.SimpleNamespace(data=object())
    a.start_btn = types.SimpleNamespace(config=lambda **k: None)
    a.status_bar = types.SimpleNamespace(status_var=_FakeVar())
    a._in_queue = queue.Queue()
    a._running = True
    a._is_video_path = lambda p: False
    a._resolve_model_path_from_dropdown = lambda **k: "model.pth"
    a._resolve_kind_from_dropdown = lambda **k: "realesrgan"
    a._wait_resume_action = lambda *a, **k: None
    a._enqueue_next = UpscaleGUI._enqueue_next.__get__(a, type(a))
    return a


@pytest.fixture(autouse=True)
def _patch_builder(monkeypatch):
    # _enqueue_next only needs the builder to return something puttable.
    monkeypatch.setattr(
        app_mod, "_make_panel_value_provider", lambda sp: (lambda key: None))
    monkeypatch.setattr(
        app_mod, "_build_run_job", lambda **kw: ("runjob", kw["job"].id))


def _job(tmp_path, jid, status):
    return Job(
        id=jid,
        input=tmp_path / ("img%d.png" % jid),
        output=tmp_path / ("img%d_out.png" % jid),
        status=status,
    )


def test_auto_continuation_skips_errored_job(tmp_path):
    a = _fake_app()
    failed = _job(tmp_path, 1, JobStatus.ERROR)
    pending = _job(tmp_path, 2, JobStatus.PENDING)

    a._enqueue_next([failed, pending])

    assert failed.status == JobStatus.ERROR, (
        "a failed job must not be auto-retried by the continuation pass")
    assert pending.status == JobStatus.RUNNING
    assert a._in_queue.qsize() == 1
    assert a._in_queue.get_nowait() == ("runjob", 2)


def test_only_errored_jobs_left_ends_the_run(tmp_path):
    a = _fake_app()
    failed = _job(tmp_path, 1, JobStatus.ERROR)
    states = []
    a.start_btn = types.SimpleNamespace(config=lambda **k: states.append(k))

    a._enqueue_next([failed])

    assert failed.status == JobStatus.ERROR
    assert a._in_queue.empty()
    assert a._running is False
    assert states and states[-1].get("state") == "normal"


def test_batch_queue_advances_past_failure(tmp_path):
    a = _fake_app()
    failed = _job(tmp_path, 1, JobStatus.ERROR)
    second = _job(tmp_path, 2, JobStatus.PENDING)
    third = _job(tmp_path, 3, JobStatus.PENDING)
    jobs = [failed, second, third]

    a._enqueue_next(jobs)           # must start #2, not retry #1
    second.status = JobStatus.DONE  # simulate completion
    a._enqueue_next(jobs)           # must continue to #3

    assert (failed.status, second.status, third.status) == (
        JobStatus.ERROR, JobStatus.DONE, JobStatus.RUNNING)
    got = []
    while not a._in_queue.empty():
        got.append(a._in_queue.get_nowait())
    assert got == [("runjob", 2), ("runjob", 3)]
