"""I1 guard: an invalid cut window is ignored (and reported), never half-applied.

`_CutWindow.is_valid()` is `end > start >= 0`. The GUI used to pass the raw
spinbox values straight into the job, so Start>0 with End=00:00:00 produced a
head-trimmed picture with full audio from 0:00. _on_start now normalises an
invalid window to "no cut" before building the RunJob.
"""
import types

import pytest

from apps.anime_upscaler_gui.anime_upscaler_gui.app import UpscaleGUI


class _FakeVar:
    def __init__(self, value=""):
        self.value = value

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


class _FakeCutWindow:
    def __init__(self, start, end):
        self._start, self._end = start, end

    def get_seconds(self):
        return self._start, self._end

    def is_valid(self):
        return self._end > self._start >= 0


def _fake_app(start=0.0, end=0.0):
    """Just enough app surface for the real _effective_cut_window to run."""
    app = types.SimpleNamespace()
    app.cut_window = _FakeCutWindow(start, end)
    app.status_bar = types.SimpleNamespace(status_var=_FakeVar())
    app._effective_cut_window = UpscaleGUI._effective_cut_window.__get__(
        app, type(app))
    return app


@pytest.mark.parametrize("start,end", [(5.0, 0.0), (8.0, 2.0)])
def test_invalid_window_becomes_no_cut_and_is_reported(start, end):
    app = _fake_app(start, end)
    assert app._effective_cut_window() == (0.0, 0.0)
    assert "Cut window ignored" in app.status_bar.status_var.get()


def test_default_zero_window_is_no_cut_without_noise():
    app = _fake_app(0.0, 0.0)
    assert app._effective_cut_window() == (0.0, 0.0)
    assert app.status_bar.status_var.get() == ""


def test_valid_window_is_used_unchanged():
    app = _fake_app(2.0, 8.0)
    assert app._effective_cut_window() == (2.0, 8.0)
    assert app.status_bar.status_var.get() == ""
