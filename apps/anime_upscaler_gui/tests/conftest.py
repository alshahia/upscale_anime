"""Shared pytest fixtures.

Tkinter allows exactly one *root* per process. Creating a second `tk.Tk()`
and destroying it leaves Tcl unable to locate `init.tcl`, so every later Tk
test in the same process dies with:

    _tkinter.TclError: invalid command name "tcl_findLibrary"

Every test that needs Tk therefore shares the single session-scoped root
below and hangs `tk.Toplevel`s off it. `_single_root_guard` enforces that
rule, turning a violation into one clear failure instead of a cascade of
confusing downstream errors.

The guard also records where each root was allocated (allocation stack plus
the current test id) so a violation names the offending test and the exact
place that asked for a second root, instead of blaming whichever test happens
to be running when Tk finally breaks. It is deliberately still just a guard:
it does not try to isolate test modules from one another.
"""
import os
import queue
import traceback
from pathlib import Path
from typing import Optional, Tuple

import tkinter as tk

import pytest


_ORIG_TK_INIT = tk.Tk.__init__
_roots_created = 0
# (test id, allocation stack) of the FIRST root seen this session -- the
# legitimate shared root. Kept so the error can contrast it with the offender.
_first_root_alloc: Optional[Tuple[str, list]] = None


def _current_test_id() -> str:
    """Return the running test's id, or a placeholder outside a test."""
    return os.environ.get("PYTEST_CURRENT_TEST") or "<no test running>"


def _allocation_stack() -> list:
    """Frames leading to the `tk.Tk()` call, oldest first (excluding this one)."""
    return traceback.extract_stack()[:-1]


def _format_stack(frames) -> str:
    return "".join(traceback.format_list(frames)) or "    <no frames available>"


def _guarded_tk_init(self, *args, **kwargs):
    global _roots_created, _first_root_alloc
    # Snapshot the allocation site first: it is the error's diagnostic payload.
    this_alloc = (_current_test_id(), _allocation_stack())
    _roots_created += 1
    if _roots_created == 1:
        _first_root_alloc = this_alloc
    if _roots_created > 1:
        owner, owner_stack = _first_root_alloc or ("<unknown>", [])
        raise AssertionError(
            "A second Tk root was created. Tk allows only one root per process: "
            "destroying a root leaves Tcl unable to find init.tcl, so every later "
            "Tk test then fails with 'invalid command name \"tcl_findLibrary\"'. "
            "Use the session-scoped `shared_tk_root` fixture and hang a "
            "`tk.Toplevel(shared_tk_root)` off it instead.\n\n"
            f"Offending test: {this_alloc[0]}\n"
            f"Allocation stack of the new root:\n{_format_stack(this_alloc[1])}\n"
            "The first (and only legitimate) root was allocated by: "
            f"{owner}\n{_format_stack(owner_stack)}"
        )
    return _ORIG_TK_INIT(self, *args, **kwargs)


@pytest.fixture(scope="session", autouse=True)
def _single_root_guard():
    """Fail loudly if any test creates a second Tk root."""
    global _roots_created, _first_root_alloc
    _roots_created = 0
    _first_root_alloc = None
    tk.Tk.__init__ = _guarded_tk_init
    yield
    tk.Tk.__init__ = _ORIG_TK_INIT


@pytest.fixture(scope="session")
def shared_tk_root(_single_root_guard):
    """One Tk root for the whole test session.

    Tests that need a parent widget should use this fixture, or chain off it
    with `tk.Toplevel(shared_tk_root)`. It is never destroyed; pytest
    reclaims the process.
    """
    r = tk.Tk()
    r.withdraw()
    yield r
    # Don't destroy on teardown -- subsequent tests in the same process can't
    # create a new Tk root on Windows after one is destroyed.

@pytest.fixture
def gui_app(shared_tk_root, tmp_path):
    """A GUI shell with the real build/queue/add methods bound.

    Enough surface to drive _build_ui, _add_files, _rebuild_ui and the queue
    listbox without a live worker or a persisted queue. Tests call
    _build_ui() first (the fixture deliberately leaves the tree unbuilt so a
    test can observe what a rebuild does), then append Job objects or call
    _add_files(paths=[...]).
    """
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

    app = tk.Toplevel(shared_tk_root)
    for name in (
        "_build_ui", "_rebuild_ui", "_on_theme_change", "_reset_settings",
        "_refresh_model_dropdown", "_make_scrollable", "_build_preview_section",
        "_build_actions_section", "_add_files", "_visible_jobs",
        "_refresh_queue_listbox", "_next_job_id", "_compute_output_path",
        "_update_cut_window_for_input",
    ):
        setattr(app, name, getattr(UpscaleGUI, name).__get__(app, type(app)))
    # staticmethod: assign the plain function or the call gains a spurious self.
    app._is_video_path = UpscaleGUI._is_video_path
    # Panels wire these app callbacks into button commands at build time; a
    # no-op is enough because the tests never invoke them.
    for name in (
        "_clear_jobs", "_retry_selected", "_on_start", "_remove_selected",
        "_pick_output_dir", "_download_preset", "_download_custom_url",
        "_import_local", "_save_settings", "_cancel_download",
        "_confirm_destructive", "_open_settings_folder", "_export_settings",
        "_import_settings", "_move_app_data", "_show_about",
        "_bind_shortcuts", "_send_pause", "_send_resume", "_send_cancel",
        "_start_download", "_on_close",
    ):
        setattr(app, name, lambda *args, **kwargs: None)
    app.paths = paths
    app.settings = settings
    app.registry = _ModelRegistry(Path(tmp_path) / "empty_pretrained")
    app._jobs = []
    app._job_counter = 0
    app._running = False
    app.input_path = None
    app._log_handler = None
    app._resume_boxes = {}
    app._ctl_queue = queue.Queue()
    app._persist_queue = lambda *args, **kwargs: None
    yield app
    handler = app.__dict__.get("_log_handler")
    if handler is not None:
        try:
            handler.set_target(None)
        except Exception:
            pass
    monitor = app.__dict__.get("gpu_monitor")
    if monitor is not None:
        try:
            monitor.stop()
        except Exception:
            pass
    try:
        app.destroy()
    except tk.TclError:
        pass
