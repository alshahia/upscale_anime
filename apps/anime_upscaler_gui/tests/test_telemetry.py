"""Phase 9: GPU monitor + log panel + logging setup + toast."""
import logging
import os
import re
import threading
import time
import tkinter as tk
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---- log panel ----

@pytest.fixture
def root(shared_tk_root):
    return shared_tk_root


def _make_app_shim():
    app = type("App", (), {})()
    app._jobs = []
    return app


def test_log_panel_constructs(root):
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.log_panel import _LogPanel
    panel = _LogPanel(root, _make_app_shim())
    assert panel.text.winfo_exists()


def test_log_panel_append_increments_text(root):
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.log_panel import _LogPanel
    panel = _LogPanel(root, _make_app_shim())
    panel._append("hello", logging.INFO)
    root.update_idletasks()
    panel._append("world", logging.WARNING)
    root.update_idletasks()
    text = panel.text.get("1.0", "end")
    assert "hello" in text
    assert "world" in text


def test_log_panel_trims_to_max_lines(root):
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.log_panel import _LogPanel
    panel = _LogPanel(root, _make_app_shim(), max_lines=3)
    for i in range(10):
        panel._append(f"line {i}", logging.INFO)
    root.update_idletasks()
    text = panel.text.get("1.0", "end")
    # Should have at most 3 lines
    lines = [ln for ln in text.splitlines() if ln]
    assert len(lines) <= 3


def test_log_panel_severity_tags(root):
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.log_panel import _LogPanel
    panel = _LogPanel(root, _make_app_shim())
    panel._append("warn-msg", logging.WARNING)
    root.update_idletasks()
    # The tag exists; the message was inserted under it.
    assert "warning" in panel.text.tag_names()


def test_log_panel_clear(root):
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.log_panel import _LogPanel
    panel = _LogPanel(root, _make_app_shim())
    panel._append("stays", logging.INFO)
    panel.clear()
    root.update_idletasks()
    text = panel.text.get("1.0", "end").strip()
    assert text == ""


# ---- log panel handler (logging.Handler bridge) ----

def test_log_handler_forwards_to_panel(root):
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.log_panel import (
        _LogPanel, _LogPanelHandler,
    )
    panel = _LogPanel(root, _make_app_shim())
    h = _LogPanelHandler()
    h.setFormatter(logging.Formatter("%(message)s"))
    h.set_target(panel)
    record = logging.LogRecord(
        name="t", level=logging.INFO, pathname="", lineno=0,
        msg="forwarded-msg", args=(), exc_info=None
    )
    h.emit(record)
    # after(0, ...) is queued; use update() to process the event loop.
    root.update()
    text = panel.text.get("1.0", "end")
    assert "forwarded-msg" in text


def test_log_handler_no_target_no_crash():
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.log_panel import _LogPanelHandler
    h = _LogPanelHandler()
    record = logging.LogRecord(
        name="t", level=logging.INFO, pathname="", lineno=0,
        msg="orphan", args=(), exc_info=None
    )
    # No target set — emit must not raise.
    h.emit(record)


def test_log_handler_unset_target_stops_forwarding(root):
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.log_panel import (
        _LogPanel, _LogPanelHandler,
    )
    panel = _LogPanel(root, _make_app_shim())
    h = _LogPanelHandler()
    h.set_target(panel)
    h.set_target(None)  # detach
    h.emit(logging.LogRecord(
        name="t", level=logging.INFO, pathname="", lineno=0,
        msg="ignored", args=(), exc_info=None))
    root.update_idletasks()
    assert "ignored" not in panel.text.get("1.0", "end")


# ---- logging_setup ----

def test_setup_logging_creates_file(tmp_path):
    from apps.anime_upscaler_gui.anime_upscaler_gui import logging_setup
    _reset_logging(logging_setup)
    logs = tmp_path / "logs"
    logger = logging_setup.setup_logging(logs)
    logger.info("hello")
    for h in logger.handlers:
        h.flush()
    log_path = logs / "app.log"
    assert log_path.exists()
    text = log_path.read_text(encoding="utf-8")
    assert "hello" in text
    _reset_logging(logging_setup)


def test_setup_logging_idempotent(tmp_path):
    from apps.anime_upscaler_gui.anime_upscaler_gui import logging_setup
    _reset_logging(logging_setup)
    a = logging_setup.setup_logging(tmp_path / "logs")
    handler_count_after_first = len(a.handlers)
    b = logging_setup.setup_logging(tmp_path / "logs")
    # Same logger; second call is a no-op (no extra handlers)
    assert a is b
    assert len(b.handlers) == handler_count_after_first
    # First-call count: file_handler + stream = 2
    assert handler_count_after_first == 2
    _reset_logging(logging_setup)


def _reset_logging(logging_setup):
    """Detach all handlers from the package logger + reset the init flag.

    Lets tests run in any order without leaking handlers across tests.
    """
    logger = logging.getLogger(logging_setup.LOGGER_NAME)
    for h in list(logger.handlers):
        try:
            h.flush()
            h.close()
        except Exception:
            pass
        logger.removeHandler(h)
    logging_setup._initialized = False


# ---- gpu monitor ----

def test_gpu_monitor_constructs_without_pynvml(root):
    """The widget must build successfully even when pynvml is absent."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.gpu_monitor import _GPUMonitor
    app = type("App", (), {})()
    monitor = _GPUMonitor(root, app)
    # Default labels show "—"
    assert "—" in monitor._util_var.get()
    assert "—" in monitor._mem_var.get()
    assert "—" in monitor._temp_var.get()
    monitor.stop()


def test_gpu_monitor_update_ui():
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.gpu_monitor import _GPUMonitor
    app = type("App", (), {})()
    monitor = _GPUMonitor(tk.Frame(), app)
    fake_mem = MagicMock(used=1024 * 1024 * 512, total=1024 * 1024 * 8192)
    monitor._update_ui(util=42, mem=fake_mem, temp=66)
    assert "42%" in monitor._util_var.get()
    assert "512" in monitor._mem_var.get() and "8192" in monitor._mem_var.get()
    assert "66" in monitor._temp_var.get()
    monitor.stop()


# ---- toast ----

def test_toast_constructs_and_positions(root):
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.toast import Toast
    t = Toast(root, "test message", duration_ms=50000)
    assert t.winfo_exists()
    t.destroy()


def test_toast_destroy_no_target_widget_error(root):
    """Destroying a Toast without showing it must not raise."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.toast import Toast
    t = Toast(root, "x", duration_ms=10)
    t.destroy()
