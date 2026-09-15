"""GPU monitor widget: shows util%, VRAM used/total, temperature.

Polls pynvml in a background thread (skipped if pynvml is not available
or no CUDA device is present). The widget degrades gracefully — labels
show '—' instead of crashing.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import tkinter as tk
from tkinter import ttk

from ..ui_constants import FONT_MONO, GROUP_PAD, TEXT_MUTED


def _try_pynvml():
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None


class _GPUMonitor(ttk.LabelFrame):
    """Live GPU stats. Polls every `interval_s` seconds (default 1.0)."""

    def __init__(self, parent, app, interval_s: float = 1.0):
        super().__init__(parent, text="GPU", padding=GROUP_PAD)
        self.app = app
        self.interval_s = interval_s

        # Labels
        row = ttk.Frame(self)
        row.pack(fill="x")
        self._util_var = tk.StringVar(value="util: —")
        self._mem_var = tk.StringVar(value="mem: —")
        self._temp_var = tk.StringVar(value="temp: —")
        ttk.Label(row, textvariable=self._util_var, font=FONT_MONO,
                  foreground=TEXT_MUTED).pack(side="left", padx=(0, GROUP_PAD))
        ttk.Label(row, textvariable=self._mem_var, font=FONT_MONO,
                  foreground=TEXT_MUTED).pack(side="left", padx=(0, GROUP_PAD))
        ttk.Label(row, textvariable=self._temp_var, font=FONT_MONO,
                  foreground=TEXT_MUTED).pack(side="left")

        self._nvml = _try_pynvml()
        self._handle = None
        if self._nvml is not None:
            try:
                self._handle = self._nvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._handle = None
                self._nvml = None  # unusable

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if self._handle is not None:
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()

    def _poll_loop(self):
        while not self._stop.wait(self.interval_s):
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
                mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                temp = self._nvml.nvmlDeviceGetTemperature(self._handle, 0)
                # Schedule UI updates on the GUI thread.
                self.after(0, lambda u=util, m=mem, t=temp: self._update_ui(u, m, t))
            except Exception:
                # GPU went away or pynvml call failed; stop polling.
                self._stop.set()
                break

    def _update_ui(self, util: int, mem, temp: int) -> None:
        try:
            self._util_var.set(f"util: {util}%")
            self._mem_var.set(f"mem: {mem.used // (1024 * 1024)} / {mem.total // (1024 * 1024)} MB")
            self._temp_var.set(f"temp: {temp}°C")
        except tk.TclError:
            self._stop.set()

    def stop(self) -> None:
        """Stop the polling thread. Call from app teardown."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
