"""Video cut-window widget: HH:MM:SS spinboxes + duration probe."""
import threading
import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from .ffmpeg import probe_duration
from .ui_constants import DISABLED, FONT_MONO


def _parse_hms(s: str) -> float:
    """Parse 'HH:MM:SS' or 'MM:SS' or seconds-as-string into total seconds."""
    s = s.strip()
    if not s:
        return 0.0
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = (int(p) for p in parts)
        elif len(parts) == 2:
            h, m, sec = 0, int(parts[0]), int(parts[1])
        else:
            return float(parts[0])
        return float(h * 3600 + m * 60 + sec)
    except (ValueError, IndexError):
        return 0.0


def _format_hms(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class _CutWindow(ttk.LabelFrame):
    """Start / End HH:MM:SS entries; auto-probes duration via ffprobe.

    Hidden by default; .show() / .hide() toggle pack.
    """

    def __init__(self, parent, on_change: Optional[Callable[[float, float], None]] = None):
        super().__init__(parent, text="Video cut", padding=6)
        self.on_change = on_change
        self._probing = False

        r = ttk.Frame(self)
        r.pack(fill="x")
        ttk.Label(r, text="Duration:").pack(side="left")
        self.duration_var = tk.StringVar(value="--:--:--")
        ttk.Label(r, textvariable=self.duration_var, font=FONT_MONO,
                  foreground=DISABLED).pack(side="left", padx=(4, 12))
        ttk.Label(r, text="Start").pack(side="left")
        self.start_var = tk.StringVar(value="00:00:00")
        ttk.Entry(r, textvariable=self.start_var, width=10, font=FONT_MONO).pack(side="left", padx=(4, 12))
        ttk.Label(r, text="End").pack(side="left")
        self.end_var = tk.StringVar(value="00:00:00")
        ttk.Entry(r, textvariable=self.end_var, width=10, font=FONT_MONO).pack(side="left", padx=(4, 12))
        ttk.Button(r, text="Full", command=self._full_range).pack(side="left")
        ttk.Button(r, text="+30s", command=lambda: self._add_seconds(30)).pack(side="left", padx=(4, 0))

        self.start_var.trace_add("write", lambda *a: self._emit())
        self.end_var.trace_add("write", lambda *a: self._emit())

    def show(self):
        self.pack(fill="x", pady=(0, 6))

    def hide(self):
        self.pack_forget()

    def probe_async(self, path):
        """Run ffprobe in a daemon thread, then update duration."""
        if self._probing:
            return
        self._probing = True

        def _run():
            try:
                d = probe_duration(path)
            except Exception:
                d = None
            def _apply():
                self._probing = False
                if d and d > 0:
                    self.duration_var.set(_format_hms(d))
                    if self.end_var.get() in ("00:00:00", ""):
                        self.end_var.set(_format_hms(min(d, 30.0)))
            try:
                self.after(0, _apply)
            except Exception:
                pass
        threading.Thread(target=_run, daemon=True).start()

    def _full_range(self):
        self.start_var.set("00:00:00")
        self.end_var.set(self.duration_var.get())

    def _add_seconds(self, n: int):
        cur = _parse_hms(self.end_var.get())
        self.end_var.set(_format_hms(cur + n))

    def _emit(self):
        if self.on_change:
            try:
                self.on_change(_parse_hms(self.start_var.get()), _parse_hms(self.end_var.get()))
            except Exception:
                pass

    def get_seconds(self) -> tuple:
        return _parse_hms(self.start_var.get()), _parse_hms(self.end_var.get())

    def is_valid(self) -> bool:
        s, e = self.get_seconds()
        return e > s >= 0