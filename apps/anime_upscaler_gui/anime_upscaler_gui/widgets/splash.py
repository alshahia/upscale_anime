"""Phase 13: Splash screen.

Borderless Toplevel shown before the main GUI. Displays the duotone app
icon + name + tagline for `display_ms`, then fades over `fade_ms`, then
calls the user's `on_done` callback (which constructs the real GUI).

Ponytail: Tk has no native alpha animation; use `wm_attributes('-alpha',
x)` with `after()` to step.
"""
from __future__ import annotations

import tkinter as tk
from typing import Callable, Optional


DEFAULT_DISPLAY_MS = 1500
DEFAULT_FADE_MS = 200
DEFAULT_WIDTH, DEFAULT_HEIGHT = 800, 600


class Splash(tk.Toplevel):
    def __init__(self, parent: Optional[tk.Misc] = None,
                 display_ms: int = DEFAULT_DISPLAY_MS,
                 fade_ms: int = DEFAULT_FADE_MS,
                 on_done: Optional[Callable[[], None]] = None):
        if parent is None:
            parent = tk.Tk()
            parent.withdraw()
        super().__init__(parent)
        self._parent = parent
        self._display_ms = display_ms
        self._fade_ms = fade_ms
        self._on_done = on_done

        self.overrideredirect(True)
        self.configure(bg="#F8FAFC")
        self.geometry(f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}")
        self._center()
        self._build()
        self.wm_attributes("-alpha", 1.0)
        self._displayed = False
        self.after(display_ms, self._begin_fade)

    def _center(self) -> None:
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = (sw - DEFAULT_WIDTH) // 2
        y = (sh - DEFAULT_HEIGHT) // 2
        self.geometry(f"+{x}+{y}")

    def _build(self) -> None:
        # Lazy import so the splash isn't dragged into every GUI import.
        from ..icons import app_256, splash_hero

        hero = splash_hero()
        if hero is not None:
            # The supplied hero PNG already contains the title + tagline baked in.
            # Render only the image; drop our own Tk labels so users don't see
            # text-on-text.
            bg = tk.Label(self, image=hero, bg="#F8FAFC", borderwidth=0)
            bg.image = hero
            bg.place(x=0, y=0, relwidth=1.0, relheight=1.0)
            return

        # Fallback: no hero asset, draw the icon + labels via Tk.
        wrap = tk.Frame(self, bg="#F8FAFC")
        wrap.place(relx=0.5, rely=0.4, anchor="center")

        icon = app_256()
        if icon is not None:
            lbl_icon = tk.Label(wrap, image=icon, bg="#F8FAFC", borderwidth=0)
            lbl_icon.image = icon
            lbl_icon.pack(pady=(0, 16))

        tk.Label(
            wrap, text="Anime Upscaler", font=("Segoe UI", 32, "bold"),
            bg="#F8FAFC", fg="#0F172A",
        ).pack()
        tk.Label(
            wrap, text="Open-source anime super-resolution",
            font=("Segoe UI", 12), bg="#F8FAFC", fg="#64748B",
        ).pack(pady=(6, 0))

    def _begin_fade(self) -> None:
        steps = max(1, self._fade_ms // 20)
        self._fade_step = steps
        self._fade_after = self.after(20, self._fade_tick)

    def _fade_tick(self) -> None:
        if self._fade_step <= 0:
            self._finish()
            return
        self._fade_step -= 1
        self.wm_attributes("-alpha", max(0.0, self._fade_step / max(1, self._fade_step + 1)))
        self.after(20, self._fade_tick)

    def _finish(self) -> None:
        try:
            self.destroy()
        except tk.TclError:
            pass
        if self._on_done is not None:
            self._on_done()