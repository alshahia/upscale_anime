"""Phase 11/13: Icon registry.

Resolves `assets/icons/phosphor/...` paths to `tk.PhotoImage` and caches
them so we don't re-decode the same file. Adds a `_safe_image()` helper
that returns `None` if the file is missing — keeps the GUI launchable
even when the asset bundle isn't built yet (e.g. fresh clone).
"""
from __future__ import annotations

import tkinter as tk
from functools import lru_cache
from pathlib import Path
from typing import Optional


def _root_assets() -> Path:
    """`apps/anime_upscaler_gui/assets/` from the package import."""
    return Path(__file__).resolve().parent.parent / "assets"


@lru_cache(maxsize=64)
def _load(path: str, master_key: int = 0) -> Optional[tk.PhotoImage]:
    """Cache by (path, master_id) so Tk images attached to different roots don't collide."""
    p = Path(path)
    if not p.exists():
        return None
    master = _master_for(master_key)
    try:
        return tk.PhotoImage(file=str(p), master=master) if master is not None else tk.PhotoImage(file=str(p))
    except tk.TclError:
        return None


def _master_for(key: int):
    """Pick a master so PhotoImage attaches to an existing Tk root.
    Ponytail: only ever used to satisfy Tk's "needs a default root" rule."""
    if key == 0:
        try:
            return tk._default_root  # type: ignore[attr-defined]
        except AttributeError:
            return None
    return None


def load(name: str, size: int = 24) -> Optional[tk.PhotoImage]:
    """Resolve a Phosphor icon by logical name. Returns None when missing."""
    base = _root_assets() / "icons" / "phosphor"
    if name == "app":
        return _load(str(base / "duotone" / f"app-{size}.png"))
    return _load(str(base / "outline" / f"{name}-{size}.png"))


def app_256() -> Optional[tk.PhotoImage]:
    return load("app", 256)


def splash_hero() -> Optional[tk.PhotoImage]:
    """Splash hero artwork 800x600."""
    return _load(str(_root_assets() / "splash" / "hero-800x600.png"))