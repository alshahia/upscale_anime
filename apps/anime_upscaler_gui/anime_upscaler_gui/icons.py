"""Phase 11/13: Icon registry.

Resolves `assets/icons/phosphor/...` paths to `tk.PhotoImage` and caches
the decoded images so we don't re-decode the same file. Returns `None`
when an asset is missing -- keeps the GUI launchable even when the asset
bundle isn't built yet (e.g. fresh clone).
"""
from __future__ import annotations

import tkinter as tk
from functools import lru_cache
from pathlib import Path
from typing import Optional


def _root_assets() -> Path:
    """Asset bundle directory (``<package>/assets``).

    Assets live *inside* the package so [tool.setuptools.package-data]
    ships them in the wheel and sdist. The legacy sibling directory is kept
    as a fallback for source trees that predate the move.
    """
    pkg = Path(__file__).resolve().parent
    local = pkg / "assets"
    if local.is_dir():
        return local
    return pkg.parent / "assets"


@lru_cache(maxsize=64)
def _decode(path: str, master_key: int) -> tk.PhotoImage:
    """Decode one icon for the interpreter selected by `master_key`.

    Only *successful* decodes end up in the cache: `lru_cache` never
    memoises a raised exception, so a transient Tk failure cannot poison
    the cache and permanently disable an icon for the rest of the session.
    """
    master = _master_for(master_key)
    if master is not None:
        return tk.PhotoImage(file=path, master=master)
    return tk.PhotoImage(file=path)


def _load(path: str, master_key: int = 0) -> Optional[tk.PhotoImage]:
    """Decode `path`, returning `None` when it is missing or undecodable.

    Tk 8.6 on Windows has a quirk where the *first* `image create photo
    ... -file` issued against a freshly created interpreter can fail with a
    **message-less** `TclError` even though the file is perfectly readable;
    an immediate identical retry succeeds. So we retry exactly once for the
    empty-message case, and never for a real error (a bad path or
    undecodable data always carries a message). Without this, one transient
    hiccup would make the icon resolve to `None` for the whole session.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        return _decode(str(p), master_key)
    except tk.TclError as exc:
        if exc.args and exc.args[0]:
            return None
        try:
            return _decode(str(p), master_key)
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
