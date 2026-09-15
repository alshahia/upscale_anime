"""_deprecation.py -- proxy + warning helpers for Phase A1 underscore aliases.

When the Phase A1 refactor renamed private classes (e.g. `_RunJob`) to
public names (`RunJob`), the old names were kept as aliases for back-compat
with existing scripts and tests. Phase E adds a soft-deprecation signal so
downstream callers know the alias will be removed, and a single source of
truth for the removal timeline.

Design:
  - DeprecatedAlias is a transparent proxy. Attribute access and calls
    forward to the underlying object (so isinstance(_RunJob(...), RunJob)
    and _RunJob.__doc__ still work).
  - On FIRST use, the proxy emits a DeprecationWarning via warnings.warn.
    The first use is recorded in a module-level set keyed by
    (public_name, removal_version, caller_filename), so the same alias
    from the same location only warns once even if it is accessed 1000
    times in a session.
  - The warning message includes the public replacement name, the removal
    milestone (e.g. 0.4.0), and a docs URL pointing at docs/PUBLIC_API.md.

Removal timeline (single source of truth; update here when bumping):
  - 0.2.x (current): DeprecationWarning emitted on first use
  - 0.3.x          : same warning, escalated to always filter in CI
  - 0.4.0          : aliases removed

Why a proxy instead of name = public_name:
  - The bare assignment runs at import time but emits no warning until
    something touches the alias later. By wrapping the underlying object
    in a proxy, every attribute access goes through __getattr__, which
    is exactly when we want to warn.
  - We avoid module-level __getattr__ (PEP 562) because several modules
    already use it for unrelated reasons, and mixing the two is brittle.
"""
from __future__ import annotations

import inspect
import warnings
from typing import Any


# One warning per (public_name, removal_version, caller_filename). Tests
# reset this in fixtures so cross-test bleed is impossible.
_warned_keys: set = set()


# Docs URL the warning points at. PUBLIC_API.md lists the public names and
# the back-compat aliases; the URL is repo-root relative so it works for
# both in-tree users and standalone-copy users.
_DOCS_URL = "../../docs/PUBLIC_API.md"


def _make_warning_message(public_name: str, removal_version: str) -> str:
    """Build the human-readable deprecation message."""
    return (
        "Deprecated alias; use " + repr(public_name)
        + " instead. The alias will be removed in " + str(removal_version)
        + ". See " + str(_DOCS_URL) + " for the migration."
    )


class DeprecatedAlias:
    """Transparent proxy that emits DeprecationWarning on first attribute access.

    Behaves exactly like the wrapped object for every Python operation;
    the only side effect is the one-shot warning. Intended use:

        from .pipeline import RunJob  # the public name
        _RunJob = DeprecatedAlias("RunJob", RunJob, removal_version="0.4.0")

    Now _RunJob is importable, instantiable, introspectable, and is an
    instance of RunJob, but any use emits a one-shot DeprecationWarning.
    """

    __slots__ = ("_public_name", "_target", "_removal_version", "_warned")

    def __init__(self, public_name: str, target: Any, removal_version: str) -> None:
        self._public_name = public_name
        self._target = target
        self._removal_version = removal_version
        self._warned = False

    # -- introspection helpers ------------------------------------------
    @property
    def __class__(self):
        # isinstance(_RunJob(...), RunJob) must succeed. Forward.
        return self._target.__class__

    @property
    def __name__(self):
        # pickle, dataclasses, and some debuggers introspect __name__.
        return getattr(self._target, "__name__", self._public_name)

    def __repr__(self) -> str:
        # __repr__ is special-cased so pdb / f-strings don't trigger the
        # warning. repr(deprecated) should be safe to call at any time.
        return "<DeprecatedAlias for " + repr(self._public_name) + " at " + hex(id(self)) + ">"

    # -- the warning trigger --------------------------------------------
    def _maybe_warn(self) -> None:
        """Emit the DeprecationWarning exactly once per (alias, call site).

        stacklevel=3 so the warning reports the caller of the attribute
        access, not _maybe_warn itself and not __getattr__. That is the
        standard pattern for proxies.
        """
        if self._warned:
            return
        try:
            caller_frame = inspect.stack()[2]
            caller_file = caller_frame.filename
        except Exception:
            caller_file = "<unknown>"
        key = (self._public_name, self._removal_version, caller_file)
        if key in _warned_keys:
            self._warned = True  # never re-warn from this proxy
            return
        _warned_keys.add(key)
        self._warned = True
        msg = _make_warning_message(self._public_name, self._removal_version)
        warnings.warn(msg, DeprecationWarning, stacklevel=3)

    # -- attribute access: the warning trigger --------------------------
    # True dunder names that introspection frameworks rely on, which we
    # route without warning so isinstance(), pickle, dataclasses, and
    # debuggers still work:
    _DUNDER_NAMES = frozenset({
        "__doc__", "__module__", "__qualname__", "__annotations__",
        "__dataclass_fields__", "__dataclass_params__",
        "__init__", "__new__", "__repr__", "__str__",
        "__wrapped__", "__func__", "__self__", "__dict__",
        "__bases__", "__mro__", "__subclasses__",
        "__slots__", "__hash__", "__eq__", "__ne__",
        "__reduce__", "__reduce_ex__", "__getstate__", "__setstate__",
        "__sizeof__", "__dir__",
        # Python 3.10+ dataclass protocol:
        "__match_args__",
    })

    def __getattr__(self, name: str) -> Any:
        # __getattr__ only fires when normal lookup fails, so
        # self._public_name etc. are read from __slots__ directly.
        if name == "__dict__":
            # pytest.monkeypatch.setattr() and some other test utilities
            # poke at __dict__ on the proxy to figure out where to store
            # the patched attribute. Forward to the target if it has a
            # __dict__ (classes and most instances do); the proxy itself
            # uses __slots__ and has no __dict__ of its own.
            target_dict = getattr(self._target, "__dict__", None)
            if target_dict is not None:
                return target_dict
            raise AttributeError(name)
        if name == "__wrapped__":
            # Property on the class -- __getattr__ only fires on miss, so
            # this branch is a no-op, but keep it explicit for clarity.
            return self._target
        if name in self._DUNDER_NAMES:
            # Dunder introspection: do NOT warn. This keeps isinstance(),
            # pickle, dataclasses, f-strings, pdb all working without
            # flooding the user with warnings. If the target object doesn't
            # actually have the dunder (e.g. simple instances that don't
            # define __qualname__), raise AttributeError as Python would
            # for a plain instance -- do NOT fall through to the warning
            # branch because that would mask the missing-attr error.
            raise AttributeError(name)
        # Everything else (public methods/attributes + private attrs
        # used by framework internals) goes through the warning.
        self._maybe_warn()
        return getattr(self._target, name)

    # -- monkeypatching support -------------------------------------------
    # pytest's monkeypatch.setattr(proxy, "name", val) goes through
    # __setattr__; we forward to the underlying target so tests that
    # patch attributes on the deprecated alias see the same effect as
    # patching the public name. __slots__ attributes on the proxy itself
    # are stored locally as usual.
    def __setattr__(self, name: str, value: Any) -> None:
        if name in type(self).__slots__:
            object.__setattr__(self, name, value)
            return
        self._maybe_warn()
        setattr(self._target, name, value)

    def __delattr__(self, name: str) -> None:
        if name in type(self).__slots__:
            object.__delattr__(self, name)
            return
        self._maybe_warn()
        delattr(self._target, name)

    # -- method calls ----------------------------------------------------
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._maybe_warn()
        return self._target(*args, **kwargs)

    # -- equality / hashing ---------------------------------------------
    def __eq__(self, other: Any) -> bool:
        return self._target == other

    def __ne__(self, other: Any) -> bool:
        return self._target != other

    def __hash__(self) -> int:
        return hash(self._target)

    def __bool__(self) -> bool:
        return bool(self._target)

    # -- target lookup for code that wants to unwrap --------------------
    @property
    def __wrapped__(self) -> Any:
        """The underlying public object. Exposed for functools.wraps-style
        unwrap, but accessing it does NOT warn (it is a documented escape
        hatch for code that needs the real object).
        """
        return self._target


def reset_warning_state() -> None:
    """Clear the one-shot dedup set. Tests call this in setUp to assert the
    warning fires again from a fresh call site.
    """
    _warned_keys.clear()


def make_alias(public_name: str, target: Any, removal_version: str) -> DeprecatedAlias:
    """Convenience constructor: build a DeprecatedAlias with the standard
    warning message. Use this everywhere so the removal-version string
    lives in exactly one place per alias site.
    """
    return DeprecatedAlias(public_name, target, removal_version)


__all__ = [
    "DeprecatedAlias",
    "make_alias",
    "reset_warning_state",
]
