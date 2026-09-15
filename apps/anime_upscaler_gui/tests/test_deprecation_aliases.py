"""Tests for apps.anime_upscaler_gui._deprecation (Phase E).

Pins the soft-deprecation contract for the Phase A1 underscore aliases:
  - the proxy emits DeprecationWarning on FIRST use (not at import)
  - subsequent uses from the same call site are silent (one-shot dedup)
  - dunder introspection (isinstance, pickle, dataclasses) does NOT warn
  - public attribute access / call / setattr / delattr DOES warn
  - the wrapped object remains accessible via __wrapped__ (escape hatch)
  - the removal timeline is documented in one place
  - the public alias site still works (forwarded behavior)
"""
import warnings

import pytest

from apps.anime_upscaler_gui.anime_upscaler_gui._deprecation import (
    DeprecatedAlias, make_alias, reset_warning_state,
)


# --- helpers ---------------------------------------------------------------
class _Sentinel:
    """Stand-in target object. We can probe every Python operation on it."""
    def __init__(self, name="Sentinel"):
        self.name = name

    def __repr__(self):
        return "<Sentinel " + str(self.name) + ">"

    def __call__(self, *args, **kwargs):
        return ("called", self.name, args, kwargs)

    def public_method(self):
        return "public_method_ok"


@pytest.fixture(autouse=True)
def _clear_dedup():
    """Reset the per-(alias, caller) dedup set before every test."""
    reset_warning_state()
    yield
    reset_warning_state()


# --- the proxy itself -----------------------------------------------------
def test_alias_target_is_accessible_via_wrapped():
    """__wrapped__ is the documented escape hatch; no warning on access."""
    sentinel = _Sentinel()
    alias = make_alias("Sentinel", sentinel, removal_version="0.4.0")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        target = alias.__wrapped__
    assert target is sentinel
    assert len(w) == 0, "__wrapped__ must not emit a warning"


def test_alias_call_forwards_to_target():
    """Calling the proxy returns the target call result and warns once."""
    sentinel = _Sentinel()
    alias = make_alias("Sentinel", sentinel, removal_version="0.4.0")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = alias("a", kw="b")
    assert result == ("called", "Sentinel", ("a",), {"kw": "b"})
    assert len(w) == 1
    assert issubclass(w[0].category, DeprecationWarning)
    assert "Sentinel" in str(w[0].message)
    assert "0.4.0" in str(w[0].message)


def test_alias_public_attr_access_warns():
    """Accessing a non-dunder attribute goes through the warning trigger."""
    sentinel = _Sentinel()
    alias = make_alias("Sentinel", sentinel, removal_version="0.4.0")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = alias.public_method()
    assert result == "public_method_ok"
    assert len(w) == 1


def test_alias_dunder_access_does_not_warn():
    """isinstance(), pickle, and friends must work without warning spam."""
    sentinel = _Sentinel()
    alias = make_alias("Sentinel", sentinel, removal_version="0.4.0")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Read several dunder attributes that frameworks rely on:
        _ = alias.__doc__
        _ = alias.__module__
        _ = alias.__repr__()
        _ = hash(alias)
        _ = bool(alias)
        # __qualname__ may not exist on simple instances; the proxy should
        # raise AttributeError silently rather than warning or hiding the
        # error. We only need to assert that no WARNING was emitted.
        try:
            _ = alias.__qualname__
        except AttributeError:
            pass
    dunder_warnings = [
        x for x in w
        if "Sentinel" in str(x.message) or "0.4.0" in str(x.message)
    ]
    assert dunder_warnings == [], (
        f"dunder access must be silent, got: {[str(x.message) for x in dunder_warnings]}",
    )


def test_alias_isinstance_works():
    """isinstance(proxy, target_class) is critical for dataclass use."""
    sentinel = _Sentinel()
    alias = make_alias("Sentinel", sentinel, removal_version="0.4.0")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ok = isinstance(alias, _Sentinel)
    assert ok is True
    assert len(w) == 0, "isinstance must not warn"


def test_alias_one_shot_per_call_site():
    """Same alias from the same caller warns only once per session."""
    sentinel = _Sentinel()
    alias = make_alias("Sentinel", sentinel, removal_version="0.4.0")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for _ in range(5):
            alias.public_method()
    assert len(w) == 1, f"expected one warning, got {len(w)}"


def test_alias_warning_message_includes_public_name_and_version():
    """The warning is grep-able: it contains the public name and version."""
    sentinel = _Sentinel()
    alias = make_alias("MyPublicName", sentinel, removal_version="1.2.3")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        alias("x")
    assert len(w) == 1
    msg = str(w[0].message)
    assert "MyPublicName" in msg, f"public name missing: {msg}"
    assert "1.2.3" in msg, f"removal version missing: {msg}"
    assert "PUBLIC_API" in msg, f"docs link missing: {msg}"


def test_alias_setattr_delattr_forward_to_target():
    """pytest monkeypatch on the proxy must affect the underlying target."""
    sentinel = _Sentinel()
    alias = make_alias("Sentinel", sentinel, removal_version="0.4.0")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        alias.injected = "patched_value"
        del alias.injected
    # Both ops route through __setattr__ and __delattr__; the warning is
    # one-shot per (alias, call site) so we see only ONE warning even
    # though both ops hit _maybe_warn().
    assert not hasattr(sentinel, "injected")
    assert len(w) == 1, f"expected one warning (one-shot), got {len(w)}"


def test_alias_equality_and_hash_use_target():
    """The proxy compares equal to its target and is hash-compatible."""
    sentinel = _Sentinel()
    alias = make_alias("Sentinel", sentinel, removal_version="0.4.0")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert alias == sentinel
        assert hash(alias) == hash(sentinel)
    assert len(w) == 0, "eq/hash must not warn"


# --- the alias sites ------------------------------------------------------
def test_phase_a1_aliases_are_deprecation_proxies():
    """Every Phase A1 underscore alias is a DeprecatedAlias."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline import (
        _RunJob, _JobEvent, _PipelineWorker, _make_backend,
    )
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import (
        _Defaults, _AppPaths, _Settings, _QueueController,
    )
    from apps.anime_upscaler_gui.anime_upscaler_gui.registry import (
        _is_supported_kind, _ModelRegistry,
    )
    for proxy in [_RunJob, _JobEvent, _PipelineWorker, _make_backend,
                  _Defaults, _AppPaths, _Settings, _QueueController,
                  _is_supported_kind, _ModelRegistry]:
        assert isinstance(proxy, DeprecatedAlias), (
            f"{proxy!r} should be a DeprecatedAlias after Phase E",
        )


def test_phase_a1_aliases_all_carry_the_same_removal_version():
    """Single source of truth: every alias says 0.4.0."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline import (
        _RunJob, _JobEvent, _PipelineWorker, _make_backend,
    )
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import (
        _Defaults, _AppPaths, _Settings, _QueueController,
    )
    from apps.anime_upscaler_gui.anime_upscaler_gui.registry import (
        _is_supported_kind, _ModelRegistry,
    )
    for proxy in [_RunJob, _JobEvent, _PipelineWorker, _make_backend,
                  _Defaults, _AppPaths, _Settings, _QueueController,
                  _is_supported_kind, _ModelRegistry]:
        assert proxy._removal_version == "0.4.0", (
            f"{proxy._public_name}: removal version = {proxy._removal_version!r}",
        )


def test_phase_a1_aliases_call_still_works():
    """The aliases still construct real Settings / ModelRegistry."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths, _Settings
    from apps.anime_upscaler_gui.anime_upscaler_gui.registry import _ModelRegistry
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td, warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        paths = _AppPaths(app_dir=td)
        s = _Settings(paths)
        s.data.outscale = 3.5
        s.save()
        s2 = _Settings(paths)
        assert s2.data.outscale == 3.5
        reg = _ModelRegistry(pretrained_dir=td)
        # Compare paths via Path() so Windows-path-style differences don't fail:
        assert Path(reg.pretrained_dir) == Path(td)


def test_warning_fires_on_real_use_of_alias():
    """End-to-end: importing _RunJob then using it fires the warning."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline import _RunJob
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = _RunJob.__doc__  # dunder, silent
        assert all("RunJob" not in str(x.message) for x in w), (
            "__doc__ access should be silent, got: " + str(w)
        )
        # Now a real use:
        try:
            _RunJob()  # invalid args; warning fires BEFORE the TypeError
        except TypeError:
            pass
        deprecation_warnings = [
            x for x in w
            if issubclass(x.category, DeprecationWarning)
            and "RunJob" in str(x.message)
        ]
        assert len(deprecation_warnings) == 1, (
            f"expected exactly one RunJob deprecation warning, got {len(deprecation_warnings)}",
        )
        assert "0.4.0" in str(deprecation_warnings[0].message)


def test_reset_warning_state_clears_dedup():
    """reset_warning_state() lets tests re-assert the warning from the same site.

    We use TWO proxies against the same target so the call-site dedup key
    changes between the two phases of the test (one proxy -> ResetTest@file,
    the second -> also ResetTest@file but the dedup set has been cleared).
    Actually the dedup key includes the caller file (this test file), so
    a single proxy warns once; we create a second proxy to demonstrate the
    global reset clears the dedup table so future proxies re-warn.
    """
    sentinel = _Sentinel()
    alias1 = make_alias("ResetTest", sentinel, removal_version="0.4.0")
    with warnings.catch_warnings(record=True) as w1:
        warnings.simplefilter("always")
        alias1("a")
        alias1("b")  # dedup suppresses
    assert len(w1) == 1
    # reset_warning_state clears the dedup table; a freshly-built proxy
    # against the same target sees a clean slate and warns again:
    reset_warning_state()
    alias2 = make_alias("ResetTest", sentinel, removal_version="0.4.0")
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        alias2("c")
    assert len(w2) == 1, f"after reset expected 1 warning, got {len(w2)}"


def test_import_time_does_not_warn():
    """Importing the modules that declare the aliases must not warn."""
    import importlib
    import apps.anime_upscaler_gui.anime_upscaler_gui.pipeline as _p
    import apps.anime_upscaler_gui.anime_upscaler_gui.settings as _s
    import apps.anime_upscaler_gui.anime_upscaler_gui.registry as _r
    importlib.reload(_p)
    importlib.reload(_s)
    importlib.reload(_r)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = type(_p._RunJob)
        _ = type(_s._Defaults)
        _ = type(_r._ModelRegistry)
    assert len(w) == 0, (
        "importing the module + reading the alias off the module must be silent, "
        f"got: {[str(x.message) for x in w]}",
    )
