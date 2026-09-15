"""Phase 14: i18n tests.

Ponytail: stdlib-only check. No babel/gettext installation gymnastics;
the messages module is two dicts and a switch.
"""
from __future__ import annotations

import importlib

import pytest


def _reload_messages():
    """Fresh import. Tests that mutate the active locale need a known starting state."""
    import apps.anime_upscaler_gui.anime_upscaler_gui.messages as m
    return importlib.reload(m)


@pytest.fixture
def msgs():
    m = _reload_messages()
    yield m
    m._reset_for_tests()()


def test_default_locale_is_english(msgs):
    assert msgs.active_locale() == "en"


def test_supported_locales(msgs):
    assert set(msgs.SUPPORTED_LOCALES) == {"en", "ar"}


def test_lookup_returns_english_default(msgs):
    assert msgs._("menu_file") == "File"


def test_lookup_uses_kwargs(msgs):
    assert msgs._("unsupported_kind", filename="x.pth", kind="srvgg") == \
        "x.pth is kind='srvgg'; not supported in this MVP."


def test_unknown_key_returns_itself(msgs):
    assert msgs._("not_a_real_key") == "not_a_real_key"


def test_unknown_key_with_kwargs_returns_self(msgs):
    assert msgs._("not_a_real_key", anything=1) == "not_a_real_key"


def test_switch_to_arabic_changes_catalogue(msgs):
    msgs.set_locale("ar")
    assert msgs.active_locale() == "ar"
    assert msgs._("menu_file") == "ملف"
    assert "النماذج" in msgs._("models_tab")


def test_unknown_locale_is_noop(msgs):
    msgs.set_locale("ar")
    msgs.set_locale("xx")  # bogus
    assert msgs.active_locale() == "ar"


def test_format_kwargs_in_arabic(msgs):
    msgs.set_locale("ar")
    out = msgs._("queue_added", n=3)
    assert "3" in out


def test_fallback_when_translation_missing(msgs):
    """If a key is absent in AR, return English so the user sees real text, not the key."""
    msgs.set_locale("ar")
    # All shipped keys have Arabic entries; introduce a missing one deliberately
    msgs._AR.pop("tainted_checkpoint_title", None)
    assert msgs._("tainted_checkpoint_title") == "Tainted checkpoint"


def test_is_rtl_flag(msgs):
    assert msgs.is_rtl() is False
    msgs.set_locale("ar")
    assert msgs.is_rtl() is True


def test_side_for_mirroring(msgs):
    assert msgs.side_for("left") == "left"
    msgs.set_locale("ar")
    assert msgs.side_for("left") == "right"
    assert msgs.side_for("right") == "left"


def test_side_for_passes_through_other_defaults(msgs):
    msgs.set_locale("ar")
    # Anything that isn't 'left'/'right' passes through (e.g. 'top', 'bottom').
    assert msgs.side_for("top") == "top"


def test_reset_for_tests_restores_locale(msgs):
    msgs.set_locale("ar")
    msgs._reset_for_tests()
    assert msgs.active_locale() == "en"


def test_locale_choice_list(msgs):
    assert msgs.locale_choices() == ["ar", "en"] or msgs.locale_choices() == ["en", "ar"]
    assert set(msgs.locale_choices()) == {"en", "ar"}


def test_settings_has_locale_field():
    """The persisted settings defaults must include locale='en'."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _Defaults
    assert _Defaults().locale == "en"


def test_coerce_drops_unknown_locale_does_not_crash():
    """An old settings.json without 'locale' must still load cleanly."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _coerce
    out = _coerce({"device": "cpu"})  # no locale
    assert out.locale == "en"  # default


def test_settings_round_trip_preserves_locale():
    """Atomic write then re-load roundtrip must keep the locale value."""
    import json
    from pathlib import Path
    import tempfile
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import (
        _AppPaths, _Settings, _atomic_dump_json,
    )
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        settings = _Settings(_AppPaths(p))
        settings.data.locale = "ar"
        settings.save()
        loaded = _Settings(_AppPaths(p))
        assert loaded.data.locale == "ar"


def test_menu_bar_includes_locale_toggle_shortcut():
    """Bindings list is matched against the README shortcut table; shortcut exists."""
    pkg = Path(__file__).resolve().parents[1] / "anime_upscaler_gui"
    # shortcuts live in app.py + window-chrome mixin: scan the package files
    joined = chr(10).join(p.read_text(encoding="utf-8") for p in sorted(pkg.glob("app*.py")))
    assert "<Control-Shift-L>" in joined
    assert "_toggle_locale" in joined


def test_readme_documents_locale_toggle():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    assert "Ctrl+Shift+L" in readme


from pathlib import Path  # local import to keep fixture scope clean
