"""Tests for apps/anime_upscaler_gui.widgets.settings_spec (Phase C3).

Audit deliverable C3: 'every SETTING_SPEC has a valid widget'. Pinning this
contract here means that adding a new spec entry is a one-line change in
settings_spec.py + zero changes in app.py.
"""
from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.settings_spec import (
    SETTING_SPECS, SettingSpec, get_spec, coerce_var,
)

VALID_WIDGETS = {"radio", "checkbox", "spinbox", "combobox"}


def test_setting_specs_is_non_empty_list():
    """The panel must declare at least one spec."""
    assert isinstance(SETTING_SPECS, list)
    assert len(SETTING_SPECS) >= 1


def test_every_spec_has_a_valid_widget():
    """No spec may use an unknown widget kind."""
    for spec in SETTING_SPECS:
        assert spec.widget in VALID_WIDGETS, (
            f"{spec.key}: widget={spec.widget!r} is not one of {VALID_WIDGETS}"
        )


def test_every_spec_has_a_unique_key():
    """Spec keys are dataclass field names on Defaults; duplicates would be a bug."""
    keys = [s.key for s in SETTING_SPECS]
    assert len(set(keys)) == len(keys), f"duplicate spec keys: {keys}"


def test_every_spec_has_a_var_name():
    """Every spec must declare a tk var attribute name (used by app.py callers)."""
    for spec in SETTING_SPECS:
        assert spec.var_name, f"{spec.key}: empty var_name"
        assert spec.var_name.endswith("_var"), (
            f"{spec.key}: var_name={spec.var_name!r} must end with _var",
        )


def test_radio_and_combobox_have_choices():
    """Radio/combobox must declare choices; otherwise the widget is empty."""
    for spec in SETTING_SPECS:
        if spec.widget in ("radio", "combobox"):
            assert spec.choices, f"{spec.key}: widget={spec.widget} has no choices"
            for choice in spec.choices:
                assert len(choice) == 2, (
                    f"{spec.key}: choices must be (value, label) tuples, got {choice!r}",
                )


def test_spinbox_has_spin_bounds():
    """Spinbox must declare non-zero (from, to, increment)."""
    for spec in SETTING_SPECS:
        if spec.widget == "spinbox":
            from_, to_, inc = spec.spin
            assert from_ < to_, (
                f"{spec.key}: spin={spec.spin} must have from_ < to_",
            )
            assert inc > 0, f"{spec.key}: spin increment must be > 0"


def test_every_spec_has_a_label():
    """Spec labels render before the widget; empty labels would render an empty prefix."""
    for spec in SETTING_SPECS:
        assert spec.label, f"{spec.key}: empty label"


def test_get_spec_known_key_returns_spec():
    """get_spec() is the canonical lookup used by make_panel_value_provider."""
    for spec in SETTING_SPECS:
        got = get_spec(spec.key)
        assert got is spec, f"get_spec({spec.key!r}) did not return the spec"


def test_get_spec_unknown_key_raises():
    """Unknown keys raise KeyError so callers can detect typos immediately."""
    import pytest
    with pytest.raises(KeyError):
        get_spec("definitely_not_a_real_key")


def test_coerce_var_int():
    """coerce_var with int spec coerces strings and floats to int."""
    spec = SettingSpec(
        key="x", label="x", widget="spinbox", var_name="x_var",
        spin=(1, 10, 1), coerce=int,
    )
    assert coerce_var(spec, "5") == 5
    assert coerce_var(spec, 5.7) == 5


def test_coerce_var_no_coerce_returns_raw():
    """coerce_var with no spec.coerce returns the raw value (identity)."""
    spec = SettingSpec(
        key="x", label="x", widget="combobox", var_name="x_var",
    )
    raw = object()
    assert coerce_var(spec, raw) is raw


def test_setting_spec_is_frozen():
    """SettingSpec is a frozen dataclass; cannot mutate after creation."""
    spec = SETTING_SPECS[0]
    import pytest
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        spec.key = "something_else"  # type: ignore[misc]


def test_every_panel_key_is_built_into_a_panel_var():
    """Every spec's var_name must become an attribute on the live SettingsPanel.

    This is the cross-module contract: SETTING_SPECS + SettingsPanel together
    expose all keys as panel.<var_name>. Failing here means a spec was added
    to the data-driven list but the SettingsPanel stopped honouring it
    (or vice versa).
    """
    import tkinter as tk
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.settings_panel import SettingsPanel
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import Defaults

    root = tk.Tk()
    try:
        root.withdraw()
        # SettingsPanel(parent, app) needs an app; the panel reads app.settings.data
        # (the Defaults dataclass) and uses app._save_settings/_on_theme_change.
        # Use a real Defaults so spinbox init coerces correctly.
        class _StubApp:
            def __init__(self):
                self.settings = type("S", (), {"data": Defaults()})()
            def _save_settings(self): pass
            def _on_theme_change(self): pass

        app = _StubApp()
        panel = SettingsPanel(root, app)
        for spec in SETTING_SPECS:
            assert hasattr(panel, spec.var_name), (
                f"SettingsPanel is missing attribute {spec.var_name!r} "
                f"(for spec key={spec.key!r}, widget={spec.widget})",
            )
            # The attribute must be a tk Variable (BoolVar/IntVar/DoubleVar/StringVar):
            var = getattr(panel, spec.var_name)
            assert isinstance(var, tk.Variable), (
                f"{spec.var_name} on SettingsPanel is not a tk.Variable: {type(var)}",
            )
    finally:
        try:
            root.destroy()
        except Exception:
            pass
