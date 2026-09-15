"""Phase 5: theme apply + swatch binding + headless theme change."""
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import pytest


@pytest.fixture
def root(shared_tk_root):
    """Use the session-shared root from conftest.py."""
    return shared_tk_root


@pytest.fixture
def fake_app(root, tmp_path):
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _Settings, _AppPaths, _Defaults
    from apps.anime_upscaler_gui.anime_upscaler_gui.registry import _ModelRegistry
    app_dir = tmp_path / "aug"
    app_dir.mkdir(parents=True, exist_ok=True)
    paths = _AppPaths(app_dir=app_dir)
    settings = _Settings(paths)
    settings.data = _Defaults()
    settings.data.pretrained_dir = str(tmp_path / "empty_pretrained")
    app = type("FakeApp", (), {})()
    app.settings = settings
    app.registry = _ModelRegistry(Path(tmp_path) / "empty_pretrained")
    app._refresh_queue_listbox = lambda: None
    app._refresh_model_dropdown = lambda: None
    app._on_theme_change = lambda: None
    app._save_settings = lambda: None
    return app


def test_themes_registry_has_three():
    from apps.anime_upscaler_gui.anime_upscaler_gui.theme import THEMES, available
    assert set(THEMES.keys()) == {"light", "dark", "high_contrast"}
    assert set(available()) == {"dark", "high_contrast", "light"}


def test_by_name_known_and_unknown():
    from apps.anime_upscaler_gui.anime_upscaler_gui.theme import by_name
    assert by_name("light").name == "light"
    with pytest.raises(KeyError):
        by_name("neon")


def test_apply_mutates_ui_constants_in_place():
    """`apply()` should update the same module attrs, not rebind imports."""
    from apps.anime_upscaler_gui.anime_upscaler_gui import ui_constants as ui
    from apps.anime_upscaler_gui.anime_upscaler_gui.theme import apply, LIGHT, DARK
    from apps.anime_upscaler_gui.anime_upscaler_gui.state import JobStatus

    apply(LIGHT)
    assert ui.BG == LIGHT.bg
    assert ui.ACCENT == LIGHT.accent
    assert ui.STATUS_COLORS[JobStatus.DONE] == LIGHT.done

    apply(DARK)
    assert ui.BG == DARK.bg
    assert ui.ACCENT == DARK.accent
    assert ui.STATUS_COLORS[JobStatus.DONE] == DARK.done


def test_apply_status_bg_dicts_updated():
    from apps.anime_upscaler_gui.anime_upscaler_gui import ui_constants as ui
    from apps.anime_upscaler_gui.anime_upscaler_gui.theme import apply, LIGHT
    from apps.anime_upscaler_gui.anime_upscaler_gui.state import JobStatus
    apply(LIGHT)
    assert ui.STATUS_BG_COLORS[JobStatus.RUNNING] == LIGHT.bg_running
    assert ui.STATUS_BG_COLORS[JobStatus.ERROR] == LIGHT.bg_error


def test_contrast_ratio_basic_known_values():
    """Black on white = 21:1, white on white = 1:1."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.theme import contrast_ratio
    assert contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0, abs=0.01)
    assert contrast_ratio("#FFFFFF", "#FFFFFF") == pytest.approx(1.0, abs=0.01)


def test_contrast_ratio_short_hex():
    from apps.anime_upscaler_gui.anime_upscaler_gui.theme import contrast_ratio
    assert contrast_ratio("#000", "#FFF") == pytest.approx(21.0, abs=0.01)


@pytest.mark.parametrize("name", ["light", "dark", "high_contrast"])
def test_all_themes_meet_wcag_aa(name):
    from apps.anime_upscaler_gui.anime_upscaler_gui.theme import by_name, assert_wcag
    failures = assert_wcag(by_name(name), min_ratio=4.5)
    assert failures == [], f"{name} WCAG AA failures: {failures}"


@pytest.mark.parametrize("name", ["light", "dark", "high_contrast"])
def test_all_themes_meet_wcag_aa_large_text(name):
    """Large text threshold is 3.0:1."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.theme import by_name, assert_wcag
    failures = assert_wcag(by_name(name), min_ratio=3.0)
    assert failures == []


def test_settings_panel_has_theme_radio(fake_app, root):
    """Headless smoke: settings panel mounts the theme row with 3 radios."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.settings_panel import _SettingsPanel

    fake_app.settings.update(theme="dark")
    panel = _SettingsPanel(tk.Frame(root), fake_app)
    try:
        assert panel.theme_var.get() == "dark"
        # Theme row is the LAST child of the panel (row5).
        theme_row = panel.winfo_children()[-1]
        canvases = [c for c in theme_row.winfo_children() if isinstance(c, tk.Canvas)]
        # 3 swatches, one per theme.
        assert len(canvases) == 3
        # Each swatch's bg color is the corresponding theme's bg.
        from apps.anime_upscaler_gui.anime_upscaler_gui.theme import by_name
        bgs = sorted(c["bg"] for c in canvases)
        assert bgs == sorted([by_name(n).bg for n in ("light", "dark", "high_contrast")])
    finally:
        panel.destroy()


def test_unknown_theme_via_by_name_raises():
    from apps.anime_upscaler_gui.anime_upscaler_gui.theme import by_name
    with pytest.raises(KeyError):
        by_name("nope")


def test_theme_field_round_trip(tmp_path):
    """Save theme=dark, reload, verify."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths, _Settings
    d = tmp_path / "appdir"
    paths = _AppPaths(d)
    s = _Settings(paths)
    s.update(theme="dark")
    s2 = _Settings(paths)
    assert s2.data.theme == "dark"


def test_settings_default_theme_is_light():
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _Defaults
    assert _Defaults().theme == "light"
