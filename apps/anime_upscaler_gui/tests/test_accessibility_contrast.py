"""WCAG AA contrast guard.

The plan calls for a thin test that locks in the WCAG AA contract for every
shipped theme, with a self-explanatory failure message. `theme.assert_wcag`
already does the heavy lifting; this file pins the contract for future
contributors.
"""
import pytest


REQUIRED_PAIRS_LIGHT_BG = [
    # (label, theme_field, expected_against_bg)
    ("text vs bg", 15.0, "text"),
    ("accent vs bg", 4.5, "accent"),
    ("error vs bg", 4.5, "error"),
    ("done vs bg", 4.5, "done"),
    ("running vs bg", 4.5, "running"),
    ("pending vs bg", 4.5, "pending"),
    ("skipped vs bg", 4.5, "skipped"),
    ("cancelled vs bg", 4.5, "cancelled"),
]


def test_light_theme_text_contrast():
    """Body text on bg must clear WCAG AAA (>7:1) for the light theme."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.theme import LIGHT, contrast_ratio
    ratio = contrast_ratio(LIGHT.text, LIGHT.bg)
    assert ratio >= 15.0, f"text/bg contrast {ratio:.2f} < 15.0 (WCAG AAA)"


def test_dark_theme_text_contrast():
    from apps.anime_upscaler_gui.anime_upscaler_gui.theme import DARK, contrast_ratio
    ratio = contrast_ratio(DARK.text, DARK.bg)
    assert ratio >= 15.0, f"text/bg contrast {ratio:.2f} < 15.0 (WCAG AAA)"


def test_high_contrast_text_contrast():
    """HC theme: pure black on pure white must be 21:1."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.theme import HIGH_CONTRAST, contrast_ratio
    ratio = contrast_ratio(HIGH_CONTRAST.text, HIGH_CONTRAST.bg)
    assert ratio == pytest.approx(21.0, abs=0.01)


def test_all_themes_accent_contrast_aa():
    from apps.anime_upscaler_gui.anime_upscaler_gui.theme import by_name, contrast_ratio
    for name in ("light", "dark", "high_contrast"):
        t = by_name(name)
        r = contrast_ratio(t.accent, t.bg)
        assert r >= 4.5, f"{name} accent/bg = {r:.2f}, want >= 4.5"


def test_all_themes_status_colors_aa():
    """Every status foreground must be >= 4.5:1 against bg."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.theme import by_name, contrast_ratio
    for name in ("light", "dark", "high_contrast"):
        t = by_name(name)
        for status in ("pending", "running", "done", "error", "skipped", "cancelled"):
            r = contrast_ratio(getattr(t, status), t.bg)
            assert r >= 4.5, f"{name}/{status} = {r:.2f}, want >= 4.5"


def test_contrast_ratio_order_independent():
    """contrast_ratio(fg, bg) and contrast_ratio(bg, fg) must return the same value."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.theme import contrast_ratio
    a = contrast_ratio("#4F46E5", "#F8FAFC")
    b = contrast_ratio("#F8FAFC", "#4F46E5")
    assert a == pytest.approx(b, abs=0.01)
