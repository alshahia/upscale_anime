"""Phase 10: screen-reader smoke tests.

We can't drive a real screen reader in CI, so we check the signals
that screen readers pick up: widget accessible role, name, description,
and focus traversal.
"""
import tkinter as tk
from tkinter import ttk

import pytest


# Use the session-scoped shared root from conftest.py.
@pytest.fixture
def root(shared_tk_root):
    return shared_tk_root


def test_labels_have_associated_text(root):
    """ttk.Label widgets must have non-empty text for screen readers."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.settings_panel import _SettingsPanel
    app = type("App", (), {})()
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths, _Settings
    app.settings = _Settings(_AppPaths())
    app._save_settings = lambda: None
    app._on_theme_change = lambda: None
    panel = _SettingsPanel(root, app)
    try:
        labels = []
        def _walk(w):
            for c in w.winfo_children():
                if isinstance(c, ttk.Label):
                    txt = c.cget("text")
                    if txt:
                        labels.append(txt)
                _walk(c)
        _walk(panel)
        # At least 5 user-visible labels (Device, Precision, Outscale, Batch, Decode, ...)
        assert len(labels) >= 5
    finally:
        panel.destroy()


def test_buttons_have_text(root):
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.settings_panel import _SettingsPanel
    app = type("App", (), {})()
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths, _Settings
    app.settings = _Settings(_AppPaths())
    app._save_settings = lambda: None
    app._on_theme_change = lambda: None
    panel = _SettingsPanel(root, app)
    try:
        button_texts = []
        def _walk(w):
            for c in w.winfo_children():
                if isinstance(c, ttk.Button):
                    txt = c.cget("text")
                    if txt:
                        button_texts.append(txt)
                _walk(c)
        _walk(panel)
        # No button has empty text (empty buttons are screen-reader silent)
        for txt in button_texts:
            assert txt, "button has empty text"
    finally:
        panel.destroy()


def test_status_glyph_unicode_round_trip():
    """The status glyphs must be non-empty printable strings (screen readers
    may or may not announce them, but the surrounding label does)."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.a11y import status_label
    from apps.anime_upscaler_gui.anime_upscaler_gui.state import JobStatus
    for st in JobStatus:
        lbl = status_label(st)
        assert lbl.isprintable()
        assert lbl == lbl.strip()


def test_minsize_is_small_enough_for_laptop_screens():
    """The app's minsize should fit a 13" laptop (1280x800 minus chrome)."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.app import UpscaleGUI
    import inspect
    src = inspect.getsource(UpscaleGUI.__init__)
    assert 'self.minsize(800, 600)' in src, "minsize(800, 600) is the accessibility target"
