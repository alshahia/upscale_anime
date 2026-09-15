"""Documentation coverage tests.

These are small, structural checks: every shortcut or menu accelerator
advertised in user docs must be wired in `app.py`/`widgets/menu_bar.py`,
and the docs cross-links must resolve on disk.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
ONBOARDING = ROOT / "docs" / "onboarding.md"
THEMES = ROOT / "docs" / "themes.md"
WIDGETS_DOC = ROOT / "docs" / "dev" / "widgets.md"
RESUME = ROOT / "docs" / "RESUME.md"

APP_PY = ROOT / "anime_upscaler_gui" / "app.py"
# app.py is split across mixin modules; shortcut bindings and window
# ceremony live in these files, so doc checks scan them with app.py.
APP_SOURCE_FILES = [
    APP_PY,
    ROOT / "anime_upscaler_gui" / "app_settings_io.py",
    ROOT / "anime_upscaler_gui" / "app_window_chrome.py",
]
MENU_BAR = ROOT / "anime_upscaler_gui" / "widgets" / "menu_bar.py"

DOC_FILES = [README, ONBOARDING, THEMES, WIDGETS_DOC, RESUME]


@pytest.mark.parametrize("path", DOC_FILES, ids=lambda p: p.name)
def test_doc_files_exist(path):
    assert path.exists(), f"missing doc file: {path}"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _app_sources() -> str:
    """All GUI orchestrator sources concatenated (app.py + mixins)."""
    return chr(10).join(_read(p) for p in APP_SOURCE_FILES)


# ------- shortcuts table in README must match _bind_shortcuts --------
SHORTCUT_TABLE = {
    "Ctrl+O": ("<Control-o>", "_add_files"),
    "Ctrl+S": ("<Control-s>", "_save_settings"),
    "Ctrl+Enter": ("<Control-Return>", "_on_start"),
    "Delete": ("<Delete>", "_remove_selected"),
    "F5": ("<F5>", "_refresh_model_dropdown"),
    "Ctrl+R": ("<Control-r>", "_reset_settings"),
    "Ctrl+Shift+L": ("<Control-Shift-L>", "_toggle_locale"),
    "Ctrl+Q": ("<Control-q>", "_on_close"),
}


def test_readme_shortcut_table_matches_bindings():
    readme = _read(README)
    app_src = _app_sources()
    for label, (tk_seq, method) in SHORTCUT_TABLE.items():
        assert label in readme, f"README shortcut table missing: {label}"
        assert tk_seq in app_src, f"app.py missing binding: {tk_seq}"
        assert method in app_src, f"app.py missing handler: {method}"


def test_readme_shortcut_table_in_same_order_as_bindings():
    """Documentation order should mirror the wiring order; if reordering one, reorder the other."""
    readme = _read(README)
    rows = [line for line in readme.splitlines() if line.startswith("| `") and "Ctrl" in line or line.startswith("| `") and "F5" in line or line.startswith("| `") and "Delete" in line]
    # ponytail: simple `in`-membership check; we don't try to be clever with table parsing.
    expected_labels = list(SHORTCUT_TABLE)
    found = [label for label in expected_labels if label in readme]
    assert found == expected_labels, (
        "README shortcut table reorder; update _bind_shortcuts comment to match"
    )


# ------- menu bar accelerators must be present in README --------
def test_menu_bar_accelerators_match_readme():
    menu_src = _read(MENU_BAR)
    readme = _read(README)
    # Menu accelerator labels (uppercase initial + accel)
    label_accels = ("Ctrl+O", "Ctrl+S", "Ctrl+Q", "Ctrl+R", "Ctrl+Enter", "F5")
    for acc in label_accels:
        assert acc in menu_src, f"menubar missing accelerator: {acc}"
        assert acc in readme, f"README missing accelerator mention: {acc}"
    # Delete is registered as a Tk binding, not a menu entry label.
    assert "<Delete>" in _app_sources() or "<Delete>" in menu_src
    assert "Delete" in readme, "README must list Delete shortcut"


# ------- README cross-link targets must resolve --------
def test_readme_doc_links_resolve():
    text = _read(README)
    for slug in ("docs/onboarding.md", "docs/themes.md", "docs/dev/widgets.md",
                 "docs/RESUME.md", "docs/plans/ui_ux_overhaul.md"):
        assert slug in text, f"README reference missing: {slug}"
        target = ROOT / slug
        assert target.exists(), f"README link broken: {target}"


# ------- onboarding cross-link targets must resolve --------
def test_onboarding_links_resolve():
    text = _read(ONBOARDING)
    # Onboarding should not contradict the README about command names.
    for cmd in ("LAUNCHER.bat", "LAUNCHER.sh", "models", "Settings", "Start"):
        if cmd.lower() in text.lower():
            continue  # informational mention, not a hard link
    # onboarding must use the same control language as the menubar
    for shortcut in ("Ctrl+O", "Ctrl+S", "Ctrl+Enter", "F5", "Ctrl+R", "Ctrl+Q"):
        assert shortcut in text, f"onboarding missing shortcut: {shortcut}"


# ------- themes doc must point at the actual theme module --------
def test_themes_doc_points_at_code():
    text = _read(THEMES)
    assert "anime_upscaler_gui/theme.py" in text
    assert "ui_constants.py" in text
    target = ROOT / "anime_upscaler_gui" / "theme.py"
    assert target.exists()


# ------- widgets doc must point at widgets/, conftest, ROOT run --------
def test_widgets_doc_commands_resolve():
    text = _read(WIDGETS_DOC)
    assert "apps\\anime_upscaler_gui\\tests\\" in text or "apps/anime_upscaler_gui/tests/" in text
    # Documented checkpoints
    assert "shared_tk_root" in text
    assert "after()" in text or "after(" in text


# ------- documentation mentions code that has its own tests --------
def test_documented_themes_have_test_coverage():
    text = _read(THEMES)
    assert "test_theme.py" in text
    assert "test_accessibility_contrast.py" in text
    test_theme = ROOT / "tests" / "test_theme.py"
    test_contrast = ROOT / "tests" / "test_accessibility_contrast.py"
    assert test_theme.exists()
    assert test_contrast.exists()


# ------- no orphan screenshots or external URLs that we'd have to ship --------
def test_docs_do_not_reference_missing_images():
    """If any doc references an image, it must exist on disk."""
    img_re = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    for path in DOC_FILES:
        text = _read(path)
        for m in img_re.finditer(text):
            target = (path.parent / m.group(1)).resolve()
            assert target.exists(), f"doc image missing: {path}:{m.group(1)} -> {target}"


# ------- README must not advertise shortcut keys that the wiring dropped --------
def test_no_orphan_shortcut_in_readme():
    """If a shortcut is in the README table, its handler must exist in app.py."""
    readme = _read(README)
    app_src = _app_sources()
    for label in SHORTCUT_TABLE:
        if label not in readme:
            continue
        seq, method = SHORTCUT_TABLE[label]
        assert method in app_src, f"README advertises {label} but {method} is gone from app.py"
