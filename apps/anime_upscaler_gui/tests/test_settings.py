"""Tests for the settings module."""
import json
import os
import sys
import tempfile
from pathlib import Path


def _fresh_appdir() -> Path:
    import uuid
    return Path(tempfile.gettempdir()) / f"aug_test_{uuid.uuid4().hex[:8]}"


def test_app_paths_creates_dir():
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths
    d = _fresh_appdir()
    p = _AppPaths(d)
    assert p.app_dir.exists()
    assert (p.app_dir / "cache").exists()
    assert (p.app_dir / "logs").exists()


def test_settings_roundtrip():
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths, _Settings, _Defaults
    d = _fresh_appdir()
    paths = _AppPaths(d)
    s = _Settings(paths)
    s.update(outscale=3.5, fp16=False, last_model="realesr-animevideov3.pth")
    s2 = _Settings(paths)
    assert s2.data.outscale == 3.5
    assert s2.data.fp16 is False
    assert s2.data.last_model == "realesr-animevideov3.pth"


def test_settings_corrupt_file_recovers():
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths, _Settings
    d = _fresh_appdir()
    paths = _AppPaths(d)
    paths.settings_path.write_text("{not valid json")
    s = _Settings(paths)
    assert s.data.outscale == 2.0  # default
    assert paths.settings_path.with_suffix(".bak").exists()


def test_settings_coerce_types():
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths, _Settings
    d = _fresh_appdir()
    paths = _AppPaths(d)
    # write with string ints/floats
    paths.settings_path.write_text(json.dumps({"outscale": "4.0", "batch_size": "8", "fp16": "true"}))
    s = _Settings(paths)
    assert s.data.outscale == 4.0
    assert s.data.batch_size == 8
    assert s.data.fp16 is True


def test_app_paths_move_to():
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths
    d1 = _fresh_appdir()
    d2 = _fresh_appdir()
    p = _AppPaths(d1)
    p.settings_path.write_text("hello")
    p.move_to(d2)
    assert (d2 / "settings.json").read_text() == "hello"
    # Old location may or may not exist depending on shutil.move semantics; that's fine.