"""M2 packaging guards (PR-05/06/07).

Two things a wheel build can silently get wrong: dependency floors drift
between pyproject.toml and requirements.txt, and package-data stops covering
the bundled assets so an installed build loses its icons and splash. These
tests fail loudly on both.
"""
from __future__ import annotations

import fnmatch
import re
import tomllib
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = APP_ROOT / "anime_upscaler_gui"


def _normalize(spec: str) -> str:
    """`Pillow>=9.0 ; marker` -> `pillow>=9.0` (PEP 503 name + flat spec)."""
    spec = spec.split(";", 1)[0].strip()
    match = re.match(r"^([A-Za-z0-9_.\-]+)\s*(.*)$", spec)
    name, version = match.group(1), match.group(2)
    name = re.sub(r"[-_.]+", "-", name).lower()
    return name + version.replace(" ", "")


def _pyproject_data() -> dict:
    return tomllib.loads((APP_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _pyproject_deps() -> list[str]:
    return list(_pyproject_data()["project"]["dependencies"])


def _requirements_deps() -> list[str]:
    out = []
    for raw in (APP_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def test_pyproject_and_requirements_agree():
    py = {_normalize(d) for d in _pyproject_deps()}
    req = {_normalize(d) for d in _requirements_deps()}
    assert py == req, (
        "dependency lists diverged\n"
        f"only in pyproject: {sorted(py - req)}\n"
        f"only in requirements: {sorted(req - py)}"
    )


def test_package_version_matches_pyproject():
    from apps.anime_upscaler_gui.anime_upscaler_gui import __version__

    assert __version__ == _pyproject_data()["project"]["version"]


def test_gpu_monitor_dependency_declared():
    assert any(_normalize(d).startswith("nvidia-ml-py") for d in _pyproject_deps())


def test_asset_bundle_is_inside_the_package():
    from apps.anime_upscaler_gui.anime_upscaler_gui import icons

    root = icons._root_assets().resolve()
    assert root == (PACKAGE / "assets").resolve(), (
        "assets must live inside the package so package-data ships them")
    assert (root / "splash" / "hero-800x600.png").is_file()
    assert (root / "icons" / "phosphor" / "duotone" / "app.ico").is_file()


def test_package_data_covers_every_asset():
    patterns = _pyproject_data()["tool"]["setuptools"]["package-data"]["anime_upscaler_gui"]
    files = [
        p.relative_to(PACKAGE).as_posix()
        for p in (PACKAGE / "assets").rglob("*")
        if p.is_file()
    ]
    assert files, "no asset files found under the package"
    uncovered = [
        f for f in files
        if not any(fnmatch.fnmatch(f, pattern) for pattern in patterns)
    ]
    assert not uncovered, f"package-data does not cover: {uncovered}"
