"""Release-branch guards: the GUI wheel must be self-contained.

`test_gui_wheel_builds` alone only proves pip returned 0: a clean checkout
can still ship a wheel without the bundled preset catalog or with a stale
vendored model tree. These tests open the built wheel and assert the exact
runtime entries the GUI needs, then import the wheel in a clean-room
subprocess that blocks `anime_sr`.
"""
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
# The importable package is `apps/anime_upscaler_gui/anime_upscaler_gui`; put its
# parent (the GUI app root) on PYTHONPATH so the clean-room subprocess imports the
# GUI from the source tree instead of depending on an installed wheel.
GUI_APP_DIR = REPO / "apps" / "anime_upscaler_gui"
PACKAGE_DIR = "anime_upscaler_gui/"

# The vendored torch-only model closure (see scripts/sync_vendored_models.py):
# every generated module plus the attribution NOTICE. An extra or missing
# entry here means the wheel and the exact-tree guard disagreed.
MODEL_CLOSURE = [
    "_models/__init__.py",
    "_models/eranet.py",
    "_models/inference/__init__.py",
    "_models/inference/tta.py",
    "_models/msrsr.py",
    "_models/NOTICE.md",
    "_models/span/__init__.py",
    "_models/span/mambair_v2.py",
    "_models/span/neosr_span.py",
    "_models/students/__init__.py",
    "_models/students/mambair.py",
    "_models/teachers/__init__.py",
    "_models/teachers/spandrel.py",
]

# Package data the GUI reads at runtime: catalog, icon, splash.
EXPECTED_PACKAGE_DATA = [
    "data/registry.json",
    "arch_registry.py",
    "assets/icons/phosphor/duotone/app.ico",
    "assets/splash/hero-800x600.png",
]


def _run(args, env=None):
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, env=env)


@pytest.fixture(scope="module")
def gui_wheel(tmp_path_factory):
    """Build the GUI wheel once for every content assertion below."""
    out = tmp_path_factory.mktemp("gui-wheel")
    r = _run([sys.executable, "-m", "pip", "wheel",
              str(GUI_APP_DIR), "--no-deps", "-w", str(out)])
    assert r.returncode == 0, r.stderr[-2000:]
    wheels = sorted(out.glob("anime_upscaler_gui-*.whl"))
    assert len(wheels) == 1, f"expected exactly one GUI wheel, got {wheels}"
    return wheels[0]


def _entries(wheel):
    with zipfile.ZipFile(wheel) as zf:
        return {info.filename: info.file_size for info in zf.infolist()}


def test_gui_wheel_builds(gui_wheel):
    assert gui_wheel.is_file()
    assert gui_wheel.stat().st_size > 0


def test_gui_wheel_contains_vendored_model_closure(gui_wheel):
    entries = _entries(gui_wheel)
    missing = [
        PACKAGE_DIR + rel for rel in MODEL_CLOSURE
        if PACKAGE_DIR + rel not in entries
    ]
    assert not missing, f"GUI wheel is missing vendored model closure entries: {missing}"
    leaked = [
        name for name in entries
        if name == "anime_sr" or name.startswith("anime_sr/")
    ]
    assert not leaked, f"GUI wheel must not vendor the anime_sr package: {leaked}"


def test_gui_wheel_contains_package_data(gui_wheel):
    entries = _entries(gui_wheel)
    missing = [
        PACKAGE_DIR + rel for rel in EXPECTED_PACKAGE_DATA
        if PACKAGE_DIR + rel not in entries
    ]
    assert not missing, f"GUI wheel is missing package data: {missing}"
    with zipfile.ZipFile(gui_wheel) as zf:
        raw = zf.read(PACKAGE_DIR + "data/registry.json")
    assert raw.strip(), "bundled preset catalog registry.json is empty"
    data = json.loads(raw.decode("utf-8"))
    assert isinstance(data, dict), f"registry.json is not a JSON object: {data!r}"
    assert "models" in data and isinstance(data["models"], list), (
        f"registry.json is missing the models list: keys={sorted(data)}")


def test_installed_wheel_imports_without_anime_sr(gui_wheel, tmp_path):
    """Unpack the wheel and import its whole model closure with anime_sr blocked."""
    unpacked = tmp_path / "unpacked"
    unpacked.mkdir()
    with zipfile.ZipFile(gui_wheel) as zf:
        zf.extractall(unpacked)
    modules = [
        "anime_upscaler_gui." + rel[: -len("/__init__.py")].replace("/", ".")
        for rel in MODEL_CLOSURE
        if rel.endswith("/__init__.py")
    ] + [
        "anime_upscaler_gui." + rel[: -len(".py")].replace("/", ".")
        for rel in MODEL_CLOSURE
        if rel.endswith(".py") and not rel.endswith("/__init__.py")
    ] + [
        "anime_upscaler_gui.arch_registry",
    ]
    code = "\n".join([
        "import sys",
        "class Block:",
        "    def find_spec(self, name, path=None, target=None):",
        "        if name == 'anime_sr' or name.startswith('anime_sr.'):",
        "            raise ModuleNotFoundError('blocked for clean-room test')",
        "sys.meta_path.insert(0, Block())",
        *[f"import {module}" for module in modules],
        "import anime_upscaler_gui",
        "assert anime_upscaler_gui.__file__.startswith(sys.argv[1]), anime_upscaler_gui.__file__",
        "print('installed-wheel import OK')",
    ])
    env = dict(os.environ)
    env["PYTHONPATH"] = str(unpacked)
    env.pop("PYTHONNOUSERSITE", None)
    r = subprocess.run([sys.executable, "-c", code, str(unpacked)],
                       cwd=str(tmp_path), capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "installed-wheel import OK" in r.stdout
