"""Release-branch guards: must fail while the release blockers exist."""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# The importable package is `apps/anime_upscaler_gui/anime_upscaler_gui`; put its
# parent (the GUI app root) on PYTHONPATH so the clean-room subprocess imports the
# GUI from the source tree instead of depending on an installed wheel.
GUI_APP_DIR = REPO / "apps" / "anime_upscaler_gui"


def _run(args, env=None):
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, env=env)


def test_gui_wheel_builds():
    out = REPO / "dist"
    out.mkdir(exist_ok=True)
    r = _run([sys.executable, "-m", "pip", "wheel",
              str(REPO / "apps" / "anime_upscaler_gui"),
              "--no-deps", "-w", str(out)])
    assert r.returncode == 0, r.stderr[-2000:]


def test_clean_room_import_does_not_need_anime_sr():
    code = "\n".join([
        "import sys",
        "class Block:",
        "    def find_spec(self, name, path=None, target=None):",
        "        if name == 'anime_sr' or name.startswith('anime_sr.'):",
        "            raise ModuleNotFoundError('blocked for clean-room test')",
        "sys.meta_path.insert(0, Block())",
        "import anime_upscaler_gui.arch_registry",
        "print('clean-room import OK')",
    ])
    pythonpath = str(GUI_APP_DIR)
    if os.environ.get("PYTHONPATH"):
        pythonpath = os.pathsep.join([pythonpath, os.environ["PYTHONPATH"]])
    env = dict(os.environ, PYTHONPATH=pythonpath)
    r = _run([sys.executable, "-c", code], env=env)
    assert r.returncode == 0, r.stderr[-2000:]
