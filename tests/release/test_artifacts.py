"""Release-branch guards: must fail while the release blockers exist."""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _run(args):
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True)


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
    r = _run([sys.executable, "-c", code])
    assert r.returncode == 0, r.stderr[-2000:]
