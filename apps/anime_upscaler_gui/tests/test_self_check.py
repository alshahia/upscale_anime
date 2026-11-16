"""The installed console script can prove its own environment."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# tests/ -> apps/anime_upscaler_gui (the GUI project root, which also
# contains the anime_upscaler_gui/ package). Running python -m from there
# resolves the checkout package without an editable install; the CI GUI job
# installs only the dependencies, not the GUI package itself.
PROJECT = Path(__file__).resolve().parents[1]


def test_self_check_reports_environment():
    r = subprocess.run(
        [sys.executable, "-m", "anime_upscaler_gui", "--self-check"],
        cwd=PROJECT, capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    report = json.loads(r.stdout)
    assert report["version"]
    assert isinstance(report["architectures"], list) and report["architectures"]
    assert report["ffmpeg"] is True
    assert report["ffprobe"] is True
