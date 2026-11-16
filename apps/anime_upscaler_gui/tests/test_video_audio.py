"""A finished video job must carry the source audio when the source has it."""
from __future__ import annotations

import shutil
import subprocess

import pytest

# Import via the repo's root-pytest convention (pythonpath = [".", "apps"]).
# The brief's plain `anime_upscaler_gui.*` paths resolve to the outer
# docstring-only marker package when pytest runs from the repo root; every
# existing GUI test imports the apps-prefixed path instead.
from apps.anime_upscaler_gui.anime_upscaler_gui.ffmpeg import has_audio_stream
from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline.worker import _mux_job_audio

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe required",
)


def _make_clip(tmp_path, with_audio=True):
    out = tmp_path / "src.mp4"
    args = ["ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=s=64x64:d=1:r=10"]
    if with_audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                 "-c:a", "aac", "-shortest"]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(args, check=True)
    return out


def test_mux_job_audio_keeps_sound(tmp_path):
    src = _make_clip(tmp_path, with_audio=True)
    silent = tmp_path / "out.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                    "-an", "-c:v", "copy", str(silent)], check=True)
    ok = _mux_job_audio(silent, src, start_seconds=0.0, duration_seconds=1.0)
    assert ok is True
    assert has_audio_stream(silent) is True


def test_mux_job_audio_returns_false_without_audio(tmp_path):
    src = _make_clip(tmp_path, with_audio=False)
    silent = tmp_path / "out.mp4"
    shutil.copyfile(src, silent)
    assert _mux_job_audio(silent, src, start_seconds=0.0,
                          duration_seconds=1.0) is False
