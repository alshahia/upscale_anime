"""A finished video job must carry the source audio when the source has it.

I1 regression: the remux window must mirror the frames that were actually
encoded (frame-exact start + probed duration), not the raw cut fields.
A start-only cut (Start>0, End=00:00:00 = "to EOF") used to remux the full
source audio from 0:00 while the picture started at Start, silently
desyncing the export.
"""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

# Import via the repo's root-pytest convention (pythonpath = [".", "apps"]).
# The brief's plain `anime_upscaler_gui.*` paths resolve to the outer
# docstring-only marker package when pytest runs from the repo root; every
# existing GUI test imports the apps-prefixed path instead.
from apps.anime_upscaler_gui.anime_upscaler_gui.ffmpeg import has_audio_stream
from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline.worker import (
    _audio_mux_window, _mux_job_audio,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe required",
)


def _make_clip(tmp_path, with_audio=True, duration=1):
    out = tmp_path / "src.mp4"
    args = ["ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"testsrc=s=64x64:d={duration}:r=10"]
    if with_audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
                 "-c:a", "aac", "-shortest"]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(args, check=True)
    return out


def _make_video_only_picture(tmp_path, source, start_seconds, duration, name):
    """Encode a video-only file representing the picture a cut window keeps."""
    out = tmp_path / name
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-ss", f"{start_seconds}", "-i", str(source), "-t", f"{duration}",
         "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
        check=True,
    )
    return out


def _stream_durations(path):
    """Probe the container/audio durations of a finished file."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(r.stdout)
    lens = {"container": float(info["format"]["duration"]), "audio": 0.0}
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "audio":
            lens["audio"] = float(stream.get("duration") or lens["container"])
    return lens


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


@pytest.mark.parametrize(
    "case, source_duration, picture_start, picture_duration, start_frame",
    [
        ("start-only", 2.0, 1.0, 1.0, 10),   # Start=1.0, End=00:00:00 -> to EOF
        ("end-only", 2.0, 0.0, 1.0, 0),      # End=1.0 with Start=0
        ("full", 1.0, 0.0, 1.0, 0),          # no cut at all
    ],
)
def test_audio_window_follows_the_encoded_picture(
        tmp_path, case, source_duration, picture_start, picture_duration,
        start_frame):
    """I1: the remux window is derived from the encoded frames, not raw cuts.

    For every cut shape the delivered audio must last as long as the picture
    that survives, and in particular a head-trimmed picture must not get the
    full source audio from 0:00 (the audible desync).
    """
    fps = 10.0
    src = _make_clip(tmp_path, with_audio=True, duration=source_duration)
    picture = _make_video_only_picture(
        tmp_path, src, picture_start, picture_duration, f"pic_{case}.mp4")

    start_seconds, duration_seconds = _audio_mux_window(
        picture, fps=fps, start_frame=start_frame,
        n_frames=int(round(picture_duration * fps)))

    assert start_seconds == pytest.approx(start_frame / fps)
    assert duration_seconds == pytest.approx(picture_duration, abs=0.1)
    assert _mux_job_audio(picture, src, start_seconds=start_seconds,
                          duration_seconds=duration_seconds) is True

    lens = _stream_durations(picture)
    assert lens["container"] == pytest.approx(picture_duration, abs=0.15), (
        "the encoded picture must survive the remux")
    assert lens["audio"] == pytest.approx(lens["container"], abs=0.15), (
        "audio must last as long as the picture that exists")
    if source_duration > picture_duration:
        assert lens["audio"] < source_duration - 0.3, (
            "muxing the full source audio from 0:00 would desync the picture")
