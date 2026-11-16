"""Audio-preservation helpers: encoder output must keep the source audio."""
from __future__ import annotations

import shutil
import subprocess

import pytest

# Import via the repo's root-pytest convention (pythonpath = [".", "apps"]).
# The brief's plain `anime_upscaler_gui.ffmpeg` resolves to the outer
# docstring-only marker package when pytest runs from the repo root; every
# existing GUI test imports the apps-prefixed path instead (e.g. test_gpu_codec).
from apps.anime_upscaler_gui.anime_upscaler_gui.ffmpeg import (
    has_audio_stream,
    mux_audio,
    probe_duration,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe required",
)


def _make_clip(tmp_path, *, with_audio: bool):
    out = tmp_path / ("av.mp4" if with_audio else "video.mp4")
    args = ["ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=s=64x64:d=1:r=10"]
    if with_audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                 "-c:a", "aac", "-shortest"]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(args, check=True)
    return out


def _strip_audio(tmp_path, src) -> "object":
    """Return a video-only copy of *src* (stream copy, no re-encode)."""
    out = tmp_path / "video_only.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                    "-an", "-c:v", "copy", str(out)], check=True)
    return out


def _mux_temp(dest):
    """The sibling temp path mux_audio must never leave behind."""
    return dest.with_name(dest.stem + ".muxing" + dest.suffix)


def test_has_audio_stream_detects_audio_and_video_only(tmp_path):
    assert has_audio_stream(_make_clip(tmp_path, with_audio=True)) is True
    assert has_audio_stream(_make_clip(tmp_path, with_audio=False)) is False


def test_mux_audio_adds_the_source_track(tmp_path):
    src = _make_clip(tmp_path, with_audio=True)
    video_only = tmp_path / "video_only.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                    "-an", "-c:v", "copy", str(video_only)], check=True)
    dest = tmp_path / "muxed.mp4"
    assert mux_audio(video_only, src, dest) is True
    assert dest.is_file()
    assert has_audio_stream(dest) is True
    assert not (tmp_path / "muxed.muxing.mp4").exists()


def test_mux_audio_accepts_str_paths(tmp_path):
    """Task 6/7 may hand the helper str paths from job dictionaries."""
    src = _make_clip(tmp_path, with_audio=True)
    video_only = _strip_audio(tmp_path, src)
    dest = tmp_path / "muxed_str.mp4"
    assert mux_audio(str(video_only), str(src), str(dest)) is True
    assert dest.is_file()
    assert has_audio_stream(dest) is True
    assert not _mux_temp(dest).exists()


def test_mux_audio_returns_false_and_leaves_no_temp_when_upstream_is_gone(tmp_path):
    """The encoded temp may already be deleted by a crashed upstream step."""
    src = _make_clip(tmp_path, with_audio=True)
    dest = tmp_path / "muxed_missing.mp4"
    assert mux_audio(tmp_path / "already_gone.mp4", src, dest) is False
    assert not dest.exists()
    assert not _mux_temp(dest).exists()


def test_mux_audio_without_source_audio_leaves_no_temp(tmp_path):
    """A video-only source must not crash the mux or leave a temp file behind."""
    video_only = _make_clip(tmp_path, with_audio=False)
    dest = tmp_path / "muxed_noaudio.mp4"
    assert mux_audio(video_only, video_only, dest) is True
    assert dest.is_file()
    assert has_audio_stream(dest) is False
    assert not _mux_temp(dest).exists()


def test_mux_audio_cut_window_trims_audio(tmp_path):
    """start_seconds/duration_seconds mirror the job's cut window."""
    src = _make_clip(tmp_path, with_audio=True)
    video_only = _strip_audio(tmp_path, src)
    dest = tmp_path / "muxed_cut.mp4"
    assert mux_audio(video_only, src, dest, start_seconds=0.5) is True
    assert has_audio_stream(dest) is True
    duration = probe_duration(dest)
    assert duration is not None and 0.2 <= duration <= 0.9
    assert not _mux_temp(dest).exists()

def test_has_audio_stream_false_when_ffprobe_unavailable(monkeypatch):
    """Degrade to False instead of raising when the app has no ffprobe."""
    from apps.anime_upscaler_gui.anime_upscaler_gui import ffmpeg as ffmpeg_mod

    monkeypatch.setattr(ffmpeg_mod, "ffprobe_available", lambda: False)
    assert has_audio_stream("does-not-need-to-exist.mp4") is False


def test_mux_audio_false_when_ffmpeg_unavailable(monkeypatch, tmp_path):
    """Degrade to False instead of raising, and write nothing, without ffmpeg."""
    from apps.anime_upscaler_gui.anime_upscaler_gui import ffmpeg as ffmpeg_mod

    monkeypatch.setattr(ffmpeg_mod, "ffmpeg_available", lambda: False)
    dest = tmp_path / "no_ffmpeg.mp4"
    assert mux_audio(tmp_path / "v.mp4", tmp_path / "a.mp4", dest) is False
    assert not dest.exists()
    assert not _mux_temp(dest).exists()

