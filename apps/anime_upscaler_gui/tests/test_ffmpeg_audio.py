"""Audio-preservation helpers: encoder output must keep the source audio."""
from __future__ import annotations

import json
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
    probe_video_meta,
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


def _make_av_clip(tmp_path, *, video_seconds: float, audio_seconds: float, name: str):
    """A clip whose audio track may be shorter than its picture (F2 repro)."""
    out = tmp_path / name
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"testsrc=s=64x64:d={video_seconds}:r=10",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={audio_seconds}",
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out),
    ], check=True)
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


def _ffprobe_json(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(r.stdout)


def _assert_full_picture(path, expected_seconds: float, tolerance: float = 0.2):
    """The muxed file must still carry a sane, full-length video stream."""
    meta = probe_video_meta(path)
    assert meta is not None, "ffprobe could not read the muxed output"
    assert meta["width"] > 0 and meta["height"] > 0
    assert abs(meta["duration"] - expected_seconds) <= tolerance
    duration = probe_duration(path)
    assert duration is not None
    assert abs(duration - expected_seconds) <= tolerance


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
    # The picture must survive the remux untouched (F1/F2 regression guard).
    _assert_full_picture(dest, expected_seconds=1.0)
    assert not (tmp_path / "muxed.muxing.mp4").exists()


def test_mux_audio_accepts_str_paths(tmp_path):
    """Task 6/7 may hand the helper str paths from job dictionaries."""
    src = _make_clip(tmp_path, with_audio=True)
    video_only = _strip_audio(tmp_path, src)
    dest = tmp_path / "muxed_str.mp4"
    assert mux_audio(str(video_only), str(src), str(dest)) is True
    assert dest.is_file()
    assert has_audio_stream(dest) is True
    _assert_full_picture(dest, expected_seconds=1.0)
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
    _assert_full_picture(dest, expected_seconds=1.0)
    assert not _mux_temp(dest).exists()


def test_mux_audio_cut_window_trims_audio_but_keeps_full_picture(tmp_path):
    """start_seconds/duration_seconds trim the AUDIO window; the picture must stay whole."""
    src = _make_clip(tmp_path, with_audio=True)
    video_only = _strip_audio(tmp_path, src)
    dest = tmp_path / "muxed_cut.mp4"
    assert mux_audio(video_only, src, dest, start_seconds=0.5) is True
    assert has_audio_stream(dest) is True
    # The encoded picture is 1.0 s: the muxed file keeps all of it (the old
    # '-shortest' behaviour truncated it to the 0.5 s remaining audio - F2).
    _assert_full_picture(dest, expected_seconds=1.0)
    audio = [s for s in _ffprobe_json(dest)["streams"] if s.get("codec_type") == "audio"]
    assert len(audio) == 1
    assert 0.2 <= float(audio[0]["duration"]) <= 0.8  # only the window remains
    assert not _mux_temp(dest).exists()


# --- F2 regression: source audio shorter than the encoded picture ----------

def test_mux_audio_short_source_audio_keeps_full_picture(tmp_path):
    """2.0 s picture + 0.9 s audio must NOT yield a 0.9 s (truncated) picture."""
    src = _make_av_clip(tmp_path, video_seconds=2, audio_seconds=0.9,
                        name="short_audio_src.mp4")
    video_only = _strip_audio(tmp_path, src)
    assert abs(probe_duration(video_only) - 2.0) <= 0.1
    dest = tmp_path / "muxed_short_audio.mp4"
    assert mux_audio(video_only, src, dest) is True
    assert has_audio_stream(dest) is True
    _assert_full_picture(dest, expected_seconds=2.0)
    video = [s for s in _ffprobe_json(dest)["streams"] if s.get("codec_type") == "video"]
    assert len(video) == 1 and float(video[0]["duration"]) >= 1.8
    assert not _mux_temp(dest).exists()


# --- F1 regression: cut window entirely past the source audio end ----------

@pytest.mark.parametrize("kwargs", [
    {"start_seconds": 5.0},
    {"start_seconds": 5.0, "duration_seconds": 0.5},
])
def test_mux_audio_cut_window_past_audio_end_fails_closed(tmp_path, kwargs):
    """A window past the audio end must fail closed, never publish a wrong picture.

    Pre-fix ffmpeg returned True after os.replace()-ing a video-less (audio-only)
    file over the encoded output - total picture loss. It also silently restarts
    the audio track when an input seek runs past EOF (verified: -ss 5 on a 1 s
    track yields the whole 1.023 s track), so a "True" would ship the entire
    track under a picture that starts later. Contract: False, encoded video_only
    untouched, no temp file, no destination written.
    """
    src = _make_av_clip(tmp_path, video_seconds=1, audio_seconds=1, name="av_src.mp4")
    video_only = _strip_audio(tmp_path, src)
    before = video_only.read_bytes()
    dest = tmp_path / "muxed_past_audio.mp4"

    result = mux_audio(video_only, src, dest, **kwargs)

    assert result is False
    assert video_only.read_bytes() == before
    assert not dest.exists()
    assert not _mux_temp(dest).exists()


def test_mux_audio_cut_window_partially_past_audio_end_keeps_full_picture(tmp_path):
    """Audio shorter than the window: keep the whole picture plus the audio that exists."""
    src = _make_av_clip(tmp_path, video_seconds=2, audio_seconds=1, name="partial_src.mp4")
    video_only = _strip_audio(tmp_path, src)
    dest = tmp_path / "muxed_partial_window.mp4"
    assert mux_audio(video_only, src, dest, start_seconds=0.5, duration_seconds=1.0) is True
    assert has_audio_stream(dest) is True
    _assert_full_picture(dest, expected_seconds=2.0)  # picture full, not audio-truncated
    audio = [s for s in _ffprobe_json(dest)["streams"] if s.get("codec_type") == "audio"]
    assert len(audio) == 1
    # 0.5 s of the 1.0 s track remains after the 0.5 s seek - never the whole track.
    assert 0.3 <= float(audio[0]["duration"]) <= 0.7
    assert not _mux_temp(dest).exists()


def test_mux_audio_fails_closed_when_ffprobe_unavailable(monkeypatch, tmp_path):
    """Without ffprobe the result cannot be verified, so nothing may be published."""
    from apps.anime_upscaler_gui.anime_upscaler_gui import ffmpeg as ffmpeg_mod

    src = _make_clip(tmp_path, with_audio=True)
    video_only = _strip_audio(tmp_path, src)
    dest = tmp_path / "unverified.mp4"
    monkeypatch.setattr(ffmpeg_mod, "ffprobe_available", lambda: False)
    assert mux_audio(video_only, src, dest) is False
    assert not dest.exists()
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
