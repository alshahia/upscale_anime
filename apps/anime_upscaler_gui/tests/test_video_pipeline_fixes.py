"""Tests for the video-pipeline fixes: NVENC functional probe, cut-window
handling with unknown frame counts, decoder-choice honoring, skip wrapper.

No Tk needed -- pure backend behavior.
"""
import subprocess

import pytest

from apps.anime_upscaler_gui.anime_upscaler_gui import decoders, ffmpeg, pipeline


# --------------------------------------------------------------------------- #
# NVENC functional probe
# --------------------------------------------------------------------------- #

@pytest.fixture
def reset_nvenc_cache():
    old = ffmpeg._NVENC_SUPPORTED
    ffmpeg._NVENC_SUPPORTED = None  # force re-probe
    yield
    ffmpeg._NVENC_SUPPORTED = old  # restore prior cache state


def test_nvenc_probe_false_when_no_ffmpeg(reset_nvenc_cache, monkeypatch):
    monkeypatch.setattr(ffmpeg, "ffmpeg_available", lambda: False)
    assert ffmpeg._detect_nvenc_support() is False


def test_nvenc_probe_actual_encode_failure_falls_back(reset_nvenc_cache, monkeypatch):
    # Encoder listed but open fails (driver mismatch / no capable device)
    # must report False so open_encoder uses libx264.
    monkeypatch.setattr(ffmpeg, "ffmpeg_available", lambda: True)

    class R:
        returncode = 1
        stdout = ""
        stderr = "No capable devices found"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())
    assert ffmpeg._detect_nvenc_support() is False


def test_nvenc_probe_actual_encode_success(reset_nvenc_cache, monkeypatch):
    monkeypatch.setattr(ffmpeg, "ffmpeg_available", lambda: True)

    class R:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())
    assert ffmpeg._detect_nvenc_support() is True


def test_open_encoder_falls_back_to_libx264_when_nvenc_unavailable(
        reset_nvenc_cache, monkeypatch):
    monkeypatch.setattr(ffmpeg, "ffmpeg_available", lambda: True)
    monkeypatch.setattr(ffmpeg, "_detect_nvenc_support", lambda: False)
    probed = {}

    def fake_popen(cmd, **kwargs):
        probed["cmd"] = cmd
        return object()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    handle = ffmpeg.open_encoder("out.mp4", 24.0, 64, 48, use_nvenc=True)
    assert handle is not None
    assert "libx264" in probed["cmd"]


# --------------------------------------------------------------------------- #
# Unknown-length video handling (PyAV stream.frames == None -> total 0)
# --------------------------------------------------------------------------- #

class FakeReader:
    def __init__(self, n):
        self.n = n
        self.released = False

    def __iter__(self):
        return iter(range(self.n))

    def release(self):
        self.released = True


def test_batched_unbounded_limit_yields_all_frames():
    frames = list(range(7))
    out = list(pipeline._batched(iter(frames), batch_size=1, limit=0))
    assert [f for _, f in out] == frames


def test_skip_first_frames_wrapper_skips_and_yields():
    r = FakeReader(10)
    wrapped = decoders._SkipFirstFrames(r, 3)
    consumed = list(wrapped)
    assert consumed == [3, 4, 5, 6, 7, 8, 9]
    wrapped.release()
    assert r.released


def test_skip_first_frames_zero_keeps_all():
    r = FakeReader(5)
    wrapped = decoders._SkipFirstFrames(r, 0)
    assert list(wrapped) == [0, 1, 2, 3, 4]


def test_skip_first_frames_skip_past_end_is_safe():
    r = FakeReader(2)
    wrapped = decoders._SkipFirstFrames(r, 99)
    assert list(wrapped) == []
