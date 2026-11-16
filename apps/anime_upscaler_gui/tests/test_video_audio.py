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


# --- worker-level wiring: the per-frame path must use the frame-exact window --
def test_per_frame_worker_muxes_the_encoded_window(monkeypatch, tmp_path):
    """I1 wiring: _run_video must hand the frame-exact window to the mux.

    A head-only cut (Start > 0, End = 00:00:00) trims the picture, so the
    remux window must start at the first encoded frame and last as long as the
    encoded picture. Raw cut fields would be start=0.42 and "to EOF"; the
    frame-exact values are 10/25 = 0.40 for both, so reverting _run_video to
    the raw cut values fails these assertions (the bug this issue fixed).
    """
    import queue

    import numpy as np

    from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline import worker as worker_mod
    from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline.jobs import RunJob

    fps = 25.0
    cut_start_seconds = 0.42  # 10.5 frames -> start_frame = round(10.5) = 10
    total_frames = 20
    expected_frames = 10  # 0.42s start, no end -> frames 10..20
    expected_window = expected_frames / fps  # 0.40 s

    class _FakeReader:
        def __init__(self, frames):
            self._frames = list(frames)
            self.fps = fps
            self.total = len(self._frames)
            self.w, self.h = 16, 16
            self.released = False

        def __iter__(self):
            return iter(self._frames)

        def release(self):
            self.released = True

    class _ScaleModel:
        scale = 4

        def to(self, *_a, **_k):
            return self

        def eval(self):
            return self

        def half(self):
            return self

        def __call__(self, x):
            return x.repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2)

    class _FakeBackend:
        def __init__(self, model):
            self.model = model

        def __call__(self, x):
            return self.model(x)

    class _OkPipe:
        """Encoder stand-in that received frames and exits 0."""

        def __init__(self, out_path):
            self._out_path = out_path
            pipe = self

            class _Stdin:
                def write(self, data):
                    pipe._out_path.write_bytes(b"encoded")

                def close(self):
                    pass

            self.stdin = _Stdin()

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(total_frames)]
    in_path = tmp_path / "in.mp4"
    in_path.write_bytes(b"not a real video")
    out_path = tmp_path / "out.mp4"

    monkeypatch.setattr(worker_mod, "_PyAvReader", lambda p: _FakeReader(frames))
    monkeypatch.setattr(worker_mod, "open_encoder",
                        lambda *a, **k: _OkPipe(out_path))
    model = _ScaleModel()
    monkeypatch.setattr(worker_mod, "build", lambda kind, ckpt: model)
    monkeypatch.setattr(worker_mod, "_make_backend",
                        lambda *a, **k: (_FakeBackend(model), "PyTorch"))
    monkeypatch.setattr(worker_mod.torch.cuda, "is_available", lambda: False)

    calls = []

    def _record_mux(_out, _inp, *, start_seconds=0.0, duration_seconds=None):
        calls.append({"start_seconds": start_seconds,
                      "duration_seconds": duration_seconds})
        return True

    monkeypatch.setattr(worker_mod, "_mux_job_audio", _record_mux)

    job = RunJob(
        job_id=11, input_path=in_path, output_path=out_path, is_video=True,
        model_filename="dummy.pth", kind="realesrgan", scale=4, outscale=4.0,
        fp16=False, device="cpu", batch_size=1, decode="pyav", prefetch="sync",
        pin_memory="off", downscale_max_edge=0, gpu_guard_mode="off",
        tile_size=0, tile_overlap=0, tta=False, use_tensorrt=False,
        use_nvenc=False, cut_start_seconds=cut_start_seconds,
        cut_end_seconds=0.0,
    )
    w = worker_mod.PipelineWorker(queue.Queue(), queue.Queue())
    w._run_one(job)

    evts = []
    while True:
        try:
            evts.append(w.out_queue.get_nowait())
        except queue.Empty:
            break
    kinds = [e.kind for e in evts]
    assert kinds[-1] == "finished", kinds

    assert len(calls) == 1, "the per-frame path must remux exactly once"
    assert calls[0]["start_seconds"] == pytest.approx(expected_window), (
        "the mux must start at the first encoded frame, not the raw cut start")
    assert calls[0]["duration_seconds"] == pytest.approx(expected_window), (
        "the mux duration must follow the encoded picture, not run to EOF")
    assert calls[0]["start_seconds"] != pytest.approx(cut_start_seconds), (
        "raw cut fields (0.42s) must not leak into the mux window")
