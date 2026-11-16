"""Task 8: the output path must never resolve to the input file.

Data loss: with output_mode=custom, the output folder set to the input file own
folder and an empty suffix (or output_mode=same_folder with an empty batch
folder name), compute_output_path() returns the input path itself. The worker
used to open an ffmpeg encoder on that destination; ffmpeg -y truncates the
file before a single frame is decoded, so a job that reports rc=0 "finished"
has destroyed the source (observed red: real 1 s clip, 3x,xxx bytes -> ~1,9xx
bytes, terminal event "finished").

These tests pin all three layers of the fix:

  * validate_output_path() - pure same-file detection (brief 8.3),
  * the enqueue site        - user-facing error, nothing is queued (8.4),
  * the worker              - refuses to touch files at run time (8.5).

Import deviation: the brief uses the plain anime_upscaler_gui.* path; pytest
runs from the repo root where that resolves to the outer docstring-only marker
package, so every file uses the apps-prefixed path (established Task 4
convention).
"""
from __future__ import annotations

import os
import queue
import shutil
import types
from pathlib import Path

import numpy as np
import pytest

from apps.anime_upscaler_gui.anime_upscaler_gui.app import UpscaleGUI
from apps.anime_upscaler_gui.anime_upscaler_gui.controllers.job_builder import (
    compute_output_path,
    validate_output_path,
)
from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline import worker as worker_mod
from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline.jobs import RunJob


# --- brief 8.1: pure helper -------------------------------------------------

def test_same_folder_and_empty_suffix_is_rejected(tmp_path):
    src = tmp_path / "ep01.mp4"
    out = compute_output_path(src, "custom", "", "batch", str(tmp_path))
    assert out == src
    with pytest.raises(ValueError, match="equals the input"):
        validate_output_path(src, out)


def test_normal_custom_output_is_allowed(tmp_path):
    src = tmp_path / "ep01.mp4"
    out = compute_output_path(src, "custom", "_x4", "batch", str(tmp_path / "out"))
    validate_output_path(src, out)  # must not raise


def test_empty_custom_folder_is_rejected(tmp_path):
    src = tmp_path / "ep01.mp4"
    with pytest.raises(ValueError, match="output folder"):
        validate_output_path(src, Path(""))


# --- extended same-file cases (Lead: spelling / relative / case) ------------

@pytest.mark.parametrize("spelling", [
    "nested/../ep01.mp4",          # different spelling, same file
    "./ep01.mp4",                  # relative spelling
    "nested/sibling.mp4",          # distinct nested file (allowed)
])
def test_same_file_via_different_spelling(tmp_path, spelling):
    src = tmp_path / "ep01.mp4"
    src.write_bytes(b"payload")
    out = tmp_path / spelling
    if spelling.endswith("sibling.mp4"):
        validate_output_path(src, out)  # a distinct file is fine
        return
    with pytest.raises(ValueError, match="equals the input"):
        validate_output_path(src, out)


@pytest.mark.skipif(os.name != "nt",
                    reason="case-insensitive filesystem semantics")
def test_same_file_via_different_case_is_rejected(tmp_path):
    src = tmp_path / "ep01.mp4"
    src.write_bytes(b"payload")
    with pytest.raises(ValueError, match="equals the input"):
        validate_output_path(src, tmp_path / "EP01.MP4")


def test_output_inside_the_input_folder_but_distinct_is_allowed(tmp_path):
    src = tmp_path / "ep01.mp4"
    validate_output_path(src, tmp_path / "exports" / "ep01_x4.mp4")
    validate_output_path(src, tmp_path / "ep01_x4.mp4")


def test_same_folder_mode_with_empty_batch_folder_is_rejected(tmp_path):
    src = tmp_path / "ep01.mp4"
    # same_folder + empty batch folder name -> <input parent>/<stem><suffix><ext>
    out = compute_output_path(src, "same_folder", "", "", str(tmp_path))
    assert out == src
    with pytest.raises(ValueError, match="equals the input"):
        validate_output_path(src, out)


# --- app layer: live widget values + enqueue refuses before touching files ---

class _FakeVar:
    def __init__(self, value=""):
        self.value = value

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


def _fake_app(tmp_path, mode="custom", suffix="", output_dir=None):
    """Just enough app surface for the real _compute_output_path to run."""
    app = types.SimpleNamespace()
    app.settings = types.SimpleNamespace(data=types.SimpleNamespace(
        output_mode="same_folder",        # stale saved settings
        single_file_suffix="_stale",
        batch_folder_name="stale_batch",
    ))
    app.output_panel = types.SimpleNamespace(
        output_mode_var=_FakeVar(mode),
        suffix_var=_FakeVar(suffix),
        batch_folder_var=_FakeVar(""),
        output_dir_var=_FakeVar(output_dir if output_dir is not None
                                else str(tmp_path)),
    )
    app._compute_output_path = UpscaleGUI._compute_output_path.__get__(
        app, type(app))
    app._compute_output_path_or_none = (
        UpscaleGUI._compute_output_path_or_none.__get__(app, type(app)))
    return app


def test_compute_output_path_reads_live_widgets_not_saved_settings(tmp_path):
    """The suffix/radios have no trace: settings.data is stale until Save."""
    app = _fake_app(tmp_path, mode="custom", suffix="",
                    output_dir=str(tmp_path))
    src = tmp_path / "ep01.mp4"
    out = app._compute_output_path(src)
    assert out == src, "live widgets (empty suffix + own folder) must be used"
    assert "stale_batch" not in str(out)


def test_enqueue_rejects_same_file_and_touches_nothing(tmp_path, monkeypatch):
    import apps.anime_upscaler_gui.anime_upscaler_gui.app as app_mod

    errors = []
    monkeypatch.setattr(app_mod.messagebox, "showerror",
                        lambda title, msg: errors.append((title, msg)))
    app = _fake_app(tmp_path, mode="custom", suffix="",
                    output_dir=str(tmp_path))
    src = tmp_path / "ep01.mp4"
    src.write_bytes(b"payload")

    assert app._compute_output_path_or_none(src) is None
    assert errors and "equals the input" in errors[0][1]
    assert "Output folder" == errors[0][0]
    assert src.read_bytes() == b"payload"  # nothing opened/truncated it


# --- worker layer: never touch a destination that is the source -----------

def _make_real_source(tmp_path):
    """A real, decodable 1 s clip; its bytes must survive the job."""
    import cv2

    src = tmp_path / "ep01.mp4"
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 25.0,
                             (64, 64))
    assert writer.isOpened()
    for _ in range(25):
        frame = np.full((64, 64, 3), 128, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    assert src.is_file() and src.stat().st_size > 0
    return src


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_worker_refuses_to_encode_over_the_source(monkeypatch, tmp_path):
    """A run whose output path equals the input must leave the source intact.

    The decoder/model are faked (this is about the guard, not super-res) but
    the ffmpeg encoder is real: before the guard existed this exact harness
    truncated the source to a small partial file and reported "finished".
    """
    src = _make_real_source(tmp_path)
    before = src.read_bytes()

    class _FakeReader:
        def __init__(self, frames):
            self._frames = list(frames)
            self.fps, self.total = 25.0, len(self._frames)
            self.w, self.h = 64, 64

        def __iter__(self):
            return iter(self._frames)

        def release(self):
            pass

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

    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(25)]
    monkeypatch.setattr(worker_mod, "_PyAvReader", lambda p: _FakeReader(frames))
    model = _ScaleModel()
    monkeypatch.setattr(worker_mod, "build", lambda kind, ckpt: model)
    monkeypatch.setattr(worker_mod, "_make_backend",
                        lambda *a, **k: (_FakeBackend(model), "PyTorch"))
    monkeypatch.setattr(worker_mod.torch.cuda, "is_available", lambda: False)

    job = RunJob(
        job_id=8, input_path=src, output_path=src,  # <-- the bug
        is_video=True, model_filename="dummy.pth", kind="realesrgan",
        scale=4, outscale=4.0, fp16=False, device="cpu", batch_size=1,
        decode="pyav", prefetch="sync", pin_memory="off",
        downscale_max_edge=0, gpu_guard_mode="off", tile_size=0,
        tile_overlap=0, tta=False, use_tensorrt=False, use_nvenc=False,
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
    assert kinds[-1] == "error", kinds
    assert "finished" not in kinds, "a refused job must not be reported DONE"
    err = [e for e in evts if e.kind == "error"][-1]
    assert "output path equals input path" in err.message, err.message
    assert src.read_bytes() == before, "the source file was modified"
