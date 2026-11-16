"""A failed ffmpeg encode must never produce a finished event (Task 7).

Two layers of regression:

  7.1  pins the ffmpeg behaviour the worker must inspect: a container that
       cannot take the chosen codec exits non-zero (the worker used to ignore
       the exit code and report the job DONE).
  7.6  pins the shared decision helper: a failed/empty encode becomes an
       "error" event, the partial output is deleted and no "finished" follows.

Import deviation: the brief uses the plain anime_upscaler_gui.* path; pytest
is run from the repo root, where that resolves to the outer docstring-only
marker package. Every existing GUI test imports the apps-prefixed path, so
this file follows that convention (documented Task 4 deviation).
"""
from __future__ import annotations

import shutil

import pytest

from apps.anime_upscaler_gui.anime_upscaler_gui.ffmpeg import open_encoder

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                reason="ffmpeg required")


def test_ffmpeg_rejects_mismatched_container(tmp_path):
    # .webm cannot take libx264; ffmpeg exits non-zero. This pins the
    # behaviour that the worker must inspect instead of ignoring.
    pipe = open_encoder(tmp_path / "bad.webm", fps=10.0, w=64, h=64)
    assert pipe is not None
    import numpy as np
    for _ in range(3):
        try:
            pipe.stdin.write(np.zeros((64, 64, 3), dtype=np.uint8).tobytes())
        except (BrokenPipeError, OSError):
            break
    pipe.stdin.close()
    rc = pipe.wait(timeout=30)
    assert rc != 0


def test_failed_encode_emits_error_not_finished(tmp_path):
    from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline import worker as worker_mod

    events = []

    class FakeWorker:
        def emit(self, evt):
            events.append(evt)

    job = type("J", (), {"job_id": 1})()
    worker_mod._handle_encode_result(FakeWorker(), job, tmp_path / "x.mp4", 0, 1)
    assert [e.kind for e in events] == ["error"]
    assert not (tmp_path / "x.mp4").exists()


# --- whole-path regression: per-frame path ignores rc today, must not ------
def test_per_frame_failed_encode_is_error_not_finished(monkeypatch, tmp_path):
    """A non-zero encoder exit on the per-frame path: ERROR, no finished, no file."""
    import queue

    import numpy as np

    from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline import worker as worker_mod
    from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline.jobs import RunJob

    class _FakeReader:
        def __init__(self, frames, fps=25.0):
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
            return (x.repeat_interleave(4, dim=-1)
                     .repeat_interleave(4, dim=-2))

    class _FakeBackend:
        def __init__(self, model):
            self.model = model

        def __call__(self, x):
            return self.model(x)

    class _FailingPipe:
        """Encoder stand-in that receives frames but exits non-zero."""

        def __init__(self, out_path, rc=1):
            self._out_path = out_path
            self._rc = rc
            pipe = self

            class _Stdin:
                closed = False

                def write(self, data):
                    pipe._out_path.write_bytes(b"partial-encode")

                def close(self):
                    self.closed = True

            self.stdin = _Stdin()

        def wait(self, timeout=None):
            return self._rc

        def kill(self):
            pass

    frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(3)]
    in_path = tmp_path / "in.mp4"
    in_path.write_bytes(b"not a real video")
    out_path = tmp_path / "out.mp4"
    monkeypatch.setattr(worker_mod, "_PyAvReader",
                        lambda p: _FakeReader(frames))
    monkeypatch.setattr(worker_mod, "open_encoder",
                        lambda *a, **k: _FailingPipe(out_path))
    model = _ScaleModel()
    monkeypatch.setattr(worker_mod, "build", lambda kind, ckpt: model)
    monkeypatch.setattr(worker_mod, "_make_backend",
                        lambda *a, **k: (_FakeBackend(model), "PyTorch"))
    monkeypatch.setattr(worker_mod.torch.cuda, "is_available", lambda: False)

    job = RunJob(
        job_id=7, input_path=in_path, output_path=out_path, is_video=True,
        model_filename="dummy.pth", kind="realesrgan", scale=4, outscale=4.0,
        fp16=False, device="cpu", batch_size=1, decode="pyav", prefetch="sync",
        pin_memory="off", downscale_max_edge=0, gpu_guard_mode="off",
        tile_size=0, tile_overlap=0, tta=False, use_tensorrt=False,
        use_nvenc=False,
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
    assert "finished" not in kinds, "a failed encode must never be reported DONE"
    err = [e for e in evts if e.kind == "error"][-1]
    assert "encoder failed (exit 1)" in err.message, err.message
    assert not out_path.exists(), "the failed encode output must be discarded"
