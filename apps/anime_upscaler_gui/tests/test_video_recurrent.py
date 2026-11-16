"""Tests for the recurrent (AnimeSR) video + image paths.

Regression pins for the block-level defects in _run_video_recurrent /
_run_image:

  B1  "error" is terminal -- no "finished" follows it, the partial output
      file is unlinked, and a failing job advances the job queue exactly
      once (a duplicate "finished" would push a second job).
  H1  pause blocks the running job at chunk granularity, cancel raises
      _JobCancelled and unlinks the output, and "progress" events with
      the same payload as _run_video are emitted while it runs.
  H2  a long cut window is consumed in sliding windows of at most
      Capability.recurrent_frames frames, so a single forward never
      holds the whole clip and VRAM stays bounded by the window.
  M3  a recurrent kind still image goes through the 5-D (b, n, c, h, w)
      contract instead of dying in MSRSW.forward.
"""
import queue
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline import worker as worker_mod
from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline.jobs import (
    JobControlEvent, JobEvent, RunJob,
)
from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline.worker import PipelineWorker


KIND = "animesr"           # recurrent: Capability.recurrent_frames == 3
FRAME = 16                 # LR frame edge; pad_multiple 4 -> no padding
SCALE = 4
FPS = 25.0


# ---- fakes ----
def _frames(n):
    """N grayscale frames with an increasing, identifiable value."""
    return [np.full((FRAME, FRAME, 3), int(255 * i / max(n - 1, 1)),
                    dtype=np.uint8) for i in range(n)]


class _FakeReader:
    """Decoder stand-in: iterable of pre-built RGB frames."""

    def __init__(self, frames, fps=FPS):
        self._frames = list(frames)
        self.fps = fps
        self.total = len(self._frames)
        self.w = FRAME
        self.h = FRAME
        self.released = False

    def __iter__(self):
        return iter(self._frames)

    def release(self):
        self.released = True


class _FakeRecurrentModel:
    """MSRSWVSR stand-in: (b, n, c, h, w) -> (b, n, c, h*s, w*s).

    Upscaling each input value as-is means the value written into the
    encoded output identifies which source frame produced it, which is
    how the chunker tests prove "every frame exactly once, in order".
    """
    scale = SCALE

    def __init__(self, gate=None):
        self.shapes = []          # every input shape, in call order
        self.calls = 0
        self.entered = threading.Event()
        self._gate = gate

    def forward(self, x):
        self.calls += 1
        self.shapes.append(tuple(int(v) for v in x.shape))
        self.entered.set()
        if self._gate is not None:
            self._gate.wait(timeout=10.0)
        return (x.repeat_interleave(self.scale, dim=-1)
                 .repeat_interleave(self.scale, dim=-2))

    def eval(self):
        return self

    def to(self, *_a, **_k):
        return self

    def half(self):
        return self

    def __call__(self, x):
        # a plain module call, the way nn.Module.__call__ would dispatch
        return self.forward(x)


class _FailAfterModel(_FakeRecurrentModel):
    """Forwards the first `fail_after` chunks, then raises."""

    def __init__(self, fail_after):
        super().__init__()
        self.fail_after = fail_after

    def forward(self, x):
        if self.calls >= self.fail_after:
            raise RuntimeError("recurrent boom")
        return super().forward(x)


class _FakeBackend:
    """backend.model holder; callable like the real backends."""

    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        return self.model(x)


class _FakePipe:
    """Stand-in for the ffmpeg Popen returned by open_encoder."""

    class _Stdin:
        def __init__(self, out_path):
            self._out_path = out_path
            self.values = []        # first pixel of every written frame
            self.closed = False

        def write(self, data):
            arr = np.frombuffer(data, dtype=np.uint8)
            self.values.append(int(arr[0]) if arr.size else -1)
            # The real encoder creates the file when it spawns; emulate
            # that so the partial-output cleanup contract is testable.
            self._out_path.write_bytes(b"")

        def close(self):
            self.closed = True

    def __init__(self, out_path):
        self.stdin = self._Stdin(out_path)
        self.waited = False
        self.killed = False

    def wait(self, timeout=None):
        self.waited = True
        return 0

    def kill(self):
        self.killed = True


# ---- helpers ----
def _force_cpu(monkeypatch):
    """Pin the device to CPU so the happy path never touches the GPU."""
    monkeypatch.setattr(worker_mod.torch.cuda, "is_available", lambda: False)


def _worker():
    return PipelineWorker(queue.Queue(), queue.Queue())


def _drain(q):
    evts = []
    while True:
        try:
            evts.append(q.get_nowait())
        except queue.Empty:
            return evts


def _kinds(evts):
    return [e.kind for e in evts]


def _job(tmp_path, **kw):
    """A recurrent VIDEO job wired to the fakes in this module."""
    in_path = Path(str(tmp_path)) / "in.mp4"
    in_path.write_bytes(b"not a real video")
    out_path = Path(str(tmp_path)) / "out.mp4"
    d = dict(
        job_id=1,
        input_path=in_path,
        output_path=out_path,
        is_video=True,
        model_filename="dummy.pth",
        kind=KIND,
        scale=4,
        outscale=4.0,
        fp16=False,
        device="cpu",
        batch_size=1,
        decode="pyav",
        prefetch="sync",
        pin_memory="off",
        downscale_max_edge=0,
        gpu_guard_mode="off",
        tile_size=0,
        tile_overlap=0,
        cut_start_seconds=0.0,
        cut_end_seconds=0.0,
        video_crf=23,
        video_preset="medium",
        use_tensorrt=False,
        use_nvenc=False,
    )
    d.update(kw)
    return RunJob(**d)


def _install(monkeypatch, model, pipe, frames):
    """Point every I/O seam the recurrent path touches at the fakes."""
    monkeypatch.setattr(worker_mod, "_PyAvReader", lambda p: _FakeReader(frames))
    monkeypatch.setattr(worker_mod, "open_encoder", lambda *a, **k: pipe)
    monkeypatch.setattr(worker_mod, "build", lambda kind, ckpt: model)
    monkeypatch.setattr(worker_mod, "_make_backend",
                            lambda *a, **k: (_FakeBackend(model), "PyTorch"))


def _wait_for(fn, timeout=10.0):
    """Poll ``fn`` until it returns truthy; return its last value."""
    deadline = time.time() + timeout
    val = False
    while time.time() < deadline:
        val = fn()
        if val:
            return val
        time.sleep(0.02)
    return val

# ---- H2: the sliding-window chunker + VRAM pre-flight ----
def test_recurrent_windows_caps_chunk_at_recurrent_frames():
    """H2: a long clip is yielded as overlapping windows of <= N frames."""
    wins = list(worker_mod._recurrent_windows(iter(range(20)), 3, 2, 0))
    assert wins, "chunker yielded nothing"
    assert all(len(w) <= 3 for w, _skip in wins)
    assert all(skip < len(w) for w, skip in wins)
    out = []
    for w, skip in wins:
        out.extend(w[skip:])
    assert out == list(range(20)), "every frame once, in order"


def test_recurrent_windows_share_one_context_frame():
    """window-1 stride keeps a neighbour, so no hard temporal cut."""
    wins = list(worker_mod._recurrent_windows(iter(range(7)), 3, 2, 0))
    assert [w for w, _skip in wins] == [[0, 1, 2], [2, 3, 4], [4, 5, 6]]
    assert [skip for _w, skip in wins] == [0, 1, 1]


def test_recurrent_windows_short_clip_and_limit():
    out = []
    for w, skip in worker_mod._recurrent_windows(iter(range(5)), 3, 2, 0):
        out.extend(w[skip:])
    assert out == list(range(5)), "clip shorter than one window still runs"

    # a 3-frame limit must stop after the first window
    wins = list(worker_mod._recurrent_windows(iter(range(20)), 3, 2, 3))
    assert wins == [([0, 1, 2], 0)]


def test_recurrent_vram_estimate_scales_with_window():
    """H2: the pre-flight budget grows with frames, shrinks with fp16."""
    one = worker_mod._recurrent_vram_estimate(1, 540, 960, False, 4)
    three = worker_mod._recurrent_vram_estimate(3, 540, 960, False, 4)
    half = worker_mod._recurrent_vram_estimate(3, 540, 960, True, 4)
    assert three > one
    assert half < three
    # 1080p x 3 frames in fp16 is a GB-scale figure, not a runaway one
    assert 1e9 < worker_mod._recurrent_vram_estimate(3, 1080, 1920, True, 4) < 20e9


# ---- B1: "error" is terminal ----
def test_recurrent_error_is_terminal_and_unlinks_partial(monkeypatch, tmp_path):
    """B1: a backend failure must not also emit "finished"."""
    _force_cpu(monkeypatch)
    out_path = Path(str(tmp_path)) / "out.mp4"
    pipe = _FakePipe(out_path)
    # chunks 0/1/2 succeed (5 frames written), chunk 3 raises
    model = _FailAfterModel(fail_after=3)
    job = _job(tmp_path)
    _install(monkeypatch, model, pipe, _frames(20))
    w = _worker()
    w._run_one(job)

    kinds = _kinds(_drain(w.out_queue))
    assert kinds[0] == "started"
    assert "error" in kinds, "backend failure must surface an error event"
    assert "finished" not in kinds, (
        "B1: a failed job must not also emit finished -- the GUI would "
        "relabel it DONE and the app would enqueue the next job twice")
    assert kinds[-1] == "error"
    # 3 successful chunks (3 + 2 + 2 frames) -> a real partial output
    assert len(pipe.stdin.values) == 7, "expected a partial output to exist"
    assert not out_path.exists(), "partial output must be unlinked"
    assert pipe.stdin.closed and pipe.waited, "encoder was not closed"


def test_recurrent_error_event_reports_stage_and_exception(monkeypatch, tmp_path):
    _force_cpu(monkeypatch)
    pipe = _FakePipe(Path(str(tmp_path)) / "out.mp4")
    job = _job(tmp_path, job_id=11)
    _install(monkeypatch, _FailAfterModel(fail_after=1), pipe, _frames(20))
    w = _worker()
    w._run_one(job)
    errs = [e for e in _drain(w.out_queue) if e.kind == "error"]
    assert len(errs) == 1
    assert "recurrent inference" in errs[0].message
    assert errs[0].job_id == 11
    assert isinstance(errs[0].exc, RuntimeError)


def test_failed_recurrent_job_advances_queue_once(monkeypatch, tmp_path):
    """B1: one terminal event per job -> exactly one follow-up job.

    The error path used to fall through to "finished" as well, so the
    GUI enqueued the next job from both branches and two jobs ran
    concurrently. Here two failing jobs must run strictly in sequence,
    each emitting exactly one terminal event.
    """
    _force_cpu(monkeypatch)
    pipe = _FakePipe(Path(str(tmp_path)) / "out.mp4")
    _install(monkeypatch, _FailAfterModel(fail_after=3), pipe, _frames(20))
    in_q, out_q = queue.Queue(), queue.Queue()
    w = PipelineWorker(in_q, out_q)
    in_q.put(_job(tmp_path, job_id=1))
    in_q.put(_job(tmp_path, job_id=2))
    w.start()
    try:
        seen, terminals = [], []
        deadline = time.time() + 60.0
        while time.time() < deadline and len(terminals) < 2:
            try:
                evt = out_q.get(timeout=0.2)
            except queue.Empty:
                continue
            seen.append(evt)
            if evt.kind in ("finished", "error", "cancelled", "fatal_error"):
                terminals.append(evt)
        assert [e.kind for e in terminals] == ["error", "error"]
        assert not any(e.kind == "finished" for e in seen)
        # The next job must not start until the previous one terminated
        # (the bug pushed a second job from the fall-through finished).
        first_j2 = next(i for i, e in enumerate(seen) if e.job_id == 2)
        last_j1 = max(i for i, e in enumerate(seen)
                      if e.job_id == 1 and e.kind == "error")
        assert first_j2 > last_j1, \
            "job 2 started before job 1 emitted its terminal event"
        assert all(e.job_id == 1 for e in seen[:last_j1 + 1])
    finally:
        w.stop()
        w.join(timeout=5.0)


# ---- H1: pause / cancel / progress at chunk granularity ----
def test_pause_blocks_recurrent_job_until_resume(monkeypatch, tmp_path):
    """H1: the running recurrent job honours a pause."""
    _force_cpu(monkeypatch)
    gate = threading.Event()
    model = _FakeRecurrentModel(gate=gate)
    pipe = _FakePipe(Path(str(tmp_path)) / "out.mp4")
    job = _job(tmp_path, job_id=7)
    _install(monkeypatch, model, pipe, _frames(20))
    w = _worker()
    t = threading.Thread(target=w._run_video_recurrent,
                         args=(job, _FakeBackend(model), torch.device("cpu")),
                         daemon=True)
    t.start()
    try:
        assert model.entered.wait(10), "first forward never started"
        # Pause while the worker is inside forward #1; it is drained at the
        # next chunk boundary, between chunks, not mid-forward.
        w.send_control(JobControlEvent(kind="pause", job_id=7))
        gate.set()
        paused = _wait_for(lambda: [
            e for e in _drain(w.out_queue)
            if e.kind == "log" and "waiting on resume" in e.message])
        assert paused, "worker never reported waiting on resume (H1)"
        assert model.calls == 1, "a paused job kept calling the model"
        assert t.is_alive(), "a paused job terminated instead of pausing"
        assert not any(e.kind in ("finished", "error")
                         for e in _drain(w.out_queue))

        w.send_control(JobControlEvent(kind="resume", job_id=7))
        t.join(timeout=30)
        assert not t.is_alive(), "job did not finish after resume"
        assert _kinds(_drain(w.out_queue))[-1] == "finished"
    finally:
        gate.set()


def test_cancel_while_paused_unlinks_partial_output(monkeypatch, tmp_path):
    """H1: cancel interrupts the pause and leaves no partial output."""
    _force_cpu(monkeypatch)
    gate = threading.Event()
    model = _FakeRecurrentModel(gate=gate)
    out_path = Path(str(tmp_path)) / "out.mp4"
    pipe = _FakePipe(out_path)
    job = _job(tmp_path, job_id=9)
    _install(monkeypatch, model, pipe, _frames(20))
    w = _worker()
    box = []

    def run():
        try:
            w._run_one(job)
        except BaseException as e:  # pragma: no cover - would be a bug
            box.append(e)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    try:
        assert model.entered.wait(10), "first forward never started"
        w.send_control(JobControlEvent(kind="pause", job_id=9))
        gate.set()
        paused = _wait_for(lambda: [
            e for e in _drain(w.out_queue)
            if e.kind == "log" and "waiting on resume" in e.message])
        assert paused, "worker never paused (H1)"
        assert model.calls == 1
        assert out_path.exists(), "expected a partial output to exist"

        # cancel from the paused state: _check_pause must see it
        w.send_control(JobControlEvent(kind="cancel", job_id=9))
        t.join(timeout=30)
        assert not t.is_alive(), "cancel did not interrupt the pause"
        kinds = _kinds(_drain(w.out_queue))
        assert "cancelled" in kinds and kinds[-1] == "cancelled"
        assert "error" not in kinds and "finished" not in kinds
        assert not out_path.exists(), "cancel must unlink the partial output"
        assert box == [], "_JobCancelled must be contained by _run_one"
        assert pipe.stdin.closed, "encoder pipe was not closed"
        assert 9 not in w._pause_events, "per-job pause state leaked"
    finally:
        gate.set()


def test_recurrent_emits_progress_events(monkeypatch, tmp_path):
    """H1 + H2: chunked inference emits the same progress payload as
    _run_video, so the status bar advances instead of freezing."""
    _force_cpu(monkeypatch)
    frames = _frames(20)
    model = _FakeRecurrentModel()
    pipe = _FakePipe(Path(str(tmp_path)) / "out.mp4")
    job = _job(tmp_path, job_id=5)
    _install(monkeypatch, model, pipe, frames)
    w = _worker()
    w._run_video_recurrent(job, _FakeBackend(model), torch.device("cpu"))
    evts = _drain(w.out_queue)
    kinds = _kinds(evts)
    assert kinds[0] == "started" and kinds[-1] == "finished"

    # H2: 20 frames must not be a single forward
    assert model.calls > 1, "20 frames were sent as one forward"
    assert all(s[1] <= 3 for s in model.shapes), model.shapes
    assert model.shapes[0] == (1, 3, 3, FRAME, FRAME)
    # every call keeps the 5-D (b, n, c, h, w) contract, h/w unpadded
    assert all(s[0] == 1 and s[2] == 3 and s[3] == FRAME and s[4] == FRAME
               for s in model.shapes), model.shapes

    # H1: progress carries the _run_video keys
    prog = [e for e in evts if e.kind == "progress"]
    assert len(prog) >= 2, kinds
    assert prog[-1].progress == pytest.approx(1.0)
    assert all(e.job_id == 5 for e in prog)
    assert all(e.fps and e.fps > 0 for e in prog)
    assert all(isinstance(e.infer_ms, float) for e in prog)
    assert all(isinstance(e.frame_idx, int) and e.frame_idx >= 0
                for e in prog)

    # H2: every output frame written exactly once, in order
    assert pipe.stdin.values == [int(f[0, 0, 0]) for f in frames]
    assert pipe.stdin.closed


# ---- M7: encoder start failure ----
def test_encoder_failure_reports_clean_error_and_releases_reader(
        monkeypatch, tmp_path):
    """M7: a failed open_encoder must not leak the reader."""
    _force_cpu(monkeypatch)
    model = _FakeRecurrentModel()
    readers = []

    def make_reader(_path):
        r = _FakeReader(_frames(4))
        readers.append(r)
        return r

    _install(monkeypatch, model, None, _frames(4))
    monkeypatch.setattr(worker_mod, "_PyAvReader", make_reader)

    def boom(*a, **k):
        raise OSError("open_encoder failed")

    monkeypatch.setattr(worker_mod, "open_encoder", boom)
    job = _job(tmp_path)
    w = _worker()
    w._run_one(job)

    kinds = _kinds(_drain(w.out_queue))
    assert kinds[0] == "started"
    assert "error" in kinds, kinds
    assert "finished" not in kinds
    assert kinds[-1] == "error"
    assert readers and all(r.released for r in readers), "reader leaked"


# ---- M3: recurrent still image ----
def test_recurrent_image_job_runs_5d_path(monkeypatch, tmp_path):
    """M3: an AnimeSR image job must reach the model with 5-D input."""
    _force_cpu(monkeypatch)
    model = _FakeRecurrentModel()
    in_path = Path(str(tmp_path)) / "in.png"
    in_path.write_bytes(b"")
    out_path = Path(str(tmp_path)) / "out.png"
    saved = {}
    monkeypatch.setattr(worker_mod, "build", lambda kind, ckpt: model)
    monkeypatch.setattr(worker_mod, "_make_backend",
                            lambda *a, **k: (_FakeBackend(model), "PyTorch"))
    monkeypatch.setattr(worker_mod, "load_image_rgb", lambda p: _frames(1)[0])
    monkeypatch.setattr(worker_mod, "save_image_rgb",
                            lambda rgb, p: (saved.update(shape=rgb.shape[:2])
                                             or True))
    job = RunJob(
        job_id=3,
        input_path=in_path,
        output_path=out_path,
        is_video=False,
        model_filename="dummy.pth",
        kind=KIND,
        scale=4,
        outscale=4.0,
        fp16=False,
        device="cpu",
        batch_size=1,
        decode="cv2",
        prefetch="sync",
        pin_memory="off",
        downscale_max_edge=0,
        gpu_guard_mode="off",
        tile_size=0,
        tile_overlap=0,
    )
    w = _worker()
    w._run_one(job)
    evts = _drain(w.out_queue)
    assert _kinds(evts) == ["started", "finished"], _kinds(evts)
    # (b=1, n=recurrent_frames, c=3, h, w) -- not a 4-D still-image tensor
    assert model.shapes == [(1, 3, 3, FRAME, FRAME)], model.shapes
    assert saved["shape"] == (FRAME * SCALE, FRAME * SCALE)


# ---- M4: the backend does not reload the checkpoint ----
def test_pytorch_backend_reuses_supplied_model(monkeypatch):
    """M4: model= must skip the second disk load."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline.backends import (
        _PyTorchBackend,
    )

    built = []

    def fake_build(kind, ckpt):
        built.append(ckpt)
        return torch.nn.Identity()

    monkeypatch.setattr(worker_mod, "build", fake_build)
    import apps.anime_upscaler_gui.anime_upscaler_gui.pipeline.backends as bk
    monkeypatch.setattr(bk, "build", fake_build)

    ready = torch.nn.Identity()
    be = _PyTorchBackend(Path("x.pth"), "realesrgan", torch.device("cpu"),
                           False, model=ready)
    assert be.model is ready, "model= was ignored / rebuilt"
    assert built == [], "model= must not re-read the checkpoint from disk"

    # the legacy signature still builds from disk
    be2 = _PyTorchBackend(Path("x.pth"), "realesrgan", torch.device("cpu"), False)
    assert built == [Path("x.pth")]
    assert be2.model is not None

# ---- I1: audio remux window follows the encoded cut ----
def test_recurrent_start_only_cut_remuxes_the_encoded_window(monkeypatch, tmp_path):
    """I1: Start-only cut (End=00:00:00 means "to EOF") must not mux from 0:00.

    The recurrent path must hand the remux helper the *effective* window: the
    frame-exact start the decoder skipped to and a duration matching the frames
    that were encoded -- not cut_end - cut_start (which is negative here).
    """
    _force_cpu(monkeypatch)
    pipe = _FakePipe(Path(str(tmp_path)) / "out.mp4")
    model = _FakeRecurrentModel()
    _install(monkeypatch, model, pipe, _frames(20))
    calls = []
    monkeypatch.setattr(
        worker_mod, "_mux_job_audio",
        lambda out_path, input_path, **kw: calls.append(kw) or True)

    # fps=25, 20 decoded frames; Start=0.4 -> skip 10 frames -> 10 encoded.
    job = _job(tmp_path, cut_start_seconds=0.4, cut_end_seconds=0.0)
    w = _worker()
    w._run_one(job)

    kinds = _kinds(_drain(w.out_queue))
    assert kinds[-1] == "finished", kinds
    assert len(calls) == 1, "audio must be remuxed exactly once"
    assert calls[0]["start_seconds"] == pytest.approx(10 / FPS), (
        "audio start must follow the picture's first encoded frame")
    # The fake encoder writes an empty file, so probe_duration fails and the
    # helper falls back to the encoded frame count / fps (10 frames @ 25 fps).
    assert calls[0]["duration_seconds"] == pytest.approx(10 / FPS), (
        "audio duration must match the encoded picture, not the full source")
