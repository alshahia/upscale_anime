"""Tests for the video-pipeline fixes: NVENC functional probe, cut-window
handling with unknown frame counts, decoder-choice honoring, skip wrapper.

No Tk needed -- pure backend behavior.
"""
import queue
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from apps.anime_upscaler_gui.anime_upscaler_gui import decoders, ffmpeg, pipeline
from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline import worker as worker_mod
from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline.jobs import (
    ABORT_JOB,
    JobControlEvent,
    RunJob,
)
from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline.worker import PipelineWorker


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
def test_nvenc_probe_uses_gpu_min_frame_size():
    # NVENC rejects frames below its minimum supported dimension (~145px),
    # so the probe must use a frame size NVENC actually accepts -- probing
    # at 64x64 falsely disabled NVENC on capable GPUs (Quadro RTX 4000).
    import re
    from pathlib import Path
    src = Path(ffmpeg.__file__).read_text(encoding="utf-8")
    m = re.search(r"color=black:s=(\d+)x(\d+)", src)
    assert m, "probe lavfi size not found"
    w, h = int(m.group(1)), int(m.group(2))
    assert min(w, h) >= 145

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


def test_pipeline_uses_skip_first_frames_wrapper():
    # Regression: pipeline.py must import the skip wrapper (a NameError at
    # runtime surfaced in _run_video when decode=pyav).
    assert pipeline._SkipFirstFrames is decoders._SkipFirstFrames


def test_open_encoder_bounds_x264_memory(monkeypatch):
    """libx264 at 4K needs a threads/lookahead cap, or x264 defaults allocate
       one working set per core (gigabytes of committed RAM on many-core
       machines)."""
    monkeypatch.setattr(ffmpeg, "ffmpeg_available", lambda: True)
    probed = {}

    def fake_popen(cmd, **kwargs):
        probed["cmd"] = cmd
        return object()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    ffmpeg.open_encoder("out.mp4", 24.0, 3840, 2160, use_nvenc=False)
    cmd = probed["cmd"]
    assert "-threads" in cmd, "encoder thread cap missing"
    assert "-x264-params" in cmd, "lookahead cap missing"
    params = " ".join(cmd)
    assert "rc-lookahead=" in params and "lookahead-threads=" in params


# --------------------------------------------------------------------------- #
# Task 9: abort / cancel semantics
#
# Contract pinned here:
#   * ABORT_JOB in the frame-error modal is a cancel, not a worker crash.
#   * A cancel that arrives during encode / during the audio remux is honoured
#     before any "finished" is emitted: exactly one terminal event, "cancelled",
#     never "finished" after it.
#   * The partial output is unlinked and the worker keeps consuming the queue.
#   * A stale cancel (unknown job id, or after the job finished) must not
#     cancel another job.
# --------------------------------------------------------------------------- #
TERMINALS = ("finished", "error", "cancelled", "fatal_error")
FRAME = 32
FPS = 25.0


def _frames(n=10):
    return [np.full((FRAME, FRAME, 3), int(255 * i / max(n - 1, 1)),
                    dtype=np.uint8) for i in range(n)]


class _FakePipe:
    """Stand-in for the ffmpeg Popen returned by open_encoder.

    The real encoder creates the destination when it spawns (before the first
    frame), so the fake does too: an abort before the first write must still
    find a partial file to clean up.
    """

    class _Stdin:
        def __init__(self, out_path):
            self._out_path = out_path
            self.closed = False
            self.writes = 0
            self.close_error = None

        def write(self, data):
            self.writes += 1
            self._out_path.write_bytes(b"encoded")

        def close(self):
            self.closed = True
            if self.close_error is not None:
                raise self.close_error

    def __init__(self, out_path, rc=0):
        self.out_path = Path(out_path)
        self.out_path.write_bytes(b"encoder-placeholder")
        self.stdin = self._Stdin(self.out_path)
        self.rc = rc
        self.waited = False
        self.killed = False

    def wait(self, timeout=None):
        self.waited = True
        return self.rc

    def kill(self):
        self.killed = True


class _FakeReader:
    """Decoder stand-in; optionally injects a control event mid-stream."""

    def __init__(self, frames, ctl=None, inject_after=None, event=None):
        self._frames = list(frames)
        self.fps = FPS
        self.total = len(self._frames)
        self.w = self.h = FRAME
        self.release_calls = 0
        self.released_after_iter = False
        self._iterated = False
        self._ctl = ctl
        self._inject_after = inject_after
        self._event = event
        self._injected = False

    def __iter__(self):
        self._iterated = True
        for i, frame in enumerate(self._frames):
            if (self._ctl is not None and not self._injected
                    and i == self._inject_after):
                self._ctl.put(self._event)
                self._injected = True
            yield frame

    def release(self):
        self.release_calls += 1
        if self._iterated:
            self.released_after_iter = True


class _FakeModel:
    scale = 2

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        if self.fail:
            raise RuntimeError("synthetic frame failure")
        return (x.repeat_interleave(2, dim=-1)
                 .repeat_interleave(2, dim=-2))

    def eval(self):
        return self

    def to(self, *_a, **_k):
        return self

    def half(self):
        return self


class _FakeBackend:
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        return self.model(x)


def _job(tmp_path, job_id=1, **kw):
    in_path = Path(str(tmp_path)) / "in.mp4"
    in_path.write_bytes(b"not a real video")
    d = dict(
        job_id=job_id,
        input_path=in_path,
        output_path=Path(str(tmp_path)) / "out.mp4",
        is_video=True,
        model_filename="dummy.pth",
        kind="realesrgan",
        scale=2,
        outscale=2.0,
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
        video_preset="ultrafast",
        use_tensorrt=False,
        use_nvenc=False,
        cascade_mode=1,
    )
    d.update(kw)
    return RunJob(**d)


def _install(monkeypatch, readers, model, pipe_factory=None):
    """Point the per-frame path decode/build/encode seams at fakes.

    The readers dict maps the job input's parent directory to its fake reader,
    so one test can script two jobs with different behaviour. The audio remux
    is stubbed to succeed: it is covered by the real-ffmpeg test below and by
    the ffmpeg module tests.
    """
    monkeypatch.setattr(worker_mod, "_PyAvReader",
                        lambda path: readers[str(Path(str(path)).parent)])
    monkeypatch.setattr(worker_mod, "build", lambda kind, ckpt: model)
    monkeypatch.setattr(worker_mod, "_make_backend",
                        lambda *a, **k: (_FakeBackend(model), "PyTorch"))
    monkeypatch.setattr(worker_mod, "_mux_job_audio", lambda *a, **k: True)
    if pipe_factory is not None:
        monkeypatch.setattr(worker_mod, "open_encoder", pipe_factory)
    monkeypatch.setattr(worker_mod.torch.cuda, "is_available", lambda: False)


def _drain_queue(q):
    evts = []
    while True:
        try:
            evts.append(q.get_nowait())
        except queue.Empty:
            return evts


def _terminal(evts):
    return [e for e in evts if e.kind in TERMINALS]


def _quiet(out_q, seconds=0.3):
    evts = []
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            evts.append(out_q.get(timeout=0.05))
        except queue.Empty:
            pass
    return evts


def _start_worker(ctl_queue=None):
    in_q = queue.Queue()
    out_q = queue.Queue()
    w = PipelineWorker(in_q, out_q, ctl_queue=ctl_queue)
    w.start()
    return w, in_q, out_q


def _collect_terminals(out_q, n, timeout=60.0):
    seen = []
    terminals = []
    deadline = time.time() + timeout
    while time.time() < deadline and len(terminals) < n:
        try:
            evt = out_q.get(timeout=0.1)
        except queue.Empty:
            continue
        seen.append(evt)
        if evt.kind in TERMINALS:
            terminals.append(evt)
    return seen, terminals


def _stop(w):
    w.stop()
    w.join(timeout=5.0)
    assert not w.is_alive(), "worker thread did not exit"


# --- 9.3: ABORT_JOB is a cancellation -------------------------------------
def test_abort_job_raises_job_cancelled_not_the_frame_error():
    """9.3: ABORT_JOB must map to _JobCancelled so the output is removed.

    Source-level pin (cheap, deterministic): the modal path must raise the
    cancellation type instead of re-raising the frame error, which escaped
    _run_one and surfaced as a "worker crashed" terminal error with the
    partial output still on disk.
    """
    import inspect
    src = inspect.getsource(worker_mod.PipelineWorker._run_video)
    assert "raise _JobCancelled()" in src
    assert "raise e" not in src


def test_abort_job_is_a_clean_cancel(monkeypatch, tmp_path):
    """ABORT_JOB from the frame-error modal: cancelled, not crashed.

    Red before the fix: terminal event "error" ("worker crashed: synthetic
    frame failure") and the partial output left behind.
    """
    pipe = _FakePipe(Path(str(tmp_path)) / "out.mp4")
    job = _job(tmp_path, job_id=3,
               on_frame_error=lambda idx, msg: ABORT_JOB)
    reader = _FakeReader(_frames(10))
    _install(monkeypatch, {str(tmp_path): reader}, _FakeModel(fail=True),
             lambda *a, **k: pipe)
    w = PipelineWorker(queue.Queue(), queue.Queue())
    w._run_one(job)
    evts = _drain_queue(w.out_queue)
    kinds = [e.kind for e in evts]
    terminals = _terminal(evts)
    assert "frame_error" in kinds, "the modal must still be prompted"
    assert len(terminals) == 1, (
        "expected exactly one terminal, got %s"
        % [e.kind for e in terminals])
    assert terminals[0].kind == "cancelled", (
        "ABORT_JOB must be a clean cancel, got %r: %r"
        % (terminals[0].kind, terminals[0].message))
    assert "finished" not in kinds
    assert "error" not in kinds, "abort surfaced as a worker crash"
    assert not job.output_path.exists(), "partial output must be unlinked"


# --- cancel mid-encode -----------------------------------------------------
@pytest.mark.skipif(not ffmpeg.ffmpeg_available(),
                    reason="real ffmpeg encoder required")
def test_cancel_mid_encode_stops_real_ffmpeg_encoder(monkeypatch, tmp_path):
    """Cancel while a REAL ffmpeg encoder is mid-stream: prompt, clean stop.

    Decode/inference are faked (deterministic) but the encoder is the real
    libx264 subprocess. The control event is injected after the 3rd frame, so
    the cancel is processed while the encoder pipe is open with frames
    already written.
    """
    ctl_q = queue.Queue()
    reader = _FakeReader(_frames(80), ctl=ctl_q, inject_after=3,
                         event=JobControlEvent(kind="cancel", job_id=5))
    job = _job(tmp_path, job_id=5)
    _install(monkeypatch, {str(tmp_path): reader}, _FakeModel())
    w, in_q, out_q = _start_worker(ctl_q)
    started = time.time()
    try:
        in_q.put(job)
        seen, terminals = _collect_terminals(out_q, 1, timeout=60.0)
        elapsed = time.time() - started
    finally:
        _stop(w)
    kinds = [e.kind for e in seen]
    assert len(terminals) == 1, [e.kind for e in terminals]
    assert terminals[0].kind == "cancelled", (
        "cancel mid-encode produced %r: %r"
        % (terminals[0].kind, terminals[0].message))
    assert "finished" not in kinds
    assert not job.output_path.exists(), "partial output must be unlinked"
    assert reader.released_after_iter, "decoder must be released"
    assert elapsed < 20.0, "cancel took %.1fs, not prompt" % elapsed


# --- cancel during the audio remux ----------------------------------------
def test_cancel_during_audio_mux_is_not_reported_as_finished(monkeypatch, tmp_path):
    """A cancel that lands while the audio remux runs must not be lost.

    The mock mux queues the cancel and returns success immediately. Red before
    the fix: the job reported "finished" even though a cancel was pending.
    """
    pipe = _FakePipe(Path(str(tmp_path)) / "out.mp4")
    job = _job(tmp_path, job_id=8)
    reader = _FakeReader(_frames(10))
    _install(monkeypatch, {str(tmp_path): reader}, _FakeModel(),
             lambda *a, **k: pipe)
    w = PipelineWorker(queue.Queue(), queue.Queue())
    mux_calls = []

    def fake_mux(out_path, input_path, *, start_seconds=0.0,
                 duration_seconds=None):
        mux_calls.append((out_path, input_path))
        w.send_control(JobControlEvent(kind="cancel", job_id=job.job_id))
        return True

    monkeypatch.setattr(worker_mod, "_mux_job_audio", fake_mux)
    w._run_one(job)
    evts = _drain_queue(w.out_queue)
    kinds = [e.kind for e in evts]
    terminals = _terminal(evts)
    assert mux_calls, "the audio remux never ran; harness is wrong"
    assert len(terminals) == 1, [e.kind for e in terminals]
    assert terminals[0].kind == "cancelled", (
        "cancel during mux produced %r: %r"
        % (terminals[0].kind, terminals[0].message))
    assert "finished" not in kinds, "finished after cancel violates the contract"
    assert not job.output_path.exists(), "partial output must be unlinked"


# --- queue survives a cancel; every terminal is exactly-once ---------------
def test_cancel_mid_encode_keeps_worker_alive_for_next_job(monkeypatch, tmp_path):
    """After a cancelled job the worker must run the next queued job."""
    d1 = Path(str(tmp_path)) / "a"
    d2 = Path(str(tmp_path)) / "b"
    d1.mkdir()
    d2.mkdir()
    ctl_q = queue.Queue()
    readers = {
        str(d1): _FakeReader(_frames(40), ctl=ctl_q, inject_after=2,
                             event=JobControlEvent(kind="cancel", job_id=1)),
        str(d2): _FakeReader(_frames(5)),
    }
    job1 = _job(d1, job_id=1)
    job2 = _job(d2, job_id=2)
    _install(monkeypatch, readers, _FakeModel(),
             lambda out_path, *a, **k: _FakePipe(out_path))
    w, in_q, out_q = _start_worker(ctl_q)
    try:
        in_q.put(job1)
        in_q.put(job2)
        seen, terminals = _collect_terminals(out_q, 2, timeout=60.0)
        assert [t.kind for t in terminals] == ["cancelled", "finished"], (
            "terminal trace: %s"
            % [(t.job_id, t.kind) for t in terminals])
        assert terminals[0].job_id == 1 and terminals[1].job_id == 2
        assert not job1.output_path.exists(), "cancelled job left a partial output"
        assert job2.output_path.exists(), "next job did not run"
        assert w.is_alive(), "worker thread died on the cancel path"
    finally:
        _stop(w)
    later = _terminal(_quiet(out_q))
    assert later == [], [e.kind for e in later]


# --- stale cancels must not cancel another job ----------------------------
def test_stale_cancel_for_unknown_job_does_not_cancel_next(monkeypatch, tmp_path):
    """A cancel for a job id that is not running is a no-op."""
    ctl_q = queue.Queue()
    ctl_q.put(JobControlEvent(kind="cancel", job_id=999))
    pipe = _FakePipe(Path(str(tmp_path)) / "out.mp4")
    job = _job(tmp_path, job_id=4)
    _install(monkeypatch, {str(tmp_path): _FakeReader(_frames(5))},
             _FakeModel(), lambda *a, **k: pipe)
    w = PipelineWorker(queue.Queue(), queue.Queue(), ctl_queue=ctl_q)
    w._run_one(job)
    evts = _drain_queue(w.out_queue)
    kinds = [e.kind for e in evts]
    assert "cancelled" not in kinds, "stale cancel cancelled an unrelated job"
    assert kinds[-1] == "finished"
    assert job.output_path.exists()


def test_cancel_after_completion_is_a_noop(monkeypatch, tmp_path):
    """A late cancel for a finished job must not add a terminal event."""
    d1 = Path(str(tmp_path)) / "a"
    d2 = Path(str(tmp_path)) / "b"
    d1.mkdir()
    d2.mkdir()
    ctl_q = queue.Queue()
    pipes = {
        str(d1): _FakePipe(d1 / "out.mp4"),
        str(d2): _FakePipe(d2 / "out.mp4"),
    }
    readers = {str(d1): _FakeReader(_frames(5)),
               str(d2): _FakeReader(_frames(5))}
    _install(monkeypatch, readers, _FakeModel(),
             lambda out_path, *a, **k: pipes[str(Path(out_path).parent)])
    w = PipelineWorker(queue.Queue(), queue.Queue(), ctl_queue=ctl_q)
    job1 = _job(d1, job_id=1)
    w._run_one(job1)
    evts1 = _drain_queue(w.out_queue)
    assert _terminal(evts1)[-1].kind == "finished"
    # The user clicks Cancel just as the job finished; the worker is already
    # on the next job when the control event is drained.
    w.send_control(JobControlEvent(kind="cancel", job_id=1))
    job2 = _job(d2, job_id=2)
    w._run_one(job2)
    evts2 = _drain_queue(w.out_queue)
    kinds2 = [e.kind for e in evts2]
    assert "cancelled" not in kinds2, "late cancel produced a spurious cancelled"
    assert _terminal(evts2)[-1].kind == "finished"
    assert job2.output_path.exists()


def test_cancel_before_job_starts_is_a_clean_cancel(monkeypatch, tmp_path):
    """A queued job cancelled before it starts: clean cancel, no output."""
    ctl_q = queue.Queue()
    job = _job(tmp_path, job_id=6)
    ctl_q.put(JobControlEvent(kind="cancel", job_id=6))
    _install(monkeypatch, {str(tmp_path): _FakeReader(_frames(5))},
             _FakeModel(),
             lambda *a, **k: _FakePipe(Path(str(tmp_path)) / "never.mp4"))
    w = PipelineWorker(queue.Queue(), queue.Queue(), ctl_queue=ctl_q)
    w._run_one(job)
    evts = _drain_queue(w.out_queue)
    kinds = [e.kind for e in evts]
    terminals = _terminal(evts)
    assert kinds[0] == "started"
    assert [t.kind for t in terminals] == ["cancelled"]
    assert "finished" not in kinds
    assert not job.output_path.exists(), "a job cancelled before start wrote output"


# --- Task 7 review follow-up: mid-stream encoder death cleanup ------------
def _dying_pipe(out_path, die_at=2, close_error=None):
    """A fake encoder whose pipe stops accepting writes after die_at frames."""
    pipe = _FakePipe(out_path)
    real_write = pipe.stdin.write

    def dying_write(data):
        if pipe.stdin.writes + 1 >= die_at:
            raise OSError(22, "Invalid argument")
        real_write(data)

    pipe.stdin.write = dying_write
    pipe.stdin.close_error = close_error
    return pipe


def test_encode_death_mid_stream_unlinks_partial_and_releases_reader(
        monkeypatch, tmp_path):
    """A mid-stream encoder death is a clean terminal error, not a crash.

    Task 7 review follow-up: the generic exception path used to bypass
    _handle_encode_result and _unlink_partial_output, leaving the truncated
    file on disk (and the reader unreleased if the pipe close failed).
    """
    out_path = Path(str(tmp_path)) / "out.mp4"
    pipe = _dying_pipe(out_path)
    reader = _FakeReader(_frames(20))
    job = _job(tmp_path, job_id=12)
    _install(monkeypatch, {str(tmp_path): reader}, _FakeModel(),
             lambda *a, **k: pipe)
    w = PipelineWorker(queue.Queue(), queue.Queue())
    try:
        w._run_one(job)
    except Exception as exc:  # pre-fix: the encoder error escaped _run_one
        escaped = exc
    else:
        escaped = None
    evts = _drain_queue(w.out_queue)
    kinds = [e.kind for e in evts]
    terminals = _terminal(evts)
    assert escaped is None, "exception escaped _run_one: %r" % (escaped,)
    assert [t.kind for t in terminals] == ["error"], [t.kind for t in terminals]
    assert "finished" not in kinds
    assert "video encoder process died mid-stream" in terminals[0].message
    assert not out_path.exists(), "truncated partial output must be unlinked"
    assert reader.released_after_iter, "decoder must be released"


def test_encode_death_close_failure_still_unlinks_and_releases(
        monkeypatch, tmp_path):
    """A failing pipe.stdin.close() must not skip reader release / cleanup."""
    out_path = Path(str(tmp_path)) / "out.mp4"
    pipe = _dying_pipe(out_path, close_error=OSError("pipe already broken"))
    reader = _FakeReader(_frames(20))
    job = _job(tmp_path, job_id=13)
    _install(monkeypatch, {str(tmp_path): reader}, _FakeModel(),
             lambda *a, **k: pipe)
    w = PipelineWorker(queue.Queue(), queue.Queue())
    try:
        w._run_one(job)
    except Exception as exc:
        escaped = exc
    else:
        escaped = None
    evts = _drain_queue(w.out_queue)
    terminals = _terminal(evts)
    assert escaped is None, "exception escaped _run_one: %r" % (escaped,)
    assert [t.kind for t in terminals] == ["error"], [t.kind for t in terminals]
    assert "finished" not in [e.kind for e in evts]
    assert not out_path.exists(), "truncated partial output must be unlinked"
    assert reader.released_after_iter, "reader release was skipped by close()"

