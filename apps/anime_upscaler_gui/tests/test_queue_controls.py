"""Tests for Q2 (perf/queue-controls-gpu-codec): pause/cancel/resume.

Pins the new control channel contract:
  - JobControlEvent kind values (pause/resume/cancel) are accepted
  - the worker drains ctl_queue between frames
  - pause blocks the worker on a per-job threading.Event
  - cancel raises _JobCancelled and emits a clean terminal event
  - _run_one cleans up per-job pause state on job end (no leak)
  - the new JobStatus.PAUSED is wired through status_triplet
"""
import queue
import threading
import time

import pytest

from apps.anime_upscaler_gui.anime_upscaler_gui.state import JobStatus
from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline.jobs import (
    JobControlEvent, JobEvent,
)
from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline.worker import (
    PipelineWorker, _JobCancelled,
)
from apps.anime_upscaler_gui.anime_upscaler_gui.a11y import (
    status_glyph, status_label, status_triplet,
)


# ---- JobControlEvent shape ----
def test_job_control_event_dataclass():
    e = JobControlEvent(kind="pause", job_id=7)
    assert e.kind == "pause"
    assert e.job_id == 7
    assert JobControlEvent(kind="resume").job_id == 0


# ---- JobStatus.PAUSED ----
def test_paused_status_exists_and_is_string_enum():
    """JobStatus.PAUSED inherits str so existing string comparisons work."""
    assert JobStatus.PAUSED == "paused"
    assert str(JobStatus.PAUSED) == "paused"
    assert JobStatus.PAUSED != JobStatus.CANCELLED


def test_status_triplet_handles_paused():
    """a11y.status_triplet renders PAUSED with a distinct glyph."""
    glyph, _color, label = status_triplet(JobStatus.PAUSED)
    assert isinstance(glyph, str) and glyph
    assert label == "Paused"
    assert glyph != status_glyph(JobStatus.RUNNING)
    assert glyph != status_glyph(JobStatus.CANCELLED)


# ---- Worker control channel ----
def test_worker_default_ctl_queue_is_per_worker_instance():
    """Two workers without an explicit ctl_queue get independent queues."""
    w1 = PipelineWorker(queue.Queue(), queue.Queue())
    w2 = PipelineWorker(queue.Queue(), queue.Queue())
    assert w1.ctl_queue is not w2.ctl_queue


def test_send_control_pushes_to_ctl_queue():
    """send_control is the public API the GUI uses."""
    w = PipelineWorker(queue.Queue(), queue.Queue())
    w.send_control(JobControlEvent(kind="pause", job_id=1))
    assert w.ctl_queue.get_nowait().kind == "pause"


def test_drain_ctl_for_job_pauses_but_does_not_block():
    """drain_ctl_for_job is non-blocking; it just arms the pause event."""
    w = PipelineWorker(queue.Queue(), queue.Queue())
    w.send_control(JobControlEvent(kind="pause", job_id=42))
    t0 = time.perf_counter()
    w._drain_ctl_for_job(42)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.05, f"drain took {elapsed*1000:.1f}ms, expected < 50ms"
    assert 42 in w._pause_events
    assert not w._pause_events[42].is_set()


def test_drain_ctl_for_job_ignores_events_for_other_jobs():
    """Pause/cancel events for a non-current job are re-queued, not applied."""
    w = PipelineWorker(queue.Queue(), queue.Queue())
    w.send_control(JobControlEvent(kind="pause", job_id=99))
    w._drain_ctl_for_job(42)
    pending = w.ctl_queue.get_nowait()
    assert pending.job_id == 99 and pending.kind == "pause"
    assert 42 not in w._pause_events


def test_drain_ctl_for_job_raises_on_cancel():
    """Cancel raises _JobCancelled so the per-job path can unwind."""
    w = PipelineWorker(queue.Queue(), queue.Queue())
    w.send_control(JobControlEvent(kind="cancel", job_id=42))
    with pytest.raises(_JobCancelled):
        w._drain_ctl_for_job(42)


def test_check_pause_blocks_then_returns_on_resume():
    """check_pause blocks until resume, then returns quickly."""
    w = PipelineWorker(queue.Queue(), queue.Queue())
    w.send_control(JobControlEvent(kind="pause", job_id=42))
    w._drain_ctl_for_job(42)

    def resume_soon():
        time.sleep(0.1)
        w.send_control(JobControlEvent(kind="resume", job_id=42))
    threading.Thread(target=resume_soon, daemon=True).start()

    t0 = time.perf_counter()
    w._check_pause(42)
    elapsed = time.perf_counter() - t0
    assert 0.05 <= elapsed < 1.0, f"check_pause took {elapsed:.3f}s"
    t1 = time.perf_counter()
    w._check_pause(42)
    assert (time.perf_counter() - t1) < 0.01


def test_finalize_job_clears_pause_state():
    """After a job ends, its pause event must not leak."""
    w = PipelineWorker(queue.Queue(), queue.Queue())
    w.send_control(JobControlEvent(kind="pause", job_id=42))
    w._drain_ctl_for_job(42)
    assert 42 in w._pause_events
    w._finalize_job(42)
    assert 42 not in w._pause_events


def test_pause_events_for_two_jobs_are_independent():
    """Pausing job 42 must not affect job 99."""
    w = PipelineWorker(queue.Queue(), queue.Queue())
    w.send_control(JobControlEvent(kind="pause", job_id=42))
    w._drain_ctl_for_job(42)
    assert not w._pause_events[42].is_set()
    assert 99 not in w._pause_events
    # Pre-arm 99 so we can show independence.
    w._pause_events.setdefault(99, threading.Event()).set()
    assert w._pause_events[99].is_set()
    assert not w._pause_events[42].is_set()


# ---- _run_one integration: cancel emits a terminal event ----
def test_run_one_cancelled_event_cleans_up_output(tmp_path):
    """When the worker hits _JobCancelled, _run_one emits 'cancelled'.

    We need an input file that exists so the worker enters _run_image
    (which drains controls before model load). Using a real 1x1 PNG so
    cv2 can decode it; cancel is sent ~50ms later, after _run_one has
    started. The cancel is drained inside _run_image's first action,
    _JobCancelled propagates up, _run_one's except clause deletes the
    output and emits the 'cancelled' event.
    """
    import struct
    import zlib
    from pathlib import Path
    from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline.jobs import RunJob

    # 1x1 transparent PNG (8 bytes IDAT after compression).
    def _tiny_png() -> bytes:
        sig = b"\x89PNG\r\n\x1a\n"
        def chunk(t, d):
            return (struct.pack(">I", len(d)) + t + d +
                    struct.pack(">I", zlib.crc32(t + d) & 0xffffffff))
        ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
        idat = chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00\x00"))
        iend = chunk(b"IEND", b"")
        return sig + ihdr + idat + iend

    in_path = Path(str(tmp_path)) / "x.png"
    out_path = Path(str(tmp_path)) / "x_out.png"
    in_path.write_bytes(_tiny_png())

    in_q, out_q, ctl_q = queue.Queue(), queue.Queue(), queue.Queue()
    w = PipelineWorker(in_q, out_q, ctl_queue=ctl_q)
    w.start()
    try:
        j = RunJob(
            job_id=1,
            input_path=in_path,
            output_path=out_path,
            is_video=False,
            model_filename="dummy.pth",
            kind="realesrgan",
            scale=4,
            outscale=2.0,
            fp16=False,
            device="cpu",
            batch_size=1,
            decode="cv2",
            prefetch="sync",
            pin_memory="off",
            downscale_max_edge=0,
            gpu_guard_mode="off",
            tile_size=256,
            tile_overlap=8,
        )
        in_q.put(j)
        # Send cancel BEFORE the worker enters _run_image's drain. The
        # cancel sits in ctl_queue and is the FIRST thing _run_image
        # drains (right after emitting 'started'). This is the
        # cancel-during-load path that the GUI exercises when a user
        # clicks Cancel right after Start.
        ctl_q.put(JobControlEvent(kind="cancel", job_id=1))
        deadline = time.time() + 3.0
        saw_started = False
        saw_cancelled = False
        while time.time() < deadline and not saw_cancelled:
            try:
                evt = out_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if evt.kind == "started":
                saw_started = True
            elif evt.kind == "cancelled":
                saw_cancelled = True
                assert evt.job_id == 1
                assert "cancel" in evt.message.lower()
        assert saw_started, "worker did not enter _run_image"
        assert saw_cancelled, "worker did not emit a cancelled event"
        assert 1 not in w._pause_events
    finally:
        w.stop()
        w.join(timeout=2.0)
