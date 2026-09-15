"""End-to-end video test: generates a real 1-second mp4 and runs the full
pipeline worker (the same backend the GUI job queue drives) over it using the
real pre-trained AnimeSR_v2 checkpoint and the real ffmpeg encoder (an NVENC-less
machine exercises the bounded libx264 fallback).

No Tk. Skipped automatically when the checkpoint is missing so the suite stays
green without pre-trained weights.
"""
import queue
from pathlib import Path

import cv2
import numpy as np
import pytest

from apps.anime_upscaler_gui.anime_upscaler_gui import pipeline

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODEL_PATH = _REPO_ROOT / "pretrained" / "AnimeSR_v2.pth"

try:
    import av  # noqa: F401
    _has_pyav = True
except Exception:
    _has_pyav = False


def _make_one_second_video(path: Path, width=64, height=64, fps=25.0) -> int:
    """Write a synthetic 1-second test clip (moving bright square)."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    assert writer.isOpened(), f"cv2 VideoWriter failed to open {path}"
    n_frames = int(round(fps))  # exactly one second
    block = max(8, height // 8)
    for i in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., 2] = 128  # static background lane so the model sees detail
        x = int((width - block) * i / max(n_frames - 1, 1))
        frame[block:block * 2, x:x + block] = (255, 255, 255)
        writer.write(frame)
    writer.release()
    return n_frames


def _probe_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _run_worker(video_path: Path, out_path: Path, decode: str):
    """Drive a real _PipelineWorker to a terminal event; return events."""
    in_q, out_q = queue.Queue(), queue.Queue()
    worker = pipeline._PipelineWorker(in_q, out_q)
    worker.start()
    job = pipeline._RunJob(
        job_id=1,
        input_path=video_path,
        output_path=out_path,
        is_video=True,
        model_filename=str(_MODEL_PATH),
        kind="animesr",
        scale=4,
        outscale=4.0,
        fp16=False,
        device="cpu",
        batch_size=1,
        decode=decode,
        prefetch="async",
        pin_memory="off",
        downscale_max_edge=0,
        gpu_guard_mode="off",
        tile_size=0,
        tile_overlap=0,
        tta=True,  # deliberately ON: animesr must gate it via Capability.tta
        use_tensorrt=False,
        use_nvenc=True,   # functional probe falls back to libx264 when unusable
        video_crf=23,
    )
    in_q.put(job)
    events = []
    try:
        while True:
            evt = out_q.get(timeout=1800)
            events.append(evt)
            if evt.kind in ("finished", "error", "fatal_error"):
                break
    finally:
        worker.stop()
        worker.join(timeout=10)
    return events


@pytest.mark.skipif(
    not _MODEL_PATH.is_file(),
    reason="pretrained/AnimeSR_v2.pth not found",
)
@pytest.mark.parametrize("decode", ["cv2", "pyav"])
def test_one_second_video_end_to_end(decode, tmp_path):
    if decode == "pyav" and not _has_pyav:
        pytest.skip("PyAV (av) not installed")
    src = tmp_path / "one_sec.mp4"
    n_frames = _make_one_second_video(src)
    assert _probe_frames(src) == n_frames

    out = tmp_path / "out" / "one_sec_x4.mp4"
    events = _run_worker(src, out, decode)
    terminal = events[-1]
    assert terminal.kind == "finished", (
        f"[{decode}] terminal event was {terminal.kind}: {terminal.message}"
    )
    assert out.is_file() and out.stat().st_size > 0
    assert _probe_frames(out) == n_frames, (
        f"[{decode}] expected {n_frames} frames in output"
    )
