"""Phase 4: _QueueController persistence."""
import json
import tempfile
import uuid
from pathlib import Path

from apps.anime_upscaler_gui.anime_upscaler_gui.state import Job, JobStatus


def _fresh_appdir() -> Path:
    return Path(tempfile.gettempdir()) / f"aug_queue_{uuid.uuid4().hex[:8]}"


def _mk_job(id: int, name: str, status: JobStatus = JobStatus.PENDING, **kw) -> Job:
    p = Path(tempfile.gettempdir()) / f"{name}.png"
    return Job(id=id, input=p, output=p.with_name(f"{name}_out.png"), status=status, **kw)


def test_persist_load_roundtrip():
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths, _QueueController
    d = _fresh_appdir()
    paths = _AppPaths(d)
    qc = _QueueController(paths)
    jobs = [
        _mk_job(1, "a", JobStatus.PENDING),
        _mk_job(2, "b", JobStatus.DONE, fps=4.2, infer_ms=12.3, model_filename="x.pth", kind="realesrgan"),
        _mk_job(3, "c", JobStatus.ERROR, error="boom"),
    ]
    qc.persist(jobs)

    qc2 = _QueueController(paths)
    loaded = qc2.load()
    assert len(loaded) == 3
    assert loaded[0].id == 1 and loaded[0].status == JobStatus.PENDING
    assert loaded[1].id == 2 and loaded[1].status == JobStatus.DONE
    assert loaded[1].fps == 4.2 and loaded[1].infer_ms == 12.3
    assert loaded[1].model_filename == "x.pth" and loaded[1].kind == "realesrgan"
    assert loaded[2].id == 3 and loaded[2].status == JobStatus.ERROR
    assert loaded[2].error == "boom"


def test_running_resets_to_pending():
    """A job that was RUNNING at shutdown belongs to a worker thread that's
    now gone; it must load as PENDING so the user can re-Start it."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths, _QueueController
    d = _fresh_appdir()
    paths = _AppPaths(d)
    qc = _QueueController(paths)
    qc.persist([_mk_job(1, "in_flight", JobStatus.RUNNING)])
    loaded = _QueueController(paths).load()
    assert loaded[0].status == JobStatus.PENDING


def test_corrupt_queue_recovers():
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths, _QueueController
    d = _fresh_appdir()
    paths = _AppPaths(d)
    qc = _QueueController(paths)
    qc.queue_path.write_text("{not valid json")
    assert qc.load() == []


def test_wrong_version_returns_empty():
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths, _QueueController
    d = _fresh_appdir()
    paths = _AppPaths(d)
    qc = _QueueController(paths)
    qc.queue_path.write_text(json.dumps({"version": 999, "jobs": [{"id": 1, "input": "x", "output": "y"}]}))
    assert qc.load() == []


def test_clear_removes_file():
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths, _QueueController
    d = _fresh_appdir()
    paths = _AppPaths(d)
    qc = _QueueController(paths)
    qc.persist([_mk_job(1, "x")])
    assert qc.queue_path.exists()
    qc.clear()
    assert not qc.queue_path.exists()


def test_clear_when_missing_is_silent():
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths, _QueueController
    d = _fresh_appdir()
    paths = _AppPaths(d)
    qc = _QueueController(paths)
    # No file yet; clear() must not raise.
    qc.clear()
    assert not qc.queue_path.exists()


def test_persist_drops_bad_rows_silently():
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths, _QueueController
    d = _fresh_appdir()
    paths = _AppPaths(d)
    qc = _QueueController(paths)
    qc.queue_path.write_text(json.dumps({
        "version": 1,
        "jobs": [
            {"id": 1, "input": str(Path("/tmp/a")), "output": str(Path("/tmp/b"))},
            {"id": "not an int", "input": "x", "output": "y"},   # bad
            {"missing_id": True},                                  # bad
        ],
    }))
    loaded = qc.load()
    assert len(loaded) == 1
    assert loaded[0].id == 1


def test_persist_is_atomic():
    """Calling persist() must leave no temp files behind."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths, _QueueController
    d = _fresh_appdir()
    paths = _AppPaths(d)
    qc = _QueueController(paths)
    qc.persist([_mk_job(1, "a")])
    qc.persist([_mk_job(1, "a"), _mk_job(2, "b")])
    leftovers = [p.name for p in d.iterdir() if p.name.startswith(".") and p.name.endswith(".tmp")]
    assert not leftovers, f"leftover: {leftovers}"
