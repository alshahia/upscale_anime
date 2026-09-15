"""Tests for the typed state layer (JobStatus, Job, FormState)."""


def test_job_status_str_enum():
    from apps.anime_upscaler_gui.anime_upscaler_gui.state import JobStatus
    assert JobStatus.PENDING == "pending"
    assert JobStatus.RUNNING == "running"
    assert JobStatus.DONE == "done"
    assert JobStatus.ERROR == "error"
    assert JobStatus.SKIPPED == "skipped"
    assert JobStatus.CANCELLED == "cancelled"


def test_job_status_str_returns_value():
    from apps.anime_upscaler_gui.anime_upscaler_gui.state import JobStatus
    assert str(JobStatus.PENDING) == "pending"
    assert f"{JobStatus.RUNNING}" == "running"
    assert f"#{JobStatus.DONE}" == "#done"


def test_job_status_lower():
    from apps.anime_upscaler_gui.anime_upscaler_gui.state import JobStatus
    assert JobStatus.PENDING.lower() == "pending"
    assert JobStatus.ERROR.lower() == "error"


def test_job_dataclass_required_and_optional_fields():
    from pathlib import Path
    from apps.anime_upscaler_gui.anime_upscaler_gui.state import Job, JobStatus
    j = Job(id=1, input=Path("x.png"), output=Path("y.png"))
    assert j.id == 1
    assert j.input.name == "x.png"
    assert j.output.name == "y.png"
    assert j.status == JobStatus.PENDING
    assert j.error == ""
    assert j.fps == 0.0
    assert j.infer_ms == 0.0
    assert j.model_filename == ""
    assert j.kind == ""
    assert j.scale == 4
    assert j.is_video is False
    assert j.cut_start == 0.0
    assert j.cut_end == 0.0


def test_job_dataclass_mutable():
    from pathlib import Path
    from apps.anime_upscaler_gui.anime_upscaler_gui.state import Job, JobStatus
    j = Job(id=1, input=Path("x.png"), output=Path("y.png"))
    j.status = JobStatus.RUNNING
    j.fps = 12.5
    j.infer_ms = 83.3
    j.error = "boom"
    j.model_filename = "/tmp/model.pth"
    j.kind = "span"
    j.scale = 4
    j.is_video = True
    j.cut_start = 1.5
    j.cut_end = 9.25
    assert j.status == JobStatus.RUNNING
    assert j.fps == 12.5
    assert j.infer_ms == 83.3
    assert j.error == "boom"


def test_form_state_default_valid():
    from apps.anime_upscaler_gui.anime_upscaler_gui.state import FormState
    fs = FormState()
    assert fs.outscale == 2.0
    assert fs.batch_size == 1
    assert fs.tile_size == 256


def test_form_state_validates_outscale():
    from apps.anime_upscaler_gui.anime_upscaler_gui.state import FormState
    import pytest
    FormState(outscale=1.0)
    FormState(outscale=8.0)
    with pytest.raises(ValueError):
        FormState(outscale=0.5)
    with pytest.raises(ValueError):
        FormState(outscale=15.0)


def test_form_state_validates_batch_size():
    from apps.anime_upscaler_gui.anime_upscaler_gui.state import FormState
    import pytest
    FormState(batch_size=1)
    FormState(batch_size=16)
    with pytest.raises(ValueError):
        FormState(batch_size=0)
    with pytest.raises(ValueError):
        FormState(batch_size=32)


def test_form_state_validates_tile_size():
    from apps.anime_upscaler_gui.anime_upscaler_gui.state import FormState
    import pytest
    FormState(tile_size=64)
    FormState(tile_size=1024)
    with pytest.raises(ValueError):
        FormState(tile_size=32)
    with pytest.raises(ValueError):
        FormState(tile_size=2048)


def test_defaults_extends_form_state():
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _Defaults
    from apps.anime_upscaler_gui.anime_upscaler_gui.state import FormState
    assert issubclass(_Defaults, FormState)
    d = _Defaults()
    assert d.outscale == 2.0
    assert d.batch_size == 1
    assert d.tile_size == 256
    assert d.version == 1


def test_defaults_construction_validates():
    """Smoke: a default _Defaults() must construct without raising."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _Defaults
    d = _Defaults()
    assert d.fp16 is True
    assert d.device == "cuda"
    assert d.geometry == "1100x800"
