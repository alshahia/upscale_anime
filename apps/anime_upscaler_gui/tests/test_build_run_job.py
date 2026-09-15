"""Unit tests for controllers/job_builder.build_run_job (Phase B2).

These tests prove the function is pure: no Tk imports, no GUI panel, just
a dict-as-value-provider and a small Defaults dataclass. The GUI's _enqueue_next
is rewritten in Phase B4 to call build_run_job() instead of the 30-arg literal,
but the GUI rewrite is independent of this test (same shape, same contract).
"""
import pytest

from apps.anime_upscaler_gui.anime_upscaler_gui.controllers.job_builder import (
    build_run_job,
    compute_output_path,
    make_panel_value_provider,
)
from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline.jobs import RunJob
from apps.anime_upscaler_gui.anime_upscaler_gui.settings import Defaults


class _FakeJob:
    """Minimal stand-in for the GUI Job (duck-typed by build_run_job)."""
    def __init__(self, id=1, input=None, output=None, is_video=False,
                 cut_start=0.0, cut_end=0.0,
                 model_filename="", kind="", scale=0):
        self.id = id
        self.input = input or "/tmp/in.mp4"
        self.output = output or "/tmp/out.mp4"
        self.is_video = is_video
        self.cut_start = cut_start
        self.cut_end = cut_end
        self.model_filename = model_filename
        self.kind = kind
        self.scale = scale


# Mirrors widgets/settings_spec.py::SETTING_SPECS defaults -- keeping
# the test self-contained (no Tk panel needed).
_TEST_VALUES = {
    "device": "cuda",
    "fp16": True,
    "outscale": 4.0,
    "batch_size": 8,
    "decode": "cv2",
    "prefetch": "sync",
    "downscale_max_edge": 2560,
    "gpu_guard_mode": "warn",
    "tile_size": 256,
    "tile_overlap": 32,
    "tta": True,
    "use_tensorrt": True,
    "use_nvenc": True,
    "nvenc_preset": "p1",
}


def _provider(d):
    return lambda key: d[key]


def _resolve_model_path():
    return "/ckpts/recommended.pth"


def _resolve_kind():
    return "spandrel"


def _on_frame_error(idx, msg):
    return "skip_frame"


def test_build_run_job_basic_image():
    """Image job: uses panel batch_size; all coerced fields match spec types."""
    job = _FakeJob(id=7, is_video=False)
    data = Defaults()  # nvenc_qp=18, cascade_mode=None
    out = build_run_job(
        job=job,
        value_provider=_provider(_TEST_VALUES),
        data=data,
        resolve_model_path=_resolve_model_path,
        resolve_kind=_resolve_kind,
        on_frame_error=_on_frame_error,
    )
    assert isinstance(out, RunJob)
    assert out.job_id == 7
    assert out.is_video is False
    assert out.batch_size == 8  # from panel
    assert out.outscale == 4.0
    assert out.fp16 is True
    assert out.device == "cuda"
    assert out.tile_size == 256
    assert out.tile_overlap == 32
    assert out.tta is True
    assert out.use_tensorrt is True
    assert out.use_nvenc is True
    assert out.nvenc_preset == "p1"
    # Advanced fields from Defaults, not the panel:
    assert out.nvenc_qp == 18
    assert out.cascade_mode is None
    # pin_memory is a constant for now:
    assert out.pin_memory == "auto"
    # Job-supplied:
    assert str(out.input_path) == "/tmp/in.mp4"
    assert str(out.output_path) == "/tmp/out.mp4"
    assert out.model_filename == "/ckpts/recommended.pth"
    assert out.kind == "spandrel"
    assert out.scale == 4  # default fallback when j.scale=0
    # Callbacks:
    assert out.on_frame_error(0, "boom") == "skip_frame"
    # Cuts:
    assert out.cut_start_seconds == 0.0
    assert out.cut_end_seconds == 0.0


def test_build_run_job_video_forces_batch_size_one():
    """Video jobs always get batch_size=1 (Phase 1.C TRT stream-sync workaround)."""
    job = _FakeJob(is_video=True)
    data = Defaults()
    out = build_run_job(
        job=job,
        value_provider=_provider(_TEST_VALUES),
        data=data,
        resolve_model_path=_resolve_model_path,
        resolve_kind=_resolve_kind,
        on_frame_error=_on_frame_error,
    )
    assert out.is_video is True
    assert out.batch_size == 1, "video jobs MUST ignore panel batch_size"


def test_fp16_is_disabled_on_cpu():
    """fp16 requires CUDA -- when device=cpu, fp16 is forced to False.

    This mirrors the app.py literal:
      fp16=bool(sp.fp16_var.get()) and sp.device_var.get() == "cuda"
    """
    job = _FakeJob(is_video=False)
    values = dict(_TEST_VALUES)
    values["device"] = "cpu"
    values["fp16"] = True  # user wants fp16 but on CPU
    data = Defaults()
    out = build_run_job(
        job=job,
        value_provider=_provider(values),
        data=data,
        resolve_model_path=_resolve_model_path,
        resolve_kind=_resolve_kind,
        on_frame_error=_on_frame_error,
    )
    assert out.device == "cpu"
    assert out.fp16 is False, "fp16 must be False on CPU regardless of panel"


def test_build_run_job_prefers_job_over_resolver():
    """If job.model_filename / kind / scale are set, the resolver is bypassed."""
    job = _FakeJob(model_filename="/batch/ckpt.pth", kind="animesr", scale=2)
    data = Defaults()
    out = build_run_job(
        job=job,
        value_provider=_provider(_TEST_VALUES),
        data=data,
        resolve_model_path=lambda: "/should/not/be/used.pth",
        resolve_kind=lambda: "spandrel",
        on_frame_error=_on_frame_error,
    )
    assert out.model_filename == "/batch/ckpt.pth"
    assert out.kind == "animesr"
    assert out.scale == 2


def test_build_run_job_cuts_propagate():
    """cut_start / cut_end flow from the Job to RunJob as floats."""
    job = _FakeJob(cut_start=1.5, cut_end=10.25)
    out = build_run_job(
        job=job,
        value_provider=_provider(_TEST_VALUES),
        data=Defaults(),
        resolve_model_path=_resolve_model_path,
        resolve_kind=_resolve_kind,
        on_frame_error=_on_frame_error,
    )
    assert out.cut_start_seconds == 1.5
    assert out.cut_end_seconds == 10.25


def test_build_run_job_cascade_and_nvenc_qp_come_from_data():
    """Advanced settings (nvenc_qp, cascade_mode) come from Defaults, not panel."""
    data = Defaults()
    data.nvenc_qp = 22  # higher QP = smaller file, lower quality
    data.cascade_mode = 2  # force 2-pass cascade
    out = build_run_job(
        job=_FakeJob(),
        value_provider=_provider(_TEST_VALUES),
        data=data,
        resolve_model_path=_resolve_model_path,
        resolve_kind=_resolve_kind,
        on_frame_error=_on_frame_error,
    )
    assert out.nvenc_qp == 22
    assert out.cascade_mode == 2


def test_make_panel_value_provider_reads_via_spec_var_name():
    """make_panel_value_provider() reads panel attrs through SETTING_SPECS.

    Uses a fake panel (no Tk) with attribute names that match the spec.
    """
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.settings_spec import get_spec

    class FakePanel:
        pass

    fp = FakePanel()
    # Set every var_name from SETTING_SPECS to a sentinel string. After coercion,
    # values are converted by spec.coerce (int/bool/float).
    for spec in __import__(
        "apps.anime_upscaler_gui.anime_upscaler_gui.widgets.settings_spec",
        fromlist=["SETTING_SPECS"],
    ).SETTING_SPECS:
        # Each Tk var would hold a Python value; mock it as a real value.
        if spec.widget == "spinbox":
            v = 7
        elif spec.widget == "checkbox":
            v = True
        else:
            v = "value"
        # Build a fake var with .get() returning v:
        var = type("Var", (), {"get": staticmethod(lambda v=v: v)})()
        setattr(fp, spec.var_name, var)

    provider = make_panel_value_provider(fp)
    # Spot-check a few keys:
    assert provider("device") == "value"  # radio -> str
    assert provider("fp16") is True       # checkbox -> bool
    assert provider("batch_size") == 7    # spinbox (int bounds) -> int
    assert provider("outscale") == 7.0    # spinbox (float bounds) -> float


def test_make_panel_value_provider_unknown_key_raises():
    """Looking up a key not in SETTING_SPECS raises KeyError -- by design."""
    class FakePanel:
        pass

    fp = FakePanel()
    provider = make_panel_value_provider(fp)
    with pytest.raises(KeyError):
        provider("not_a_real_setting")


def test_compute_output_path_same_folder():
    """compute_output_path() is unchanged from Phase A3 -- smoke test."""
    from pathlib import Path, PurePosixPath
    out = compute_output_path(
        Path("/in/clip.mp4"),
        "same_folder", "_upscaled", "upscaled_4x", "/unused",
    )
    # Path equality is OS-normalized; check the parts instead so the test
    # works on both Windows (backslashes) and POSIX (forward slashes).
    assert out.name == "clip_upscaled.mp4"
    assert out.parent.name == "upscaled_4x"
    assert PurePosixPath(out.parent.parent).name == "in"


def test_compute_output_path_custom_dir():
    """compute_output_path() with output_mode="custom" uses custom_output_dir."""
    from pathlib import Path, PurePosixPath
    out = compute_output_path(
        Path("/in/clip.mp4"),
        "custom", "_upscaled", "upscaled_4x", "/my/out",
    )
    assert out.name == "clip_upscaled.mp4"
    assert PurePosixPath(out.parent).name == "out"
    assert PurePosixPath(out.parent.parent).name == "my"
