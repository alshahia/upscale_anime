"""Tests pinning the Phase B2/B4 contract: build_run_job flows every
SETTING_SPECS key into the returned RunJob.

Audit deliverable C3: given a Job + Settings + InstalledModel,
build_run_job returns a RunJob with every specd field. We pin that
contract here so that adding a new SettingSpec entry does not silently
drop it from the RunJob builder.
"""
from pathlib import Path

from apps.anime_upscaler_gui.anime_upscaler_gui.settings import (
    Defaults, Job, JobStatus,
)
from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.settings_spec import (
    SETTING_SPECS, coerce_var, get_spec,
)
from apps.anime_upscaler_gui.anime_upscaler_gui.controllers.job_builder import (
    build_run_job,
)


# --- helpers ---------------------------------------------------------------
class _FakePanel:
    """Minimal duck-typed settings panel: one tk Variable per SETTING_SPECS entry.

    The build_run_job -> make_panel_value_provider code path calls
    getattr(panel, spec.var_name).get() and coerces via spec.coerce. We
    supply a panel whose attributes match the spec var_names exactly.
    """
    def __init__(self, values):
        # values: {spec_key: raw_value} (will be coerced by coerce_var)
        self._values = values

    def _get(self, key):
        spec = get_spec(key)
        raw = self._values[key]
        return coerce_var(spec, raw)


def _build(values):
    """Build a RunJob using a _FakePanel-like value dict."""
    panel = _FakePanel(values)
    value_provider = panel._get

    job = Job(
        id=1, input=Path("/in/a.png"), output=Path("/out/a_x4.png"),
        is_video=False, status=JobStatus.PENDING, error="",
        model_filename="m.pth", kind="rfdn_student", scale=4,
        cut_start=0.0, cut_end=0.0,
    )
    return build_run_job(
        job=job,
        value_provider=value_provider,
        data=Defaults(),
        resolve_model_path=lambda: "m.pth",
        resolve_kind=lambda: "rfdn_student",
        on_frame_error=lambda idx, msg: "skip",
    )


# Baseline values that satisfy all coerce types:
_BASE_VALUES = {
    "device": "cuda",
    "fp16": True,
    "outscale": 4.0,
    "batch_size": 4,
    "decode": "cv2",
    "prefetch": "async",
    "downscale_max_edge": 0,
    "gpu_guard_mode": "auto_downscale",
    "tile_size": 256,
    "tile_overlap": 8,
    "tta": False,
    "use_tensorrt": False,
    "use_nvenc": True,
    "nvenc_preset": "p4",
    # Q3 (perf/queue-controls-gpu-codec): NVENC QP widget added to the panel.
    "nvenc_qp": 20,
}


# --- the audit headline test --------------------------------------------
def test_every_panel_key_flows_into_runjob():
    """Audit C3: build_run_job returns a RunJob with every specd field.

    For each SettingSpec, we mutate the value, call build_run_job, and
    assert the corresponding RunJob field reflects the change. This is the
    structural guarantee that adding a SettingSpec is sufficient to
    propagate a new setting to the worker.
    """
    # SETTING_SPECS key -> RunJob field name they map to inside build_run_job.
    # Anything not listed here is intentionally read from Defaults (advanced),
    # or hardcoded for pin_memory, or derived from job.is_video.
    PANEL_TO_RUNJOB = {
        "device": "device",
        "fp16": "fp16",
        "outscale": "outscale",
        "decode": "decode",
        "prefetch": "prefetch",
        "downscale_max_edge": "downscale_max_edge",
        "gpu_guard_mode": "gpu_guard_mode",
        "tile_size": "tile_size",
        "tile_overlap": "tile_overlap",
        "tta": "tta",
        "use_tensorrt": "use_tensorrt",
        "use_nvenc": "use_nvenc",
        "nvenc_preset": "nvenc_preset",
        # Q3 (perf/queue-controls-gpu-codec): QP widget added to the panel
        # and now flows through build_run_job (previously read from data).
        "nvenc_qp": "nvenc_qp",
        # batch_size handled by build_run_job is_video branch; tested below.
    }
    for spec_key, runjob_field in PANEL_TO_RUNJOB.items():
        spec = get_spec(spec_key)
        baseline = _BASE_VALUES[spec_key]
        if spec.widget in ("radio", "combobox") and len(spec.choices) >= 2:
            new_value = spec.choices[1][0]
        elif spec.widget == "spinbox":
            from_, to_, inc = spec.spin
            new_value = from_ + inc
            if new_value == baseline:
                new_value = to_
        elif spec.widget == "checkbox":
            new_value = not bool(baseline)
        else:
            new_value = baseline
        values = dict(_BASE_VALUES)
        values[spec_key] = new_value
        out = _build(values)
        actual = getattr(out, runjob_field)
        coerced = coerce_var(spec, new_value)
        assert actual == coerced, (
            f"spec_key={spec_key!r} (panel.{spec.var_name}) did not propagate "
            f"to RunJob.{runjob_field}: expected {coerced!r}, got {actual!r}",
        )


def test_batch_size_propagation_for_image():
    """batch_size flows from the panel for image jobs (non-video)."""
    out = _build({**_BASE_VALUES, "batch_size": 8})
    assert out.batch_size == 8


def test_all_setting_spec_keys_have_defaults_in_baseline():
    """_BASE_VALUES must cover every SettingSpec.key used by build_run_job."""
    spec_keys = {s.key for s in SETTING_SPECS}
    baseline_keys = set(_BASE_VALUES.keys())
    PANEL_KEYS_USED_BY_BUILDER = {
        "device", "fp16", "outscale", "batch_size", "decode",
        "prefetch", "downscale_max_edge", "gpu_guard_mode",
        "tile_size", "tile_overlap", "tta", "use_tensorrt",
        "use_nvenc", "nvenc_preset",
    }
    missing_in_baseline = PANEL_KEYS_USED_BY_BUILDER - baseline_keys
    assert not missing_in_baseline, (
        f"_BASE_VALUES is missing keys used by build_run_job: {missing_in_baseline}",
    )
    extra_in_spec = spec_keys - baseline_keys
    assert not extra_in_spec, (
        f"SETTING_SPECS has new keys not covered by _BASE_VALUES: {extra_in_spec}. "
        f"Add them so test_every_panel_key_flows_into_runjob covers the propagation.",
    )


def test_is_video_derived_from_job():
    """is_video flows from job.is_video into RunJob.is_video."""
    job = Job(
        id=2, input=Path("/in/a.mp4"), output=Path("/out/a_x4.mp4"),
        is_video=True, status=JobStatus.PENDING, error="",
        model_filename="m.pth", kind="srvgg", scale=4,
        cut_start=0.0, cut_end=0.0,
    )
    panel = _FakePanel(_BASE_VALUES)
    out = build_run_job(
        job=job,
        value_provider=panel._get,
        data=Defaults(),
        resolve_model_path=lambda: "m.pth",
        resolve_kind=lambda: "srvgg",
        on_frame_error=lambda idx, msg: "skip",
    )
    assert out.is_video is True


def test_resolver_used_only_when_job_fields_missing():
    """job.model_filename and job.kind take precedence over the resolvers."""
    called = []
    def _model_path():
        called.append("model")
        return "RESOLVED.pth"
    def _kind():
        called.append("kind")
        return "resolved_kind"

    job = Job(
        id=3, input=Path("/in/a.png"), output=Path("/out/a_x4.png"),
        is_video=False, status=JobStatus.PENDING, error="",
        model_filename="JOB.pth", kind="job_kind", scale=4,
        cut_start=0.0, cut_end=0.0,
    )
    panel = _FakePanel(_BASE_VALUES)
    out = build_run_job(
        job=job,
        value_provider=panel._get,
        data=Defaults(),
        resolve_model_path=_model_path,
        resolve_kind=_kind,
        on_frame_error=lambda idx, msg: "skip",
    )
    assert out.model_filename == "JOB.pth"
    assert out.kind == "job_kind"
    assert called == []


def test_resolver_used_when_job_fields_missing():
    """Resolvers ARE called when job.model_filename or job.kind are empty."""
    job = Job(
        id=4, input=Path("/in/a.png"), output=Path("/out/a_x4.png"),
        is_video=False, status=JobStatus.PENDING, error="",
        model_filename="", kind="", scale=4,
        cut_start=0.0, cut_end=0.0,
    )
    panel = _FakePanel(_BASE_VALUES)
    out = build_run_job(
        job=job,
        value_provider=panel._get,
        data=Defaults(),
        resolve_model_path=lambda: "RESOLVED.pth",
        resolve_kind=lambda: "resolved_kind",
        on_frame_error=lambda idx, msg: "skip",
    )
    assert out.model_filename == "RESOLVED.pth"
    assert out.kind == "resolved_kind"
