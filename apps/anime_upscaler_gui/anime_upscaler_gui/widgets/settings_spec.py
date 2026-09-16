"""widgets/settings_spec.py -- data-driven spec for the Settings panel.

A SettingSpec describes one user-facing setting: its key on the Settings
dataclass, the display label, which Tk widget kind to render, and the
coercion used when the orchestrator reads the var back as a Python value.

This module replaces the previous "one method per setting" pattern in
``widgets/settings_panel.py``. To add a new setting, append one entry to
``SETTING_SPECS`` -- the panel widget loop renders it, the RunJob builder
(Phase B2) reads it by key, and the tests assert the spec list matches
the dataclass fields.

Phase B1 of the GUI extensibility refactor (see docs/gui_audit.md).
"""
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union


# Widget kinds:
#   "radio"     -> ttk.Radiobutton per choice (label=label, value=value)
#   "checkbox"  -> ttk.Checkbutton (one boolean)
#   "spinbox"   -> ttk.Spinbox (numeric; from_/to/increment)
#   "combobox"  -> ttk.Combobox (string, state="readonly", choices=values)
WidgetKind = str  # noqa: F811 -- semantic type alias kept as plain str


@dataclass(frozen=True)
class SettingSpec:
    """Declarative description of one setting entry in the panel.

    Attributes:
      key:       dataclass field name on Defaults (e.g. "device", "fp16").
      label:     human-readable text rendered before the widget(s).
      widget:    one of "radio" | "checkbox" | "spinbox" | "combobox".
      var_name:  tk.StringVar/BooleanVar/IntVar/DoubleVar attribute name on
                 the panel (e.g. "device_var"). Preserved exactly to keep
                 existing app.py callers ("sp.device_var.get()") working.
      choices:   for "radio" / "combobox": list of (value, label) pairs.
      spin:      for "spinbox": (from_, to, increment).
      coerce:    callable converting tk var -> Python value when read back.
      help_text: optional hint shown via a tooltip-like label after the
                 widget (kept short so it fits the row).
    """
    key: str
    label: str
    widget: str
    var_name: str
    choices: Sequence[Tuple[str, str]] = ()
    spin: Tuple[float, float, float] = (0, 0, 0)
    coerce: Optional[Callable[[Any], Any]] = None
    help_text: str = ""


# --- helpers: coerce tk.StringVar/IntVar/DoubleVar/BooleanVar values ---
def _int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _bool(v: Any) -> bool:
    return bool(v)


def _str(v: Any) -> str:
    return str(v) if v is not None else ""


# --- the actual spec list ---
#
# Order matters: rows are rendered in this order, top to bottom. Each entry
# becomes one labeled widget (or one row of radios for "radio" kind).
SETTING_SPECS: List[SettingSpec] = [
    # Row 1: device (radio) + fp16 (checkbox) + outscale (spinbox)
    SettingSpec(
        key="device", label="Device:", widget="radio", var_name="device_var",
        choices=(("cuda", "GPU"), ("cpu", "CPU")),
        coerce=_str,
        help_text="Inference device (CUDA needs an NVIDIA GPU).",
    ),
    SettingSpec(
        key="fp16", label="Precision:", widget="checkbox", var_name="fp16_var",
        coerce=_bool,
        help_text="Halve weights + activations (faster, slight quality cost).",
    ),
    SettingSpec(
        key="outscale", label="Outscale:", widget="spinbox", var_name="outscale_var",
        spin=(1.0, 8.0, 0.5),
        coerce=_float,
        help_text="Final output scale relative to input.",
    ),
    # Row 2: batch + decode + prefetch
    SettingSpec(
        key="batch_size", label="Batch:", widget="spinbox", var_name="batch_var",
        spin=(1, 16, 1),
        coerce=_int,
        help_text="Frames per forward pass (image jobs only).",
    ),
    SettingSpec(
        key="decode", label="Decode:", widget="combobox", var_name="decode_var",
        # Q3: adds nvdec (PyAV hwaccel=cuda) and auto (prefer nvdec, then
        # pyav, then cv2). The existing cv2 / pyav choices are preserved
        # so legacy settings files load unchanged.
        choices=(("auto", "auto"), ("nvdec", "nvdec"),
                 ("pyav", "pyav"), ("cv2", "cv2")),
        coerce=_str,
    ),
    SettingSpec(
        key="prefetch", label="Prefetch:", widget="combobox", var_name="prefetch_var",
        choices=(("sync", "sync"), ("async", "async")),
        coerce=_str,
        help_text="async = background decode thread (lower latency).",
    ),
    # Row 3: downscale pre-process
    SettingSpec(
        key="downscale_max_edge", label="Pre-process: downscale if max edge >",
        widget="spinbox", var_name="downscale_var",
        spin=(0, 8192, 64),
        coerce=_int,
        help_text="Cap input edge before SR (0 = no cap).",
    ),
    # Row 4: gpu_guard + tile_size + tile_overlap
    SettingSpec(
        key="gpu_guard_mode", label="GPU guard:", widget="combobox",
        var_name="gpu_guard_var",
        choices=(("warn", "warn"), ("auto_downscale", "auto_downscale"),
                 ("tiled", "tiled"), ("off", "off")),
        coerce=_str,
    ),
    SettingSpec(
        key="tile_size", label="Tile:", widget="spinbox", var_name="tile_size_var",
        spin=(64, 1024, 32),
        coerce=_int,
    ),
    SettingSpec(
        key="tile_overlap", label="Overlap:", widget="spinbox",
        var_name="tile_overlap_var",
        spin=(0, 128, 4),
        coerce=_int,
    ),
    # Row 5: TTA + acceleration
    SettingSpec(
        key="tta", label="TTA:", widget="checkbox", var_name="tta_var",
        coerce=_bool,
        help_text="D4 group augmentation (8x cost, +0.3 dB).",
    ),
    SettingSpec(
        key="use_tensorrt", label="Acceleration: TensorRT engine (~2-4x faster)",
        widget="checkbox", var_name="use_tensorrt_var",
        coerce=_bool,
    ),
    SettingSpec(
        key="use_nvenc", label="NVENC hardware encoder",
        widget="checkbox", var_name="use_nvenc_var",
        coerce=_bool,
    ),
    SettingSpec(
        key="nvenc_preset", label="NVENC preset:", widget="combobox",
        var_name="nvenc_preset_var",
        choices=(("p1", "p1"), ("p2", "p2"), ("p3", "p3"), ("p4", "p4")),
        coerce=_str,
        help_text="p1 = fastest, p4 = best (we expose p1..p4).",
    ),
    # Q3: was previously read from settings as a hardcoded default 18.
    # Lower = better quality / larger file; 18 is "visually lossless"
    # for SDR anime content. Range mirrors x264 CRF conventions but
    # tuned for NVENC's constant-QP curve.
    SettingSpec(
        key="nvenc_qp", label="NVENC QP:", widget="spinbox",
        var_name="nvenc_qp_var",
        spin=(16, 28, 1),
        coerce=_int,
        help_text="constant-QP target (lower = better quality, larger file)",
    ),
]


# Public helpers for Phase B2 (build_run_job) -- they centralize the
# "what does this tk var look like as a Python value?" question so the
# job builder does not need to know about widget kinds.
def coerce_var(spec: SettingSpec, raw: Any) -> Any:
    """Apply the spec's coerce callable, or fall back to identity for strings."""
    if spec.coerce is None:
        return raw
    return spec.coerce(raw)


def get_spec(key: str) -> SettingSpec:
    """Look up a spec by its dataclass-field key. Raises KeyError if missing."""
    for s in SETTING_SPECS:
        if s.key == key:
            return s
    raise KeyError(f"unknown setting key: {key!r}")


__all__ = [
    "SettingSpec",
    "SETTING_SPECS",
    "coerce_var",
    "get_spec",
]
