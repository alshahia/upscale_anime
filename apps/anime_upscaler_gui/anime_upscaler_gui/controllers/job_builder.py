"""controllers/job_builder.py -- pure output-path construction.

Pulled out of the Tk orchestrator so the "where does this file get written?"
logic is testable without spinning up a Tk root.

The orchestrator still owns the settings panel + output panel widgets; this
module just answers "given (input_path, settings, custom_output_dir), what
is the destination Path?".

Phase A3 of the GUI extensibility refactor (see docs/gui_audit.md).

Phase B2 adds build_run_job(), which replaces the 30-arg RunJob(...) literal
previously inlined in app.py::_enqueue_next. The function is pure: it takes
a queued Job, a value-provider callable (so the GUI panel and unit tests can
both call it without depending on Tk), and the few orchestrator-supplied
callables (model/kind resolvers, frame-error callback). No Tk imports here.
"""
from pathlib import Path
from typing import Any, Callable, Optional

from ..widgets.settings_spec import SETTING_SPECS, coerce_var, get_spec
from ..pipeline.jobs import RunJob


# Settings that the panel reads but that are NOT in SETTING_SPECS.
# Either advanced fields hidden from the panel ("nvenc_qp", "cascade_mode"),
# or constants hard-coded at the call site ("pin_memory"). build_run_job
# reads these from `data` (the Defaults dataclass) directly.
_PANEL_KEYS = {spec.key for spec in SETTING_SPECS}


def compute_output_path(
    input_path: Path,
    output_mode: str,
    single_file_suffix: str,
    batch_folder_name: str,
    custom_output_dir: str,
) -> Path:
    """Compute where the upscaled output for ``input_path`` should be written.

    Logic mirrors the orchestrator's _compute_output_path exactly:

      * output_mode == "same_folder" ->
            <input_path.parent>/<batch_folder_name>/<stem><suffix><ext>
        (used for multi-file jobs; the batch folder sits next to the inputs).
      * output_mode == "custom" (or anything else) ->
            <custom_output_dir>/<stem><suffix><ext>
        (custom_output_dir is a free string from the output panel var).

    Parameters are primitives instead of an object so this stays trivially
    unit-testable and so the orchestrator can keep its own data model
    (it has a `Defaults` dataclass accessed via `self.settings.data`).
    """
    if output_mode == "same_folder":
        out_dir = input_path.parent / batch_folder_name
    else:
        out_dir = Path(custom_output_dir)
    return out_dir / f"{input_path.stem}{single_file_suffix}{input_path.suffix}"


# --- Phase B2: build_run_job -------------------------------------------


def make_panel_value_provider(
    panel: Any,
) -> Callable[[str], Any]:
    """Build a value_provider that reads the settings panel via SETTING_SPECS.

    Used by app.py to wire the live Tk panel into build_run_job() without
    importing Tk here. The lookup is spec-key -> panel.<var_name>.get();
    values are coerced via spec.coerce so the returned Python types match
    what RunJob expects (int/int/float/bool/str).

    Note: panel is duck-typed (Any). The only attribute access is
    `getattr(panel, spec.var_name).get()`. Any object that quacks that way
    works -- including a fake panel built for unit tests.
    """
    def _get(key: str) -> Any:
        spec = get_spec(key)  # raises KeyError for non-panel keys
        raw = getattr(panel, spec.var_name).get()
        return coerce_var(spec, raw)
    return _get


def build_run_job(
    *,
    job: Any,
    value_provider: Callable[[str], Any],
    data: Any,
    resolve_model_path: Callable[[], str],
    resolve_kind: Callable[[], str],
    on_frame_error: Callable[[int, str], str],
    is_video_for_batch: Optional[Callable[[], int]] = None,
) -> RunJob:
    """Construct a fully-built RunJob from a queued Job + panel + defaults.

    Parameters:
      job:                the queued Job (any object with id/input/output/
                          is_video/cut_start/cut_end/model_filename/kind/scale).
      value_provider:     callable(spec_key) -> coerced Python value. In
                          production this wraps the live Tk settings panel
                          via make_panel_value_provider(). In tests it can
                          be a plain lambda key: settings_dict[key].
      data:               the Defaults dataclass (advanced settings not on
                          the panel: nvenc_qp, cascade_mode).
      resolve_model_path: callable returning the absolute model path
                          (e.g. from the model dropdown).
      resolve_kind:       callable returning "spandrel" | "animesr" | etc.
      on_frame_error:     callable for the worker's resume-action prompt.
      is_video_for_batch: optional override for the video batch_size=1
                          shortcut (default: use job.is_video).

    Returns:
      A fully-built RunJob dataclass instance ready to put on the queue.

    This function replaces the 30-arg RunJob(...) literal that previously
    lived in app.py::_enqueue_next. The panel-reading loop is data-driven
    by SETTING_SPECS, so adding a new setting only requires appending one
    entry there and (if RunJob needs it) one new key in _RUN_JOB_KEYS below.
    """
    is_video = bool(job.is_video)

    # Phase 1.C STATUS: batched video is currently forced to batch_size=1.
    # This mirrors the orchestrator's existing logic (see the long block
    # comment in app.py::_enqueue_next) and keeps the well-tested per-frame
    # path running for video. Image jobs use the panel's batch_size.
    if is_video_for_batch is not None:
        batch_size = int(is_video_for_batch())
    elif is_video:
        batch_size = 1
    else:
        batch_size = int(value_provider("batch_size"))

    device = value_provider("device")
    fp16 = bool(value_provider("fp16")) and device == "cuda"

    return RunJob(
        job_id=job.id,
        input_path=job.input,
        output_path=job.output,
        is_video=is_video,
        model_filename=job.model_filename or resolve_model_path(),
        kind=job.kind or resolve_kind(),
        scale=job.scale or 4,
        outscale=float(value_provider("outscale")),
        fp16=fp16,
        device=device,
        batch_size=batch_size,
        decode=value_provider("decode"),
        prefetch=value_provider("prefetch"),
        # pin_memory is a constant for now; expose in the panel later if users ask.
        pin_memory="auto",
        downscale_max_edge=int(value_provider("downscale_max_edge")),
        gpu_guard_mode=value_provider("gpu_guard_mode"),
        tile_size=int(value_provider("tile_size")),
        tile_overlap=int(value_provider("tile_overlap")),
        tta=bool(value_provider("tta")),
        use_tensorrt=bool(value_provider("use_tensorrt")),
        use_nvenc=bool(value_provider("use_nvenc")),
        nvenc_preset=str(value_provider("nvenc_preset")),
        # Advanced fields read directly from the Defaults dataclass.
        nvenc_qp=int(data.nvenc_qp),
        cascade_mode=data.cascade_mode,
        cut_start_seconds=float(job.cut_start),
        cut_end_seconds=float(job.cut_end),
        on_frame_error=on_frame_error,
    )


__all__ = [
    "compute_output_path",
    "build_run_job",
    "make_panel_value_provider",
]
