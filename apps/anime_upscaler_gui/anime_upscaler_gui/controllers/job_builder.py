"""controllers/job_builder.py -- pure output-path construction.

Pulled out of the Tk orchestrator so the "where does this file get written?"
logic is testable without spinning up a Tk root.

The orchestrator still owns the settings panel + output panel widgets; this
module just answers "given (input_path, settings, custom_output_dir), what
is the destination Path?".

Phase A3 of the GUI extensibility refactor (see docs/gui_audit.md).
"""
from pathlib import Path


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


__all__ = ["compute_output_path"]
