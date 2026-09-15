"""controllers -- pure-logic helpers extracted from the Tk orchestrator.

These are pure functions (or thin data carriers) that the orchestrator
used to do inline. They are pulled out so they can be unit-tested without
spinning up a Tk root and so the orchestrator stays focused on widget
event handling.

Phase A3 of the GUI extensibility refactor (see docs/gui_audit.md).
"""
from .job_builder import compute_output_path  # noqa: F401
from .model_resolver import (  # noqa: F401
    parse_dropdown_filename,
    resolve_from_dropdown,
    resolve_kind_from_dropdown,
    resolve_path_from_dropdown,
)

__all__ = [
    "compute_output_path",
    "parse_dropdown_filename",
    "resolve_from_dropdown",
    "resolve_path_from_dropdown",
    "resolve_kind_from_dropdown",
]
