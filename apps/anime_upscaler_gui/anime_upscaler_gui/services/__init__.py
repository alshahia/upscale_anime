"""services -- pure application services extracted from the Tk orchestrator.

These are stateless helpers that the orchestrator used to do inline; pulled
out so they can be unit-tested without a Tk root and so the orchestrator
stays focused on widget event handling.

Phase A4 of the GUI extensibility refactor (see docs/gui_audit.md).
"""
from .model_service import (  # noqa: F401
    TAG_NO_ARCH,
    TAG_TAINTED,
    TAG_TRAINED,
    TAG_UNSUPPORTED,
    format_dropdown_label,
    pick_default_index,
    sort_trained_first,
)

__all__ = [
    "TAG_NO_ARCH",
    "TAG_TAINTED",
    "TAG_TRAINED",
    "TAG_UNSUPPORTED",
    "format_dropdown_label",
    "pick_default_index",
    "sort_trained_first",
]
