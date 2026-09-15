"""controllers/model_resolver.py -- pure model-selection helpers.

Pulled out of the Tk orchestrator so the "what model did the user pick?"
logic is testable without spinning up a Tk root.

The orchestrator still owns the dropdown widget; this module just answers
"given a dropdown text + a registry, which InstalledModel (if any) is this?".

Phase A3 of the GUI extensibility refactor (see docs/gui_audit.md).
"""
from typing import Optional

from ..registry import InstalledModel


def parse_dropdown_filename(text: str) -> str:
    """Pick the filename out of a dropdown label like "rfdn_distill_v1.pth  4x  ...".

    The orchestrator renders each dropdown entry as "<filename>  <scale>x  <display>",
    so the first whitespace-delimited token is always the filename. Empty /
    whitespace-only input returns "" so callers can short-circuit cleanly.
    """
    if not text:
        return ""
    return text.split()[0]


def resolve_from_dropdown(text: str, registry: "ModelRegistry") -> Optional[InstalledModel]:
    """Resolve "whatever the user clicked in the model dropdown" to an InstalledModel.

    Returns None when:
      * the dropdown text is empty / whitespace-only,
      * the filename token does not match any installed model,
      * the registry has not been scanned yet (empty _installed list).

    Callers should fall back to a default model in the first two cases.
    """
    filename = parse_dropdown_filename(text)
    if not filename:
        return None
    return next((m for m in registry._installed if m.filename == filename), None)


def resolve_path_from_dropdown(text: str, registry: "ModelRegistry") -> str:
    """Convenience wrapper that returns the resolved Path string, or "" when missing.

    Replaces the orchestrator's _resolve_model_path_from_dropdown helper.
    """
    m = resolve_from_dropdown(text, registry)
    return str(m.path.resolve()) if m else ""


def resolve_kind_from_dropdown(text: str, registry: "ModelRegistry") -> str:
    """Convenience wrapper that returns the kind string, or "" when missing.

    Replaces the orchestrator's _resolve_kind_from_dropdown helper.
    """
    m = resolve_from_dropdown(text, registry)
    return (m.kind if m and m.kind else "")


__all__ = [
    "parse_dropdown_filename",
    "resolve_from_dropdown",
    "resolve_path_from_dropdown",
    "resolve_kind_from_dropdown",
]
