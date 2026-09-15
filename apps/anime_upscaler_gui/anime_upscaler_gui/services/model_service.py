"""services/model_service.py -- pure model-dropdown presentation helpers.

Pulled out of the Tk orchestrator so the "how do we present installed models
in the dropdown?" logic is unit-testable without spinning up a Tk root.

Three concerns live here:
  1. sort order: trained (in-house) models first, community models after;
  2. per-row label: "  (<kind>, <scale>x, <size_mb> MB)<tag>";
  3. default selection: last_model substring -> first trained -> first item.

The orchestrator still owns the registry (calls scan_installed + writes the
dropdown widget); this module just answers "given (models, last_model), what
should the dropdown labels be and which row should be selected?".

Phase A4 of the GUI extensibility refactor (see docs/gui_audit.md).
"""
from typing import Iterable, List

from ..registry import InstalledModel

# Visual tags appended to a dropdown label so the user can spot warnings
# and provenance at a glance. Public so widgets/tests can match on them.
TAG_TAINTED = "  [TAINTED]"
TAG_TRAINED = "  [TRAINED]"
TAG_UNSUPPORTED = "  [unsupported]"
TAG_NO_ARCH = "  [no arch]"


def sort_trained_first(models: Iterable[InstalledModel]) -> List[InstalledModel]:
    """Return a new list with trained (in-house) entries before community entries.

    Within each group the original scan order is preserved; the registry
    already returns them in a useful order (typically newest-first), and we
    do not want to surprise the user with alphabetical reshuffles.
    """
    models = list(models)
    trained = [m for m in models if m.is_trained]
    others = [m for m in models if not m.is_trained]
    return trained + others


def _pick_tag(model: InstalledModel, supported_kind: bool) -> str:
    """Decide which warning/provenance tag (if any) goes on the dropdown row.

    Priority order matters -- a tainted trained file should still warn the
    user it is tainted before the more general "[TRAINED]" provenance cue.
    """
    if model.tainted:
        return TAG_TAINTED
    if model.is_trained:
        return TAG_TRAINED
    if not model.supported:
        return TAG_UNSUPPORTED
    if not supported_kind:
        return TAG_NO_ARCH
    return ""


def format_dropdown_label(model: InstalledModel, supported_kind: bool = True) -> str:
    """Render one InstalledModel as a single dropdown label string.

    Example output:
      "rfdn_distill_v1.pth  (rfdn_student, 4x, 18.3 MB)  [TRAINED]"
      "animesr_v2.pth      (animesr, 2x, 40.1 MB)"
      "garbage.pth         (?, 4x, 0.0 MB)  [TAINTED]"

    `supported_kind` is passed in by the caller (typically `is_supported_kind(model.kind)`)
    so this function stays import-cycle-free of the archs registry.
    """
    tag = _pick_tag(model, supported_kind)
    head = f"{model.filename}  ({model.kind or chr(63)}, {model.scale}x, {model.size_mb:.1f} MB)"
    return head + tag


def pick_default_index(items: List[str], last_model: str = "") -> int:
    """Pick which dropdown row should be selected after a refresh.

    Priority:
      1. `last_model` substring match (so the previous pick survives
         filename reformatting across releases).
      2. First row containing "[TRAINED]" -- so a fresh launch with a
         trained artifact on disk defaults to it instead of an arbitrary
         community preset (the original bug fixed when this was added).
      3. First row (any installed model).
      4. Returns -1 when `items` is empty (caller decides whether to disable
         the dropdown or leave it unselected).
    """
    if not items:
        return -1
    if last_model:
        for i, it in enumerate(items):
            if last_model in it:
                return i
    for i, it in enumerate(items):
        if TAG_TRAINED in it:
            return i
    return 0


__all__ = [
    "TAG_TAINTED",
    "TAG_TRAINED",
    "TAG_UNSUPPORTED",
    "TAG_NO_ARCH",
    "sort_trained_first",
    "format_dropdown_label",
    "pick_default_index",
]
