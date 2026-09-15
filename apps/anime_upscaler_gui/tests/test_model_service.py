"""Tests for apps.anime_upscaler_gui.services.model_service (Phase C3).

Audit deliverable C3: 'model_service -- trained-first sort, default pick'.
Pinning these contracts here means the GUI dropdown's behavior is fully
covered without spinning up Tk.
"""
from pathlib import Path

from apps.anime_upscaler_gui.anime_upscaler_gui.registry import InstalledModel
from apps.anime_upscaler_gui.anime_upscaler_gui.services.model_service import (
    TAG_TAINTED, TAG_TRAINED, TAG_UNSUPPORTED, TAG_NO_ARCH,
    sort_trained_first, format_dropdown_label, pick_default_index,
)


# --- helpers ---------------------------------------------------------------
def _m(filename, *, kind="rfdn_student", scale=4, size_mb=18.0,
        tainted=False, supported=True, is_trained=False):
    """Build an InstalledModel with minimal fields populated."""
    return InstalledModel(
        filename=filename,
        path=Path(f"/fake/{filename}"),
        kind=kind,
        scale=scale,
        size_mb=size_mb,
        tainted=tainted,
        supported=supported,
        is_trained=is_trained,
    )


# --- sort_trained_first ----------------------------------------------------
def test_sort_trained_first_empty():
    """Empty input -> empty output."""
    assert sort_trained_first([]) == []


def test_sort_trained_first_already_sorted():
    """If trained already come first, the order is unchanged."""
    a = _m("a.pth", is_trained=True)
    b = _m("b.pth")
    c = _m("c.pth")
    out = sort_trained_first([a, b, c])
    assert [m.filename for m in out] == ["a.pth", "b.pth", "c.pth"]


def test_sort_trained_first_moves_trained_to_front():
    """All trained entries are pulled to the front, in original order."""
    a = _m("a.pth")
    b = _m("b.pth", is_trained=True)
    c = _m("c.pth")
    d = _m("d.pth", is_trained=True)
    out = sort_trained_first([a, b, c, d])
    # Original scan order preserved within each group:
    assert [m.filename for m in out] == ["b.pth", "d.pth", "a.pth", "c.pth"]


def test_sort_trained_first_does_not_mutate_input():
    """Input list is left untouched (sort returns a new list)."""
    a = _m("a.pth")
    b = _m("b.pth", is_trained=True)
    inp = [a, b]
    _ = sort_trained_first(inp)
    assert [m.filename for m in inp] == ["a.pth", "b.pth"]


# --- format_dropdown_label ------------------------------------------------
def test_format_label_basic():
    """A clean supported model has no tag suffix."""
    m = _m("foo.pth", kind="rfdn_student", scale=4, size_mb=18.3)
    label = format_dropdown_label(m, supported_kind=True)
    assert "foo.pth" in label
    assert "rfdn_student" in label
    assert "4x" in label
    assert "18.3 MB" in label
    # No tag suffix:
    assert TAG_TAINTED not in label
    assert TAG_TRAINED not in label
    assert TAG_UNSUPPORTED not in label
    assert TAG_NO_ARCH not in label


def test_format_label_trained():
    """In-house trained models get the [TRAINED] tag."""
    m = _m("foo.pth", is_trained=True)
    label = format_dropdown_label(m)
    assert label.endswith(TAG_TRAINED)


def test_format_label_tainted_overrides_trained():
    """Tainted tag wins over trained tag (taint is a stronger warning)."""
    m = _m("foo.pth", is_trained=True, tainted=True)
    label = format_dropdown_label(m)
    assert TAG_TAINTED in label
    assert TAG_TRAINED not in label


def test_format_label_unsupported():
    """supported=False on the model itself triggers the [unsupported] tag."""
    m = _m("foo.pth", supported=False)
    label = format_dropdown_label(m, supported_kind=True)
    assert TAG_UNSUPPORTED in label


def test_format_label_no_arch():
    """supported=True on the model but kind not registered -> [no arch]."""
    m = _m("foo.pth", supported=True)
    label = format_dropdown_label(m, supported_kind=False)
    assert TAG_NO_ARCH in label


def test_format_label_uses_question_mark_for_unknown_kind():
    """kind=None renders as a literal '?' so users see something is wrong."""
    m = _m("foo.pth", kind=None)
    label = format_dropdown_label(m, supported_kind=False)
    assert "?" in label


# --- pick_default_index ---------------------------------------------------
def test_pick_default_empty_list():
    """Empty list -> -1 (caller decides how to handle)."""
    assert pick_default_index([]) == -1
    assert pick_default_index([], last_model="anything.pth") == -1


def test_pick_default_last_model_substring_wins():
    """A substring match on last_model returns its index (priority 1)."""
    items = ["a.pth  [TRAINED]", "b.pth", "c.pth"]
    # Substring match in items[1]:
    assert pick_default_index(items, last_model="b.pth") == 1
    # Even though items[0] has [TRAINED], last_model substring wins:
    assert pick_default_index(items, last_model="a.pth") == 0


def test_pick_default_falls_back_to_first_trained():
    """When last_model has no match, pick the first [TRAINED] row."""
    items = ["a.pth", "b.pth  [TRAINED]", "c.pth  [TRAINED]"]
    assert pick_default_index(items) == 1


def test_pick_default_falls_back_to_first_row():
    """When no last_model match and no [TRAINED], pick row 0."""
    items = ["a.pth", "b.pth", "c.pth"]
    assert pick_default_index(items) == 0
    # last_model that doesn't match: still row 0
    assert pick_default_index(items, last_model="not_there.pth") == 0


def test_pick_default_empty_last_model_string():
    """last_model='' is treated as 'no preference' (skips priority 1)."""
    items = ["a.pth", "b.pth"]
    # last_model="" should NOT match the empty substring of any item.
    # (Empty-string substring matches everything, but the implementation
    # guards `if last_model:` so empty is treated as no preference.)
    assert pick_default_index(items, last_model="") == 0


# --- integration: sort + label + pick_default chain ----------------------
def test_trained_first_labeling_then_default_pick():
    """The full model_service chain produces a sane default selection.

    Simulates the orchestrator's flow: scan -> sort -> label -> default pick.
    """
    community = _m("realesr-animevideov3.pth", kind="srvgg", is_trained=False)
    trained_a = _m("RFDN_distill_v1_4x_student.pth", kind="rfdn_student", is_trained=True)
    trained_b = _m("SRVGG_distill_v1_4x_student.pth", kind="srvgg_student", is_trained=True)
    community2 = _m("animesr_v2.pth", kind="animesr", is_trained=False)

    sorted_models = sort_trained_first([community, trained_a, trained_b, community2])
    labels = [format_dropdown_label(m) for m in sorted_models]
    # Trained entries are first:
    assert labels[0].startswith("RFDN_distill")
    assert labels[1].startswith("SRVGG_distill")
    # No last_model -> first [TRAINED] row is picked:
    idx = pick_default_index(labels)
    assert idx == 0
    # last_model='realesr' matches the 3rd item (community_a):
    assert pick_default_index(labels, last_model="realesr") == 2
