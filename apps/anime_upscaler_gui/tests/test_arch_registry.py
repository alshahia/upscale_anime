"""Tests for the extensible architecture registry and preset extension file.

No Tk needed -- pure backend behavior.
"""
import json

import pytest

from apps.anime_upscaler_gui.anime_upscaler_gui import archs
from apps.anime_upscaler_gui.anime_upscaler_gui.registry import (
    _ModelRegistry,
    _load_preset_catalog,
)


# --------------------------------------------------------------------------- #
# Built-in registry
# --------------------------------------------------------------------------- #

def test_builtin_kinds_registered():
    kinds = archs.registered_kinds()
    for expected in ("span", "srvgg", "srvgg_student", "era", "animesr", "rfdn_student"):
        assert expected in kinds


def test_is_supported_kind_unknown_is_false():
    assert archs.is_supported_kind("srvgg") is True
    assert archs.is_supported_kind("not_a_kind") is False


def test_capabilities_default_for_unknown_kind():
    caps = archs.capabilities("not_a_kind")
    assert caps == archs.Capability()


def test_animesr_capabilities():
    caps = archs.capabilities("animesr")
    assert caps.tiled is False
    assert caps.tensorrt is False
    assert caps.batch_video is False
    assert caps.recurrent_frames == 3
    assert caps.pad_multiple == 4
    assert caps.out_multiple == 4


def test_build_unknown_kind_raises_with_registered_kinds():
    with pytest.raises(ValueError) as ei:
        archs.build("not_a_kind", "whatever.pth")
    assert "span" in str(ei.value)


# --------------------------------------------------------------------------- #
# Detection delegation
# --------------------------------------------------------------------------- #

def test_detection_matches_registry_kind_strings():
    cases = [
        ({"conv_1.sk.weight": 1, "upsampler.0.weight": 1}, "span"),
        ({"body.0.weight": 1, "body.34.weight": 1}, "srvgg"),
        ({"head.weight": 1, "tail.weight": 1, "body.0.conv.ek.weight": 1}, "era"),
        ({"recurrent_cell.conv_s1_first.0.weight": 1}, "animesr"),
        ({"head.weight": 1, "body_tail.weight": 1, "pa.pa_conv.weight": 1,
          "upsampler.0.weight": 1, "blocks.0.d1.weight": 1}, "rfdn_student"),
    ]
    for state, expected in cases:
        assert archs.detect_kind_from_state(state) == expected


def test_detect_and_support_stay_in_sync():
    for kind in ("span", "srvgg", "srvgg_student", "era", "animesr", "rfdn_student"):
        assert archs.spec_of(kind) is not None


# --------------------------------------------------------------------------- #
# Extension contract: adding a model without touching pipeline/registry code
# --------------------------------------------------------------------------- #

class _DummyArch(object):
    def __init__(self, path):
        self.path = path
        self.upscale = 4
        self.evaluated = False

    def eval(self):
        self.evaluated = True
        return self


@pytest.fixture
def dummy_arch_registered():
    spec = archs.ArchSpec(
        kind="dummykind",
        detect=lambda state: "dummykind" if "dummy.magic.weight" in state else None,
        loader=lambda ckpt: _DummyArch(ckpt),
        default_scale=8,
        capability=archs.Capability(tiled=False),
    )
    archs.register_arch(spec)
    try:
        yield spec
    finally:
        archs.unregister_arch("dummykind")
    assert "dummykind" not in archs.registered_kinds()


def test_register_new_kind_is_buildable_and_supported(dummy_arch_registered):
    assert archs.is_supported_kind("dummykind") is True
    assert archs.capabilities("dummykind").tiled is False
    assert archs.capabilities("dummykind").tensorrt is True  # defaults preserved
    m = archs.build("dummykind", "fake.ckpt")
    assert isinstance(m, _DummyArch)
    assert m.evaluated is True
    assert m.path == "fake.ckpt"


def test_registering_kind_extends_kind_detection(dummy_arch_registered):
    assert archs.detect_kind_from_state({"dummy.magic.weight": 1}) == "dummykind"


# --------------------------------------------------------------------------- #
# Preset catalog extension file (registry.json)
# --------------------------------------------------------------------------- #

def test_load_preset_catalog_bundled_defaults():
    entries = _load_preset_catalog(None)
    ids = {e.id for e in entries}
    assert "realesr-animevideov3" in ids
    assert "rfdn_distill_v1_4x" in ids


def test_registry_merges_extension_entries(tmp_path):
    ext_file = tmp_path / "registry.json"
    custom = {"models": [
        {
            "id": "my_custom",
            "filename": "my_custom.pth",
            "url": "https://example.com/my_custom.pth",
            "scale": 2,
            "kind": "srvgg",
            "license": "MIT",
            "size_mb": 4.5,
            "description": "local extension preset",
            "source": "community",
            "display_name": "My Custom",
        },
        {"id": "broken", "filename": "broken.pth"},
        {"id": "badscale", "filename": "b.pth", "scale": -1,
         "kind": "srvgg", "license": "MIT", "size_mb": 1.0,
         "description": "nope"},
    ]}
    ext_file.write_text(json.dumps(custom), encoding="utf-8")

    reg = _ModelRegistry(tmp_path / "pretrained", presets_path=ext_file)
    pids = [p.id for p in reg.presets()]
    assert "my_custom" in pids
    assert "broken" not in pids
    assert "badscale" not in pids
    found = reg.preset_by_id("my_custom")
    assert found.scale == 2
    assert found.display_name == "My Custom"


def test_registry_extension_overrides_by_id(tmp_path):
    ext_file = tmp_path / "registry.json"
    custom = {"models": [{
        "id": "realesr-animevideov3",
        "filename": "realesr-animevideov3.pth",
        "scale": 4, "kind": "srvgg", "license": "CC0", "size_mb": 3.0,
        "description": "re-skinned from data",
    }]}
    ext_file.write_text(json.dumps(custom), encoding="utf-8")
    reg = _ModelRegistry(tmp_path / "pretrained", presets_path=ext_file)
    entry = reg.preset_by_id("realesr-animevideov3")
    assert entry.license == "CC0"
    children = [p for p in reg.presets() if p.id == "realesr-animevideov3"]
    assert len(children) == 1


def test_registry_malformed_extension_file_does_not_break(tmp_path):
    ext_file = tmp_path / "registry.json"
    ext_file.write_text("{ this is not json", encoding="utf-8")
    reg = _ModelRegistry(tmp_path / "pretrained", presets_path=ext_file)
    ids = {p.id for p in reg.presets()}
    assert "rfdn_distill_v1_4x" in ids  # baseline falls back intact


def test_registry_app_data_preset_path_merges_bundled(tmp_path):
    # In-tree bundled extension file + an app-data file: both merge.
    ext_file = tmp_path / "registry.json"
    custom = {"models": [{
        "id": "appdata_only",
        "filename": "appdata_only.pth",
        "scale": 3, "kind": "srvgg", "license": "MIT", "size_mb": 2,
        "description": "from app data",
    }]}
    ext_file.write_text(json.dumps(custom), encoding="utf-8")
    reg = _ModelRegistry(tmp_path / "pretrained", presets_path=ext_file)
    by_id = {p.id: p for p in reg.presets()}
    assert "appdata_only" in by_id
    assert "realesr-animevideov3" in by_id  # bundled baseline still present


def test_add_trained_prefix_extends_detection():
    from apps.anime_upscaler_gui.anime_upscaler_gui.registry import (
        add_trained_prefix, is_trained_filename,
    )
    assert not is_trained_filename("mynewmodel_v1.pth")
    add_trained_prefix("mynewmodel_")
    try:
        assert is_trained_filename("mynewmodel_v1.pth")
        # idempotent: re-adding does not duplicate
        add_trained_prefix("mynewmodel_")
        add_trained_prefix("   ")  # ignored
    finally:
        import apps.anime_upscaler_gui.anime_upscaler_gui.registry as reg_mod
        reg_mod._TRAINED_FILENAME_PATTERNS.remove("mynewmodel_")
    assert not is_trained_filename("mynewmodel_v1.pth")
