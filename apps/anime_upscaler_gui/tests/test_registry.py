"""Tests for the model registry."""
import sys
import tempfile
import uuid
from pathlib import Path


def _fresh_pretrained_dir() -> Path:
    d = Path(tempfile.gettempdir()) / f"aug_pretrained_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_registry_detects_srvgg_keys(tmp_path=Path(tempfile.gettempdir()) / f"aug_{uuid.uuid4().hex[:6]}"):
    from apps.anime_upscaler_gui.anime_upscaler_gui.registry import _detect_kind_from_state
    # Fake srvgg state
    state = {f"body.{i}.weight": None for i in range(35)}
    assert _detect_kind_from_state(state) == "srvgg"


def test_registry_detects_span_keys():
    from apps.anime_upscaler_gui.anime_upscaler_gui.registry import _detect_kind_from_state
    state = {
        "conv_1.sk.weight": None,
        "conv_1.eval_conv.weight": None,
        "block_1.c1_r.sk.weight": None,
        "upsampler.0.weight": None,
    }
    assert _detect_kind_from_state(state) == "span"


def test_registry_detects_era_keys():
    from apps.anime_upscaler_gui.anime_upscaler_gui.registry import _detect_kind_from_state
    state = {
        "head.weight": None,
        "body.0.conv.ek": None,
        "body.0.conv.sk.weight": None,
        "tail.weight": None,
    }
    assert _detect_kind_from_state(state) == "era"


def test_registry_detects_animesr_keys():
    from apps.anime_upscaler_gui.anime_upscaler_gui.registry import _detect_kind_from_state
    state = {
        "recurrent_cell.conv_s1_first.0.weight": None,
        "body_s1_first.0.conv1.weight": None,
        "fusion.0.weight": None,
    }
    assert _detect_kind_from_state(state) == "animesr"


def test_registry_detects_rfdn_student_keys():
    """Our in-house RFDN student (anime_upscaler.student.RFDN)."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.registry import _detect_kind_from_state
    state = {
        "head.weight": None,
        "blocks.0.d1.weight": None,
        "blocks.0.r1.weight": None,
        "blocks.5.pa.pa_conv.weight": None,
        "body_tail.weight": None,
        "pa.pa_conv.weight": None,
        "upsampler.0.weight": None,
    }
    assert _detect_kind_from_state(state) == "rfdn_student"
    # Without the unique blocks.<N>.d1.weight marker, should not match rfdn.
    state_no_d1 = {k: v for k, v in state.items() if ".d1.weight" not in k}
    assert _detect_kind_from_state(state_no_d1) is None


def test_registry_detects_rfdn_inside_student_wrapper():
    """KD trainer wraps the student under a top-level 'student' key.

    `scan_installed` must unwrap that before calling _detect_kind_from_state.
    """
    from apps.anime_upscaler_gui.anime_upscaler_gui.registry import _detect_kind_from_state
    student_inner = {
        "head.weight": None,
        "blocks.0.d1.weight": None,
        "blocks.0.r1.weight": None,
        "body_tail.weight": None,
        "pa.pa_conv.weight": None,
        "upsampler.0.weight": None,
    }
    wrapped = {"epoch": 40, "val_psnr": 29.32, "args": {},
               "student": student_inner, "adapters": {"a.0.weight": None}}
    # _detect_kind_from_state does NOT unwrap -- that's scan_installed's job --
    # so calling it directly on the wrapped dict returns None (no head.weight at top).
    assert _detect_kind_from_state(wrapped) is None
    # But scan_installed must extract student_inner first. We test that path
    # implicitly via test_registry_scan_with_real_checkpoints which scans
    # pretrained/ -- the v1 student is placed there.


def test_registry_scan_finds_rfdn_student():
    """If the v1 student checkpoint is in pretrained/, the scan must detect it."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.registry import _ModelRegistry
    pre = Path(__file__).resolve().parents[3] / "pretrained"
    if not pre.exists():
        return  # skip in minimal envs
    reg = _ModelRegistry(pre)
    installed = reg.scan_installed()
    rfdn = next((m for m in installed if m.filename == "RFDN_distill_v1_4x_student.pth"), None)
    if rfdn is None:
        return  # vendored file not present in this checkout
    assert rfdn.kind == "rfdn_student"
    assert rfdn.scale == 4
    assert rfdn.supported is True



def test_registry_detects_srvgg_student_keys():
    """Our Step-2 TinySRVGG student (anime_upscaler/student.py::TinySRVGGStudent)."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.registry import _detect_kind_from_state
    # Layout mirrors the real ckpt: convs at even body indices, PReLU weights at
    # odd indices + the final even index, upsampler has no params.
    keys = {}
    for i in range(0, 27, 2):
        keys[f"body.{i}.weight"] = None
        keys[f"body.{i}.bias"] = None
    for i in range(1, 26, 2):
        keys[f"body.{i}.weight"] = None
    assert _detect_kind_from_state(keys) == "srvgg_student"


def test_registry_scan_finds_srvgg_student():
    """If the step-2 student checkpoint is in pretrained/, the scan must detect it."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.registry import _ModelRegistry
    pre = Path(__file__).resolve().parents[3] / "pretrained"
    if not (pre / "SRVGG_distill_v1_4x_student.pth").exists():
        return  # vendored file not present in this checkout
    reg = _ModelRegistry(pre)
    installed = reg.scan_installed()
    m = next((x for x in installed if x.filename == "SRVGG_distill_v1_4x_student.pth"), None)
    assert m is not None
    assert m.kind == "srvgg_student"
    assert m.scale == 4
    assert m.supported is True
    assert m.is_trained is True

def test_registry_scan_with_real_checkpoints():
    from apps.anime_upscaler_gui.anime_upscaler_gui.registry import _ModelRegistry
    # Use the repo's real pretrained/ to confirm scan produces sane metadata.
    pre = Path(__file__).resolve().parents[3] / "pretrained"
    if not pre.exists():
        return  # skip when repo layout isn't present
    reg = _ModelRegistry(pre)
    installed = reg.scan_installed()
    names = [m.filename for m in installed]
    assert "realesr-animevideov3.pth" in names
    srvgg = next(m for m in installed if m.filename == "realesr-animevideov3.pth")
    assert srvgg.kind == "srvgg"
    assert srvgg.scale == 4
    assert srvgg.supported is True
    assert srvgg.tainted is False


def test_registry_presets_listed():
    from apps.anime_upscaler_gui.anime_upscaler_gui.registry import _ModelRegistry
    pre = _fresh_pretrained_dir()
    reg = _ModelRegistry(pre)
    presets = reg.presets()
    assert len(presets) >= 5
    ids = [p.id for p in presets]
    assert "realesr-animevideov3" in ids
    assert "span_pix_pretrain_4x" in ids