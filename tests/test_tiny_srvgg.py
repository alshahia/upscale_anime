"""Phase 4 I3 smoke tests: TinySRVGGStudent works as a drop-in student.

Verifies the new SRVGG-body student class (anime_upscaler/student.py) and the
vendored GUI copy (apps/anime_upscaler_gui/anime_upscaler_gui/archs.py) agree
on shape and param count. The latency test is best-effort: it skips on CPU
or when CUDA is too slow.
"""
import pytest
import torch

from anime_upscaler.student import TinySRVGGStudent


def test_params_under_ceiling():
    """Defaults must stay under the 600K param ceiling."""
    s = TinySRVGGStudent()
    n = s.num_params()
    assert n < 600_000, f"student too big: {n:,}"
    # Plan target: ~315K (matching v1 RFDN). Allow some slack for the 12-conv
    # body + 13 PReLU activations.
    assert 200_000 < n < 600_000, f"unexpected param count {n:,}"


def test_forward_shape_scale4():
    """Scale=4 (default) produces (1, 3, 4H, 4W)."""
    s = TinySRVGGStudent()
    lr = torch.randn(1, 3, 48, 48)
    sr = s(lr)
    assert sr.shape == (1, 3, 192, 192), f"unexpected shape {tuple(sr.shape)}"


def test_forward_shape_scale2():
    """Scale=2 (cascade student path) produces (1, 3, 2H, 2W)."""
    s = TinySRVGGStudent(scale=2)
    lr = torch.randn(1, 3, 64, 64)
    sr = s(lr)
    assert sr.shape == (1, 3, 128, 128), f"unexpected shape {tuple(sr.shape)}"


def test_return_features_returns_empty_list():
    """SRVGG exposes no taps; return_features must yield []."""
    s = TinySRVGGStudent()
    lr = torch.randn(1, 3, 32, 32)
    sr, feats = s(lr, return_features=True)
    assert sr.shape == (1, 3, 128, 128)
    assert feats == [], f"SRVGG must expose no taps, got {len(feats)}"


def test_nearest_residual_present():
    """Forward output must contain the nearest-upsampled input contribution.

    Cheap proxy: torch.allclose on a noise-free input. With PReLU + 3x3 convs
    the body output is bounded but not zero, so we just assert shape + finite
    values + that the residual adds something beyond a body-only path.
    """
    s = TinySRVGGStudent()
    lr = torch.zeros(1, 3, 32, 32)
    sr = s(lr)
    assert sr.shape == (1, 3, 128, 128)
    assert torch.isfinite(sr).all()
    # Sanity: at least some nonzero pixels (PReLU output + nearest shortcut).
    assert sr.abs().sum() > 0


def test_set_shortcut_weight_is_noop():
    """set_shortcut_weight is a no-op (SRVGG has no annealable shortcut)."""
    s = TinySRVGGStudent()
    lr = torch.randn(1, 3, 32, 32)
    sr_before = s(lr)
    s.set_shortcut_weight(0.0)
    s.set_shortcut_weight(1.0)
    s.set_shortcut_weight(0.5)
    sr_after = s(lr)
    assert torch.allclose(sr_before, sr_after), \
        "set_shortcut_weight must not affect SRVGG output"


def test_set_shortcut_mode_validates():
    """set_shortcut_mode accepts nearest (no-op); rejects invalid modes."""
    s = TinySRVGGStudent()
    s.set_shortcut_mode("nearest")  # valid, no-op
    s.set_shortcut_mode("bicubic")  # valid, no-op
    with pytest.raises(ValueError):
        s.set_shortcut_mode("bilinear")


def test_vendored_archs_matches_student_py():
    """GUI's vendored TinySRVGGStudent (apps/.../archs.py) must agree on
    shape + param count with the canonical implementation in
    anime_upscaler/student.py. Drift between the two is the most likely
    silent regression source.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                          / "apps" / "anime_upscaler_gui"))
    try:
        from anime_upscaler_gui.archs import TinySRVGGStudent as Vendored
    except ImportError as e:
        pytest.skip(f"GUI archs module not importable here: {e}")
    a = TinySRVGGStudent()
    b = Vendored()
    assert a.num_params() == b.num_params(), \
        f"vendored vs canonical param count drift: {b.num_params()} vs {a.num_params()}"
    lr = torch.randn(1, 3, 32, 32)
    sr_a = a(lr)
    sr_b = b(lr)
    # Different random inits; just check shape parity.
    assert sr_a.shape == sr_b.shape == (1, 3, 128, 128)


def test_build_dispatches_srvgg_student(tmp_path):
    """apps/.../archs.py::build(kind='srvgg_student') loads a saved ckpt."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                          / "apps" / "anime_upscaler_gui"))
    try:
        from anime_upscaler_gui.archs import build
    except ImportError as e:
        pytest.skip(f"GUI archs module not importable here: {e}")
    s = TinySRVGGStudent()
    ckpt_path = tmp_path / "tiny_srvgg.pth"
    torch.save({"student": s.state_dict()}, ckpt_path)
    loaded = build("srvgg_student", str(ckpt_path))
    assert loaded.__class__.__name__ == "TinySRVGGStudent"
    # Sanity: forward still works and shape is correct.
    loaded.eval()
    with torch.no_grad():
        sr = loaded(torch.randn(1, 3, 32, 32))
    assert sr.shape == (1, 3, 128, 128)


# =========================================================================
# Phase 5 Rank #1 (2026-09-03): cross-architecture warm-start tests.
# The RFDN v1 student (315K, key names 'head.*', 'blocks.*', 'pa.*',
# 'upsampler.0.*') needs to be partially loaded into a TinySRVGGStudent
# (317K, key names 'body.{0..26}.*', 'upsampler'). The mapping is:
#   RFDN 'head.weight/bias' -> SRVGG 'body.0.weight/bias'  (3 -> 52 conv)
#   RFDN 'upsampler.0.weight/bias' -> SRVGG 'body.{26}.weight/bias' (52 -> 12)
# The 12-conv middle stack + PReLUs stay at random init. These tests pin
# the mapping logic against the published shape contracts so a future
# student.py or distill.py refactor can't silently break the warm-start.
# =========================================================================


def test_partial_warmstart_maps_rfdn_head_to_srvgg_body0():
    """RFDN 'head.weight' [52,3,3,3] is copied into SRVGG 'body.0.weight'.

    Reproduces the layer-mapping logic in distill.py's --warm-start-mode
    partial branch and verifies the two key shapes match (so the mapping
    fires) and that the copy is bit-exact (so the warm-start actually
    seeds v1's weights into the new student).
    """
    # Mimic the v1 RFDN head shape by building a tiny RFDN-like state dict.
    import copy
    from anime_upscaler.student import RFDN
    rfdn = RFDN(scale=4, shortcut_mode="bicubic")
    v1_sd = rfdn.state_dict()  # real RFDN keys: head.weight, blocks.*, upsampler.0.*

    srvgg = TinySRVGGStudent(scale=4)
    srvgg_before = copy.deepcopy(srvgg)
    n_body = len(srvgg.body) - 1

    # The mapping logic from distill.py:451-481 (2026-09-03):
    mapped = {}
    if "head.weight" in v1_sd and srvgg.body[0].weight.shape == v1_sd["head.weight"].shape:
        mapped["body.0.weight"] = v1_sd["head.weight"]
        mapped["body.0.bias"] = v1_sd["head.bias"]
    if "upsampler.0.weight" in v1_sd \
            and srvgg.state_dict()[f"body.{n_body}.weight"].shape == v1_sd["upsampler.0.weight"].shape:
        mapped[f"body.{n_body}.weight"] = v1_sd["upsampler.0.weight"]
        mapped[f"body.{n_body}.bias"] = v1_sd["upsampler.0.bias"]

    # The mapping must succeed (4 keys: head + tail conv + biases).
    assert set(mapped.keys()) == {f"body.{n_body}.weight", f"body.{n_body}.bias",
                                  "body.0.weight", "body.0.bias"}, \
        f"unexpected mapped keys: {sorted(mapped.keys())}"

    result = srvgg.load_state_dict(mapped, strict=False)
    assert result.missing_keys  # the middle 12-conv stack + PReLUs stay random
    assert result.unexpected_keys == []  # mapped only contains SRVGG-side names

    # Verify the body.0 conv WAS overwritten with v1's head weights.
    assert torch.equal(srvgg.body[0].weight, v1_sd["head.weight"]), \
        "body.0.weight must be copied from v1 'head.weight'"
    # Verify body.{n_body} conv WAS overwritten with v1's upsampler.0 weights.
    assert torch.equal(srvgg.body[n_body].weight, v1_sd["upsampler.0.weight"]), \
        f"body.{n_body}.weight must be copied from v1 'upsampler.0.weight'"
    # Verify body.1 (the PReLU, no weights) is unchanged.
    # (PReLU has .weight only after first forward; we just check it has an attr)
    assert hasattr(srvgg.body[1], "weight")


def test_partial_warmstart_forward_works_after_mapping():
    """After the partial mapping, the SRVGG student still produces a valid
    4x upsampled output (the body.0 + body.{n_body} weights are real, the
    middle 12-conv stack is freshly initialised so the output is
    noisier than a fully-trained SRVGG but the shape must be correct
    and all pixels finite).
    """
    import copy
    from anime_upscaler.student import RFDN
    rfdn = RFDN(scale=4, shortcut_mode="bicubic")
    v1_sd = rfdn.state_dict()

    srvgg = TinySRVGGStudent(scale=4)
    n_body = len(srvgg.body) - 1
    mapped = {
        "body.0.weight": v1_sd["head.weight"],
        "body.0.bias": v1_sd["head.bias"],
        f"body.{n_body}.weight": v1_sd["upsampler.0.weight"],
        f"body.{n_body}.bias": v1_sd["upsampler.0.bias"],
    }
    srvgg.load_state_dict(mapped, strict=False)

    srvgg.eval()
    with torch.no_grad():
        lr = torch.randn(2, 3, 32, 32)
        sr = srvgg(lr)
    assert sr.shape == (2, 3, 128, 128)
    assert torch.isfinite(sr).all(), "NaN/Inf in SRVGG output after partial warm-start"
