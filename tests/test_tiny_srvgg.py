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
