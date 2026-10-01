"""Phase 4 I1 smoke test: RFDN works with both shortcut_mode values.

Verifies that switching shortcut_mode between "bicubic" and "nearest"
produces the same output shape and a different (non-equal) value when
the mode actually differs. Pure sanity check for the new arg.
"""
import pytest
import torch

from anime_sr.models.students import RFDN


def test_shortcut_mode_default():
    """Default shortcut_mode remains 'bicubic' for backward compatibility."""
    s = RFDN(scale=4)
    assert s.shortcut_mode == "bicubic", "default shortcut_mode must stay 'bicubic' for backward compat"


def test_shortcut_mode_nearest():
    """Nearest mode constructs and runs forward without error."""
    s = RFDN(scale=4, shortcut_mode="nearest")
    assert s.shortcut_mode == "nearest"
    lr = torch.randn(1, 3, 32, 32)
    sr, feats = s(lr, return_features=True)
    assert sr.shape == (1, 3, 128, 128)
    assert len(feats) == 4  # shallow + blk1 + blk3 + body_out


def test_set_shortcut_mode_runtime():
    """set_shortcut_mode() flips the active mode and changes the shortcut."""
    s = RFDN(scale=4)
    lr = torch.randn(1, 3, 32, 32)
    sr_b, _ = s(lr, return_features=True)
    s.set_shortcut_mode("nearest")
    assert s.shortcut_mode == "nearest"
    sr_n, _ = s(lr, return_features=True)
    # Output shapes must match; values must differ (different upsampling modes).
    assert sr_b.shape == sr_n.shape
    assert not torch.allclose(sr_b, sr_n), "bicubic and nearest shortcuts must produce different SR"


def test_shortcut_mode_invalid():
    """Invalid mode raises ValueError (no silent acceptance)."""
    with pytest.raises(ValueError):
        RFDN(scale=4, shortcut_mode="bilinear")
    s = RFDN(scale=4)
    with pytest.raises(ValueError):
        s.set_shortcut_mode("area")
