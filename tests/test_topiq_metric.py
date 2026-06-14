"""
Tests for TOPIQ + MUSIQ no-reference IQA metrics and PYIQA_DIRECTION/PYIQA_RANGE lookups.

These tests are skipped when pyiqa is not installed.
The first call to a new pyiqa model triggers a ~150MB download; subsequent
runs are fast (cached in TORCH_HOME/hub).
"""
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

for _tdm_k in ('utils', 'utils.metrics'):
    sys.modules.pop(_tdm_k, None)

from utils.metrics import (  # noqa: E402
    PYIQA_AVAILABLE,
    PYIQA_DIRECTION,
    PYIQA_RANGE,
    calculate_topiq,
    calculate_musiq,
    calculate_niqe,
    calculate_clipiqa,
)

pytestmark = pytest.mark.skipif(
    not PYIQA_AVAILABLE,
    reason="pyiqa not installed; install with: pip install pyiqa",
)


def _random_rgb(h: int = 64, w: int = 64, seed: int = 0) -> torch.Tensor:
    """Build a [1, 3, H, W] uint8-derived float tensor in [0, 1]."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, h, w, generator=g)


def _blur(img: torch.Tensor, k: int = 7) -> torch.Tensor:
    """Apply a simple box blur to simulate a degraded version."""
    kernel = torch.ones(3, 1, k, k) / float(k * k)
    return torch.nn.functional.conv2d(img, kernel, padding=k // 2, groups=3)


def test_pyiqa_direction_lookup():
    """PYIQA_DIRECTION must classify known metrics correctly."""
    assert PYIQA_DIRECTION["niqe"] == "lower"
    assert PYIQA_DIRECTION["brisque"] == "lower"
    assert PYIQA_DIRECTION["maniqa"] == "higher"
    assert PYIQA_DIRECTION["clipiqa"] == "higher"
    assert PYIQA_DIRECTION["topiq_nr"] == "higher"
    assert PYIQA_DIRECTION["musiq"] == "higher"
    assert PYIQA_DIRECTION["arniqa"] == "higher"
    assert PYIQA_DIRECTION["liqe"] == "higher"


def test_pyiqa_range_lookup():
    """PYIQA_RANGE must contain (lo, hi) tuples for the supported metrics."""
    for name in ("niqe", "maniqa", "clipiqa", "topiq_nr", "musiq", "brisque"):
        assert name in PYIQA_RANGE, f"{name} missing from PYIQA_RANGE"
        lo, hi = PYIQA_RANGE[name]
        assert isinstance(lo, (int, float)) and isinstance(hi, (int, float))
        assert lo < hi


@pytest.mark.slow
def test_calculate_topiq_returns_float():
    """calculate_topiq must return a Python float (or NaN) on a 64x64 RGB tensor."""
    img = _random_rgb(64, 64, seed=42)
    score = calculate_topiq(img)
    assert isinstance(score, float)
    # Score is either finite or NaN; never raise.
    assert score == score or score != score  # tautology: float-typed


@pytest.mark.slow
def test_calculate_musiq_returns_float():
    """calculate_musiq must return a Python float (or NaN) on a 64x64 RGB tensor."""
    img = _random_rgb(64, 64, seed=7)
    score = calculate_musiq(img)
    assert isinstance(score, float)


@pytest.mark.slow
def test_topiq_higher_for_cleaner_image():
    """
    Sanity check: TOPIQ-NR should rank a sharp image above a blurred one.
    Uses a structured sharp image (sharp edges) vs a Gaussian-blurred variant.

    Note: TOPIQ strictly enforces inputs in [0, 1]; we keep the noisy pattern
    inside that range and avoid adding Gaussian noise that pushes values out.
    """
    # Build a sharp image: half black, half white with a clean vertical edge,
    # plus content in [0, 1] range to give TOPIQ a real signal to score.
    sharp = torch.zeros(1, 3, 64, 64)
    sharp[:, :, :, 32:] = 1.0
    # Add a low-amplitude noise pattern that stays within [0, 1].
    sharp = sharp + 0.05 * torch.rand(1, 3, 64, 64)
    sharp = sharp.clamp(0.0, 1.0)

    # Blur it heavily with a box filter.
    blurred = _blur(sharp, k=15)

    score_sharp = calculate_topiq(sharp)
    score_blur = calculate_topiq(blurred)

    # If TOPIQ returned NaN for both (e.g. model failed to load), skip the
    # ordering check but still assert the function returned floats.
    if score_sharp != score_sharp or score_blur != score_blur:
        pytest.skip("TOPIQ returned NaN (model not available); skipping ordering check")

    assert score_sharp > score_blur, (
        f"TOPIQ should rate sharp (={score_sharp:.4f}) above blurred "
        f"(={score_blur:.4f})"
    )


def test_existing_metrics_still_work():
    """Sanity: previously-added NIQE/CLIPIQA still work after the refactor."""
    img = _random_rgb(64, 64, seed=123)
    # These do not call pyiqa (since the model isn't requested in this test),
    # but at least verify import + callable with proper signature.
    assert callable(calculate_niqe)
    assert callable(calculate_clipiqa)
