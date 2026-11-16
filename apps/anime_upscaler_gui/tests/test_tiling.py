"""Tests for the tiled inference module."""
import pytest
import torch
import torch.nn as nn


class Identity(nn.Module):
    def forward(self, x):
        return x


class Upscale2x(nn.Module):
    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")


def test_tile_identity_single_tile():
    from apps.anime_upscaler_gui.anime_upscaler_gui.tiling import _TileProcessor
    x = torch.arange(256 * 256, dtype=torch.float32).view(1, 1, 256, 256)
    proc = _TileProcessor(Identity(), tile_size=256, overlap=0)
    y = proc.forward(x)
    assert torch.allclose(x, y, atol=1e-5)


def test_tile_upscale_with_overlap():
    from apps.anime_upscaler_gui.anime_upscaler_gui.tiling import _TileProcessor
    x = torch.randn(1, 3, 128, 128)
    proc = _TileProcessor(Upscale2x(), tile_size=64, overlap=16)
    y = proc.forward(x)
    assert y.shape == (1, 3, 256, 256)


def test_raised_cosine_window():
    from apps.anime_upscaler_gui.anime_upscaler_gui.tiling import _raised_cosine_window
    w = _raised_cosine_window(128, 32, "cpu", torch.float32)
    assert w.shape == (1, 1, 128, 128)
    assert float(w.min()) >= 0.0
    assert float(w.max()) <= 1.0
    # Center should be 1, edges should fade to ~0
    assert w[0, 0, 64, 64] > 0.99
    assert w[0, 0, 0, 64] < 0.05
    assert w[0, 0, 127, 64] < 0.05


def test_tile_invalid_overlap():
    from apps.anime_upscaler_gui.anime_upscaler_gui.tiling import _TileProcessor
    # overlap == tile/2 -> stride <= 0; raises on forward, not on init
    proc = _TileProcessor(Identity(), tile_size=64, overlap=64)
    try:
        proc.forward(torch.zeros(1, 1, 64, 64))
        raised = False
    except ValueError:
        raised = True
    assert raised

class Shrink2x(nn.Module):
    """Model whose output is smaller than its input (invalid for a tiler)."""

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=0.5, mode="nearest")


class SmallConv(nn.Module):
    """Local 3x3 conv; tiled output must equal the full-image pass (no seams/shifts)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return self.conv(x)


def _pattern(h, w):
    """Deterministic distinct-valued gradient in [0, 1); catches shifted content."""
    return torch.linspace(0.0, 1.0, 3 * h * w).reshape(1, 3, h, w)


def _processor(model, tile_size=64, overlap=16):
    from apps.anime_upscaler_gui.anime_upscaler_gui.tiling import _TileProcessor
    return _TileProcessor(model, tile_size=tile_size, overlap=overlap)


# Below, at and around the tile size; 100x130 and 37x91 are not multiples of the
# 64 px tile and do not divide it.
IMAGE_SIZES = [(32, 24), (64, 64), (100, 130), (130, 100), (37, 91)]


def test_image_smaller_than_tile_does_not_crash():
    proc = _processor(Identity(), tile_size=64, overlap=16)
    out = proc.forward(torch.rand(1, 3, 32, 24))
    assert out.shape == (1, 3, 32, 24)


def test_non_multiple_size_is_exact():
    proc = _processor(Identity(), tile_size=64, overlap=16)
    out = proc.forward(torch.rand(1, 3, 100, 130))
    assert out.shape == (1, 3, 100, 130)


@pytest.mark.parametrize("h,w", IMAGE_SIZES)
def test_tiled_identity_matches_full_image(h, w):
    """Tiling must be invisible: every pixel equals the one-pass reference."""
    x = _pattern(h, w)
    out = _processor(Identity(), tile_size=64, overlap=16).forward(x)
    torch.testing.assert_close(out, x, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("h,w", IMAGE_SIZES)
def test_tiled_upscale_matches_full_image(h, w):
    """A pixel-wise 2x model must match the full-image result exactly."""
    x = _pattern(h, w)
    model = Upscale2x()
    out = _processor(model, tile_size=64, overlap=16).forward(x)
    ref = model(x)
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("h,w", [(32, 24), (64, 64), (100, 130), (130, 100)])
def test_tiled_local_conv_has_no_seams(h, w):
    """A 3x3 conv sees real neighbours on every tile edge, so seams must vanish.

    For images at least one tile large the tiled result equals the full-image
    pass everywhere. For smaller images the duplicate padding only affects the
    outermost output pixel, so compare the interior (1 px margin).
    """
    torch.manual_seed(0)
    model = SmallConv()
    x = _pattern(h, w)
    out = _processor(model, tile_size=64, overlap=16).forward(x)
    ref = model(x)
    margin = 1 if (h < 64 or w < 64) else 0
    if margin:
        out = out[:, :, margin:h - margin, margin:w - margin]
        ref = ref[:, :, margin:h - margin, margin:w - margin]
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=0.0)


def test_shrinking_model_raises_value_error():
    """The scale probe must fail loudly, not silently fall back to scale 4."""
    proc = _processor(Shrink2x(), tile_size=64, overlap=16)
    with pytest.raises(ValueError, match="smaller image"):
        proc.forward(torch.rand(1, 3, 64, 64))
