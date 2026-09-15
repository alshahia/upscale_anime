"""Tests for the tiled inference module."""
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