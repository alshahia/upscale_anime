"""Tiled inference: split input into overlapping tiles, blend with raised-cosine window.

Each tile is padded with `reflect` so the model sees a full context window. Overlap
regions between adjacent tiles are blended using a Hann-like window so seams vanish.

API:
    proc = _TileProcessor(model, tile_size=256, overlap=32)
    sr = proc.forward(x)  # x: (1, 3, H, W)
"""
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _raised_cosine_window(tile: int, overlap: int, device, dtype) -> torch.Tensor:
    """Build a 2D raised-cosine (Hann) blend window for one tile.

    Output: (1, 1, tile, tile) where the overlap regions fade 1 -> 0 toward the
    matching edge so adjacent tiles sum to ~1 in their overlap.
    """
    if overlap <= 0:
        return torch.ones(1, 1, tile, tile, device=device, dtype=dtype)
    ramp = torch.ones(tile, device=device, dtype=dtype)
    # Left ramp: tile[0:overlap] fades 0 -> 1
    ramp[:overlap] = 0.5 * (1 - torch.cos(torch.linspace(0, torch.pi, overlap, device=device, dtype=dtype)))
    # Right ramp: tile[-overlap:] fades 1 -> 0
    ramp[-overlap:] = torch.flip(ramp[:overlap], dims=[0])
    # Same for vertical axis
    return ramp[None, None, :, None] * ramp[None, None, None, :]  # outer product


class _TileProcessor:
    def __init__(self, model: nn.Module, tile_size: int = 256, overlap: int = 32):
        self.model = model
        self.tile = tile_size
        self.overlap = overlap

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (1, 3, H, W) -> (1, 3, H*scale, W*scale)."""
        b, c, h, w = x.shape
        scale = self._infer_scale(x)
        out_h, out_w = h * scale, w * scale
        device, dtype = x.device, x.dtype
        out = torch.zeros(1, c, out_h, out_w, device=device, dtype=dtype)
        weight = torch.zeros(1, 1, out_h, out_w, device=device, dtype=dtype)

        stride = self.tile - 2 * self.overlap
        if stride <= 0:
            raise ValueError("overlap too large for tile_size")

        win = _raised_cosine_window(self.tile * scale, self.overlap * scale, device, dtype)
        for ty in range(0, max(1, h - self.overlap), stride):
            ty = min(ty, max(0, h - self.tile))
            for tx in range(0, max(1, w - self.overlap), stride):
                tx = min(tx, max(0, w - self.tile))
                tile = x[:, :, ty:ty + self.tile, tx:tx + self.tile]
                # Pad if tile is at the edge and smaller than tile_size.
                pad_h = self.tile - tile.shape[2]
                pad_w = self.tile - tile.shape[3]
                if pad_h or pad_w:
                    tile = F.pad(tile, (0, pad_w, 0, pad_h), mode="reflect")
                y_tile = self.model(tile)
                y_tile = y_tile * win
                out[:, :, ty * scale:ty * scale + self.tile * scale,
                    tx * scale:tx * scale + self.tile * scale] += y_tile
                weight[:, :, ty * scale:ty * scale + self.tile * scale,
                       tx * scale:tx * scale + self.tile * scale] += win
        out = out / weight.clamp(min=1e-6)
        return out

    def _infer_scale(self, x: torch.Tensor) -> int:
        """Probe the model to figure out its upscaling factor."""
        try:
            with torch.no_grad():
                probe = self.model(x[:, :, :self.tile, :self.tile])
            return probe.shape[2] // self.tile
        except Exception:
            return 4