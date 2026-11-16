"""Tiled inference: split input into overlapping tiles, blend with raised-cosine window.

Each tile is padded with `replicate` so the model sees a full context window (also
for images smaller than one tile). Overlap
regions between adjacent tiles are blended using a Hann-like window so seams vanish.

API:
    proc = _TileProcessor(model, tile_size=256, overlap=32)
    sr = proc.forward(x)  # x: (1, 3, H, W)
"""
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _raised_cosine_ramp(tile: int, overlap: int, device, dtype) -> torch.Tensor:
    """Build a 1D raised-cosine (Hann) blend ramp of `tile` samples.

    The first `overlap` samples fade 0 -> 1 and the last `overlap` samples fade
    1 -> 0, so adjacent ramps overlap and sum to ~1 where two tiles meet.
    """
    ramp = torch.ones(tile, device=device, dtype=dtype)
    if overlap <= 0:
        return ramp
    # Rising edge: [0:overlap] fades 0 -> 1
    rise = 0.5 * (1 - torch.cos(torch.linspace(0, torch.pi, overlap, device=device, dtype=dtype)))
    ramp[:overlap] = rise
    # Falling edge: [-overlap:] fades 1 -> 0
    ramp[-overlap:] = torch.flip(rise, dims=[0])
    return ramp


def _raised_cosine_window(tile: int, overlap: int, device, dtype) -> torch.Tensor:
    """Build a 2D raised-cosine (Hann) blend window for one tile.

    Output: (1, 1, tile, tile) where the overlap regions fade 1 -> 0 toward the
    matching edge so adjacent tiles sum to ~1 in their overlap.
    """
    ramp = _raised_cosine_ramp(tile, overlap, device, dtype)
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

        tile_out = self.tile * scale
        fade = self.overlap * scale

        def positions(size: int) -> list[int]:
            if size <= self.tile:
                return [0]
            pos = list(range(0, size - self.tile + 1, stride))
            if pos[-1] != size - self.tile:
                pos.append(size - self.tile)
            return pos

        for ty in positions(h):
            valid_h = min(self.tile, h - ty)
            for tx in positions(w):
                valid_w = min(self.tile, w - tx)
                tile = x[:, :, ty:ty + self.tile, tx:tx + self.tile]
                pad_h = self.tile - tile.shape[-2]
                pad_w = self.tile - tile.shape[-1]
                if pad_h or pad_w:
                    # replicate is defined even when pad >= dimension;
                    # reflect raises in torch for small images.
                    tile = F.pad(tile, (0, pad_w, 0, pad_h), mode="replicate")

                ramp_v = _raised_cosine_ramp(tile_out, fade, device, dtype)
                ramp_u = _raised_cosine_ramp(tile_out, fade, device, dtype)
                # A tile that touches the image border has no neighbour on that
                # side: keep full weight there. Otherwise the normalised output
                # fades out at the image edge (and a single-tile image, whose
                # tile sits on both edges, would come back all zeros).
                if ty == 0:
                    ramp_v[:fade] = 1.0
                if ty + valid_h >= h:
                    ramp_v[tile_out - fade:] = 1.0
                if tx == 0:
                    ramp_u[:fade] = 1.0
                if tx + valid_w >= w:
                    ramp_u[tile_out - fade:] = 1.0
                win = ramp_v[:, None] * ramp_u[None, :]

                # Replicated padding only exists for images smaller than a tile;
                # crop it back out before accumulating so no duplicate content
                # leaks into the result.
                crop_h, crop_w = valid_h * scale, valid_w * scale
                y_tile = self.model(tile) * win
                out[:, :, ty * scale:ty * scale + crop_h,
                    tx * scale:tx * scale + crop_w] += y_tile[:, :, :crop_h, :crop_w]
                weight[:, :, ty * scale:ty * scale + crop_h,
                       tx * scale:tx * scale + crop_w] += win[:crop_h, :crop_w]
        return out / weight.clamp(min=1e-6)

    def _infer_scale(self, x: torch.Tensor) -> int:
        """Probe the model to learn its upscaling factor."""
        probe_h = min(x.shape[-2], self.tile)
        probe_w = min(x.shape[-1], self.tile)
        with torch.no_grad():
            probe = self.model(x[:, :, :probe_h, :probe_w])
        ratio = probe.shape[-2] // probe_h
        if ratio <= 0:
            raise ValueError(
                f"model returned a smaller image ({tuple(probe.shape)}) than its input")
        return int(ratio)
