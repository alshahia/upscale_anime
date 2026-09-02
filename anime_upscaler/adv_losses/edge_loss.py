# anime_upscaler/losses/edge_loss.py
"""Sobel edge loss for adversarial SR (Phase 3).

EdgeLoss(sr, hr): L1 between |Sobel(sr_Y)| and |Sobel(hr_Y)| on the Y (luma)
channel only. Sobel kernels are fixed buffers -- no learnable parameters.

Why luma-only: chroma edges in anime are usually saturated flat fills (e.g. blue
sky, red hair) where high-frequency detail carries little semantic weight;
saving gradients on chroma only adds noise to G. BT.601 weights
(0.299, 0.587, 0.114) are the standard cv2-style RGB->Y.

Phase 3 handoff: docs/plans/student_adversarial_handoff_2026_08.md (A.2).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# Sobel kernels (3x3); magnitude is taken on [0, 1] floats -- scale cancels.
_SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0],
     [-2.0, 0.0, 2.0],
     [-1.0, 0.0, 1.0]], dtype=torch.float32
).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor(
    [[-1.0, -2.0, -1.0],
     [ 0.0,  0.0,  0.0],
     [ 1.0,  2.0,  1.0]], dtype=torch.float32
).view(1, 1, 3, 3)


def _rgb_to_y(rgb):
    """[B, 3, H, W] in [0, 1] -> [B, 1, H, W] luma (BT.601)."""
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


class EdgeLoss(nn.Module):
    """L1 edge loss on luma. L = mean(|grad(sr_Y)| - |grad(hr_Y)|)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("_kx", _SOBEL_X)
        self.register_buffer("_ky", _SOBEL_Y)

    def forward(self, sr, hr):
        y_sr = _rgb_to_y(sr.clamp(0, 1))
        y_hr = _rgb_to_y(hr.clamp(0, 1))
        gx_sr = F.conv2d(y_sr, self._kx, padding=1)
        gy_sr = F.conv2d(y_sr, self._ky, padding=1)
        gx_hr = F.conv2d(y_hr, self._kx, padding=1)
        gy_hr = F.conv2d(y_hr, self._ky, padding=1)
        mag_sr = torch.sqrt(gx_sr ** 2 + gy_sr ** 2 + 1e-12)
        mag_hr = torch.sqrt(gx_hr ** 2 + gy_hr ** 2 + 1e-12)
        return F.l1_loss(mag_sr, mag_hr)


__all__ = ["EdgeLoss"]
