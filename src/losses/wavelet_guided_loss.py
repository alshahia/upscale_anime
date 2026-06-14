"""
Wavelet-Guided GAN Loss for stable super-resolution training.
Uses Haar wavelet decomposition to guide GAN training and reduce artifacts.

Reference: WGSR (Wavelet-Guided Super-Resolution)
The wavelet decomposition separates low-frequency (structure) and high-frequency (texture)
components, allowing targeted guidance for each.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveletGuidedLoss(nn.Module):
    """
    Wavelet-Guided Loss for GAN stability and artifact reduction.

    Performs Haar wavelet decomposition on SR and HR images,
    then matches wavelet coefficients across subbands:
    - LL: Low-low (approximation/structure)
    - LH: Low-high (horizontal details)
    - HL: High-low (vertical details)
    - HH: High-high (diagonal details/texture)

    Reference: WGSR paper - wavelet_guided loss in neosr
    https://github.com/neosr-project/neosr/wiki/Losses
    """

    def __init__(
        self,
        weight: float = 1.0,
        hh_weight: float = 2.0,
        wavelet_init: int = 0,
    ):
        super().__init__()
        self.weight = weight
        self.hh_weight = hh_weight
        self.wavelet_init = wavelet_init

        haar_low = torch.tensor([0.5, 0.5], dtype=torch.float32)
        haar_high = torch.tensor([0.5, -0.5], dtype=torch.float32)
        self.register_buffer('low', haar_low)
        self.register_buffer('high', haar_high)

    def _haar_wavelet_dec(self, x: torch.Tensor) -> dict:
        """Perform single-level Haar wavelet decomposition."""
        b, c, h, w = x.shape

        if h % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1), mode='reflect')
        if w % 2 != 0:
            x = F.pad(x, (0, 1, 0, 0), mode='reflect')

        low_row = F.conv2d(x, self.low.view(1, 1, -1, 1).expand(c, -1, -1, -1), groups=c, stride=(2, 1))
        high_row = F.conv2d(x, self.high.view(1, 1, -1, 1).expand(c, -1, -1, -1), groups=c, stride=(2, 1))

        ll = F.conv2d(low_row, self.low.view(1, 1, 1, -1).expand(c, -1, -1, -1), groups=c, stride=(1, 2))
        lh = F.conv2d(low_row, self.high.view(1, 1, 1, -1).expand(c, -1, -1, -1), groups=c, stride=(1, 2))
        hl = F.conv2d(high_row, self.low.view(1, 1, 1, -1).expand(c, -1, -1, -1), groups=c, stride=(1, 2))
        hh = F.conv2d(high_row, self.high.view(1, 1, 1, -1).expand(c, -1, -1, -1), groups=c, stride=(1, 2))

        return {'ll': ll, 'lh': lh, 'hl': hl, 'hh': hh}

    def forward(self, pred: torch.Tensor, target: torch.Tensor, epoch: int = 0) -> torch.Tensor:
        if epoch < self.wavelet_init:
            return torch.tensor(0.0, device=pred.device, requires_grad=False)

        pred_w = self._haar_wavelet_dec(pred)
        target_w = self._haar_wavelet_dec(target)

        ll_loss = F.l1_loss(pred_w['ll'], target_w['ll'])
        lh_loss = F.l1_loss(pred_w['lh'], target_w['lh'])
        hl_loss = F.l1_loss(pred_w['hl'], target_w['hl'])
        hh_loss = F.l1_loss(pred_w['hh'], target_w['hh'])

        total = ll_loss + lh_loss + hl_loss + self.hh_weight * hh_loss

        if torch.isnan(total) or torch.isinf(total):
            return torch.tensor(0.0, device=pred.device, requires_grad=False)

        return self.weight * total
