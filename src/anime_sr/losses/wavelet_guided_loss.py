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

        # Phase 3: precompute the 4-output-channel 2x2 Haar kernel for the
        # fused decomposition. Each entry is the outer product of the row
        # filter [low|high] with the col filter [low|high], scaled by 0.25
        # because the original sequential path applies two 0.5-scaled 1D
        # convolutions in cascade. Channel order matches the dict keys below.
        # kernel row index -> H (rows of input); kernel col index -> W (cols of input).
        haar_kernel = torch.tensor([
            [[+0.25, +0.25], [+0.25, +0.25]],   # LL = avg_H x avg_W
            [[+0.25, -0.25], [+0.25, -0.25]],   # LH = avg_H x diff_W (row=avg, col=diff)
            [[+0.25, +0.25], [-0.25, -0.25]],   # HL = diff_H x avg_W (row=diff, col=avg)
            [[+0.25, -0.25], [-0.25, +0.25]],   # HH = diff_H x diff_W
        ], dtype=torch.float32)                  # shape [4, 2, 2]
        self.register_buffer('haar_kernel', haar_kernel)

    def _haar_wavelet_dec(self, x: torch.Tensor) -> dict:
        """Perform single-level Haar wavelet decomposition.

        Phase 3: fused into a single grouped 2x2 conv producing 4*C output
        channels (LL, LH, HL, HH per input channel). Slice channels to
        extract subbands. Previously: 6 sequential F.conv2d calls.
        """
        b, c, h, w = x.shape

        if h % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1), mode='reflect')
        if w % 2 != 0:
            x = F.pad(x, (0, 1, 0, 0), mode='reflect')

        # Expand the 4x2x2 kernel to depthwise groups=c (one set of 4 per channel).
        kernel = self.haar_kernel.view(4, 1, 2, 2).repeat(c, 1, 1, 1)
        out = F.conv2d(x, kernel, stride=2, groups=c)   # [B, 4*C, H/2, W/2]
        return {
            'll': out[:, 0::4],
            'lh': out[:, 1::4],
            'hl': out[:, 2::4],
            'hh': out[:, 3::4],
        }

    def forward(self, pred: torch.Tensor, target: torch.Tensor, epoch: int = 0) -> torch.Tensor:
        if epoch < self.wavelet_init:
            return torch.tensor(0.0, device=pred.device, requires_grad=False)

        # Target Haar decomposition is constant w.r.t. the generator; skip the
        # autograd graph to halve wavelet compute + memory.
        with torch.no_grad():
            target_w = self._haar_wavelet_dec(target)

        pred_w = self._haar_wavelet_dec(pred)

        ll_loss = F.l1_loss(pred_w['ll'], target_w['ll'])
        lh_loss = F.l1_loss(pred_w['lh'], target_w['lh'])
        hl_loss = F.l1_loss(pred_w['hl'], target_w['hl'])
        hh_loss = F.l1_loss(pred_w['hh'], target_w['hh'])

        total = ll_loss + lh_loss + hl_loss + self.hh_weight * hh_loss

        if torch.isnan(total) or torch.isinf(total):
            return torch.tensor(0.0, device=pred.device, requires_grad=False)

        return self.weight * total
