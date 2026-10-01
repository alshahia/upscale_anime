"""PatchGAN discriminator + Hinge GAN loss for adversarial SR (Phase 3).

PatchGAN70 (4 strided conv blocks C64->C128->C256->C512, kernel=4 stride=2,
LeakyReLU 0.2, spectral norm on every conv, 1x1 head to a 1-channel logit map).
Input is concat(LR_upsampled_to_HR, SR_or_HR) = 6 channels; output for 256x256
input is [B, 1, 16, 16] (spatial /16 because 4 strided layers).

Reference: src/losses/adversarial_loss.py::SRDiscriminator (used for shape only).
We REPLACE BatchNorm with spectral-norm-only (Ganin et al., 2016), and use Hinge
loss (Lim & Ye, 2017) for both D and G; no-logits are exposed.

Phase 3 handoff: docs/plans/student_adversarial_handoff_2026_08.md (A.1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _spectral_conv(in_ch, out_ch, kernel, stride, padding):
    """Conv2d wrapped in spectral_norm (Miyato 2018)."""
    return nn.utils.spectral_norm(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, bias=True)
    )


class PatchGAN70(nn.Module):
    """70x70 receptive-field PatchGAN (Isola et al., 2017) with spectral norm.

    4 strided conv blocks (C64,C128,C256,C512), kernel=4 stride=2 padding=1,
    LeakyReLU(0.2). Spectral norm on every conv. Final 1x1 to 1-channel logit map.
    Input: 6 channels = concat(LR_upsampled_to_HR, SR_or_HR).
    Output: [B, 1, H/16, W/16] for 4 stride-2 layers.

    Why spectral norm: stabilizes D, avoids checkerboard artifacts that BatchNorm
    produces in low-res SR. Why Hinge (not BCE/relativistic): converges faster at
    the small batch sizes (16) we use on RTX 4000, and is the standard recipe in
    ESRGAN / RealESRGAN follow-on work.
    """

    def __init__(self, in_channels=6, num_features=64):
        super().__init__()
        nf = num_features
        self.conv1 = _spectral_conv(in_channels, nf, 4, 2, 1)
        self.conv2 = _spectral_conv(nf, nf * 2, 4, 2, 1)
        self.conv3 = _spectral_conv(nf * 2, nf * 4, 4, 2, 1)
        self.conv4 = _spectral_conv(nf * 4, nf * 8, 4, 2, 1)
        self.head = _spectral_conv(nf * 8, 1, kernel=1, stride=1, padding=0)
        self.act = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))
        h = self.act(self.conv4(h))
        return self.head(h)


class HingeGANLoss(nn.Module):
    """Hinge GAN loss (Lim & Ye, 2017).

    D loss = mean(relu(1 - D(real))) + mean(relu(1 + D(fake)))
    G loss = -mean(D(fake))

    forward(d_real, d_fake) returns (d_loss, g_loss) -- a single call
    gives both losses; the caller zeroes/updates the right optimizer.
    """

    def forward(self, d_real, d_fake):
        d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
        g_loss = -d_fake.mean()
        return d_loss, g_loss


__all__ = ["PatchGAN70", "HingeGANLoss"]
