"""
RCAN: Residual Channel Attention Networks
Reference: https://arxiv.org/abs/1807.02758
Channel Attention mechanism for SR with very deep networks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention (CA) module.
    Squeeze-and-Excitation style attention.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C, H, W]
        Returns:
            Channel-attended features [B, C, H, W]
        """
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    """
    Residual Channel Attention Block (RCAB).
    Residual block with channel attention.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        reduction: int = 16,
        res_scale: float = 1.0,
    ):
        super().__init__()
        self.res_scale = res_scale

        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            ChannelAttention(channels, reduction),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x) * self.res_scale
        return x + res


class ResidualGroup(nn.Module):
    """
    Residual Group (RG) containing multiple RCABs.
    Implements RIR (Residual In Residual) structure.
    """

    def __init__(
        self,
        channels: int,
        n_rcabs: int,
        kernel_size: int = 3,
        reduction: int = 16,
        res_scale: float = 1.0,
    ):
        super().__init__()
        modules = [RCAB(channels, kernel_size, reduction, res_scale) for _ in range(n_rcabs)]
        modules.append(nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False))
        self.body = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x)
        return x + res * 0.1  # Short skip connection with scaling


class Upsampler(nn.Module):
    """Upsampling module using pixel shuffle"""

    def __init__(self, scale: int, channels: int):
        super().__init__()
        self.scale = scale

        if scale == 2 or scale == 3:
            self.conv = nn.Conv2d(channels, channels * scale * scale, kernel_size=3, padding=1, bias=False)
            self.pixel_shuffle = nn.PixelShuffle(scale)
        elif scale == 4:
            self.conv1 = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1, bias=False)
            self.pixel_shuffle1 = nn.PixelShuffle(2)
            self.conv2 = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1, bias=False)
            self.pixel_shuffle2 = nn.PixelShuffle(2)
        else:
            raise ValueError(f"Unsupported scale: {scale}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2 or self.scale == 3:
            x = self.conv(x)
            x = self.pixel_shuffle(x)
        elif self.scale == 4:
            x = self.conv1(x)
            x = self.pixel_shuffle1(x)
            x = self.conv2(x)
            x = self.pixel_shuffle2(x)
        return x


class RCAN(nn.Module):
    """
    RCAN: Residual Channel Attention Network

    Args:
        n_groups: Number of residual groups (default: 10)
        n_rcabs: Number of RCABs per group (default: 20)
        n_feats: Number of feature channels (default: 64)
        scale: Upsampling scale (2, 3, or 4)
        reduction: Channel reduction factor for attention (default: 16)
        n_colors: Number of input/output channels (default: 3 for RGB)
    """

    def __init__(
        self,
        n_groups: int = 10,
        n_rcabs: int = 20,
        n_feats: int = 64,
        scale: int = 4,
        reduction: int = 16,
        n_colors: int = 3,
    ):
        super().__init__()
        self.scale = scale
        self.n_groups = n_groups
        self.n_rcabs = n_rcabs
        self.n_feats = n_feats

        # Head
        self.head = nn.Conv2d(n_colors, n_feats, kernel_size=3, padding=1, bias=False)

        # Body: residual groups
        modules = [
            ResidualGroup(n_feats, n_rcabs, reduction=reduction)
            for _ in range(n_groups)
        ]
        modules.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=False))
        self.body = nn.Sequential(*modules)

        # Tail
        self.tail = nn.Sequential(
            Upsampler(scale, n_feats),
            nn.Conv2d(n_feats, n_colors, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input LR image [B, 3, H, W]
        Returns:
            HR image [B, 3, scale*H, scale*W]
        """
        # Head
        x = self.head(x)
        res = x

        # Body
        x = self.body(x)

        # Long skip connection (RIR)
        x = x + res

        # Tail
        x = self.tail(x)

        return x

    def get_features(self, x: torch.Tensor) -> dict:
        """Extract intermediate features for FAKD"""
        features = {}

        # Head
        x = self.head(x)
        features['head'] = x
        res = x

        # Residual groups (sample a few)
        for i, group in enumerate(self.body[:-1]):  # Exclude last conv
            x = group(x)
            if (i + 1) % 3 == 0:  # Save every 3rd group
                features[f'rg_{i+1}'] = x

        # Long skip and body out
        x = x + res
        features['body_out'] = x

        return features


def create_rcan(scale: int = 4) -> RCAN:
    """
    Create RCAN model with default configuration.
    """
    return RCAN(n_groups=10, n_rcabs=20, n_feats=64, scale=scale)
