"""
EDSR: Enhanced Deep Residual Networks for Single Image Super-Resolution
Reference: https://arxiv.org/abs/1707.02921
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """EDSR Residual Block with residual scaling"""

    def __init__(self, channels: int, res_scale: float = 0.1):
        super().__init__()
        self.res_scale = res_scale

        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x) * self.res_scale
        return x + res


class Upsampler(nn.Module):
    """Upsampling module using pixel shuffle"""

    def __init__(self, scale: int, channels: int):
        super().__init__()
        self.scale = scale

        if scale == 2 or scale == 3:
            # Single upsampling layer
            self.conv = nn.Conv2d(channels, channels * scale * scale, kernel_size=3, padding=1, bias=False)
            self.pixel_shuffle = nn.PixelShuffle(scale)
        elif scale == 4:
            # Two 2x upsampling layers
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


class EDSR(nn.Module):
    """
    EDSR: Enhanced Deep Residual Network

    Args:
        n_resblocks: Number of residual blocks (default: 16, can be 32 for large model)
        n_feats: Number of feature channels (default: 64, can be 256 for large model)
        scale: Upsampling scale (2, 3, or 4)
        res_scale: Residual scaling factor (default: 0.1)
        n_colors: Number of input/output channels (default: 3 for RGB)
    """

    def __init__(
        self,
        n_resblocks: int = 16,
        n_feats: int = 64,
        scale: int = 4,
        res_scale: float = 0.1,
        n_colors: int = 3,
    ):
        super().__init__()
        self.scale = scale
        self.n_resblocks = n_resblocks
        self.n_feats = n_feats

        # Head: feature extraction
        self.head = nn.Conv2d(n_colors, n_feats, kernel_size=3, padding=1, bias=False)

        # Body: residual blocks
        modules = [ResidualBlock(n_feats, res_scale) for _ in range(n_resblocks)]
        modules.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=False))
        self.body = nn.Sequential(*modules)

        # Tail: upsampling + reconstruction
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

        # Body with skip connection
        res = self.body(x)
        x = x + res

        # Tail
        x = self.tail(x)

        return x

    def get_features(self, x: torch.Tensor) -> dict:
        """Extract intermediate features for FAKD"""
        features = {}

        # Head features
        x = self.head(x)
        features['head'] = x

        # Residual block features (sample every few blocks)
        for i, block in enumerate(self.body[:-1]):  # Exclude last conv
            x = block(x)
            if (i + 1) % 4 == 0:  # Save every 4th block
                features[f'resblock_{i+1}'] = x

        # Before tail
        features['body_out'] = x

        return features


def create_edsr(scale: int = 4, large: bool = False) -> EDSR:
    """
    Create EDSR model.

    Args:
        scale: Upsampling scale (2, 3, or 4)
        large: If True, create large EDSR (32 blocks, 256 features)
               If False, create base EDSR (16 blocks, 64 features)
    """
    if large:
        return EDSR(n_resblocks=32, n_feats=256, scale=scale)
    else:
        return EDSR(n_resblocks=16, n_feats=64, scale=scale)
