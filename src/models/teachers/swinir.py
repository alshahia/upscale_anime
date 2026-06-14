"""
SwinIR: Image Restoration Using Swin Transformer
Reference: https://arxiv.org/abs/2108.10257
Swin Transformer based architecture for classical image SR.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MLP(nn.Module):
    """Multi-layer perceptron for Swin Transformer"""

    def __init__(self, in_features: int, hidden_features: int = None, drop: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """
    Window-based Multi-head Self-Attention (W-MSA).
    Supports both shifted and non-shifted windows.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        shifted: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.shifted = shifted
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Get relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B*N, window_size*window_size, C]
        Returns:
            Attended features [B*N, window_size*window_size, C]
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            attn = attn + mask.unsqueeze(1).unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block with alternating shifted and non-shifted window attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        shifted: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shifted = shifted
        self.shift_size = window_size // 2 if shifted else 0

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size, num_heads, qkv_bias, attn_drop, drop, shifted
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, drop)

    def window_partition(self, x: torch.Tensor) -> torch.Tensor:
        """Partition into windows"""
        B, H, W, C = x.shape
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)
        return windows

    def window_reverse(self, windows: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Reverse window partition"""
        B = int(windows.shape[0] / (H * W / self.window_size / self.window_size))
        x = windows.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [B, H, W, C]
        Returns:
            Output [B, H, W, C]
        """
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        # Cyclic shift if needed
        if self.shifted:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition into windows
        x_windows = self.window_partition(shifted_x)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        # Merge windows
        shifted_x = self.window_reverse(attn_windows, H, W)

        # Reverse cyclic shift
        if self.shifted:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class RSTB(nn.Module):
    """
    Residual Swin Transformer Block (RSTB).
    Multiple Swin Transformer layers with residual connection.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                shifted=(i % 2 == 1),  # Alternate between regular and shifted
            )
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [B, C, H, W]
        Returns:
            Output [B, C, H, W]
        """
        shortcut = x
        x = x.permute(0, 2, 3, 1)  # B, H, W, C

        for layer in self.layers:
            x = layer(x)

        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        x = self.conv(x)
        x = x + shortcut
        return x


class Upsampler(nn.Module):
    """Pixel shuffle upsampler"""

    def __init__(self, scale: int, channels: int):
        super().__init__()
        if scale == 2 or scale == 3:
            self.conv = nn.Conv2d(channels, channels * scale * scale, kernel_size=3, padding=1, bias=False)
            self.pixel_shuffle = nn.PixelShuffle(scale)
        elif scale == 4:
            self.conv1 = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1, bias=False)
            self.pixel_shuffle1 = nn.PixelShuffle(2)
            self.conv2 = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1, bias=False)
            self.pixel_shuffle2 = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'conv'):
            x = self.conv(x)
            x = self.pixel_shuffle(x)
        else:
            x = self.conv1(x)
            x = self.pixel_shuffle1(x)
            x = self.conv2(x)
            x = self.pixel_shuffle2(x)
        return x


class SwinIR(nn.Module):
    """
    SwinIR: Image Restoration Using Swin Transformer
    Classical SR variant (SwinIR-S for classical SR).

    Args:
        img_size: Input image size (default: 64)
        patch_size: Patch embedding size (default: 1)
        in_chans: Number of input channels (default: 3)
        embed_dim: Patch embedding dimension (default: 180 for SwinIR-M, 60 for SwinIR-S)
        depths: Number of RSTB blocks at each level
        num_heads: Number of attention heads at each level
        window_size: Window size for attention (default: 8)
        mlp_ratio: MLP hidden dimension ratio (default: 2.0)
        scale: Upsampling scale (2, 3, or 4)
    """

    def __init__(
        self,
        img_size: int = 64,
        in_chans: int = 3,
        embed_dim: int = 180,
        depths: list = None,
        num_heads: list = None,
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        scale: int = 4,
    ):
        super().__init__()
        self.scale = scale
        self.embed_dim = embed_dim
        self.window_size = window_size

        if depths is None:
            depths = [6, 6, 6, 6, 6, 6]
        if num_heads is None:
            num_heads = [6, 6, 6, 6, 6, 6]

        # Shallow feature extraction
        self.conv_first = nn.Conv2d(in_chans, embed_dim, kernel_size=3, padding=1, bias=False)

        # Deep feature extraction (RSTBs)
        self.layers = nn.ModuleList([
            RSTB(
                dim=embed_dim,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
            )
            for i in range(len(depths))
        ])
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False)

        # Upsampling
        self.upsample = Upsampler(scale, embed_dim)
        self.conv_last = nn.Conv2d(embed_dim, in_chans, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input LR [B, 3, H, W]
        Returns:
            HR [B, 3, scale*H, scale*W]
        """
        # Shallow feature extraction
        x = self.conv_first(x)
        shortcut = x

        # Deep feature extraction
        for layer in self.layers:
            x = layer(x)
        x = self.conv_after_body(x)
        x = x + shortcut  # Long skip connection

        # Upsampling
        x = self.upsample(x)
        x = self.conv_last(x)

        return x

    def get_features(self, x: torch.Tensor) -> dict:
        """Extract intermediate features for FAKD"""
        features = {}

        # Shallow feature
        x = self.conv_first(x)
        features['shallow'] = x
        shortcut = x

        # RSTB features (sample every few layers)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if (i + 1) % 2 == 0:  # Save every 2nd layer
                features[f'rstb_{i+1}'] = x

        # After body
        x = self.conv_after_body(x)
        x = x + shortcut
        features['deep'] = x

        return features


def create_swinir(scale: int = 4, small: bool = False) -> SwinIR:
    """
    Create SwinIR model.

    Args:
        scale: Upsampling scale
        small: If True, create lightweight SwinIR-S (embed_dim=60)
               If False, create SwinIR-M (embed_dim=180)
    """
    if small:
        return SwinIR(embed_dim=60, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6], scale=scale)
    else:
        return SwinIR(embed_dim=180, depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6], scale=scale)
