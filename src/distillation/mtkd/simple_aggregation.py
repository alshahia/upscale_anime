"""
Simple Knowledge Aggregation Network (NaN-free alternative)
Replaces complex DCTSwinBlocks with simple residual convolutions.
"""
import torch
import torch.nn as nn
from typing import List


class SimpleConvBlock(nn.Module):
    """Simple residual conv block without normalization (NaN-safe)."""
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.residual_scale = 0.1  # Small residual to prevent explosion
        
        # Initialize with small weights
        nn.init.kaiming_normal_(self.conv1.weight, a=0.1)
        nn.init.kaiming_normal_(self.conv2.weight, a=0.1)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return shortcut + self.residual_scale * x


class SimpleKnowledgeAggregation(nn.Module):
    """
    Simplified Knowledge Aggregation Network.
    
    This version:
    - Uses simple conv blocks instead of DCTSwinBlocks
    - No BatchNorm/InstanceNorm (no small-batch issues)
    - No DCT/IDCT (no potential numerical issues)
    - Small residual scales for stability
    
    Args:
        num_teachers: Number of teacher models
        in_channels: Input channels per teacher (typically 3 for RGB)
        embed_dim: Embedding dimension
        num_blocks: Number of residual blocks
        scale: Upscale factor
    """
    def __init__(
        self,
        num_teachers: int = 3,
        in_channels: int = 3,
        embed_dim: int = 64,
        num_blocks: int = 6,
        scale: int = 4,
    ):
        super().__init__()
        self.num_teachers = num_teachers
        self.scale = scale
        
        # Pixel unshuffle to reduce spatial resolution
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=scale)
        
        # Input projection (simple conv, no normalization)
        self.input_conv = nn.Conv2d(
            num_teachers * in_channels * scale * scale,
            embed_dim,
            kernel_size=3,
            padding=1,
        )
        nn.init.kaiming_normal_(self.input_conv.weight, a=0.1)
        nn.init.zeros_(self.input_conv.bias)
        
        # Simple residual blocks
        self.blocks = nn.ModuleList([
            SimpleConvBlock(embed_dim)
            for _ in range(num_blocks)
        ])
        
        # Output projection (simple conv, no normalization)
        self.output_conv = nn.Conv2d(embed_dim, in_channels * scale * scale, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.output_conv.weight, a=0.1)
        nn.init.zeros_(self.output_conv.bias)
        
        # Pixel shuffle for upsampling
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scale)
        
        self.use_gradient_checkpointing = False
    
    def gradient_checkpointing_enable(self):
        self.use_gradient_checkpointing = True
    
    def forward(self, teacher_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate multiple teacher outputs.
        
        Args:
            teacher_outputs: List of [B, C, H*scale, W*scale] tensors
        
        Returns:
            Enhanced output [B, C, H*scale, W*scale]
        """
        # Sanitize teacher outputs (clip extreme values)
        sanitized_outputs = []
        for out in teacher_outputs:
            # Clamp to valid range and replace NaN/Inf
            out = torch.clamp(out, min=-10.0, max=10.0)
            out = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
            sanitized_outputs.append(out)
        
        # Unshuffle each teacher output
        unshuffled = [self.pixel_unshuffle(out) for out in sanitized_outputs]
        
        # Concatenate
        x = torch.cat(unshuffled, dim=1)
        
        # Input projection
        x = self.input_conv(x)
        
        # Residual blocks
        if self.use_gradient_checkpointing and self.training:
            for block in self.blocks:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x)
        
        # Output projection
        x = self.output_conv(x)
        
        # Upsample
        x = self.pixel_shuffle(x)
        
        # Final sanitization
        x = torch.clamp(x, min=-10.0, max=10.0)
        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return x
