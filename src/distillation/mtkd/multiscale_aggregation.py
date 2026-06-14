"""
Multi-Scale Knowledge Aggregation Network
Processes teacher outputs at multiple resolutions for better detail capture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ScaleAggregationBlock(nn.Module):
    """Aggregation block for a specific scale."""
    
    def __init__(self, num_teachers: int, in_channels: int, embed_dim: int, scale: int):
        super().__init__()
        
        # Pixel unshuffle for this scale
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=scale)
        
        # Input projection
        total_channels = num_teachers * in_channels * scale * scale
        self.input_conv = nn.Sequential(
            nn.Conv2d(total_channels, embed_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Simple residual blocks
        self.conv1 = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Output projection
        self.output_conv = nn.Conv2d(embed_dim, in_channels * scale * scale, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scale)
        
        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, teacher_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Process at this scale."""
        # Unshuffle
        unshuffled = [self.pixel_unshuffle(out) for out in teacher_outputs]
        x = torch.cat(unshuffled, dim=1)
        
        # Project
        x = self.input_conv(x)
        
        # Residual block
        shortcut = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x = shortcut + 0.1 * x
        
        # Output
        x = self.output_conv(x)
        x = self.pixel_shuffle(x)
        
        return x


class MultiScaleKnowledgeAggregation(nn.Module):
    """
    Multi-scale knowledge aggregation that fuses teacher outputs at multiple resolutions.
    
    Architecture:
    - Process at original resolution + downsampled scales
    - Fuse multi-scale features
    - Upsample and combine
    """
    
    def __init__(
        self,
        num_teachers: int = 3,
        in_channels: int = 3,
        embed_dim: int = 96,
        scale: int = 4,
        scale_factors: List[int] = None,
    ):
        super().__init__()
        
        self.num_teachers = num_teachers
        self.scale = scale
        self.scale_factors = scale_factors or [1, 2]
        
        # Create aggregation block for each scale
        self.scale_blocks = nn.ModuleList([
            ScaleAggregationBlock(num_teachers, in_channels, embed_dim, scale)
            for _ in self.scale_factors
        ])
        
        # Fusion network
        fusion_channels = in_channels * len(self.scale_factors)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_channels, embed_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, in_channels, 3, padding=1),
        )
        
        self.use_gradient_checkpointing = False
    
    def forward(self, teacher_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate at multiple scales and fuse.
        
        Args:
            teacher_outputs: List of [B, C, H*scale, W*scale] tensors
            
        Returns:
            Enhanced output [B, C, H*scale, W*scale]
        """
        # Get original size
        orig_size = teacher_outputs[0].shape[-2:]
        
        # Process at each scale
        scale_outputs = []
        for block in self.scale_blocks:
            # Process at this scale
            out = block(teacher_outputs)
            # Resize to original
            if out.shape[-2:] != orig_size:
                out = F.interpolate(out, size=orig_size, mode='bilinear', align_corners=False)
            scale_outputs.append(out)
        
        # Concatenate all scales
        x = torch.cat(scale_outputs, dim=1)
        
        # Fuse
        x = self.fusion_conv(x)
        
        return x
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing."""
        self.use_gradient_checkpointing = True
        print("  MultiScaleKnowledgeAggregation: gradient checkpointing enabled")
