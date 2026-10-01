"""
SPAN: Swift Parameter-free Attention Network for Efficient Super-Resolution
NTIRE 2025 winning architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
from anime_sr.models.base import BaseSRModel
from anime_sr.models.span.parameter_free_attention import SwiftParameterFreeAttentionBlock
from anime_sr.models.span.conv_lora import ConvLoRA3x3


class UpsampleBlock(nn.Module):
    """Upsampling block using PixelShuffle + Conv"""
    
    def __init__(self, in_channels: int, out_channels: int, scale: int = 4):
        super().__init__()
        self.scale = scale
        
        # Conv before upsampling
        self.conv = nn.Conv2d(in_channels, out_channels * scale * scale, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x


class SPANModel(BaseSRModel):
    """
    SPAN: Swift Parameter-free Attention Network.
    
    Architecture:
    - Shallow feature extraction
    - Multiple SPAB (Swift Parameter-free Attention Blocks)
    - Upsampling via PixelShuffle
    - Residual connection from bicubic upsampled input
    
    Variants:
    - SPAN-Tiny: 26 channels, 12 blocks (EMSR NTIRE winner)
    - SPANF: 32 channels, fewer blocks (XiaomiMM variant)
    """
    
    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        num_channels: int = 26,  # 26 for Tiny, 32 for larger
        num_blocks: int = 12,    # 12 for Tiny
        use_lora: bool = True,
        lora_rank: int = 4,
    ):
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)
        
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.use_lora = use_lora
        
        # Shallow feature extraction
        if use_lora:
            self.shallow_conv = ConvLoRA3x3(in_channels, num_channels, r=lora_rank)
        else:
            self.shallow_conv = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1)
        
        # SPAB blocks
        self.blocks = nn.ModuleList([
            SwiftParameterFreeAttentionBlock(num_channels)
            for _ in range(num_blocks)
        ])
        
        # After blocks conv
        if use_lora:
            self.after_blocks_conv = ConvLoRA3x3(num_channels, num_channels, r=lora_rank)
        else:
            self.after_blocks_conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        
        # Upsampling
        self.upsample = UpsampleBlock(num_channels, out_channels, scale)
        
        # Store feature extraction points for FAKD
        self.feature_extraction_layers = [2, 4, 6, 8, 10, 12][:min(6, num_blocks)]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Low-resolution input [B, C, H, W]
        
        Returns:
            High-resolution output [B, C, H*scale, W*scale]
        """
        # Bicubic upsample for residual
        lr_upsampled = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        
        # Shallow feature extraction
        feat = self.shallow_conv(x)
        
        # SPAB blocks with residual
        shortcut = feat
        for block in self.blocks:
            feat = block(feat)
        
        feat = self.after_blocks_conv(feat)
        feat = feat + shortcut
        
        # Upsample
        out = self.upsample(feat)
        
        # Residual connection
        out = out + lr_upsampled
        
        # Clamp to valid range
        out = torch.clamp(out, 0, 1)
        
        return out
    
    def forward_with_features(self, x: torch.Tensor) -> tuple:
        """
        Forward pass with intermediate feature extraction (for FAKD).
        
        Returns:
            (output, features_dict)
        """
        features = {}
        
        lr_upsampled = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        feat = self.shallow_conv(x)
        
        # SPAB blocks
        shortcut = feat
        for i, block in enumerate(self.blocks):
            feat = block(feat)
            # Store features at specified layers
            if (i + 1) in self.feature_extraction_layers:
                features[i + 1] = feat.clone()
        
        feat = self.after_blocks_conv(feat)
        feat = feat + shortcut
        
        out = self.upsample(feat)
        out = out + lr_upsampled
        out = torch.clamp(out, 0, 1)
        
        return out, features
    
    def merge_lora_weights(self):
        """Merge all LoRA weights into main convolutions (for inference)"""
        if not self.use_lora:
            return
        
        # Merge shallow conv
        if hasattr(self.shallow_conv, 'merge_lora'):
            self.shallow_conv.merge_lora()
        
        # Merge after-blocks conv
        if hasattr(self.after_blocks_conv, 'merge_lora'):
            self.after_blocks_conv.merge_lora()
        
        print("LoRA weights merged. Model now has zero LoRA overhead.")
    
    def get_feature_maps(self, x: torch.Tensor, layer_indices: Optional[List[int]] = None) -> Dict[int, torch.Tensor]:
        """
        Extract intermediate feature maps for FAKD distillation.
        
        Args:
            x: Input tensor
            layer_indices: Specific layers to extract (None = all registered)
        
        Returns:
            Dictionary mapping layer index to feature tensor
        """
        _, features = self.forward_with_features(x)
        
        if layer_indices is not None:
            features = {k: v for k, v in features.items() if k in layer_indices}
        
        return features


class SPANTiny(SPANModel):
    """SPAN-Tiny: 26 channels, 12 blocks (EMSR NTIRE 2025 winner config)"""
    
    def __init__(self, scale: int = 4, **kwargs):
        super().__init__(
            scale=scale,
            num_channels=26,
            num_blocks=12,
            use_lora=True,
            lora_rank=4,
            **kwargs
        )


class SPANF(SPANModel):
    """SPANF: 32 channels, optimized for speed (XiaomiMM variant)"""
    
    def __init__(self, scale: int = 4, **kwargs):
        super().__init__(
            scale=scale,
            num_channels=32,
            num_blocks=10,  # Fewer blocks for speed
            use_lora=True,
            lora_rank=4,
            **kwargs
        )


def create_span_model(config: dict) -> SPANModel:
    """
    Factory function to create SPAN model from config.
    
    Args:
        config: Model configuration dict
    
    Returns:
        SPANModel instance
    """
    model_type = config.get('type', 'span').lower()
    scale = config.get('scale', 4)
    
    if model_type == 'span_tiny' or model_type == 'tiny':
        model = SPANTiny(scale=scale)
    elif model_type == 'spanf' or model_type == 'f':
        model = SPANF(scale=scale)
    elif model_type == 'checkpoint_compatible' or model_type == 'checkpoint':
        # Import here to avoid circular imports
        from .checkpoint_compatible_exact import CheckpointCompatibleSPANExact
        model = CheckpointCompatibleSPANExact(
            scale=scale,
            in_channels=config.get('in_channels', 3),
            out_channels=config.get('out_channels', 3),
            channels=config.get('channels', 48),
            hidden_channels=config.get('hidden_channels', 96),
            num_blocks=config.get('num_blocks', 6),
        )
    else:
        # Custom config
        model = SPANModel(
            scale=scale,
            num_channels=config.get('channels', 26),
            num_blocks=config.get('num_blocks', 12),
            use_lora=config.get('use_lora', True),
            lora_rank=config.get('lora_rank', 4),
        )
    
    return model
