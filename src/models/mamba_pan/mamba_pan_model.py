"""
Mamba-PAN: State Space Model with Pixel Attention Network
Combines Mamba's linear complexity with PAN's efficient attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
from models.base import BaseSRModel
from .hierarchical_mamba import HierarchicalMambaBlock, PixelAttentionMamba
from .mamba_utils import MAMBA_AVAILABLE


class UpsampleBlock(nn.Module):
    """Upsampling using PixelShuffle"""
    
    def __init__(self, in_channels: int, out_channels: int, scale: int = 4):
        super().__init__()
        self.scale = scale
        
        self.conv = nn.Conv2d(in_channels, out_channels * scale * scale, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x


class MambaPANModel(BaseSRModel):
    """
    Mamba-PAN Model for Efficient Super-Resolution.
    
    Architecture:
    - Shallow feature extraction
    - Multiple Hierarchical Mamba Blocks (HMB) with 4-direction scanning
    - Pixel attention between blocks
    - Upsampling via PixelShuffle
    - Residual connection
    
    Key advantages:
    - O(N) complexity instead of O(N²) like transformers
    - Direction-aware scanning captures multi-scale patterns
    - Efficient for anime content with sharp edges
    """
    
    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        num_channels: int = 32,
        num_blocks: int = 8,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        use_direction_fusion: bool = True,
    ):
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)
        
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        
        # Shallow feature extraction
        self.shallow_conv = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1)
        
        # Hierarchical Mamba Blocks with Pixel Attention
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleDict({
                'hmb': HierarchicalMambaBlock(
                    dim=num_channels,
                    d_state=mamba_d_state,
                    d_conv=mamba_d_conv,
                    expand=mamba_expand,
                    use_direction_fusion=use_direction_fusion,
                ),
                'pa': PixelAttentionMamba(num_channels),
            }))
        
        # After blocks conv
        self.after_blocks_conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        
        # Upsampling
        self.upsample = UpsampleBlock(num_channels, out_channels, scale)
        
        # Feature extraction points for FAKD
        self.feature_extraction_layers = list(range(2, num_blocks + 1, 2))
        
        # Print warning if using fallback
        if not MAMBA_AVAILABLE:
            print("Warning: Using fallback Mamba implementation. For better performance,")
            print("install mamba-ssm: pip install mamba-ssm causal-conv1d>=1.1.0")
    
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
        
        # Hierarchical Mamba Blocks
        shortcut = feat
        for block in self.blocks:
            # HMB with 4-direction scanning
            hmb_out = block['hmb'](feat)
            feat = hmb_out['fused']
            
            # Pixel attention
            feat = block['pa'](feat)
        
        # After blocks conv
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
        Forward with intermediate feature extraction (for FAKD).
        
        Returns:
            (output, features_dict)
        """
        features = {}
        
        lr_upsampled = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        feat = self.shallow_conv(x)
        
        # Hierarchical Mamba Blocks
        shortcut = feat
        for i, block in enumerate(self.blocks):
            # HMB with 4-direction scanning
            hmb_out = block['hmb'](feat)
            feat = hmb_out['fused']
            
            # Pixel attention
            feat = block['pa'](feat)
            
            # Store features at specified layers
            if (i + 1) in self.feature_extraction_layers:
                features[i + 1] = feat.clone()
        
        feat = self.after_blocks_conv(feat)
        feat = feat + shortcut
        
        out = self.upsample(feat)
        out = out + lr_upsampled
        out = torch.clamp(out, 0, 1)
        
        return out, features
    
    def get_direction_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get outputs from all 4 directions before fusion.
        Useful for directional consistency loss.
        
        Args:
            x: Input [B, C, H, W]
        
        Returns:
            Dict with 'h', 'v', 'rh', 'rv' keys
        """
        feat = self.shallow_conv(x)
        
        # Use first block to get direction outputs
        hmb_out = self.blocks[0]['hmb'](feat)
        
        # Return all direction outputs
        return {
            'h': hmb_out['h'],
            'v': hmb_out['v'],
            'rh': hmb_out['rh'],
            'rv': hmb_out['rv'],
        }
    
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


def create_mamba_pan_model(config: dict) -> MambaPANModel:
    """
    Factory function to create Mamba-PAN model from config.
    
    Args:
        config: Model configuration dict
    
    Returns:
        MambaPANModel instance
    """
    scale = config.get('scale', 4)
    
    model = MambaPANModel(
        scale=scale,
        in_channels=config.get('in_channels', 3),
        out_channels=config.get('out_channels', 3),
        num_channels=config.get('channels', 32),
        num_blocks=config.get('num_blocks', 8),
        mamba_d_state=config.get('mamba_d_state', 16),
        mamba_d_conv=config.get('mamba_d_conv', 4),
        mamba_expand=config.get('mamba_expand', 2),
        use_direction_fusion=config.get('use_direction_fusion', True),
    )
    
    return model
