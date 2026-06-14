"""
Hierarchical Mamba Block (HMB) with 4-direction scanning
Directional scanning: Horizontal, Vertical, Reverse-H, Reverse-V
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
from .mamba_utils import (
    create_mamba_layer,
    scan_horizontal,
    scan_vertical,
    reverse_scan_horizontal,
    reverse_scan_vertical,
)


class DirectionalMambaLayer(nn.Module):
    """
    Single Mamba layer with one directional scanning.
    """
    
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        scan_type: str = 'h',  # 'h', 'v', 'rh', 'rv'
    ):
        super().__init__()
        self.dim = dim
        self.scan_type = scan_type
        self.flip = scan_type.startswith('r')  # Reverse scans flip the input
        
        # LayerNorm
        self.norm = nn.LayerNorm(dim)
        
        # Mamba layer
        self.mamba = create_mamba_layer(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward with directional scanning.
        
        Args:
            x: Input [B, C, H, W]
        
        Returns:
            Output [B, C, H, W]
        """
        B, C, H, W = x.shape
        shortcut = x
        
        # Normalize and reshape
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm(x)
        
        # Scan
        if 'h' in self.scan_type:
            x_seq, shape = scan_horizontal(x.permute(0, 3, 1, 2), flip=self.flip)
        else:  # vertical
            x_seq, shape = scan_vertical(x.permute(0, 3, 1, 2), flip=self.flip)
        
        # Apply Mamba
        x_seq = self.mamba(x_seq)
        
        # Reverse scan
        if 'h' in self.scan_type:
            x = reverse_scan_horizontal(x_seq, shape, flip=self.flip)
        else:
            x = reverse_scan_vertical(x_seq, shape, flip=self.flip)
        
        # Residual
        x = x + shortcut
        
        return x


class HierarchicalMambaBlock(nn.Module):
    """
    Hierarchical Mamba Block with 4-direction scanning.
    Combines H, V, RH, RV scans with learnable fusion.
    """
    
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_direction_fusion: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.use_direction_fusion = use_direction_fusion
        
        # Four directional Mamba layers
        self.mamba_h = DirectionalMambaLayer(dim, d_state, d_conv, expand, 'h')
        self.mamba_v = DirectionalMambaLayer(dim, d_state, d_conv, expand, 'v')
        self.mamba_rh = DirectionalMambaLayer(dim, d_state, d_conv, expand, 'rh')
        self.mamba_rv = DirectionalMambaLayer(dim, d_state, d_conv, expand, 'rv')
        
        if use_direction_fusion:
            # Learnable fusion weights
            self.fusion_weights = nn.Parameter(torch.ones(4))
            self.fusion_conv = nn.Conv2d(dim * 4, dim, kernel_size=1)
        else:
            # Simple averaging
            self.fusion_conv = None
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward with 4-direction scanning.
        
        Args:
            x: Input [B, C, H, W]
        
        Returns:
            Dict of direction outputs: {'h': tensor, 'v': tensor, 'rh': tensor, 'rv': tensor}
            Plus 'fused' key with combined output
        """
        # Apply each direction
        out_h = self.mamba_h(x)
        out_v = self.mamba_v(x)
        out_rh = self.mamba_rh(x)
        out_rv = self.mamba_rv(x)
        
        outputs = {
            'h': out_h,
            'v': out_v,
            'rh': out_rh,
            'rv': out_rv,
        }
        
        # Fusion
        if self.use_direction_fusion:
            # Apply learnable weights
            weights = F.softmax(self.fusion_weights, dim=0)
            
            # Concatenate all directions
            concat = torch.cat([out_h, out_v, out_rh, out_rv], dim=1)  # [B, C*4, H, W]
            
            # Learnable fusion
            fused = self.fusion_conv(concat)  # [B, C, H, W]
        else:
            # Simple average
            fused = (out_h + out_v + out_rh + out_rv) / 4.0
        
        outputs['fused'] = fused
        
        return outputs


class PixelAttentionMamba(nn.Module):
    """
    PAN-style Pixel Attention combined with Mamba.
    Uses channel attention + spatial attention before Mamba processing.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Pixel attention (similar to PAN)
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.pa_sigmoid = nn.Sigmoid()
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply pixel and channel attention.
        
        Args:
            x: [B, C, H, W]
        
        Returns:
            [B, C, H, W]
        """
        shortcut = x
        
        # Pixel attention
        pa = self.pa_conv(x)
        pa = self.pa_sigmoid(pa)
        x = x * pa
        
        # Channel attention
        ca = self.avg_pool(x)
        ca = self.ca_conv(ca)
        x = x * ca
        
        # Residual
        x = x + shortcut
        
        return x
