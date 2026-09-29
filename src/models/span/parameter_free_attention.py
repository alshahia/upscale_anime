"""
Parameter-Free Attention for SPAN
Uses channel mean and standard deviation (no learnable parameters)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ParameterFreeChannelAttention(nn.Module):
    """
    Parameter-Free Channel Attention (PFCA).
    Computes attention weights from channel statistics (mean, std).
    No learnable parameters - pure function of input.
    
    Reference: SPAN paper (Swift Parameter-free Attention Network)
    """
    
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply parameter-free channel attention.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Attention-weighted output [B, C, H, W]
        """
        b, c, h, w = x.shape
        
        # Compute channel-wise mean and std
        # Mean: [B, C, 1, 1]
        mean = x.mean(dim=[2, 3], keepdim=True)
        
        # Std: [B, C, 1, 1]
        std = x.std(dim=[2, 3], keepdim=True) + self.eps
        
        # Compute attention weights: higher activation -> higher weight
        # weight = mean / std (coefficient of variation inspired)
        # This emphasizes channels with high mean relative to variance
        weight = mean / (std + self.eps)
        
        # Normalize weights across channels
        weight = weight / (weight.sum(dim=1, keepdim=True) + self.eps)
        
        # Apply attention
        out = x * weight
        
        return out


class ParameterFreeSpatialAttention(nn.Module):
    """
    Parameter-Free Spatial Attention.
    Uses channel-wise statistics to create spatial attention map.
    """
    
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply parameter-free spatial attention.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Attention-weighted output [B, C, H, W]
        """
        # Compute spatial attention map from channel statistics
        # Max and mean across channels
        max_val, _ = x.max(dim=1, keepdim=True)  # [B, 1, H, W]
        mean_val = x.mean(dim=1, keepdim=True)   # [B, 1, H, W]
        
        # Combine max and mean
        spatial_att = max_val + mean_val
        
        # Normalize with softmax over spatial positions. Sum-normalization
        # would scale each pixel's weight to ~1/(H*W), shrinking the output.
        B, _, H, W = x.shape
        spatial_att = F.softmax(spatial_att.view(B, 1, -1), dim=-1).view(B, 1, H, W)

        # Apply
        out = x * spatial_att
        
        return out


class ParameterFreeAttentionBlock(nn.Module):
    """
    Combined parameter-free attention: channel + spatial.
    Core building block for SPAN.
    """
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.channel_att = ParameterFreeChannelAttention()
        self.spatial_att = ParameterFreeSpatialAttention()
        
        # Learnable fusion weights (only learnable parameters in this block)
        self.channel_weight = nn.Parameter(torch.ones(1))
        self.spatial_weight = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply both channel and spatial attention.
        
        Args:
            x: Input [B, C, H, W]
        
        Returns:
            Output [B, C, H, W]
        """
        # Channel attention branch
        ch_out = self.channel_att(x)
        
        # Spatial attention branch
        sp_out = self.spatial_att(x)
        
        # Weighted fusion
        out = self.channel_weight * ch_out + self.spatial_weight * sp_out
        
        return out


class SwiftParameterFreeAttentionBlock(nn.Module):
    """
    SPAB: Swift Parameter-free Attention Block.
    Main block for SPAN network.
    
    Architecture:
    - Two-branch structure (main branch + attention branch)
    - Parameter-free attention in attention branch
    - Residual connection
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        use_att: bool = True,
    ):
        super().__init__()
        
        # Main branch: simple conv
        self.main_conv = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        
        # Attention branch
        self.use_att = use_att
        if use_att:
            self.attention = ParameterFreeAttentionBlock(channels)
        
        # Activation
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [B, C, H, W]
        
        Returns:
            [B, C, H, W]
        """
        shortcut = x
        
        # Main branch
        main = self.main_conv(x)
        
        # Attention branch (if enabled)
        if self.use_att:
            att = self.attention(x)
            # Combine: main + attention-modulated input
            out = main + att
        else:
            out = main
        
        # Activation
        out = self.activation(out)
        
        # Residual
        out = out + shortcut
        
        return out
