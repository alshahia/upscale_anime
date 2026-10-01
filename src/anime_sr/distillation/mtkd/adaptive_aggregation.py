"""
Adaptive Teacher Aggregation Network
Learns per-input teacher importance weights using attention mechanism
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class TeacherGatingNetwork(nn.Module):
    """
    Learns input-dependent weights for each teacher.
    Uses SE-style channel attention to determine teacher importance.
    """
    
    def __init__(self, num_teachers: int, embed_dim: int, reduction: int = 4):
        super().__init__()
        self.num_teachers = num_teachers
        
        # Global feature extraction
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Teacher importance prediction
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // reduction, num_teachers, bias=True),
        )
        
        # Initialize to uniform weights (softmax of 0 = 1/num_teachers)
        nn.init.zeros_(self.fc[-1].bias)
        nn.init.kaiming_normal_(self.fc[0].weight, mode='fan_in', nonlinearity='relu')
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute teacher weights based on input features.
        
        Args:
            features: [B, embed_dim, H, W] concatenated teacher features
            
        Returns:
            Teacher weights [B, num_teachers] (sum to 1)
        """
        # Global pooling
        pooled = self.global_pool(features).flatten(1)  # [B, embed_dim]
        
        # Predict teacher weights
        logits = self.fc(pooled)  # [B, num_teachers]
        weights = F.softmax(logits, dim=1)  # Sum to 1
        
        return weights


class AdaptiveConvBlock(nn.Module):
    """
    Residual conv block with adaptive instance normalization.
    More stable than batch norm for varying batch sizes.
    """
    
    def __init__(self, dim: int, use_norm: bool = True):
        super().__init__()
        self.use_norm = use_norm
        
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        
        if use_norm:
            # InstanceNorm is more stable than BatchNorm for small batches
            self.norm1 = nn.InstanceNorm2d(dim, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim, affine=True)
        
        self.relu = nn.ReLU(inplace=True)
        self.residual_scale = 0.1
        
        # Initialize
        nn.init.kaiming_normal_(self.conv1.weight, a=0.1)
        nn.init.kaiming_normal_(self.conv2.weight, a=0.1)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        
        x = self.conv1(x)
        if self.use_norm:
            x = self.norm1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        if self.use_norm:
            x = self.norm2(x)
        
        return shortcut + self.residual_scale * x


class AdaptiveTeacherAggregation(nn.Module):
    """
    Adaptive Knowledge Aggregation Network with learned teacher weighting.
    
    Features:
    - Input-dependent teacher importance weighting
    - Residual conv blocks with instance normalization
    - Optional gradient checkpointing for memory efficiency
    """
    
    def __init__(
        self,
        num_teachers: int = 3,
        in_channels: int = 3,
        embed_dim: int = 96,
        num_blocks: int = 6,
        scale: int = 4,
        use_norm: bool = True,
    ):
        super().__init__()
        
        self.num_teachers = num_teachers
        self.scale = scale
        self.use_gradient_checkpointing = False
        
        # Pixel unshuffle to reduce spatial resolution
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=scale)
        
        # Input projection
        total_channels = num_teachers * in_channels * scale * scale
        self.input_conv = nn.Sequential(
            nn.Conv2d(total_channels, embed_dim, 3, padding=1),
            nn.InstanceNorm2d(embed_dim, affine=True) if use_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        
        # Teacher gating network
        self.teacher_gating = TeacherGatingNetwork(num_teachers, embed_dim)
        
        # Teacher-specific processing branches
        self.teacher_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // num_teachers, 3, padding=1),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_teachers)
        ])
        
        # Fusion convolution
        fusion_channels = (embed_dim // num_teachers) * num_teachers
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_channels, embed_dim, 3, padding=1),
            nn.InstanceNorm2d(embed_dim, affine=True) if use_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            AdaptiveConvBlock(embed_dim, use_norm=use_norm)
            for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_conv = nn.Sequential(
            nn.InstanceNorm2d(embed_dim, affine=True) if use_norm else nn.Identity(),
            nn.Conv2d(embed_dim, in_channels * scale * scale, 3, padding=1),
        )
        
        # Pixel shuffle
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scale)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu', a=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d)):
                if m.affine:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
    
    def forward(self, teacher_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate multiple teacher outputs with learned weighting.
        
        Args:
            teacher_outputs: List of [B, C, H*scale, W*scale] tensors
            
        Returns:
            Enhanced output [B, C, H*scale, W*scale]
        """
        # Unshuffle each teacher output
        unshuffled = [self.pixel_unshuffle(out) for out in teacher_outputs]
        
        # Concatenate all teacher outputs
        x = torch.cat(unshuffled, dim=1)  # [B, num_teachers*C*scale^2, H, W]
        
        # Input projection
        features = self.input_conv(x)  # [B, embed_dim, H, W]
        
        # Compute teacher importance weights
        teacher_weights = self.teacher_gating(features)  # [B, num_teachers]
        
        # Process each teacher branch with learned weighting
        branch_outputs = []
        for i, branch in enumerate(self.teacher_branches):
            # Extract features for this teacher
            branch_out = branch(features)
            # Apply learned weight
            weight = teacher_weights[:, i:i+1, None, None]  # [B, 1, 1, 1]
            branch_outputs.append(weight * branch_out)
        
        # Concatenate all branches
        fused = torch.cat(branch_outputs, dim=1)
        
        # Fusion convolution
        x = self.fusion_conv(fused)
        
        # Residual blocks with optional gradient checkpointing
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
        
        return x
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.use_gradient_checkpointing = True
        print("  AdaptiveTeacherAggregation: gradient checkpointing enabled")
    
    def get_teacher_weights(self, teacher_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Get learned teacher weights for analysis/debugging.
        
        Args:
            teacher_outputs: List of teacher output tensors
            
        Returns:
            Teacher weights [B, num_teachers]
        """
        with torch.no_grad():
            unshuffled = [self.pixel_unshuffle(out) for out in teacher_outputs]
            x = torch.cat(unshuffled, dim=1)
            features = self.input_conv(x)
            weights = self.teacher_gating(features)
        return weights
