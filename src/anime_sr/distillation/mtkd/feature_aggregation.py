"""
Feature-Based Knowledge Aggregation
Aggregates teacher features in low-dimensional space before upsampling.
More efficient than output aggregation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


class FeatureExtractor(nn.Module):
    """
    Extract intermediate features from teacher models.
    Hooks into bottleneck layers before upsampling.
    """
    
    def __init__(self, teacher_model: nn.Module, layer_names: List[str] = None):
        super().__init__()
        self.teacher = teacher_model
        self.layer_names = layer_names or ['bottleneck', 'body']
        self.features = {}
        self.hooks = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate features."""
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
        
        # Try to find common layer names
        for name, module in self.teacher.named_modules():
            if any(target in name.lower() for target in self.layer_names):
                handle = module.register_forward_hook(get_hook(name))
                self.hooks.append(handle)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass and extract features."""
        self.features = {}  # Clear previous features
        with torch.no_grad():
            _ = self.teacher(x)
        return self.features
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class FeatureAlignmentModule(nn.Module):
    """
    Align features from different teachers to common dimension.
    Uses learnable projections.
    """
    
    def __init__(self, input_dims: List[int], target_dim: int = 64):
        super().__init__()
        
        self.input_dims = input_dims
        self.target_dim = target_dim
        
        # Create projection layers for each teacher
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, target_dim, 1),  # 1x1 conv for channel alignment
                nn.ReLU(inplace=True),
                nn.InstanceNorm2d(target_dim, affine=True),
            )
            for in_dim in input_dims
        ])
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Align and concatenate features from multiple teachers.
        
        Args:
            features: List of feature tensors [B, C_i, H, W]
            
        Returns:
            Aligned and concatenated features [B, N*target_dim, H, W]
        """
        # Project each to common dimension
        projected = []
        for feat, proj in zip(features, self.projections):
            # Resize if spatial dimensions differ
            target_size = features[0].shape[-2:]
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            
            # Project
            p = proj(feat)
            projected.append(p)
        
        # Concatenate
        return torch.cat(projected, dim=1)


class CrossTeacherAttention(nn.Module):
    """
    Cross-attention between different teacher features.
    Allows teachers to "communicate" and share knowledge.
    """
    
    def __init__(self, dim: int, num_teachers: int, num_heads: int = 4):
        super().__init__()
        
        self.dim = dim
        self.num_teachers = num_teachers
        self.num_heads = num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        # Layer norm
        self.norm = nn.LayerNorm(dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply cross-attention between teacher features.
        
        Args:
            features: List of [B, C, H, W] tensors
            
        Returns:
            List of attended features
        """
        b, c, h, w = features[0].shape
        
        # Flatten spatial dimensions and stack
        # [num_teachers, B, H*W, C]
        stacked = torch.stack([f.view(b, c, -1).permute(0, 2, 1) for f in features])
        
        # Reshape for attention: [B*H*W, num_teachers, C]
        num_spatial = h * w
        stacked = stacked.permute(1, 2, 0, 3).reshape(b * num_spatial, self.num_teachers, c)
        
        # Cross-attention (each position attends to all teachers)
        attended, _ = self.attention(stacked, stacked, stacked)
        
        # Reshape back
        attended = attended.reshape(b, num_spatial, self.num_teachers, c)
        attended = attended.permute(2, 0, 1, 3)  # [num_teachers, B, H*W, C]
        
        # Unflatten and return
        result = []
        for i, feat in enumerate(features):
            out = attended[i].permute(0, 2, 1).view(b, c, h, w)
            result.append(out)
        
        return result


class FeatureKnowledgeAggregation(nn.Module):
    """
    Aggregate teacher features in feature space (before upsampling).
    Much more efficient than output aggregation.
    """
    
    def __init__(
        self,
        num_teachers: int = 3,
        feature_dims: List[int] = None,
        aligned_dim: int = 64,
        num_blocks: int = 4,
        scale: int = 4,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        
        self.num_teachers = num_teachers
        self.scale = scale
        self.use_cross_attention = use_cross_attention
        
        # Default feature dimensions if not specified
        if feature_dims is None:
            # EDSR: 64, RCAN: 64, SwinIR: 180 (typical)
            feature_dims = [64, 64, 180]
        
        # Feature alignment
        self.alignment = FeatureAlignmentModule(feature_dims, aligned_dim)
        
        # Cross-teacher attention (optional)
        if use_cross_attention:
            self.cross_attention = CrossTeacherAttention(aligned_dim, num_teachers)
        
        # Aggregation blocks
        total_dim = aligned_dim * num_teachers
        self.aggregation_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(total_dim, total_dim, 3, padding=1, groups=num_teachers),  # Grouped conv
                nn.ReLU(inplace=True),
                nn.InstanceNorm2d(total_dim, affine=True),
                nn.Conv2d(total_dim, total_dim, 3, padding=1),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_blocks)
        ])
        
        # Fusion to single representation
        self.fusion = nn.Sequential(
            nn.Conv2d(total_dim, aligned_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(aligned_dim * 2, aligned_dim, 3, padding=1),
        )
        
        # Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(aligned_dim, 3 * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
        )
        
        self.use_gradient_checkpointing = False
    
    def forward(self, teacher_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate teacher features.
        
        Args:
            teacher_features: List of feature tensors from each teacher
            
        Returns:
            Aggregated output [B, 3, H*scale, W*scale]
        """
        # Align features to common dimension
        aligned = self.alignment(teacher_features)
        
        # Split back per-teacher for cross-attention
        b, _, h, w = aligned.shape
        aligned_dim = aligned.shape[1] // self.num_teachers
        split = torch.split(aligned, aligned_dim, dim=1)
        
        # Cross-teacher attention
        if self.use_cross_attention:
            attended = self.cross_attention(list(split))
            x = torch.cat(attended, dim=1)
        else:
            x = aligned
        
        # Aggregation blocks
        for block in self.aggregation_blocks:
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = x + 0.1 * block(x)  # Residual with scaling
        
        # Fusion
        x = self.fusion(x)
        
        # Upsample to HR
        x = self.upsample(x)
        
        return x
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing."""
        self.use_gradient_checkpointing = True
        print("  FeatureKnowledgeAggregation: gradient checkpointing enabled")


class HybridAggregation(nn.Module):
    """
    Hybrid aggregation combining feature-space and output-space.
    Best of both worlds.
    """
    
    def __init__(
        self,
        num_teachers: int = 3,
        feature_dims: List[int] = None,
        scale: int = 4,
        feature_weight: float = 0.6,
        output_weight: float = 0.4,
    ):
        super().__init__()
        
        self.feature_weight = feature_weight
        self.output_weight = output_weight
        
        # Feature aggregation branch
        self.feature_agg = FeatureKnowledgeAggregation(
            num_teachers=num_teachers,
            feature_dims=feature_dims,
            scale=scale,
        )
        
        # Output aggregation branch
        from anime_sr.distillation.mtkd.simple_aggregation import SimpleKnowledgeAggregation
        self.output_agg = SimpleKnowledgeAggregation(
            num_teachers=num_teachers,
            scale=scale,
            num_blocks=4,
        )
    
    def forward(
        self,
        teacher_outputs: List[torch.Tensor],
        teacher_features: List[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Hybrid forward using both features and outputs.
        
        Args:
            teacher_outputs: List of teacher SR outputs
            teacher_features: List of teacher intermediate features (optional)
            
        Returns:
            Aggregated output
        """
        # Output branch always available
        output_branch = self.output_agg(teacher_outputs)
        
        # Feature branch if features provided
        if teacher_features is not None:
            feature_branch = self.feature_agg(teacher_features)
            
            # Resize to match if needed
            if feature_branch.shape != output_branch.shape:
                feature_branch = F.interpolate(
                    feature_branch,
                    size=output_branch.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
            
            # Weighted combination
            return self.feature_weight * feature_branch + self.output_weight * output_branch
        else:
            return output_branch
