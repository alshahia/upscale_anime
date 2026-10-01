"""
Feature Affinity Knowledge Distillation (FAKD)
Transfers second-order statistics (feature correlations) from teacher to student
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


class FeatureAffinityLoss(nn.Module):
    """
    FAKD: Feature Affinity-based Knowledge Distillation
    
    Computes spatial affinity matrices (second-order statistics) from features
    and distills structural knowledge from teacher to student.
    """
    
    def __init__(
        self,
        layers: List[int] = [2, 4, 6, 8],
        layer_weights: List[float] = None,
        affinity_type: str = 'spatial',  # 'spatial' or 'channel'
        temperature: float = 1.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.layers = layers
        self.affinity_type = affinity_type
        self.temperature = temperature
        self.reduction = reduction
        
        # Normalize layer weights
        if layer_weights is None:
            self.layer_weights = [1.0] * len(layers)
        else:
            total = sum(layer_weights)
            self.layer_weights = [w / total for w in layer_weights]
    
    def compute_spatial_affinity(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial affinity matrix (pixel correlations).
        
        Args:
            features: [B, C, H, W]
        
        Returns:
            Affinity matrix [B, H*W, H*W]
        """
        b, c, h, w = features.shape
        
        # Downsample to prevent OOM for large spatial dimensions
        max_spatial = 32
        if h * w > max_spatial * max_spatial:
            features = F.adaptive_avg_pool2d(features, (max_spatial, max_spatial))
            b, c, h, w = features.shape
        
        # Flatten spatial dimensions: [B, C, H*W]
        features_flat = features.view(b, c, -1)
        
        # Compute Gram matrix (spatial correlations): [B, H*W, H*W]
        # A[i,j] = correlation between pixel i and pixel j across all channels
        affinity = torch.bmm(features_flat.transpose(1, 2), features_flat)
        
        # Normalize
        affinity = affinity / (c + 1e-8)
        
        return affinity
    
    def compute_channel_affinity(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute channel affinity matrix (channel correlations).
        
        Args:
            features: [B, C, H, W]
        
        Returns:
            Affinity matrix [B, C, C]
        """
        b, c, h, w = features.shape
        
        # Global average pooling: [B, C, 1, 1]
        features_pooled = F.adaptive_avg_pool2d(features, 1)
        
        # Flatten: [B, C, 1]
        features_flat = features_pooled.view(b, c, -1)
        
        # Compute Gram matrix (channel correlations): [B, C, C]
        affinity = torch.bmm(features_flat, features_flat.transpose(1, 2))
        
        # Normalize
        affinity = affinity / (h * w + 1e-8)
        
        return affinity
    
    def affinity_loss(
        self,
        student_affinity: torch.Tensor,
        teacher_affinity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss between student and teacher affinity matrices.
        
        Args:
            student_affinity: Student affinity matrix
            teacher_affinity: Teacher affinity matrix
        
        Returns:
            Affinity loss (Frobenius norm)
        """
        # Normalize by temperature for stability
        student_affinity = student_affinity / self.temperature
        teacher_affinity = teacher_affinity / self.temperature
        
        # Frobenius norm of difference
        diff = student_affinity - teacher_affinity
        
        if self.reduction == 'mean':
            loss = (diff ** 2).mean()
        elif self.reduction == 'sum':
            loss = (diff ** 2).sum()
        else:
            loss = diff ** 2
        
        return loss
    
    def forward(
        self,
        student_features: Dict[int, torch.Tensor],
        teacher_features: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute FAKD loss across multiple layers.
        
        Args:
            student_features: Dict mapping layer index to student feature tensor
            teacher_features: Dict mapping layer index to teacher feature tensor
        
        Returns:
            Combined FAKD loss
        """
        total_loss = None
        
        for layer_idx, weight in zip(self.layers, self.layer_weights):
            if layer_idx not in student_features or layer_idx not in teacher_features:
                continue
            
            student_feat = student_features[layer_idx]
            teacher_feat = teacher_features[layer_idx]
            
            # Resize teacher features to match student if sizes differ
            if student_feat.shape != teacher_feat.shape:
                teacher_feat = F.interpolate(
                    teacher_feat,
                    size=student_feat.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
            
            # Compute affinities
            if self.affinity_type == 'spatial':
                student_aff = self.compute_spatial_affinity(student_feat)
                teacher_aff = self.compute_spatial_affinity(teacher_feat)
            elif self.affinity_type == 'channel':
                student_aff = self.compute_channel_affinity(student_feat)
                teacher_aff = self.compute_channel_affinity(teacher_feat)
            else:
                raise ValueError(f"Unknown affinity type: {self.affinity_type}")
            
            # Compute loss for this layer
            layer_loss = self.affinity_loss(student_aff, teacher_aff)
            if total_loss is None:
                total_loss = weight * layer_loss
            else:
                total_loss += weight * layer_loss
        
        if total_loss is None:
            # No matching layers found - return zero tensor on first available device
            device = next(iter(student_features.values()), next(iter(teacher_features.values()), torch.tensor(0.0))).device
            return torch.tensor(0.0, device=device)
        
        return total_loss


class DirectionalFeatureAffinityLoss(nn.Module):
    """
    Direction-aware FAKD for Mamba-PAN.
    Computes affinity loss separately for each directional scan.
    """
    
    def __init__(
        self,
        directions: List[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.directions = directions or ['h', 'v', 'rh', 'rv']
        
        # Create FAKD loss for each direction
        self.fakd_losses = nn.ModuleDict({
            direction: FeatureAffinityLoss(**kwargs)
            for direction in self.directions
        })
    
    def forward(
        self,
        student_directions: Dict[str, Dict[int, torch.Tensor]],
        teacher_features: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute directional FAKD loss.
        
        Args:
            student_directions: Dict mapping direction to feature dict
            teacher_features: Teacher feature dict
        
        Returns:
            Combined directional loss
        """
        total_loss = 0.0
        
        for direction, fakd in self.fakd_losses.items():
            if direction in student_directions:
                direction_loss = fakd(student_directions[direction], teacher_features)
                total_loss += direction_loss
        
        return total_loss / len(self.directions)


class CrossDirectionConsistencyLoss(nn.Module):
    """
    Cross-direction consistency loss for Mamba-PAN.
    Ensures outputs from different directional scans are consistent.
    """
    
    def __init__(
        self,
        consistency_type: str = 'variance',
        weight: float = 0.1,
    ):
        super().__init__()
        self.consistency_type = consistency_type
        self.weight = weight
    
    def forward(self, direction_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute consistency loss across directions.
        
        Args:
            direction_outputs: Dict mapping direction to output tensor
        
        Returns:
            Consistency loss
        """
        outputs = list(direction_outputs.values())
        
        if len(outputs) < 2:
            device = outputs[0].device if outputs else None
            return torch.tensor(0.0, device=device) if device is not None else torch.tensor(0.0)
        
        if self.consistency_type == 'variance':
            # Stack outputs: [num_directions, B, C, H, W]
            stacked = torch.stack(outputs)
            
            # Compute variance across directions (should be low for consistency)
            variance = torch.var(stacked, dim=0).mean()
            
            return self.weight * variance
        
        elif self.consistency_type == 'cosine':
            # Compute pairwise cosine similarity (should be high)
            loss = 0.0
            count = 0
            
            for i in range(len(outputs)):
                for j in range(i + 1, len(outputs)):
                    # Flatten for cosine similarity
                    flat_i = outputs[i].flatten(1)
                    flat_j = outputs[j].flatten(1)
                    
                    # Cosine similarity: -1 to 1, want close to 1
                    sim = F.cosine_similarity(flat_i, flat_j, dim=1).mean()
                    
                    # Loss: 1 - similarity (want to minimize)
                    loss += (1 - sim)
                    count += 1
            
            return self.weight * (loss / count)
        
        else:
            raise ValueError(f"Unknown consistency type: {self.consistency_type}")
