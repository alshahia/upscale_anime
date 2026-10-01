"""
Feature distillation for super-resolution.
Distills intermediate features from teacher to student model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional


class FeatureDistillationLoss(nn.Module):
    """
    Feature-level knowledge distillation loss.
    Matches intermediate feature maps between teacher and student.
    """
    
    def __init__(self, 
                 layer_indices: List[int] = None,
                 weights: List[float] = None,
                 loss_type: str = 'l2'):
        """
        Args:
            layer_indices: Which layer outputs to match (e.g., [2, 4, 6, 8])
            weights: Weight for each layer
            loss_type: 'l2', 'l1', or 'cosine'
        """
        super().__init__()
        
        self.layer_indices = layer_indices or [2, 4, 6, 8]
        
        if weights is None:
            self.weights = [1.0 / len(self.layer_indices)] * len(self.layer_indices)
        else:
            self.weights = weights
        
        self.loss_type = loss_type
        
        # Hooks for extracting features
        self.teacher_features = {}
        self.student_features = {}
        self.teacher_hooks = []
        self.student_hooks = []
    
    def register_hooks(self, teacher: nn.Module, student: nn.Module):
        """Register forward hooks to capture intermediate features."""
        self._remove_hooks()
        
        # Register teacher hooks
        if hasattr(teacher, 'blocks'):
            for idx in self.layer_indices:
                if idx < len(teacher.blocks):
                    hook = teacher.blocks[idx].register_forward_hook(
                        self._make_hook('teacher', idx)
                    )
                    self.teacher_hooks.append(hook)
        
        # Register student hooks
        if hasattr(student, 'blocks'):
            for idx in self.layer_indices:
                if idx < len(student.blocks):
                    hook = student.blocks[idx].register_forward_hook(
                        self._make_hook('student', idx)
                    )
                    self.student_hooks.append(hook)
    
    def _make_hook(self, model_type: str, layer_idx: int):
        """Create a hook function."""
        def hook(module, input, output):
            if model_type == 'teacher':
                self.teacher_features[layer_idx] = output
            else:
                self.student_features[layer_idx] = output
        return hook
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.teacher_hooks:
            hook.remove()
        for hook in self.student_hooks:
            hook.remove()
        self.teacher_hooks.clear()
        self.student_hooks.clear()
        self.teacher_features.clear()
        self.student_features.clear()
    
    def forward(self, student: nn.Module, teacher: nn.Module, 
                input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute feature distillation loss.
        
        Args:
            student: Student model
            teacher: Teacher model (pre-trained)
            input_tensor: Input LR image
        
        Returns:
            Feature distillation loss
        """
        # Register hooks if not already done
        if not self.teacher_hooks:
            self.register_hooks(teacher, student)
        
        # Clear previous features
        self.teacher_features.clear()
        self.student_features.clear()
        
        # Forward pass through both models
        with torch.no_grad():
            _ = teacher(input_tensor)
        _ = student(input_tensor)
        
        # Compute loss for each layer
        total_loss = 0.0
        
        for idx, weight in zip(self.layer_indices, self.weights):
            if idx in self.teacher_features and idx in self.student_features:
                teacher_feat = self.teacher_features[idx]
                student_feat = self.student_features[idx]
                
                # Handle size mismatch (e.g., different scales)
                if teacher_feat.shape != student_feat.shape:
                    # Adapt student features to teacher size
                    student_feat = F.interpolate(
                        student_feat,
                        size=teacher_feat.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Compute loss
                if self.loss_type == 'l2':
                    layer_loss = F.mse_loss(student_feat, teacher_feat)
                elif self.loss_type == 'l1':
                    layer_loss = F.l1_loss(student_feat, teacher_feat)
                elif self.loss_type == 'cosine':
                    layer_loss = 1 - F.cosine_similarity(
                        student_feat.flatten(1),
                        teacher_feat.flatten(1),
                        dim=1
                    ).mean()
                else:
                    raise ValueError(f"Unknown loss type: {self.loss_type}")
                
                total_loss += weight * layer_loss
        
        return total_loss
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        self._remove_hooks()


class MultiTeacherFeatureDistillation(nn.Module):
    """
    Feature distillation from multiple teachers.
    Combines features from multiple teacher models.
    """
    
    def __init__(self,
                 teachers: List[nn.Module],
                 teacher_weights: List[float] = None,
                 layer_indices: List[int] = None,
                 loss_weight: float = 0.1):
        """
        Args:
            teachers: List of teacher models
            teacher_weights: Weight for each teacher
            layer_indices: Which layers to distill
            loss_weight: Overall weight for this loss
        """
        super().__init__()
        
        self.teachers = teachers
        
        if teacher_weights is None:
            self.teacher_weights = [1.0 / len(teachers)] * len(teachers)
        else:
            self.teacher_weights = teacher_weights
        
        self.layer_indices = layer_indices or [2, 4, 6, 8]
        self.loss_weight = loss_weight
        
        # Create distillation loss for each teacher
        self.distillation_losses = nn.ModuleList([
            FeatureDistillationLoss(
                layer_indices=self.layer_indices,
                weights=None,
                loss_type='l2'
            )
            for _ in teachers
        ])
        
        # Freeze all teachers
        for teacher in self.teachers:
            for param in teacher.parameters():
                param.requires_grad = False
            teacher.eval()
    
    def forward(self, student: nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-teacher feature distillation loss.
        
        Args:
            student: Student model
            input_tensor: Input LR image
        
        Returns:
            Weighted feature distillation loss
        """
        total_loss = 0.0
        
        for teacher, weight, dist_loss in zip(
            self.teachers, self.teacher_weights, self.distillation_losses
        ):
            loss = dist_loss(student, teacher, input_tensor)
            total_loss += weight * loss
        
        return self.loss_weight * total_loss
