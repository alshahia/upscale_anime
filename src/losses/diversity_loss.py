"""
Teacher diversity loss for multi-teacher knowledge distillation.
Encourages student to learn from teacher disagreements.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class TeacherDiversityLoss(nn.Module):
    """
    Loss that encourages learning from teacher disagreements.
    
    The student should capture the variance/diversity among teachers,
    not just the mean prediction. This helps the student learn
    more robust and generalizable representations.
    """
    
    def __init__(self, diversity_weight: float = 0.1):
        super().__init__()
        self.diversity_weight = diversity_weight
    
    def forward(
        self,
        student_output: torch.Tensor,
        teacher_outputs: List[torch.Tensor],
    ) -> dict:
        """
        Compute diversity loss.
        
        Args:
            student_output: Student prediction [B, C, H, W]
            teacher_outputs: List of teacher predictions [B, C, H, W] each
            
        Returns:
            Dict with loss components
        """
        # Stack teacher outputs: [num_teachers, B, C, H, W]
        teacher_stack = torch.stack(teacher_outputs)
        
        # Compute teacher disagreement (variance)
        teacher_variance = torch.var(teacher_stack, dim=0).mean()
        
        # Compute student-teacher agreement
        teacher_mean = teacher_stack.mean(dim=0)
        student_mean_error = F.l1_loss(student_output, teacher_mean)
        
        # The loss: student should capture variance, not just mean
        # Higher variance among teachers = more important to capture
        # We want student to have error proportional to teacher disagreement
        diversity_loss = -teacher_variance * torch.exp(-student_mean_error)
        
        return {
            'diversity': diversity_loss * self.diversity_weight,
            'teacher_variance': teacher_variance.detach(),
            'student_mean_error': student_mean_error.detach(),
        }


class TeacherDisagreementLoss(nn.Module):
    """
    Loss based on where teachers disagree the most.
    
    Puts more weight on regions where teachers have high variance,
    indicating these are "hard" regions worth learning.
    """
    
    def __init__(self, base_loss: nn.Module = None):
        super().__init__()
        self.base_loss = base_loss or nn.L1Loss(reduction='none')
    
    def forward(
        self,
        student_output: torch.Tensor,
        teacher_outputs: List[torch.Tensor],
        target: torch.Tensor,
    ) -> dict:
        """
        Compute disagreement-weighted loss.
        
        Args:
            student_output: Student prediction
            teacher_outputs: List of teacher predictions
            target: Ground truth
            
        Returns:
            Dict with losses
        """
        # Stack teachers: [N, B, C, H, W]
        teacher_stack = torch.stack(teacher_outputs)
        
        # Compute per-pixel teacher disagreement
        teacher_std = torch.std(teacher_stack, dim=0)  # [B, C, H, W]
        
        # Normalize to get weights
        disagreement_weights = teacher_std / (teacher_std.mean() + 1e-8)
        
        # Base loss (per-pixel)
        base_loss = self.base_loss(student_output, target)
        
        # Weight by disagreement
        weighted_loss = (base_loss * disagreement_weights).mean()
        
        return {
            'disagreement_weighted': weighted_loss,
            'base_loss': base_loss.mean(),
            'mean_disagreement': teacher_std.mean().detach(),
            'max_disagreement': teacher_std.max().detach(),
        }


class TeacherConsistencyLoss(nn.Module):
    """
    Loss that ensures student is consistent with best teacher.
    
    Identifies the best teacher for each sample and ensures
    student matches that teacher closely.
    """
    
    def __init__(self, base_loss: nn.Module = None):
        super().__init__()
        self.base_loss = base_loss or nn.L1Loss()
    
    def forward(
        self,
        student_output: torch.Tensor,
        teacher_outputs: List[torch.Tensor],
        target: torch.Tensor,
    ) -> dict:
        """
        Compute consistency loss with best teacher.
        
        Args:
            student_output: Student prediction
            teacher_outputs: List of teacher predictions
            target: Ground truth
            
        Returns:
            Dict with losses
        """
        # Compute each teacher's error
        teacher_errors = []
        for teacher_out in teacher_outputs:
            error = F.l1_loss(teacher_out, target, reduction='mean')
            teacher_errors.append(error)
        
        # Stack errors: [num_teachers, B]
        teacher_errors = torch.stack(teacher_errors)
        
        # Find best teacher per sample
        best_teacher_idx = torch.argmin(teacher_errors, dim=0)
        
        # Gather best teacher outputs
        # This is tricky with batches - for simplicity, use mean best
        mean_errors = teacher_errors.mean(dim=1)
        best_teacher_overall = torch.argmin(mean_errors)
        best_teacher_output = teacher_outputs[best_teacher_overall]
        
        # Loss: student should match best teacher
        consistency_loss = self.base_loss(student_output, best_teacher_output)
        
        return {
            'consistency': consistency_loss,
            'best_teacher_idx': best_teacher_overall.item(),
            'best_teacher_error': mean_errors[best_teacher_overall].detach(),
        }
