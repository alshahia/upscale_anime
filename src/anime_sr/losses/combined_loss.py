"""
Combined loss module for Stage 1 training.
Integrates pixel loss, wavelet loss, gradient loss, and diversity loss.
"""
import torch
import torch.nn as nn
from typing import Dict, List

from anime_sr.losses.wavelet_loss import WaveletLoss
from anime_sr.losses.gradient_loss import GradientLoss
from anime_sr.losses.diversity_loss import TeacherDiversityLoss


class Stage1CombinedLoss(nn.Module):
    """
    Combined loss for Stage 1 (Knowledge Aggregation) training.
    
    Components:
    - L1 pixel loss (0.4): Basic pixel-wise accuracy
    - Wavelet loss (0.3): Frequency domain fidelity
    - Gradient loss (0.2): Edge preservation
    - Diversity loss (0.1): Learn from teacher disagreements
    
    All weights are configurable.
    """
    
    def __init__(
        self,
        l1_weight: float = 0.4,
        wavelet_weight: float = 0.3,
        gradient_weight: float = 0.2,
        diversity_weight: float = 0.1,
        wavelet_levels: int = 3,
    ):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.wavelet_weight = wavelet_weight
        self.gradient_weight = gradient_weight
        self.diversity_weight = diversity_weight
        
        # Initialize loss components
        self.l1_loss = nn.L1Loss()
        self.wavelet_loss = WaveletLoss(levels=wavelet_levels)
        self.gradient_loss = GradientLoss()
        self.diversity_loss = TeacherDiversityLoss(diversity_weight=diversity_weight)
        
        # Normalize weights to sum to 1.0
        total = l1_weight + wavelet_weight + gradient_weight + diversity_weight
        if total > 0:
            self.l1_weight /= total
            self.wavelet_weight /= total
            self.gradient_weight /= total
            self.diversity_weight /= total
    
    def forward(
        self,
        student_output: torch.Tensor,
        target: torch.Tensor,
        teacher_outputs: List[torch.Tensor] = None,
    ) -> tuple:
        """
        Compute combined loss.
        
        Args:
            student_output: Model prediction [B, C, H, W]
            target: Ground truth [B, C, H, W]
            teacher_outputs: List of teacher predictions (for diversity loss)
            
        Returns:
            (total_loss, loss_dict)
        """
        losses = {}
        
        # L1 pixel loss
        l1 = self.l1_loss(student_output, target)
        losses['l1'] = l1
        
        # Wavelet loss
        wavelet = self.wavelet_loss(student_output, target)
        losses['wavelet'] = wavelet
        
        # Gradient loss
        gradient = self.gradient_loss(student_output, target)
        losses['gradient'] = gradient
        
        # Diversity loss (if teachers provided)
        diversity = torch.tensor(0.0, device=student_output.device)
        if teacher_outputs is not None and len(teacher_outputs) > 0 and self.diversity_weight > 0:
            diversity_dict = self.diversity_loss(student_output, teacher_outputs)
            diversity = diversity_dict['diversity']
            losses['teacher_variance'] = diversity_dict['teacher_variance']
        losses['diversity'] = diversity
        
        # Compute weighted total
        total = (
            self.l1_weight * l1 +
            self.wavelet_weight * wavelet +
            self.gradient_weight * gradient +
            self.diversity_weight * diversity
        )
        losses['total'] = total
        
        # Also return individual weights for monitoring
        losses['weights'] = {
            'l1': self.l1_weight,
            'wavelet': self.wavelet_weight,
            'gradient': self.gradient_weight,
            'diversity': self.diversity_weight,
        }
        
        return total, losses


class AdaptiveCombinedLoss(nn.Module):
    """
    Combined loss with adaptive weighting based on gradient magnitudes.
    
    Automatically balances loss components based on their relative
    gradient magnitudes to prevent one loss from dominating.
    """
    
    def __init__(
        self,
        initial_weights: Dict[str, float] = None,
        wavelet_levels: int = 3,
        adapt_every: int = 100,
    ):
        super().__init__()
        
        self.weights = initial_weights or {
            'l1': 0.4,
            'wavelet': 0.3,
            'gradient': 0.2,
            'diversity': 0.1,
        }
        
        # Initialize loss components
        self.l1_loss = nn.L1Loss()
        self.wavelet_loss = WaveletLoss(levels=wavelet_levels)
        self.gradient_loss = GradientLoss()
        self.diversity_loss = TeacherDiversityLoss()
        
        # Adaptive weighting parameters
        self.adapt_every = adapt_every
        self.step_count = 0
        
        # Running averages of gradient norms
        self.register_buffer('grad_norms', torch.zeros(4))
        self.register_buffer('weight_buffer', torch.ones(4) / 4)
    
    def forward(
        self,
        student_output: torch.Tensor,
        target: torch.Tensor,
        teacher_outputs: List[torch.Tensor] = None,
    ) -> tuple:
        """Compute adaptive combined loss."""
        
        # Compute individual losses (detached for weight computation)
        with torch.no_grad():
            l1_val = self.l1_loss(student_output, target).item()
            wavelet_val = self.wavelet_loss(student_output, target).item()
            gradient_val = self.gradient_loss(student_output, target).item()
            
            if teacher_outputs:
                diversity_dict = self.diversity_loss(student_output, teacher_outputs)
                diversity_val = diversity_dict['diversity'].item()
            else:
                diversity_val = 0.0
            
            values = torch.tensor([l1_val, wavelet_val, gradient_val, diversity_val])
            
            # Normalize to get adaptive weights (inverse of loss magnitude)
            # Lower loss = higher weight (focus on what's working)
            inv_values = 1.0 / (values + 1e-8)
            adaptive_weights = inv_values / inv_values.sum()
            
            # Smooth with running average
            self.weight_buffer = 0.9 * self.weight_buffer + 0.1 * adaptive_weights
        
        # Compute weighted loss with adaptive weights
        l1 = self.l1_loss(student_output, target)
        wavelet = self.wavelet_loss(student_output, target)
        gradient = self.gradient_loss(student_output, target)
        
        diversity = torch.tensor(0.0, device=student_output.device)
        if teacher_outputs:
            diversity_dict = self.diversity_loss(student_output, teacher_outputs)
            diversity = diversity_dict['diversity']
        
        total = (
            self.weight_buffer[0] * l1 +
            self.weight_buffer[1] * wavelet +
            self.weight_buffer[2] * gradient +
            self.weight_buffer[3] * diversity
        )
        
        losses = {
            'total': total,
            'l1': l1,
            'wavelet': wavelet,
            'gradient': gradient,
            'diversity': diversity,
            'adaptive_weights': {
                'l1': self.weight_buffer[0].item(),
                'wavelet': self.weight_buffer[1].item(),
                'gradient': self.weight_buffer[2].item(),
                'diversity': self.weight_buffer[3].item(),
            }
        }
        
        return total, losses
