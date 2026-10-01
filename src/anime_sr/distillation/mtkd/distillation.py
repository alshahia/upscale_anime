"""
MTKD Stage 2: Wavelet-based Distillation Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from anime_sr.losses.wavelet_loss import WaveletLoss, DirectionalWaveletLoss


class WaveletDistillationLoss(nn.Module):
    """
    MTKD Stage 2 distillation loss.
    Combines:
    1. L1 loss to ground truth
    2. Wavelet-based distillation loss to aggregated teacher output
    """
    
    def __init__(
        self,
        pixel_weight: float = 1.0,
        wavelet_weight: float = 1.0,
        wavelet_levels: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.pixel_weight = pixel_weight
        self.wavelet_weight = wavelet_weight
        
        self.l1_loss = nn.L1Loss()
        self.wavelet_loss = WaveletLoss(levels=wavelet_levels, **kwargs)
    
    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> tuple:
        """
        Compute MTKD distillation loss.
        
        Args:
            student_output: Student prediction [B, C, H, W]
            teacher_output: Aggregated teacher output [B, C, H, W]
            ground_truth: Ground truth HR [B, C, H, W]
        
        Returns:
            (total_loss, loss_dict)
        """
        # L1 loss to ground truth
        l1 = self.l1_loss(student_output, ground_truth)
        
        # Wavelet distillation loss to teacher
        wavelet = self.wavelet_loss(student_output, teacher_output)
        
        # Total loss
        total = self.pixel_weight * l1 + self.wavelet_weight * wavelet
        
        loss_dict = {
            'total': total.item(),
            'l1': l1.item(),
            'wavelet': wavelet.item(),
        }
        
        return total, loss_dict


class FullMTKDLoss(nn.Module):
    """
    Full MTKD loss combining:
    - L1 pixel loss
    - Wavelet distillation loss  
    - FAKD feature affinity loss (optional, added externally)
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        wavelet_weight: float = 1.0,
        wavelet_levels: int = 3,
    ):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.wavelet_weight = wavelet_weight
        
        self.l1_loss = nn.L1Loss()
        self.wavelet_loss = WaveletLoss(levels=wavelet_levels)
    
    def forward(
        self,
        student_pred: torch.Tensor,
        teacher_pred: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> tuple:
        """
        Compute combined MTKD loss.
        
        Returns:
            (total_loss, loss_dict)
        """
        l1 = self.l1_loss(student_pred, ground_truth)
        wavelet = self.wavelet_loss(student_pred, teacher_pred)
        
        total = self.l1_weight * l1 + self.wavelet_weight * wavelet
        
        # NaN/Inf protection
        if torch.isnan(total) or torch.isinf(total):
            return torch.tensor(0.0, device=total.device), {
                'total': 0.0,
                'l1': 0.0,
                'wavelet_distill': 0.0,
            }
        
        return total, {
            'total': total.item(),
            'l1': l1.item(),
            'wavelet_distill': wavelet.item(),
        }
