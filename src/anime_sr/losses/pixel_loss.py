"""
Pixel-level loss functions for Super-Resolution
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    """L1 (MAE) loss"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target, reduction=self.reduction)


class L2Loss(nn.Module):
    """L2 (MSE) loss"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target, reduction=self.reduction)


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss (variant of L1 with epsilon for stability).
    More robust to outliers than L1/L2.
    """
    
    def __init__(self, eps: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LaplacianPyramidLoss(nn.Module):
    """
    Laplacian pyramid loss for multi-scale pixel comparison.
    Captures details at different frequencies.
    """
    
    def __init__(self, levels: int = 3, reduction: str = 'mean'):
        super().__init__()
        self.levels = levels
        self.l1 = L1Loss(reduction=reduction)
    
    def _build_laplacian_pyramid(self, img: torch.Tensor, levels: int):
        """Build Laplacian pyramid"""
        pyramid = []
        current = img
        
        for _ in range(levels):
            # Downsample
            down = F.avg_pool2d(current, 2, stride=2)
            
            # Upsample back
            up = F.interpolate(down, size=current.shape[-2:], mode='bilinear', align_corners=False)
            
            # Laplacian level
            laplacian = current - up
            pyramid.append(laplacian)
            
            current = down
        
        # Add the final low-frequency residual
        pyramid.append(current)
        return pyramid
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_pyramid = self._build_laplacian_pyramid(pred, self.levels)
        target_pyramid = self._build_laplacian_pyramid(target, self.levels)
        
        loss = 0
        for p, t in zip(pred_pyramid, target_pyramid):
            loss += self.l1(p, t)
        
        return loss / (self.levels + 1)


class LossFactory:
    """Factory for creating loss functions from config"""
    
    @staticmethod
    def create(loss_type: str, **kwargs) -> nn.Module:
        """
        Create loss function by name.
        
        Args:
            loss_type: 'l1', 'l2', 'mse', 'charbonnier', 'laplacian'
            **kwargs: Additional arguments for loss constructor
        
        Returns:
            Loss module
        """
        loss_map = {
            'l1': L1Loss,
            'l2': L2Loss,
            'mse': L2Loss,
            'charbonnier': CharbonnierLoss,
            'laplacian': LaplacianPyramidLoss,
        }
        
        if loss_type.lower() not in loss_map:
            raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(loss_map.keys())}")
        
        return loss_map[loss_type.lower()](**kwargs)
