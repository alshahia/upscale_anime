"""
Gradient-based loss functions for edge preservation.
Useful for maintaining sharp edges in super-resolution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientLoss(nn.Module):
    """
    Loss based on gradient (edge) differences.
    Uses Sobel filters to compute gradients in x and y directions.
    
    Helps preserve edges and fine details in super-resolution.
    """
    
    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        
        # Sobel filters for gradient computation
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        
        self.loss_type = loss_type
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
    
    def compute_gradients(self, x: torch.Tensor) -> tuple:
        """
        Compute x and y gradients using Sobel filters.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            (grad_x, grad_y) each [B, C, H, W]
        """
        b, c, h, w = x.shape
        
        # Pad for Sobel filter
        x_pad = F.pad(x, (1, 1, 1, 1), mode='replicate')
        
        # Expand filters for all channels
        sobel_x = self.sobel_x.expand(c, 1, 3, 3)
        sobel_y = self.sobel_y.expand(c, 1, 3, 3)
        
        # Compute gradients
        grad_x = F.conv2d(x_pad, sobel_x, groups=c)
        grad_y = F.conv2d(x_pad, sobel_y, groups=c)
        
        return grad_x, grad_y
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient loss.
        
        Args:
            pred: Predictions [B, C, H, W]
            target: Targets [B, C, H, W]
            
        Returns:
            Gradient loss value
        """
        # Compute gradients
        pred_grad_x, pred_grad_y = self.compute_gradients(pred)
        target_grad_x, target_grad_y = self.compute_gradients(target)
        
        # Compute gradient magnitude
        pred_grad_mag = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2 + 1e-8)
        target_grad_mag = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2 + 1e-8)
        
        # Loss on gradient magnitude
        loss = self.loss_fn(pred_grad_mag, target_grad_mag)
        
        return loss


class LaplacianLoss(nn.Module):
    """
    Loss based on Laplacian (second derivative).
    Penalizes differences in curvature.
    """
    
    def __init__(self):
        super().__init__()
        
        # Laplacian filter
        self.register_buffer('laplacian', torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian loss."""
        b, c, h, w = pred.shape
        
        # Pad
        pred_pad = F.pad(pred, (1, 1, 1, 1), mode='replicate')
        target_pad = F.pad(target, (1, 1, 1, 1), mode='replicate')
        
        # Expand filter
        laplacian = self.laplacian.expand(c, 1, 3, 3)
        
        # Compute Laplacian
        pred_lap = F.conv2d(pred_pad, laplacian, groups=c)
        target_lap = F.conv2d(target_pad, laplacian, groups=c)
        
        # L1 loss
        loss = F.l1_loss(pred_lap, target_lap)
        
        return loss


class CombinedGradientLoss(nn.Module):
    """
    Combined gradient loss with multiple components.
    """
    
    def __init__(
        self,
        gradient_weight: float = 1.0,
        laplacian_weight: float = 0.5,
    ):
        super().__init__()
        
        self.gradient_loss = GradientLoss()
        self.laplacian_loss = LaplacianLoss()
        self.gradient_weight = gradient_weight
        self.laplacian_weight = laplacian_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Compute combined gradient loss.
        
        Returns:
            Dict with 'total', 'gradient', 'laplacian' losses
        """
        grad_loss = self.gradient_loss(pred, target)
        lap_loss = self.laplacian_loss(pred, target)
        
        total = self.gradient_weight * grad_loss + self.laplacian_weight * lap_loss
        
        return {
            'total': total,
            'gradient': grad_loss,
            'laplacian': lap_loss,
        }
