"""
Wavelet-based loss for Multi-Teacher Knowledge Distillation (MTKD)
Implements multi-scale wavelet decomposition loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def dwt_haar(x: torch.Tensor, levels: int = 1):
    """
    Discrete Wavelet Transform using Haar wavelets.
    
    Args:
        x: Input tensor [B, C, H, W]
        levels: Number of decomposition levels
    
    Returns:
        List of wavelet coefficients per level.
        Each level: (LL, (LH, HL, HH)) where:
        - LL: Low frequency (approximation)
        - LH: Horizontal high frequency
        - HL: Vertical high frequency
        - HH: Diagonal high frequency
    """
    coeffs = []
    current = x
    
    for _ in range(levels):
        b, c, h, w = current.shape
        
        # Pad if dimensions are odd
        if h % 2 == 1:
            current = F.pad(current, (0, 0, 0, 1))
        if w % 2 == 1:
            current = F.pad(current, (0, 1, 0, 0))
        
        # Unfold into 2x2 blocks
        # Reshape: [B, C, H/2, 2, W/2, 2] -> [B, C, H/2, W/2, 4]
        h2, w2 = current.shape[-2] // 2, current.shape[-1] // 2
        x_unfold = current.view(b, c, h2, 2, w2, 2).permute(0, 1, 2, 4, 3, 5)
        x_unfold = x_unfold.reshape(b, c, h2, w2, 4)
        
        # Haar wavelet filters
        # LL: (a + b + c + d) / 2
        # LH: (a + b - c - d) / 2 (horizontal edges)
        # HL: (a - b + c - d) / 2 (vertical edges)
        # HH: (a - b - c + d) / 2 (diagonal edges)
        
        a = x_unfold[..., 0]
        b = x_unfold[..., 1]
        c = x_unfold[..., 2]
        d = x_unfold[..., 3]
        
        ll = (a + b + c + d) / 2.0
        lh = (a + b - c - d) / 2.0
        hl = (a - b + c - d) / 2.0
        hh = (a - b - c + d) / 2.0
        
        coeffs.append((ll, (lh, hl, hh)))
        current = ll  # Continue decomposing the low-frequency component
    
    return coeffs


class WaveletLoss(nn.Module):
    """
    Multi-scale wavelet decomposition loss for MTKD.
    Compares wavelet coefficients between student and teacher at multiple levels.
    """
    
    def __init__(
        self,
        levels: int = 3,
        wavelet_type: str = 'haar',
        ll_weight: float = 1.0,
        lh_weight: float = 1.0,
        hl_weight: float = 1.0,
        hh_weight: float = 1.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.levels = levels
        self.wavelet_type = wavelet_type
        self.ll_weight = ll_weight
        self.lh_weight = lh_weight
        self.hl_weight = hl_weight
        self.hh_weight = hh_weight
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute wavelet loss.
        
        Args:
            pred: Student prediction [B, C, H, W]
            target: Teacher target [B, C, H, W]
        
        Returns:
            Wavelet loss value
        """
        pred_coeffs = dwt_haar(pred, self.levels)
        target_coeffs = dwt_haar(target, self.levels)
        
        loss = 0.0
        
        for level_idx, (pred_level, target_level) in enumerate(zip(pred_coeffs, target_coeffs)):
            pred_ll, (pred_lh, pred_hl, pred_hh) = pred_level
            target_ll, (target_lh, target_hl, target_hh) = target_level
            
            # Low-frequency loss (smooth regions)
            ll_loss = F.l1_loss(pred_ll, target_ll, reduction=self.reduction)
            
            # High-frequency losses (edges/details)
            lh_loss = F.l1_loss(pred_lh, target_lh, reduction=self.reduction)
            hl_loss = F.l1_loss(pred_hl, target_hl, reduction=self.reduction)
            hh_loss = F.l1_loss(pred_hh, target_hh, reduction=self.reduction)
            
            # Weighted combination
            level_loss = (
                self.ll_weight * ll_loss +
                self.lh_weight * lh_loss +
                self.hl_weight * hl_loss +
                self.hh_weight * hh_loss
            )
            
            # Higher levels (finer details) get less weight
            level_weight = 1.0 / (level_idx + 1)
            loss += level_weight * level_loss
        
        return loss / self.levels


class DirectionalWaveletLoss(nn.Module):
    """
    Direction-aware wavelet loss for Mamba-PAN.
    Applies wavelet loss separately for each directional scan output.
    """
    
    def __init__(
        self,
        levels: int = 3,
        directions: list = None,
        direction_weights: list = None,
        **kwargs,
    ):
        super().__init__()
        self.levels = levels
        self.directions = directions or ['h', 'v', 'rh', 'rv']
        self.direction_weights = direction_weights or [1.0] * len(self.directions)
        
        # Create wavelet loss for each direction
        self.wavelet_losses = nn.ModuleList([
            WaveletLoss(levels=levels, **kwargs)
            for _ in self.directions
        ])
    
    def forward(
        self,
        pred_directions: dict,  # {'h': tensor, 'v': tensor, ...}
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute directional wavelet loss.
        
        Args:
            pred_directions: Dict of predictions per direction
            target: Teacher target
        
        Returns:
            Combined directional loss
        """
        total_loss = 0.0
        
        for direction, weight, loss_fn in zip(
            self.directions, self.direction_weights, self.wavelet_losses
        ):
            if direction in pred_directions:
                direction_loss = loss_fn(pred_directions[direction], target)
                total_loss += weight * direction_loss
        
        return total_loss / sum(self.direction_weights)
