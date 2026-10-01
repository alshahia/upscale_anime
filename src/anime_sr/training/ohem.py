"""
Online Hard Example Mining (OHEM) for training.
Selects hardest examples in each batch for backpropagation.
"""
import torch
import torch.nn as nn


class OHEMLoss(nn.Module):
    """
    Wrapper that applies OHEM to any loss function.
    
    Args:
        loss_fn: Base loss function (e.g., nn.L1Loss())
        ratio: Fraction of hard examples to keep (0.0-1.0)
        reduction: How to reduce the loss ('mean' or 'sum')
    """
    
    def __init__(
        self,
        loss_fn: nn.Module,
        ratio: float = 0.7,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.ratio = ratio
        self.reduction = reduction
        
        # Store statistics for monitoring
        self.stats = {
            'total_samples': 0,
            'selected_samples': 0,
            'threshold': 0.0,
        }
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute OHEM loss.
        
        Args:
            pred: Predictions [B, C, H, W] or [B, ...]
            target: Targets [B, C, H, W] or [B, ...]
            
        Returns:
            Loss value (scalar)
        """
        # Compute per-sample losses
        batch_size = pred.shape[0]
        
        # Compute loss without reduction to get per-sample values
        if hasattr(self.loss_fn, 'reduction'):
            original_reduction = self.loss_fn.reduction
            self.loss_fn.reduction = 'none'
        
        per_sample_loss = self.loss_fn(pred, target)
        
        if hasattr(self.loss_fn, 'reduction'):
            self.loss_fn.reduction = original_reduction
        
        # Flatten per-sample loss if needed
        if per_sample_loss.dim() > 1:
            per_sample_loss = per_sample_loss.view(batch_size, -1).mean(dim=1)
        
        # Determine threshold for hard examples
        num_hard = max(1, int(batch_size * self.ratio))
        threshold = torch.topk(per_sample_loss, num_hard, largest=True)[0][-1]
        
        # Create mask for hard examples
        hard_mask = per_sample_loss >= threshold
        
        # Compute loss only on hard examples
        hard_loss = per_sample_loss[hard_mask]
        
        # Reduce
        if self.reduction == 'mean':
            loss = hard_loss.mean()
        elif self.reduction == 'sum':
            loss = hard_loss.sum()
        else:
            loss = hard_loss
        
        # Update stats
        self.stats['total_samples'] = batch_size
        self.stats['selected_samples'] = hard_mask.sum().item()
        self.stats['threshold'] = threshold.item()
        
        return loss
    
    def get_stats(self) -> dict:
        """Get OHEM statistics from last forward pass."""
        return self.stats.copy()


class OHEMSampler:
    """
    Sampler that selects hard examples based on loss values.
    Can be used to re-sample batches or weight examples.
    """
    
    def __init__(self, ratio: float = 0.7):
        self.ratio = ratio
        self.hard_indices = []
        self.easy_indices = []
    
    def select_hard_examples(
        self,
        losses: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select indices of hard examples.
        
        Args:
            losses: Per-example losses [N]
            
        Returns:
            Indices of hard examples
        """
        num_samples = len(losses)
        num_hard = max(1, int(num_samples * self.ratio))
        
        # Get top-k hardest examples
        hard_indices = torch.topk(losses, num_hard, largest=True)[1]
        
        # Store for analysis
        all_indices = set(range(num_samples))
        hard_set = set(hard_indices.cpu().numpy())
        self.hard_indices = list(hard_set)
        self.easy_indices = list(all_indices - hard_set)
        
        return hard_indices
    
    def get_hard_ratio(self) -> float:
        """Get ratio of hard examples selected."""
        total = len(self.hard_indices) + len(self.easy_indices)
        if total == 0:
            return 0.0
        return len(self.hard_indices) / total


def compute_ohem_loss(
    loss_fn: nn.Module,
    pred: torch.Tensor,
    target: torch.Tensor,
    ratio: float = 0.7,
) -> tuple:
    """
    Compute OHEM loss and return additional info.
    
    Args:
        loss_fn: Base loss function
        pred: Predictions
        target: Targets
        ratio: Hard example ratio
        
    Returns:
        (loss, info_dict)
    """
    batch_size = pred.shape[0]
    
    # Compute per-sample losses
    original_reduction = getattr(loss_fn, 'reduction', 'mean')
    if hasattr(loss_fn, 'reduction'):
        loss_fn.reduction = 'none'
    
    per_sample_loss = loss_fn(pred, target)
    
    if hasattr(loss_fn, 'reduction'):
        loss_fn.reduction = original_reduction
    
    # Flatten
    if per_sample_loss.dim() > 1:
        per_sample_loss = per_sample_loss.view(batch_size, -1).mean(dim=1)
    
    # Select hard examples
    num_hard = max(1, int(batch_size * ratio))
    hard_losses, hard_indices = torch.topk(per_sample_loss, num_hard, largest=True)
    
    # Compute mean of hard losses
    loss = hard_losses.mean()
    
    info = {
        'threshold': hard_losses[-1].item(),
        'max_loss': hard_losses[0].item(),
        'min_hard_loss': hard_losses[-1].item(),
        'num_hard': num_hard,
        'hard_indices': hard_indices.cpu().numpy(),
    }
    
    return loss, info
