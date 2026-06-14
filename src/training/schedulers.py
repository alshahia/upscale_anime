"""
Custom learning rate schedulers for training.
Includes warmup + cosine annealing, multi-step with warmup, etc.
"""
import torch
from torch.optim.lr_scheduler import _LRScheduler
import math


class WarmupCosineScheduler(_LRScheduler):
    """
    Cosine annealing scheduler with linear warmup.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs for linear warmup
        total_epochs: Total number of training epochs
        min_lr: Minimum learning rate after cosine annealing
        last_epoch: Last epoch index (for resuming)
    """
    
    def __init__(
        self,
        optimizer,
        warmup_epochs: int = 10,
        total_epochs: int = 100,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = float(min_lr)  # Ensure float type
        self.base_lrs = [float(group['lr']) for group in optimizer.param_groups]  # Ensure float type
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [float(base_lr) * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            remaining_epochs = self.total_epochs - self.warmup_epochs
            if remaining_epochs <= 0:
                progress = 1.0  # Already at end of training
            else:
                progress = (self.last_epoch - self.warmup_epochs) / remaining_epochs
            progress = min(1.0, progress)
            
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (float(base_lr) - self.min_lr) * cosine_decay 
                    for base_lr in self.base_lrs]


class WarmupMultiStepScheduler(_LRScheduler):
    """
    Multi-step LR scheduler with linear warmup.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        milestones: List of epoch indices to reduce LR
        gamma: LR decay factor at each milestone
        min_lr: Minimum learning rate
    """
    
    def __init__(
        self,
        optimizer,
        warmup_epochs: int = 10,
        milestones: list = None,
        gamma: float = 0.5,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.milestones = milestones or [50, 100, 150]
        self.gamma = gamma
        self.min_lr = float(min_lr)  # Ensure float type
        self.base_lrs = [float(group['lr']) for group in optimizer.param_groups]  # Ensure float type
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute learning rate."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [float(base_lr) * alpha for base_lr in self.base_lrs]
        else:
            # Multi-step decay
            decay_factor = self.gamma ** sum(1 for m in self.milestones if m <= self.last_epoch)
            return [max(self.min_lr, float(base_lr) * decay_factor) for base_lr in self.base_lrs]


class WarmupConstantScheduler(_LRScheduler):
    """
    Constant LR after warmup (useful for fine-tuning).
    """
    
    def __init__(
        self,
        optimizer,
        warmup_epochs: int = 10,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.base_lrs = [float(group['lr']) for group in optimizer.param_groups]  # Ensure float type
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute learning rate."""
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [float(base_lr) * alpha for base_lr in self.base_lrs]
        else:
            return [float(base_lr) for base_lr in self.base_lrs]


class WarmupCosinePlateauScheduler(_LRScheduler):
    """
    Cosine annealing scheduler with warmup and plateau detection.
    
    Combines:
    1. Linear warmup
    2. Cosine annealing until plateau detected
    3. Step reduction on plateau
    4. Return to cosine after cooldown
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs for linear warmup
        total_epochs: Total number of training epochs
        plateau_patience: Epochs to wait before reducing LR on plateau
        reduction_factor: Factor to reduce LR by (0.5 = halve)
        min_lr: Minimum learning rate
        max_reductions: Maximum number of LR reductions
        cooldown: Epochs to wait after reduction before resuming
    """
    
    def __init__(
        self,
        optimizer,
        warmup_epochs: int = 10,
        total_epochs: int = 100,
        plateau_patience: int = 8,
        reduction_factor: float = 0.5,
        min_lr: float = 1e-7,
        max_reductions: int = 3,
        cooldown: int = 3,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.plateau_patience = plateau_patience
        self.reduction_factor = reduction_factor
        self.min_lr = float(min_lr)
        self.max_reductions = max_reductions
        self.cooldown = cooldown
        self.base_lrs = [float(group['lr']) for group in optimizer.param_groups]
        
        # Plateau detection state
        self.best_loss = float('inf')
        self.plateau_counter = 0
        self.num_reductions = 0
        self.cooldown_counter = 0
        self.current_lr_scale = 1.0
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute learning rate for current epoch."""
        # Warmup phase
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [max(self.min_lr, float(base_lr) * alpha) for base_lr in self.base_lrs]
        
        # Check if in cooldown after reduction
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        
        # Cosine annealing phase (adjusted by current scale)
        remaining_epochs = self.total_epochs - self.warmup_epochs
        if remaining_epochs <= 0:
            progress = 1.0
        else:
            progress = (self.last_epoch - self.warmup_epochs) / remaining_epochs
        progress = min(1.0, progress)
        
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        
        # Apply current scale from plateau reductions
        return [
            max(self.min_lr, float(base_lr) * self.current_lr_scale * cosine_decay)
            for base_lr in self.base_lrs
        ]
    
    def step(self, loss=None, epoch=None):
        """Step the scheduler with optional loss for plateau detection."""
        if loss is not None:
            self._update_plateau_state(loss)
        
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
    
    def _update_plateau_state(self, loss: float):
        """Update plateau detection state and apply LR reduction if needed."""
        # Check for improvement
        if loss < self.best_loss:
            self.best_loss = loss
            self.plateau_counter = 0
        else:
            self.plateau_counter += 1
        
        # Check if should reduce LR
        if (self.plateau_counter >= self.plateau_patience and 
            self.cooldown_counter == 0 and 
            self.num_reductions < self.max_reductions):
            
            # Reduce learning rate
            self.current_lr_scale *= self.reduction_factor
            self.num_reductions += 1
            self.plateau_counter = 0
            self.cooldown_counter = self.cooldown
            
            print(f"  LR reduced to {self.current_lr_scale:.3f}x due to plateau "
                  f"(reduction {self.num_reductions}/{self.max_reductions})")
    
    def state_dict(self):
        """Get state for checkpointing."""
        return {
            'best_loss': self.best_loss,
            'plateau_counter': self.plateau_counter,
            'num_reductions': self.num_reductions,
            'cooldown_counter': self.cooldown_counter,
            'current_lr_scale': self.current_lr_scale,
            'last_epoch': self.last_epoch,
        }
    
    def load_state_dict(self, state_dict):
        """Load state from checkpoint."""
        self.best_loss = state_dict.get('best_loss', float('inf'))
        self.plateau_counter = state_dict.get('plateau_counter', 0)
        self.num_reductions = state_dict.get('num_reductions', 0)
        self.cooldown_counter = state_dict.get('cooldown_counter', 0)
        self.current_lr_scale = state_dict.get('current_lr_scale', 1.0)
        self.last_epoch = state_dict.get('last_epoch', -1)


class PlateauLRScheduler:
    """
    Wrapper around ReduceLROnPlateau with warmup support.
    
    Args:
        optimizer: PyTorch optimizer
        mode: 'min' or 'max'
        factor: LR reduction factor
        patience: Epochs to wait before reducing
        threshold: Minimum change to count as improvement
        cooldown: Epochs after reduction before resuming
        min_lr: Minimum learning rate
        warmup_epochs: Number of warmup epochs
    """
    
    def __init__(
        self,
        optimizer,
        mode: str = 'min',
        factor: float = 0.5,
        patience: int = 8,
        threshold: float = 0.0001,
        cooldown: int = 3,
        min_lr: float = 1e-8,
        warmup_epochs: int = 5,
    ):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
        
        # Create ReduceLROnPlateau (will be used after warmup)
        self.plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            cooldown=cooldown,
            min_lr=min_lr,
            verbose=False,
        )
    
    def step(self, loss=None, epoch=None):
        """Step the scheduler."""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        # Warmup phase
        if self.current_epoch < self.warmup_epochs:
            alpha = self.current_epoch / max(1, self.warmup_epochs)
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * alpha
        elif loss is not None:
            # Use ReduceLROnPlateau
            self.plateau_scheduler.step(loss)
        else:
            # Just increment internal counter
            self.plateau_scheduler.step()
    
    def state_dict(self):
        """Get state for checkpointing."""
        return {
            'plateau_scheduler': self.plateau_scheduler.state_dict(),
            'current_epoch': self.current_epoch,
            'base_lrs': self.base_lrs,
        }
    
    def load_state_dict(self, state_dict):
        """Load state from checkpoint."""
        self.plateau_scheduler.load_state_dict(state_dict['plateau_scheduler'])
        self.current_epoch = state_dict.get('current_epoch', 0)
        self.base_lrs = state_dict.get('base_lrs', self.base_lrs)


def create_scheduler(optimizer, config: dict, stage: str = 'stage1'):
    """
    Factory function to create scheduler based on config.
    
    Args:
        optimizer: PyTorch optimizer
        config: Training configuration dict
        stage: 'stage1' or 'stage2'
        
    Returns:
        Scheduler instance
    """
    stage_cfg = config.get('training', {}).get(stage, {})
    adaptive_cfg = config.get('training', {}).get('adaptive_lr', {})
    
    scheduler_name = stage_cfg.get('scheduler', 'cosine')
    warmup_epochs = stage_cfg.get('warmup_epochs', 10)
    epochs = stage_cfg.get('epochs', 100)
    min_lr = stage_cfg.get('min_lr', 1e-7)
    
    # Check if adaptive LR is enabled for this stage
    use_adaptive = adaptive_cfg.get('enabled', False)
    adaptive_scheduler = adaptive_cfg.get('scheduler', 'plateau')
    
    # Override with adaptive scheduler if enabled
    if use_adaptive and adaptive_scheduler == 'plateau':
        plateau_cfg = adaptive_cfg.get('plateau_config', {})
        scheduler = PlateauLRScheduler(
            optimizer,
            mode='min',
            factor=plateau_cfg.get('factor', 0.5),
            patience=plateau_cfg.get('patience', 8),
            threshold=plateau_cfg.get('threshold', 0.0001),
            cooldown=plateau_cfg.get('cooldown', 3),
            min_lr=plateau_cfg.get('min_lr', 1e-8),
            warmup_epochs=warmup_epochs,
        )
        print(f"  Using PlateauLRScheduler (patience={plateau_cfg.get('patience', 8)}, "
              f"factor={plateau_cfg.get('factor', 0.5)})")
        return scheduler
    
    if use_adaptive and adaptive_scheduler == 'warmup_cosine_plateau':
        wc_plateau_cfg = adaptive_cfg.get('warmup_cosine_plateau_config', {})
        scheduler = WarmupCosinePlateauScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=epochs,
            plateau_patience=wc_plateau_cfg.get('plateau_patience', 8),
            reduction_factor=wc_plateau_cfg.get('reduction_factor', 0.5),
            min_lr=min_lr,
            max_reductions=wc_plateau_cfg.get('max_reductions', 3),
            cooldown=wc_plateau_cfg.get('cooldown', 3),
        )
        print(f"  Using WarmupCosinePlateauScheduler (plateau_patience={wc_plateau_cfg.get('plateau_patience', 8)})")
        return scheduler
    
    # Standard schedulers (non-adaptive)
    if scheduler_name == 'cosine':
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=epochs,
            min_lr=min_lr,
        )
    elif scheduler_name == 'multistep':
        milestones = stage_cfg.get('milestones', [epochs // 3, epochs * 2 // 3])
        gamma = stage_cfg.get('gamma', 0.5)
        scheduler = WarmupMultiStepScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            milestones=milestones,
            gamma=gamma,
            min_lr=min_lr,
        )
    elif scheduler_name == 'constant':
        scheduler = WarmupConstantScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
        )
    else:
        # Fallback to base trainer's scheduler
        return None
    
    return scheduler
