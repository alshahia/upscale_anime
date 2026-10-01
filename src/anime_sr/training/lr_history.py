"""
Learning rate history tracking and analysis.
Tracks LR changes and correlates with training metrics.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class LRHistory:
    """
    Track learning rate history and correlate with training metrics.
    
    Features:
    - Records LR at each epoch
    - Tracks LR reduction events
    - Correlates LR with loss improvements
    - Recommends optimal LR for future runs
    """
    
    def __init__(self):
        self.epochs: List[int] = []
        self.lr_values: List[float] = []
        self.loss_values: List[float] = []
        self.reduction_events: List[Dict] = []
        self.best_loss = float('inf')
        self.best_lr = None
    
    def record(self, epoch: int, lr: float, loss: float):
        """
        Record LR and loss for an epoch.
        
        Args:
            epoch: Epoch number
            lr: Learning rate
            loss: Loss value
        """
        self.epochs.append(epoch)
        self.lr_values.append(lr)
        self.loss_values.append(loss)
        
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_lr = lr
    
    def record_reduction(self, epoch: int, old_lr: float, new_lr: float, 
                         reason: str = "plateau"):
        """
        Record a learning rate reduction event.
        
        Args:
            epoch: Epoch when reduction occurred
            old_lr: Learning rate before reduction
            new_lr: Learning rate after reduction
            reason: Reason for reduction (plateau, warmup_end, etc.)
        """
        self.reduction_events.append({
            'epoch': epoch,
            'old_lr': old_lr,
            'new_lr': new_lr,
            'factor': new_lr / old_lr if old_lr > 0 else 1.0,
            'reason': reason,
        })
    
    def get_lr_at_epoch(self, epoch: int) -> Optional[float]:
        """Get learning rate at a specific epoch."""
        if epoch < len(self.lr_values):
            return self.lr_values[epoch]
        return None
    
    def get_reduction_events(self) -> List[Dict]:
        """Get all LR reduction events."""
        return self.reduction_events.copy()
    
    def get_best_lr(self) -> Optional[float]:
        """Get the learning rate that achieved best loss."""
        return self.best_lr
    
    def get_lr_range(self) -> tuple:
        """Get min and max LR values."""
        if not self.lr_values:
            return (0.0, 0.0)
        return (min(self.lr_values), max(self.lr_values))
    
    def get_stats(self) -> Dict:
        """Get LR history statistics."""
        if not self.lr_values:
            return {}
        
        return {
            'total_epochs': len(self.epochs),
            'num_reductions': len(self.reduction_events),
            'lr_range': self.get_lr_range(),
            'final_lr': self.lr_values[-1] if self.lr_values else 0,
            'best_lr': self.best_lr,
            'best_loss': self.best_loss,
        }
    
    def recommend_lr(self, mode: str = "best") -> float:
        """
        Recommend learning rate for next run.
        
        Args:
            mode: Recommendation mode
                - "best": Use LR that achieved best loss
                - "final": Use final LR
                - "median": Use median LR
                
        Returns:
            Recommended learning rate
        """
        if not self.lr_values:
            return 1e-4  # Default
        
        if mode == "best" and self.best_lr is not None:
            return self.best_lr
        elif mode == "final":
            return self.lr_values[-1]
        elif mode == "median":
            return float(np.median(self.lr_values))
        else:
            return self.lr_values[-1]
    
    def get_lr_efficiency(self) -> Dict:
        """
        Analyze LR efficiency - how well each LR improved loss.
        
        Returns:
            Dictionary with efficiency metrics
        """
        if len(self.lr_values) < 2:
            return {}
        
        # Compute loss improvement per LR
        lr_improvements = {}
        for i in range(1, len(self.loss_values)):
            lr = self.lr_values[i-1]
            improvement = self.loss_values[i-1] - self.loss_values[i]
            if lr not in lr_improvements:
                lr_improvements[lr] = []
            lr_improvements[lr].append(improvement)
        
        # Average improvement per LR
        avg_improvements = {
            lr: np.mean(improvements) 
            for lr, improvements in lr_improvements.items()
        }
        
        # Find most efficient LR
        if avg_improvements:
            best_lr = max(avg_improvements, key=avg_improvements.get)
            return {
                'best_lr': best_lr,
                'best_improvement': avg_improvements[best_lr],
                'per_lr_improvements': avg_improvements,
            }
        return {}
    
    def save(self, path: Path):
        """Save LR history to file."""
        data = {
            'epochs': self.epochs,
            'lr_values': self.lr_values,
            'loss_values': self.loss_values,
            'reduction_events': self.reduction_events,
            'best_loss': self.best_loss,
            'best_lr': self.best_lr,
            'stats': self.get_stats(),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: Path):
        """Load LR history from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.epochs = data.get('epochs', [])
        self.lr_values = data.get('lr_values', [])
        self.loss_values = data.get('loss_values', [])
        self.reduction_events = data.get('reduction_events', [])
        self.best_loss = data.get('best_loss', float('inf'))
        self.best_lr = data.get('best_lr')


class LRHistoryTracker:
    """
    High-level tracker that integrates with training loop.
    Automatically records LR and loss after each epoch.
    """
    
    def __init__(self, log_dir: str = "logs", enabled: bool = True):
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = LRHistory()
        self.last_lr = None
    
    def on_epoch_end(self, epoch: int, optimizer, loss: float):
        """Record LR and loss at epoch end."""
        if not self.enabled:
            return
        
        # Get current LR from optimizer
        lr = optimizer.param_groups[0]['lr']
        
        # Check for reduction
        if self.last_lr is not None and lr < self.last_lr:
            self.history.record_reduction(
                epoch=epoch,
                old_lr=self.last_lr,
                new_lr=lr,
                reason="plateau"
            )
        
        # Record values
        self.history.record(epoch, lr, loss)
        self.last_lr = lr
    
    def save(self, filename: str = "lr_history.json"):
        """Save history to file."""
        if self.enabled:
            self.history.save(self.log_dir / filename)
    
    def get_summary(self) -> str:
        """Get text summary of LR history."""
        if not self.enabled:
            return "LR history tracking disabled"
        
        stats = self.history.get_stats()
        lines = [
            "LR History Summary:",
            f"  Total epochs: {stats.get('total_epochs', 0)}",
            f"  LR reductions: {stats.get('num_reductions', 0)}",
            f"  LR range: {stats.get('lr_range', (0, 0))}",
            f"  Final LR: {stats.get('final_lr', 0):.2e}",
            f"  Best LR: {stats.get('best_lr', 0):.2e}" if stats.get('best_lr') else "  Best LR: N/A",
        ]
        
        # Add reduction events
        events = self.history.get_reduction_events()
        if events:
            lines.append("  Reduction events:")
            for event in events:
                lines.append(
                    f"    Epoch {event['epoch']}: "
                    f"{event['old_lr']:.2e} -> {event['new_lr']:.2e} "
                    f"(factor: {event['factor']:.2f})"
                )
        
        return "\n".join(lines)
