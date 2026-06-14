"""
Training Callbacks
EMA, checkpointing, logging, and learning rate scheduling
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Any, Callable
import json
import time


class Callback:
    """Base callback class"""
    
    def on_train_begin(self, logs: Dict = None):
        pass
    
    def on_train_end(self, logs: Dict = None):
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Dict = None):
        pass
    
    def on_epoch_end(self, epoch: int, logs: Dict = None):
        pass
    
    def on_batch_begin(self, batch: int, logs: Dict = None):
        pass
    
    def on_batch_end(self, batch: int, logs: Dict = None):
        pass


class EMACallback(Callback):
    """
    Exponential Moving Average callback.
    Maintains EMA version of model parameters.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.ema_model = self._create_ema_model(model)
        self.num_updates = 0
    
    def _create_ema_model(self, model: nn.Module) -> nn.Module:
        """Create EMA model copy"""
        import copy
        ema_model = copy.deepcopy(model)
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model
    
    def update(self):
        """Update EMA parameters"""
        self.num_updates += 1
        
        # Compute decay rate with warmup
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)
    
    def on_batch_end(self, batch: int, logs: Dict = None):
        self.update()
    
    def get_model(self) -> nn.Module:
        """Get EMA model"""
        return self.ema_model
    
    def apply_ema(self):
        """Apply EMA weights to original model (for evaluation)"""
        self.backup = {}
        for name, param in self.model.named_parameters():
            self.backup[name] = param.data.clone()
            param.data.copy_(self.ema_model.state_dict()[name])
    
    def restore_original(self):
        """Restore original model weights"""
        if hasattr(self, 'backup'):
            for name, param in self.model.named_parameters():
                param.data.copy_(self.backup[name])
            self.backup = {}


class CheckpointCallback(Callback):
    """
    Model checkpointing callback.
    Saves checkpoints at intervals and keeps best model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_dir: str = "checkpoints",
        save_interval: int = 10,
        keep_last_n: int = 5,
        monitor: str = "val_loss",
        mode: str = "min",
    ):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_interval = save_interval
        self.keep_last_n = keep_last_n
        self.monitor = monitor
        self.mode = mode
        
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.checkpoints = []
    
    def on_epoch_end(self, epoch: int, logs: Dict = None):
        if logs is None:
            logs = {}
        
        # Save regular checkpoint
        if epoch % self.save_interval == 0:
            self._save_checkpoint(epoch, logs, is_best=False)
        
        # Save best checkpoint
        current_value = logs.get(self.monitor)
        if current_value is not None:
            is_better = (current_value < self.best_value) if self.mode == "min" else (current_value > self.best_value)
            
            if is_better:
                self.best_value = current_value
                self._save_checkpoint(epoch, logs, is_best=True)
    
    def _save_checkpoint(self, epoch: int, logs: Dict, is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'logs': logs,
        }
        
        # Save checkpoint
        if is_best:
            path = self.checkpoint_dir / 'best.pth'
        else:
            path = self.checkpoint_dir / f'epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)
        
        if not is_best:
            self.checkpoints.append(path)
            
            # Cleanup old checkpoints
            if len(self.checkpoints) > self.keep_last_n:
                old_checkpoint = self.checkpoints.pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()


class TensorBoardCallback(Callback):
    """TensorBoard logging callback"""
    
    def __init__(self, log_dir: str = "logs", enabled: bool = True):
        self.enabled = enabled
        self.writer = None
        
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                print("Warning: TensorBoard not available. Install with: pip install tensorboard")
                self.enabled = False
        
        self.global_step = 0
    
    def on_batch_end(self, batch: int, logs: Dict = None):
        if not self.enabled or logs is None:
            return
        
        # Log metrics
        for name, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'train/{name}', value, self.global_step)
        
        self.global_step += 1
    
    def on_epoch_end(self, epoch: int, logs: Dict = None):
        if not self.enabled or logs is None:
            return
        
        # Log epoch metrics
        for name, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'epoch/{name}', value, epoch)
    
    def on_train_end(self, logs: Dict = None):
        if self.enabled and self.writer:
            self.writer.close()


class LRSchedulerCallback(Callback):
    """Learning rate scheduling callback"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, scheduler: Any):
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def on_epoch_end(self, epoch: int, logs: Dict = None):
        if self.scheduler:
            self.scheduler.step()
            
            # Log learning rate
            lr = self.optimizer.param_groups[0]['lr']
            if logs is not None:
                logs['lr'] = lr


class ProgressCallback(Callback):
    """Progress printing callback"""
    
    def __init__(self, print_interval: int = 1):
        self.print_interval = print_interval
        self.epoch_times = []
    
    def on_epoch_begin(self, epoch: int, logs: Dict = None):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch: int, logs: Dict = None):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        
        if epoch % self.print_interval == 0:
            avg_time = sum(self.epoch_times[-10:]) / min(len(self.epoch_times), 10)
            
            log_str = f"Epoch {epoch}"
            if logs:
                for key, value in logs.items():
                    if isinstance(value, float):
                        log_str += f" - {key}: {value:.4f}"
            
            log_str += f" - time: {epoch_time:.1f}s (avg: {avg_time:.1f}s)"
            print(log_str)


class EarlyStoppingCallback(Callback):
    """
    Early stopping callback with multiple trigger conditions.
    
    Stops training when:
    - No improvement for `patience` epochs
    - Loss change becomes negligible (< min_delta)
    - Loss diverges (increases for divergence_patience epochs)
    - Training is stable but not improving (plateau detection)
    
    Args:
        monitor: Metric to monitor (e.g., "val_loss", "train_loss", "psnr")
        mode: "min" for loss (lower is better), "max" for metrics (higher is better)
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to count as improvement
        divergence_patience: Epochs of increasing loss before stopping
        restore_best_weights: Whether to load best checkpoint when stopped
        min_epochs: Minimum epochs before early stopping can trigger
        plateau_slope_threshold: Slope threshold for plateau detection
        verbose: Whether to print stopping information
    """
    
    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 15,
        min_delta: float = 0.0001,
        divergence_patience: int = 5,
        restore_best_weights: bool = True,
        min_epochs: int = 50,
        plateau_slope_threshold: float = 1e-6,
        verbose: bool = True,
    ):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.divergence_patience = divergence_patience
        self.restore_best_weights = restore_best_weights
        self.min_epochs = min_epochs
        self.plateau_slope_threshold = plateau_slope_threshold
        self.verbose = verbose
        
        # State tracking
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.best_epoch = 0
        self.counter = 0
        self.divergence_counter = 0
        self.stopped_epoch = 0
        self.should_stop = False
        self.best_checkpoint_path = None
        
        # Loss history for trend analysis
        self.values = []
        self.epochs = []
    
    def on_train_begin(self, logs: Dict = None):
        """Reset state at training start."""
        self.best_value = float('inf') if self.mode == "min" else float('-inf')
        self.best_epoch = 0
        self.counter = 0
        self.divergence_counter = 0
        self.stopped_epoch = 0
        self.should_stop = False
        self.best_checkpoint_path = None
        self.values = []
        self.epochs = []
    
    def on_epoch_end(self, epoch: int, logs: Dict = None):
        """Check for early stopping conditions."""
        if logs is None:
            logs = {}
        
        current_value = logs.get(self.monitor)
        if current_value is None:
            return
        
        # Track history
        self.values.append(current_value)
        self.epochs.append(epoch)
        
        # Check for improvement
        if self._is_better(current_value, self.best_value):
            improvement = abs(self.best_value - current_value)
            if improvement > self.min_delta:
                self.best_value = current_value
                self.best_epoch = epoch
                self.counter = 0
                self.divergence_counter = 0
                
                # Track best checkpoint path if available in logs
                if 'checkpoint_path' in logs:
                    self.best_checkpoint_path = logs['checkpoint_path']
            else:
                self.counter += 1
        else:
            self.counter += 1
            # Check for divergence (only for loss metrics in min mode)
            if self.mode == "min" and len(self.values) >= 2:
                if current_value > self.values[-2]:
                    self.divergence_counter += 1
                else:
                    self.divergence_counter = 0
        
        # Check stopping conditions (only after min_epochs)
        if epoch >= self.min_epochs:
            stop_reason = self._should_stop(epoch, current_value)
            if stop_reason:
                self.should_stop = True
                self.stopped_epoch = epoch
                if self.verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch}: {stop_reason}")
                    print(f"  Best {self.monitor}: {self.best_value:.6f} at epoch {self.best_epoch}")
        
        # Add convergence info to logs
        logs['epochs_without_improvement'] = self.counter
        logs['best_epoch'] = self.best_epoch
        logs['best_value'] = self.best_value
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current value is better than best."""
        if self.mode == "min":
            return current < best
        else:
            return current > best
    
    def _should_stop(self, epoch: int, current_value: float) -> Optional[str]:
        """Check if training should stop. Returns stop reason or None."""
        # Check patience (no improvement)
        if self.counter >= self.patience:
            return f"No improvement in {self.monitor} for {self.patience} epochs"
        
        # Check divergence
        if self.divergence_counter >= self.divergence_patience:
            return f"Loss diverging for {self.divergence_patience} consecutive epochs"
        
        # Check plateau (flat loss curve)
        if len(self.values) >= 5:
            recent_values = self.values[-5:]
            slope = self._compute_slope(recent_values)
            if abs(slope) < self.plateau_slope_threshold:
                return f"Loss curve plateaued (slope={slope:.2e})"
        
        return None
    
    def _compute_slope(self, values: list) -> float:
        """Compute linear regression slope."""
        if len(values) < 2:
            return 0.0
        try:
            import numpy as np
            x = np.arange(len(values))
            slope, _, _, _, _ = np.polyfit(x, values, 1, full=True)
            return float(slope[0]) if not np.isnan(slope[0]) else 0.0
        except ImportError:
            # Simple finite difference
            return (values[-1] - values[0]) / (len(values) - 1)
    
    def on_train_end(self, logs: Dict = None):
        """Handle training end - restore best weights if needed."""
        if self.should_stop and self.verbose:
            print(f"Training stopped early at epoch {self.stopped_epoch}")
            print(f"Restoring best weights from epoch {self.best_epoch}")
        
        # Add early stopping info to logs
        if logs is not None:
            logs['early_stopped'] = self.should_stop
            logs['stopped_epoch'] = self.stopped_epoch if self.should_stop else None
            logs['best_epoch'] = self.best_epoch
            logs['best_value'] = self.best_value
    
    def get_status(self) -> Dict:
        """Get current early stopping status."""
        return {
            'should_stop': self.should_stop,
            'stopped_epoch': self.stopped_epoch,
            'best_epoch': self.best_epoch,
            'best_value': self.best_value,
            'counter': self.counter,
            'divergence_counter': self.divergence_counter,
            'patience': self.patience,
        }
    
    def state_dict(self) -> Dict:
        """Get state for checkpointing."""
        return {
            'best_value': self.best_value,
            'best_epoch': self.best_epoch,
            'counter': self.counter,
            'divergence_counter': self.divergence_counter,
            'stopped_epoch': self.stopped_epoch,
            'should_stop': self.should_stop,
            'values': self.values.copy(),
            'epochs': self.epochs.copy(),
        }
    
    def load_state_dict(self, state: Dict):
        """Load state from checkpoint."""
        self.best_value = state.get('best_value', float('inf') if self.mode == "min" else float('-inf'))
        self.best_epoch = state.get('best_epoch', 0)
        self.counter = state.get('counter', 0)
        self.divergence_counter = state.get('divergence_counter', 0)
        self.stopped_epoch = state.get('stopped_epoch', 0)
        self.should_stop = state.get('should_stop', False)
        self.values = state.get('values', [])
        self.epochs = state.get('epochs', [])


class LRMonitorCallback(Callback):
    """
    Monitor and log learning rate changes.
    Tracks LR reduction events and correlates with training progress.
    """
    
    def __init__(self, optimizer, log_interval: int = 1):
        self.optimizer = optimizer
        self.log_interval = log_interval
        self.lr_history = []
        self.last_lr = None
        self.reduction_count = 0
    
    def on_epoch_begin(self, epoch: int, logs: Dict = None):
        """Record LR at epoch start."""
        lr = self.optimizer.param_groups[0]['lr']
        
        # Detect LR reduction
        if self.last_lr is not None and lr < self.last_lr * 0.99:
            self.reduction_count += 1
            if logs is not None:
                logs['lr_reduced'] = True
                logs['lr_reduction_count'] = self.reduction_count
                logs['lr_before'] = self.last_lr
                logs['lr_after'] = lr
        
        self.last_lr = lr
        self.lr_history.append((epoch, lr))
        
        if logs is not None:
            logs['lr'] = lr
    
    def get_lr_history(self) -> list:
        """Get recorded LR history."""
        return self.lr_history.copy()
    
    def get_reduction_count(self) -> int:
        """Get number of LR reductions."""
        return self.reduction_count


class StageMonitorCallback(Callback):
    """
    Monitor stage progress and emit transition signals.
    
    Integrates with StageTransitionController to:
    - Track stage metrics
    - Detect stage completion
    - Trigger stage transitions
    - Log stage events to TensorBoard
    """
    
    def __init__(
        self,
        stage_controller,
        current_stage: int = 1,
        log_interval: int = 1,
    ):
        self.stage_controller = stage_controller
        self.current_stage = current_stage
        self.log_interval = log_interval
        self.epoch_count = 0
        self.stage_complete = False
    
    def on_epoch_end(self, epoch: int, logs: Dict = None):
        """Check for stage transition at epoch end."""
        if logs is None:
            logs = {}
        
        self.epoch_count += 1
        
        # Update stage progress
        self.stage_controller.update_stage_progress(
            stage=self.current_stage,
            epoch=epoch,
            metrics=logs,
        )
        
        # Check if should advance stage
        if self.stage_controller.should_advance_stage(
            current_stage=self.current_stage,
            epoch=epoch,
            metrics=logs,
        ):
            logs['stage_complete'] = True
            logs['advance_to_stage'] = self.current_stage + 1
            self.stage_complete = True
            
            print(f"\n*** Stage {self.current_stage} complete! Ready to advance to Stage {self.current_stage + 1} ***\n")
        
        # Add stage info to logs
        logs['current_stage'] = self.current_stage
        logs['stage_epochs'] = self.epoch_count
    
    def on_train_end(self, logs: Dict = None):
        """Handle training end - mark stage complete."""
        if logs is None:
            logs = {}
        
        # Mark current stage complete
        logs['stage_completed'] = self.current_stage
        logs['total_stage_epochs'] = self.epoch_count
        
        # Print final status
        self.stage_controller.print_status()
    
    def set_stage(self, stage: int):
        """Set current stage (for stage transitions)."""
        self.current_stage = stage
        self.epoch_count = 0
        self.stage_complete = False
    
    def is_stage_complete(self) -> bool:
        """Check if current stage is complete."""
        return self.stage_complete


class ProgressiveCropCallback(Callback):
    """
    Progressive crop size callback.
    Increases crop size during training for better generalization.
    """
    
    def __init__(self, sizes: list = None, epochs_per_size: int = 25):
        """
        Args:
            sizes: List of crop sizes to progress through (e.g., [64, 96, 128, 160])
            epochs_per_size: Number of epochs per crop size
        """
        self.sizes = sizes or [64, 96, 128, 160]
        self.epochs_per_size = epochs_per_size
        self.current_size_idx = 0
    
    def on_train_begin(self, logs: Dict = None):
        """Initialize crop size."""
        self.current_size_idx = 0
    
    def on_epoch_begin(self, epoch: int, logs: Dict = None):
        """Update crop size based on current epoch."""
        size_idx = min(epoch // self.epochs_per_size, len(self.sizes) - 1)
        
        if size_idx != self.current_size_idx:
            self.current_size_idx = size_idx
            new_size = self.sizes[self.current_size_idx]
            print(f"\n[ProgressiveCrop] Epoch {epoch}: Increasing crop size to {new_size}")
        
        # Store current crop size in logs
        if logs is not None:
            logs['crop_size'] = self.sizes[self.current_size_idx]
    
    def get_current_size(self) -> int:
        """Get current crop size."""
        return self.sizes[min(self.current_size_idx, len(self.sizes) - 1)]


class LayerFreezeCallback(Callback):
    """
    Callback for progressive layer unfreezing during transfer learning.
    """
    
    def __init__(self, model: nn.Module, freeze_schedule: Dict):
        """
        Args:
            model: The model to freeze/unfreeze
            freeze_schedule: Dictionary with freeze/unfreeze schedule
        """
        self.model = model
        self.freeze_schedule = freeze_schedule
        self.frozen_blocks = set()
    
    def on_epoch_begin(self, epoch: int, logs: Dict = None):
        """Apply freeze/unfreeze schedule."""
        if 'unfreeze_schedule' in self.freeze_schedule:
            for entry in self.freeze_schedule['unfreeze_schedule']:
                if epoch == entry.get('epoch', -1):
                    if 'unfreeze_blocks' in entry:
                        self._unfreeze_blocks(entry['unfreeze_blocks'])
                    if 'unfreeze_all' in entry and entry['unfreeze_all']:
                        self._unfreeze_all()
                    
                    lr_mult = entry.get('lr_multiplier', 1.0)
                    print(f"[LayerFreeze] Epoch {epoch}: Unfroze layers, LR multiplier = {lr_mult}")
                    
                    if logs is not None:
                        logs['lr_multiplier'] = lr_mult
    
    def _freeze_initial_blocks(self, num_blocks: int):
        """Freeze first N blocks."""
        if hasattr(self.model, 'blocks'):
            for i, block in enumerate(self.model.blocks[:num_blocks]):
                for param in block.parameters():
                    param.requires_grad = False
                self.frozen_blocks.add(i)
                print(f"[LayerFreeze] Frozen block {i}")
    
    def _unfreeze_blocks(self, count: int):
        """Unfreeze next N blocks (from beginning, lowest indices first)."""
        if hasattr(self.model, 'blocks'):
            # Get blocks that are currently frozen, sort ascending (lowest first)
            to_unfreeze = sorted([b for b in self.frozen_blocks])[:count]
            for i in to_unfreeze:
                for param in self.model.blocks[i].parameters():
                    param.requires_grad = True
                self.frozen_blocks.discard(i)
                print(f"[LayerFreeze] Unfrozen block {i}")
    
    def _unfreeze_all(self):
        """Unfreeze all layers."""
        for param in self.model.parameters():
            param.requires_grad = True
        self.frozen_blocks.clear()
        print("[LayerFreeze] All layers unfrozen")


class CallbackList:
    """Manager for multiple callbacks"""
    
    def __init__(self, callbacks: Optional[list] = None):
        self.callbacks = callbacks or []
    
    def add(self, callback: Callback):
        self.callbacks.append(callback)
    
    def on_train_begin(self, logs: Dict = None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)
    
    def on_train_end(self, logs: Dict = None):
        for callback in self.callbacks:
            callback.on_train_end(logs)
    
    def on_epoch_begin(self, epoch: int, logs: Dict = None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Dict = None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
    
    def on_batch_begin(self, batch: int, logs: Dict = None):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Dict = None):
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)
    
    def should_stop(self) -> bool:
        """Check if any callback is requesting training stop (e.g., early stopping)."""
        for callback in self.callbacks:
            if isinstance(callback, EarlyStoppingCallback) and callback.should_stop:
                return True
        return False
    
    def get_early_stopping_status(self) -> Optional[Dict]:
        """Get status from early stopping callback if present."""
        for callback in self.callbacks:
            if isinstance(callback, EarlyStoppingCallback):
                return callback.get_status()
        return None
