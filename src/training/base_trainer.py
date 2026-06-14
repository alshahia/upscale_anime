"""
Base trainer class for all Super-Resolution models
"""
import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch.amp import GradScaler  # PyTorch 2.0+
except ImportError:
    from torch.cuda.amp import GradScaler  # PyTorch 1.x
import os
from pathlib import Path
from typing import Dict, Optional, Any
from tqdm import tqdm

# Optional tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


class BaseTrainer:
    """
    Base trainer with common functionality:
    - Training loop management
    - Mixed precision (AMP)
    - Gradient accumulation
    - EMA (Exponential Moving Average)
    - Checkpointing
    - TensorBoard logging
    - Callback system (early stopping, LR scheduling)
    """
    
    def __init__(self, config: Dict, model: nn.Module):
        self.config = config
        self.model = model
        
        # Device setup
        device_str = config.get('training', {}).get('device', 'cuda') if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_str)
        self.model = self.model.to(self.device)
        
        # Training config
        train_cfg = config.get('training', {})
        self.epochs = train_cfg.get('epochs', 1000)
        
        # Resolve batch_size with fallback notification
        data_cfg = config.get('data', {})
        training_batch = train_cfg.get('batch_size')
        data_batch = data_cfg.get('batch_size')
        
        if training_batch is not None:
            self.batch_size = training_batch
        elif data_batch is not None:
            self.batch_size = data_batch
            print(f"[Config] Using data.batch_size={data_batch} for training")
        else:
            self.batch_size = 8
            print(f"[WARNING] batch_size not found in config, using default=8")
            print(f"  To fix: Add 'training.batch_size' or 'data.batch_size' to your config file")
        
        self.accumulation_steps = train_cfg.get('accumulation_steps', 1)
        
        # Optimization
        self.lr = train_cfg.get('lr', 1e-4)
        self.optimizer_name = train_cfg.get('optimizer', 'adam')
        self.weight_decay = train_cfg.get('weight_decay', 0)
        
        # Mixed precision
        self.use_amp = train_cfg.get('mixed_precision', True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient checkpointing
        self.use_gradient_checkpointing = train_cfg.get('gradient_checkpointing', False)
        if self.use_gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            else:
                print(f"  Warning: {type(self.model).__name__} does not support gradient checkpointing")
        
        # EMA
        self.use_ema = train_cfg.get('use_ema', True)
        self.ema_decay = train_cfg.get('ema_decay', 0.999)
        self.ema_model = None
        if self.use_ema:
            self.ema_model = self._create_ema_model()
        
        # SWA (Stochastic Weight Averaging)
        self.use_swa = train_cfg.get('use_swa', False)
        self.swa_model = None
        self.swa_scheduler = None
        self.swa_start_epoch = train_cfg.get('swa_start_epoch', int(self.epochs * 0.75))
        if self.use_swa:
            self._setup_swa()
        
        # Setup logging (moved to separate method)
        self._setup_logging(config, train_cfg)
    
    def _setup_swa(self):
        """Setup SWA model and scheduler"""
        try:
            from torch.optim.swa_utils import AveragedModel, SWALR
            self.swa_model = AveragedModel(self.model)
            # Create SWA scheduler - will be initialized once optimizer is available
            self.swa_scheduler_class = SWALR
            print(f"  SWA enabled (starts at epoch {self.swa_start_epoch})")
        except ImportError:
            print("  Warning: SWA requires PyTorch 1.12+, disabling SWA")
            self.use_swa = False
    
    def _update_swa(self):
        """Update SWA model (call once per epoch after swa_start_epoch)"""
        if not self.use_swa or self.swa_model is None:
            return
        
        if self.current_epoch >= self.swa_start_epoch:
            self.swa_model.update_parameters(self.model)
            if self.current_epoch == self.swa_start_epoch:
                print(f"  SWA averaging started at epoch {self.current_epoch}")
    
    def _setup_logging(self, config, train_cfg):
        """Setup logging and checkpointing"""
        # Logging
        self.log_interval = train_cfg.get('log_interval', 100)
        self.val_interval = train_cfg.get('val_interval', 10)
        self.save_interval = train_cfg.get('save_interval', 50)
        
        # TensorBoard
        self.use_tensorboard = train_cfg.get('use_tensorboard', True) and TENSORBOARD_AVAILABLE
        self.writer = None
        if self.use_tensorboard:
            log_dir = Path(config.get('paths', {}).get('log_dir', 'logs'))
            log_dir.mkdir(parents=True, exist_ok=True)
            if SummaryWriter is not None:
                self.writer = SummaryWriter(log_dir=log_dir)
            else:
                print("Warning: TensorBoard not available. Install with: pip install tensorboard")
                self.use_tensorboard = False
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('paths', {}).get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = config.get('checkpoint', {}).get('keep_last_n', 5)
        self.keep_best = config.get('checkpoint', {}).get('keep_best', True)
        
        self.best_loss = float('inf')
        self.current_epoch = 0
        self.global_step = 0
        
        # Callbacks system
        self.callbacks = None
        self._setup_callbacks()
    
    def _setup_callbacks(self):
        """Setup training callbacks including early stopping and LR monitoring."""
        from training.callbacks import CallbackList, EarlyStoppingCallback, LRMonitorCallback
        
        callback_list = []
        
        # Early stopping callback
        early_stop_cfg = self.config.get('training', {}).get('early_stopping', {})
        if early_stop_cfg.get('enabled', False):
            early_stopping = EarlyStoppingCallback(
                monitor=early_stop_cfg.get('monitor', 'val_loss'),
                mode=early_stop_cfg.get('mode', 'min'),
                patience=early_stop_cfg.get('patience', 15),
                min_delta=early_stop_cfg.get('min_delta', 0.0001),
                divergence_patience=early_stop_cfg.get('divergence_patience', 5),
                restore_best_weights=early_stop_cfg.get('restore_best_weights', True),
                min_epochs=early_stop_cfg.get('min_epochs', 50),
                plateau_slope_threshold=early_stop_cfg.get('plateau_slope_threshold', 1e-6),
                verbose=True,
            )
            callback_list.append(early_stopping)
            print(f"  Early stopping enabled (patience={early_stopping.patience}, monitor={early_stopping.monitor})")
        
        self.callbacks = CallbackList(callback_list)
        
        # LR history tracker
        self.lr_history_tracker = None
        adaptive_cfg = self.config.get('training', {}).get('adaptive_lr', {})
        if adaptive_cfg.get('enabled', False):
            from training.lr_history import LRHistoryTracker
            log_dir = self.config.get('paths', {}).get('log_dir', 'logs')
            self.lr_history_tracker = LRHistoryTracker(log_dir=log_dir, enabled=True)
    
    def _create_ema_model(self):
        """Create EMA version of model"""
        import copy
        # Use deepcopy for all models to preserve architecture and weights
        ema_model = copy.deepcopy(self.model).to(self.device)
        ema_model.requires_grad_(False)
        return ema_model
    
    def _update_ema(self):
        """Update EMA model parameters"""
        if self.ema_model is None:
            return
        
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(model_param.data, alpha=1 - self.ema_decay)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config"""
        if self.optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.99)
            )
        elif self.optimizer_name.lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        
        return optimizer
    
    def _create_scheduler(self, optimizer: optim.Optimizer, stage: str = 'stage1'):
        """
        Create learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            stage: 'stage1' or 'stage2' for stage-specific config
        """
        from training.schedulers import create_scheduler
        
        # Try to use new scheduler system with warmup support
        scheduler = create_scheduler(optimizer, self.config, stage)
        if scheduler is not None:
            return scheduler
        
        # Fallback to legacy scheduler system
        scheduler_name = self.config.get('training', {}).get('scheduler', 'step')
        
        if scheduler_name == 'step':
            step_size = self.config.get('training', {}).get('step_size', 200000)
            gamma = self.config.get('training', {}).get('gamma', 0.5)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == 'multistep':
            milestones = self.config.get('training', {}).get('milestones', [200000, 400000, 600000])
            gamma = self.config.get('training', {}).get('gamma', 0.5)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        elif scheduler_name == 'cosine':
            T_max = self.config.get('training', {}).get('T_max', self.epochs)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self, dataloader, optimizer, loss_fn) -> Dict[str, float]:
        """
        Train for one epoch.
        To be implemented by subclass.
        """
        raise NotImplementedError
    
    def validate(self, dataloader) -> Dict[str, float]:
        """
        Validate model.
        To be implemented by subclass.
        """
        raise NotImplementedError
    
    def save_checkpoint(self, epoch: int, optimizer: optim.Optimizer, is_best: bool = False, metrics: Dict = None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
        }
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        if self.ema_model is not None:
            checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()
        
        if self.swa_model is not None:
            checkpoint['swa_model_state_dict'] = self.swa_model.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        if hasattr(self, 'scheduler') and self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save callback states (including early stopping and all other stateful callbacks)
        if self.callbacks is not None:
            callback_states = {}
            for callback in self.callbacks.callbacks:
                if hasattr(callback, 'state_dict'):
                    callback_name = type(callback).__name__
                    callback_states[callback_name] = callback.state_dict()
            if callback_states:
                checkpoint['callback_states'] = callback_states

        # Save latest
        latest_path = self.checkpoint_dir / 'latest.pth'
        try:
            torch.save(checkpoint, latest_path)
        except Exception as e:
            print(f"Warning: Failed to save latest checkpoint: {e}")
        
        # Save epoch checkpoint
        epoch_path = self.checkpoint_dir / f'epoch_{epoch}.pth'
        try:
            torch.save(checkpoint, epoch_path)
        except Exception as e:
            print(f"Warning: Failed to save epoch checkpoint: {e}")
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            try:
                torch.save(checkpoint, best_path)
                print(f"Saved best checkpoint with loss {self.best_loss:.4f}")
            except Exception as e:
                print(f"Warning: Failed to save best checkpoint: {e}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Keep only N most recent checkpoints"""
        checkpoints = sorted(self.checkpoint_dir.glob('epoch_*.pth'))
        if len(checkpoints) > self.keep_last_n:
            for ckpt in checkpoints[:-self.keep_last_n]:
                try:
                    ckpt.unlink()
                except (OSError, PermissionError) as e:
                    import warnings
                    warnings.warn(f"Could not delete old checkpoint {ckpt}: {e}")
    
    def load_checkpoint(self, checkpoint_path: str, optimizer=None) -> int:
        """Load checkpoint and return starting epoch"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except Exception as e:
            # Checkpoint may contain custom objects - require explicit opt-in for security
            security_cfg = self.config.get('security') or {}
            allow_pickle = security_cfg.get('allow_pickle_checkpoint', False)
            if not allow_pickle:
                raise RuntimeError(
                    f"Checkpoint contains unsafe content that requires pickle loading. "
                    f"Set 'security.allow_pickle_checkpoint: true' in config to allow. "
                    f"Only use with trusted checkpoints! Error: {e}"
                )
            import warnings
            warnings.warn(
                "Loading checkpoint with pickle fallback - only use with trusted sources!",
                UserWarning,
                stacklevel=2
            )
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_loss = checkpoint.get('best_loss', float('inf'))

        # Restore optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"  Restored optimizer state")
            except Exception as e:
                print(f"  Warning: Could not restore optimizer state: {e}")

        if 'ema_model_state_dict' in checkpoint and self.ema_model is not None:
            self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])

        if 'swa_model_state_dict' in checkpoint and self.swa_model is not None:
            try:
                self.swa_model.load_state_dict(checkpoint['swa_model_state_dict'])
                print(f"  Restored SWA model state")
            except Exception as e:
                print(f"  Warning: Could not restore SWA state: {e}")

        # Restore callback states (new generic format and backward compatibility)
        if self.callbacks is not None:
            restored_callbacks = []
            
            # Try new format first (all callback states in dict)
            if 'callback_states' in checkpoint:
                for callback in self.callbacks.callbacks:
                    callback_name = type(callback).__name__
                    if callback_name in checkpoint['callback_states']:
                        if hasattr(callback, 'load_state_dict'):
                            try:
                                callback.load_state_dict(checkpoint['callback_states'][callback_name])
                                restored_callbacks.append(callback_name)
                            except Exception as e:
                                print(f"  Warning: Could not restore {callback_name} state: {e}")
            
            # Backward compatibility: old format with 'early_stopping_state' key
            if 'early_stopping_state' in checkpoint:
                from training.callbacks import EarlyStoppingCallback
                for callback in self.callbacks.callbacks:
                    if isinstance(callback, EarlyStoppingCallback):
                        callback.load_state_dict(checkpoint['early_stopping_state'])
                        if 'EarlyStoppingCallback' not in restored_callbacks:
                            restored_callbacks.append('EarlyStoppingCallback')
            
            if restored_callbacks:
                print(f"  Restored callback states: {restored_callbacks}")

        # Store checkpoint for deferred scheduler loading (scheduler created after model initialization)
        self._pending_checkpoint = checkpoint

        # Try to load scheduler state immediately if scheduler exists
        self._load_scheduler_state_if_available()

        start_epoch = checkpoint.get('epoch', 0) + 1
        self.current_epoch = start_epoch

        print(f"Loaded checkpoint from epoch {start_epoch - 1}")
        return start_epoch

    def _load_scheduler_state_if_available(self):
        """Load scheduler state if both checkpoint and scheduler are available"""
        if hasattr(self, '_pending_checkpoint') and hasattr(self, 'scheduler') and self.scheduler is not None:
            checkpoint = self._pending_checkpoint
            if 'scheduler_state_dict' in checkpoint:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print(f"  Restored scheduler state (last_epoch: {self.scheduler.last_epoch})")
                except Exception as e:
                    print(f"  Warning: Could not restore scheduler state: {e}")
            # Clear pending checkpoint after attempting scheduler load
            delattr(self, '_pending_checkpoint')

    def set_scheduler(self, scheduler):
        """Set scheduler and attempt to load its state from pending checkpoint"""
        self.scheduler = scheduler
        self._load_scheduler_state_if_available()
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ''):
        """Log metrics to TensorBoard"""
        if self.writer is None:
            return
        
        for name, value in metrics.items():
            self.writer.add_scalar(f'{prefix}/{name}', value, step)
    
    def train(self, train_loader, val_loader=None):
        """
        Main training loop.
        To be implemented by subclass.
        """
        raise NotImplementedError
