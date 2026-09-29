"""
Ensemble Trainer - Distill from ensemble teacher to tiny student
"""
import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch.amp import autocast  # PyTorch 2.0+
except ImportError:
    from torch.cuda.amp import autocast as _autocast  # PyTorch 1.x
    
    # Wrapper for PyTorch 1.x compatibility (ignores device argument)
    def autocast(device=None, enabled=True):
        return _autocast(enabled=enabled)
from typing import Dict, Optional
from tqdm import tqdm

from training.base_trainer import BaseTrainer
from models.ensemble import EnsembleTeacher, TinyStudent, create_ensemble_teacher, create_tiny_student
from distillation.fakd.affinity_loss import FeatureAffinityLoss
from losses import L1Loss


class EnsembleTrainer(BaseTrainer):
    """
    Trainer for ensemble distillation.
    
    Process:
    1. Load frozen Model A and B
    2. Create ensemble teacher
    3. Train tiny student with FAKD from ensemble
    
    No MTKD needed since we have direct ensemble output.
    """
    
    def __init__(
        self,
        config: Dict,
        ensemble_teacher: EnsembleTeacher,
        tiny_student: Optional[TinyStudent] = None,
    ):
        # Create student if not provided
        if tiny_student is None:
            tiny_student = create_tiny_student(scale=config.get('model', {}).get('scale', 4))
        
        self.ensemble_teacher = ensemble_teacher
        
        super().__init__(config, tiny_student)
        
        self.config = config
        
        # Losses
        self.l1_loss = L1Loss()
        
        # FAKD from ensemble
        self.fakd_loss = None
        if config.get('loss', {}).get('distillation', {}).get('fakd_enabled', True):
            self.fakd_loss = FeatureAffinityLoss(
                layers=config.get('loss', {}).get('distillation', {}).get('fakd_layers', [1, 2, 3]),
                layer_weights=config.get('loss', {}).get('distillation', {}).get('fakd_layer_weights'),
                affinity_type='spatial',
            )
        
        # Training settings
        self.epochs = config.get('training', {}).get('epochs', 100)
    
    def train_epoch(self, dataloader, optimizer) -> Dict[str, float]:
        """
        Train tiny student for one epoch.
        """
        self.model.train()
        self.ensemble_teacher.eval()
        
        total_loss = 0
        total_l1 = 0
        total_fakd = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Ensemble Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            lr = batch['lr'].to(self.device, non_blocking=True)
            hr = batch['hr'].to(self.device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast('cuda', enabled=self.use_amp):
                # Get ensemble prediction (teacher)
                with torch.no_grad():
                    teacher_pred = self.ensemble_teacher(lr)
                
                # Student forward with features
                if hasattr(self.model, 'forward_with_features'):
                    student_pred, student_features = self.model.forward_with_features(lr)
                else:
                    student_pred = self.model(lr)
                    student_features = {}
                
                # Extract teacher features (if model supports it)
                teacher_features = {}
                if hasattr(self.ensemble_teacher.model_a, 'forward_with_features'):
                    with torch.no_grad():
                        _, feat_a = self.ensemble_teacher.model_a.forward_with_features(lr)
                        _, feat_b = self.ensemble_teacher.model_b.forward_with_features(lr)
                        # Average features from both teachers
                        teacher_features = {
                            k: (feat_a.get(k, torch.zeros_like(v)) + feat_b.get(k, torch.zeros_like(v))) / 2
                            for k, v in student_features.items()
                        }
                
                # L1 loss to ground truth
                l1 = self.l1_loss(student_pred, hr)
                
                # L1 loss to teacher (distillation)
                distill = self.l1_loss(student_pred, teacher_pred)
                
                # Combined pixel loss
                pixel_loss = 0.5 * l1 + 0.5 * distill
                
                loss = pixel_loss
                loss_dict = {
                    'l1': l1.item(),
                    'distill': distill.item(),
                    'pixel_total': pixel_loss.item(),
                }
                
                # FAKD loss
                if self.fakd_loss is not None and len(student_features) > 0:
                    fakd = self.fakd_loss(student_features, teacher_features)
                    fakd_weight = self.config.get('loss', {}).get('distillation', {}).get('fakd_weight', 0.5)
                    loss = loss + fakd_weight * fakd
                    loss_dict['fakd'] = fakd.item()
                
                loss_dict['total'] = loss.item()
            
            # Backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Update EMA
            self._update_ema()
            
            total_loss += loss_dict['total']
            total_l1 += loss_dict['l1']
            total_fakd += loss_dict.get('fakd', 0)
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'l1': f"{loss_dict['l1']:.4f}",
            })
            
            # Log
            if batch_idx % self.log_interval == 0:
                self.log_metrics(loss_dict, self.global_step, 'ensemble_train')
        
        if num_batches == 0:
            return {'loss': 0.0, 'l1': 0.0, 'fakd': 0.0}

        return {
            'loss': total_loss / num_batches,
            'l1': total_l1 / num_batches,
            'fakd': total_fakd / num_batches,
        }
    
    def validate(self, dataloader) -> Dict[str, float]:
        """Validation"""
        self.model.eval()
        self.ensemble_teacher.eval()
        
        total_psnr = 0
        total_psnr_teacher = 0
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation'):
                lr = batch['lr'].to(self.device, non_blocking=True)
                hr = batch['hr'].to(self.device, non_blocking=True)
                
                # Forward
                if self.use_amp:
                    with autocast('cuda'):
                        student_pred = self.model(lr)
                        teacher_pred = self.ensemble_teacher(lr)
                else:
                    student_pred = self.model(lr)
                    teacher_pred = self.ensemble_teacher(lr)
                
                # Compute metrics
                loss = self.l1_loss(student_pred, hr)
                
                # PSNR for student (with epsilon for numerical stability)
                student_pred = student_pred.clamp(0, 1)
                mse = torch.mean((student_pred - hr) ** 2)
                psnr = 10 * torch.log10(1.0 / (mse + 1e-10))

                # PSNR for teacher (with epsilon for numerical stability)
                teacher_pred = teacher_pred.clamp(0, 1)
                mse_teacher = torch.mean((teacher_pred - hr) ** 2)
                psnr_teacher = 10 * torch.log10(1.0 / (mse_teacher + 1e-10))
                
                total_loss += loss.item()
                total_psnr += psnr.item()
                total_psnr_teacher += psnr_teacher.item()
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'psnr': total_psnr / num_batches,
            'psnr_teacher': total_psnr_teacher / num_batches,
        }
    
    def train(self, train_loader, val_loader=None):
        """Main training loop"""
        # Wrap train_loader with async prefetcher if enabled
        use_prefetcher = self.config.get('training', {}).get('use_async_prefetcher', False)
        if use_prefetcher and torch.cuda.is_available():
            from data.async_prefetcher import create_prefetcher
            train_loader = create_prefetcher(train_loader, self.config, self.device)
        
        print(f"\n{'='*60}")
        print(f"Training Tiny Student with Ensemble Distillation")
        print(f"Teacher: Ensemble (Model A + Model B)")
        print(f"Student: TinyStudent ({self.model.count_parameters():,} params)")
        print(f"{'='*60}\n")
        
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer)
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer)
            print(f"Epoch {epoch}: loss={train_metrics['loss']:.4f}, l1={train_metrics['l1']:.4f}")
            
            # Validate
            if val_loader is not None and epoch % self.val_interval == 0:
                val_metrics = self.validate(val_loader)
                print(f"Validation: loss={val_metrics['loss']:.4f}")
                print(f"  Student PSNR: {val_metrics['psnr']:.2f} dB")
                print(f"  Teacher PSNR: {val_metrics['psnr_teacher']:.2f} dB")
                print(f"  Gap: {val_metrics['psnr_teacher'] - val_metrics['psnr']:.2f} dB")
                self.log_metrics(val_metrics, self.global_step, 'ensemble_val')
                
                if val_metrics['loss'] < self.best_loss:
                    self.best_loss = val_metrics['loss']
                    self.save_checkpoint(epoch, optimizer, is_best=True)
            
            # Save checkpoint
            if epoch % self.save_interval == 0:
                self.save_checkpoint(epoch, optimizer)
            
            if scheduler is not None:
                scheduler.step()
        
        # Save final
        self.save_checkpoint(self.epochs - 1, optimizer)
        print(f"\nEnsemble training complete. Best loss: {self.best_loss:.4f}")
        print(f"Student parameters: {self.model.count_parameters():,}")


def train_ensemble_distillation(
    config: Dict,
    model_a_checkpoint: str,
    model_b_checkpoint: str,
    train_loader,
    val_loader=None,
    device: str = 'cuda',
) -> TinyStudent:
    """
    Convenience function to train ensemble distillation.
    
    Args:
        config: Configuration dict
        model_a_checkpoint: Path to Model A checkpoint
        model_b_checkpoint: Path to Model B checkpoint
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        device: Device to train on
    
    Returns:
        Trained TinyStudent model
    """
    # Create ensemble teacher
    ensemble = create_ensemble_teacher(
        model_a_checkpoint,
        model_b_checkpoint,
        device=device,
    )
    
    # Create trainer
    trainer = EnsembleTrainer(config, ensemble)
    
    # Train
    trainer.train(train_loader, val_loader)
    
    return trainer.model
