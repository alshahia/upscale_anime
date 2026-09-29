"""
Trainer for Model B: Mamba-PAN + MTKD + FAKD
With gradient checkpointing for 8GB VRAM optimization
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
from typing import Dict
from tqdm import tqdm

from training.base_trainer import BaseTrainer
from models.mamba_pan import create_mamba_pan_model, MAMBA_AVAILABLE
from distillation.mtkd.aggregation import KnowledgeAggregationNetwork
from distillation.mtkd.distillation import FullMTKDLoss
from distillation.fakd.affinity_loss import DirectionalFeatureAffinityLoss, CrossDirectionConsistencyLoss
from losses import L1Loss


class ModelBTrainer(BaseTrainer):
    """
    Trainer for Mamba-PAN with MTKD + FAKD.
    
    Key differences from Model A:
    - Lower batch size (4 vs 8)
    - Gradient checkpointing enabled
    - Direction-aware losses
    - Cross-direction consistency loss
    """
    
    def __init__(self, config: Dict):
        self.stage = 1
        
        # Create model
        model = self._create_model(config)
        super().__init__(config, model)
        
        self.config = config
        
        # Stage configs
        stage1_cfg = config.get('training', {}).get('stage1', {})
        stage2_cfg = config.get('training', {}).get('stage2', {})
        
        self.stage1_epochs = stage1_cfg.get('epochs', 100) if stage1_cfg.get('enabled', False) else 0
        self.stage2_epochs = stage2_cfg.get('epochs', 250) if stage2_cfg.get('enabled', False) else 0
        
        # Teachers
        self.teachers = None
        self.knowledge_aggregation = None
        
        # Losses
        self.l1_loss = L1Loss()
        self.mtkd_loss = None
        self.fakd_loss = None
        self.consistency_loss = None
        
        self._setup_losses()
        # Note: Gradient checkpointing is already enabled in BaseTrainer if configured

    def _create_model(self, config: Dict) -> nn.Module:
        """Create Mamba-PAN model"""
        if self.stage == 1:
            # Knowledge Aggregation Network
            num_teachers = len(config.get('training', {}).get('stage1', {}).get('teachers', []))
            model = KnowledgeAggregationNetwork(
                num_teachers=num_teachers,
                embed_dim=64,
                num_blocks=6,
                scale=config.get('model', {}).get('scale', 4),
            )
        else:
            # Mamba-PAN student
            model = create_mamba_pan_model(config.get('model', {}))
        
        return model
    
    def _setup_losses(self):
        """Setup loss functions with direction-aware variants"""
        distillation_cfg = self.config.get('loss', {}).get('distillation', {})
        
        # Perceptual loss (DISTS or VGG)
        self.perceptual_loss = None
        self.perceptual_weight = 0.0
        perceptual_cfg = self.config.get('loss', {}).get('perceptual', {})
        if perceptual_cfg.get('enabled', False):
            loss_type = perceptual_cfg.get('type', 'vgg')
            self.perceptual_weight = perceptual_cfg.get('weight', 0.1)
            
            if loss_type == 'dists':
                from losses import DISTSLoss
                self.perceptual_loss = DISTSLoss(alpha=perceptual_cfg.get('alpha', 0.5))
                print(f"  Enabled: DISTS perceptual loss (weight={self.perceptual_weight})")
            elif loss_type == 'vgg':
                from losses import VGGPerceptualLoss
                self.perceptual_loss = VGGPerceptualLoss()
                print(f"  Enabled: VGG perceptual loss (weight={self.perceptual_weight})")
            elif loss_type == 'resnet':
                from losses import ResNetPerceptualLoss
                self.perceptual_loss = ResNetPerceptualLoss()
                print(f"  Enabled: ResNet perceptual loss (weight={self.perceptual_weight})")
            
            # Move perceptual loss to the same device as the model
            if self.perceptual_loss is not None:
                self.perceptual_loss = self.perceptual_loss.to(self.device)
        
        if distillation_cfg.get('enabled', False):
            self.mtkd_loss = FullMTKDLoss(
                l1_weight=self.config.get('loss', {}).get('pixel_loss', {}).get('weight', 1.0),
                wavelet_weight=distillation_cfg.get('mtkd_weight', 1.0),
                wavelet_levels=3,
            )
            
            # Direction-aware FAKD for Mamba
            self.fakd_loss = DirectionalFeatureAffinityLoss(
                directions=['h', 'v', 'rh', 'rv'],
                layers=distillation_cfg.get('fakd_layers', [2, 4, 6, 8]),
                layer_weights=distillation_cfg.get('fakd_layer_weights'),
                affinity_type='spatial',
            )
            
            # Cross-direction consistency loss
            if self.config.get('distillation', {}).get('consistency', {}).get('enabled', True):
                self.consistency_loss = CrossDirectionConsistencyLoss(
                    consistency_type='variance',
                    weight=self.config.get('distillation', {}).get('consistency', {}).get('weight', 0.1),
                )
    
    def train_stage2_epoch(self, dataloader, optimizer) -> Dict[str, float]:
        """
        Train Mamba-PAN student with direction-aware distillation.
        """
        self.model.train()
        
        total_loss = 0
        total_l1 = 0
        total_wavelet = 0
        total_fakd = 0
        total_consistency = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Model B Stage 2 Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            lr = batch['lr'].to(self.device, non_blocking=True)
            hr = batch['hr'].to(self.device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast('cuda', enabled=self.use_amp):
                # Student forward with features
                student_dir_features = {}
                if hasattr(self.model, 'forward_with_direction_features'):
                    # Mamba-PAN: extract direction-specific features in a single forward pass
                    student_pred, student_dir_features = self.model.forward_with_direction_features(lr)
                    student_features = {}
                elif hasattr(self.model, 'forward_with_features'):
                    student_pred, student_features = self.model.forward_with_features(lr)
                else:
                    student_pred = self.model(lr)
                    student_features = {}
                
                # Get direction outputs for consistency loss
                direction_outputs = {}
                if hasattr(self.model, 'get_direction_outputs') and self.consistency_loss is not None:
                    direction_outputs = self.model.get_direction_outputs(lr)
                
                # Get teacher predictions (if available)
                with torch.no_grad():
                    teacher_features = {}
                    aggregated = None
                    
                    if self.teachers and len(self.teachers) > 0:
                        teacher_outputs = [teacher(lr) for teacher in self.teachers]
                        
                        if self.knowledge_aggregation is not None:
                            aggregated = self.knowledge_aggregation(teacher_outputs)
                        else:
                            aggregated = teacher_outputs[0] if teacher_outputs else None
                
                # Compute losses
                l1 = self.l1_loss(student_pred, hr)

                loss = l1
                loss_dict = {'l1': l1.item()}

                # MTKD wavelet loss (add to L1, matching Model A pattern)
                if self.mtkd_loss is not None and aggregated is not None:
                    mtkd_loss, mtkd_dict = self.mtkd_loss(student_pred, aggregated, hr)
                    loss = loss + mtkd_loss  # Additive, not overwrite
                    loss_dict.update(mtkd_dict)
                
                # Direction-aware FAKD
                if self.fakd_loss is not None:
                    if student_dir_features:
                        # Mamba-PAN: direction-specific features from forward_with_direction_features
                        fakd = self.fakd_loss(student_dir_features, teacher_features)
                    elif len(student_features) > 0:
                        fakd = self.fakd_loss(student_features, teacher_features)
                    else:
                        fakd = None

                    if fakd is not None:
                        fakd_weight = self.config.get('loss', {}).get('distillation', {}).get('fakd_weight', 0.5)
                        loss = loss + fakd_weight * fakd
                        loss_dict['fakd'] = fakd.item()
                
                # Cross-direction consistency loss
                if self.consistency_loss is not None and len(direction_outputs) > 0:
                    consistency = self.consistency_loss(direction_outputs)
                    loss = loss + consistency
                    loss_dict['consistency'] = consistency.item()
                
                # Add perceptual loss (DISTS/VGG)
                if self.perceptual_loss is not None:
                    p_loss = self.perceptual_loss(student_pred, hr)
                    loss = loss + self.perceptual_weight * p_loss
                    loss_dict['perceptual'] = p_loss.item()
                
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
            total_l1 += loss_dict.get('l1', 0)
            total_wavelet += loss_dict.get('wavelet_distill', 0)
            total_fakd += loss_dict.get('fakd', 0)
            total_consistency += loss_dict.get('consistency', 0)
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'l1': f"{loss_dict.get('l1', 0):.4f}",
            })
            
            # Log
            if batch_idx % self.log_interval == 0:
                self.log_metrics(loss_dict, self.global_step, 'model_b_train')
        
        if num_batches == 0:
            return {'loss': 0.0, 'l1': 0.0, 'wavelet': 0.0, 'fakd': 0.0, 'consistency': 0.0}

        return {
            'loss': total_loss / num_batches,
            'l1': total_l1 / num_batches,
            'wavelet': total_wavelet / num_batches,
            'fakd': total_fakd / num_batches,
            'consistency': total_consistency / num_batches,
        }
    
    def train_epoch(self, dataloader, optimizer, loss_fn=None) -> Dict[str, float]:
        """Route to appropriate stage training"""
        if self.stage == 1:
            # Use base trainer's default for stage 1 (L1 only)
            return super().train_epoch(dataloader, optimizer, loss_fn)
        else:
            return self.train_stage2_epoch(dataloader, optimizer)
    
    def validate(self, dataloader) -> Dict[str, float]:
        """Validation"""
        self.model.eval()
        
        total_psnr = 0
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation'):
                lr = batch['lr'].to(self.device, non_blocking=True)
                hr = batch['hr'].to(self.device, non_blocking=True)
                
                # Forward
                if self.use_amp:
                    with autocast('cuda'):
                        pred = self.model(lr)
                else:
                    pred = self.model(lr)
                
                # Compute metrics
                loss = self.l1_loss(pred, hr)
                
                # PSNR (with epsilon for numerical stability)
                pred = pred.clamp(0, 1)
                mse = torch.mean((pred - hr) ** 2)
                psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
                
                total_loss += loss.item()
                total_psnr += psnr.item()
                num_batches += 1
        
        if num_batches == 0:
            return {'loss': 0.0, 'psnr': 0.0}

        return {
            'loss': total_loss / num_batches,
            'psnr': total_psnr / num_batches,
        }
    
    def train(self, train_loader, val_loader=None):
        """Main training loop - Stage 2 only for Model B (similar to Model A)"""
        
        # Wrap train_loader with async prefetcher if enabled
        use_prefetcher = self.config.get('training', {}).get('use_async_prefetcher', False)
        if use_prefetcher and torch.cuda.is_available():
            from data.async_prefetcher import create_prefetcher
            train_loader = create_prefetcher(train_loader, self.config, self.device)
        
        if self.stage2_epochs > 0:
            print(f"\n{'='*50}")
            print(f"Training Mamba-PAN Student with Direction-Aware Distillation")
            if not MAMBA_AVAILABLE:
                print(f"(Using fallback implementation - install mamba-ssm for better performance)")
            print(f"{'='*50}\n")
            
            # Recreate model as Mamba-PAN student
            self.stage = 2
            self.model = self._create_model(self.config).to(self.device)
            # Recreate EMA for the new architecture (deepcopy of stage-1 model is incompatible)
            if self.use_ema:
                self.ema_model = self._create_ema_model()
            self.best_loss = float('inf')
            
            optimizer = self._create_optimizer()
            self.set_scheduler(self._create_scheduler(optimizer))  # Use set_scheduler for deferred loading

            for epoch in range(self.stage2_epochs):
                self.current_epoch = epoch

                # Train
                train_metrics = self.train_stage2_epoch(train_loader, optimizer)
                print(f"Epoch {epoch}: loss={train_metrics['loss']:.4f}, l1={train_metrics['l1']:.4f}")

                # Validate
                if val_loader is not None and epoch % self.val_interval == 0:
                    val_metrics = self.validate(val_loader)
                    print(f"Validation: loss={val_metrics['loss']:.4f}, PSNR={val_metrics['psnr']:.2f}")
                    self.log_metrics(val_metrics, self.global_step, 'model_b_val')

                    if val_metrics['loss'] < self.best_loss:
                        self.best_loss = val_metrics['loss']
                        self.save_checkpoint(epoch, optimizer, is_best=True)

                # Save checkpoint
                if epoch % self.save_interval == 0:
                    self.save_checkpoint(epoch, optimizer)

                if self.scheduler is not None:
                    self.scheduler.step()
            
            # Save final checkpoint
            self.save_checkpoint(self.stage2_epochs - 1, optimizer)
            print(f"\nStage 2 complete. Best loss: {self.best_loss:.4f}\n")
        
        print("Training complete!")
