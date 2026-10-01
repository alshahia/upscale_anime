"""
Trainer for Model A: NTIRE + MTKD + FAKD
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pickle import UnpicklingError
import logging
try:
    from torch.amp import autocast  # PyTorch 2.0+
except ImportError:
    from torch.cuda.amp import autocast as _autocast  # PyTorch 1.x
    
    # Wrapper for PyTorch 1.x compatibility (ignores device argument)
    def autocast(device=None, enabled=True):
        return _autocast(enabled=enabled)
from typing import Dict
from tqdm import tqdm
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from anime_sr.training.base_trainer import BaseTrainer
    from anime_sr.training.feature_distillation import FeatureDistillationLoss, MultiTeacherFeatureDistillation
    from anime_sr.models.span import create_span_model
    from anime_sr.distillation.mtkd.aggregation_factory import create_aggregation_network
    from anime_sr.distillation.mtkd.distillation import FullMTKDLoss
    from anime_sr.distillation.fakd.affinity_loss import FeatureAffinityLoss
    from anime_sr.distillation.mtkd.simple_aggregation import SimpleKnowledgeAggregation
    from anime_sr.distillation.mtkd.aggregation import KnowledgeAggregationNetwork
    from anime_sr.losses import L1Loss
    from anime_sr.utils.pretrained_models import get_teacher_model_path, download_all_teacher_models
    from anime_sr.utils.checkpoint_loader import load_pretrained_weights
    from anime_sr.data.augmentation import AugmentationPipeline
except ImportError:
    from anime_sr.training.base_trainer import BaseTrainer
    from anime_sr.training.feature_distillation import FeatureDistillationLoss, MultiTeacherFeatureDistillation
    from anime_sr.models.span import create_span_model
    from anime_sr.distillation.mtkd.aggregation_factory import create_aggregation_network
    from anime_sr.distillation.mtkd.distillation import FullMTKDLoss
    from anime_sr.distillation.fakd.affinity_loss import FeatureAffinityLoss
    from anime_sr.distillation.mtkd.simple_aggregation import SimpleKnowledgeAggregation
    from anime_sr.distillation.mtkd.aggregation import KnowledgeAggregationNetwork
    from losses import L1Loss
    from anime_sr.utils.pretrained_models import get_teacher_model_path, download_all_teacher_models
    from anime_sr.utils.checkpoint_loader import load_pretrained_weights
    from anime_sr.data.augmentation import AugmentationPipeline


class ModelATrainer(BaseTrainer):
    """
    Trainer for NTIRE SPAN model with MTKD + FAKD.
    
    Supports two stages:
    - Stage 1: Train Knowledge Aggregation network
    - Stage 2: Train SPAN student with MTKD + FAKD
    """
    
    def __init__(self, config: Dict):
        self.stage = 1  # Current training stage
        self.config = config
        
        # Store pretrained checkpoint path for Stage 2 loading
        self.pretrained_checkpoint = config.get('training', {}).get('pretrained_checkpoint')
        
        # Create model based on stage
        model = self._create_model(config)
        super().__init__(config, model)
        
        # Stage configs
        stage1_cfg = config.get('training', {}).get('stage1', {})
        stage2_cfg = config.get('training', {}).get('stage2', {})
        
        self.stage1_epochs = stage1_cfg.get('epochs', 100) if stage1_cfg.get('enabled', False) else 0
        self.stage2_epochs = stage2_cfg.get('epochs', 200) if stage2_cfg.get('enabled', False) else 0
        
        # Teachers (for Stage 1 and 2)
        self.teachers = None
        self.knowledge_aggregation = None
        
        # Losses
        self.l1_loss = L1Loss()
        self.mtkd_loss = None
        self.fakd_loss = None
        
        self._setup_losses()
    
    def _create_model(self, config: Dict) -> nn.Module:
        """Create model based on current stage"""
        # Stage 1: Knowledge Aggregation (or SPAN if no teachers)
        # Stage 2: SPAN student
        
        if self.stage == 1:
            # Check if teachers are configured
            stage1_cfg = config.get('training', {}).get('stage1', {})
            teachers = stage1_cfg.get('teachers', [])
            if not teachers:
                # No teachers - use SPAN directly for demo/basic training
                print("[Stage 1] No teachers configured, using SPAN student directly")
                from anime_sr.models.span import create_span_model
                model = create_span_model(config.get('model', {}))
            else:
                # Use factory to create appropriate aggregation network
                model = create_aggregation_network(config)
        else:
            # SPAN student
            from anime_sr.models.span import create_span_model
            model = create_span_model(config.get('model', {}))
            
            # Load pretrained checkpoint for Stage 2 SPAN model
            if self.pretrained_checkpoint:
                print(f"\n[Transfer Learning] Loading pretrained checkpoint for Stage 2:")
                print(f"  Path: {self.pretrained_checkpoint}")
                
                # Check if this is a checkpoint-compatible model
                model_type = config.get('model', {}).get('type', '')
                if model_type in ['checkpoint_compatible', 'checkpoint']:
                    # Use specialized loader for checkpoint-compatible models
                    try:
                        from anime_sr.models.span.checkpoint_compatible_exact import load_checkpoint_compatible_weights_exact
                        success, loading_info = load_checkpoint_compatible_weights_exact(
                            model,
                            self.pretrained_checkpoint,
                            device='cpu',
                            verbose=True
                        )
                    except ImportError:
                        print(f"  Warning: Could not import specialized loader, falling back to generic loader")
                        from anime_sr.utils.checkpoint_loader import load_pretrained_weights
                        success = load_pretrained_weights(
                            model,
                            self.pretrained_checkpoint,
                            strict=False,
                            device='cpu',
                            verbose=True
                        )
                else:
                    # Use generic loader for regular models
                    from anime_sr.utils.checkpoint_loader import load_pretrained_weights
                    success = load_pretrained_weights(
                        model,
                        self.pretrained_checkpoint,
                        strict=False,  # Allow partial loading for fine-tuning
                        device='cpu',  # Load on CPU first, then move to device
                        verbose=True
                    )
                
                if success:
                    print("  Pretrained weights loaded successfully!")
                    print("  Ready for fine-tuning with low learning rate.\n")
                else:
                    print("  Warning: Failed to load pretrained weights.")
                    print("  Training will start from scratch.\n")
        
        return model
    
    def transition_to_stage2(self):
        """Transition from Stage 1 to Stage 2 and create SPAN student model"""
        print("\n" + "="*60)
        print("Transitioning to Stage 2: SPAN Student Training")
        print("="*60)
        
        self.stage = 2
        
        # Create new SPAN student model (this will load checkpoint)
        new_model = self._create_model(self.config)
        
        # Update the model in the base trainer
        self.model = new_model
        self.model.to(self.device)
        
        # Reinitialize EMA model for new architecture
        if self.use_ema:
            self.ema_model = self._create_ema_model()
            print("  Reinitialized EMA model for Stage 2")
        
        print("Stage 2 transition complete. SPAN student model ready.")
    
    def _apply_gpu_degradation(self, hr: torch.Tensor, scale: int = 4) -> torch.Tensor:
        """
        Apply anime degradation on GPU using the AnimeDegradationPipeline.
        This offloads heavy CPU processing to GPU, eliminating data loading bottleneck.
        
        Args:
            hr: High-res images on GPU [B, C, H, W]
            scale: Downsampling factor
            
        Returns:
            lr: Low-res degraded images on GPU [B, C, H//scale, W//scale]
        """
        from anime_sr.data.anime_degradation import AnimeDegradationPipeline
        import torch.nn.functional as F
        
        # Initialize degradation pipeline (cached after first call)
        if not hasattr(self, '_anime_degradation'):
            use_anime = self.config.get('data', {}).get('degradation', {}).get('anime_degradation', False)
            
            self._anime_degradation = AnimeDegradationPipeline(
                enable_blur=True,
                enable_directional=True,
                enable_quantization=use_anime,
                enable_banding=use_anime,
                enable_ringing=use_anime,
            ).to(self.device)
            self._anime_degradation.eval()  # No gradients needed
            print(f"[GPU Degradation] Initialized AnimeDegradationPipeline on GPU (anime={use_anime})")
        
        # Apply degradation
        with torch.no_grad():
            hr_degraded = self._anime_degradation(hr)
            
            # Downsample with bicubic
            lr = F.interpolate(
                hr_degraded,
                scale_factor=1.0 / scale,
                mode='bicubic',
                align_corners=False,
            )
            lr = torch.clamp(lr, 0, 1)
        
        return lr
    
    def _setup_losses(self):
        """Setup loss functions including anime-specific and perceptual losses"""
        distillation_cfg = self.config.get('loss', {}).get('distillation', {})
        
        # Base L1 loss
        self.l1_loss = L1Loss()
        
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
                # Verify buffers moved correctly (prevents device mismatch in AMP)
                # Compare devices by normalizing to string representation
                if hasattr(self.perceptual_loss, 'mean'):
                    actual_device = str(self.perceptual_loss.mean.device)
                    expected_device = str(self.device)
                    # Handle equivalence: cuda == cuda:0 when there's only one GPU
                    if actual_device != expected_device:
                        # Normalize cuda:0 to cuda for single GPU case
                        if expected_device == 'cuda' and actual_device == 'cuda:0':
                            pass  # Equivalent
                        elif expected_device.startswith('cuda:') and actual_device.startswith('cuda:'):
                            raise RuntimeError(
                                f"Perceptual loss buffers on wrong device: "
                                f"{actual_device} vs {expected_device}. "
                                f"This may cause errors during mixed-precision training."
                            )
        
        # MTKD + FAKD losses
        if distillation_cfg.get('enabled', False):
            self.mtkd_loss = FullMTKDLoss(
                l1_weight=0.0,  # L1 already added separately in train_stage2_epoch
                wavelet_weight=distillation_cfg.get('mtkd_weight', 1.0),
                wavelet_levels=3,
            )
            
            self.fakd_loss = FeatureAffinityLoss(
                layers=distillation_cfg.get('fakd_layers', [2, 4, 6, 8]),
                layer_weights=distillation_cfg.get('fakd_layer_weights'),
                affinity_type='spatial',
            )
        
        # Stage 1 config for anime losses
        stage1_cfg = self.config.get('training', {}).get('stage1', {})
        anime_cfg = stage1_cfg.get('anime', {})
        
        # Anime-specific losses
        self.line_art_loss = None
        self.color_loss = None
        self.flat_loss = None
        self.temporal_loss = None
        
        if anime_cfg.get('line_art_preservation', False):
            from anime_sr.losses.anime_losses import LineArtPreservationLoss
            self.line_art_loss = LineArtPreservationLoss(
                edge_weight=anime_cfg.get('line_art_weight', 1.0)
            )
            print(f"  Enabled: Line-art preservation loss (weight={anime_cfg.get('line_art_weight', 1.0)})")
        
        if anime_cfg.get('color_consistency', False):
            from anime_sr.losses.anime_losses import ColorConsistencyLoss
            self.color_loss = ColorConsistencyLoss(
                color_weight=anime_cfg.get('color_weight', 0.5)
            )
            print(f"  Enabled: Color consistency loss (weight={anime_cfg.get('color_weight', 0.5)})")
        
        if anime_cfg.get('flat_region_preservation', False):
            from anime_sr.losses.anime_losses import FlatRegionPreservationLoss
            self.flat_loss = FlatRegionPreservationLoss(
                preserve_weight=anime_cfg.get('flat_weight', 0.5)
            )
            print(f"  Enabled: Flat region preservation loss (weight={anime_cfg.get('flat_weight', 0.5)})")
        
        # Temporal consistency for video
        video_cfg = self.config.get('data', {}).get('video', {})
        if video_cfg.get('temporal_training', False):
            from anime_sr.losses.temporal_loss import TemporalConsistencyLoss, VideoAwareLoss
            base_temporal = TemporalConsistencyLoss()
            self.temporal_loss = VideoAwareLoss(base_temporal)
            print(f"  Enabled: Temporal consistency loss")
        
        # Feature distillation loss (for transfer learning with teachers)
        self.feature_distillation_loss = None
        feature_dist_cfg = self.config.get('loss', {}).get('feature_distillation', {})
        if feature_dist_cfg.get('enabled', False):
            layer_indices = feature_dist_cfg.get('layers', [2, 4, 6, 8])
            fd_weight = feature_dist_cfg.get('weight', 0.1)
            
            self.feature_distillation_loss = FeatureDistillationLoss(
                layer_indices=layer_indices,
                weights=None,
                loss_type='l2'
            )
            print(f"  Enabled: Feature distillation loss (layers={layer_indices}, weight={fd_weight})")
        
        # Multi-teacher feature distillation (deferred initialization - teachers not loaded yet)
        # Will be initialized in load_teachers() after teachers are available
        self.multi_teacher_fd = None
        self._mt_fd_cfg = self.config.get('training', {}).get('stage1', {}).get('feature_distillation', {})
        
        # Advanced augmentation pipeline
        self.aug_pipeline = None
        aug_cfg = self.config.get('training', {}).get('augmentation', {})
        if aug_cfg.get('enabled', False):
            self.aug_pipeline = AugmentationPipeline(
                mixup_prob=aug_cfg.get('mixup_prob', 0.3),
                cutmix_prob=aug_cfg.get('cutmix_prob', 0.3),
                random_resize_prob=aug_cfg.get('random_resize_prob', 0.5)
            )
            print(f"  Enabled: Advanced augmentation (Mixup/CutMix)")
    
    def load_teachers(self, auto_download: bool = None):
        """
        Load pretrained teacher models.
        
        Args:
            auto_download: If True, automatically download missing teacher models.
                          If None, uses config value training.stage1.auto_download_teachers (default: True)
        """
        stage1_cfg = self.config.get('training', {}).get('stage1', {})
        teacher_configs = stage1_cfg.get('teachers', [])
        strict_mode = stage1_cfg.get('strict_teacher_loading', False)
        
        # Determine auto-download behavior
        if auto_download is None:
            auto_download = stage1_cfg.get('auto_download_teachers', True)
        scale = self.config.get('model', {}).get('scale', 4)
        
        self.teachers = []
        missing_teachers = []
        failed_teachers = []
        
        # First pass: check which teachers need to be downloaded
        for teacher_cfg in teacher_configs:
            teacher_name = teacher_cfg.get('name', '').lower()
            teacher_path = teacher_cfg.get('path')
            
            # Check if explicit path exists
            if teacher_path and Path(teacher_path).exists():
                continue
            
            # Try to get/download model
            if auto_download:
                actual_path = get_teacher_model_path(teacher_name, scale, self.config, auto_download=False)
                if actual_path is None:
                    missing_teachers.append(teacher_name)
            else:
                if teacher_path and not Path(teacher_path).exists():
                    missing_teachers.append(teacher_name)
        
        # Auto-download missing models
        if missing_teachers and auto_download:
            print(f"\n{'='*60}")
            print(f"Auto-downloading missing teacher models: {missing_teachers}")
            print(f"{'='*60}\n")
            
            # Download all at once for better progress display
            download_all_teacher_models(scale, self.config)
        
        # Second pass: load teachers
        for teacher_cfg in teacher_configs:
            teacher_name = teacher_cfg.get('name', '').lower()
            teacher_path = teacher_cfg.get('path')
            
            # Resolve actual path
            actual_path = None
            if teacher_path and Path(teacher_path).exists():
                actual_path = teacher_path
            else:
                # Try to get from auto-downloaded models
                actual_path = get_teacher_model_path(teacher_name, scale, self.config, auto_download=False)
            
            if actual_path is None or not Path(actual_path).exists():
                error_msg = f"Could not find teacher model '{teacher_name}'"
                if strict_mode:
                    raise FileNotFoundError(
                        f"{error_msg}. Set 'strict_teacher_loading: false' to skip missing teachers. "
                        f"Run manually: python -m utils.pretrained_models --model {teacher_name} --scale {scale}"
                    )
                print(f"Warning: {error_msg}")
                print(f"  Run manually: python -m utils.pretrained_models --model {teacher_name} --scale {scale}")
                failed_teachers.append(teacher_name)
                continue
            
            print(f"Loading teacher: {teacher_name} from {actual_path}")
            
            try:
                # Load teacher model based on architecture type
                teacher = self._load_teacher_model(teacher_name, actual_path)
                
                if teacher is not None:
                    teacher.eval()
                    for param in teacher.parameters():
                        param.requires_grad = False
                    
                    # Validate teacher by running dummy input
                    try:
                        with torch.no_grad():
                            dummy_input = torch.randn(1, 3, 64, 64, device=self.device)
                            teacher = teacher.to(self.device)
                            test_output = teacher(dummy_input)
                            
                            if torch.isnan(test_output).any() or torch.isinf(test_output).any():
                                print(f"  [WARNING] Teacher {teacher_name} produced NaN/Inf on validation input - excluding from ensemble")
                                failed_teachers.append(teacher_name)
                                if strict_mode:
                                    raise RuntimeError(f"Teacher '{teacher_name}' failed validation - produces NaN/Inf")
                                continue
                            
                            self.teachers.append(teacher)
                            print(f"  [OK] Loaded {teacher_name}")
                    except Exception as val_err:
                        print(f"  [WARNING] Teacher {teacher_name} validation failed: {val_err} - excluding from ensemble")
                        failed_teachers.append(teacher_name)
                        if strict_mode:
                            raise RuntimeError(f"Teacher '{teacher_name}' validation failed: {val_err}")
                else:
                    failed_teachers.append(teacher_name)
                    print(f"  [ERROR] Failed to load {teacher_name}")
                    if strict_mode:
                        raise RuntimeError(f"Failed to load required teacher '{teacher_name}'")
                    
            except Exception as e:
                failed_teachers.append(teacher_name)
                print(f"  [ERROR] Error loading {teacher_name}: {e}")
                if strict_mode:
                    raise RuntimeError(f"Error loading required teacher '{teacher_name}': {e}")
        
        # Summary with clear warning if teachers failed to load
        total_requested = len(teacher_configs)
        loaded = len(self.teachers)
        print(f"Loaded {loaded}/{total_requested} teacher models")
        
        if failed_teachers:
            print(f"\n[WARNING] {len(failed_teachers)} teacher(s) failed to load: {failed_teachers}")
            print(f"   Training will continue with reduced ensemble quality.")
            if loaded == 0 and total_requested > 0:
                raise RuntimeError(
                    "No teachers loaded but teachers required for training. "
                    "Check model paths or enable auto_download_teachers."
                )
        
        # Initialize multi-teacher feature distillation now that teachers are loaded
        if self._mt_fd_cfg.get('enabled', False) and self.teachers:
            teacher_weights = [t.get('weight', 1.0) for t in self.config.get('training', {}).get('stage1', {}).get('teachers', [])]
            layer_indices = self._mt_fd_cfg.get('layers', [2, 4, 6, 8])
            fd_weight = self._mt_fd_cfg.get('weight', 0.1)
            
            self.multi_teacher_fd = MultiTeacherFeatureDistillation(
                teachers=list(self.teachers.values()) if isinstance(self.teachers, dict) else self.teachers,
                teacher_weights=teacher_weights,
                layer_indices=layer_indices,
                loss_weight=fd_weight
            )
            print(f"  Enabled: Multi-teacher feature distillation")
    
    def _load_teacher_model(self, teacher_name: str, checkpoint_path: str):
        """
        Load a specific teacher model architecture.

        Args:
            teacher_name: Name of the teacher (edsr, rcan, swinir)
            checkpoint_path: Path to checkpoint file

        Returns:
            Loaded model or None if loading failed
        """
        from anime_sr.models.teachers import load_teacher_model

        teacher_name = teacher_name.lower()
        scale = self.config.get('model', {}).get('scale', 4)
        device = self.device

        try:
            # Use the teacher loader with auto-detection
            model = load_teacher_model(
                checkpoint_path=checkpoint_path,
                scale=scale,
                arch=teacher_name,  # Provide hint for architecture
                device=device,
                auto_detect=True,
            )
            return model

        except Exception as e:
            print(f"    Error loading teacher model: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_teacher_features(self, lr: torch.Tensor) -> tuple:
        """
        Extract outputs and intermediate features from all teachers for FAKD.
        
        Args:
            lr: Low-resolution input [B, C, H, W]
            
        Returns:
            (teacher_outputs, aggregated_features)
            - teacher_outputs: List of teacher predictions
            - aggregated_features: Dict of aggregated intermediate features
        """
        if not self.teachers:
            return [], {}
        
        teacher_outputs = []
        teacher_features_list = []
        
        for teacher in self.teachers:
            # Get prediction
            output = teacher(lr)
            teacher_outputs.append(output)
            
            # Get intermediate features if available
            if hasattr(teacher, 'get_features'):
                features = teacher.get_features(lr)
                teacher_features_list.append(features)
        
        # Aggregate features across teachers
        aggregated_features = self._aggregate_teacher_features(teacher_features_list)
        
        return teacher_outputs, aggregated_features

    def _aggregate_teacher_features(self, teacher_feature_list: list) -> dict:
        """
        Aggregate intermediate features from multiple teachers for FAKD.
        Uses average pooling across teachers.
        
        Args:
            teacher_feature_list: List of feature dicts from each teacher
            
        Returns:
            Dict of aggregated features
        """
        if not teacher_feature_list:
            return {}
        
        if len(teacher_feature_list) == 1:
            return teacher_feature_list[0]
        
        aggregated = {}
        
        # Get all feature keys from first teacher
        reference_features = teacher_feature_list[0]
        
        for key in reference_features.keys():
            # Collect features for this key from all teachers
            features_for_key = []
            for teacher_features in teacher_feature_list:
                if key in teacher_features:
                    features_for_key.append(teacher_features[key])
            
            if features_for_key:
                # Average across teachers
                aggregated[key] = torch.stack(features_for_key).mean(dim=0)
        
        return aggregated
    
    def _check_tensor_stats(self, tensor, name, batch_idx):
        """Debug helper: check tensor for NaN/Inf and print statistics"""
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        stats = {
            'min': tensor.min().item(),
            'max': tensor.max().item(),
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'shape': list(tensor.shape),
        }
        status = "OK" if not (has_nan or has_inf) else f"{'NaN' if has_nan else ''}{' Inf' if has_inf else ''}"
        print(f"  [{name}] {status} | shape={stats['shape']} min={stats['min']:.4f} max={stats['max']:.4f} mean={stats['mean']:.4f} std={stats['std']:.4f}")
        return has_nan or has_inf

    def train_stage1_epoch(self, dataloader, optimizer) -> Dict[str, float]:
        """
        Train Knowledge Aggregation for one epoch.
        Supports gradient accumulation and OHEM.
        """
        self.model.train()

        if not self.teachers:
            self.load_teachers()
            if not self.teachers:
                raise RuntimeError("No teacher models could be loaded. Check teacher paths in config.")

        # Get stage 1 config
        stage1_cfg = self.config.get('training', {}).get('stage1', {})
        
        # Enable debug mode for first epoch or if explicitly configured
        debug_mode = stage1_cfg.get('debug_nan', True)
        if debug_mode and self.current_epoch == 0:
            logger.info("\n" + "="*60)
            logger.info("DEBUG MODE: NaN detection enabled for Stage 1")
            logger.info("="*60)

        # Gradient accumulation settings
        accumulation_steps = stage1_cfg.get('gradient_accumulation_steps', 1)
        
        # OHEM settings
        use_ohem = stage1_cfg.get('ohem_enabled', False)
        ohem_ratio = stage1_cfg.get('ohem_ratio', 0.7)
        
        total_loss = 0
        total_l1 = 0
        total_line_art = 0
        total_color = 0
        total_flat = 0
        num_batches = 0
        has_scaled_batch = False  # Track if any scaler.scale() was called
        optimizer.zero_grad()  # Zero gradients at start for accumulation

        pbar = tqdm(dataloader, desc=f"Stage 1 Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):
            lr = batch['lr'].to(self.device, non_blocking=True) if batch['lr'] is not None else None
            hr = batch['hr'].to(self.device, non_blocking=True)
            
            # Apply GPU degradation if lr is None (indicates gpu_degradation mode)
            if lr is None:
                scale = self.config.get('model', {}).get('scale', 4)
                lr = self._apply_gpu_degradation(hr, scale)
            
            # Handle temporal data: reshape [B, T, C, H, W] -> [B*T, C, H, W]
            is_temporal = lr.dim() == 5
            if is_temporal:
                # LR and HR have different spatial dims, get shapes separately
                B_lr, T_lr, C_lr, H_lr, W_lr = lr.shape
                B_hr, T_hr, C_hr, H_hr, W_hr = hr.shape
                lr = lr.view(B_lr * T_lr, C_lr, H_lr, W_lr)
                hr = hr.view(B_hr * T_hr, C_hr, H_hr, W_hr)
            
            # Zero gradients only at start of accumulation cycle
            if batch_idx % accumulation_steps == 0:
                optimizer.zero_grad()
            
            if debug_mode and batch_idx < 3:  # Debug first 3 batches only
                self._check_tensor_stats(lr, "LR input", batch_idx)
                self._check_tensor_stats(hr, "HR target", batch_idx)

            # Get teacher outputs with normalization and NaN filtering
            with torch.no_grad():
                teacher_outputs = []
                valid_teachers = []
                for i, teacher in enumerate(self.teachers):
                    to = teacher(lr)
                    
                    # Check for NaN/Inf in teacher output
                    if torch.isnan(to).any() or torch.isinf(to).any():
                        print(f"\n[WARNING] Teacher {i} ({self.teacher_names[i] if hasattr(self, 'teacher_names') else 'unknown'}) produced NaN/Inf! Skipping this teacher.")
                        continue
                    
                    # Clip extreme values and normalize
                    to = torch.clamp(to, min=0.0, max=1.0)
                    teacher_outputs.append(to)
                    valid_teachers.append(i)
                
                # Need at least one valid teacher
                if len(teacher_outputs) == 0:
                    print(f"\n[ERROR] All teachers produced NaN at batch {batch_idx}! Skipping batch.")
                    continue
                
                # Skip batch if too many teachers failed (more than half)
                expected_teachers = len(self.teachers)
                failed_count = expected_teachers - len(teacher_outputs)
                if failed_count > expected_teachers // 2:
                    logger.warning(f"Too many teachers failed ({failed_count}/{expected_teachers}) at batch {batch_idx}. Skipping batch.")
                    continue
                
                # Pad with zeros to maintain expected teacher count for aggregation network
                # The model was created expecting a specific number of teachers
                while len(teacher_outputs) < expected_teachers:
                    # Create zero tensor with same shape as valid teachers
                    zero_teacher = torch.zeros_like(teacher_outputs[0])
                    teacher_outputs.append(zero_teacher)
                    valid_teachers.append(-1)  # Mark as padded

            if debug_mode and batch_idx < 3:
                for i, idx in enumerate(valid_teachers):
                    label = f"Teacher {idx} output (valid)" if idx >= 0 else f"Teacher {i} output (padded)"
                    self._check_tensor_stats(teacher_outputs[i], label, batch_idx)

            # Forward through aggregation network
            with autocast('cuda', enabled=self.use_amp):
                aggregated = self.model(teacher_outputs)

                if debug_mode and batch_idx < 3:
                    self._check_tensor_stats(aggregated, "Aggregated output", batch_idx)

                # Compute loss (with OHEM if enabled)
                if use_ohem:
                    from anime_sr.training.ohem import compute_ohem_loss
                    loss, ohem_info = compute_ohem_loss(self.l1_loss, aggregated, hr, ratio=ohem_ratio)
                    if debug_mode and batch_idx < 3:
                        print(f"  [OHEM] threshold={ohem_info['threshold']:.4f}, max_loss={ohem_info['max_loss']:.4f}")
                else:
                    loss = self.l1_loss(aggregated, hr)
                
                # Add anime-specific losses (Stage 1)
                loss_dict = {'l1': loss.item()}
                
                if self.line_art_loss is not None:
                    line_result = self.line_art_loss(aggregated, hr)
                    line_loss = line_result['line_art']  # Extract main loss from dict
                    loss = loss + line_loss
                    loss_dict['line_art'] = line_loss.item()
                
                if self.color_loss is not None:
                    color_result = self.color_loss(aggregated, hr)
                    color_loss = color_result['color_consistency']  # Extract main loss from dict
                    loss = loss + color_loss
                    loss_dict['color'] = color_loss.item()
                
                if self.flat_loss is not None:
                    flat_result = self.flat_loss(aggregated, hr)
                    flat_loss = flat_result['flat_preservation']  # Extract main loss from dict
                    loss = loss + flat_loss
                    loss_dict['flat'] = flat_loss.item()
                
                # Add perceptual loss (DISTS/VGG)
                if self.perceptual_loss is not None:
                    p_loss = self.perceptual_loss(aggregated, hr)
                    
                    # Only add perceptual loss if it's valid
                    if torch.isnan(p_loss).any() or torch.isinf(p_loss).any():
                        if debug_mode and batch_idx < 3:
                            print(f"  [WARNING] NaN/Inf in perceptual loss, skipping for this batch")
                        # Don't add NaN loss - continue with other losses
                    else:
                        loss = loss + self.perceptual_weight * p_loss
                        loss_dict['perceptual'] = p_loss.item()

                if debug_mode and batch_idx < 3:
                    loss_str = "  [Loss] " + " | ".join([f"{k}={v:.4f}" for k, v in loss_dict.items()])
                    print(loss_str)
                
                # Scale loss for gradient accumulation
                if accumulation_steps > 1:
                    loss = loss / accumulation_steps

            # Check for NaN in loss
            if torch.isnan(loss) or torch.isinf(loss):
                if debug_mode:
                    logger.error(f"\n*** NaN/Inf DETECTED at batch {batch_idx} ***")
                    # Run deep forward analysis (if available)
                    if hasattr(self.model, 'debug_forward'):
                        logger.info("\nRunning deep forward analysis...")
                        with torch.no_grad():
                            debug_info = self.model.debug_forward(teacher_outputs)
                        # Check which layer first produced NaN
                        logger.info("\nLayer-by-layer analysis:")
                        for key, tensor in debug_info.items():
                            has_nan = torch.isnan(tensor).any().item()
                            has_inf = torch.isinf(tensor).any().item()
                            status = "CLEAN" if not (has_nan or has_inf) else f"{'NaN' if has_nan else ''}{' Inf' if has_inf else ''}"
                            logger.info(f"  {key}: {status} | min={tensor.min():.4f} max={tensor.max():.4f}")
                    else:
                        logger.warning("[Note: Model does not have debug_forward method]")
                    # Also check model parameters
                    logger.info("\nParameter analysis:")
                    for name, param in self.model.named_parameters():
                        if torch.isnan(param).any() or torch.isinf(param).any():
                            logger.error(f"  {name}: NaN/Inf found! shape={param.shape}")
                        elif param.abs().max() > 100:
                            logger.warning(f"  {name}: WARNING large values (max={param.abs().max().item():.2f})")
                    logger.info("\n*** END DEBUG ***\n")
                else:
                    logger.warning(f"Warning: NaN/Inf loss at batch {batch_idx}, skipping...")
                continue

            # Backward with gradient clipping
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Only update weights after accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(optimizer)
                
                # Configurable gradient clipping
                max_grad_norm = self.config.get('training', {}).get('max_grad_norm', 1.0)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                
                if debug_mode and batch_idx < 3:
                    print(f"  [Grad norm] {grad_norm:.4f}")
                
                if self.use_amp:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()

            # Scale loss back for reporting
            report_loss = loss.item() * accumulation_steps
            total_loss += report_loss
            total_l1 += loss_dict.get('l1', 0) * accumulation_steps
            total_line_art += loss_dict.get('line_art', 0) * accumulation_steps
            total_color += loss_dict.get('color', 0) * accumulation_steps
            total_flat += loss_dict.get('flat', 0) * accumulation_steps
            num_batches += 1
            self.global_step += 1
            has_scaled_batch = True  # Mark that we successfully scaled this batch

            # Update progress bar with detailed losses
            postfix_dict = {'loss': f"{report_loss:.4f}", 'l1': f"{loss_dict.get('l1', 0):.4f}"}
            if 'line_art' in loss_dict:
                postfix_dict['line'] = f"{loss_dict['line_art']:.4f}"
            if 'color' in loss_dict:
                postfix_dict['color'] = f"{loss_dict['color']:.4f}"
            pbar.set_postfix(postfix_dict)

            # Log
            if batch_idx % self.log_interval == 0:
                self.log_metrics(loss_dict, self.global_step, 'stage1_train')

        # Handle any remaining accumulated gradients (only if we have valid gradients)
        if num_batches > 0 and has_scaled_batch and (batch_idx + 1) % accumulation_steps != 0:
            if self.use_amp:
                self.scaler.unscale_(optimizer)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        if num_batches == 0:
            print("Warning: No training batches in epoch (dataset may be too small)")
            return {'loss': 0.0, 'l1': 0.0, 'line_art': 0.0, 'color': 0.0, 'flat': 0.0}
        
        avg_loss = total_loss / num_batches
        return {
            'loss': avg_loss,
            'l1': total_l1 / num_batches,
            'line_art': total_line_art / num_batches,
            'color': total_color / num_batches,
            'flat': total_flat / num_batches,
        }
    
    def train_stage2_epoch(self, dataloader, optimizer) -> Dict[str, float]:
        """
        Train SPAN student with MTKD + FAKD for one epoch.
        """
        self.model.train()
        
        if not self.teachers:
            self.load_teachers()
            if not self.teachers:
                raise RuntimeError("No teacher models could be loaded. Check teacher paths in config.")
        
        if self.knowledge_aggregation is None and not getattr(self, '_ka_load_attempted', False):
            self._ka_load_attempted = True  # Remember we tried to load

            # Check if loading is enabled in config
            stage2_cfg = self.config.get('training', {}).get('stage2', {})
            if not stage2_cfg.get('use_stage1_checkpoint', False):
                print("  Knowledge Aggregation loading disabled (use_stage1_checkpoint: false)")
                print("  Stage 2 will use first teacher only")
            else:
                # Load frozen Stage 1 model
                stage1_path = stage2_cfg.get('stage1_checkpoint')
                if stage1_path:
                    print(f"Loading Knowledge Aggregation from {stage1_path}")
                    try:
                        # Determine which aggregation class to use (check metadata or config)
                        use_simple = stage2_cfg.get('use_simple_aggregation', True)  # Default to simple for stability
                        aggregation_class = SimpleKnowledgeAggregation if use_simple else KnowledgeAggregationNetwork
                        model_name = "SimpleKnowledgeAggregation" if use_simple else "KnowledgeAggregationNetwork"
                        print(f"  Using {model_name}")

                        # Get architecture params from config
                        num_blocks = stage2_cfg.get('num_blocks', 3)
                        embed_dim = stage2_cfg.get('embed_dim', 64)

                        # Create aggregation network with correct parameters
                        self.knowledge_aggregation = aggregation_class(
                            num_teachers=len(self.teachers) if self.teachers else 3,
                            embed_dim=embed_dim,
                            num_blocks=num_blocks,
                            scale=self.config.get('model', {}).get('scale', 4),
                        ).to(self.device)

                        # Load checkpoint
                        try:
                            checkpoint = torch.load(stage1_path, map_location=self.device, weights_only=True)
                        except UnpicklingError as e:
                            logger.warning(f"Stage 1 checkpoint requires pickle ({e}). Only load from trusted sources")
                            checkpoint = torch.load(stage1_path, map_location=self.device, weights_only=False)
                        self.knowledge_aggregation.load_state_dict(checkpoint['model_state_dict'])

                        # Freeze and set to eval mode
                        self.knowledge_aggregation.eval()
                        for param in self.knowledge_aggregation.parameters():
                            param.requires_grad = False

                        print(f"[OK] Loaded frozen Knowledge Aggregation from epoch {checkpoint.get('epoch', 'unknown')}")
                    except Exception as e:
                        print(f"[ERROR] Failed to load Knowledge Aggregation: {e}")
                        print("  Stage 2 will use first teacher only")
                        self.knowledge_aggregation = None
        
        total_loss = 0
        total_l1 = 0
        total_wavelet = 0
        total_fakd = 0
        total_perceptual = 0
        num_batches = 0
        
        # Gradient accumulation setup
        accumulation_steps = self.config.get('training', {}).get('gradient_accumulation_steps', 1)
        has_scaled_batch = False
        
        pbar = tqdm(dataloader, desc=f"Stage 2 Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            lr = batch['lr'].to(self.device, non_blocking=True) if batch['lr'] is not None else None
            hr = batch['hr'].to(self.device, non_blocking=True)
            
            # Apply GPU degradation if lr is None (indicates gpu_degradation mode)
            if lr is None:
                scale = self.config.get('model', {}).get('scale', 4)
                lr = self._apply_gpu_degradation(hr, scale)
            
            # Handle temporal data: reshape [B, T, C, H, W] -> [B*T, C, H, W]
            is_temporal = lr.dim() == 5
            if is_temporal:
                # LR and HR have different spatial dims, get shapes separately
                B_lr, T_lr, C_lr, H_lr, W_lr = lr.shape
                B_hr, T_hr, C_hr, H_hr, W_hr = hr.shape
                lr = lr.view(B_lr * T_lr, C_lr, H_lr, W_lr)
                hr = hr.view(B_hr * T_hr, C_hr, H_hr, W_hr)
            
            # Zero gradients only at start of accumulation cycle
            if batch_idx % accumulation_steps == 0:
                optimizer.zero_grad()
            
            with autocast('cuda', enabled=self.use_amp):
                # Student forward
                if hasattr(self.model, 'forward_with_features'):
                    student_pred, student_features = self.model.forward_with_features(lr)
                else:
                    student_pred = self.model(lr)
                    student_features = {}
                
                # Get teacher predictions and features (if available)
                with torch.no_grad():
                    aggregated = None
                    
                    if self.teachers and len(self.teachers) > 0:
                        # Extract both outputs and intermediate features
                        teacher_outputs, teacher_features = self._extract_teacher_features(lr)
                        
                        if self.knowledge_aggregation is not None:
                            aggregated = self.knowledge_aggregation(teacher_outputs)
                        else:
                            # Fallback: use first teacher
                            aggregated = teacher_outputs[0] if teacher_outputs else None
                    else:
                        teacher_features = {}
                
                # Compute losses
                l1 = self.l1_loss(student_pred, hr)

                loss = l1
                loss_dict = {'l1': l1.item()}

                # MTKD wavelet loss (only if teacher/aggregation available)
                if self.mtkd_loss is not None and aggregated is not None:
                    mtkd_loss, mtkd_dict = self.mtkd_loss(student_pred, aggregated, hr)
                    loss = loss + mtkd_loss
                    loss_dict.update({k: v.item() if isinstance(v, torch.Tensor) else v for k, v in mtkd_dict.items()})

                # FAKD feature affinity loss (now with actual teacher features)
                if self.fakd_loss is not None and len(student_features) > 0 and len(teacher_features) > 0:
                    fakd = self.fakd_loss(student_features, teacher_features)
                    fakd_weight = self.config.get('loss', {}).get('distillation', {}).get('fakd_weight', 0.5)
                    loss = loss + fakd_weight * fakd
                    loss_dict['fakd'] = fakd.item() if isinstance(fakd, torch.Tensor) else fakd
                    
                    # Debug: Log when FAKD is active
                    if num_batches == 0 and self.current_epoch == 0:
                        print(f"  FAKD active: student_layers={list(student_features.keys())}, "
                              f"teacher_layers={list(teacher_features.keys())}")
                
                # Add anime-specific losses (Stage 2)
                if self.line_art_loss is not None:
                    line_result = self.line_art_loss(student_pred, hr)
                    line_loss = line_result['line_art']  # Extract main loss from dict
                    loss = loss + line_loss
                    loss_dict['line_art'] = line_loss.item()
                
                if self.color_loss is not None:
                    color_result = self.color_loss(student_pred, hr)
                    color_loss = color_result['color_consistency']  # Extract main loss from dict
                    loss = loss + color_loss
                    loss_dict['color'] = color_loss.item()
                
                if self.flat_loss is not None:
                    flat_result = self.flat_loss(student_pred, hr)
                    flat_loss = flat_result['flat_preservation']  # Extract main loss from dict
                    loss = loss + flat_loss
                    loss_dict['flat'] = flat_loss.item()
                
                # Add perceptual loss (DISTS/VGG)
                if self.perceptual_loss is not None:
                    p_loss = self.perceptual_loss(student_pred, hr)
                    
                    # Debug: Check for NaN in perceptual loss
                    if torch.isnan(p_loss).any() or torch.isinf(p_loss).any():
                        print(f"\n[WARNING] NaN/Inf in perceptual loss!")
                        print(f"  pred range: [{student_pred.min():.4f}, {student_pred.max():.4f}]")
                        print(f"  hr range: [{hr.min():.4f}, {hr.max():.4f}]")
                    else:
                        loss = loss + self.perceptual_weight * p_loss
                        loss_dict['perceptual'] = p_loss.item()
                
                # Temporal consistency for video
                if self.temporal_loss is not None:
                    temporal_dict = self.temporal_loss(student_pred, hr)
                    if 'temporal' in temporal_dict:
                        loss = loss + temporal_dict['temporal']
                        loss_dict['temporal'] = temporal_dict['temporal'].item()

                loss_dict['total'] = loss.item()
            
            # NaN/Inf detection before backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf detected in Stage 2 loss at batch {batch_idx}, skipping")
                optimizer.zero_grad()
                continue
            
            # Backward with gradient accumulation
            loss = loss / accumulation_steps
            has_scaled_batch = True
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Step optimizer only at end of accumulation cycle
            if (batch_idx + 1) % accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.get('training', {}).get('max_grad_norm', 1.0))
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.get('training', {}).get('max_grad_norm', 1.0))
                    optimizer.step()
                
                # Update EMA after optimizer step
                self._update_ema()
            
            total_loss += loss_dict['total']
            total_l1 += loss_dict.get('l1', 0)
            total_wavelet += loss_dict.get('wavelet_distill', 0)
            total_fakd += loss_dict.get('fakd', 0)
            total_perceptual += loss_dict.get('perceptual', 0)
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar with all losses
            postfix = {
                'loss': f"{loss_dict['total']:.4f}",
                'l1': f"{loss_dict.get('l1', 0):.4f}",
            }
            if 'perceptual' in loss_dict:
                postfix['perc'] = f"{loss_dict['perceptual']:.4f}"
            if 'fakd' in loss_dict:
                postfix['fakd'] = f"{loss_dict['fakd']:.4f}"
            if 'wavelet_distill' in loss_dict:
                postfix['wav'] = f"{loss_dict['wavelet_distill']:.4f}"
            pbar.set_postfix(postfix)
            
            # Log
            if batch_idx % self.log_interval == 0:
                self.log_metrics(loss_dict, self.global_step, 'stage2_train')
        
        # Handle any remaining accumulated gradients
        if num_batches > 0 and has_scaled_batch and (batch_idx + 1) % accumulation_steps != 0:
            if self.use_amp:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.get('training', {}).get('max_grad_norm', 1.0))
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.get('training', {}).get('max_grad_norm', 1.0))
                optimizer.step()
            optimizer.zero_grad()
        
        if num_batches == 0:
            print("Warning: No training batches processed in Stage 2")
            return {'loss': 0.0, 'l1': 0.0, 'wavelet': 0.0, 'fakd': 0.0}
        
        # Build return dict with all tracked losses
        result = {
            'loss': total_loss / num_batches,
            'l1': total_l1 / num_batches,
            'wavelet': total_wavelet / num_batches,
            'fakd': total_fakd / num_batches,
        }
        if total_perceptual > 0:
            result['perceptual'] = total_perceptual / num_batches
        return result
    
    def train_epoch(self, dataloader, optimizer, loss_fn=None) -> Dict[str, float]:
        """Route to appropriate stage training"""
        if self.stage == 1:
            return self.train_stage1_epoch(dataloader, optimizer)
        else:
            return self.train_stage2_epoch(dataloader, optimizer)
    
    def validate(self, dataloader) -> Dict[str, float]:
        """Validation - handles both Stage 1 (Knowledge Aggregation) and Stage 2 (SPAN student)"""
        self.model.eval()

        total_psnr = 0
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation'):
                lr = batch['lr'].to(self.device, non_blocking=True) if batch['lr'] is not None else None
                hr = batch['hr'].to(self.device, non_blocking=True)
                
                # Apply GPU degradation if lr is None (indicates gpu_degradation mode)
                if lr is None:
                    scale = self.config.get('model', {}).get('scale', 4)
                    lr = self._apply_gpu_degradation(hr, scale)
                
                # Handle temporal data: reshape [B, T, C, H, W] -> [B*T, C, H, W]
                is_temporal = lr.dim() == 5
                if is_temporal:
                    B_lr, T_lr, C_lr, H_lr, W_lr = lr.shape
                    B_hr, T_hr, C_hr, H_hr, W_hr = hr.shape
                    lr = lr.view(B_lr * T_lr, C_lr, H_lr, W_lr)
                    hr = hr.view(B_hr * T_hr, C_hr, H_hr, W_hr)

                # Forward based on current stage
                has_teachers = self.teachers is not None and len(self.teachers) > 0
                if self.use_amp:
                    with autocast('cuda'):
                        if self.stage == 1 and has_teachers:
                            # Stage 1 with teachers: Get teacher outputs first, then aggregate
                            teacher_outputs = []
                            for teacher in self.teachers:
                                to = teacher(lr)
                                # Check for NaN/Inf
                                if torch.isnan(to).any() or torch.isinf(to).any():
                                    continue
                                teacher_outputs.append(to)
                            
                            # Pad with zeros if some teachers failed
                            expected_teachers = getattr(self.model, 'num_teachers', len(self.teachers))
                            scale = self.config.get('model', {}).get('scale', 4)
                            while len(teacher_outputs) < expected_teachers:
                                zero_teacher = torch.zeros_like(teacher_outputs[0]) if teacher_outputs else torch.zeros(lr.size(0), 3, lr.size(2) * scale, lr.size(3) * scale, device=lr.device)
                                teacher_outputs.append(zero_teacher)
                            
                            pred = self.model(teacher_outputs)
                        else:
                            # Stage 2 or no teachers: SPAN student takes LR directly
                            pred = self.model(lr)
                else:
                    if self.stage == 1 and has_teachers:
                        # Stage 1 with teachers: Get teacher outputs first, then aggregate
                        teacher_outputs = []
                        for teacher in self.teachers:
                            to = teacher(lr)
                            # Check for NaN/Inf
                            if torch.isnan(to).any() or torch.isinf(to).any():
                                continue
                            teacher_outputs.append(to)
                        
                        # Pad with zeros if some teachers failed
                        expected_teachers = getattr(self.model, 'num_teachers', len(self.teachers))
                        scale = self.config.get('model', {}).get('scale', 4)
                        while len(teacher_outputs) < expected_teachers:
                            zero_teacher = torch.zeros_like(teacher_outputs[0]) if teacher_outputs else torch.zeros(lr.size(0), 3, lr.size(2) * scale, lr.size(3) * scale, device=lr.device)
                            teacher_outputs.append(zero_teacher)
                        
                        pred = self.model(teacher_outputs)
                    else:
                        # Stage 2 or no teachers: SPAN student takes LR directly
                        pred = self.model(lr)

                # Compute metrics
                loss = self.l1_loss(pred, hr)

                # NaN/Inf protection for predictions
                if torch.isnan(pred).any() or torch.isinf(pred).any():
                    continue

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
        """Full training loop with both stages"""
        
        # Wrap train_loader with async prefetcher if enabled
        use_prefetcher = self.config.get('training', {}).get('use_async_prefetcher', False)
        if use_prefetcher and torch.cuda.is_available():
            from anime_sr.data.async_prefetcher import create_prefetcher
            train_loader = create_prefetcher(train_loader, self.config, self.device)

        # Load teachers if available (needed for both Stage 1 and Stage 2)
        stage1_cfg = self.config.get('training', {}).get('stage1', {})
        teachers_cfg = stage1_cfg.get('teachers', [])
        if teachers_cfg and not self.teachers:
            self.load_teachers()
            if not self.teachers:
                raise RuntimeError("No teacher models could be loaded. Check teacher paths in config.")

        # Stage 1: Knowledge Aggregation (or simple SPAN training if no teachers)
        if self.stage1_epochs > 0:
            # Check if we have teachers loaded
            if not self.teachers:
                # Train SPAN directly with pixel loss (demo/basic mode)
                print("\n" + "="*50)
                print("Stage 1: Training SPAN (pixel loss only, no teachers)")
                print("="*50)
                
                self.stage = 1
                optimizer = self._create_optimizer()
                self.set_scheduler(self._create_scheduler(optimizer, stage='stage1'))
                optimizer._step_count = 0
                
                if self.callbacks is not None:
                    self.callbacks.on_train_begin()

                for epoch in range(self.stage1_epochs):
                    self.current_epoch = epoch
                    
                    if self.callbacks is not None:
                        self.callbacks.on_epoch_begin(epoch)

                    # Train with pixel loss only
                    self.model.train()
                    epoch_loss = 0.0
                    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.stage1_epochs}")
                    for batch in pbar:
                        lr = batch['lr'].to(self.device)
                        hr = batch['hr'].to(self.device)
                        
                        optimizer.zero_grad()
                        sr = self.model(lr)
                        loss = self.l1_loss(sr, hr)
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
                    avg_loss = epoch_loss / len(train_loader)
                    print(f"Epoch {epoch}: loss={avg_loss:.4f}")

                    # Validate
                    val_metrics = {}
                    if val_loader is not None and epoch % self.val_interval == 0:
                        val_metrics = self.validate(val_loader)
                        print(f"Validation: loss={val_metrics['loss']:.4f}, PSNR={val_metrics['psnr']:.2f}")
                        self.log_metrics(val_metrics, self.global_step, 'stage1_val')

                        if val_metrics['loss'] < self.best_loss:
                            self.best_loss = val_metrics['loss']
                            self.save_checkpoint(epoch, optimizer, is_best=True)

                    if epoch % self.save_interval == 0:
                        self.save_checkpoint(epoch, optimizer)

                    if self.scheduler is not None:
                        self.scheduler.step()

                    self.global_step += len(train_loader)

                    if self.callbacks is not None:
                        self.callbacks.on_epoch_end(epoch, {'loss': avg_loss, **val_metrics})

                if self.callbacks is not None:
                    self.callbacks.on_train_end()
                    
                print("Stage 1 complete (SPAN pixel loss training)")
            else:
                print("\n" + "="*50)
                print("Stage 1: Training Knowledge Aggregation")
                print("="*50)
                
                self.stage = 1
                optimizer = self._create_optimizer()
                self.set_scheduler(self._create_scheduler(optimizer, stage='stage1'))  # Use set_scheduler for deferred loading
                optimizer._step_count = 0  # Reset to prevent lr_scheduler warning
                
                # Initialize callbacks for Stage 1
                if self.callbacks is not None:
                    self.callbacks.on_train_begin()

                for epoch in range(self.stage1_epochs):
                    self.current_epoch = epoch
                    
                    # Callback: epoch begin
                    if self.callbacks is not None:
                        self.callbacks.on_epoch_begin(epoch)

                    # Train
                    train_metrics = self.train_stage1_epoch(train_loader, optimizer)
                    print(f"Epoch {epoch}: loss={train_metrics['loss']:.4f}")

                    # Validate
                    val_metrics = {}
                    if val_loader is not None and epoch % self.val_interval == 0:
                        val_metrics = self.validate(val_loader)
                        print(f"Validation: loss={val_metrics['loss']:.4f}, PSNR={val_metrics['psnr']:.2f}")
                        self.log_metrics(val_metrics, self.global_step, 'stage1_val')

                        # Save best
                        if val_metrics['loss'] < self.best_loss:
                            self.best_loss = val_metrics['loss']
                            self.save_checkpoint(epoch, optimizer, is_best=True)

                    # Save checkpoint
                    if epoch % self.save_interval == 0:
                        self.save_checkpoint(epoch, optimizer)

                    if self.scheduler is not None:
                        self.scheduler.step()
                    
                    # Callback: epoch end with metrics
                    if self.callbacks is not None:
                        # Combine train and val metrics for callbacks
                        combined_metrics = {**train_metrics, **val_metrics}
                        self.callbacks.on_epoch_end(epoch, combined_metrics)
                        
                        # Check early stopping
                        if self.callbacks.should_stop():
                            print(f"\nEarly stopping triggered at epoch {epoch}")
                            break
                
                # Callback: train end
                if self.callbacks is not None:
                    self.callbacks.on_train_end({'stage': 1, 'best_loss': self.best_loss})
                
                print(f"\nStage 1 complete. Best loss: {self.best_loss:.4f}\n")
        
        # Stage 2: Student Distillation
        if self.stage2_epochs > 0:
            print(f"\n{'='*50}")
            print(f"Stage 2: Training SPAN Student with MTKD + FAKD")
            print(f"{'='*50}\n")
            
            # Recreate model as SPAN student
            self.stage = 2
            self.model = self._create_model(self.config).to(self.device)
            self.best_loss = float('inf')
            
            # Reinitialize EMA model for new architecture
            if self.use_ema:
                self.ema_model = self._create_ema_model()
            
            optimizer = self._create_optimizer()
            self.set_scheduler(self._create_scheduler(optimizer, stage='stage2'))  # Use set_scheduler for deferred loading
            optimizer._step_count = 0  # Reset to prevent lr_scheduler warning
            
            # Re-initialize callbacks for Stage 2 (reset early stopping state)
            self._setup_callbacks()
            if self.callbacks is not None:
                self.callbacks.on_train_begin()

            for epoch in range(self.stage2_epochs):
                self.current_epoch = epoch
                
                # Callback: epoch begin
                if self.callbacks is not None:
                    self.callbacks.on_epoch_begin(epoch)
                
                # Train
                train_metrics = self.train_stage2_epoch(train_loader, optimizer)
                
                # Verbose epoch summary
                metric_str = f"Epoch {epoch}: loss={train_metrics['loss']:.4f}, l1={train_metrics['l1']:.4f}"
                if 'perceptual' in train_metrics:
                    metric_str += f", perc={train_metrics['perceptual']:.4f}"
                if 'wavelet' in train_metrics:
                    metric_str += f", wav={train_metrics['wavelet']:.4f}"
                if 'fakd' in train_metrics:
                    metric_str += f", fakd={train_metrics['fakd']:.4f}"
                print(metric_str)
                
                # Validate
                val_metrics = {}
                if val_loader is not None and epoch % self.val_interval == 0:
                    val_metrics = self.validate(val_loader)
                    print(f"Validation: loss={val_metrics['loss']:.4f}, PSNR={val_metrics['psnr']:.2f}")
                    self.log_metrics(val_metrics, self.global_step, 'stage2_val')
                    
                    if val_metrics['loss'] < self.best_loss:
                        self.best_loss = val_metrics['loss']
                        self.save_checkpoint(epoch, optimizer, is_best=True)
                
                # Save checkpoint
                if epoch % self.save_interval == 0:
                    self.save_checkpoint(epoch, optimizer)
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Callback: epoch end with metrics
                if self.callbacks is not None:
                    combined_metrics = {**train_metrics, **val_metrics}
                    self.callbacks.on_epoch_end(epoch, combined_metrics)
                    
                    # Check early stopping
                    if self.callbacks.should_stop():
                        print(f"\nEarly stopping triggered at epoch {epoch}")
                        break
            
            # Callback: train end
            if self.callbacks is not None:
                self.callbacks.on_train_end({'stage': 2, 'best_loss': self.best_loss})

            # Save final Stage 2 checkpoint
            final_epoch = self.current_epoch
            self.save_checkpoint(final_epoch, optimizer)
            print(f"\nStage 2 complete. Best loss: {self.best_loss:.4f}\n")
        
        print("Training complete!")

    def save_checkpoint(self, epoch: int, optimizer, is_best: bool = False, metrics: Dict = None):
        """Override to dual-save Stage 1 best checkpoint."""
        # Call parent save_checkpoint (includes full state: EMA, SWA, scaler, callbacks)
        super().save_checkpoint(epoch, optimizer, is_best, metrics)
        
        # Dual-save: also save as model_a_stage1_best.pth during Stage 1 when is_best
        # Include full state for proper resumption
        if is_best and self.stage == 1 and hasattr(self, 'stage'):
            stage1_best_path = self.checkpoint_dir / "model_a_stage1_best.pth"
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else {},
                'best_loss': self.best_loss,
                'metrics': metrics or {},
                'stage': 1,
                'aggregation_type': 'simple' if isinstance(self.model, SimpleKnowledgeAggregation) else 'complex',
            }
            
            # Include EMA state if available
            if self.ema_model is not None:
                checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()
            
            # Include SWA state if available
            if self.swa_model is not None:
                checkpoint['swa_model_state_dict'] = self.swa_model.state_dict()
            
            # Include scaler state if available
            if self.scaler is not None:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
            torch.save(checkpoint, stage1_best_path)
            print(f"  Also saved as model_a_stage1_best.pth (full state)")
