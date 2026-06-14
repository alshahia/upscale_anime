"""
Fine-tuning trainer for neosr SPAN models.
Supports loading pretrained neosr weights and fine-tuning on custom data.
Enhanced with multi-loss training (perceptual, anime-specific, adversarial).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pickle import UnpicklingError
import logging
import os
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional

try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast as _autocast
    def autocast(device=None, enabled=True):
        return _autocast(enabled=enabled)
    try:
        from torch.cuda.amp import GradScaler as _GradScaler
        class GradScaler:  # minimal shim for PyTorch<2
            def __init__(self, device='cuda', enabled=True):
                self._inner = _GradScaler(enabled=enabled)
            def scale(self, loss):
                return self._inner.scale(loss)
            def unscale_(self, optimizer):
                self._inner.unscale_(optimizer)
            def step(self, optimizer):
                self._inner.step(optimizer)
            def update(self):
                self._inner.update()
            def state_dict(self):
                return self._inner.state_dict()
            def load_state_dict(self, sd):
                self._inner.load_state_dict(sd)
    except ImportError:
        class GradScaler:  # CPU / no-AMP fallback
            def __init__(self, device='cuda', enabled=True):
                self._enabled = bool(enabled)
            def scale(self, loss):
                return loss
            def unscale_(self, optimizer):
                return None
            def step(self, optimizer):
                optimizer.step()
            def update(self):
                return None
            def state_dict(self):
                return {}
            def load_state_dict(self, sd):
                return None

logger = logging.getLogger(__name__)

try:
    from src.models.span.neosr_span import create_neosr_span
    from src.utils.checkpoint_loader import load_pretrained_weights
except ImportError:
    from models.span.neosr_span import create_neosr_span
    from utils.checkpoint_loader import load_pretrained_weights


class NeosrSPANFinetuner:
    """Fine-tuning trainer for neosr SPAN models with multi-loss support."""

    def __init__(self, config: Dict, device: str = 'cuda', checkpoint_manager=None):
        # Phase 2 #8: read config values that are needed BEFORE _setup_losses
        # runs (they're used in startup print blocks). Extracted into
        # _read_config_early() so the ordering is explicit and testable.
        self.config = config
        self.device = device
        self.finetune_cfg = config.get('training', {}).get('finetune', {})
        self.checkpoint_manager = checkpoint_manager

        # Set the early-needed attributes first so anything downstream
        # (e.g. print blocks, the loss-construction paths) can rely on them.
        self._read_config_early()

        # Setup checkpoint directory from manager or fallback
        if checkpoint_manager:
            self.checkpoint_dir = checkpoint_manager.get_checkpoint_dir()
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if not checkpoint_manager.is_resumed():
                checkpoint_manager.register_run()
        else:
            self.checkpoint_dir = Path(config.get('paths', {}).get('checkpoint_dir',
                                       config.get('training', {}).get('checkpoint_dir', 'checkpoints')))
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create model
        model_cfg = config.get('model', config)
        self.model = create_neosr_span(model_cfg)

        # Load pretrained weights
        pretrained_path = self.finetune_cfg.get('pretrained_path') or model_cfg.get('pretrained_path')
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"\n[Pretrained] Loading weights from {pretrained_path}")
            result = self.model.load_neosr_weights(pretrained_path, strict=False)
            print(f"  Loaded: {result['loaded']}/{result['total_model_keys']} weights")
            if result['skipped']:
                print(f"  Skipped: {result['skipped']}")
            if result['mismatched']:
                print(f"  Mismatched: {result['mismatched']}")
        else:
            print("\n[Pretrained] No pretrained weights found, training from scratch")

        self.model = self.model.to(device)

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True

        use_compile = self.finetune_cfg.get('torch_compile', False)
        if use_compile and hasattr(torch, 'compile'):
            print(f"[torch.compile] Compiling model with mode='reduce-overhead'")
            self.model = torch.compile(self.model, mode="reduce-overhead")

        use_checkpointing = self.finetune_cfg.get('gradient_checkpointing', False)
        if use_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            print(f"[Gradient Checkpointing] Enabled for memory saving")
            self.model.gradient_checkpointing_enable()

        # Freeze layers if specified
        freeze_layers = self.finetune_cfg.get('freeze_layers', [])
        if freeze_layers:
            self._freeze_layers(freeze_layers)

        # Setup optimizer
        # NOTE: self.lr is owned by _read_config_early().
        self.use_amp = self.config.get('training', {}).get('mixed_precision', True) and torch.cuda.is_available()
        adam_eps = 1e-7 if self.use_amp else 1e-8
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=1e-4,
            eps=adam_eps,
        )

        # Setup GradScaler for AMP (generator + content losses only).
        # Discriminator step stays in FP32 and does NOT use the scaler
        # (it is already wrapped in torch.amp.autocast('cuda', enabled=False)).
        self.scaler = GradScaler('cuda', enabled=self.use_amp)

        # Setup scheduler
        # NOTE: self.epochs is owned by _read_config_early().
        self.warmup_epochs = self.finetune_cfg.get('warmup_epochs', 5)
        self.min_lr = self.finetune_cfg.get('min_lr', 1e-5)
        self.scheduler = self._create_scheduler()

        # Phase D-EMA: Generator EMA. Gated by training.use_ema (default True
        # via base.yaml). Decay default 0.999, also from base.yaml. The
        # shadow is used for validation/inference to stabilize quality
        # over the noisy per-step weights.
        self.use_ema = bool(self.config.get('training', {}).get('use_ema', True))
        self.ema_decay = float(self.config.get('training', {}).get('ema_decay', 0.999))
        self.ema = None
        if self.use_ema:
            try:
                from training.ema import GeneratorEMA
                self.ema = GeneratorEMA(self.model, decay=self.ema_decay)
                # Match the live model's device (handles CPU/GPU moves).
                self.ema.ema_model = self.ema.ema_model.to(self.device)
                print(f"[EMA] Generator EMA enabled (decay={self.ema_decay})")
            except Exception as e:
                logger.warning("Failed to initialize GeneratorEMA: %s", e)
                self.ema = None

        # Setup multi-loss system
        self._setup_losses()

        # Training state
        self.best_loss = float('inf')
        self.current_epoch = 0
        self.global_step = 0
        self.val_interval = self.finetune_cfg.get('val_interval', 5)
        self.save_interval = self.finetune_cfg.get('save_interval', 10)
        self.gradient_accumulation = self.finetune_cfg.get('gradient_accumulation_steps', 1)

        # TensorBoard logging
        self.use_tensorboard = config.get('training', {}).get('use_tensorboard', False)
        self.writer = None
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = Path(config.get('paths', {}).get('log_dir', 'logs')) / 'finetune'
                log_dir.mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(log_dir=str(log_dir))
                print(f"[TensorBoard] Logging to {log_dir}")
            except ImportError:
                print("[WARNING] TensorBoard not available. Install with: pip install tensorboard")
                self.use_tensorboard = False

        # Early stopping
        es_cfg = self.finetune_cfg.get('early_stopping') or {}
        self.use_early_stopping = es_cfg.get('enabled', False)
        self.early_stopping_metric = es_cfg.get('metric', 'lpips')
        self.early_stopping_patience = es_cfg.get('patience', 20)
        self.early_stopping_mode = es_cfg.get('mode', 'min')
        self.best_metric_value = float('inf') if self.early_stopping_mode == 'min' else float('-inf')
        self.early_stopping_counter = 0
        self.should_stop = False

        # Two-phase training configuration
        two_phase_cfg = self.finetune_cfg.get('two_phase_training') or {}
        self.use_two_phase = two_phase_cfg.get('enabled', False)
        # NOTE: self.phase1_epochs and self.phase_adversarial_ramp_epochs are
        # owned by _read_config_early().

        phase1_weights = two_phase_cfg.get('phase1') or {}
        self.phase1_pixel_weight = phase1_weights.get('pixel_weight', 1.0)
        self.phase1_perceptual_weight = phase1_weights.get('perceptual_weight', 0.1)
        self.phase1_adversarial_weight = phase1_weights.get('adversarial_weight', 0.0)
        self.phase1_line_art_weight = phase1_weights.get('line_art_weight', 1.0)
        self.phase1_flat_weight = phase1_weights.get('flat_weight', 0.05)
        self.phase1_frequency_weight = phase1_weights.get('frequency_weight', 0.025)

        phase2_weights = two_phase_cfg.get('phase2') or {}
        self.phase2_pixel_weight = phase2_weights.get('pixel_weight', 0.6)
        self.phase2_perceptual_weight = phase2_weights.get('perceptual_weight', 0.12)
        self.phase2_adversarial_weight = phase2_weights.get('adversarial_weight', 0.0001)
        self.phase2_line_art_weight = phase2_weights.get('line_art_weight', 1.0)
        self.phase2_flat_weight = phase2_weights.get('flat_weight', 0.05)
        self.phase2_frequency_weight = phase2_weights.get('frequency_weight', 0.075)

        if self.use_two_phase:
            print(f"[Two-Phase Training] Enabled (Phase1: epochs 1-{self.phase1_epochs}, Phase2: epochs {self.phase1_epochs+1}-{self.epochs})")
            print(f"  Phase1: pixel={self.phase1_pixel_weight}, perc={self.phase1_perceptual_weight}, adv={self.phase1_adversarial_weight} (starts epoch {self.adversarial_start_epoch}), line_art={self.phase1_line_art_weight}, flat={self.phase1_flat_weight}, freq={self.phase1_frequency_weight}")
            print(f"  Phase2: pixel={self.phase2_pixel_weight}, perc={self.phase2_perceptual_weight}, adv={self.phase2_adversarial_weight}, line_art={self.phase2_line_art_weight}, flat={self.phase2_flat_weight}, freq={self.phase2_frequency_weight}")
            print(f"  Adversarial ramp: {self.phase_adversarial_ramp_epochs} epochs after start_epoch")

        # Progressive crop configuration
        progressive_cfg = self.finetune_cfg.get('progressive_crop') or {}
        self.use_progressive_crop = progressive_cfg.get('enabled', False)

        if self.use_progressive_crop:
            # Phase 4.3: pull default stages from base.yaml when not specified.
            pc_defaults = self.config.get('progressive_crop_defaults') or {}
            default_stages = pc_defaults.get('stages', [
                {'crop_size': 128, 'epochs': 10},
                {'crop_size': 256, 'epochs': 30},
                {'crop_size': 512, 'epochs': 40},
            ])
            self.progressive_stages = progressive_cfg.get('stages', default_stages)
            self.current_crop_size = self.progressive_stages[0].get('crop_size', 128)
            self.progressive_epoch_index = 0

            try:
                from utils.training_state import set_progressive_crop
                # Phase 4.3: read default crop from base.yaml
                default_crop = progressive_cfg.get(
                    'default_crop_size',
                    pc_defaults.get('default_crop_size', 128),
                )
                set_progressive_crop(True, self.progressive_stages, default_crop)
                print(f"[Progressive Crop] Initialized in training_state")
            except ImportError as e:
                print(f"[Progressive Crop] WARNING: Failed to set training_state: {e}")

            total_progressive_epochs = sum(s.get('epochs', 10) for s in self.progressive_stages)
            print(f"[Progressive Crop] Enabled ({len(self.progressive_stages)} stages, total: {total_progressive_epochs} epochs)")
            for i, stage in enumerate(self.progressive_stages):
                print(f"  Stage {i+1}: crop_size={stage.get('crop_size', 128)}, epochs={stage.get('epochs', 10)}")
        else:
            self.progressive_stages = []
            self.current_crop_size = self.finetune_cfg.get('crop_size', 128)
            try:
                from utils.training_state import set_progressive_crop
                set_progressive_crop(False, [], self.current_crop_size)
            except ImportError as e:
                print(f"[Progressive Crop] WARNING: Failed to disable: {e}")

        if self.use_early_stopping:
            print(f"[Early Stopping] Enabled (metric={self.early_stopping_metric}, patience={self.early_stopping_patience})")

    def _read_config_early(self):
        """Phase 2 #8: explicit early config reads.

        These values are referenced by print blocks and adversarial-weight
        logic BEFORE `_setup_losses()` runs. Reading them here (as the first
        thing __init__ does) makes the ordering explicit and removes the
        fragility of relying on `_setup_losses` to side-effect these names.

        Values owned here:
        - self.lr
        - self.epochs
        - self.phase1_epochs
        - self.phase_adversarial_ramp_epochs
        - self.adversarial_start_epoch
        """
        # Optimizer / scheduler hyperparameters
        self.lr = self.finetune_cfg.get('lr', 0.0001)
        self.epochs = self.finetune_cfg.get('epochs', 50)

        # Two-phase training schedule
        two_phase_cfg = self.finetune_cfg.get('two_phase_training') or {}
        self.phase1_epochs = two_phase_cfg.get('phase1_epochs', 30)
        self.phase_adversarial_ramp_epochs = two_phase_cfg.get('phase_adversarial_ramp_epochs', 10)

        # Adversarial schedule start
        loss_cfg = self.finetune_cfg.get('loss', {}) or {}
        adv_cfg = loss_cfg.get('adversarial', {}) or {}
        self.adversarial_start_epoch = adv_cfg.get('start_epoch', 20)

    def _setup_losses(self):
        """Setup multi-loss system based on config."""
        loss_cfg = self.finetune_cfg.get('loss') or {}
        
        # Pixel loss (L1/L2)
        pixel_cfg = loss_cfg.get('pixel', {})
        pixel_type = pixel_cfg.get('type', 'l1')
        self.pixel_weight = float(pixel_cfg.get('weight', 1.0))

        if pixel_type == 'l1':
            self.pixel_loss = nn.L1Loss()
        elif pixel_type == 'l2' or pixel_type == 'mse':
            self.pixel_loss = nn.MSELoss()
        elif pixel_type == 'charbonnier':
            from losses import CharbonnierLoss
            eps = float(pixel_cfg.get('eps', 1e-6))
            self.pixel_loss = CharbonnierLoss(eps=eps)
        else:
            self.pixel_loss = nn.L1Loss()
        
        print(f"\n[Loss] Pixel loss: {pixel_type} (weight={self.pixel_weight})")
        
        # Perceptual loss - DISTS for texture, Dual for structure
        perceptual_cfg = loss_cfg.get('perceptual', {})
        self.use_perceptual = perceptual_cfg.get('enabled', False)
        self.perceptual_weight = perceptual_cfg.get('weight', 0.1)
        self.perceptual_warmup_epochs = perceptual_cfg.get('warmup_epochs', 5)
        self.perceptual_loss = None
        
        if self.use_perceptual:
            perceptual_type = perceptual_cfg.get('type', 'dists')
            try:
                if perceptual_type == 'dists':
                    from losses import DISTSLoss
                    self.perceptual_loss = DISTSLoss()
                    print(f"[Loss] Perceptual loss: DISTS (texture) (weight={self.perceptual_weight}, warmup={self.perceptual_warmup_epochs} epochs)")
                elif perceptual_type == 'vgg':
                    from losses import VGGPerceptualLoss
                    self.perceptual_loss = VGGPerceptualLoss()
                    print(f"[Loss] Perceptual loss: VGG (weight={self.perceptual_weight}, warmup={self.perceptual_warmup_epochs} epochs)")
                elif perceptual_type == 'resnet':
                    from losses import ResNetPerceptualLoss
                    self.perceptual_loss = ResNetPerceptualLoss()
                    print(f"[Loss] Perceptual loss: ResNet (weight={self.perceptual_weight}, warmup={self.perceptual_warmup_epochs} epochs)")
                elif perceptual_type == 'dual':
                    from losses import DualPerceptualLoss
                    self.perceptual_loss = DualPerceptualLoss()
                    print(f"[Loss] Perceptual loss: Dual VGG+ResNet (structure) (weight={self.perceptual_weight}, warmup={self.perceptual_warmup_epochs} epochs)")
                elif perceptual_type == 'twin':
                    from losses import TwinPerceptualLoss
                    twin_delta = perceptual_cfg.get('delta', 0.1)
                    twin_kwargs = {'delta': twin_delta}
                    if 'danbooru_weight' in perceptual_cfg and 'vgg_weight' in perceptual_cfg:
                        twin_kwargs['danbooru_weight'] = float(perceptual_cfg['danbooru_weight'])
                        twin_kwargs['vgg_weight'] = float(perceptual_cfg['vgg_weight'])
                        print(f"[Loss] Perceptual loss: Twin VGG19+ResNet50 (APISR) (danbooru={twin_kwargs['danbooru_weight']}, vgg={twin_kwargs['vgg_weight']}, weight={self.perceptual_weight})")
                    else:
                        print(f"[Loss] Perceptual loss: Twin VGG19+ResNet50 (APISR) (delta={twin_delta}, weight={self.perceptual_weight})")
                    self.perceptual_loss = TwinPerceptualLoss(**twin_kwargs)
                elif perceptual_type == 'none' or perceptual_type == 'disabled':
                    self.use_perceptual = False
                    print(f"[Loss] Perceptual loss: disabled")
                else:
                    from losses import DISTSLoss
                    self.perceptual_loss = DISTSLoss()
                    print(f"[Loss] Perceptual loss: DISTS (fallback) (weight={self.perceptual_weight})")
                
                if self.perceptual_loss is not None:
                    self.perceptual_loss = self.perceptual_loss.to(self.device)
            except Exception as e:
                logger.warning("Failed to load perceptual loss: %s", e)
                self.use_perceptual = False
        
        # Anime-specific losses
        anime_cfg = loss_cfg.get('anime') or {}
        
        # Line art preservation with multi-scale edge detection
        line_art_cfg = anime_cfg.get('line_art_preservation', {})
        self.use_line_art = line_art_cfg.get('enabled', False)
        self.line_art_weight = line_art_cfg.get('weight', 0.5)
        self.line_art_edge_method = line_art_cfg.get('edge_method', 'multi_scale')
        self.line_art_edge_weight = line_art_cfg.get('edge_weight', 3.0)
        self.line_art_loss = None
        
        if self.use_line_art:
            try:
                from losses import LineArtPreservationLoss
                self.line_art_loss = LineArtPreservationLoss(
                    edge_weight=self.line_art_edge_weight,
                    edge_method=self.line_art_edge_method
                ).to(self.device)
                print(f"[Loss] Line art preservation (weight={self.line_art_weight}, edge_method={self.line_art_edge_method})")
            except Exception as e:
                logger.warning("Failed to load line art loss: %s", e)
                self.use_line_art = False
        
        # Color consistency
        color_cfg = anime_cfg.get('color_consistency', {})
        self.use_color = color_cfg.get('enabled', False)
        self.color_weight = color_cfg.get('weight', 0.25)
        self.color_loss = None
        
        if self.use_color:
            try:
                from losses import ColorConsistencyLoss
                self.color_loss = ColorConsistencyLoss(color_weight=self.color_weight)
                print(f"[Loss] Color consistency (weight={self.color_weight})")
            except Exception as e:
                logger.warning("Failed to load color loss: %s", e)
                self.use_color = False
        
        # Flat region preservation
        flat_cfg = anime_cfg.get('flat_region_preservation', {})
        self.use_flat = flat_cfg.get('enabled', False)
        self.flat_weight = flat_cfg.get('weight', 0.3)
        self.flat_loss = None
        
        if self.use_flat:
            try:
                from losses import FlatRegionPreservationLoss
                self.flat_loss = FlatRegionPreservationLoss(preserve_weight=self.flat_weight).to(self.device)
                print(f"[Loss] Flat region preservation (weight={self.flat_weight})")
            except Exception as e:
                logger.warning("Failed to load flat region loss: %s", e)
                self.use_flat = False
        
        # Adversarial loss (GAN)
        adv_cfg = loss_cfg.get('adversarial') or {}
        self.use_adversarial = adv_cfg.get('enabled', False)
        self.adversarial_weight = adv_cfg.get('weight', 0.01)
        # NOTE: self.adversarial_start_epoch is owned by _read_config_early().
        self.adversarial_max_weight = adv_cfg.get('max_weight', 0.1)
        self.use_gradient_penalty = adv_cfg.get('gradient_penalty', True)
        # Phase 4.6: read lambda from config (default 1.0)
        self.gradient_penalty_lambda = float(adv_cfg.get('gradient_penalty_lambda', 1.0))
        # NOTE: 'label_smoothing' config was removed (was a no-op since v4 uses
        # RelativisticGANLoss, which is not compatible with binary target
        # smoothing). The config key is silently ignored for backward compat.
        self.discriminator = None
        self.discriminator_optimizer = None
        self.adversarial_loss_fn = None

        if self.use_adversarial:
            try:
                from losses import SRDiscriminator, RelativisticGANLoss, AdversarialLoss
                adv_type = adv_cfg.get('type', 'relativistic')
                
                # Create discriminator
                self.discriminator = SRDiscriminator(
                    in_channels=3,
                    num_features=64,
                    num_layers=3,
                ).to(self.device)

                if self.finetune_cfg.get('torch_compile', False) and hasattr(torch, 'compile'):
                    print(f"[torch.compile] Compiling discriminator")
                    self.discriminator = torch.compile(self.discriminator, mode="reduce-overhead")
                
                # Create discriminator optimizer (slower learning rate for stability)
                disc_lr = adv_cfg.get('discriminator_lr', self.lr * 0.25)
                # CRITICAL FIX: Use larger epsilon for discriminator stability
                self.discriminator_optimizer = optim.Adam(
                    self.discriminator.parameters(),
                    lr=disc_lr,
                    betas=(0.9, 0.99),
                    eps=1e-7,  # Increased for FP16 stability
                )
                
                # Create adversarial loss
                self.adversarial_loss_fn = AdversarialLoss(loss_type=adv_type)
                
                print(f"[Loss] Adversarial loss: {adv_type} (weight={self.adversarial_weight}, start_epoch={self.adversarial_start_epoch})")
                print(f"  Discriminator LR: {disc_lr}")
                print(f"  Gradient penalty: {self.use_gradient_penalty}")
            except Exception as e:
                logger.warning("Failed to load adversarial loss: %s", e)
                self.use_adversarial = False
        
        # Frequency-aware loss (ESPAN)
        freq_cfg = loss_cfg.get('frequency') or {}
        self.use_frequency = freq_cfg.get('enabled', False)
        self.frequency_weight = freq_cfg.get('weight', 0.1)
        self.frequency_block_size = freq_cfg.get('block_size', 8)
        self.frequency_high_weight = freq_cfg.get('high_freq_weight', 2.0)
        self.frequency_loss = None
        
        if self.use_frequency:
            try:
                from losses import FrequencyAwareLoss
                self.frequency_loss = FrequencyAwareLoss(
                    block_size=self.frequency_block_size,
                    high_freq_weight=self.frequency_high_weight,
                ).to(self.device)
                print(f"[Loss] Frequency-aware loss (weight={self.frequency_weight}, block_size={self.frequency_block_size})")
            except Exception as e:
                logger.warning("Failed to load frequency-aware loss: %s", e)
                self.use_frequency = False

        # FDL (Frequency Distribution Loss) with DINOv2
        fdl_cfg = loss_cfg.get('fdl') or {}
        self.use_fdl = fdl_cfg.get('enabled', False)
        self.fdl_weight = fdl_cfg.get('weight', 0.05)
        self.fdl_num_proj = fdl_cfg.get('num_proj', 24)
        self.fdl_compute_every = fdl_cfg.get('compute_every', 1)
        self.fdl_loss = None
        self._fdl_last_value = 0.0

        if self.use_fdl:
            try:
                from losses import FDLLoss
                self.fdl_loss = FDLLoss(
                    num_proj=self.fdl_num_proj,
                    dino_variant=fdl_cfg.get('dino_variant', 'small'),
                    weight=self.fdl_weight,
                ).to(self.device)
                if self.fdl_compute_every > 1:
                    print(f"[Loss] FDL loss (DINOv2) (weight={self.fdl_weight}, num_proj={self.fdl_num_proj}, compute_every={self.fdl_compute_every})")
                else:
                    print(f"[Loss] FDL loss (DINOv2) (weight={self.fdl_weight}, num_proj={self.fdl_num_proj})")
            except Exception as e:
                logger.warning("Failed to load FDL loss: %s", e)
                self.use_fdl = False

        # Wavelet-Guided Loss for GAN stability
        wg_cfg = loss_cfg.get('wavelet_guided') or {}
        self.use_wavelet_guided = wg_cfg.get('enabled', False)
        self.wavelet_guided_weight = wg_cfg.get('weight', 1.0)
        # Phase 4.8: prefer `start_epoch`; accept `wavelet_init` as a
        # deprecated alias. Default to 40 (legacy).
        if 'start_epoch' in wg_cfg:
            self.wavelet_init_epoch = wg_cfg.get('start_epoch', 40)
        else:
            self.wavelet_init_epoch = wg_cfg.get('wavelet_init', 40)
            if 'wavelet_init' in wg_cfg:
                logger.warning(
                    "loss.wavelet_guided.wavelet_init is deprecated; "
                    "rename to `start_epoch` in your config."
                )
        self.wavelet_guided_loss = None

        if self.use_wavelet_guided:
            try:
                from losses import WaveletGuidedLoss
                self.wavelet_guided_loss = WaveletGuidedLoss(
                    weight=self.wavelet_guided_weight,
                    hh_weight=wg_cfg.get('hh_weight', 2.0),
                    wavelet_init=self.wavelet_init_epoch,
                ).to(self.device)
                print(f"[Loss] Wavelet-guided loss (weight={self.wavelet_guided_weight}, init_epoch={self.wavelet_init_epoch})")
            except Exception as e:
                logger.warning("Failed to load wavelet-guided loss: %s", e)
                self.use_wavelet_guided = False

        print()
    
    def _get_perceptual_weight(self) -> float:
        """Get current perceptual loss weight with warmup."""
        if not self.use_perceptual or self.perceptual_loss is None:
            return 0.0
        
        # Linear warmup
        if self.current_epoch < self.perceptual_warmup_epochs:
            warmup_ratio = (self.current_epoch + 1) / self.perceptual_warmup_epochs
            return self.perceptual_weight * warmup_ratio
        
        return self.perceptual_weight
    
    def _get_phase_weights(self) -> Dict[str, float]:
        """
        Get loss weights for current phase in two-phase training.

        Phase 1 (epochs 0 to phase1_epochs-1): Learn structure with high pixel/perceptual
        Phase 2 (epochs phase1_epochs to end): Refine texture with adversarial

        All losses are phase-aware. Adversarial respects start_epoch config.

        Returns:
            Dict with 'pixel', 'perceptual', 'adversarial', 'line_art', 'flat', 'frequency' weights
        """
        if not self.use_two_phase:
            return {
                'pixel': self.pixel_weight,
                'perceptual': self._get_perceptual_weight(),
                'adversarial': self._get_adversarial_weight(),
                'line_art': self.line_art_weight,
                'flat': self.flat_weight,
                'frequency': self.frequency_weight,
            }

        if self.current_epoch < self.phase1_epochs:
            perc_weight = self.phase1_perceptual_weight
            if self.perceptual_warmup_epochs > 0 and self.current_epoch < self.perceptual_warmup_epochs:
                warmup_ratio = (self.current_epoch + 1) / self.perceptual_warmup_epochs
                perc_weight = perc_weight * warmup_ratio

            adversarial_w = self._compute_adversarial_weight_phase_aware(
                phase1_weight=self.phase1_adversarial_weight,
                phase2_weight=self.phase2_adversarial_weight,
                in_phase1=True
            )

            return {
                'pixel': self.phase1_pixel_weight,
                'perceptual': perc_weight,
                'adversarial': adversarial_w,
                'line_art': self.phase1_line_art_weight,
                'flat': self.phase1_flat_weight,
                'frequency': self.phase1_frequency_weight,
            }
        else:
            # Phase 2 weights interpolate from phase1 -> phase2 over the
            # phase2 epoch range. Guard against `total_phase2_epochs == 0`
            # (e.g. when phase1_epochs == epochs, which is unusual but valid).
            total_phase2_epochs = max(1, self.epochs - self.phase1_epochs)
            phase2_progress = (self.current_epoch - self.phase1_epochs) / total_phase2_epochs
            # Clamp to [0, 1] so out-of-range epochs (e.g. resumed beyond
            # configured `epochs`) don't extrapolate outside the phase range.
            phase2_progress = max(0.0, min(1.0, phase2_progress))

            pixel_w = self.phase1_pixel_weight + (self.phase2_pixel_weight - self.phase1_pixel_weight) * phase2_progress
            perceptual_w = self.phase1_perceptual_weight + (self.phase2_perceptual_weight - self.phase1_perceptual_weight) * phase2_progress
            line_art_w = self.phase1_line_art_weight + (self.phase2_line_art_weight - self.phase1_line_art_weight) * phase2_progress
            flat_w = self.phase1_flat_weight + (self.phase2_flat_weight - self.phase1_flat_weight) * phase2_progress
            frequency_w = self.phase1_frequency_weight + (self.phase2_frequency_weight - self.phase1_frequency_weight) * phase2_progress

            adversarial_w = self._compute_adversarial_weight_phase_aware(
                phase1_weight=self.phase1_adversarial_weight,
                phase2_weight=self.phase2_adversarial_weight,
                in_phase1=False
            )

            return {
                'pixel': pixel_w,
                'perceptual': perceptual_w,
                'adversarial': adversarial_w,
                'line_art': line_art_w,
                'flat': flat_w,
                'frequency': frequency_w,
            }

    def _compute_adversarial_weight_phase_aware(self, phase1_weight: float, phase2_weight: float, in_phase1: bool) -> float:
        """
        Compute phase-aware adversarial weight respecting start_epoch and ramp.

        Args:
            phase1_weight: Target adversarial weight for Phase 1
            phase2_weight: Target adversarial weight for Phase 2
            in_phase1: True if current epoch is in Phase 1

        Returns:
            Current adversarial weight (0.0 if before start_epoch)
        """
        if not self.use_adversarial or self.discriminator is None:
            return 0.0

        if self.current_epoch < self.adversarial_start_epoch:
            return 0.0

        if in_phase1:
            ramp_target = phase1_weight
        else:
            ramp_target = phase2_weight

        ramp_epochs = self.phase_adversarial_ramp_epochs
        ramp_end_epoch = self.adversarial_start_epoch + ramp_epochs

        if self.current_epoch < ramp_end_epoch:
            progress = (self.current_epoch - self.adversarial_start_epoch) / ramp_epochs
            return ramp_target * progress

        return ramp_target
    
    def _get_adversarial_weight(self) -> float:
        """Get current adversarial loss weight with progressive schedule.

        Phase 4.7: thin wrapper around the phase-aware method, using the
        single-phase target weight. This eliminates the previous hard-coded
        10-epoch ramp; the schedule now respects `phase_adversarial_ramp_epochs`
        consistently for both two-phase and single-phase training.
        """
        target = min(self.adversarial_weight, self.adversarial_max_weight)
        return self._compute_adversarial_weight_phase_aware(
            phase1_weight=target,
            phase2_weight=target,
            in_phase1=True,
        )
    
    def _get_current_crop_size(self, epoch: int) -> int:
        """
        Get the current crop size based on progressive schedule.
        
        Args:
            epoch: Current epoch number (0-indexed)
        
        Returns:
            Current crop size
        """
        if not self.use_progressive_crop or not self.progressive_stages:
            return self.current_crop_size
        
        # Calculate cumulative epochs
        cumulative_epochs = 0
        for stage in self.progressive_stages:
            stage_epochs = stage.get('epochs', 10)
            if epoch < cumulative_epochs + stage_epochs:
                return stage.get('crop_size', 128)
            cumulative_epochs += stage_epochs
        
        # Return last stage crop size if we've exceeded total
        return self.progressive_stages[-1].get('crop_size', 128)
    
    def _check_crop_size_change(self, epoch: int) -> bool:
        """
        Check if crop size should change at this epoch.
        
        Args:
            epoch: Current epoch number (0-indexed)
        
        Returns:
            True if crop size changes at this epoch
        """
        if not self.use_progressive_crop or not self.progressive_stages:
            return False
        
        new_crop_size = self._get_current_crop_size(epoch)
        return new_crop_size != self.current_crop_size
    
    def _update_crop_size(self, new_crop_size: int):
        """
        Update the current crop size and notify user.

        Args:
            new_crop_size: New crop size to use
        """
        old_crop_size = self.current_crop_size
        self.current_crop_size = new_crop_size

        try:
            from utils.training_state import get_training_state
            state = get_training_state()
            state.progressive_crop.update_crop_size(new_crop_size)
            print(f"\n[Progressive Crop] Changing crop size: {old_crop_size} -> {new_crop_size}")
            print(f"  Crop size updated in training state")
        except ImportError:
            print(f"\n[Progressive Crop] Changing crop size: {old_crop_size} -> {new_crop_size}")
            print(f"  Note: Update training_state module for dataloader sync")
    
    def _compute_gradient_penalty(self, real_samples: torch.Tensor, fake_samples: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient penalty for WGAN-GP style training.
        This regularizes the discriminator to have Lipschitz constraint.
        
        CRITICAL FIX: Add epsilon to norm calculation to prevent NaN when gradients are zero.
        Reference: https://github.com/pytorch/pytorch/issues/2534
        """
        batch_size = real_samples.size(0)
        
        # Random interpolation factor
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        
        # Interpolated samples
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated.requires_grad_(True)
        
        # Discriminator output on interpolated (FP32 for stability)
        with torch.amp.autocast('cuda', enabled=False):
            disc_interpolated = self.discriminator(interpolated.float())
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=disc_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(disc_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        # CRITICAL FIX: Add epsilon before norm to prevent NaN
        # torch.norm() produces NaN when all gradients are zero
        gradients = gradients.view(batch_size, -1)
        gradient_norm = torch.sqrt((gradients ** 2).sum(dim=1) + 1e-12)
        penalty = ((gradient_norm - 1) ** 2).mean()
        
        # Safety check
        if torch.isnan(penalty) or torch.isinf(penalty):
            print(f"  [WARNING] NaN/Inf in gradient penalty, using zero penalty")
            penalty = torch.tensor(0.0, device=self.device, requires_grad=False)
        
        return penalty
    
    def _train_discriminator(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> Dict[str, float]:
        """
        Train discriminator for one step.
        
        Args:
            real_images: Real HR images
            fake_images: Generated SR images (from generator)
        
        Returns:
            Dict with discriminator loss and metrics
        """
        self.discriminator_optimizer.zero_grad()
        
        # CRITICAL FIX: Run discriminator in FP32 to avoid type mismatch
        with torch.amp.autocast('cuda', enabled=False):
            # Discriminator on real
            disc_real = self.discriminator(real_images.float())
            
            # Discriminator on fake (detach to prevent gradients flowing to generator)
            disc_fake = self.discriminator(fake_images.detach().float())
        
        # Compute discriminator loss
        from losses import RelativisticGANLoss
        if isinstance(self.adversarial_loss_fn.loss_fn, RelativisticGANLoss):
            d_loss = self.adversarial_loss_fn.loss_fn.discriminator_loss(disc_real, disc_fake)
        else:
            loss_real = self.adversarial_loss_fn.loss_fn(disc_real, is_real=True)
            loss_fake = self.adversarial_loss_fn.loss_fn(disc_fake, is_real=False)
            d_loss = (loss_real + loss_fake) / 2
        
        # Add gradient penalty for stability
        if self.use_gradient_penalty:
            gp = self._compute_gradient_penalty(real_images, fake_images)
            d_loss = d_loss + self.gradient_penalty_lambda * gp
        
        d_loss.backward()
        
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        
        self.discriminator_optimizer.step()
        
        return {
            'd_loss': d_loss.item(),
            'disc_real': disc_real.mean().item(),
            'disc_fake': disc_fake.mean().item(),
        }
    
    def _compute_total_loss(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
        batch_idx: int = 0,
        include_adversarial: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss from all enabled loss components.

        CRITICAL FIX: Perceptual losses MUST run in FP32, not FP16.
        Uses phase-aware weights for two-phase training.

        Args:
            sr: Super-resolved prediction.
            hr: Ground truth.
            batch_idx: Current batch index (for FDL compute_every gating).
            include_adversarial: Reserved flag for future adversarial integration
                into this method. Currently a no-op because adversarial is
                handled separately in train_epoch() (after the discriminator
                step), so validate()'s call to _compute_total_loss already
                skips the discriminator update by virtue of not entering
                train_epoch. Kept in the signature so callers can pass it
                explicitly and so the contract is documented.

        Returns:
            Dict with 'total' loss and individual loss components
        """
        loss_dict = {}
        
        # Get phase-aware weights
        weights = self._get_phase_weights()
        pixel_weight = weights['pixel']
        perceptual_weight = weights['perceptual']
        
        # Pixel loss
        pixel_loss = self.pixel_loss(sr, hr)
        loss_dict['pixel'] = pixel_loss
        total_loss = pixel_weight * pixel_loss
        
        # Perceptual loss (DISTS) - MUST run in FP32
        if self.use_perceptual and self.perceptual_loss is not None:
            if perceptual_weight > 0:
                with torch.amp.autocast('cuda', enabled=False):
                    sr_fp32 = sr.float()
                    hr_fp32 = hr.float()
                    perc_loss = self.perceptual_loss(sr_fp32, hr_fp32)
                
                if torch.isnan(perc_loss) or torch.isinf(perc_loss):
                    print(f"  [WARNING] NaN/Inf in perceptual loss, skipping")
                    perc_loss = torch.tensor(0.0, device=sr.device, requires_grad=False)
                else:
                    loss_dict['perceptual'] = perc_loss
                    total_loss = total_loss + perceptual_weight * perc_loss
        
        # Line art preservation loss
        if self.use_line_art and self.line_art_loss is not None:
            line_result = self.line_art_loss(sr, hr)
            line_loss = line_result['line_art'] if isinstance(line_result, dict) else line_result

            if torch.isnan(line_loss) or torch.isinf(line_loss):
                print(f"  [WARNING] NaN/Inf in line art loss, skipping")
                line_loss = torch.tensor(0.0, device=sr.device, requires_grad=False)
            else:
                loss_dict['line_art'] = line_loss
                total_loss = total_loss + weights.get('line_art', self.line_art_weight) * line_loss
        
        # Color consistency loss
        if self.use_color and self.color_loss is not None:
            color_result = self.color_loss(sr, hr)
            color_loss = color_result['color_consistency'] if isinstance(color_result, dict) else color_result
            
            if torch.isnan(color_loss) or torch.isinf(color_loss):
                print(f"  [WARNING] NaN/Inf in color loss, skipping")
                color_loss = torch.tensor(0.0, device=sr.device, requires_grad=False)
            else:
                loss_dict['color'] = color_loss
                total_loss = total_loss + self.color_weight * color_loss
        
        # Flat region preservation loss
        if self.use_flat and self.flat_loss is not None:
            flat_result = self.flat_loss(sr, hr)
            flat_loss = flat_result['flat_preservation'] if isinstance(flat_result, dict) else flat_result

            if torch.isnan(flat_loss) or torch.isinf(flat_loss):
                print(f"  [WARNING] NaN/Inf in flat region loss, skipping")
                flat_loss = torch.tensor(0.0, device=sr.device, requires_grad=False)
            else:
                loss_dict['flat'] = flat_loss
                total_loss = total_loss + weights.get('flat', self.flat_weight) * flat_loss
        
        # Frequency-aware loss (ESPAN)
        if self.use_frequency and self.frequency_loss is not None:
            with torch.amp.autocast('cuda', enabled=False):
                freq_loss = self.frequency_loss(sr.float(), hr.float())

            if torch.isnan(freq_loss) or torch.isinf(freq_loss):
                print(f"  [WARNING] NaN/Inf in frequency loss, skipping")
                freq_loss = torch.tensor(0.0, device=sr.device, requires_grad=False)
            else:
                loss_dict['frequency'] = freq_loss
                total_loss = total_loss + weights.get('frequency', self.frequency_weight) * freq_loss

        # FDL loss (DINOv2) - MUST run in FP32
        # compute_every > 1 skips computation on some batches for speed
        if self.use_fdl and self.fdl_loss is not None:
            if batch_idx % self.fdl_compute_every == 0:
                with torch.amp.autocast('cuda', enabled=False):
                    fdl_loss = self.fdl_loss(sr.float(), hr.float())

                if torch.isnan(fdl_loss) or torch.isinf(fdl_loss):
                    logger.warning("NaN/Inf in FDL loss on batch %d, skipping", batch_idx)
                    fdl_loss = torch.tensor(0.0, device=sr.device, requires_grad=False)
                else:
                    self._fdl_last_value = fdl_loss.item()
                    loss_dict['fdl'] = fdl_loss
                    total_loss = total_loss + self.fdl_weight * fdl_loss
            else:
                # Phase 3.4: skipped batch — log the cached value only, do NOT add
                # to total_loss. The cached value has requires_grad=False so it
                # contributed nothing to gradients, but it also inflated the
                # displayed total_loss for no reason.
                logger.debug(
                    "FDL skipped batch %d (compute_every=%d); cached=%g",
                    batch_idx, self.fdl_compute_every, self._fdl_last_value,
                )
                loss_dict['fdl_cached'] = self._fdl_last_value

        # Wavelet-Guided Loss - MUST run in FP32
        if self.use_wavelet_guided and self.wavelet_guided_loss is not None:
            with torch.amp.autocast('cuda', enabled=False):
                wg_loss = self.wavelet_guided_loss(sr.float(), hr.float(), epoch=self.current_epoch)

            if torch.isnan(wg_loss) or torch.isinf(wg_loss):
                print(f"  [WARNING] NaN/Inf in wavelet-guided loss, skipping")
                wg_loss = torch.tensor(0.0, device=sr.device, requires_grad=False)
            else:
                loss_dict['wavelet_guided'] = wg_loss
                total_loss = total_loss + self.wavelet_guided_weight * wg_loss
        
        # Final safety check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"  [CRITICAL] Total loss is NaN/Inf, returning pixel loss only")
            total_loss = pixel_loss.detach()
            loss_dict = {'pixel': pixel_loss, 'total': total_loss}
        
        loss_dict['total'] = total_loss
        return loss_dict
    
    def _freeze_layers(self, freeze_layers: list):
        """Freeze specified layers."""
        for name, param in self.model.named_parameters():
            for pattern in freeze_layers:
                if pattern in name:
                    param.requires_grad = False
                    print(f"  Frozen: {name}")
                    break
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({trainable/total*100:.1f}%)")
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        scheduler_type = self.finetune_cfg.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            base_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs - self.warmup_epochs,
                eta_min=self.min_lr,
            )
        elif scheduler_type == 'step':
            step_size = self.finetune_cfg.get('step_size', max(1, self.epochs // 3))
            gamma = self.finetune_cfg.get('gamma', 0.5)
            base_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma,
            )
        elif scheduler_type == 'multistep':
            milestones = self.finetune_cfg.get('milestones', [self.epochs // 2, self.epochs * 3 // 4])
            gamma = self.finetune_cfg.get('gamma', 0.5)
            base_scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=gamma,
            )
        elif scheduler_type == 'constant':
            base_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda _: 1.0,
            )
        else:
            print(f"[WARNING] Unknown scheduler type '{scheduler_type}', using cosine")
            base_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs - self.warmup_epochs,
                eta_min=self.min_lr,
            )
        
        class WarmupScheduler:
            def __init__(self, base, warmup_epochs, base_lr, min_lr):
                self.base = base
                self.warmup_epochs = warmup_epochs
                self.base_lr = base_lr
                self.min_lr = min_lr
                self.current_epoch = 0
            
            def step(self):
                self.current_epoch += 1
                if self.current_epoch <= self.warmup_epochs:
                    lr = self.min_lr + (self.base_lr - self.min_lr) * (self.current_epoch / self.warmup_epochs)
                    for param_group in self.base.optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    self.base.step()
            
            def get_last_lr(self):
                return [g['lr'] for g in self.base.optimizer.param_groups]
        
        return WarmupScheduler(base_scheduler, self.warmup_epochs, self.lr, self.min_lr)
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train one epoch with multi-loss and adversarial support."""
        self.model.train()
        
        # Accumulators for all loss components
        loss_accumulators = {'total': 0.0}
        num_batches = 0
        num_optimizer_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}/{self.epochs}")

        for batch_idx, batch in enumerate(pbar):
            lr = batch['lr'].to(self.device, non_blocking=True)
            hr = batch['hr'].to(self.device, non_blocking=True)

            # Handle GPU degradation if lr is None
            if lr is None:
                scale = self.config.get('model', {}).get('upscale', 4)
                lr = F.interpolate(hr, scale_factor=1/scale, mode='bicubic', align_corners=False)

            # Forward with gradient accumulation
            # CRITICAL: Run model in autocast, but compute losses carefully
            with autocast('cuda'):
                sr = self.model(lr)

            # Compute non-adversarial losses
            loss_dict = self._compute_total_loss(sr, hr, batch_idx)

            # Train discriminator and compute adversarial loss for generator
            # CRITICAL: This must happen BEFORE total_loss.backward() to avoid graph conflicts
            # The discriminator training modifies weights, so adversarial loss must be computed AFTER
            if self.use_adversarial and self.discriminator is not None:
                # Get phase-aware adversarial weight (Phase 1 has minimal adversarial)
                weights = self._get_phase_weights()
                adv_weight = weights['adversarial']

                if adv_weight > 0:
                    # Step 1: Train discriminator (modifies discriminator weights)
                    sr_detached = sr.detach().clone()
                    disc_metrics = self._train_discriminator(hr, sr_detached)
                    loss_dict['discriminator'] = torch.tensor(disc_metrics['d_loss'])
                    loss_dict['disc_real'] = torch.tensor(disc_metrics['disc_real'])
                    loss_dict['disc_fake'] = torch.tensor(disc_metrics['disc_fake'])

                    # Step 2: Compute adversarial loss for generator using UPDATED discriminator
                    with torch.amp.autocast('cuda', enabled=False):
                        disc_real = self.discriminator(hr.float())
                        disc_fake = self.discriminator(sr.float())

                    adv_loss = self.adversarial_loss_fn.generator_loss(disc_real, disc_fake)

                    if torch.isnan(adv_loss) or torch.isinf(adv_loss):
                        print(f"  [WARNING] NaN/Inf in adversarial loss, skipping")
                        adv_loss = torch.tensor(0.0, device=sr.device, requires_grad=False)
                    else:
                        loss_dict['adversarial'] = adv_loss
                        loss_dict['total'] = loss_dict['total'] + adv_weight * adv_loss

            # Check for NaN in total loss BEFORE backward
            if torch.isnan(loss_dict['total']) or torch.isinf(loss_dict['total']):
                print(f"\n[WARNING] NaN/Inf detected in loss at batch {batch_idx}")
                print(f"  Loss components: {', '.join([f'{k}={v.item():.4f}' for k, v in loss_dict.items() if k != 'total'])}")
                print(f"  Skipping batch and zeroing gradients")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            # Scale loss for gradient accumulation
            total_loss = loss_dict['total'] / self.gradient_accumulation

            # Backward pass (use GradScaler when AMP is enabled)
            if self.scaler is not None:
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            # Step optimizer after accumulation
            if (batch_idx + 1) % self.gradient_accumulation == 0:
                num_optimizer_steps += 1

                # Unscale gradients before sanitizing/clipping (GradScaler)
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                # CRITICAL FIX: Sanitize gradients before clipping
                # Replace any NaN/Inf gradients with zeros to prevent corruption
                nan_count = 0
                offender_names = []
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        nan_mask = torch.isnan(param.grad) | torch.isinf(param.grad)
                        n_bad = nan_mask.sum().item()
                        if n_bad > 0:
                            nan_count += n_bad
                            # Phase 3.5: log up to 5 offender param names per epoch
                            if len(offender_names) < 5:
                                offender_names.append(
                                    f"{name} ({n_bad} bad)"
                                )
                            param.grad[nan_mask] = 0.0

                if nan_count > 0:
                    suffix = (
                        f" offenders={offender_names}"
                        if offender_names
                        else ""
                    )
                    logger.warning(
                        "Sanitized %d NaN/Inf gradient values (epoch=%d, batch=%d)%s",
                        nan_count, self.current_epoch, batch_idx, suffix,
                    )

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Step optimizer (GradScaler skips step if inf/nan found and updates)
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

                # Phase D-EMA: update shadow params after each optimizer step.
                # Done after zero_grad so we don't accidentally wipe the
                # shadow update if zero_grad ever operates on the shadow
                # (it doesn't today, but defensive ordering costs nothing).
                # Use getattr() so unit tests that stub-build the trainer
                # (see test_grad_scaler.py) keep working.
                ema = getattr(self, 'ema', None)
                if ema is not None:
                    ema.update(self.model)

                self.global_step += 1

            # Accumulate losses (multiply back by gradient_accumulation for reporting)
            for key, value in loss_dict.items():
                # Skip non-tensor values (e.g. fdl_cached = Python float)
                if not isinstance(value, torch.Tensor):
                    continue
                if key not in loss_accumulators:
                    loss_accumulators[key] = 0.0
                loss_accumulators[key] += value.item() * self.gradient_accumulation

            num_batches += 1

            # Build progress bar postfix with key losses and step counter
            step_info = f"step {num_optimizer_steps}"
            if self.gradient_accumulation > 1:
                step_info += f"/{num_batches} batches"
            postfix = {'loss': f"{loss_dict['total'].item():.4f}", 'step': step_info}
            if 'perceptual' in loss_dict:
                postfix['perc'] = f"{loss_dict['perceptual'].item():.4f}"
            if 'line_art' in loss_dict:
                postfix['line'] = f"{loss_dict['line_art'].item():.4f}"
            if 'adversarial' in loss_dict:
                postfix['adv'] = f"{loss_dict['adversarial'].item():.4f}"
            if 'frequency' in loss_dict:
                postfix['freq'] = f"{loss_dict['frequency'].item():.4f}"
            if 'fdl' in loss_dict:
                postfix['fdl'] = f"{loss_dict['fdl'].item():.4f}"
            if 'wavelet_guided' in loss_dict:
                postfix['wg'] = f"{loss_dict['wavelet_guided'].item():.4f}"
            pbar.set_postfix(postfix)
        
        # Average losses
        avg_losses = {key: val / max(num_batches, 1) for key, val in loss_accumulators.items()}
        return avg_losses
    
    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate model with multi-loss and perceptual metrics."""
        self.model.eval()

        # Phase D-EMA: run validation with the shadow (averaged) weights
        # so PSNR/SSIM/LPIPS reflect the stabilized model rather than
        # the noisy per-step weights. The contextmanager swaps back to
        # the live weights on exit, so training continues unaffected.
        # Use getattr() so stub-built trainers (see test_grad_scaler.py)
        # don't AttributeError out of the validation entry point.
        ema = getattr(self, 'ema', None)
        if ema is not None:
            ema_ctx = ema.averaged(self.model)
        else:
            from contextlib import nullcontext
            ema_ctx = nullcontext()

        with ema_ctx:
            loss_accumulators = {'total': 0.0}
            total_psnr = 0.0
            total_ssim = 0.0
            total_lpips = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(tqdm(val_loader, desc='Validation')):
                lr = batch['lr'].to(self.device, non_blocking=True)
                hr = batch['hr'].to(self.device, non_blocking=True)

                if lr is None:
                    scale = self.config.get('model', {}).get('upscale', 4)
                    lr = F.interpolate(hr, scale_factor=1/scale, mode='bicubic', align_corners=False)

                sr = self.model(lr)
                loss_dict = self._compute_total_loss(sr, hr, batch_idx)

                # Accumulate losses
                for key, value in loss_dict.items():
                    # Skip non-tensor values (e.g. fdl_cached = Python float)
                    if not isinstance(value, torch.Tensor):
                        continue
                    if key not in loss_accumulators:
                        loss_accumulators[key] = 0.0
                    loss_accumulators[key] += value.item()

                # Calculate PSNR
                mse = torch.mean((sr.clamp(0, 1) - hr.clamp(0, 1)) ** 2)
                psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
                total_psnr += psnr.item()

                # Calculate SSIM using existing utility
                try:
                    from utils.metrics import calculate_ssim
                    ssim = calculate_ssim(sr.clamp(0, 1), hr.clamp(0, 1))
                    total_ssim += ssim
                except Exception:
                    pass

                # Calculate LPIPS for perceptual quality
                try:
                    from utils.metrics import calculate_lpips
                    lpips_val = calculate_lpips(sr.clamp(0, 1), hr.clamp(0, 1))
                    total_lpips += lpips_val
                except Exception:
                    pass

                num_batches += 1

            # Average losses and metrics
            avg_losses = {key: val / max(num_batches, 1) for key, val in loss_accumulators.items()}
            avg_losses['psnr'] = total_psnr / max(num_batches, 1)
            avg_losses['ssim'] = total_ssim / max(num_batches, 1)
            avg_losses['lpips'] = total_lpips / max(num_batches, 1)
        
        return avg_losses
    
    def _check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """
        Check if early stopping should be triggered.
        
        Args:
            metrics: Current validation metrics
        
        Returns:
            True if should stop early
        """
        if not self.use_early_stopping:
            return False
        
        # Get current metric value
        current_value = metrics.get(self.early_stopping_metric)
        if current_value is None:
            return False
        
        # Check if improved
        if self.early_stopping_mode == 'min':
            improved = current_value < self.best_metric_value - 1e-6
        else:
            improved = current_value > self.best_metric_value + 1e-6
        
        if improved:
            self.best_metric_value = current_value
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
        
        if self.early_stopping_counter >= self.early_stopping_patience:
            print(f"\n[Early Stopping] Triggered after {self.early_stopping_patience} epochs without improvement")
            print(f"  Best {self.early_stopping_metric}: {self.best_metric_value:.4f}")
            return True
        
        return False
    
    def _log_to_tensorboard(self, epoch: int, train_metrics: Dict, val_metrics: Optional[Dict] = None):
        """Log metrics to TensorBoard."""
        if not self.use_tensorboard or self.writer is None:
            return
        
        # Log training metrics
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'train/{key}', value, epoch)
        
        # Log validation metrics
        if val_metrics is not None:
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'val/{key}', value, epoch)
        
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('lr', current_lr, epoch)
        
        self.writer.flush()
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint to resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            epoch to resume from (0-indexed)
        """
        if not Path(checkpoint_path).exists():
            print(f"[WARNING] Checkpoint not found: {checkpoint_path}")
            return 0

        # PR-4 (Issue #4): safe checkpoint load.
        # Pattern: try weights_only=True first; fall back to weights_only=False
        # only if the user has explicitly opted in via
        # `security.allow_pickle_checkpoint: true`. This matches the canonical
        # pattern from base_trainer.py:354-372.
        checkpoint = None
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=True)
        except UnpicklingError as e:
            security_cfg = self.config.get('security') or {}
            allow_pickle = security_cfg.get('allow_pickle_checkpoint', False)
            if not allow_pickle:
                raise RuntimeError(
                    f"Checkpoint requires pickle deserialization but "
                    f"`security.allow_pickle_checkpoint` is not set. "
                    f"Set it to true in the config to allow (only with trusted "
                    f"checkpoints). Error: {e}"
                )
            import warnings
            warnings.warn(
                "Loading checkpoint with pickle fallback - only use with "
                "trusted sources!",
                UserWarning,
                stacklevel=2,
            )
            checkpoint = torch.load(checkpoint_path, weights_only=False)

        try:
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[Resume] Loaded model weights from epoch {checkpoint.get('epoch', 0) + 1}")

            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"[Resume] Loaded optimizer state")

            # Load scheduler state
            if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
                if checkpoint['scheduler_state_dict'] is not None:
                    self.scheduler.base.load_state_dict(checkpoint['scheduler_state_dict'])
                    print(f"[Resume] Loaded scheduler state")

            # Load GradScaler state (PR-2). Old checkpoints (pre-2026-06) do not
            # contain scaler_state_dict; the scaler then starts fresh and prints
            # a one-time warning. This is safe and forward-compatible.
            if 'scaler_state_dict' in checkpoint and self.scaler is not None:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print(f"[Resume] Loaded GradScaler state")
            elif self.scaler is not None and self.scaler._enabled:
                # Only warn if AMP is actually active and the checkpoint predates
                # the scaler change (no scaler_state_dict key present).
                print(f"[Resume] WARNING: AMP active but checkpoint has no scaler_state_dict; scaler will start fresh")

            # Restore best loss
            if 'best_loss' in checkpoint:
                self.best_loss = checkpoint['best_loss']
                print(f"[Resume] Restored best loss: {self.best_loss:.4f}")

            # Restore early stopping state
            if 'early_stopping_counter' in checkpoint:
                self.early_stopping_counter = checkpoint['early_stopping_counter']
                print(f"[Resume] Restored early stopping counter: {self.early_stopping_counter}")
            if 'best_metric_value' in checkpoint:
                self.best_metric_value = checkpoint['best_metric_value']
                print(f"[Resume] Restored best metric value: {self.best_metric_value:.4f}")

            # Restore crop size for progressive crop
            if self.use_progressive_crop and 'crop_size' in checkpoint:
                saved_crop = checkpoint['crop_size']
                self.current_crop_size = saved_crop
                try:
                    from utils.training_state import get_training_state
                    state = get_training_state()
                    state.progressive_crop.update_crop_size(saved_crop)
                    state.progressive_crop.current_crop_size = saved_crop
                except ImportError:
                    pass
                print(f"[Resume] Restored crop_size: {saved_crop}")

            # Restore phase1_epochs from checkpoint if it exists
            # CRITICAL: This ensures phase determination uses the original config value
            # Fixes bug where resuming with different phase1_epochs would cause incorrect phase
            if 'phase1_epochs' in checkpoint:
                saved_phase1_epochs = checkpoint['phase1_epochs']
                if saved_phase1_epochs != self.phase1_epochs:
                    print(f"[Resume] Restoring phase1_epochs: {saved_phase1_epochs} (config: {self.phase1_epochs})")
                self.phase1_epochs = saved_phase1_epochs

            # Phase D-EMA: restore shadow weights if present. Backward
            # compat: checkpoints saved before EMA was wired (pre-2026-06)
            # have no `ema_state_dict` / `ema_model_state_dict` key; the
            # shadow then starts at its initial (deepcopy-of-model) values
            # and is harmless — we just log a warning so the user knows.
            # Use getattr() so stub-built trainers (see test_grad_scaler.py)
            # don't AttributeError out of the whole load.
            ema = getattr(self, 'ema', None)
            if ema is not None:
                ema_payload = (
                    checkpoint.get('ema_state_dict')
                    if 'ema_state_dict' in checkpoint
                    else checkpoint.get('ema_model_state_dict')
                )
                if ema_payload is not None:
                    try:
                        ema.load_state_dict(ema_payload)
                        print(f"[Resume] Loaded EMA shadow state")
                    except Exception as e:
                        logger.warning("Failed to load EMA shadow state: %s", e)
                else:
                    logger.warning(
                        "Checkpoint predates EMA; shadow starts at initial "
                        "(deepcopy-of-model) values."
                    )

            # Restore epoch
            resume_epoch = checkpoint.get('epoch', 0) + 1
            print(f"[Resume] Resuming from epoch {resume_epoch + 1}")

            return resume_epoch

        except Exception as e:
            logger.warning("Failed to load checkpoint: %s", e)
            print(f"[Resume] Starting from epoch 0")
            return 0
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, val_metrics: Optional[Dict] = None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.base.state_dict() if hasattr(self.scheduler, 'base') else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler is not None else None,
            'best_loss': self.best_loss,
            'early_stopping_counter': self.early_stopping_counter,
            'best_metric_value': self.best_metric_value,
            'config': self.config,
            'crop_size': self.current_crop_size,
            'progressive_crop_enabled': self.use_progressive_crop,
            'phase1_epochs': self.phase1_epochs,
        }

        # Phase D-EMA: save shadow weights under both `ema_state_dict`
        # (canonical, matches base_trainer key shape) and
        # `ema_model_state_dict` (matches what the inference engine
        # already loads — see src/inference/engine.py:159-163). Saving
        # under both keys keeps resume and inference paths consistent.
        # Use getattr() so unit tests that stub-build the trainer via
        # `NeosrSPANFinetuner.__new__()` (see test_grad_scaler.py) keep
        # working — they don't run __init__ and so don't have `self.ema`.
        ema = getattr(self, 'ema', None)
        if ema is not None:
            try:
                ema_sd = ema.state_dict()
                checkpoint['ema_state_dict'] = ema_sd
                checkpoint['ema_model_state_dict'] = ema_sd
            except Exception as e:
                logger.warning("Failed to serialize EMA state: %s", e)
        
        # Save latest
        latest_path = self.checkpoint_dir / 'finetune_latest.pth'
        torch.save(checkpoint, latest_path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'finetune_best.pth'
            torch.save(checkpoint, best_path)
            print(f"  Saved best checkpoint with loss {self.best_loss:.4f}")
        
        if (epoch + 1) % self.save_interval == 0:
            epoch_path = self.checkpoint_dir / f'finetune_epoch_{epoch+1}.pth'
            torch.save(checkpoint, epoch_path)
            print(f"  Saved epoch checkpoint: {epoch_path}")

        if self.checkpoint_manager:
            status = 'interrupted' if epoch < self.epochs - 1 else 'completed'
            updates = {
                'status': status,
                'best_loss': float(self.best_loss),
                'final_epoch': epoch + 1,
            }
            if val_metrics:
                updates['psnr'] = f"{val_metrics.get('psnr', 0.0):.2f}"
                updates['ssim'] = f"{val_metrics.get('ssim', 0.0):.4f}"
                updates['lpips'] = f"{val_metrics.get('lpips', 0.0):.4f}"
            self.checkpoint_manager.update_run(updates)
    
    def train(self, train_loader, val_loader=None, resume_from: str = None):
        """Full training loop with multi-loss and adversarial support.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            resume_from: Checkpoint path to resume from (optional)
        """
        print("\n" + "="*50)
        print("Fine-tuning neosr SPAN (Multi-Loss + GAN)")
        print("="*50)
        print(f"  Epochs: {self.epochs}")
        print(f"  LR: {self.lr} -> {self.min_lr}")
        print(f"  Batch size: {self.finetune_cfg.get('batch_size', 4)}")
        print(f"  Gradient accumulation: {self.gradient_accumulation}x (effective batch: {self.finetune_cfg.get('batch_size', 4) * self.gradient_accumulation})")
        print(f"  Device: {self.device}")
        print(f"\n  Loss Configuration:")
        print(f"    Pixel: {self.pixel_weight}")
        if self.use_perceptual:
            print(f"    Perceptual: {self.perceptual_weight} (warmup: {self.perceptual_warmup_epochs} epochs)")
        if self.use_line_art:
            print(f"    Line Art: {self.line_art_weight}")
        if self.use_flat:
            print(f"    Flat Region: {self.flat_weight}")
        if self.use_adversarial:
            print(f"    Adversarial: {self.adversarial_weight} (starts epoch {self.adversarial_start_epoch}, max: {self.adversarial_max_weight})")
        if self.use_frequency:
            print(f"    Frequency: {self.frequency_weight} (block_size: {self.frequency_block_size})")
        
        # Handle resume
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
            print(f"\n[Resume] Starting from epoch {start_epoch + 1}/{self.epochs}")

        # Print initial crop size
        if self.use_progressive_crop:
            print(f"  Crop size: {self.current_crop_size} (progressive mode)")
        else:
            print(f"  Crop size: {self.current_crop_size} (fixed mode)")

        # Print phase info on resume
        if resume_from and self.use_two_phase:
            if start_epoch < self.phase1_epochs:
                print(f"  Phase: Phase 1 (structure learning) - {self.phase1_epochs - start_epoch} epochs remaining")
            else:
                print(f"  Phase: Phase 2 (texture refinement) - {self.epochs - start_epoch} epochs remaining")
            print(f"  Phase1 epochs: {self.phase1_epochs} (restored from checkpoint)")
        
        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch

            # Update training state for progressive crop (dataloader sync)
            if self.use_progressive_crop:
                try:
                    from utils.training_state import get_training_state
                    state = get_training_state()
                    state._current_epoch = epoch
                    state.progressive_crop.current_crop_size = self.current_crop_size
                    if epoch % 3 == 0:
                        print(f"[Finetuner PC] epoch={epoch}, current_crop_size={self.current_crop_size}, state.crop={state.progressive_crop.current_crop_size}")
                except ImportError:
                    pass

            # Check for crop size change (for progressive mode)
            # On first resumed epoch, use saved crop; for subsequent epochs, check schedule
            if self.use_progressive_crop:
                new_crop_size = self._get_current_crop_size(epoch)
                expected_crop = new_crop_size

                # Only update if we're past the expected transition point
                # and current crop doesn't match expected (to handle resume correctly)
                if epoch == start_epoch:
                    # First epoch of resume - use saved crop
                    pass
                elif new_crop_size != self.current_crop_size:
                    self._update_crop_size(new_crop_size)

            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Build training summary with crop size
            crop_info = f" [crop={self.current_crop_size}]" if self.use_progressive_crop else ""
            summary = f"Epoch {epoch+1}/{self.epochs}{crop_info}: loss={train_metrics['total']:.4f}"
            if 'perceptual' in train_metrics:
                summary += f", perc={train_metrics['perceptual']:.4f}"
            if 'line_art' in train_metrics:
                summary += f", line={train_metrics['line_art']:.4f}"
            if 'flat' in train_metrics:
                summary += f", flat={train_metrics['flat']:.4f}"
            if 'adversarial' in train_metrics:
                summary += f", adv={train_metrics['adversarial']:.4f}"
            if 'discriminator' in train_metrics:
                summary += f", D={train_metrics['discriminator']:.4f}"
            if 'frequency' in train_metrics:
                summary += f", freq={train_metrics['frequency']:.4f}"
            print(summary)
            
            # Validate
            val_metrics = None
            if val_loader is not None and (epoch + 1) % self.val_interval == 0:
                val_metrics = self.validate(val_loader)
                val_summary = f"  Validation: loss={val_metrics['total']:.4f}, PSNR={val_metrics['psnr']:.2f}, SSIM={val_metrics['ssim']:.4f}, LPIPS={val_metrics.get('lpips', 0.0):.4f}"
                if 'perceptual' in val_metrics:
                    val_summary += f", perc={val_metrics['perceptual']:.4f}"
                print(val_summary)
                
                # Use LPIPS for best checkpoint selection if early stopping uses LPIPS
                if self.early_stopping_metric == 'lpips':
                    current_best = val_metrics.get('lpips', 1.0)
                    is_best = current_best < self.best_loss - 1e-6
                    if is_best:
                        self.best_loss = current_best
                        self.save_checkpoint(epoch, is_best=True, val_metrics=val_metrics)
                else:
                    if val_metrics['total'] < self.best_loss:
                        self.best_loss = val_metrics['total']
                        self.save_checkpoint(epoch, is_best=True, val_metrics=val_metrics)
                
                # Check early stopping
                if self._check_early_stopping(val_metrics):
                    print(f"\nStopping training at epoch {epoch+1}")
                    break
            
            # Log to TensorBoard
            self._log_to_tensorboard(epoch, train_metrics, val_metrics)
            
            # Step scheduler
            self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(epoch, val_metrics=val_metrics)
        
        # Final save
        self.save_checkpoint(self.epochs - 1, val_metrics=val_metrics)
        
        # Close TensorBoard
        if self.writer is not None:
            self.writer.close()
        
        print("\n" + "="*50)
        print(f"Fine-tuning complete! Best loss: {self.best_loss:.4f}")
        print(f"Checkpoint: {self.checkpoint_dir / 'finetune_best.pth'}")
        print("="*50)


def create_finetuner(config: Dict, device: str = 'cuda', checkpoint_manager=None) -> NeosrSPANFinetuner:
    """Create finetuner from config."""
    return NeosrSPANFinetuner(config, device, checkpoint_manager)
