"""
PreprocessingManager - Unified preprocessing orchestrator.

Handles all preprocessing modes:
- on_the_fly: CPU-based degradation during training
- precomputed: Load pre-generated LR/HR pairs from disk
- gpu_degradation: GPU-accelerated degradation (speed focus)
- hybrid: Precomputed base + runtime augmentation (DEFAULT)
- quality_adaptive: Per-image quality-based degradation selection

Usage:
    manager = PreprocessingManager(config, device)
    lr = manager.process(hr_tensor)  # Returns LR tensor
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple, Callable
from pathlib import Path
import numpy as np
from PIL import Image

try:
    from data.anime_degradation import (
        AnimeBlur, DirectionalBlur, ColorQuantization, 
        BandingArtifact, RingingArtifact
    )
    from data.quality_analyzer import QualityAnalyzer
    from data.compression_modules import CompressionPipeline
except ImportError:
    try:
        from src.data.anime_degradation import (
            AnimeBlur, DirectionalBlur, ColorQuantization, 
            BandingArtifact, RingingArtifact
        )
        from src.data.quality_analyzer import QualityAnalyzer
        from src.data.compression_modules import CompressionPipeline
    except ImportError:
        from anime_degradation import (
            AnimeBlur, DirectionalBlur, ColorQuantization, 
            BandingArtifact, RingingArtifact
        )
        from quality_analyzer import QualityAnalyzer
        from compression_modules import CompressionPipeline


class PreprocessingManager:
    """
    Unified preprocessing orchestrator.
    
    Selects and routes to appropriate handler based on mode configuration.
    Default mode is 'hybrid' (precomputed base + runtime augmentation).
    """
    
    MODES = ['on_the_fly', 'precomputed', 'gpu_degradation', 'hybrid', 'quality_adaptive']
    
    def __init__(
        self,
        config: Dict,
        device: torch.device = None
    ):
        """
        Initialize PreprocessingManager.
        
        Args:
            config: Preprocessing configuration dict
            device: torch device for GPU operations
        """
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get mode
        self.mode = config.get('mode', 'hybrid')
        if self.mode not in self.MODES:
            raise ValueError(f"Invalid mode '{self.mode}'. Must be one of {self.MODES}")
        
        # Initialize handlers based on mode
        self._init_handlers()
    
    def _init_handlers(self):
        """Initialize appropriate handlers for the selected mode."""
        # Always init base capabilities
        self._init_quality_analyzer()
        self._init_line_enhancement()
        
        # Mode-specific initialization
        if self.mode in ['on_the_fly', 'quality_adaptive']:
            self._init_on_the_fly_handler()
        
        if self.mode in ['precomputed', 'hybrid']:
            self._init_precomputed_handler()
        
        if self.mode in ['gpu_degradation', 'hybrid']:
            self._init_gpu_degradation_handler()
        
        if self.mode == 'quality_adaptive':
            # Quality adaptive uses on_the_fly with quality-based config selection
            pass
    
    def _init_quality_analyzer(self):
        """Initialize quality analyzer if enabled in config."""
        qa_config = self.config.get('quality_analyzer', {})
        if qa_config.get('enabled', True):
            cache_scores = qa_config.get('cache_scores', True)
            self.quality_analyzer = QualityAnalyzer(cache_scores=cache_scores)
        else:
            self.quality_analyzer = None
    
    def _init_line_enhancement(self):
        """Initialize line enhancement (XDoG) if enabled."""
        le_config = self.config.get('line_enhancement', {})
        if le_config.get('enabled', False):
            try:
                from src.data.line_enhancement import LineEnhancer
            except ImportError:
                try:
                    from data.line_enhancement import LineEnhancer
                except ImportError:
                    from line_enhancement import LineEnhancer
            
            self.line_enhancer = LineEnhancer(
                sigma1=le_config.get('sigma1', 1.0),
                sigma2=le_config.get('sigma2', 16.0),
                alpha=le_config.get('alpha', 0.1),
                gamma=le_config.get('gamma', 0.5),
            )
        else:
            self.line_enhancer = None
    
    def _init_on_the_fly_handler(self):
        """Initialize on-the-fly degradation handler."""
        otf_config = self.config.get('on_the_fly', {})
        
        self.degrade_before_crop = otf_config.get('degrade_before_crop', True)
        self.shuffled_resize = otf_config.get('shuffled_resize', True)
        self.two_stage_compression = otf_config.get('two_stage_compression', True)
        
        # Compression pipelines
        stage1_types = otf_config.get('compression_stage1', ['jpeg', 'webp'])
        stage2_types = otf_config.get('compression_stage2', ['avif', 'h264', 'jpeg'])
        
        if self.two_stage_compression and CompressionPipeline is not None:
            try:
                self.stage1_pipeline = CompressionPipeline(
                    compression_types=stage1_types,
                    quality_ranges=[(60, 95)] * len(stage1_types),
                )
                self.stage2_pipeline = CompressionPipeline(
                    compression_types=stage2_types,
                    quality_ranges=[(60, 90)] * len(stage2_types),
                )
            except Exception:
                self.stage1_pipeline = None
                self.stage2_pipeline = None
        else:
            self.stage1_pipeline = None
            self.stage2_pipeline = None
        
        # On-the-fly degradation config
        self.otf_degradation_config = otf_config.get('degradation', {})
    
    def _init_precomputed_handler(self):
        """Initialize precomputed dataset handler.

        Phase 4.5: pre-flight check. If the user has set
        `hybrid.precomputed_base: true` (or mode is `precomputed`) but the
        configured `base_dir` does not exist, raise FileNotFoundError with a
        clear actionable message. This catches typos and forgotten
        `precompute_pairs.py` runs at startup, before the first batch.

        Opt-in rules (Issue #4 — loosen pre-flight trigger):
        - `mode='precomputed'`: preflight fires only if `precomputed.enabled` is True (default True).
          Set `precomputed.enabled: false` to skip the preflight (e.g. for tests/smoke runs).
        - `mode='hybrid'`: preflight fires only if `hybrid.precomputed_base` is True.
        - All other modes: never fire the preflight (no precomputed dependency).
        """
        pc_config = self.config.get('precomputed', {})
        hybrid_config = self.config.get('hybrid', {})

        if self.mode == 'hybrid' and hybrid_config.get('precomputed_base'):
            self.precomputed_base_dir = hybrid_config.get('base_dir', 'data/precomputed_4x')
        else:
            self.precomputed_base_dir = pc_config.get('base_dir', 'data/precomputed_4x')

        self.cache_in_memory = pc_config.get('cache_in_memory', False)

        base_path = Path(self.precomputed_base_dir)
        self.precomputed_available = base_path.exists() and (base_path / 'lr').exists() and (base_path / 'hr').exists()

        # Pre-flight: only fire when the user explicitly opted in.
        opt_in_precomputed = (
            (self.mode == 'precomputed' and pc_config.get('enabled', True))
            or (self.mode == 'hybrid' and hybrid_config.get('precomputed_base'))
        )
        if opt_in_precomputed and not self.precomputed_available:
            if self.mode == 'precomputed':
                msg = (
                    f"PreprocessingManager: mode='precomputed' but base_dir "
                    f"'{self.precomputed_base_dir}' does not exist (or missing 'lr'/'hr' subdirs). "
                    f"Run `python scripts/precompute_pairs.py` to populate it, or set "
                    f"`precomputed.enabled: false` to fall back to on-the-fly degradation."
                )
            else:
                msg = (
                    f"PreprocessingManager: hybrid.precomputed_base=true but base_dir "
                    f"'{self.precomputed_base_dir}' does not exist (or missing 'lr'/'hr' subdirs). "
                    f"Run `python scripts/precompute_pairs.py` to populate it, or set "
                    f"`hybrid.precomputed_base: false` to fall back to on-the-fly degradation."
                )
            raise FileNotFoundError(msg)

        if self.precomputed_available:
            lr_dir = base_path / 'lr'
            hr_dir = base_path / 'hr'
            self.lr_files = sorted(lr_dir.glob('*.png')) + sorted(lr_dir.glob('*.jpg'))
            self.hr_files = sorted(hr_dir.glob('*.png')) + sorted(hr_dir.glob('*.jpg'))

            if self.cache_in_memory:
                from PIL import Image
                self._lr_cache = []
                for f in self.lr_files:
                    with Image.open(f) as img:
                        self._lr_cache.append(np.array(img))
                self._hr_cache = []
                for f in self.hr_files:
                    with Image.open(f) as img:
                        self._hr_cache.append(np.array(img))
            else:
                self._lr_cache = None
                self._hr_cache = None
        else:
            self.lr_files = []
            self.hr_files = []
            self._lr_cache = None
            self._hr_cache = None
    
    def _init_gpu_degradation_handler(self):
        """Initialize GPU degradation handler (speed focus)."""
        gpu_config = self.config.get('gpu_degradation', {})
        modules_config = gpu_config.get('modules', {})
        
        # Initialize GPU modules
        self.gpu_modules = nn.ModuleDict()
        
        if modules_config.get('blur', True):
            self.gpu_modules['blur'] = AnimeBlur().to(self.device)
        
        if modules_config.get('directional_blur', True):
            self.gpu_modules['directional_blur'] = DirectionalBlur().to(self.device)
        
        if modules_config.get('quantization', True):
            self.gpu_modules['quantization'] = ColorQuantization().to(self.device)
        
        if modules_config.get('banding', True):
            self.gpu_modules['banding'] = BandingArtifact().to(self.device)
        
        if modules_config.get('ringing', True):
            self.gpu_modules['ringing'] = RingingArtifact().to(self.device)
        
        # Probabilities for each module
        self.gpu_blur_prob = gpu_config.get('blur_prob', 0.5)
        self.gpu_noise_prob = gpu_config.get('noise_prob', 0.4)
        self.gpu_quant_prob = gpu_config.get('quantization_prob', 0.3)
    
    def process(self, hr_tensor: torch.Tensor, idx: int = 0, hr_path: Path = None) -> torch.Tensor:
        """
        Main entry point - process HR tensor to LR tensor.
        
        Args:
            hr_tensor: HR image tensor [B, C, H, W] in range [0, 1]
            idx: Index for precomputed loading (if applicable)
            hr_path: Path to HR image for hybrid mode (to find matching LR file)
        
        Returns:
            LR tensor [B, C, H/scale, W/scale]
        """
        if self.mode == 'on_the_fly':
            return self._process_on_the_fly(hr_tensor)
        elif self.mode == 'precomputed':
            return self._process_precomputed(idx)
        elif self.mode == 'gpu_degradation':
            return self._process_gpu_degradation(hr_tensor)
        elif self.mode == 'hybrid':
            if hr_path:
                return self._process_hybrid(hr_path)
            else:
                return self._process_hybrid_fallback(hr_tensor, idx)
        elif self.mode == 'quality_adaptive':
            return self._process_quality_adaptive(hr_tensor)
        else:
            raise NotImplementedError(f"Mode '{self.mode}' not implemented")
    
    def _process_hybrid_fallback(self, hr_tensor: torch.Tensor, idx: int) -> torch.Tensor:
        """Fallback for hybrid mode when hr_path is not available."""
        if not self.precomputed_available:
            return self._process_on_the_fly(hr_tensor)
        
        if self._lr_cache is not None:
            lr = self._lr_cache[idx]
        else:
            with Image.open(self.lr_files[idx]) as img:
                lr = np.array(img)
        
        lr_tensor = torch.from_numpy(lr).permute(2, 0, 1).float() / 255.0
        lr_tensor = lr_tensor.to(self.device)
        lr_tensor = self._apply_runtime_augmentation(lr_tensor)
        return lr_tensor
    
    def _process_on_the_fly(self, hr_tensor: torch.Tensor) -> torch.Tensor:
        """On-the-fly CPU-based degradation using the shared DegradationPipeline.

        Respects the configured preset/two_stage and degrades on the full image
        before resize (APISR-style 'degrade-before-crop'). Returns a CxHxW
        float32 tensor in [0, 1] on self.device.
        """
        from data.degradation_pipeline import DegradationPipeline

        if hr_tensor.dim() == 4:
            hr_tensor = hr_tensor[0]
        hr_np = hr_tensor.detach().cpu().numpy()
        if hr_np.shape[0] in (1, 3) and hr_np.ndim == 3:
            hr_np = np.transpose(hr_np, (1, 2, 0))
        if hr_np.dtype != np.uint8:
            hr_np = (hr_np * 255.0).clip(0, 255).astype(np.uint8)

        cfg = self.config.get('preprocessing', {}) or {}
        otf_cfg = cfg.get('on_the_fly', {}) or {}
        pipeline = DegradationPipeline(
            mode=otf_cfg.get('mode', self.config.get('mode', 'anime_heavy')),
            cfg=otf_cfg.get('cfg'),
            scale=self.config.get('scale', 4),
            two_stage=otf_cfg.get('two_stage', False),
            compression_stage1=otf_cfg.get('compression_stage1'),
            compression_stage2=otf_cfg.get('compression_stage2'),
            seed=otf_cfg.get('seed'),
        )
        lr_np = pipeline(hr_np)
        lr_tensor = torch.from_numpy(lr_np).permute(2, 0, 1).float() / 255.0
        return lr_tensor.to(self.device)
    
    def _process_precomputed(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load precomputed LR/HR pair."""
        if not self.precomputed_available:
            raise RuntimeError(
                f"Precomputed data not found at {self.precomputed_base_dir}. "
                "Run precompute_pairs.py first or use 'hybrid' mode."
            )
        
        if self._lr_cache is not None:
            lr = self._lr_cache[idx]
            hr = self._hr_cache[idx]
        else:
            with Image.open(self.lr_files[idx]) as img:
                lr = np.array(img)
            with Image.open(self.hr_files[idx]) as img:
                hr = np.array(img)
        
        # Convert to tensors
        lr_tensor = torch.from_numpy(lr).permute(2, 0, 1).float() / 255.0
        hr_tensor = torch.from_numpy(hr).permute(2, 0, 1).float() / 255.0
        
        return lr_tensor.to(self.device), hr_tensor.to(self.device)
    
    def _process_gpu_degradation(self, hr_tensor: torch.Tensor) -> torch.Tensor:
        """GPU-accelerated degradation (speed focus)."""
        x = hr_tensor.to(self.device)
        generator = torch.Generator(device=self.device)
        
        # Apply enabled GPU modules
        if 'blur' in self.gpu_modules and torch.rand(1, generator=generator) < self.gpu_blur_prob:
            x = self.gpu_modules['blur'](x, generator)
        
        if 'directional_blur' in self.gpu_modules and torch.rand(1, generator=generator) < 0.3:
            x = self.gpu_modules['directional_blur'](x, generator)
        
        if 'quantization' in self.gpu_modules and torch.rand(1, generator=generator) < self.gpu_quant_prob:
            x = self.gpu_modules['quantization'](x, generator)
        
        # Downsample for LR
        scale = self.config.get('scale', 4)
        lr_h = x.shape[2] // scale
        lr_w = x.shape[3] // scale
        x = torch.nn.functional.interpolate(x, size=(lr_h, lr_w), mode='bicubic')
        
        if 'banding' in self.gpu_modules and torch.rand(1, generator=generator) < 0.4:
            x = self.gpu_modules['banding'](x, generator)
        
        if 'ringing' in self.gpu_modules and torch.rand(1, generator=generator) < 0.3:
            x = self.gpu_modules['ringing'](x, generator)
        
        return x
    
    def _process_hybrid(self, hr_path: Path, precomputed_base_dir: Path = None) -> torch.Tensor:
        """
        Hybrid mode: Load precomputed base + apply runtime augmentation.
        
        This is the DEFAULT mode for best speed/quality balance.
        
        Args:
            hr_path: Path to the HR image (to find matching LR)
            precomputed_base_dir: Base directory for precomputed data (uses self.precomputed_base_dir if None)
        """
        base_dir = precomputed_base_dir or Path(self.precomputed_base_dir)
        
        if not self.precomputed_available:
            return self._process_on_the_fly(torch.from_numpy(np.zeros((1,1,3), dtype=np.uint8)))
        
        lr_name = hr_path.name
        lr_path = base_dir / 'lr' / lr_name
        
        if not lr_path.exists():
            lr_path = base_dir / 'lr' / lr_name.replace('.png', '.jpg')
        
        # Derive fallback LR size from the HR image and scale factor
        with Image.open(hr_path) as hr_img:
            hr_w, hr_h = hr_img.size
        scale = self.config.get('scale', 4)
        fallback_lr = np.zeros((max(1, hr_h // scale), max(1, hr_w // scale), 3), dtype=np.uint8)
        
        if lr_path.exists():
            with Image.open(lr_path) as lr_img:
                lr = np.array(lr_img)
        elif self._lr_cache:
            idx = self._find_file_index(hr_path.name)
            if idx is not None and idx < len(self._lr_cache):
                lr = self._lr_cache[idx]
            else:
                lr = fallback_lr
        else:
            lr = fallback_lr
        
        lr_tensor = torch.from_numpy(lr).permute(2, 0, 1).float() / 255.0
        lr_tensor = lr_tensor.to(self.device)
        
        lr_tensor = self._apply_runtime_augmentation(lr_tensor)
        
        return lr_tensor
    
    def _find_file_index(self, filename: str) -> Optional[int]:
        """Find the index of a file by name in lr_files."""
        for i, f in enumerate(self.lr_files):
            if f.name == filename:
                return i
        return None
    
    def _apply_runtime_augmentation(self, lr_tensor: torch.Tensor) -> torch.Tensor:
        """Apply runtime augmentation for hybrid mode."""
        x = lr_tensor.unsqueeze(0) if lr_tensor.dim() == 3 else lr_tensor
        generator = torch.Generator(device=self.device)
        
        hybrid_config = self.config.get('hybrid', {})
        runtime_config = hybrid_config.get('runtime_augmentation', {})
        
        # Blur
        if runtime_config.get('blur_prob', 0.3) > 0 and torch.rand(1) < runtime_config.get('blur_prob', 0.3):
            if 'blur' in self.gpu_modules:
                x = self.gpu_modules['blur'](x, generator)
        
        # Noise (additive Gaussian)
        noise_sigma = runtime_config.get('noise_sigma', [0, 15])
        if isinstance(noise_sigma, list) and len(noise_sigma) == 2:
            sigma = torch.rand(1).item() * (noise_sigma[1] - noise_sigma[0]) + noise_sigma[0]
            if sigma > 0:
                noise = torch.randn_like(x) * sigma / 255.0
                x = x + noise
        
        # Color quantization
        if runtime_config.get('quantization', True) and torch.rand(1) < runtime_config.get('quantization_prob', 0.3):
            if 'quantization' in self.gpu_modules:
                x = self.gpu_modules['quantization'](x, generator)
        
        return x.squeeze(0) if lr_tensor.dim() == 3 else x
    
    def _process_quality_adaptive(self, hr_tensor: torch.Tensor) -> torch.Tensor:
        """Quality-adaptive degradation based on per-image analysis.

        The HR image is scored by the quality analyzer; the score maps to a
        tier ('light' | 'medium' | 'heavy'), which selects a DegradationPipeline
        preset. The selected pipeline is then applied to the full image.
        """
        from data.degradation_pipeline import DegradationPipeline

        if hr_tensor.dim() == 4:
            hr_tensor = hr_tensor[0]
        hr_np = hr_tensor.detach().cpu().numpy()
        if hr_np.shape[0] in (1, 3) and hr_np.ndim == 3:
            hr_np = np.transpose(hr_np, (1, 2, 0))
        if hr_np.dtype != np.uint8:
            hr_np = (hr_np * 255.0).clip(0, 255).astype(np.uint8)

        if self.quality_analyzer is None:
            return self._process_on_the_fly(hr_tensor)

        quality_score = self.quality_analyzer.analyze_image(hr_np)
        tier = self.quality_analyzer.get_degradation_tier(quality_score)

        qa_cfg = self.config.get('preprocessing', {}).get('quality_adaptive', {}) or {}
        tier_to_preset = {
            'light': qa_cfg.get('light_degradation', 'light'),
            'medium': qa_cfg.get('medium_degradation', 'medium'),
            'heavy': qa_cfg.get('heavy_degradation', 'heavy'),
        }
        preset = tier_to_preset.get(tier, 'medium')
        pipeline = DegradationPipeline(
            mode=preset,
            scale=self.config.get('scale', 4),
            two_stage=qa_cfg.get('two_stage', False),
            compression_stage1=qa_cfg.get('compression_stage1'),
            compression_stage2=qa_cfg.get('compression_stage2'),
            seed=qa_cfg.get('seed'),
        )
        lr_np = pipeline(hr_np)
        lr_tensor = torch.from_numpy(lr_np).permute(2, 0, 1).float() / 255.0
        return lr_tensor.to(self.device)
    
    def get_info(self) -> Dict:
        """Get information about the preprocessing configuration."""
        return {
            'mode': self.mode,
            'precomputed_available': getattr(self, 'precomputed_available', None),
            'gpu_modules': list(self.gpu_modules.keys()) if hasattr(self, 'gpu_modules') else [],
            'line_enhancement': self.line_enhancer is not None if hasattr(self, 'line_enhancer') else False,
            'quality_analyzer': self.quality_analyzer is not None if hasattr(self, 'quality_analyzer') else False,
        }
    
    def __repr__(self) -> str:
        return f"PreprocessingManager(mode={self.mode}, device={self.device})"


class OnTheFlyProcessor:
    """Handles on-the-fly degradation (legacy from base.py)."""
    
    def __init__(self, config: Dict, device: torch.device = None):
        self.config = config
        self.device = device or torch.device('cpu')
        
        # APISR-style options
        self.degrade_before_crop = config.get('degrade_before_crop', True)
        self.shuffled_resize = config.get('shuffled_resize', True)
        self.two_stage_compression = config.get('two_stage_compression', True)
        
        # Initialize compression pipelines if enabled
        if self.two_stage_compression:
            self._init_compression_pipelines()
    
    def _init_compression_pipelines(self):
        """Initialize compression pipelines for two-stage degradation."""
        stage1_types = self.config.get('compression_stage1', ['jpeg', 'webp'])
        stage2_types = self.config.get('compression_stage2', ['avif', 'h264', 'jpeg'])
        
        if CompressionPipeline is not None:
            try:
                self.stage1_pipeline = CompressionPipeline(
                    compression_types=stage1_types,
                    quality_ranges=[(60, 95)] * len(stage1_types),
                )
                self.stage2_pipeline = CompressionPipeline(
                    compression_types=stage2_types,
                    quality_ranges=[(60, 90)] * len(stage2_types),
                )
            except Exception:
                self.stage1_pipeline = None
                self.stage2_pipeline = None
    
    def process(self, hr_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply on-the-fly degradation.
        
        Args:
            hr_tensor: HR tensor [B, C, H, W]
        
        Returns:
            LR tensor [B, C, H/scale, W/scale]
        """
        raise NotImplementedError(
            "OnTheFlyProcessor.process is a legacy stub that only performed "
            "bicubic downsampling. Use PreprocessingManager (mode='on_the_fly') "
            "which runs the full DegradationPipeline instead."
        )


if __name__ == "__main__":
    # Test PreprocessingManager creation
    config = {
        'mode': 'hybrid',
        'scale': 4,
        'gpu_degradation': {
            'enabled': True,
            'modules': {'blur': True, 'quantization': True}
        },
        'line_enhancement': {'enabled': False}
    }
    
    manager = PreprocessingManager(config)
    print(manager)
    print(manager.get_info())