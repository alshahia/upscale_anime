"""
Base dataset class for Super-Resolution
Supports multiple datasets, degradation pipelines, and flexible data loading
Includes auto crop_size detection and smart quality-aware degradation
"""
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import cv2
from typing import Optional, Tuple, List, Dict, Callable, Any
import torchvision.transforms as transforms
import warnings

try:
    from src.data.quality_analyzer import (
        QualityAnalyzer, compute_optimal_crop_size, get_degradation_preset_config
    )
    from src.data.compression_modules import CompressionPipeline, JPEGCompression, WebPCompression, AVIFCompression, VideoCodecCompression
except ImportError:
    try:
        from data.quality_analyzer import (
            QualityAnalyzer, compute_optimal_crop_size, get_degradation_preset_config
        )
        from data.compression_modules import CompressionPipeline, JPEGCompression, WebPCompression, AVIFCompression, VideoCodecCompression
    except ImportError:
        from quality_analyzer import (
            QualityAnalyzer, compute_optimal_crop_size, get_degradation_preset_config
        )
        CompressionPipeline = None
        JPEGCompression = None
        WebPCompression = None
        AVIFCompression = None
        VideoCodecCompression = None


class BaseDataset(Dataset):
    """
    Universal base dataset for Super-Resolution.
    Handles HR/LR image loading, degradation, and augmentation.
    """
    
    def __init__(
        self,
        hr_dir: str,
        lr_dir: Optional[str] = None,
        scale: int = 4,
        crop_size: int = 128,
        augment: bool = True,
        degradation: Optional[Dict] = None,
        preload: bool = False,
        max_images: Optional[int] = None,
        gpu_degradation: bool = False,
        crop_size_mode: str = 'manual',
        auto_crop_max: int = 960,
        auto_crop_min: int = 64,
        use_quality_analyzer: bool = False,
        quality_analyzer: Optional[QualityAnalyzer] = None,
        degrade_before_crop: bool = False,
        shuffled_resize: bool = False,
        two_stage_compression: bool = False,
        compression_stage1: Optional[List[str]] = None,
        compression_stage2: Optional[List[str]] = None,
        line_enhancement: Optional[Dict] = None,
        preprocessing_manager: Optional[Any] = None,
        sample_ratio: Optional[float] = None,
    ):
        """
        Args:
            hr_dir: Directory with high-resolution images
            lr_dir: Directory with low-resolution images (optional)
            scale: Upsampling factor
            crop_size: Training patch size (HR space)
            augment: Enable random flip/rotate
            degradation: Degradation config for synthetic LR generation
            preload: Load all images into RAM
            max_images: Limit dataset size (for testing)
            crop_size_mode: 'auto' to detect from dataset, 'manual' to use crop_size
            auto_crop_max: Maximum crop size in auto mode (prevents OOM)
            auto_crop_min: Minimum crop size
            use_quality_analyzer: Enable per-image quality-based degradation
            quality_analyzer: Optional pre-configured QualityAnalyzer instance
            degrade_before_crop: Apply degradation on full image before cropping (APISR style)
            shuffled_resize: Randomize order of degradation operations
            two_stage_compression: Use two-stage compression pipeline
            compression_stage1: List of compression types for stage 1 (before resize)
            compression_stage2: List of compression types for stage 2 (after resize)
            line_enhancement: Config for XDoG-based line enhancement
            preprocessing_manager: PreprocessingManager instance for unified preprocessing
            sample_ratio: Optional ratio (0.0-1.0) to sample from dataset
        """
        super().__init__()
        
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir) if lr_dir else None
        self.scale = scale
        self.crop_size = crop_size
        self.augment = augment
        self.degradation = degradation or {}
        self.preload = preload
        self.sample_ratio = sample_ratio  # NEW: for dataset sampling
        
        # Validate directory exists
        if not self.hr_dir.exists():
            raise FileNotFoundError(
                f"HR directory does not exist: {self.hr_dir}\n"
                f"Please create the directory and add images, or update the config.\n"
                f"To create dummy test data, run:\n"
                f"  python -c \"from data.data_utils import create_dummy_dataset; "
                f"create_dummy_dataset('{self.hr_dir}', num_images=10)\""
            )
        
        # Get image list
        self.hr_images = sorted(self.hr_dir.glob("*.png")) + \
                         sorted(self.hr_dir.glob("*.jpg")) + \
                         sorted(self.hr_dir.glob("*.jpeg"))
        
        # Validate images found
        if len(self.hr_images) == 0:
            raise ValueError(
                f"No images found in {self.hr_dir}\n"
                f"Supported formats: .png, .jpg, .jpeg\n"
                f"To create dummy test data, run:\n"
                f"  python -c \"from data.data_utils import create_dummy_dataset; "
                f"create_dummy_dataset('{self.hr_dir}', num_images=10)\""
            )
        
        if max_images:
            self.hr_images = self.hr_images[:max_images]
        
        # Apply dataset sampling if sample_ratio provided (NEW)
        # This is applied AFTER max_images but BEFORE any other processing
        if hasattr(self, 'sample_ratio') and self.sample_ratio is not None and self.sample_ratio < 1.0:
            num_to_sample = int(len(self.hr_images) * self.sample_ratio)
            self.hr_images = self.hr_images[:num_to_sample]
            print(f"[Dataset Sampling] Using {len(self.hr_images)} images ({self.sample_ratio:.0%} of total)")
        
        print(f"Dataset loaded: {len(self.hr_images)} images from {self.hr_dir}")
        
        # Auto-detect optimal crop size if enabled
        if crop_size_mode == 'auto':
            computed_crop = compute_optimal_crop_size(
                self.hr_images,
                mode='auto',
                max_size=auto_crop_max,
                min_size=auto_crop_min
            )
            if computed_crop != self.crop_size:
                print(f"[Auto Crop] Adjusted crop_size from {self.crop_size} to {computed_crop}")
                self.crop_size = computed_crop
        elif crop_size_mode == 'manual' and len(self.hr_images) > 0:
            # Warn if manual crop_size might be inappropriate for dataset
            self._validate_crop_size()
        
        # Initialize quality analyzer if enabled
        self.use_quality_analyzer = use_quality_analyzer
        self.quality_analyzer = quality_analyzer
        if use_quality_analyzer:
            if self.quality_analyzer is None:
                self.quality_analyzer = QualityAnalyzer()
            print(f"[Quality Analyzer] Enabled - per-image quality-based degradation")
            if self.degradation.get('mode') == 'auto':
                self._setup_auto_degradation()

        # APISR-style degradation options
        self.degrade_before_crop = degrade_before_crop
        self.shuffled_resize = shuffled_resize
        self.two_stage_compression = two_stage_compression
        self.compression_stage1 = compression_stage1 or ['jpeg', 'webp']
        self.compression_stage2 = compression_stage2 or ['avif', 'h264', 'jpeg']

        # Build compression pipelines
        self.stage1_pipeline = None
        self.stage2_pipeline = None
        if two_stage_compression and CompressionPipeline is not None:
            try:
                stage1_ranges = self._get_quality_ranges(self.compression_stage1, default=(60, 95))
                stage2_ranges = self._get_quality_ranges(self.compression_stage2, default=(60, 90))
                self.stage1_pipeline = CompressionPipeline(
                    compression_types=self.compression_stage1,
                    quality_ranges=stage1_ranges,
                )
                self.stage2_pipeline = CompressionPipeline(
                    compression_types=self.compression_stage2,
                    quality_ranges=stage2_ranges,
                )
                print(f"[Compression] Two-stage: stage1={self.compression_stage1}, stage2={self.compression_stage2}")
            except Exception as e:
                print(f"[Compression] WARNING: Failed to init pipelines: {e}")

        # Line enhancement
        self.line_enhancement = line_enhancement
        # apply_to_gt gates whether the enhancer is applied to HR (pseudo-GT).
        # Defaults to True to preserve existing behavior (was always applied when enabled).
        self.apply_to_gt = (
            line_enhancement.get('apply_to_gt', True)
            if isinstance(line_enhancement, dict) else True
        )
        if line_enhancement and line_enhancement.get('enabled', False):
            try:
                from src.data.line_enhancement import LineEnhancer
            except ImportError:
                try:
                    from data.line_enhancement import LineEnhancer
                except ImportError:
                    from line_enhancement import LineEnhancer
            try:
                self.line_enhancer = LineEnhancer(
                    sigma1=line_enhancement.get('sigma1', 1.0),
                    sigma2=line_enhancement.get('sigma2', 16.0),
                    alpha=line_enhancement.get('alpha', 0.1),
                    gamma=line_enhancement.get('gamma', 0.5),
                    pseudo_gt_mode=line_enhancement.get('pseudo_gt_mode', 'lite'),
                    usm_rounds=line_enhancement.get('usm_rounds', 3),
                    usm_radius=line_enhancement.get('usm_radius', 50),
                    usm_sigma=line_enhancement.get('usm_sigma', 0),
                    usm_threshold=line_enhancement.get('usm_threshold', 10),
                    usm_weight=line_enhancement.get('usm_weight', 0.5),
                    xdog_sigma=line_enhancement.get('xdog_sigma', 0.6),
                    xdog_k=line_enhancement.get('xdog_k', 2.5),
                    xdog_gamma=line_enhancement.get('xdog_gamma', 0.97),
                    xdog_eps=line_enhancement.get('xdog_eps', -15.0),
                    xdog_phi=line_enhancement.get('xdog_phi', 1e9),
                    outlier_min_size=line_enhancement.get('outlier_min_size', 32),
                    dilation_threshold=line_enhancement.get('dilation_threshold', 3),
                )
                le_mode = line_enhancement.get('pseudo_gt_mode', 'lite')
                print(
                    f"[Line Enhancement] Enabled (mode={le_mode}, "
                    f"sigma1={line_enhancement.get('sigma1', 1.0)}, "
                    f"alpha={line_enhancement.get('alpha', 0.1)}, "
                    f"apply_to_gt={self.apply_to_gt})"
                )
            except Exception as e:
                print(f"[Line Enhancement] WARNING: Failed to init: {e}")
                self.line_enhancer = None
        else:
            self.line_enhancer = None

        # PreprocessingManager for unified preprocessing (NEW)
        self.preprocessing_manager = preprocessing_manager
        
        # Preload images if requested
        self.gpu_degradation = gpu_degradation
        
        self.cached_hr = {}
        if preload:
            for img_path in self.hr_images:
                self.cached_hr[img_path] = self._load_image(img_path)
        
        # Precompute LR paths if provided
        if self.lr_dir:
            self.lr_paths = {}
            for hr_path in self.hr_images:
                lr_name = hr_path.name
                lr_path = self.lr_dir / lr_name
                if lr_path.exists():
                    self.lr_paths[hr_path] = lr_path
            print(f"Paired LR images: {len(self.lr_paths)}")
        else:
            self.lr_paths = {}
        
        # Initialize RNG for reproducibility
        self._rng = torch.Generator()
        self._rng.manual_seed(torch.initial_seed() % (2**32))

    def _get_quality_ranges(self, compression_types: List[str], default: Tuple[int, int] = (60, 95)) -> List[Tuple[int, int]]:
        """Get appropriate quality ranges for each compression type."""
        video_codecs = ('h264', 'h265', 'mpeg4', 'mpeg2')
        ranges = []
        for ctype in compression_types:
            if ctype in video_codecs:
                ranges.append((18, 35))
            elif ctype == 'avif':
                ranges.append((30, 80))
            else:
                ranges.append(default)
        return ranges

    def _load_image(self, path: Path) -> np.ndarray:
        """Load image as numpy array [H, W, C], RGB, uint8"""
        try:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to load image: {path}")
            # Validate image dimensions
            if img.ndim != 3:
                raise ValueError(f"Invalid image dimensions (expected 3D, got {img.ndim}D): {path}")
            if img.shape[2] != 3:
                raise ValueError(f"Invalid number of channels (expected 3, got {img.shape[2]}): {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Error loading image {path}: {e}") from e
    
    def _validate_crop_size(self):
        """Validate that crop_size is appropriate for dataset images."""
        try:
            # Sample a few images to check dimensions
            sample_paths = self.hr_images[:5]
            min_h, min_w = float('inf'), float('inf')
            
            for path in sample_paths:
                with Image.open(path) as img:
                    w, h = img.size
                    min_h = min(min_h, h)
                    min_w = min(min_w, w)
            
            smallest_dim = min(min_h, min_w)
            
            if self.crop_size > smallest_dim:
                warnings.warn(
                    f"[BaseDataset] crop_size ({self.crop_size}) is larger than smallest image dimension "
                    f"({smallest_dim}). This will cause upscaling during training. "
                    f"Consider using crop_size_mode='auto' or setting crop_size to {smallest_dim // 2}. "
                    f"Dataset: {self.hr_dir}"
                )
        except Exception:
            # If validation fails, continue silently
            pass
    
    def _setup_auto_degradation(self):
        """Setup auto degradation based on dataset quality analysis."""
        try:
            print("[Auto Degradation] Analyzing dataset quality...")
            from src.data.quality_analyzer import analyze_dataset_quality
            
            stats = analyze_dataset_quality(self.hr_images, sample_size=30)
            
            if 'error' in stats:
                print(f"[Auto Degradation] Failed to analyze: {stats['error']}")
                return
            
            recommended = stats['recommended_preset']
            mean_quality = stats['mean_quality']
            
            print(f"[Auto Degradation] Mean quality: {mean_quality:.2f}")
            print(f"[Auto Degradation] Tier distribution: {stats['tier_distribution']}")
            print(f"[Auto Degradation] Recommended preset: '{recommended}'")
            
            # Apply preset
            preset_config = get_degradation_preset_config(recommended, self.degradation)
            self.degradation.update(preset_config)
            
            print(f"[Auto Degradation] Applied '{recommended}' preset")
            
        except Exception as e:
            print(f"[Auto Degradation] Error during setup: {e}")
    
    def _get_degradation_for_image(self, img: np.ndarray, img_path: Optional[Path] = None) -> Dict:
        """
        Get degradation config for a specific image based on quality.
        
        Returns:
            Degradation config dict appropriate for image quality tier
        """
        if not self.use_quality_analyzer or self.quality_analyzer is None:
            return self.degradation
        
        # Analyze image quality
        quality_score = self.quality_analyzer.analyze_image(img, img_path)
        tier = self.quality_analyzer.get_degradation_tier(quality_score)
        
        # Get preset for this tier
        tier_config = get_degradation_preset_config(tier, self.degradation)
        
        return tier_config
    
    def _apply_degradation_with_quality(self, img: np.ndarray, img_path: Optional[Path] = None) -> np.ndarray:
        """
        Apply degradation with quality-aware selection.
        
        Args:
            img: Input image
            img_path: Optional path for caching
            
        Returns:
            Degraded LR image
        """
        if not self.degradation.get('enabled', False):
            # Simple bicubic downsample
            h, w = img.shape[:2]
            lr_h, lr_w = h // self.scale, w // self.scale
            lr = cv2.resize(img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
            return lr
        
        # Get quality-aware degradation config
        if self.use_quality_analyzer:
            cfg = self._get_degradation_for_image(img, img_path)
        else:
            cfg = self.degradation
        
        # Apply first-order degradation with selected config
        img = self._apply_first_degradation_with_config(img, cfg)
        
        # Resize to LR size
        h, w = img.shape[:2]
        lr_h, lr_w = h // self.scale, w // self.scale
        img = cv2.resize(img, (lr_w, lr_h), interpolation=cv2.INTER_LINEAR)
        
        # Apply second-order degradation
        img = self._apply_second_degradation_with_config(img, cfg)
        
        return img
    
    def _apply_first_degradation_with_config(self, img: np.ndarray, cfg: Dict) -> np.ndarray:
        """Apply first-order degradation with specific config."""
        # Blur
        if random.random() < cfg.get('blur_prob', 0.7):
            kernel_sizes = cfg.get('blur_kernel_size', [7, 9, 11])
            kernel_size = random.choice(kernel_sizes)
            sigma_range = cfg.get('blur_sigma', [0.1, 3.0])
            sigma = random.uniform(*sigma_range[:2])
            img = self._apply_blur(img, kernel_size, sigma)
        
        # Noise
        if random.random() < cfg.get('noise_prob', 0.5):
            sigma_range = cfg.get('noise_sigma', [0, 25])
            sigma = random.uniform(*sigma_range[:2])
            img = self._apply_noise(img, sigma)
        
        # Random resize
        resize_probs = cfg.get('resize_prob', [0.2, 0.7, 0.1])
        resize_type = random.choices(['up', 'down', 'keep'], weights=resize_probs)[0]
        
        if resize_type != 'keep':
            h, w = img.shape[:2]
            resize_range = cfg.get('resize_range', [0.15, 1.5])
            
            if resize_type == 'up':
                factor = random.uniform(1.0, resize_range[1])
            else:
                factor = random.uniform(resize_range[0], 1.0)
            
            new_h, new_w = int(h * factor), int(w * factor)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # JPEG compression
        if random.random() < cfg.get('jpeg_prob', 0.5):
            jpeg_qualities = cfg.get('jpeg_quality', [60, 70, 80, 90, 100])
            quality = random.choice(jpeg_qualities)
            img = self._apply_jpeg(img, quality)
        
        return img
    
    def _apply_second_degradation_with_config(self, img: np.ndarray, cfg: Dict) -> np.ndarray:
        """Apply second-order degradation with specific config."""
        # Use lower probabilities for second-order
        second_order_prob = cfg.get('second_order_prob', 0.3)
        
        if random.random() < second_order_prob:
            kernel_size = random.choice([3, 5, 7])
            sigma = random.uniform(0.1, 1.0)
            img = self._apply_blur(img, kernel_size, sigma)
        
        if random.random() < second_order_prob:
            sigma = random.uniform(0, 10)
            img = self._apply_noise(img, sigma)
        
        if random.random() < second_order_prob:
            jpeg_qualities = cfg.get('jpeg_quality', [70, 80, 90, 100])
            quality = random.choice(jpeg_qualities)
            img = self._apply_jpeg(img, quality)
        
        return img
    
    def _apply_degradation(self, img: np.ndarray) -> np.ndarray:
        """
        Apply RealESRGAN-style high-order degradation to create LR image.
        img: [H, W, C], RGB, uint8
        Returns: LR image at scale factor
        """
        if not self.degradation.get('enabled', False):
            # Simple bicubic downsample
            h, w = img.shape[:2]
            lr_h, lr_w = h // self.scale, w // self.scale
            lr = cv2.resize(img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
            return lr
        
        # First-order degradation
        img = self._first_degradation(img)
        
        # Resize to LR size
        h, w = img.shape[:2]
        lr_h, lr_w = h // self.scale, w // self.scale
        img = cv2.resize(img, (lr_w, lr_h), interpolation=cv2.INTER_LINEAR)
        
        # Second-order degradation
        img = self._second_degradation(img)
        
        return img
    
    def _first_degradation(self, img: np.ndarray) -> np.ndarray:
        """First-order degradation: blur, noise, resize, jpeg"""
        cfg = self.degradation
        
        # Blur
        if random.random() < cfg.get('blur_prob', 0.7):
            kernel_size = random.choice(cfg.get('blur_kernel_size', [7, 9, 11]))
            sigma = random.uniform(*cfg.get('blur_sigma', [0.1, 3.0])[:2])
            img = self._apply_blur(img, kernel_size, sigma)
        
        # Noise
        if random.random() < cfg.get('noise_prob', 0.5):
            sigma = random.uniform(*cfg.get('noise_sigma', [0, 25])[:2])
            img = self._apply_noise(img, sigma)
        
        # Random resize
        resize_probs = cfg.get('resize_prob', [0.2, 0.7, 0.1])
        resize_type = random.choices(['up', 'down', 'keep'], weights=resize_probs)[0]
        
        if resize_type != 'keep':
            h, w = img.shape[:2]
            resize_range = cfg.get('resize_range', [0.15, 1.5])
            
            if resize_type == 'up':
                factor = random.uniform(1.0, resize_range[1])
            else:
                factor = random.uniform(resize_range[0], 1.0)
            
            new_h, new_w = int(h * factor), int(w * factor)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # JPEG compression
        if random.random() < cfg.get('jpeg_prob', 0.5):
            quality = random.choice(cfg.get('jpeg_quality', [60, 70, 80, 90, 100]))
            img = self._apply_jpeg(img, quality)
        
        return img
    
    def _second_degradation(self, img: np.ndarray) -> np.ndarray:
        """Second-order degradation (lighter)"""
        cfg = self.degradation

        if random.random() < 0.3:
            kernel_size = random.choice([3, 5, 7])
            sigma = random.uniform(0.1, 1.0)
            img = self._apply_blur(img, kernel_size, sigma)

        if random.random() < 0.3:
            sigma = random.uniform(0, 10)
            img = self._apply_noise(img, sigma)

        if random.random() < 0.3:
            quality = random.choice([70, 80, 90, 100])
            img = self._apply_jpeg(img, quality)

        return img

    def _apply_shuffled_degradation(self, img: np.ndarray, cfg: Dict) -> np.ndarray:
        """
        APISR-style shuffled degradation: randomize order of operations.
        Operations: blur, noise, resize, compression
        """
        operations = []

        if random.random() < cfg.get('blur_prob', 0.7):
            operations.append('blur')
        if random.random() < cfg.get('noise_prob', 0.5):
            operations.append('noise')
        operations.append('resize')
        if random.random() < cfg.get('jpeg_prob', 0.5):
            operations.append('compression')

        random.shuffle(operations)

        for op in operations:
            if op == 'blur':
                kernel_size = random.choice(cfg.get('blur_kernel_size', [7, 9, 11]))
                sigma = random.uniform(*cfg.get('blur_sigma', [0.1, 3.0])[:2])
                img = self._apply_blur(img, kernel_size, sigma)
            elif op == 'noise':
                sigma = random.uniform(*cfg.get('noise_sigma', [0, 25])[:2])
                img = self._apply_noise(img, sigma)
            elif op == 'resize':
                resize_probs = cfg.get('resize_prob', [0.2, 0.7, 0.1])
                resize_type = random.choices(['up', 'down', 'keep'], weights=resize_probs)[0]
                if resize_type != 'keep':
                    h, w = img.shape[:2]
                    resize_range = cfg.get('resize_range', [0.15, 1.5])
                    factor = random.uniform(1.0, resize_range[1]) if resize_type == 'up' else random.uniform(resize_range[0], 1.0)
                    new_h, new_w = int(h * factor), int(w * factor)
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            elif op == 'compression':
                if self.stage1_pipeline is not None:
                    img = self.stage1_pipeline(img)
                else:
                    jpeg_qualities = cfg.get('jpeg_quality', [60, 70, 80, 90, 100])
                    quality = random.choice(jpeg_qualities)
                    img = self._apply_jpeg(img, quality)

        return img

    def _apply_two_stage_degradation(self, img: np.ndarray, cfg: Dict, scale: int) -> np.ndarray:
        """
        APISR-style two-stage degradation:
        Stage 1: Apply degradations on full image (before resize)
        Stage 2: Apply degradations after resize to LR size
        """
        h, w = img.shape[:2]

        # Stage 1: Full image degradation
        img = self._apply_shuffled_degradation(img, cfg)

        # Apply stage 1 compression if two-stage enabled
        if self.stage1_pipeline is not None and random.random() < cfg.get('compression_prob', 0.7):
            img = self.stage1_pipeline(img)

        # Resize to LR size
        lr_h, lr_w = h // scale, w // scale
        img = cv2.resize(img, (lr_w, lr_h), interpolation=cv2.INTER_LINEAR)

        # Stage 2: Post-resize degradation
        if random.random() < cfg.get('second_order_prob', 0.3):
            kernel_size = random.choice([3, 5, 7])
            sigma = random.uniform(0.1, 1.0)
            img = self._apply_blur(img, kernel_size, sigma)

        if random.random() < cfg.get('second_order_prob', 0.3):
            sigma = random.uniform(0, 10)
            img = self._apply_noise(img, sigma)

        # Apply stage 2 compression
        if self.stage2_pipeline is not None and random.random() < cfg.get('compression_prob', 0.5):
            img = self.stage2_pipeline(img)
        elif random.random() < cfg.get('jpeg_prob', 0.3):
            jpeg_qualities = cfg.get('jpeg_quality', [70, 80, 90, 100])
            quality = random.choice(jpeg_qualities)
            img = self._apply_jpeg(img, quality)

        return img

    def _apply_degradation_full_image(self, img: np.ndarray, cfg: Dict, scale: int) -> np.ndarray:
        """
        Apply degradation on the full image (not patch), then return LR.
        APISR approach: degrade full image first, then crop.
        """
        if self.two_stage_compression:
            return self._apply_two_stage_degradation(img, cfg, scale)
        elif self.shuffled_resize:
            img = self._apply_shuffled_degradation(img, cfg)
            h, w = img.shape[:2]
            lr_h, lr_w = h // scale, w // scale
            return cv2.resize(img, (lr_w, lr_h), interpolation=cv2.INTER_LINEAR)
        else:
            return self._apply_degradation(img)
    
    def _apply_blur(self, img: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
        """Apply Gaussian blur"""
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = kernel * kernel.T
        img = cv2.filter2D(img, -1, kernel)
        return img
    
    def _apply_noise(self, img: np.ndarray, sigma: float) -> np.ndarray:
        """Apply Gaussian noise"""
        if sigma == 0:
            return img
        noise = torch.randn(img.shape, generator=self._rng).numpy().astype(np.float32) * sigma
        img = img.astype(np.float32) + noise
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    
    def _apply_jpeg(self, img: np.ndarray, quality: int) -> np.ndarray:
        """Apply JPEG compression"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(encimg, 1)
        return img
    
    def _random_crop(self, hr: np.ndarray, lr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Random crop paired patches - supports progressive crop via training state"""
        hr_h, hr_w = hr.shape[:2]
        lr_h, lr_w = lr.shape[:2]

        # Check for dynamic crop_size from training state (progressive crop)
        crop_size = self.crop_size
        pc_enabled = False
        pc_epoch = -1
        try:
            from utils.training_state import get_training_state
            state = get_training_state()
            pc_enabled = state.progressive_crop.enabled
            pc_epoch = getattr(state, '_current_epoch', -1)
            if pc_enabled:
                new_crop = state.progressive_crop.get_crop_size(pc_epoch)
                self.crop_size = new_crop
                crop_size = new_crop
        except (ImportError, AttributeError):
            pass

        if pc_enabled and pc_epoch % 3 == 0 and pc_epoch >= 0:
            print(f"[Dataset PC] epoch={pc_epoch}, using crop_size={crop_size}, self.crop_size={self.crop_size}")

        # Ensure LR dimensions match expected scale
        expected_lr_h, expected_lr_w = hr_h // self.scale, hr_w // self.scale
        if lr_h != expected_lr_h or lr_w != expected_lr_w:
            # Resize LR to match expected dimensions
            lr = cv2.resize(lr, (expected_lr_w, expected_lr_h), interpolation=cv2.INTER_CUBIC)
            lr_h, lr_w = expected_lr_h, expected_lr_w

        # Check if crop size is valid
        if hr_h < crop_size or hr_w < crop_size:
            # Resize HR to crop_size if too small
            hr = cv2.resize(hr, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
            lr = cv2.resize(lr, (crop_size // self.scale, crop_size // self.scale),
                           interpolation=cv2.INTER_CUBIC)
            return hr, lr

        # Random crop position
        top = random.randint(0, hr_h - crop_size)
        left = random.randint(0, hr_w - crop_size)

        # Crop HR
        hr_crop = hr[top:top + crop_size, left:left + crop_size]

        # Crop LR at corresponding position
        lr_top = top // self.scale
        lr_left = left // self.scale
        lr_crop_size = crop_size // self.scale

        # Validate LR crop is within bounds
        if lr_top + lr_crop_size > lr_h or lr_left + lr_crop_size > lr_w:
            # Resize to ensure valid crop
            lr = cv2.resize(lr, (expected_lr_w, expected_lr_h), interpolation=cv2.INTER_CUBIC)

        lr_crop = lr[lr_top:lr_top + lr_crop_size, lr_left:lr_left + lr_crop_size]

        # Final validation
        if hr_crop.shape[0] != crop_size or hr_crop.shape[1] != crop_size:
            hr_crop = cv2.resize(hr_crop, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
        if lr_crop.shape[0] != lr_crop_size or lr_crop.shape[1] != lr_crop_size:
            lr_crop = cv2.resize(lr_crop, (lr_crop_size, lr_crop_size), interpolation=cv2.INTER_CUBIC)

        return hr_crop, lr_crop
    
    def _augment(self, hr: np.ndarray, lr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Random augmentation: flip and rotate"""
        # Horizontal flip
        if random.random() < 0.5:
            hr = np.fliplr(hr).copy()
            lr = np.fliplr(lr).copy()
        
        # Vertical flip
        if random.random() < 0.5:
            hr = np.flipud(hr).copy()
            lr = np.flipud(lr).copy()
        
        # Rotation (0, 90, 180, 270)
        rot = random.choice([0, 1, 2, 3])
        if rot > 0:
            hr = np.rot90(hr, rot).copy()
            lr = np.rot90(lr, rot).copy()
        
        return hr, lr
    
    def __len__(self) -> int:
        return len(self.hr_images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with 'lr' [C, H, W], 'hr' [C, H, W] tensors, normalized to [0, 1]
            If gpu_degradation=True, 'lr' will be None and must be generated on GPU.
        """
        hr_path = self.hr_images[idx]
        if self.preload and hr_path in self.cached_hr:
            hr = self.cached_hr[hr_path].copy()
        else:
            hr = self._load_image(hr_path)

        if self.gpu_degradation:
            hr = self._center_crop_if_needed(hr, self.crop_size)
            lr = None
        else:
            lr_path = self.lr_paths.get(hr_path)
            if lr_path is not None and lr_path.exists():
                lr = self._load_image(lr_path)
                hr, lr = self._random_crop(hr, lr)
            elif self.preprocessing_manager is not None:
                hr, lr = self._apply_preprocessing_manager(hr, idx, hr_path)
            else:
                hr, lr = self._apply_standard_degradation(hr)

        if self.augment:
            if lr is not None:
                hr, lr = self._augment(hr, lr)
            else:
                hr = self._augment_single(hr)

        hr = torch.from_numpy(hr.transpose(2, 0, 1)).float() / 255.0
        if lr is not None:
            lr = torch.from_numpy(lr.transpose(2, 0, 1)).float() / 255.0

        return {
            'lr': lr,
            'hr': hr,
            'name': hr_path.stem,
        }
    
    def _center_crop_if_needed(self, img: np.ndarray, crop_size: int) -> np.ndarray:
        """Center crop image if larger than crop_size."""
        h, w = img.shape[:2]
        if h > crop_size or w > crop_size:
            # Random crop position (for training variety)
            top = torch.randint(0, max(1, h - crop_size + 1), (1,), generator=self._rng).item()
            left = torch.randint(0, max(1, w - crop_size + 1), (1,), generator=self._rng).item()
            img = img[top:top + crop_size, left:left + crop_size]
        return img

    def _apply_hybrid_preprocessing(self, hr: np.ndarray, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Apply hybrid preprocessing using PreprocessingManager."""
        pm = self.preprocessing_manager
        hr_path = self.hr_images[idx]

        if pm and pm.mode == 'hybrid' and pm.precomputed_available:
            lr_path = self.lr_paths.get(hr_path)
            if lr_path is None:
                pm_base = Path(pm.precomputed_base_dir)
                candidate_lr = pm_base / 'lr' / hr_path.name
                if candidate_lr.exists():
                    lr_path = candidate_lr

            if lr_path and lr_path.exists():
                lr = self._load_image(lr_path)
                hr, lr = self._random_crop(hr, lr)
            else:
                lr_tensor = pm.process(hr, idx, hr_path=hr_path)
                lr = lr_tensor.cpu().numpy().transpose(1, 2, 0)
                hr, lr = self._random_crop(hr, lr)

            if self.line_enhancer is not None and self.apply_to_gt:
                hr = self.line_enhancer(hr)
            return hr, lr
        else:
            return self._apply_standard_degradation(hr)

    def _apply_preprocessing_manager(
        self, hr: np.ndarray, idx: int, hr_path
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Dispatch to the correct PreprocessingManager mode (all 5).

        - on_the_fly      : degrade HR via DegradationPipeline each batch
        - precomputed     : load precomputed LR/HR pair
        - gpu_degradation : degrade on GPU (returns None lr; trainer handles)
        - hybrid          : precomputed base + runtime augmentation
        - quality_adaptive: degrade HR via tier-selected DegradationPipeline
        """
        pm = self.preprocessing_manager
        pm_mode = getattr(pm, 'mode', None)
        pm_available = getattr(pm, 'precomputed_available', False)

        if pm_mode == 'precomputed':
            if not pm_available:
                raise RuntimeError(
                    f"Precomputed mode requires {pm.precomputed_base_dir} to exist. "
                    "Run precompute_pairs.py first or switch to 'hybrid' mode."
                )
            lr_tensor, hr_tensor = pm._process_precomputed(idx)
            lr = lr_tensor.cpu().numpy().transpose(1, 2, 0)
            hr_np = hr_tensor.cpu().numpy().transpose(1, 2, 0)
            hr, lr = self._random_crop(hr_np, lr)
            if self.line_enhancer is not None and self.apply_to_gt:
                hr = self.line_enhancer(hr)
            return hr, lr

        if pm_mode == 'on_the_fly':
            hr_tensor = torch.from_numpy(hr.transpose(2, 0, 1)).float() / 255.0
            hr_tensor = hr_tensor.unsqueeze(0).to(pm.device)
            lr_tensor = pm._process_on_the_fly(hr_tensor)
            lr = lr_tensor.cpu().numpy().transpose(1, 2, 0)
            if self.line_enhancer is not None and self.apply_to_gt:
                hr = self.line_enhancer(hr)
            hr, lr = self._random_crop(hr, lr)
            return hr, lr

        if pm_mode == 'quality_adaptive':
            hr_tensor = torch.from_numpy(hr.transpose(2, 0, 1)).float() / 255.0
            hr_tensor = hr_tensor.unsqueeze(0).to(pm.device)
            lr_tensor = pm._process_quality_adaptive(hr_tensor)
            lr = lr_tensor.cpu().numpy().transpose(1, 2, 0)
            if self.line_enhancer is not None and self.apply_to_gt:
                hr = self.line_enhancer(hr)
            hr, lr = self._random_crop(hr, lr)
            return hr, lr

        if pm_mode == 'gpu_degradation':
            if self.gpu_degradation:
                hr = self._center_crop_if_needed(hr, self.crop_size)
                return hr, None
            hr_tensor = torch.from_numpy(hr.transpose(2, 0, 1)).float() / 255.0
            hr_tensor = hr_tensor.unsqueeze(0).to(pm.device)
            lr_tensor = pm._process_gpu_degradation(hr_tensor)
            lr = lr_tensor.cpu().numpy().transpose(1, 2, 0)
            hr, lr = self._random_crop(hr, lr)
            return hr, lr

        # hybrid (default) or unknown mode: existing hybrid logic
        if pm_mode == 'hybrid' and pm_available:
            return self._apply_hybrid_preprocessing(hr, idx)
        return self._apply_standard_degradation(hr)

    def _apply_standard_degradation(self, hr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply standard degradation pipeline."""
        if self.degrade_before_crop and self.degradation.get('enabled', False):
            hr_original = hr.copy()
            if self.line_enhancer is not None and self.apply_to_gt:
                hr = self.line_enhancer(hr)
            lr = self._apply_degradation_full_image(hr_original, self.degradation, self.scale)
            hr, lr = self._random_crop(hr, lr)
        else:
            if self.line_enhancer is not None and self.apply_to_gt:
                hr = self.line_enhancer(hr)
            if self.use_quality_analyzer:
                lr = self._apply_degradation_with_quality(hr, None)
            else:
                lr = self._apply_degradation(hr)
            hr, lr = self._random_crop(hr, lr)
        return hr, lr

    def _augment_single(self, img: np.ndarray) -> np.ndarray:
        """Apply augmentation to single image."""
        # Horizontal flip
        if torch.rand(1, generator=self._rng).item() < 0.5:
            img = np.fliplr(img).copy()
        
        # Vertical flip
        if torch.rand(1, generator=self._rng).item() < 0.5:
            img = np.flipud(img).copy()
        
        # Rotation (0, 90, 180, 270)
        rot = torch.randint(0, 4, (1,), generator=self._rng).item()
        if rot > 0:
            img = np.rot90(img, rot).copy()
        
        return img


class DatasetFactory:
    """Factory for creating datasets from config"""
    
    @staticmethod
    def create(config: Dict, preprocessing_config: Optional[Dict] = None) -> BaseDataset:
        """Create dataset from config dict
        
        Args:
            config: Dataset configuration dict
            preprocessing_config: Optional preprocessing configuration for PreprocessingManager
        """
        degradation = config.get('degradation') or {}

        use_quality_analyzer = (
            degradation.get('mode') in ['auto', 'smart'] or
            config.get('use_quality_analyzer', False)
        )

        degrade_before_crop = degradation.get('degrade_before_crop', False)
        shuffled_resize = degradation.get('shuffled_resize', False)
        two_stage_compression = degradation.get('two_stage_compression', False)
        compression_stage1 = degradation.get('compression_stage1', ['jpeg', 'webp'])
        compression_stage2 = degradation.get('compression_stage2', ['avif', 'h264', 'jpeg'])
        line_enhancement = config.get('line_enhancement', None)
        
        # Create PreprocessingManager if preprocessing config provided (NEW)
        preprocessing_manager = None
        if preprocessing_config and preprocessing_config.get('mode'):
            try:
                from src.data.preprocessing_manager import PreprocessingManager
            except ImportError:
                try:
                    from data.preprocessing_manager import PreprocessingManager
                except ImportError:
                    preprocessing_manager = None
            else:
                preprocessing_manager = PreprocessingManager(preprocessing_config)

        return BaseDataset(
            hr_dir=config['hr_dir'],
            lr_dir=config.get('lr_dir'),
            scale=config.get('scale', 4),
            crop_size=config.get('crop_size', 128),
            augment=config.get('augment', True),
            degradation=degradation,
            preload=config.get('preload', False),
            max_images=config.get('max_images'),
            gpu_degradation=config.get('gpu_degradation', False),
            crop_size_mode=config.get('crop_size_mode', 'manual'),
            auto_crop_max=config.get('auto_crop_max', 960),
            auto_crop_min=config.get('auto_crop_min', 64),
            use_quality_analyzer=use_quality_analyzer,
            quality_analyzer=config.get('quality_analyzer'),
            degrade_before_crop=degrade_before_crop,
            shuffled_resize=shuffled_resize,
            two_stage_compression=two_stage_compression,
            compression_stage1=compression_stage1,
            compression_stage2=compression_stage2,
            line_enhancement=line_enhancement,
            preprocessing_manager=preprocessing_manager,
            sample_ratio=config.get('sample_ratio'),
        )
    
    @staticmethod
    def create_multi_dataset(configs: List[Dict], weights: Optional[List[float]] = None,
                              weight_mode: str = 'fixed',
                              preprocessing_config: Optional[Dict] = None):
        """
        Create weighted combination of multiple datasets.
        Returns a MultiDataset that samples from each dataset according to weights.
        
        Args:
            configs: List of dataset configurations
            weights: Optional list of weights (ignored if weight_mode='size_proportional')
            weight_mode: 'fixed' for user weights, 'size_proportional' for auto-calculated
            preprocessing_config: Optional preprocessing configuration for PreprocessingManager
        """
        datasets = [DatasetFactory.create(cfg, preprocessing_config=preprocessing_config) for cfg in configs]
        return MultiDataset(datasets, weights, weight_mode=weight_mode)


class MultiDataset(Dataset):
    """Weighted combination of multiple datasets
    
    Supports two weight calculation modes:
    - 'fixed': Use provided weights (default behavior)
    - 'size_proportional': Calculate weights based on dataset sizes
    
    Automatically standardizes crop_size across all datasets to ensure
    consistent tensor shapes for batching.
    """
    
    def __init__(self, datasets: List[Dataset], weights: Optional[List[float]] = None, 
                 weight_mode: str = 'fixed'):
        self.datasets = datasets
        self.weight_mode = weight_mode
        
        # Standardize crop_size across all datasets
        # This is necessary because PyTorch DataLoader requires consistent tensor shapes
        self._standardize_crop_sizes()
        
        if weight_mode == 'size_proportional':
            # Calculate weights proportional to dataset sizes
            dataset_sizes = []
            for ds in datasets:
                try:
                    dataset_sizes.append(len(ds))
                except (TypeError, AttributeError):
                    # Fallback to equal weights if length unknown
                    dataset_sizes.append(1.0)
            
            total_size = sum(dataset_sizes)
            if total_size > 0:
                self.weights = [size / total_size for size in dataset_sizes]
                print(f"[MultiDataset] Size-proportional weights: {self.weights}")
            else:
                self.weights = [1.0 / len(datasets)] * len(datasets)
        else:
            # Use provided or equal weights
            self.weights = weights or [1.0] * len(datasets)
        
        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
        
        # Compute cumulative weights for sampling
        self.cum_weights = [sum(self.weights[:i+1]) for i in range(len(self.weights))]
    
    def _standardize_crop_sizes(self):
        """
        Standardize crop_size across all datasets to ensure consistent tensor shapes.
        Uses the minimum crop_size from all datasets.
        """
        if not self.datasets:
            return
        
        # Collect crop_sizes from all datasets
        crop_sizes = []
        for ds in self.datasets:
            if hasattr(ds, 'crop_size'):
                crop_sizes.append(ds.crop_size)
        
        if not crop_sizes:
            return
        
        # Find minimum crop_size
        min_crop_size = min(crop_sizes)
        
        # Check if standardization is needed
        unique_sizes = set(crop_sizes)
        if len(unique_sizes) > 1:
            print(f"[MultiDataset] Standardizing crop_sizes: {crop_sizes} -> {min_crop_size}")
            
            # Adjust all datasets to use the minimum crop_size
            for ds in self.datasets:
                if hasattr(ds, 'crop_size') and ds.crop_size != min_crop_size:
                    ds.crop_size = min_crop_size
    
    def __len__(self) -> int:
        # Approximate length
        return max(len(d) for d in self.datasets) * len(self.datasets)
    
    def __getitem__(self, idx: int):
        # Select dataset based on weights
        r = random.random()
        dataset_idx = 0
        for i, cw in enumerate(self.cum_weights):
            if r <= cw:
                dataset_idx = i
                break
        
        # Get random sample from selected dataset
        dataset = self.datasets[dataset_idx]
        sample_idx = random.randint(0, len(dataset) - 1)
        return dataset[sample_idx]
