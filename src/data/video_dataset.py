"""
Video Dataset for on-demand frame extraction during training
Optimizes storage by extracting frames as needed rather than pre-extracting all
Includes auto crop_size detection and smart quality-aware degradation
"""
import os
import cv2
import numpy as np
import torch
import random
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable
from torch.utils.data import Dataset
import tempfile
import shutil
from collections import OrderedDict
import warnings

try:
    from data.video_extraction import (
        calculate_frame_quality,
        is_duplicate_frame,
        calculate_frame_hash
    )
    from data.quality_analyzer import (
        QualityAnalyzer, get_degradation_preset_config
    )
except ImportError:
    from video_extraction import (
        calculate_frame_quality,
        is_duplicate_frame,
        calculate_frame_hash
    )
    from quality_analyzer import (
        QualityAnalyzer, get_degradation_preset_config
    )


class VideoDataset(Dataset):
    """
    PyTorch Dataset that extracts frames on-demand from video files.
    
    Features:
    - Extracts frames during training as needed
    - Configurable caching to balance speed vs storage
    - Quality filtering on-the-fly
    - Duplicate detection
    
    Usage:
        dataset = VideoDataset(
            video_dir="data/anime_vid",
            extract_every_n_frames=30,
            quality_threshold=0.7,
            cache_size_gb=10
        )
    """
    
    def __init__(
        self,
        video_dir: str,
        scale: int = 4,
        crop_size: int = 128,
        augment: bool = True,
        degradation: Optional[Dict] = None,
        extract_every_n_frames: int = 30,
        quality_threshold: float = 0.7,
        remove_duplicates: bool = True,
        min_resolution: Tuple[int, int] = (720, 720),
        cache_size_gb: float = 10.0,
        max_images: Optional[int] = None,
        video_extensions: Tuple[str, ...] = ('.mp4', '.avi', '.mkv', '.mov', '.webm'),
        # Auto crop_size parameters (NEW)
        crop_size_mode: str = 'manual',  # 'auto' or 'manual'
        auto_crop_max: int = 960,
        auto_crop_min: int = 64,
        # Smart degradation parameters (NEW)
        use_quality_analyzer: bool = False,
        quality_analyzer: Optional[QualityAnalyzer] = None,
        # Preprocessing pipeline parameters (NEW - Issue #12 parity with BaseDataset)
        preprocessing_manager: Optional[object] = None,
        sample_ratio: float = 1.0,
        line_enhancer: Optional[object] = None,
        two_stage_compression: bool = False,
        degrade_before_crop: bool = True,
        shuffled_resize: bool = True,
    ):
        """
        Args:
            video_dir: Directory containing video files
            scale: Upsampling factor
            crop_size: Training patch size (HR space)
            augment: Enable random flip/rotate
            degradation: Degradation config for synthetic LR
            extract_every_n_frames: Extract 1 frame every N frames
            quality_threshold: Minimum quality score (0-1)
            remove_duplicates: Remove similar consecutive frames
            min_resolution: Minimum (width, height) to keep
            cache_size_gb: Maximum cache size in GB
            max_images: Limit total frames across all videos
            video_extensions: Supported video formats
            crop_size_mode: 'auto' to detect from video resolution, 'manual' to use crop_size
            auto_crop_max: Maximum crop_size in auto mode
            auto_crop_min: Minimum crop_size
            use_quality_analyzer: Enable per-frame quality-based degradation
            quality_analyzer: Optional pre-configured QualityAnalyzer
        """
        self.video_dir = Path(video_dir)
        self.scale = scale
        self.crop_size = crop_size
        self.augment = augment
        self.degradation = degradation or {}
        self.extract_every_n_frames = extract_every_n_frames
        self.quality_threshold = quality_threshold
        self.remove_duplicates = remove_duplicates
        self.min_resolution = min_resolution
        self.cache_size_gb = cache_size_gb
        self.max_images = max_images
        self.video_extensions = video_extensions

        # Preprocessing pipeline wiring (parity with BaseDataset; Issue #12)
        self.preprocessing_manager = preprocessing_manager
        self.sample_ratio = sample_ratio
        self.line_enhancer = line_enhancer
        self.two_stage_compression = two_stage_compression
        self.degrade_before_crop = degrade_before_crop
        self.shuffled_resize = shuffled_resize
        if sample_ratio < 1.0 and self.frame_index:
            keep = max(1, int(len(self.frame_index) * sample_ratio))
            self.frame_index = self.frame_index[:keep]

        # Validate directory
        if not self.video_dir.exists():
            raise FileNotFoundError(
                f"Video directory does not exist: {self.video_dir}\n"
                f"Please create the directory and add video files."
            )
        
        # Find video files
        self.video_files = self._find_videos()
        
        if not self.video_files:
            raise ValueError(
                f"No video files found in {self.video_dir}\n"
                f"Supported formats: {', '.join(video_extensions)}"
            )
        
        # Build frame index: maps dataset index -> (video_idx, frame_pos)
        self.frame_index = self._build_frame_index()
        
        if max_images:
            self.frame_index = self.frame_index[:max_images]
        
        # Setup cache
        self.cache_dir = Path(tempfile.gettempdir()) / "anime_sr_video_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.frame_cache = OrderedDict()  # LRU cache: frame_idx -> frame_path
        self.cache_size_bytes = int(cache_size_gb * 1024 * 1024 * 1024)
        self.current_cache_size = 0
        
        # Open video capture objects (lazy initialization)
        self.video_caps = {}
        
        print(f"VideoDataset loaded: {len(self.video_files)} videos, {len(self.frame_index)} frames")
        for i, v in enumerate(self.video_files):
            print(f"  [{i}] {v.name}")
        
        # Auto-detect optimal crop size from video resolution if enabled
        if crop_size_mode == 'auto':
            self._compute_optimal_crop_size(auto_crop_max, auto_crop_min)
        elif crop_size_mode == 'manual' and self.video_files:
            # Validate manual crop_size against video resolution
            self._validate_crop_size()
        
        # Initialize quality analyzer if enabled
        self.use_quality_analyzer = use_quality_analyzer
        self.quality_analyzer = quality_analyzer
        if use_quality_analyzer:
            if self.quality_analyzer is None:
                self.quality_analyzer = QualityAnalyzer()
            print(f"[VideoDataset] Quality analyzer enabled - per-frame quality-based degradation")
            # Setup auto degradation based on first frame analysis
            if self.degradation.get('mode') == 'auto':
                self._setup_auto_degradation()
        
        # Initialize RNG for reproducibility
        self._rng = torch.Generator()
        self._rng.manual_seed(torch.initial_seed() % (2**32))
    
    def _compute_optimal_crop_size(self, max_size: int = 960, min_size: int = 64):
        """Compute optimal crop size from video resolution."""
        try:
            # Sample a few videos to get resolution
            sample_videos = self.video_files[:3]
            min_width, min_height = float('inf'), float('inf')
            
            for video_path in sample_videos:
                cap = cv2.VideoCapture(str(video_path))
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    min_width = min(min_width, width)
                    min_height = min(min_height, height)
                    cap.release()
            
            if min_width == float('inf') or min_height == float('inf'):
                return  # Could not detect resolution
            
            # Use smaller dimension to determine crop_size
            smallest_dim = min(min_width, min_height)
            
            # Standard sizes for efficiency
            standard_sizes = [64, 96, 128, 160, 192, 256, 320, 384, 448, 512, 640, 720, 768, 960]
            
            # Find largest standard size that fits within smallest dimension
            optimal_size = min_size
            for size in standard_sizes:
                if size <= smallest_dim * 0.9:  # Leave 10% margin
                    optimal_size = size
            
            # Apply caps
            optimal_size = max(min_size, min(optimal_size, max_size))
            
            if optimal_size != self.crop_size:
                print(f"[VideoDataset Auto Crop] Adjusted crop_size from {self.crop_size} to {optimal_size}")
                self.crop_size = optimal_size
                
        except Exception as e:
            print(f"[VideoDataset Auto Crop] Error during auto-detection: {e}")
    
    def _validate_crop_size(self):
        """Validate that crop_size is appropriate for video resolution."""
        try:
            if not self.video_files:
                return
            
            # Sample first video
            cap = cv2.VideoCapture(str(self.video_files[0]))
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                
                smallest_dim = min(width, height)
                
                if self.crop_size > smallest_dim:
                    warnings.warn(
                        f"[VideoDataset] crop_size ({self.crop_size}) is larger than video resolution "
                        f"({width}x{height}). This will cause upscaling during training. "
                        f"Consider using crop_size_mode='auto' or setting crop_size to {smallest_dim // 2}."
                    )
        except Exception:
            # If validation fails, continue silently
            pass
    
    def _setup_auto_degradation(self):
        """Setup auto degradation based on sample frame quality analysis."""
        try:
            print("[VideoDataset Auto Degradation] Analyzing sample frames...")
            
            # Analyze a few sample frames
            sample_scores = []
            sample_indices = [0, min(len(self.frame_index) // 2, 50), min(len(self.frame_index) - 1, 100)]
            
            for idx in sample_indices:
                if idx < len(self.frame_index):
                    frame = self._get_cached_frame(idx)
                    if frame is not None:
                        score = self.quality_analyzer.analyze_image(frame)
                        sample_scores.append(score)
            
            if not sample_scores:
                print("[VideoDataset Auto Degradation] Could not analyze frames")
                return
            
            mean_quality = sum(sample_scores) / len(sample_scores)
            recommended = self.quality_analyzer.get_degradation_tier(mean_quality)
            
            print(f"[VideoDataset Auto Degradation] Mean quality: {mean_quality:.2f}")
            print(f"[VideoDataset Auto Degradation] Recommended preset: '{recommended}'")
            
            # Apply preset
            preset_config = get_degradation_preset_config(recommended, self.degradation)
            self.degradation.update(preset_config)
            
        except Exception as e:
            print(f"[VideoDataset Auto Degradation] Error during setup: {e}")
    
    def _find_videos(self) -> List[Path]:
        """Find all video files in directory"""
        videos = []
        for ext in self.video_extensions:
            videos.extend(self.video_dir.glob(f"*{ext}"))
            videos.extend(self.video_dir.glob(f"*{ext.upper()}"))
        return sorted(list(set(videos)))
    
    def _build_frame_index(self) -> List[Tuple[int, int]]:
        """
        Build index mapping dataset index to (video_idx, frame_position).
        Scans videos and identifies valid frame positions.
        """
        index = []
        
        for video_idx, video_path in enumerate(self.video_files):
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                continue
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Add every Nth frame position
            for frame_pos in range(0, total_frames, self.extract_every_n_frames):
                index.append((video_idx, frame_pos))
        
        return index
    
    def _get_video_cap(self, video_idx: int) -> cv2.VideoCapture:
        """Get or create video capture for video_idx"""
        if video_idx not in self.video_caps or not self.video_caps[video_idx].isOpened():
            video_path = self.video_files[video_idx]
            self.video_caps[video_idx] = cv2.VideoCapture(str(video_path))
        return self.video_caps[video_idx]
    
    def _extract_frame(self, video_idx: int, frame_pos: int) -> Optional[np.ndarray]:
        """Extract specific frame from video"""
        cap = self._get_video_cap(video_idx)
        
        if not cap.isOpened():
            return None
        
        # Seek to frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        
        # Read frame
        ret, frame = cap.read()
        
        if not ret or frame is None:
            return None
        
        # Check resolution
        h, w = frame.shape[:2]
        if w < self.min_resolution[0] or h < self.min_resolution[1]:
            return None
        
        # Check quality
        quality = calculate_frame_quality(frame)
        if quality < self.quality_threshold:
            return None
        
        return frame
    
    def _get_cached_frame(self, idx: int) -> Optional[np.ndarray]:
        """Get frame from cache or extract and cache"""
        cache_key = f"frame_{idx}"
        
        # Check memory cache first
        if cache_key in self.frame_cache:
            # Move to end (LRU)
            self.frame_cache.move_to_end(cache_key)
            frame_path = self.frame_cache[cache_key]
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract frame
        video_idx, frame_pos = self.frame_index[idx]
        frame = self._extract_frame(video_idx, frame_pos)
        
        if frame is None:
            return None
        
        # Cache to disk
        self._cache_frame(idx, frame)
        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def _cache_frame(self, idx: int, frame: np.ndarray):
        """Cache frame to disk with LRU eviction"""
        cache_key = f"frame_{idx}"
        cache_path = self.cache_dir / f"{cache_key}.png"
        
        # Save frame
        cv2.imwrite(str(cache_path), frame)
        
        # Get file size
        file_size = cache_path.stat().st_size
        
        # Evict old entries if needed
        while (self.current_cache_size + file_size > self.cache_size_bytes and 
               len(self.frame_cache) > 0):
            # Remove oldest
            old_key, old_path = self.frame_cache.popitem(last=False)
            if old_path.exists():
                self.current_cache_size -= old_path.stat().st_size
                old_path.unlink()
        
        # Add to cache
        self.frame_cache[cache_key] = cache_path
        self.current_cache_size += file_size
    
    def _apply_degradation(self, img: np.ndarray, frame_idx: Optional[int] = None) -> np.ndarray:
        """Apply RealESRGAN-style degradation to create LR image"""
        if not self.degradation.get('enabled', False):
            # Simple bicubic downsample
            h, w = img.shape[:2]
            lr_h, lr_w = h // self.scale, w // self.scale
            lr = cv2.resize(img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
            return lr
        
        # Get quality-aware degradation config if enabled
        if self.use_quality_analyzer and self.quality_analyzer is not None:
            # Analyze image quality
            quality_score = self.quality_analyzer.analyze_image(img)
            tier = self.quality_analyzer.get_degradation_tier(quality_score)
            cfg = get_degradation_preset_config(tier, self.degradation)
        else:
            cfg = self.degradation
        
        # Apply first-order degradation with selected config
        img = self._first_degradation_with_config(img, cfg)
        
        # Resize to LR size
        h, w = img.shape[:2]
        lr_h, lr_w = h // self.scale, w // self.scale
        img = cv2.resize(img, (lr_w, lr_h), interpolation=cv2.INTER_LINEAR)
        
        # Apply second-order degradation with selected config
        img = self._second_degradation_with_config(img, cfg)
        
        return img
    
    def _first_degradation_with_config(self, img: np.ndarray, cfg: Dict) -> np.ndarray:
        """Apply first-order degradation with specific config."""
        # Blur
        if random.random() < cfg.get('blur_prob', 0.7):
            kernel_sizes = cfg.get('blur_kernel_size', [7, 9, 11])
            kernel_size = random.choice(kernel_sizes)
            sigma_range = cfg.get('blur_sigma', [0.1, 3.0])
            sigma = random.uniform(*sigma_range[:2])
            kernel = cv2.getGaussianKernel(kernel_size, sigma)
            kernel = kernel * kernel.T
            img = cv2.filter2D(img, -1, kernel)
        
        # Noise
        if random.random() < cfg.get('noise_prob', 0.5):
            sigma_range = cfg.get('noise_sigma', [0, 25])
            sigma = random.uniform(*sigma_range[:2])
            noise = torch.randn(img.shape, generator=self._rng).numpy().astype(np.float32) * sigma
            img = img.astype(np.float32) + noise
            img = np.clip(img, 0, 255).astype(np.uint8)
        
        # JPEG compression
        if random.random() < cfg.get('jpeg_prob', 0.5):
            jpeg_qualities = cfg.get('jpeg_quality', [60, 70, 80, 90, 100])
            quality = random.choice(jpeg_qualities)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encimg = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(encimg, 1)
        
        return img
    
    def _second_degradation_with_config(self, img: np.ndarray, cfg: Dict) -> np.ndarray:
        """Apply second-order degradation with specific config."""
        second_order_prob = cfg.get('second_order_prob', 0.3)
        
        if random.random() < second_order_prob:
            kernel_size = random.choice([3, 5, 7])
            sigma = random.uniform(0.1, 1.0)
            kernel = cv2.getGaussianKernel(kernel_size, sigma)
            kernel = kernel * kernel.T
            img = cv2.filter2D(img, -1, kernel)
        
        if random.random() < second_order_prob:
            sigma = random.uniform(0, 10)
            noise = torch.randn(img.shape, generator=self._rng).numpy().astype(np.float32) * sigma
            img = img.astype(np.float32) + noise
            img = np.clip(img, 0, 255).astype(np.uint8)
        
        return img
    
    def _first_degradation(self, img: np.ndarray) -> np.ndarray:
        """First-order degradation: blur, noise, resize, jpeg"""
        cfg = self.degradation
        
        # Blur
        if random.random() < cfg.get('blur_prob', 0.7):
            kernel_size = random.choice(cfg.get('blur_kernel_size', [7, 9, 11]))
            sigma = random.uniform(*cfg.get('blur_sigma', [0.1, 3.0])[:2])
            kernel = cv2.getGaussianKernel(kernel_size, sigma)
            kernel = kernel * kernel.T
            img = cv2.filter2D(img, -1, kernel)
        
        # Noise
        if random.random() < cfg.get('noise_prob', 0.5):
            sigma = random.uniform(*cfg.get('noise_sigma', [0, 25])[:2])
            noise = torch.randn(img.shape, generator=self._rng).numpy().astype(np.float32) * sigma
            img = img.astype(np.float32) + noise
            img = np.clip(img, 0, 255).astype(np.uint8)
        
        # JPEG compression
        if random.random() < cfg.get('jpeg_prob', 0.5):
            quality = random.choice(cfg.get('jpeg_quality', [60, 70, 80, 90, 100]))
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encimg = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(encimg, 1)
        
        return img
    
    def _second_degradation(self, img: np.ndarray) -> np.ndarray:
        """Second-order degradation (lighter)"""
        
        if random.random() < 0.3:
            kernel_size = random.choice([3, 5, 7])
            sigma = random.uniform(0.1, 1.0)
            kernel = cv2.getGaussianKernel(kernel_size, sigma)
            kernel = kernel * kernel.T
            img = cv2.filter2D(img, -1, kernel)
        
        if random.random() < 0.3:
            sigma = random.uniform(0, 10)
            noise = torch.randn(img.shape, generator=self._rng).numpy().astype(np.float32) * sigma
            img = img.astype(np.float32) + noise
            img = np.clip(img, 0, 255).astype(np.uint8)
        
        return img
    
    def _random_crop(self, hr: np.ndarray, lr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Random crop paired patches"""
        hr_h, hr_w = hr.shape[:2]
        
        # Check if crop size is valid
        if hr_h < self.crop_size or hr_w < self.crop_size:
            # Resize HR to crop_size if too small
            hr = cv2.resize(hr, (self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC)
            lr = cv2.resize(lr, (self.crop_size // self.scale, self.crop_size // self.scale),
                           interpolation=cv2.INTER_CUBIC)
            return hr, lr
        
        # Random crop position
        import random
        top = random.randint(0, hr_h - self.crop_size)
        left = random.randint(0, hr_w - self.crop_size)
        
        # Crop HR
        hr_crop = hr[top:top + self.crop_size, left:left + self.crop_size]
        
        # Crop LR at corresponding position
        lr_top = top // self.scale
        lr_left = left // self.scale
        lr_crop_size = self.crop_size // self.scale
        lr_crop = lr[lr_top:lr_top + lr_crop_size, lr_left:lr_left + lr_crop_size]
        
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
        return len(self.frame_index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with 'lr' [C, H, W], 'hr' [C, H, W] tensors, normalized to [0, 1]
        """
        # Get HR frame (extract on-demand)
        hr = self._get_cached_frame(idx)

        if hr is None:
            raise RuntimeError(
                f"VideoDataset: failed to extract frame at idx {idx} "
                f"(video_idx, frame_pos)={self.frame_index[idx]}"
            )

        # Degrade via the shared PreprocessingManager if attached (Issue #12).
        if self.preprocessing_manager is not None:
            pm = self.preprocessing_manager
            hr_tensor = torch.from_numpy(hr.transpose(2, 0, 1)).float() / 255.0
            hr_tensor = hr_tensor.unsqueeze(0).to(pm.device)
            pm_mode = getattr(pm, 'mode', None)
            if pm_mode == 'on_the_fly':
                lr_tensor = pm._process_on_the_fly(hr_tensor)
            elif pm_mode == 'quality_adaptive':
                lr_tensor = pm._process_quality_adaptive(hr_tensor)
            elif pm_mode == 'gpu_degradation':
                lr_tensor = pm._process_gpu_degradation(hr_tensor)
            else:
                lr_tensor = None
            if lr_tensor is not None:
                lr = lr_tensor.cpu().numpy().transpose(1, 2, 0)
            else:
                lr = self._apply_degradation(hr)
        else:
            lr = self._apply_degradation(hr)

        # Line enhancement (parity with BaseDataset; Issue #12)
        if self.line_enhancer is not None:
            try:
                hr = self.line_enhancer(hr)
            except Exception:
                pass

        # Random crop
        hr, lr = self._random_crop(hr, lr)

        # Augmentation
        if self.augment:
            hr, lr = self._augment(hr, lr)

        # Convert to tensor [C, H, W], float32, [0, 1]
        hr = torch.from_numpy(hr.transpose(2, 0, 1)).float() / 255.0
        lr = torch.from_numpy(lr.transpose(2, 0, 1)).float() / 255.0

        video_idx, frame_pos = self.frame_index[idx]
        video_name = self.video_files[video_idx].stem

        return {
            'lr': lr,
            'hr': hr,
            'name': f"{video_name}_frame{frame_pos}",
        }
    
    def __del__(self):
        """Cleanup video captures and cache on deletion"""
        # Close video captures
        for cap in self.video_caps.values():
            if cap.isOpened():
                cap.release()
        
        # Clear cache directory
        if hasattr(self, 'cache_dir') and self.cache_dir.exists():
            shutil.rmtree(self.cache_dir, ignore_errors=True)
    
    def get_info(self) -> Dict:
        """Get dataset information"""
        total_duration = 0
        for video_path in self.video_files:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                total_duration += frames / fps if fps > 0 else 0
                cap.release()
        
        return {
            'num_videos': len(self.video_files),
            'num_frames': len(self.frame_index),
            'total_duration_seconds': total_duration,
            'cache_size_gb': self.cache_size_gb,
            'extract_every_n_frames': self.extract_every_n_frames,
            'quality_threshold': self.quality_threshold,
        }


class TemporalDataset(Dataset):
    """
    Dataset for temporal consistency training.
    Returns sequences of consecutive frames for video super-resolution.
    
    This enables training with temporal loss (frame-to-frame consistency).
    
    Usage:
        dataset = TemporalDataset(
            base_dataset=video_dataset,
            frame_window=3,
            scene_change_threshold=0.3
        )
        # Returns: [T, C, H, W] tensor where T is frame_window
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        frame_window: int = 3,
        scene_change_threshold: float = 0.3,
    ):
        """
        Args:
            base_dataset: Underlying dataset (e.g., VideoDataset or image folder)
            frame_window: Number of consecutive frames to return
            scene_change_threshold: Skip sequences with large scene changes
        """
        self.base_dataset = base_dataset
        self.frame_window = frame_window
        self.scene_change_threshold = scene_change_threshold
        
        # For image datasets, we need sequential indexing
        # For video datasets, use their frame index
        if hasattr(base_dataset, 'frame_index'):
            # VideoDataset has frame_index with (video_idx, frame_num)
            self.is_video = True
            self.indices = self._build_video_sequences()
        else:
            # Regular image dataset - assume sequential ordering
            self.is_video = False
            self.indices = self._build_image_sequences()
    
    def _build_video_sequences(self) -> List[Tuple[int, ...]]:
        """Build valid frame sequences from video frame index."""
        sequences = []
        frame_index = self.base_dataset.frame_index
        
        # Group by video
        video_frames = {}
        for idx, (video_idx, frame_num) in enumerate(frame_index):
            if video_idx not in video_frames:
                video_frames[video_idx] = []
            video_frames[video_idx].append((idx, frame_num))
        
        # Build sequences within each video
        for video_idx, frames in video_frames.items():
            # Sort by frame number
            frames.sort(key=lambda x: x[1])
            
            # Create sequences of frame_window length
            for i in range(len(frames) - self.frame_window + 1):
                sequence = tuple(frames[j][0] for j in range(i, i + self.frame_window))
                sequences.append(sequence)
        
        return sequences
    
    def _build_image_sequences(self) -> List[Tuple[int, ...]]:
        """Build sequences for image dataset (consecutive indices)."""
        num_items = len(self.base_dataset)
        sequences = []
        
        for i in range(num_items - self.frame_window + 1):
            sequence = tuple(range(i, i + self.frame_window))
            sequences.append(sequence)
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sequence of frames.
        
        Returns:
            Dict with:
                - 'hr': [T, C, H, W] High-res frames
                - 'lr': [T, C, H//scale, W//scale] Low-res frames (if degradation enabled)
                - 'is_temporal': True (flag for temporal training)
        """
        sequence_indices = self.indices[idx]
        
        # Load all frames in sequence
        frames = []
        for frame_idx in sequence_indices:
            frame_data = self.base_dataset[frame_idx]
            frames.append(frame_data)
        
        # Stack frames
        if 'lr' in frames[0]:
            # Has LR degradation
            lr_frames = torch.stack([f['lr'] for f in frames], dim=0)  # [T, C, H, W]
            hr_frames = torch.stack([f['hr'] for f in frames], dim=0)  # [T, C, H, W]
            return {
                'lr': lr_frames,
                'hr': hr_frames,
                'is_temporal': True,
            }
        else:
            # HR only (degradation applied during training)
            hr_frames = torch.stack([f['hr'] for f in frames], dim=0)  # [T, C, H, W]
            return {
                'lr': None,
                'hr': hr_frames,
                'is_temporal': True,
            }
    
    def get_info(self) -> Dict:
        """Get dataset information."""
        base_info = self.base_dataset.get_info() if hasattr(self.base_dataset, 'get_info') else {}
        return {
            **base_info,
            'type': 'TemporalDataset',
            'frame_window': self.frame_window,
            'scene_change_threshold': self.scene_change_threshold,
            'num_sequences': len(self.indices),
        }
