"""
Quality analysis module for adaptive degradation and crop size detection.
Analyzes images to determine optimal training parameters.
"""
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from PIL import Image


def calculate_image_quality(img: np.ndarray) -> float:
    """
    Calculate image quality score (0-1) using multiple metrics.
    
    Metrics:
    - Sharpness (Laplacian variance)
    - Contrast (standard deviation)
    - Color variance (indicates rich vs flat colors)
    - Compression artifacts detection
    
    Args:
        img: Image array [H, W, C] or [H, W], RGB/BGR/Grayscale
        
    Returns:
        Quality score from 0.0 (low quality) to 1.0 (high quality)
    """
    if img is None or img.size == 0:
        return 0.0
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        if img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.dtype == np.uint8 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img[:, :, 0]
    else:
        gray = img
    
    # Metric 1: Sharpness (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize: good sharpness > 100, excellent > 500
    sharpness_score = min(laplacian_var / 500, 1.0)
    
    # Metric 2: Contrast
    contrast = np.std(gray) / 128.0
    contrast_score = min(contrast, 1.0)
    
    # Metric 3: Color richness (if RGB)
    color_score = 0.5  # Default if grayscale
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Calculate color variance across channels
        color_std = np.std(img, axis=2).mean() / 128.0
        color_score = min(color_std, 1.0)
    
    # Metric 4: Artifact detection (DCT-based)
    artifact_score = detect_compression_artifacts(gray)
    
    # Weighted combination
    # High sharpness + contrast + color = HD/clean source
    # Low artifact score indicates compression
    score = (
        sharpness_score * 0.35 +
        contrast_score * 0.25 +
        color_score * 0.20 +
        (1.0 - artifact_score) * 0.20  # Lower artifacts = higher quality
    )
    
    return float(np.clip(score, 0.0, 1.0))


def calculate_anime_quality(img: np.ndarray) -> float:
    """
    Calculate image quality score optimized for anime content.
    
    Anime-specific metrics:
    - Edge density (anime has strong, clean lines)
    - Color flatness (anime uses flat shading)
    - Line art sharpness (anime relies on clear outlines)
    - Color quantization detection (indicates compression banding)
    
    Args:
        img: Image array [H, W, C], RGB format
        
    Returns:
        Quality score from 0.0 (low quality) to 1.0 (high quality)
    """
    if img is None or img.size == 0:
        return 0.0
    
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img if len(img.shape) == 2 else img[:, :, 0]
    
    # Metric 1: Edge density (anime has distinct outlines)
    # Use Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    # Anime typically has 10-30% edge density; too low = blurry, too high = noisy
    edge_score = min(edge_density * 5, 1.0) if edge_density > 0.02 else edge_density * 10
    
    # Metric 2: Line sharpness (Sobel gradient magnitude)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    line_sharpness = np.percentile(sobel_mag, 90) / 255.0
    line_score = min(line_sharpness * 2, 1.0)
    
    # Metric 3: Color flatness (anime uses flat shading with clean color regions)
    # Low local variance indicates intentional flat shading (good anime)
    # High local variance indicates photo-realistic or noise
    if len(img.shape) == 3:
        local_std = cv2.blur(img.astype(np.float32), (15, 15))
        local_std = cv2.blur((img.astype(np.float32) - local_std)**2, (15, 15))
        if len(local_std.shape) == 3:
            local_std = local_std.mean(axis=2)
        flatness = 1.0 - min(np.mean(local_std) / 1000.0, 1.0)
    else:
        local_std = cv2.blur(gray.astype(np.float32), (15, 15))
        local_std = cv2.blur((gray.astype(np.float32) - local_std)**2, (15, 15))
        flatness = 1.0 - min(np.mean(local_std) / 1000.0, 1.0)
    
    # Metric 4: Compression artifacts (blocking and banding)
    # Detect blocking artifacts (8x8 DCT blocks from JPEG)
    block_score = detect_blocking_artifacts(gray)
    
    # Metric 5: Color quantization (banding due to heavy compression)
    quant_score = detect_color_quantization(img)
    
    # Weighted combination for anime
    score = (
        edge_score * 0.30 +
        line_score * 0.30 +
        flatness * 0.15 +
        (1.0 - block_score) * 0.15 +
        (1.0 - quant_score) * 0.10
    )
    
    return float(np.clip(score, 0.0, 1.0))


def detect_blocking_artifacts(gray_img: np.ndarray) -> float:
    """
    Detect JPEG blocking artifacts (8x8 grid pattern).
    Returns artifact score (0 = clean, 1 = heavy blocking).
    """
    h, w = gray_img.shape
    block_size = 8
    
    if h < 16 or w < 16:
        return 0.0
    
    # Analyze variance at block boundaries
    artifact_scores = []
    
    for i in range(8, h - 8, 16):
        for j in range(8, w - 8, 16):
            # Compare pixels across block boundaries
            if j + 1 < w:
                diff_h = np.abs(int(gray_img[i, j]) - int(gray_img[i, j + 1]))
                artifact_scores.append(min(diff_h / 20.0, 1.0))
            if i + 1 < h:
                diff_v = np.abs(int(gray_img[i, j]) - int(gray_img[i + 1, j]))
                artifact_scores.append(min(diff_v / 20.0, 1.0))
    
    return float(np.mean(artifact_scores)) if artifact_scores else 0.0


def detect_color_quantization(img: np.ndarray) -> float:
    """
    Detect color banding/quantization artifacts (indicates heavy compression).
    Returns quantization score (0 = smooth gradients, 1 = heavy banding).
    """
    if len(img.shape) != 3 or img.shape[2] != 3:
        return 0.0
    
    h, w, c = img.shape
    
    if h < 40 or w < 40:
        return 0.0
    
    # Sample smooth regions and check for stepped colors
    scores = []
    
    for _ in range(50):
        y = np.random.randint(20, h - 20)
        x = np.random.randint(20, w - 20)
        
        # Extract 32x32 patch
        patch = img[y:y+32, x:x+32].astype(np.float32)
        
        # Check color uniformity
        color_std = np.std(patch.reshape(-1, 3), axis=0).mean()
        
        # Low variance + high local variation = quantization
        if color_std < 5:
            # Check for stepping (large jumps in gradients)
            grad_y = np.abs(np.diff(patch, axis=0)).mean()
            grad_x = np.abs(np.diff(patch, axis=1)).mean()
            
            if grad_y > 3 or grad_x > 3:
                scores.append(0.8)  # Likely quantized
            else:
                scores.append(0.2)  # Smooth region, likely fine
        else:
            scores.append(0.3)  # High variance, likely natural content
    
    return float(np.mean(scores)) if scores else 0.0


def detect_compression_artifacts(gray_img: np.ndarray) -> float:
    """
    Detect JPEG/compression artifacts using DCT analysis.
    Returns artifact score (0 = clean, 1 = heavy compression).
    """
    # Resize to 8x8 blocks for DCT analysis
    h, w = gray_img.shape
    block_size = 8
    
    # Sample a few blocks
    num_blocks = min(100, (h // block_size) * (w // block_size))
    if num_blocks == 0:
        return 0.0
    
    artifact_scores = []
    
    for _ in range(num_blocks):
        # Random block position
        y = np.random.randint(0, max(1, h - block_size))
        x = np.random.randint(0, max(1, w - block_size))
        
        block = gray_img[y:y+block_size, x:x+block_size].astype(np.float32)
        if block.shape[0] < block_size or block.shape[1] < block_size:
            continue
        
        # Apply DCT
        dct = cv2.dct(block / 255.0)
        
        # Analyze high-frequency coefficients (compression removes these)
        high_freq = np.abs(dct[4:, 4:]).mean()
        low_freq = np.abs(dct[:4, :4]).mean() + 1e-8
        
        # Ratio indicates compression level
        ratio = high_freq / low_freq
        artifact_scores.append(1.0 - min(ratio * 10, 1.0))
    
    return float(np.mean(artifact_scores)) if artifact_scores else 0.0


def analyze_dataset_quality(image_paths: List[Path], sample_size: int = 50, anime_mode: bool = False) -> Dict:
    """
    Analyze quality distribution of a dataset.
    
    Args:
        image_paths: List of image file paths
        sample_size: Number of images to sample for analysis
        anime_mode: If True, use anime-optimized quality metrics
        
    Returns:
        Dictionary with quality statistics and tier distribution
    """
    if not image_paths:
        return {'error': 'No images provided'}
    
    # Sample images for efficiency
    if len(image_paths) > sample_size:
        indices = np.random.choice(len(image_paths), sample_size, replace=False)
        sample_paths = [image_paths[i] for i in indices]
    else:
        sample_paths = image_paths
    
    quality_scores = []
    edge_scores = []
    line_scores = []
    
    for path in sample_paths:
        try:
            img = cv2.imread(str(path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if anime_mode:
                    score = calculate_anime_quality(img)
                else:
                    score = calculate_image_quality(img)
                
                quality_scores.append(score)
        except Exception:
            continue
    
    if not quality_scores:
        return {'error': 'Could not analyze any images'}
    
    quality_scores = np.array(quality_scores)
    
    # Calculate tier distribution
    hd_count = np.sum(quality_scores > 0.8)
    standard_count = np.sum((quality_scores > 0.5) & (quality_scores <= 0.8))
    low_count = np.sum(quality_scores <= 0.5)
    
    result = {
        'mean_quality': float(np.mean(quality_scores)),
        'median_quality': float(np.median(quality_scores)),
        'min_quality': float(np.min(quality_scores)),
        'max_quality': float(np.max(quality_scores)),
        'std_quality': float(np.std(quality_scores)),
        'tier_distribution': {
            'hd_clean': int(hd_count),
            'standard': int(standard_count),
            'low_compressed': int(low_count),
            'total_analyzed': len(quality_scores)
        },
        'recommended_preset': recommend_degradation_preset(float(np.mean(quality_scores)), anime_mode),
        'anime_mode': anime_mode
    }
    
    # Additional anime-specific stats
    if anime_mode:
        result['notes'] = [
            'Edge density analyzed (anime has distinct outlines)',
            'Line sharpness measured (Sobel gradient)',
            'Color flatness checked (anime uses flat shading)',
            'Compression artifacts detected'
        ]
    
    return result


def recommend_degradation_preset(mean_quality: float, anime_mode: bool = False) -> str:
    """
    Recommend degradation preset based on mean quality score.
    
    Args:
        mean_quality: Mean quality score from dataset analysis
        anime_mode: If True, apply anime-specific thresholds
    """
    if anime_mode:
        # Anime-specific thresholds (anime content is typically cleaner than photos)
        if mean_quality > 0.6:
            return 'light'
        elif mean_quality > 0.4:
            return 'medium'
        else:
            return 'heavy'
    else:
        # Standard thresholds
        if mean_quality > 0.8:
            return 'light'
        elif mean_quality > 0.5:
            return 'medium'
        else:
            return 'heavy'


def compute_optimal_crop_size(
    image_paths: List[Path],
    mode: str = 'auto',
    manual_size: Optional[int] = None,
    max_size: int = 960,
    min_size: int = 64,
    sample_size: int = 30
) -> int:
    """
    Compute optimal crop size for a dataset.
    
    Args:
        image_paths: List of image file paths
        mode: 'auto' or 'manual'
        manual_size: Crop size when mode='manual'
        max_size: Maximum crop size in auto mode (prevents OOM)
        min_size: Minimum crop size
        sample_size: Number of images to sample
        
    Returns:
        Optimal crop size
    """
    if mode == 'manual' and manual_size is not None:
        return max(min_size, min(manual_size, max_size * 2))  # Allow larger for manual
    
    if not image_paths:
        return 128  # Default
    
    # Sample images
    if len(image_paths) > sample_size:
        indices = np.random.choice(len(image_paths), sample_size, replace=False)
        sample_paths = [image_paths[i] for i in indices]
    else:
        sample_paths = image_paths
    
    # Collect dimensions
    heights = []
    widths = []
    
    for path in sample_paths:
        try:
            # Use PIL for faster dimension reading (no full decode)
            with Image.open(path) as img:
                w, h = img.size
                heights.append(h)
                widths.append(w)
        except Exception:
            # Fallback to cv2
            try:
                img = cv2.imread(str(path))
                if img is not None:
                    h, w = img.shape[:2]
                    heights.append(h)
                    widths.append(w)
            except Exception:
                continue
    
    if not heights or not widths:
        return 128  # Default
    
    # Compute statistics
    min_h, min_w = min(heights), min(widths)
    smallest_dim = min(min_h, min_w)
    
    # Choose crop size based on smallest dimension
    # Use nearest power of 2 or standard sizes for efficiency
    standard_sizes = [64, 96, 128, 160, 192, 256, 320, 384, 448, 512, 640, 720, 768, 960]
    
    # Find largest standard size that fits within smallest dimension
    optimal_size = min_size
    for size in standard_sizes:
        if size <= smallest_dim * 0.9:  # Leave 10% margin
            optimal_size = size
    
    # Apply caps
    optimal_size = max(min_size, min(optimal_size, max_size))
    
    return optimal_size


class QualityAnalyzer:
    """
    Analyzer for dataset quality and optimal training parameters.
    
    Provides:
    - Per-image quality scoring
    - Dataset-wide quality statistics
    - Optimal crop size computation
    - Degradation preset recommendations
    """
    
    def __init__(
        self,
        auto_quality_thresholds: Optional[Dict] = None,
        cache_scores: bool = False
    ):
        """
        Args:
            auto_quality_thresholds: Dict with 'hd_threshold' and 'standard_threshold'
            cache_scores: Whether to cache quality scores for determinism
        """
        self.thresholds = auto_quality_thresholds or {
            'hd_threshold': 0.8,
            'standard_threshold': 0.5
        }
        self.cache_scores = cache_scores
        self._score_cache: Dict[Path, float] = {}
    
    def analyze_image(self, img: np.ndarray, img_path: Optional[Path] = None) -> float:
        """Analyze single image quality."""
        if img_path and self.cache_scores and img_path in self._score_cache:
            return self._score_cache[img_path]
        
        score = calculate_image_quality(img)
        
        if img_path and self.cache_scores:
            self._score_cache[img_path] = score
        
        return score
    
    def get_degradation_tier(self, quality_score: float) -> str:
        """
        Map quality score to degradation tier.
        
        Returns:
            'light', 'medium', or 'heavy'
        """
        if quality_score > self.thresholds.get('hd_threshold', 0.8):
            return 'light'
        elif quality_score > self.thresholds.get('standard_threshold', 0.5):
            return 'medium'
        else:
            return 'heavy'
    
    def analyze_dataset(
        self,
        image_paths: List[Path],
        sample_size: int = 50
    ) -> Dict:
        """Analyze dataset quality distribution."""
        return analyze_dataset_quality(image_paths, sample_size)
    
    def compute_crop_size(
        self,
        image_paths: List[Path],
        mode: str = 'auto',
        manual_size: Optional[int] = None,
        max_size: int = 960,
        min_size: int = 64
    ) -> int:
        """Compute optimal crop size for dataset."""
        return compute_optimal_crop_size(
            image_paths, mode, manual_size, max_size, min_size
        )


def get_degradation_preset_config(preset: str, base_config: Optional[Dict] = None) -> Dict:
    """
    Get degradation configuration for a preset.
    
    Args:
        preset: 'light', 'medium', 'heavy', 'anime', or 'disabled'
        base_config: User-provided config to merge with preset
        
    Returns:
        Merged degradation configuration
    """
    presets = {
        'disabled': {
            'enabled': False,
        },
        'light': {
            'enabled': True,
            'blur_prob': 0.3,
            'blur_sigma': [0.1, 1.5],
            'noise_prob': 0.2,
            'noise_sigma': [0, 15],
            'jpeg_prob': 0.2,
            'jpeg_quality': [80, 90, 95, 100],
            'resize_prob': [0.1, 0.3, 0.6],  # Less aggressive resizing
        },
        'medium': {
            'enabled': True,
            'blur_prob': 0.7,
            'blur_sigma': [0.1, 2.5],
            'noise_prob': 0.5,
            'noise_sigma': [0, 25],
            'jpeg_prob': 0.5,
            'jpeg_quality': [70, 80, 90, 95, 100],
            'resize_prob': [0.2, 0.7, 0.1],
        },
        'heavy': {
            'enabled': True,
            'blur_prob': 0.9,
            'blur_sigma': [0.5, 4.0],
            'noise_prob': 0.8,
            'noise_sigma': [5, 50],
            'jpeg_prob': 0.7,
            'jpeg_quality': [60, 70, 75, 80, 85],
            'resize_prob': [0.3, 0.6, 0.1],
        },
        'anime': {
            'enabled': True,
            'use_anime_pipeline': True,
            'blur_prob': 0.4,
            'blur_sigma': [0.2, 2.0],
        }
    }
    
    preset_config = presets.get(preset, presets['medium']).copy()
    
    # Merge with user config
    if base_config:
        preset_config.update({
            k: v for k, v in base_config.items()
            if v is not None and k != 'mode' and k != 'presets'
        })
    
    return preset_config
