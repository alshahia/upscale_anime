"""
DegradationPipeline - Single source of truth for image degradation.

This module consolidates the degradation logic that previously lived in
BaseDataset and the stub `_apply_degradation_cpu` in PreprocessingManager.
Both the dataset and the manager now use this class so that the same
algorithmic behavior is guaranteed end-to-end.

Supports:
- 'bicubic'       : pure bicubic downsample (fastest, no degradation)
- 'light'         : mild blur + noise + JPEG
- 'medium'        : moderate blur + noise + JPEG
- 'heavy'         : strong blur + noise + JPEG
- 'anime'         : anime-tuned (banding, color quantization)
- 'anime_heavy'   : APISR-style anime heavy degradation (default)
- 'two_stage'     : APISR two-stage compression (degrade -> resize -> second-order)
- 'shuffled'      : randomize order of blur/noise/resize/compression

The pipeline is a callable: img_uint8_in -> img_uint8_out.
"""
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    from data.compression_modules import CompressionPipeline
except ImportError:
    try:
        from src.data.compression_modules import CompressionPipeline
    except ImportError:
        CompressionPipeline = None
        logger.warning("CompressionPipeline not available; two-stage compression will fall back to JPEG.")


PRESETS: Dict[str, Dict] = {
    'bicubic': {},
    'light': {
        'blur_prob': 0.3, 'blur_sigma': [0.1, 1.5],
        'noise_prob': 0.2, 'noise_sigma': [0, 15],
        'jpeg_prob': 0.2, 'jpeg_quality': [80, 90, 95],
        'compression_prob': 0.3, 'second_order_prob': 0.1,
    },
    'medium': {
        'blur_prob': 0.7, 'blur_sigma': [0.1, 2.5],
        'noise_prob': 0.5, 'noise_sigma': [0, 25],
        'jpeg_prob': 0.5, 'jpeg_quality': [70, 80, 90],
        'compression_prob': 0.5, 'second_order_prob': 0.3,
    },
    'heavy': {
        'blur_prob': 0.9, 'blur_sigma': [0.5, 4.0],
        'noise_prob': 0.8, 'noise_sigma': [5, 50],
        'jpeg_prob': 0.7, 'jpeg_quality': [60, 70, 80],
        'compression_prob': 0.7, 'second_order_prob': 0.4,
    },
    'anime': {
        'blur_prob': 0.7, 'blur_sigma': [0.1, 3.0],
        'noise_prob': 0.5, 'noise_sigma': [0, 30],
        'jpeg_prob': 0.6, 'jpeg_quality': [60, 70, 80, 90],
        'compression_prob': 0.7, 'second_order_prob': 0.3,
        'anime_degradation': True, 'color_quantization': True,
    },
    'anime_heavy': {
        'blur_prob': 0.8, 'blur_sigma': [0.1, 3.0],
        'noise_prob': 0.6, 'noise_sigma': [0, 30],
        'jpeg_prob': 0.6, 'jpeg_quality': [50, 60, 70, 80, 90],
        'compression_prob': 0.8, 'second_order_prob': 0.4,
        'anime_degradation': True, 'color_quantization': True,
        'banding_simulation': True,
    },
}


def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return np.clip(img, 0, 255).astype(np.uint8)


def _apply_blur(img: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = kernel * kernel.T
    return cv2.filter2D(img, -1, kernel)


def _apply_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma == 0:
        return img
    noise = rng.standard_normal(img.shape).astype(np.float32) * sigma
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encimg, 1)


class DegradationPipeline:
    """
    Callable image degradation pipeline.

    Args:
        mode: One of the keys in PRESETS, or a custom name to use `cfg` directly.
        cfg: Optional dict of degradation parameters. If None, the preset is used.
        scale: Upscale factor (HR size / LR size).
        two_stage: If True and mode is one of the heavy presets, use two-stage
            compression (Stage 1 degrade full image, resize, Stage 2 light degrade).
        compression_stage1: List of compression types for stage 1
            (e.g. ['jpeg', 'webp']). Defaults to ['jpeg', 'webp'].
        compression_stage2: List of compression types for stage 2
            (e.g. ['avif', 'h264', 'jpeg']). Defaults to ['avif', 'h264', 'jpeg'].
        seed: Optional integer for deterministic degradation.
    """

    def __init__(
        self,
        mode: str = 'anime_heavy',
        cfg: Optional[Dict] = None,
        scale: int = 4,
        two_stage: bool = False,
        compression_stage1: Optional[List[str]] = None,
        compression_stage2: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ):
        self.mode = mode
        self.scale = scale
        self.two_stage = two_stage
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        if cfg is None and mode in PRESETS:
            self.cfg = dict(PRESETS[mode])
        else:
            self.cfg = dict(cfg or {})

        if mode == 'bicubic':
            self.cfg = {}

        self._stage1_types = compression_stage1 or ['jpeg', 'webp']
        self._stage2_types = compression_stage2 or ['avif', 'h264', 'jpeg']
        self._stage1_pipeline = None
        self._stage2_pipeline = None
        if two_stage and CompressionPipeline is not None:
            try:
                self._stage1_pipeline = CompressionPipeline(
                    compression_types=self._stage1_types,
                    quality_ranges=[(60, 95)] * len(self._stage1_types),
                )
                self._stage2_pipeline = CompressionPipeline(
                    compression_types=self._stage2_types,
                    quality_ranges=[(60, 90)] * len(self._stage2_types),
                )
            except Exception as e:
                logger.warning(f"Failed to init CompressionPipeline: {e}")
                self._stage1_pipeline = None
                self._stage2_pipeline = None

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Apply degradation. Input/output are HxWxC uint8 RGB."""
        if img is None:
            return img
        img = _to_uint8(img)
        cfg = self.cfg
        if not cfg and self.mode == 'bicubic':
            h, w = img.shape[:2]
            lr_h, lr_w = max(1, h // self.scale), max(1, w // self.scale)
            return cv2.resize(img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
        if not cfg:
            return self._bicubic_only(img)

        if self.two_stage:
            return self._apply_two_stage(img, cfg)
        return self._apply_shuffled(img, cfg)

    def _bicubic_only(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        lr_h, lr_w = max(1, h // self.scale), max(1, w // self.scale)
        return cv2.resize(img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)

    def _apply_shuffled(self, img: np.ndarray, cfg: Dict) -> np.ndarray:
        """APISR-style shuffled degradation. Random order of ops."""
        operations: List[str] = []
        if self._rng.random() < cfg.get('blur_prob', 0.7):
            operations.append('blur')
        if self._rng.random() < cfg.get('noise_prob', 0.5):
            operations.append('noise')
        operations.append('resize')
        if self._rng.random() < cfg.get('jpeg_prob', 0.5):
            operations.append('compression')
        self._rng.shuffle(operations)

        for op in operations:
            if op == 'blur':
                ks = self._rng.choice(cfg.get('blur_kernel_size', [7, 9, 11]))
                sigma_range = cfg.get('blur_sigma', [0.1, 3.0])
                sigma = float(self._rng.uniform(sigma_range[0], sigma_range[1]))
                img = _apply_blur(img, int(ks), sigma)
            elif op == 'noise':
                sigma_range = cfg.get('noise_sigma', [0, 25])
                sigma = float(self._rng.uniform(sigma_range[0], sigma_range[1]))
                img = _apply_noise(img, sigma, self._rng)
            elif op == 'resize':
                img = self._random_resize(img, cfg)
            elif op == 'compression':
                if self._stage1_pipeline is not None and self.two_stage is False:
                    img = self._stage1_pipeline(img)
                else:
                    qualities = cfg.get('jpeg_quality', [60, 70, 80, 90, 100])
                    q = int(self._rng.choice(qualities))
                    img = _apply_jpeg(img, q)
        return img

    def _random_resize(self, img: np.ndarray, cfg: Dict) -> np.ndarray:
        resize_probs = cfg.get('resize_prob', [0.2, 0.7, 0.1])
        types = ['up', 'down', 'keep']
        idx = int(self._rng.choice(len(types), p=resize_probs))
        resize_type = types[idx]
        if resize_type == 'keep':
            return img
        h, w = img.shape[:2]
        resize_range = cfg.get('resize_range', [0.15, 1.5])
        if resize_type == 'up':
            factor = float(self._rng.uniform(1.0, resize_range[1]))
        else:
            factor = float(self._rng.uniform(resize_range[0], 1.0))
        new_h, new_w = max(1, int(h * factor)), max(1, int(w * factor))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def _apply_two_stage(self, img: np.ndarray, cfg: Dict) -> np.ndarray:
        """Stage 1: degrade full image -> resize -> Stage 2: light degrade."""
        h, w = img.shape[:2]
        img = self._apply_shuffled(img, cfg)
        if self._stage1_pipeline is not None and self._rng.random() < cfg.get('compression_prob', 0.7):
            img = self._stage1_pipeline(img)
        lr_h, lr_w = max(1, h // self.scale), max(1, w // self.scale)
        img = cv2.resize(img, (lr_w, lr_h), interpolation=cv2.INTER_LINEAR)

        if self._rng.random() < cfg.get('second_order_prob', 0.3):
            ks = int(self._rng.choice([3, 5, 7]))
            sigma = float(self._rng.uniform(0.1, 1.0))
            img = _apply_blur(img, ks, sigma)
        if self._rng.random() < cfg.get('second_order_prob', 0.3):
            sigma = float(self._rng.uniform(0, 10))
            img = _apply_noise(img, sigma, self._rng)

        if self._stage2_pipeline is not None and self._rng.random() < cfg.get('compression_prob', 0.5):
            img = self._stage2_pipeline(img)
        elif self._rng.random() < cfg.get('jpeg_prob', 0.3):
            qualities = cfg.get('jpeg_quality', [70, 80, 90, 100])
            q = int(self._rng.choice(qualities))
            img = _apply_jpeg(img, q)
        return img


__all__ = ['DegradationPipeline', 'PRESETS']
