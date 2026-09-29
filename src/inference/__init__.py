"""
Inference module for Anime Super-Resolution.

Provides:
- InferenceEngine: High-level inference interface
- ModelSoup: Weight averaging across checkpoints
- Test-Time Augmentation (TTA) support
- EMA model loading for stable inference

Example usage:
    >>> from inference import InferenceEngine, quick_inference
    >>> 
    >>> # Basic inference with EMA weights
    >>> engine = InferenceEngine.from_checkpoint('checkpoints/best.pth', use_ema=True)
    >>> sr_image = engine.run(lr_image)
    >>> 
    >>> # Inference with TTA for better quality
    >>> sr_image = engine.run_tta(lr_image, augmentations=['identity', 'flip_h', 'flip_v'])
    >>> 
    >>> # Quick inference
    >>> quick_inference('checkpoints/best.pth', 'input.png', 'output.png', use_ema=True)
    >>> 
    >>> # Create model soup from multiple checkpoints
    >>> from inference import ModelSoup
    >>> soup = ModelSoup.from_checkpoints([
    ...     'checkpoints/epoch_100.pth',
    ...     'checkpoints/epoch_150.pth',
    ...     'checkpoints/best.pth'
    ... ])
    >>> soup.save_soup('checkpoints/soup.pth')
"""

from inference.engine import (
    InferenceEngine,
    quick_inference,
    ModelSoup,
    TTA_TRANSFORMS,
)

from inference.tta import (
    tta_forward,
    D4_AUGMENTATIONS,
)

__all__ = [
    'InferenceEngine',
    'quick_inference',
    'ModelSoup',
    'TTA_TRANSFORMS',
    'tta_forward',
    'D4_AUGMENTATIONS',
]
