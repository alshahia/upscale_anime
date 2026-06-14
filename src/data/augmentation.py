"""
Advanced augmentation techniques for super-resolution training.
Includes Mixup, CutMix, and progressive augmentation strategies.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class MixupAugmentation:
    """
    Mixup augmentation for super-resolution.
    Blends two HR/LR patch pairs with random weights.
    """
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self._rng = torch.Generator()
        self._rng.manual_seed(torch.initial_seed() % (2**32))
    
    def __call__(self, hr1: torch.Tensor, hr2: torch.Tensor,
                 lr1: torch.Tensor, lr2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup to HR and LR pairs.
        
        Args:
            hr1, hr2: High-resolution images (C, H, W)
            lr1, lr2: Low-resolution images (C, h, w)
        
        Returns:
            hr_mixed, lr_mixed, lambda
        """
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        hr_mixed = lam * hr1 + (1 - lam) * hr2
        lr_mixed = lam * lr1 + (1 - lam) * lr2
        return hr_mixed, lr_mixed, lam.item()


class CutMixAugmentation:
    """
    CutMix augmentation for super-resolution.
    Cuts a region from one image and pastes into another.
    """
    def __init__(self, min_ratio: float = 0.2, max_ratio: float = 0.8):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self._rng = torch.Generator()
        self._rng.manual_seed(torch.initial_seed() % (2**32))
    
    def __call__(self, hr1: torch.Tensor, hr2: torch.Tensor,
                 lr1: torch.Tensor, lr2: torch.Tensor,
                 scale: int = 4) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix to HR and LR pairs.
        
        Args:
            hr1, hr2: High-resolution images (C, H, W)
            lr1, lr2: Low-resolution images (C, h, w)
            scale: Super-resolution scale factor
        
        Returns:
            hr_mixed, lr_mixed, ratio
        """
        _, h, w = hr1.shape
        
        # Random box size
        ratio = torch.empty(1).uniform_(self.min_ratio, self.max_ratio, generator=self._rng).item()
        cut_h = int(h * np.sqrt(ratio))
        cut_w = int(w * np.sqrt(ratio))
        
        # Random position
        cx = torch.randint(0, w - cut_w + 1, (1,), generator=self._rng).item()
        cy = torch.randint(0, h - cut_h + 1, (1,), generator=self._rng).item()
        
        # Apply CutMix to HR
        hr_mixed = hr1.clone()
        hr_mixed[:, cx:cx+cut_h, cy:cy+cut_w] = hr2[:, cx:cx+cut_h, cy:cy+cut_w]
        
        # Apply corresponding CutMix to LR
        lr_mixed = lr1.clone()
        lr_cx, lr_cy = cx // scale, cy // scale
        lr_ch, lr_cw = cut_h // scale, cut_w // scale
        lr_mixed[:, lr_cx:lr_cx+lr_ch, lr_cy:lr_cy+lr_cw] = lr2[:, lr_cx:lr_cx+lr_ch, lr_cy:lr_cy+lr_cw]
        
        return hr_mixed, lr_mixed, ratio


class RandomResizedCrop:
    """
    Random resized crop augmentation.
    Randomly crops and resizes to target size.
    """
    def __init__(self, scale_range: Tuple[float, float] = (0.5, 2.0)):
        self.scale_range = scale_range
        self._rng = torch.Generator()
        self._rng.manual_seed(torch.initial_seed() % (2**32))
    
    def __call__(self, image: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Apply random resized crop.
        
        Args:
            image: Input image (C, H, W)
            target_size: (H, W) target size
        
        Returns:
            Resized image
        """
        _, h, w = image.shape
        
        # Random scale
        scale = torch.empty(1).uniform_(*self.scale_range, generator=self._rng).item()
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize
        resized = F.interpolate(
            image.unsqueeze(0),
            size=(new_h, new_w),
            mode='bicubic',
            align_corners=False
        ).squeeze(0)
        
        # Random crop to target size
        if new_h > target_size[0] and new_w > target_size[1]:
            x = torch.randint(0, new_h - target_size[0] + 1, (1,), generator=self._rng).item()
            y = torch.randint(0, new_w - target_size[1] + 1, (1,), generator=self._rng).item()
            return resized[:, x:x+target_size[0], y:y+target_size[1]]
        else:
            # Pad if too small
            return F.interpolate(
                resized.unsqueeze(0),
                size=target_size,
                mode='bicubic',
                align_corners=False
            ).squeeze(0)


class AugmentationPipeline:
    """
    Complete augmentation pipeline for SR training.
    """
    def __init__(self, 
                 mixup_alpha: float = 0.4,
                 cutmix_alpha: float = 1.0,
                 mixup_prob: float = 0.3,
                 cutmix_prob: float = 0.3,
                 random_resize_prob: float = 0.5,
                 random_resize_range: Tuple[float, float] = (0.5, 2.0)):
        
        self.mixup = MixupAugmentation(alpha=mixup_alpha)
        self.cutmix = CutMixAugmentation()
        self.random_resize = RandomResizedCrop(scale_range=random_resize_range)
        
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        self.random_resize_prob = random_resize_prob
        
        self.use_mixup = mixup_prob > 0
        self.use_cutmix = cutmix_prob > 0
        self.use_random_resize = random_resize_prob > 0
        
        self._rng = torch.Generator()
        self._rng.manual_seed(torch.initial_seed() % (2**32))
    
    def apply_to_batch(self, hr_batch: torch.Tensor, lr_batch: torch.Tensor,
                       scale: int = 4) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply augmentation to a batch.
        
        Args:
            hr_batch: (B, C, H, W) HR images
            lr_batch: (B, C, h, w) LR images
            scale: SR scale factor
        
        Returns:
            hr_aug, lr_aug, mixup_lambdas (if applicable)
        """
        batch_size = hr_batch.size(0)
        device = hr_batch.device
        
        mixup_lambdas = None
        
        # Apply Mixup
        if self.use_mixup and torch.rand(1, generator=self._rng).item() < self.mixup_prob:
            indices = torch.randperm(batch_size, device=device)
            hr_shuffled = hr_batch[indices]
            lr_shuffled = lr_batch[indices]
            
            # Sample lambdas for each image
            dist = torch.distributions.Beta(0.4, 0.4)
            lambdas = dist.sample((batch_size,)).float().to(device)
            
            # Expand for broadcasting
            lambdas_expanded = lambdas.view(batch_size, 1, 1, 1)
            
            hr_batch = lambdas_expanded * hr_batch + (1 - lambdas_expanded) * hr_shuffled
            lr_batch = lambdas_expanded * lr_batch + (1 - lambdas_expanded) * lr_shuffled
            
            mixup_lambdas = lambdas
        
        # Apply CutMix (simpler version - per-image)
        if self.use_cutmix and torch.rand(1, generator=self._rng).item() < self.cutmix_prob:
            # For now, skip cutmix for simplicity
            # Can be implemented similarly to mixup
            pass
        
        return hr_batch, lr_batch, mixup_lambdas


def apply_geometric_augmentation(hr: torch.Tensor, lr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply random geometric augmentations.
    
    Args:
        hr: HR image (C, H, W)
        lr: LR image (C, h, w)
    
    Returns:
        Augmented (hr, lr)
    """
    # Random horizontal flip
    if torch.rand(1).item() > 0.5:
        hr = torch.flip(hr, dims=[-1])
        lr = torch.flip(lr, dims=[-1])
    
    # Random vertical flip
    if torch.rand(1).item() > 0.5:
        hr = torch.flip(hr, dims=[-2])
        lr = torch.flip(lr, dims=[-2])
    
    # Random rotation (0, 90, 180, 270)
    if torch.rand(1).item() > 0.5:
        k = torch.randint(1, 4, (1,)).item()  # 1, 2, or 3 times 90 degrees
        hr = torch.rot90(hr, k=k, dims=[-2, -1])
        lr = torch.rot90(lr, k=k, dims=[-2, -1])
    
    return hr, lr


def apply_color_jitter(image: torch.Tensor, 
                       brightness: float = 0.2,
                       contrast: float = 0.2,
                       saturation: float = 0.2) -> torch.Tensor:
    """
    Apply color jittering.
    
    Args:
        image: (C, H, W) RGB image
        brightness: Brightness jitter factor
        contrast: Contrast jitter factor
        saturation: Saturation jitter factor
    
    Returns:
        Jittered image
    """
    # Convert to HSV for easier manipulation
    # This is a simplified version - full implementation would use proper color space conversion
    
    # Random brightness
    if brightness > 0:
        factor = torch.empty(1).uniform_(1 - brightness, 1 + brightness).item()
        image = image * factor
    
    # Random contrast
    if contrast > 0:
        mean = image.mean(dim=(-2, -1), keepdim=True)
        factor = torch.empty(1).uniform_(1 - contrast, 1 + contrast).item()
        image = (image - mean) * factor + mean
    
    # Clamp to valid range
    image = torch.clamp(image, 0, 1)
    
    return image
