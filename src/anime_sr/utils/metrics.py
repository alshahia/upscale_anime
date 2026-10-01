"""
Metrics for Super-Resolution evaluation
PSNR, SSIM, LPIPS, and no-reference metrics (NIQE, MANIQA, CLIPIQA, TOPIQ, MUSIQ)
"""
import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Union, Optional

logger = logging.getLogger(__name__)

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

try:
    import pyiqa
    PYIQA_AVAILABLE = True
except ImportError:
    PYIQA_AVAILABLE = False

# Module-level model caches to avoid reinitialization
_lpips_model_cache = {}
_pyiqa_model_cache = {}

# Direction and typical range lookup for pyiqa metrics. Used by
# scripts/evaluate_quality.py to label "higher is better" vs "lower is better"
# and to clamp / interpret score output.
PYIQA_DIRECTION = {
    'niqe': 'lower',
    'maniqa': 'higher',
    'clipiqa': 'higher',
    'topiq_nr': 'higher',
    'musiq': 'higher',
    'brisque': 'lower',
    'arniqa': 'higher',
    'liqe': 'higher',
}

PYIQA_RANGE = {
    'niqe': (0, 20),
    'maniqa': (0, 1),
    'clipiqa': (0, 1),
    'topiq_nr': (0, 1),
    'musiq': (0, 100),
    'brisque': (0, 100),
}


def get_lpips_model(net: str = 'alex') -> Optional[object]:
    """Get or create cached LPIPS model."""
    if not LPIPS_AVAILABLE:
        return None
    
    cache_key = net
    if cache_key not in _lpips_model_cache:
        try:
            _lpips_model_cache[cache_key] = lpips.LPIPS(net=net).eval()
        except Exception:
            _lpips_model_cache[cache_key] = None
    
    return _lpips_model_cache[cache_key]


def get_pyiqa_model(metric_name: str, device: torch.device) -> Optional[object]:
    """Get or create cached pyiqa metric model."""
    if not PYIQA_AVAILABLE:
        return None
    
    cache_key = f"{metric_name}_{device}"
    if cache_key not in _pyiqa_model_cache:
        try:
            _pyiqa_model_cache[cache_key] = pyiqa.create_metric(metric_name, device=device).eval()
        except Exception:
            _pyiqa_model_cache[cache_key] = None
    
    return _pyiqa_model_cache[cache_key]


def _move_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move tensor to device, handling non_blocking properly."""
    return tensor.to(device, non_blocking=True)


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Calculate PSNR between two images.
    
    Args:
        img1: First image [B, C, H, W] or [C, H, W]
        img2: Second image [B, C, H, W] or [C, H, W]
        max_val: Maximum pixel value (1.0 for normalized images)
    
    Returns:
        PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse + 1e-10))
    return psnr.item()


def calculate_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    max_val: float = 1.0,
) -> float:
    """
    Calculate SSIM between two images.
    Simplified implementation.
    
    Args:
        img1: First image [B, C, H, W]
        img2: Second image [B, C, H, W]
        window_size: Size of sliding window
        max_val: Maximum pixel value
    
    Returns:
        SSIM value (0 to 1)
    """
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
    sigma = 1.5
    gauss = torch.Tensor([
        np.exp(-(x - window_size//2)**2 / (2 * sigma**2))
        for x in range(window_size)
    ])
    gauss = gauss / gauss.sum()
    
    window = gauss.unsqueeze(1) * gauss.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0).to(img1.device)
    window = window.repeat(img1.size(1), 1, 1, 1)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 ** 2, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img2.size(1)) - mu1_mu2
    # Clamp variances to >= 0 to avoid NaN from floating-point roundoff.
    sigma1_sq = sigma1_sq.clamp(min=0)
    sigma2_sq = sigma2_sq.clamp(min=0)
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def calculate_lpips(img1: torch.Tensor, img2: torch.Tensor, net: str = 'alex') -> float:
    """
    Calculate LPIPS (Learned Perceptual Image Patch Similarity).
    
    LPIPS is a perceptual metric that correlates better with human judgment
    than PSNR/SSIM. Lower values indicate better perceptual quality.
    
    Reference: "The Unreasonable Effectiveness of Deep Features as a 
    Perceptual Metric" (Zhang et al., 2018)
    
    Args:
        img1: First image [B, C, H, W], range [0, 1]
        img2: Second image [B, C, H, W], range [0, 1]
        net: LPIPS network backbone ('alex', 'vgg', or 'squeeze')
    
    Returns:
        LPIPS distance (lower is better, typical range 0.0 - 0.5)
    """
    if not LPIPS_AVAILABLE:
        return 0.0
    
    lpips_model = get_lpips_model(net)
    if lpips_model is None:
        return 0.0
    
    # Get device from input tensor
    device = img1.device
    
    # Move model to device if needed
    lpips_model = lpips_model.to(device)
    
    # Ensure inputs are in [0, 1] range for LPIPS
    img1 = _move_to_device(img1, device)
    img2 = _move_to_device(img2, device)
    
    if img1.min() < 0 or img1.max() > 1:
        img1 = img1.clamp(0, 1)
    if img2.min() < 0 or img2.max() > 1:
        img2 = img2.clamp(0, 1)
    
    with torch.no_grad():
        # LPIPS expects input in range [-1, 1], so convert
        img1_lpips = img1 * 2 - 1
        img2_lpips = img2 * 2 - 1
        distance = lpips_model(img1_lpips, img2_lpips)
    
    return distance.mean().item()


def calculate_niqe(img: torch.Tensor) -> float:
    """
    Calculate NIQE (Natural Image Quality Evaluator).

    NIQE is a no-reference image quality metric based on natural scene statistics.
    Lower values indicate better quality.

    Args:
        img: Image tensor [B, C, H, W], range [0, 1]

    Returns:
        NIQE score (lower is better, typical range 0 - 10). Returns NaN on failure.
    """
    if not PYIQA_AVAILABLE:
        return float('nan')

    try:
        niqe_metric = get_pyiqa_model('niqe', img.device)
        if niqe_metric is None:
            return float('nan')

        img_batch = img.to(niqe_metric.device)
        if img_batch.min() < 0 or img_batch.max() > 1:
            img_batch = img_batch.clamp(0, 1)
        if img_batch.shape[2] < 200 or img_batch.shape[3] < 200:
            img_batch = F.interpolate(img_batch, size=(200, 200), mode='bicubic', align_corners=False)
        # bicubic overshoots; re-clamp to keep values in [0, 1].
        if img_batch.min() < 0 or img_batch.max() > 1:
            img_batch = img_batch.clamp(0, 1)
        score = niqe_metric(img_batch)
        return score.mean().item() if score.numel() > 0 else float('nan')
    except Exception as e:
        logger.warning(f"NIQE failed: {e}")
        return float('nan')


def calculate_maniqa(img: torch.Tensor) -> float:
    """
    Calculate MANIQA (Multi-Attention No-Reference IQA).

    MANIQA is a learning-based no-reference IQA metric that uses attention
    mechanisms. Higher values indicate better quality.

    Reference: NTIRE 2022 NR-IQA Challenge Winner

    Args:
        img: Image tensor [B, C, H, W], range [0, 1]

    Returns:
        MANIQA score (higher is better, typical range 0 - 1). Returns NaN on failure.
    """
    if not PYIQA_AVAILABLE:
        return float('nan')

    try:
        maniqa_metric = get_pyiqa_model('maniqa', img.device)
        if maniqa_metric is None:
            return float('nan')

        img_batch = img.to(maniqa_metric.device)
        if img_batch.min() < 0 or img_batch.max() > 1:
            img_batch = img_batch.clamp(0, 1)
        if img_batch.shape[2] < 224 or img_batch.shape[3] < 224:
            img_batch = F.interpolate(img_batch, size=(224, 224), mode='bicubic', align_corners=False)
        # bicubic overshoots [0, 1] — re-clamp AFTER resize
        if img_batch.min() < 0 or img_batch.max() > 1:
            img_batch = img_batch.clamp(0, 1)
        score = maniqa_metric(img_batch)
        return score.mean().item() if score.numel() > 0 else float('nan')
    except Exception as e:
        logger.warning(f"MANIQA failed: {e}")
        return float('nan')


def calculate_clipiqa(img: torch.Tensor) -> float:
    """
    Calculate CLIPIQA (CLIP-based Image Quality Assessment).
    
    CLIPIQA uses CLIP embeddings to assess image quality. Higher values 
    indicate better quality. Particularly effective for anime content.
    
    Reference: APISR (CVPR 2024)
    
    Args:
        img: Image tensor [B, C, H, W], range [0, 1]
    
    Returns:
        CLIPIQA score (higher is better, typical range 0 - 1). Returns NaN on failure.
    """
    if not PYIQA_AVAILABLE:
        return float('nan')
    
    try:
        clipiqa_metric = get_pyiqa_model('clipiqa', img.device)
        if clipiqa_metric is None:
            return float('nan')
        
        img_batch = img.to(clipiqa_metric.device)
        img_min = img_batch.min()
        img_max = img_batch.max()
        if img_min < 0 or img_max > 1:
            img_batch = (img_batch - img_min) / (img_max - img_min + 1e-8)
        if img_batch.shape[2] < 224 or img_batch.shape[3] < 224:
            img_batch = F.interpolate(img_batch, size=(224, 224), mode='bicubic', align_corners=False)
        score = clipiqa_metric(img_batch)
        return score.mean().item() if score.numel() > 0 else float('nan')
    except Exception as e:
        logger.warning(f"CLIPIQA failed: {e}")
        return float('nan')


def calculate_topiq(img: torch.Tensor) -> float:
    """
    Calculate TOPIQ-NR (No-Reference version, koniq-pretrained).

    TOPIQ is a top-down perceptual IQA model that predicts human opinion scores.
    Higher values indicate better quality. Effective across natural, face, and
    anime content.

    Reference: "TOPIQ: A Top-down Pipeline from Coarse to Fine for Image Quality
    Assessment" (Chen et al., 2024)

    Args:
        img: Image tensor [B, C, H, W], range [0, 1]

    Returns:
        TOPIQ score (higher is better, typical range 0 - 1). Returns NaN on failure.
    """
    if not PYIQA_AVAILABLE:
        return float('nan')

    try:
        topiq_metric = get_pyiqa_model('topiq_nr', img.device)
        if topiq_metric is None:
            return float('nan')

        img_batch = img.to(topiq_metric.device)
        if img_batch.min() < 0 or img_batch.max() > 1:
            img_batch = img_batch.clamp(0, 1)
        if img_batch.shape[2] < 224 or img_batch.shape[3] < 224:
            img_batch = F.interpolate(img_batch, size=(224, 224), mode='bicubic', align_corners=False)
        # bicubic overshoots [0, 1]; TOPIQ strictly enforces the range. Re-clamp AFTER resize.
        if img_batch.min() < 0 or img_batch.max() > 1:
            img_batch = img_batch.clamp(0, 1)
        score = topiq_metric(img_batch)
        return score.mean().item() if score.numel() > 0 else float('nan')
    except Exception as e:
        logger.warning(f"TOPIQ failed: {e}")
        return float('nan')


def calculate_musiq(img: torch.Tensor) -> float:
    """
    Calculate MUSIQ (Multi-Scale Image Quality Transformer, koniq-pretrained).

    MUSIQ uses a Vision Transformer trained on multi-scale image patches to
    predict human opinion scores. Higher values indicate better quality.

    Reference: "MUSIQ: Multi-Scale Image Quality Transformer" (Ke et al., ICCV 2021)

    Args:
        img: Image tensor [B, C, H, W], range [0, 1]

    Returns:
        MUSIQ score (higher is better, typical range 0 - 100). Returns NaN on failure.
    """
    if not PYIQA_AVAILABLE:
        return float('nan')

    try:
        musiq_metric = get_pyiqa_model('musiq', img.device)
        if musiq_metric is None:
            return float('nan')

        img_batch = img.to(musiq_metric.device)
        if img_batch.min() < 0 or img_batch.max() > 1:
            img_batch = img_batch.clamp(0, 1)
        # MUSIQ supports multi-scale; let pyiqa handle resizing internally
        # but ensure min side is at least 224 for stable inference.
        if img_batch.shape[2] < 224 or img_batch.shape[3] < 224:
            img_batch = F.interpolate(img_batch, size=(224, 224), mode='bicubic', align_corners=False)
        # bicubic overshoots; re-clamp to keep values in [0, 1].
        if img_batch.min() < 0 or img_batch.max() > 1:
            img_batch = img_batch.clamp(0, 1)
        score = musiq_metric(img_batch)
        return score.mean().item() if score.numel() > 0 else float('nan')
    except Exception as e:
        logger.warning(f"MUSIQ failed: {e}")
        return float('nan')


def calculate_batch_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    max_val: float = 1.0,
    include_lpips: bool = True,
) -> dict:
    """
    Calculate metrics for a batch.
    
    Args:
        preds: Predictions [B, C, H, W]
        targets: Ground truth [B, C, H, W]
        max_val: Maximum pixel value
        include_lpips: Whether to compute LPIPS (requires lpips library)
    
    Returns:
        Dictionary with PSNR, SSIM, and optionally LPIPS
    """
    batch_size = preds.size(0)
    
    psnr_total = 0.0
    ssim_total = 0.0
    lpips_total = 0.0
    
    for i in range(batch_size):
        psnr = calculate_psnr(preds[i], targets[i], max_val)
        ssim = calculate_ssim(preds[i:i+1], targets[i:i+1], max_val=max_val)
        
        psnr_total += psnr
        ssim_total += ssim
        
        if include_lpips:
            lpips_val = calculate_lpips(preds[i:i+1], targets[i:i+1])
            lpips_total += lpips_val
    
    result = {
        'psnr': psnr_total / batch_size,
        'ssim': ssim_total / batch_size,
    }
    
    if include_lpips:
        result['lpips'] = lpips_total / batch_size
    
    return result


class MetricsTracker:
    """Track metrics during training/validation"""
    
    def __init__(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, metrics_dict: dict):
        """Update metrics"""
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            
            self.metrics[key] += value
            self.counts[key] += 1
    
    def get_average(self) -> dict:
        """Get average metrics"""
        return {
            key: self.metrics[key] / self.counts[key]
            for key in self.metrics.keys()
        }
    
    def reset(self):
        """Reset metrics"""
        self.metrics = {}
        self.counts = {}