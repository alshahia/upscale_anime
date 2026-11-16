"""
Anime-specific degradation models for training.
Simulates realistic compression artifacts common in anime distribution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AnimeBlur(nn.Module):
    """
    Anime-optimized blur kernels.
    Preserves sharp lines better than standard blur.
    """
    
    def __init__(self, kernel_sizes=[3, 5, 7, 9], sigma_range=(0.3, 2.5)):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.sigma_range = sigma_range
    
    def get_gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """Create Gaussian kernel."""
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g.outer(g)
        return kernel
    
    def forward(self, x: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Apply anime-optimized blur."""
        skip_prob = torch.rand(1, generator=generator, device=x.device)
        if skip_prob.item() > 0.7:  # 70% chance to skip
            return x
        
        # Random kernel size and sigma
        idx = torch.randint(len(self.kernel_sizes), (1,), generator=generator, device=x.device)
        size = self.kernel_sizes[idx.item()]
        sigma = torch.rand(1, generator=generator, device=x.device) * (self.sigma_range[1] - self.sigma_range[0]) + self.sigma_range[0]
        
        # Create kernel
        kernel = self.get_gaussian_kernel(size, sigma).to(x.device)
        kernel = kernel.view(1, 1, size, size).expand(x.shape[1], -1, -1, -1)
        
        # Apply padding
        pad = size // 2
        x_padded = F.pad(x, (pad, pad, pad, pad), mode='replicate')
        
        # Convolve (depthwise)
        blurred = F.conv2d(x_padded, kernel, groups=x.shape[1])
        
        return blurred


class DirectionalBlur(nn.Module):
    """
    Motion blur for anime (horizontal/vertical).
    Simulates interlacing or motion artifacts.
    """
    
    def __init__(self, max_kernel=7):
        super().__init__()
        self.max_kernel = max_kernel
    
    def forward(self, x: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Apply directional blur."""
        skip_prob = torch.rand(1, generator=generator, device=x.device)
        if skip_prob.item() > 0.3:
            return x
        
        direction_choice = torch.rand(1, generator=generator, device=x.device)
        direction = 'horizontal' if direction_choice.item() < 0.5 else 'vertical'
        strength = torch.randint(3, self.max_kernel + 1, (1,), generator=generator, device=x.device)
        
        if direction == 'horizontal':
            kernel_size = (strength.item(), 1)
        else:
            kernel_size = (1, strength.item())
        
        # Box blur
        kernel = torch.ones(kernel_size, device=x.device)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, *kernel_size).expand(x.shape[1], -1, -1, -1)
        
        # Pad
        pad_h = kernel_size[0] // 2
        pad_w = kernel_size[1] // 2
        x_padded = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
        
        # Apply
        blurred = F.conv2d(x_padded, kernel, groups=x.shape[1])
        
        return blurred


class ColorQuantization(nn.Module):
    """
    Simulates 8-bit color banding and posterization.
    Common in anime due to limited color palettes.
    """
    
    def __init__(self, levels_range=[4, 8, 16, 32]):
        super().__init__()
        self.levels_range = levels_range
    
    def forward(self, x: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Apply color quantization."""
        skip_prob = torch.rand(1, generator=generator, device=x.device)
        if skip_prob.item() > 0.5:
            return x
        
        # Random quantization levels
        idx = torch.randint(len(self.levels_range), (1,), generator=generator, device=x.device)
        levels = self.levels_range[idx.item()]
        
        # Quantize
        x_quantized = torch.round(x * (levels - 1)) / (levels - 1)
        
        # Add small noise to hide banding (dithering simulation)
        noise = torch.rand_like(x, generator=generator) * (1.0 / levels) * 0.5
        x_quantized = x_quantized + noise
        
        return torch.clamp(x_quantized, 0, 1)


class BandingArtifact(nn.Module):
    """
    Simulates gradient banding common in compressed anime.
    Visible in smooth color transitions (sky, gradients).
    """
    
    def __init__(self, band_strength_range=(0.01, 0.05)):
        super().__init__()
        self.band_strength_range = band_strength_range
    
    def forward(self, x: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Add banding artifacts."""
        skip_prob = torch.rand(1, generator=generator, device=x.device)
        if skip_prob.item() > 0.4:
            return x
        
        strength = torch.rand(1, generator=generator, device=x.device) * (self.band_strength_range[1] - self.band_strength_range[0]) + self.band_strength_range[0]
        
        # Detect smooth regions (low gradient)
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        grad_x = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
        grad_y = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
        
        # Smooth regions (combine horizontal and vertical gradients)
        grad_x_mean = grad_x.mean(dim=(2, 3), keepdim=True)
        grad_y_mean = grad_y.mean(dim=(2, 3), keepdim=True)
        smooth_mask = (((grad_x_mean + grad_y_mean) / 2) < 0.1).float()
        
        # Add banding in smooth regions
        banding = torch.round(x / strength) * strength
        
        # Blend
        result = smooth_mask * banding + (1 - smooth_mask) * x
        
        return result


class RingingArtifact(nn.Module):
    """
    Simulates ringing/ghosting artifacts from sharp edges.
    Common in anime with hard edges.
    """
    
    def __init__(self, strength_range=(0.02, 0.08)):
        super().__init__()
        self.strength_range = strength_range
    
    def forward(self, x: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Add ringing artifacts."""
        skip_prob = torch.rand(1, generator=generator, device=x.device)
        if skip_prob.item() > 0.3:
            return x
        
        strength = torch.rand(1, generator=generator, device=x.device) * (self.strength_range[1] - self.strength_range[0]) + self.strength_range[0]
        
        # Detect edges
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device).float()
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=x.device).float()
        
        # Pad
        gray_padded = F.pad(gray, (1, 1, 1, 1), mode='replicate')
        
        # Convolve
        gx = F.conv2d(gray_padded, sobel_x.view(1, 1, 3, 3))
        gy = F.conv2d(gray_padded, sobel_y.view(1, 1, 3, 3))
        
        edge_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        edge_mask = (edge_mag > 0.2).float()
        
        # Dilate edge mask
        edge_mask = F.max_pool2d(edge_mask, 3, stride=1, padding=1)
        
        # Add ringing (opposite polarity near edges)
        ringing = -strength * edge_mask * torch.sign(gx + gy)
        ringing = ringing.expand(-1, 3, -1, -1)
        
        return torch.clamp(x + ringing, 0, 1)


class AnimeDegradationPipeline(nn.Module):
    """
    Complete anime-specific degradation pipeline.
    Combines all anime artifacts in realistic order.
    """
    
    def __init__(
        self,
        enable_blur: bool = True,
        enable_directional: bool = True,
        enable_quantization: bool = True,
        enable_banding: bool = True,
        enable_ringing: bool = True,
    ):
        super().__init__()
        
        self.enable_blur = enable_blur
        self.enable_directional = enable_directional
        self.enable_quantization = enable_quantization
        self.enable_banding = enable_banding
        self.enable_ringing = enable_ringing
        
        # Initialize modules
        if enable_blur:
            self.blur = AnimeBlur()
        if enable_directional:
            self.directional = DirectionalBlur()
        if enable_quantization:
            self.quantization = ColorQuantization()
        if enable_banding:
            self.banding = BandingArtifact()
        if enable_ringing:
            self.ringing = RingingArtifact()
    
    def forward(self, x: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Apply anime degradation pipeline.
        
        Order matters for realistic artifacts:
        1. Blur (capture/encoding blur)
        2. Directional (interlacing)
        3. Ringing (compression artifacts)
        4. Banding (gradient compression)
        5. Quantization (final color depth)
        """
        if self.enable_blur and hasattr(self, 'blur'):
            x = self.blur(x, generator=generator)
        
        if self.enable_directional and hasattr(self, 'directional'):
            x = self.directional(x, generator=generator)
        
        if self.enable_ringing and hasattr(self, 'ringing'):
            x = self.ringing(x, generator=generator)
        
        if self.enable_banding and hasattr(self, 'banding'):
            x = self.banding(x, generator=generator)
        
        if self.enable_quantization and hasattr(self, 'quantization'):
            x = self.quantization(x, generator=generator)
        
        return torch.clamp(x, 0, 1)
    
    def __repr__(self):
        enabled = []
        if self.enable_blur:
            enabled.append('blur')
        if self.enable_directional:
            enabled.append('directional')
        if self.enable_quantization:
            enabled.append('quantization')
        if self.enable_banding:
            enabled.append('banding')
        if self.enable_ringing:
            enabled.append('ringing')
        return f"AnimeDegradationPipeline({', '.join(enabled)})"


def apply_anime_degradation(
    hr_image: torch.Tensor,
    scale: int = 4,
    enable_all: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply anime degradation to create LR/HR pair.
    
    Args:
        hr_image: High-res anime image [B, C, H, W]
        scale: Downsampling scale
        enable_all: Enable all degradation types
        
    Returns:
        (lr_image, hr_degraded) tuple
    """
    # Apply anime-specific degradation
    degradation = AnimeDegradationPipeline(
        enable_blur=enable_all,
        enable_directional=enable_all,
        enable_quantization=enable_all,
        enable_banding=enable_all,
        enable_ringing=enable_all,
    )
    
    hr_degraded = degradation(hr_image)
    
    # Downsample with bicubic (preserves anime degradation patterns)
    lr_image = F.interpolate(
        hr_degraded,
        scale_factor=1.0 / scale,
        mode='bicubic',
        align_corners=False,
    )
    
    # Clamp
    lr_image = torch.clamp(lr_image, 0, 1)
    hr_degraded = torch.clamp(hr_degraded, 0, 1)
    
    return lr_image, hr_degraded
