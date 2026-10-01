"""
Frequency-Aware Loss for Super-Resolution
Based on ESPAN (CVPRW 2025): Expanded SPAN for Efficient Super-Resolution

Uses DCT (Discrete Cosine Transform) to separate frequency components and
applies higher weights to high-frequency regions (edges, textures).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FrequencyAwareLoss(nn.Module):
    """
    Frequency-aware loss that emphasizes high-frequency regions.
    
    Based on ESPAN (CVPRW 2025), this loss uses DCT to separate
    frequency components and applies higher weights to high-frequency
    areas (edges, textures, fine details).
    
    This helps the model focus on recovering important high-frequency
    details rather than just minimizing pixel-wise errors.
    """
    
    def __init__(
        self,
        block_size: int = 8,
        high_freq_weight: float = 2.0,
        mid_freq_weight: float = 1.0,
        low_freq_weight: float = 0.5,
        use_high_pass_filter: bool = True,
    ):
        """
        Initialize frequency-aware loss.
        
        Args:
            block_size: DCT block size (default: 8)
            high_freq_weight: Weight for high-frequency components
            mid_freq_weight: Weight for mid-frequency components
            low_freq_weight: Weight for low-frequency components
            use_high_pass_filter: If True, also apply high-pass filter
        """
        super().__init__()
        self.block_size = block_size
        self.high_freq_weight = high_freq_weight
        self.mid_freq_weight = mid_freq_weight
        self.low_freq_weight = low_freq_weight
        self.use_high_pass_filter = use_high_pass_filter

        # Pre-compute frequency weight mask for DCT coefficients
        self._create_frequency_weight_mask(block_size)

        # Phase 3: pre-compute DCT basis matrix as a registered buffer.
        # Previously rebuilt on every forward via 64 torch.cos() kernel launches.
        self.register_buffer('dct_basis', self._build_dct_basis(block_size))

    @staticmethod
    def _build_dct_basis(block_size: int) -> torch.Tensor:
        """Build the DCT-II basis matrix of shape [block_size, block_size].

        Built once on CPU using math.cos (cheap) and registered as a buffer
        so the trainer's .to(device) call moves it alongside model weights.
        """
        n = block_size
        basis = torch.zeros(n, n, dtype=torch.float32)
        for k in range(n):
            for i in range(n):
                val = math.cos(math.pi * (2 * i + 1) * k / (2 * n))
                if k == 0:
                    val /= math.sqrt(n)
                else:
                    val *= math.sqrt(2.0 / n)
                basis[k, i] = val
        return basis
    
    def _create_frequency_weight_mask(self, block_size: int):
        """
        Create frequency weight mask for DCT coefficients.
        
        DCT coefficients are organized by frequency:
        - Top-left: Low frequency (DC component)
        - Middle: Mid frequency
        - Bottom-right: High frequency
        """
        weight_mask = torch.zeros(block_size, block_size)
        
        for i in range(block_size):
            for j in range(block_size):
                # Distance from DC component (top-left)
                distance = (i ** 2 + j ** 2) ** 0.5
                max_distance = (2 * (block_size - 1) ** 2) ** 0.5
                normalized_distance = distance / max_distance
                
                if normalized_distance < 0.33:
                    weight_mask[i, j] = self.low_freq_weight
                elif normalized_distance < 0.66:
                    weight_mask[i, j] = self.mid_freq_weight
                else:
                    weight_mask[i, j] = self.high_freq_weight
        
        # Register as buffer (persists with model)
        self.register_buffer('weight_mask', weight_mask)
    
    def _dct_2d_block(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D DCT to 8x8 blocks using matrix multiplication.

        Phase 3: the DCT basis matrix is now a registered buffer (built once
        in __init__) instead of being rebuilt via 64 separate torch.cos()
        kernel launches per forward call.

        Args:
            x: Input tensor [B, C, H, W] where H,W are multiples of block_size

        Returns:
            DCT coefficients [B, C, H, W]
        """
        B, C, H, W = x.shape
        bs = self.block_size

        # Reshape into blocks: [B, C, H/bs, bs, W/bs, bs]
        x = x.view(B, C, H // bs, bs, W // bs, bs)
        x = x.permute(0, 1, 2, 4, 3, 5)  # [B, C, n_h, n_w, bs, bs]

        # Use pre-computed DCT basis (registered buffer, device/dtype-aware via .to)
        dct_basis = self.dct_basis.to(device=x.device, dtype=x.dtype)

        # Apply DCT: basis @ x @ basis^T
        x = torch.matmul(dct_basis, x)
        x = torch.matmul(x, dct_basis.t())

        # Reshape back: [B, C, H, W]
        x = x.permute(0, 1, 2, 4, 3, 5)
        x = x.reshape(B, C, H, W)

        return x
    
    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """
        Compute frequency-aware loss.
        
        Args:
            sr: Super-resolved image [B, C, H, W]
            hr: High-resolution ground truth [B, C, H, W]
        
        Returns:
            Frequency-aware loss
        """
        # Compute absolute error
        error = (sr - hr).abs()
        
        # Get spatial dimensions
        B, C, H, W = error.shape
        bs = self.block_size
        
        # Pad to multiple of block_size if needed
        pad_h = (bs - H % bs) % bs
        pad_w = (bs - W % bs) % bs
        if pad_h > 0 or pad_w > 0:
            error = F.pad(error, (0, pad_w, 0, pad_h), mode='reflect')
            H_padded = H + pad_h
            W_padded = W + pad_w
        else:
            H_padded = H
            W_padded = W
        
        # Apply DCT to error
        dct_error = self._dct_2d_block(error)
        
        # Create weight mask for the full spatial dimensions
        # Tile the block_size weight mask to match H_padded x W_padded.
        # weight_mask is a registered buffer, already moved by .to(device) in
        # the trainer setup — no per-call .to() needed.
        weight_mask = self.weight_mask
        n_h = H_padded // bs
        n_w = W_padded // bs
        
        # Tile weight mask: [bs, bs] -> [n_h*bs, n_w*bs]
        full_weight_mask = weight_mask.repeat(n_h, n_w)
        
        # Expand to batch and channel dimensions
        full_weight_mask = full_weight_mask.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
        
        # Apply frequency weighting to DCT coefficients
        weighted_error = dct_error.abs() * full_weight_mask
        
        # Compute loss (mean over all dimensions)
        loss = weighted_error.mean()
        
        # Add high-pass filter component
        if self.use_high_pass_filter:
            hp_sr = self._high_pass_filter(sr)
            hp_hr = self._high_pass_filter(hr)
            hp_loss = F.l1_loss(hp_sr, hp_hr)
            loss = loss + hp_loss
        
        return loss
    
    def _high_pass_filter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply simple high-pass filter to extract high-frequency content.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            High-frequency content [B, C, H, W]
        """
        # Simple Laplacian-like filter
        kernel = torch.tensor([
            [[[-1, -1, -1],
              [-1,  8, -1],
              [-1, -1, -1]]]
        ], dtype=x.dtype, device=x.device).expand(x.size(1), -1, -1, -1)
        
        # Apply convolution
        hp = F.conv2d(x, kernel, padding=1, groups=x.size(1))
        
        return hp.abs()


class CombinedFrequencyLoss(nn.Module):
    """
    Combined loss that integrates frequency-aware loss with pixel loss.
    
    This provides a convenient wrapper for using frequency-aware loss
    alongside other losses.
    """
    
    def __init__(
        self,
        pixel_weight: float = 1.0,
        frequency_weight: float = 0.1,
        block_size: int = 8,
        high_freq_weight: float = 2.0,
    ):
        """
        Initialize combined frequency loss.
        
        Args:
            pixel_weight: Weight for pixel (L1) loss
            frequency_weight: Weight for frequency-aware loss
            block_size: DCT block size
            high_freq_weight: Weight for high-frequency components
        """
        super().__init__()
        self.pixel_weight = pixel_weight
        self.frequency_weight = frequency_weight
        
        self.pixel_loss = nn.L1Loss()
        self.frequency_loss = FrequencyAwareLoss(
            block_size=block_size,
            high_freq_weight=high_freq_weight,
        )
    
    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            sr: Super-resolved image [B, C, H, W]
            hr: High-resolution ground truth [B, C, H, W]
        
        Returns:
            Combined loss
        """
        pixel_loss = self.pixel_loss(sr, hr)
        freq_loss = self.frequency_loss(sr, hr)
        
        return self.pixel_weight * pixel_loss + self.frequency_weight * freq_loss
