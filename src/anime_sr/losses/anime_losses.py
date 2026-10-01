"""
Anime-specific loss functions for super-resolution.
Designed to preserve line art, color consistency, and flat regions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LineArtPreservationLoss(nn.Module):
    """
    Loss that preserves sharp lines common in anime.
    Uses multi-scale edge detection to capture both thick and thin lines.
    
    Edge methods:
    - sobel: Standard 3x3 Sobel edge detection
    - laplacian: Laplacian edge detection for fine details
    - multi_scale: Combines Sobel at multiple kernel sizes (default, best for anime)
    """
    
    def __init__(self, edge_weight: float = 3.0, edge_method: str = 'multi_scale'):
        super().__init__()
        self.edge_weight = edge_weight
        self.edge_method = edge_method
        
        # Sobel filters for edge detection (3x3)
        sobel_3x3 = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        # Sobel filters for edge detection (5x5 - captures thinner lines)
        sobel_5x5 = torch.tensor([
            [-1, -2, 0, 2, 1],
            [-2, -4, 0, 4, 2],
            [-4, -8, 0, 8, 4],
            [-2, -4, 0, 4, 2],
            [-1, -2, 0, 2, 1]
        ], dtype=torch.float32).view(1, 1, 5, 5)
        
        # Laplacian kernel for fine edge detection
        laplacian = torch.tensor([
            [0, 0, -1, 0, 0],
            [0, -1, -2, -1, 0],
            [-1, -2, 17, -2, -1],
            [0, -1, -2, -1, 0],
            [0, 0, -1, 0, 0]
        ], dtype=torch.float32).view(1, 1, 5, 5) / 9.0
        
        self.register_buffer('sobel_x_3x3', sobel_3x3)
        self.register_buffer('sobel_y_3x3', sobel_3x3.transpose(-1, -2).contiguous())
        self.register_buffer('sobel_x_5x5', sobel_5x5)
        self.register_buffer('sobel_y_5x5', sobel_5x5.transpose(-1, -2).contiguous())
        self.register_buffer('laplacian', laplacian)
    
    def detect_edges_sobel(self, x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """Detect edges using Sobel filters."""
        b, c, h, w = x.shape
        
        if c == 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x
        
        pad = kernel_size // 2
        gray_pad = F.pad(gray, (pad, pad, pad, pad), mode='replicate')
        
        if kernel_size == 3:
            sobel_x = self.sobel_x_3x3.to(device=gray_pad.device, dtype=gray_pad.dtype)
            sobel_y = self.sobel_y_3x3.to(device=gray_pad.device, dtype=gray_pad.dtype)
        else:
            sobel_x = self.sobel_x_5x5.to(device=gray_pad.device, dtype=gray_pad.dtype)
            sobel_y = self.sobel_y_5x5.to(device=gray_pad.device, dtype=gray_pad.dtype)
        
        grad_x = F.conv2d(gray_pad, sobel_x)
        grad_y = F.conv2d(gray_pad, sobel_y)
        
        edge_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        edge_mag = edge_mag / (edge_mag.max() + 1e-8)
        
        return edge_mag
    
    def detect_edges_laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """Detect edges using Laplacian kernel."""
        b, c, h, w = x.shape
        
        if c == 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x
        
        gray_pad = F.pad(gray, (2, 2, 2, 2), mode='replicate')
        
        laplacian = self.laplacian.to(device=gray_pad.device, dtype=gray_pad.dtype)
        edge_mag = torch.abs(F.conv2d(gray_pad, laplacian))
        edge_mag = edge_mag / (edge_mag.max() + 1e-8)
        
        return edge_mag
    
    def detect_edges_multi_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Detect edges using multiple scales for better coverage of anime lines."""
        # Sobel 3x3 for thicker lines
        edges_3x3 = self.detect_edges_sobel(x, kernel_size=3)
        
        # Sobel 5x5 for thinner lines (common in anime detail)
        edges_5x5 = self.detect_edges_sobel(x, kernel_size=5)
        
        # Laplacian for very fine lines
        edges_lap = self.detect_edges_laplacian(x)
        
        # Combine: take maximum across scales for union coverage
        # This ensures we capture both thick character outlines and thin details
        combined = torch.maximum(edges_3x3, torch.maximum(edges_5x5, edges_lap))
        
        return combined
    
    def detect_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Detect edges using configured method."""
        if self.edge_method == 'laplacian':
            return self.detect_edges_laplacian(x)
        elif self.edge_method == 'multi_scale':
            return self.detect_edges_multi_scale(x)
        else:  # 'sobel' as default
            return self.detect_edges_sobel(x, kernel_size=3)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        # Detect edges in target
        target_edges = self.detect_edges(target)
        
        # Compute base L1 loss
        l1_loss = F.l1_loss(pred, target, reduction='none')
        
        # Weight by edge importance
        if l1_loss.dim() == 4:
            # Average over channels
            l1_loss = l1_loss.mean(dim=1, keepdim=True)
        
        # Edge regions get higher weight
        edge_mask = (target_edges > 0.1).float()
        weights = 1.0 + (self.edge_weight - 1.0) * edge_mask
        
        weighted_loss = (l1_loss * weights).mean()
        base_loss = l1_loss.mean()
        
        return {
            'line_art': weighted_loss,
            'base_l1': base_loss,
            'edge_coverage': edge_mask.mean().detach(),
        }


class ColorConsistencyLoss(nn.Module):
    """
    Loss that ensures color consistency in flat regions.
    Prevents color bleeding and banding in anime content.
    """
    
    def __init__(self, color_weight: float = 1.0, use_histogram: bool = True):
        super().__init__()
        self.color_weight = color_weight
        self.use_histogram = use_histogram
    
    def compute_histogram_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        bins: int = 32,
    ) -> torch.Tensor:
        """
        Compute histogram-based color consistency loss.
        
        Args:
            pred: Predictions [B, C, H, W]
            target: Targets [B, C, H, W]
            bins: Number of histogram bins
            
        Returns:
            Histogram loss
        """
        b, c, h, w = pred.shape
        
        # Flatten spatial dimensions
        pred_flat = pred.view(b, c, -1)
        target_flat = target.view(b, c, -1)
        
        # Compute histograms per channel
        # Convert to float32 for histc (doesn't support half precision)
        loss = 0.0
        for i in range(c):
            pred_hist = torch.histc(pred_flat[:, i].float(), bins=bins, min=0, max=1)
            target_hist = torch.histc(target_flat[:, i].float(), bins=bins, min=0, max=1)
            
            # Normalize
            pred_hist = pred_hist / (pred_hist.sum() + 1e-8)
            target_hist = target_hist / (target_hist.sum() + 1e-8)
            
            # L1 between histograms
            loss += F.l1_loss(pred_hist, target_hist)
        
        return loss / c
    
    def compute_variance_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        kernel_size: int = 7,
    ) -> torch.Tensor:
        """
        Compute variance-based loss for flat regions.
        Flat regions should have similar variance between pred and target.
        
        Args:
            pred: Predictions [B, C, H, W]
            target: Targets [B, C, H, W]
            kernel_size: Size of local window
            
        Returns:
            Variance loss
        """
        pad = kernel_size // 2
        
        # Compute local variance
        pred_pooled = F.avg_pool2d(
            F.pad(pred, (pad, pad, pad, pad), mode='reflect'),
            kernel_size,
            stride=1,
        )
        target_pooled = F.avg_pool2d(
            F.pad(target, (pad, pad, pad, pad), mode='reflect'),
            kernel_size,
            stride=1,
        )
        
        # Variance difference
        pred_var = (pred - pred_pooled).pow(2).mean(dim=1, keepdim=True)
        target_var = (target - target_pooled).pow(2).mean(dim=1, keepdim=True)
        
        # Loss: variance should match
        loss = F.l1_loss(pred_var, target_var)
        
        return loss
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Compute color consistency loss.
        
        Returns:
            Dict with loss components
        """
        losses = {}
        
        # Variance loss (flat region preservation)
        var_loss = self.compute_variance_loss(pred, target)
        losses['variance'] = var_loss
        
        # Histogram loss (global color distribution)
        if self.use_histogram:
            hist_loss = self.compute_histogram_loss(pred, target)
            losses['histogram'] = hist_loss
            total = var_loss + hist_loss
        else:
            total = var_loss
        
        losses['color_consistency'] = total * self.color_weight
        
        return losses


class FlatRegionPreservationLoss(nn.Module):
    """
    Loss specifically for preserving flat color regions in anime.
    Detects flat regions and enforces smoothness within them.
    """
    
    def __init__(self, flat_threshold: float = 0.05, preserve_weight: float = 1.0):
        super().__init__()
        self.flat_threshold = flat_threshold
        self.preserve_weight = preserve_weight
    
    def detect_flat_regions(self, x: torch.Tensor, window_size: int = 5) -> torch.Tensor:
        """
        Detect flat regions based on local variance.
        
        Args:
            x: Input tensor [B, C, H, W]
            window_size: Size of analysis window
            
        Returns:
            Flat region mask [B, 1, H, W]
        """
        # Convert to grayscale
        if x.shape[1] == 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x
        
        # Compute local variance
        pad = window_size // 2
        gray_padded = F.pad(gray, (pad, pad, pad, pad), mode='reflect')
        
        # Mean in window
        local_mean = F.avg_pool2d(
            gray_padded,
            window_size,
            stride=1,
        )
        
        # Variance in window
        local_var = F.avg_pool2d(
            gray_padded ** 2,
            window_size,
            stride=1,
        ) - local_mean ** 2
        
        # Flat regions have low variance
        flat_mask = (local_var < self.flat_threshold).float()
        
        return flat_mask
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Compute flat region preservation loss.
        
        Returns:
            Dict with losses
        """
        # Detect flat regions in target
        flat_mask = self.detect_flat_regions(target)
        
        # Within flat regions, prediction should be smooth
        pred_var = torch.var(pred, dim=1, keepdim=True)
        
        # Penalize variance in flat regions
        flat_loss = (pred_var * flat_mask).mean()
        
        # Also ensure prediction matches target in flat regions
        l1_in_flat = (F.l1_loss(pred, target, reduction='none').mean(dim=1, keepdim=True) * flat_mask).mean()
        
        total = flat_loss + l1_in_flat
        
        return {
            'flat_preservation': total * self.preserve_weight,
            'flat_coverage': flat_mask.mean().detach(),
            'flat_l1': l1_in_flat.detach(),
        }


class AnimeCombinedLoss(nn.Module):
    """
    Combined anime-specific loss.
    Integrates line art, color consistency, and flat region preservation.
    """
    
    def __init__(
        self,
        line_art_weight: float = 1.0,
        color_weight: float = 0.5,
        flat_weight: float = 0.5,
    ):
        super().__init__()
        
        self.line_art_weight = line_art_weight
        self.color_weight = color_weight
        self.flat_weight = flat_weight
        
        self.line_art_loss = LineArtPreservationLoss()
        self.color_loss = ColorConsistencyLoss()
        self.flat_loss = FlatRegionPreservationLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Compute combined anime loss.
        
        Returns:
            Dict with all losses
        """
        # Line art preservation
        line_dict = self.line_art_loss(pred, target)
        line_loss = line_dict['line_art']
        
        # Color consistency
        color_dict = self.color_loss(pred, target)
        color_loss = color_dict['color_consistency']
        
        # Flat region preservation
        flat_dict = self.flat_loss(pred, target)
        flat_loss = flat_dict['flat_preservation']
        
        # Total
        total = (
            self.line_art_weight * line_loss +
            self.color_weight * color_loss +
            self.flat_weight * flat_loss
        )
        
        return {
            'total': total,
            'line_art': line_loss,
            'color_consistency': color_loss,
            'flat_preservation': flat_loss,
            'details': {
                **line_dict,
                **color_dict,
                **flat_dict,
            }
        }
