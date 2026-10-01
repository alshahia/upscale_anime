"""
Mamba utility functions and fallback implementations
Handles optional mamba-ssm dependency
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Try to import mamba_ssm, provide fallback if not available
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    # Only warn in main process (suppress in DataLoader workers)
    try:
        from torch.utils.data import get_worker_info
        if get_worker_info() is None:
            print("Warning: mamba-ssm not available. Using fallback implementation.")
    except Exception:
        print("Warning: mamba-ssm not available. Using fallback implementation.")


class FallbackMamba(nn.Module):
    """
    Fallback Mamba implementation using standard PyTorch operations.
    Not as efficient as the CUDA-optimized version, but functional.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        # Input projection (x, z, B, C, Δ)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2 + d_state * 2 + self.d_inner, bias=False)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        # A (negative semi-definite) and D (skip) for the selective scan
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, L, C]
        
        Returns:
            Output tensor [B, L, C]
        """
        B, L, C = x.shape
        
        # Input projection
        x_and_z_and_rest = self.in_proj(x)  # [B, L, d_inner*2 + d_state*2 + d_inner]
        
        # Split into components
        x_proj = x_and_z_and_rest[..., :self.d_inner]  # [B, L, d_inner]
        z = x_and_z_and_rest[..., self.d_inner:self.d_inner*2]  # [B, L, d_inner]
        B_param = x_and_z_and_rest[..., self.d_inner*2:self.d_inner*2 + self.d_state]  # [B, L, d_state]
        C_param = x_and_z_and_rest[..., self.d_inner*2 + self.d_state:self.d_inner*2 + self.d_state*2]
        # Convolution
        x_conv = x_proj.transpose(1, 2)  # [B, d_inner, L]
        x_conv = self.conv1d(x_conv)[..., :L]  # [B, d_inner, L]
        x_conv = x_conv.transpose(1, 2)  # [B, L, d_inner]
        x_conv = F.silu(x_conv)
        
        # Selective scan (S6): h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * u_t
        #                      y_t = C_t @ h_t
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        delta = F.softplus(self.dt_proj(x_conv))  # [B, L, d_inner]

        h = x_conv.new_zeros(B, self.d_inner, self.d_state)
        ys = []
        for t in range(L):
            delta_t = delta[:, t].unsqueeze(-1)  # [B, d_inner, 1]
            dA = torch.exp(delta_t * A.unsqueeze(0))  # [B, d_inner, d_state]
            dB_u = delta_t * B_param[:, t].unsqueeze(1) * x_conv[:, t].unsqueeze(-1)
            h = dA * h + dB_u
            y_t = (h * C_param[:, t].unsqueeze(1)).sum(dim=-1)  # [B, d_inner]
            ys.append(y_t)
        x_out = torch.stack(ys, dim=1)  # [B, L, d_inner]

        # Skip connection
        x_out = x_out + x_conv * self.D.view(1, -1)
        
        # Gating
        z = F.silu(z)
        x_out = x_out * z
        
        # Output projection
        output = self.out_proj(x_out)  # [B, L, d_model]
        
        return output


def create_mamba_layer(
    d_model: int,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
) -> nn.Module:
    """
    Create Mamba layer, using optimized version if available.
    
    Args:
        d_model: Model dimension
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        expand: Expansion factor
    
    Returns:
        Mamba layer module
    """
    if MAMBA_AVAILABLE:
        return Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
    else:
        return FallbackMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )


def scan_horizontal(x: torch.Tensor, flip: bool = False) -> torch.Tensor:
    """
    Horizontal scan.
    
    Args:
        x: [B, C, H, W]
        flip: Whether to reverse the scan direction
    
    Returns:
        Scanned tensor [B, C, H, W]
    """
    if flip:
        x = x.flip(-1)
    
    # Reshape to sequence: [B, C, H, W] -> [B*H, W, C]
    B, C, H, W = x.shape
    x_seq = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
    
    return x_seq, (B, H, W)


def scan_vertical(x: torch.Tensor, flip: bool = False) -> torch.Tensor:
    """
    Vertical scan.
    
    Args:
        x: [B, C, H, W]
        flip: Whether to reverse the scan direction
    
    Returns:
        Scanned tensor as sequence [B*W, H, C]
    """
    if flip:
        x = x.flip(-2)
    
    # Reshape to sequence: [B, C, H, W] -> [B*W, H, C]
    B, C, H, W = x.shape
    x_seq = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
    
    return x_seq, (B, H, W)


def reverse_scan_horizontal(x_seq: torch.Tensor, shape: Tuple, flip: bool = False) -> torch.Tensor:
    """Reverse horizontal scan"""
    B, H, W = shape
    # Reshape back: [B*H, W, C] -> [B, C, H, W]
    x = x_seq.reshape(B, H, W, -1).permute(0, 3, 1, 2)
    
    if flip:
        x = x.flip(-1)
    
    return x


def reverse_scan_vertical(x_seq: torch.Tensor, shape: Tuple, flip: bool = False) -> torch.Tensor:
    """Reverse vertical scan"""
    B, H, W = shape
    # Reshape back: [B*W, H, C] -> [B, C, H, W]
    x = x_seq.reshape(B, W, H, -1).permute(0, 3, 2, 1)
    
    if flip:
        x = x.flip(-2)
    
    return x
