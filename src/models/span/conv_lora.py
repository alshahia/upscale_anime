"""
ConvLoRA: Low-Rank Adaptation for Convolutional Layers
Train with LoRA, merge for inference (zero overhead)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvLoRA(nn.Module):
    """
    Convolutional layer with LoRA (Low-Rank Adaptation).
    
    During training:
        y = W * x + b + (alpha/rank) * B * A * x
    
    After merging:
        y = W_merged * x + b
        where W_merged = W + (alpha/rank) * B * A
    
    This allows efficient fine-tuning without inference overhead.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        bias: bool = True,
        r: int = 4,  # LoRA rank
        lora_alpha: float = 1.0,
    ):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # Main convolution (pretrained, frozen during LoRA training)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        # Freeze main conv during LoRA training
        self.conv.weight.requires_grad_(False)
        if bias:
            self.conv.bias.requires_grad_(False)
        
        # LoRA parameters (trainable)
        # For conv: A is [r, in_channels, 1, 1], B is [out_channels, r, k, k]
        self.lora_A = nn.Parameter(torch.zeros(r, in_channels, 1, 1))
        self.lora_B = nn.Parameter(torch.zeros(out_channels, r, kernel_size, kernel_size))
        
        # Initialize A with normal, B with zeros (so initial LoRA contribution is zero)
        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.zeros_(self.lora_B)
        
        # Freeze main conv during LoRA training
        self.lora_enabled = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [B, in_channels, H, W]
        
        Returns:
            [B, out_channels, H', W']
        """
        # Main convolution
        out = self.conv(x)
        
        # LoRA branch (if enabled)
        if self.lora_enabled:
            # LoRA: B @ A @ x
            # First: A @ x -> [B, r, H, W]
            tmp = F.conv2d(x, self.lora_A, stride=1, padding=0)
            # Then: B @ tmp -> [B, out_channels, H, W]
            lora_out = F.conv2d(tmp, self.lora_B, stride=self.stride, padding=self.padding)
            
            out = out + self.scaling * lora_out
        
        return out
    
    def merge_lora(self):
        """
        Merge LoRA weights into main convolution.
        After merging, inference has zero LoRA overhead.
        """
        if not self.lora_enabled:
            return
        
        # Store original weights so unmerge_lora() can restore them.
        # persistent=False keeps these out of the state_dict.
        self.register_buffer(
            '_original_weight', self.conv.weight.data.clone(), persistent=False
        )
        if self.conv.bias is not None:
            self.register_buffer(
                '_original_bias', self.conv.bias.data.clone(), persistent=False
            )
        
        # Compute effective weight: W + scaling * B @ A
        # lora_A: [r, in, 1, 1], lora_B: [out, r, k, k]
        # Effective LoRA weight: [out, in, k, k]
        A_matrix = self.lora_A.squeeze(-1).squeeze(-1)  # [r, in]
        B_matrix = self.lora_B.reshape(self.out_channels, self.r, -1)  # [out, r, k*k]
        lora_weight = torch.einsum('ork,ri->oik', B_matrix, A_matrix)
        lora_weight = lora_weight.reshape(
            self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        )

        # Merge into main conv weight
        self.conv.weight.data += self.scaling * lora_weight
        
        # Disable LoRA
        self.lora_enabled = False
        
        # Clear LoRA parameters (optional, saves memory)
        self.lora_A.requires_grad = False
        self.lora_B.requires_grad = False
    
    def unmerge_lora(self):
        """Unmerge LoRA weights (restore original conv weights)"""
        if self.lora_enabled:
            return
        if hasattr(self, '_original_weight'):
            self.conv.weight.data.copy_(self._original_weight)
        if hasattr(self, '_original_bias') and self.conv.bias is not None:
            self.conv.bias.data.copy_(self._original_bias)
        self.lora_enabled = True


class ConvLoRA3x3(ConvLoRA):
    """Convenience class for 3x3 ConvLoRA"""
    
    def __init__(self, in_channels: int, out_channels: int, r: int = 4, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size=3, r=r, **kwargs)


class ConvLoRA1x1(ConvLoRA):
    """Convenience class for 1x1 ConvLoRA"""
    
    def __init__(self, in_channels: int, out_channels: int, r: int = 4, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size=1, padding=0, r=r, **kwargs)
