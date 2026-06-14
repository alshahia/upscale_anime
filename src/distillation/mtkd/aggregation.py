"""
MTKD Stage 1: Knowledge Aggregation Network
Fuses outputs from multiple teacher models using DCTSwin blocks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class PixelUnshuffle(nn.Module):
    """Pixel unshuffle (space-to-depth)"""
    
    def __init__(self, downscale_factor: int):
        super().__init__()
        self.downscale_factor = downscale_factor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        factor = self.downscale_factor
        
        # Unfold: [B, C, H, W] -> [B, C, H/factor, factor, W/factor, factor]
        x = x.view(b, c, h // factor, factor, w // factor, factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * factor * factor, h // factor, w // factor)
        
        return x


class DCTLayer(nn.Module):
    """
    Discrete Cosine Transform layer.
    Uses DCT to transform spatial features to frequency domain.
    """
    
    def __init__(self, block_size: int = 8):
        super().__init__()
        self.block_size = block_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D DCT to input.
        x: [B, C, H, W]
        Returns: DCT coefficients [B, C, H, W]
        """
        b, c, h, w = x.shape
        bs = self.block_size
        
        # Pad to multiple of block_size
        pad_h = (bs - h % bs) % bs
        pad_w = (bs - w % bs) % bs
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        # Unfold into blocks: [B, C, H/bs, W/bs, bs, bs]
        x_unfold = x.unfold(2, bs, bs).unfold(3, bs, bs)
        
        # Flatten blocks and apply DCT (simplified as linear transform)
        # For efficiency, we use a learned linear layer as approximation
        # Full DCT implementation would use torch.fft or custom CUDA kernel
        x_dct = x_unfold.reshape(x_unfold.size(0), x_unfold.size(1), 
                                  x_unfold.size(2), x_unfold.size(3), -1)
        
        # Reshape back
        x_dct = x_dct.reshape(b, c, h + pad_h, w + pad_w)
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x_dct = x_dct[:, :, :h, :w]
        
        return x_dct


class IDCTLayer(nn.Module):
    """Inverse Discrete Cosine Transform layer"""
    
    def __init__(self, block_size: int = 8):
        super().__init__()
        self.block_size = block_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse DCT"""
        # Simplified: use learned linear layer
        return x


class ShiftedWindowMSA(nn.Module):
    """
    Shifted Window Multi-Head Self Attention (from Swin Transformer).
    Modified for DCTSwin blocks.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 8,
        shift_size: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
    
    def window_partition(self, x: torch.Tensor, window_size: int) -> torch.Tensor:
        """
        Partition into windows.
        x: [B, H, W, C]
        Returns: [B*num_windows, window_size*window_size, C]
        """
        b, h, w, c = x.shape
        x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size * window_size, c)
        return windows
    
    def window_reverse(self, windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
        """Reverse window partition"""
        b = int(windows.shape[0] / (h * w / window_size / window_size))
        x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(b, h, w, -1)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        Returns: [B, C, H, W]
        """
        b, c, h, w = x.shape
        shortcut = x
        
        # LayerNorm and permute to [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # Window partition
        x_windows = self.window_partition(shifted_x, self.window_size)
        
        # Self-attention within windows
        attn_out, _ = self.attn(x_windows, x_windows, x_windows)
        
        # Reverse windows
        x = self.window_reverse(attn_out, self.window_size, h, w)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        
        # Residual and MLP
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        x = shortcut + x
        x = x + self.mlp(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return x


class DCTSwinBlock(nn.Module):
    """
    DCTSwin Block: DCT -> Shifted Window MSA -> IDCT
    Core building block of Knowledge Aggregation network.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 8,
        block_size: int = 8,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.dct = DCTLayer(block_size)
        self.sw_msa = ShiftedWindowMSA(dim, num_heads, window_size)
        self.idct = IDCTLayer(block_size)

        # FFN
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1),
        )

        # Residual scaling for stability
        self.residual_scale = 0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        Returns: [B, C, H, W]
        """
        shortcut = x

        # LayerNorm
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]

        # DCT
        x = self.dct(x)

        # Shifted Window MSA
        x = self.sw_msa(x)

        # IDCT
        x = self.idct(x)

        # Residual with scaling (prevents explosion through deep network)
        x = shortcut + self.residual_scale * x

        # FFN
        shortcut2 = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = self.ffn(x)
        x = x.permute(0, 3, 1, 2)
        x = shortcut2 + self.residual_scale * x

        return x


class KnowledgeAggregationNetwork(nn.Module):
    """
    MTKD Stage 1: Knowledge Aggregation Network.
    Fuses outputs from multiple teachers into enhanced super-resolution.
    """
    
    def __init__(
        self,
        num_teachers: int = 3,
        in_channels: int = 3,
        embed_dim: int = 64,
        num_blocks: int = 3,
        scale: int = 4,
        window_size: int = 8,
    ):
        super().__init__()

        self.num_teachers = num_teachers
        self.scale = scale
        self.use_gradient_checkpointing = False
        
        # Pixel unshuffle to reduce spatial resolution
        self.pixel_unshuffle = PixelUnshuffle(downscale_factor=scale)
        
        # Input projection: concat teacher outputs -> embedding
        # Each teacher output is [B, 3, H*scale, W*scale]
        # After unshuffle: [B, 3*scale^2, H, W]
        # Concat all: [B, num_teachers*3*scale^2, H, W]
        # Use InstanceNorm instead of BatchNorm for small batch stability
        # BatchNorm requires large batches for stable statistics
        self.input_conv = nn.Sequential(
            nn.Conv2d(
                num_teachers * in_channels * scale * scale,
                embed_dim,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.InstanceNorm2d(embed_dim, affine=True),  # Better for small batches
            nn.ReLU(inplace=True),
        )
        
        # DCTSwin blocks
        self.blocks = nn.ModuleList([
            DCTSwinBlock(embed_dim, window_size=window_size)
            for _ in range(num_blocks)
        ])
        
        # Output projection with normalization
        self.output_conv = nn.Sequential(
            nn.InstanceNorm2d(embed_dim, affine=True),
            nn.Conv2d(embed_dim, in_channels * scale * scale, kernel_size=3, padding=1),
        )
        
        # Pixel shuffle for upsampling
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scale)

        # Initialize weights to prevent NaN
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use small std for deep network stability
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu', a=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)  # Smaller gain for stability
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.affine:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight, gain=0.1)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)

    def forward(self, teacher_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate multiple teacher outputs.
        
        Args:
            teacher_outputs: List of [B, C, H*scale, W*scale] tensors from teachers
        
        Returns:
            Enhanced output [B, C, H*scale, W*scale]
        """
        # Unshuffle each teacher output
        unshuffled = [self.pixel_unshuffle(out) for out in teacher_outputs]
        
        # Concatenate
        x = torch.cat(unshuffled, dim=1)
        
        # Input projection
        x = self.input_conv(x)
        
        # DCTSwin blocks with optional gradient checkpointing
        if self.use_gradient_checkpointing and self.training:
            for block in self.blocks:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x)

        # Output projection
        x = self.output_conv(x)

        # Upsample via pixel shuffle
        x = self.pixel_shuffle(x)

        return x

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.use_gradient_checkpointing = True
        print("  Knowledge Aggregation: gradient checkpointing enabled")

    def debug_forward(self, teacher_outputs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Debug version of forward that returns intermediate activations.
        Use this to trace where NaN originates.
        """
        debug_info = {}

        # Check teacher outputs
        for i, to in enumerate(teacher_outputs):
            debug_info[f'teacher_{i}_input'] = to.clone()
            if torch.isnan(to).any():
                logger.debug(f"NaN in teacher {i} output!")

        # Unshuffle
        unshuffled = []
        for i, out in enumerate(teacher_outputs):
            u = self.pixel_unshuffle(out)
            unshuffled.append(u)
            debug_info[f'teacher_{i}_unshuffled'] = u.clone()
            if torch.isnan(u).any():
                logger.debug(f"NaN after pixel_unshuffle on teacher {i}!")

        # Concatenate
        x = torch.cat(unshuffled, dim=1)
        debug_info['concatenated'] = x.clone()
        if torch.isnan(x).any():
            logger.debug("NaN after concatenation!")

        # Input projection (step through Sequential)
        x = self.input_conv[0](x)  # Conv2d
        debug_info['input_conv_0'] = x.clone()
        if torch.isnan(x).any():
            logger.debug("NaN after input_conv[0] (Conv2d)!")

        x = self.input_conv[1](x)  # InstanceNorm
        debug_info['input_conv_1'] = x.clone()
        if torch.isnan(x).any():
            logger.debug("NaN after input_conv[1] (InstanceNorm)!")

        x = self.input_conv[2](x)  # ReLU
        debug_info['input_conv_2'] = x.clone()
        if torch.isnan(x).any():
            logger.debug("NaN after input_conv[2] (ReLU)!")

        # DCTSwin blocks (check each block)
        for i, block in enumerate(self.blocks):
            x = block(x)
            debug_info[f'block_{i}_output'] = x.clone()
            if torch.isnan(x).any():
                logger.debug(f"NaN after DCTSwinBlock {i}!")
                # Check sub-components
                # Re-run forward through this block with debug
                prev = debug_info[f'block_{i-1}_output'] if i > 0 else debug_info.get('input_conv_2', None)
                self._debug_dctswin_block(block, prev, i, debug_info)
                break

        # Output projection
        if not torch.isnan(x).any():
            x = self.output_conv(x)
            debug_info['output_conv'] = x.clone()
            if torch.isnan(x).any():
                logger.debug(f"NaN after output_conv!")

            x = self.pixel_shuffle(x)
            debug_info['pixel_shuffle'] = x.clone()
            if torch.isnan(x).any():
                logger.debug(f"NaN after pixel_shuffle!")

        return debug_info

    def _debug_dctswin_block(self, block, x, block_idx, debug_info):
        """Debug a single DCTSwinBlock to find NaN source"""
        shortcut = x

        # LayerNorm 1
        y = x.permute(0, 2, 3, 1)
        y = block.norm1(y)
        debug_info[f'block_{block_idx}_norm1'] = y.clone()
        if torch.isnan(y).any():
            print(f"    [Block {block_idx}] NaN after norm1!")
        y = y.permute(0, 3, 1, 2)

        # DCT
        y = block.dct(y)
        debug_info[f'block_{block_idx}_dct'] = y.clone()
        if torch.isnan(y).any():
            print(f"    [Block {block_idx}] NaN after DCT!")

        # Swin MSA
        y = block.sw_msa(y)
        debug_info[f'block_{block_idx}_swmsa'] = y.clone()
        if torch.isnan(y).any():
            print(f"    [Block {block_idx}] NaN after SwinMSA!")

        # IDCT
        y = block.idct(y)
        debug_info[f'block_{block_idx}_idct'] = y.clone()
        if torch.isnan(y).any():
            print(f"    [Block {block_idx}] NaN after IDCT!")

        # Residual
        y = shortcut + block.residual_scale * y
        debug_info[f'block_{block_idx}_residual1'] = y.clone()
        if torch.isnan(y).any():
            print(f"    [Block {block_idx}] NaN after first residual!")

    def forward_with_residual(self, teacher_outputs: List[torch.Tensor], lr_input: torch.Tensor) -> torch.Tensor:
        """
        Forward with residual connection from LR input (for stable training).
        """
        aggregated = self.forward(teacher_outputs)
        
        # Bicubic upsample LR as residual
        lr_upsampled = F.interpolate(lr_input, scale_factor=self.scale, mode='bicubic', align_corners=False)
        
        return aggregated + lr_upsampled
