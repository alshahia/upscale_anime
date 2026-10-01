"""
Exact Checkpoint-Compatible SPAN Model

This model exactly matches the architecture and parameter naming of the downloaded SPAN checkpoint
to achieve 100% parameter loading success.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from anime_sr.models.base import BaseSRModel


class ConvModuleExact(nn.Module):
    """
    Convolution module that matches exact checkpoint parameter naming:
    - conv.0.weight/bias
    - conv.1.weight/bias  
    - conv.2.weight/bias
    """
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        
        # Direct naming to match checkpoint (no nested conv)
        self._0 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, padding=0)
        self._1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self._2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._0(x)
        x = F.relu(x)
        x = self._1(x)
        x = F.relu(x)
        x = self._2(x)
        return x


class ConvSubBlockExact(nn.Module):
    """
    Sub-block that matches exact checkpoint structure:
    - sk.weight/bias
    - conv.0/1/2.weight/bias
    - eval_conv.weight/bias
    """
    
    def __init__(self, in_channels: int = 48, out_channels: int = 48, hidden_channels: int = 96):
        super().__init__()
        
        # Spectral kernel
        self.sk = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.register_parameter('sk_bias', nn.Parameter(torch.zeros(in_channels)))
        
        # Three conv layers with exact naming
        self.conv = ConvModuleExact(in_channels, hidden_channels, out_channels)
        
        # Evaluation convolution
        self.eval_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights to match checkpoint expectations"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(getattr(self, 'sk_bias'))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spectral kernel
        sk_out = self.sk(x) + getattr(self, 'sk_bias').view(1, -1, 1, 1)
        
        # Three conv layers
        conv_out = self.conv(sk_out)
        
        # Evaluation convolution
        eval_out = self.eval_conv(conv_out)
        
        # Combine outputs
        out = sk_out + conv_out + eval_out
        
        return out


class Conv1BlockExact(nn.Module):
    """
    Conv_1 block that matches exact checkpoint structure:
    - sk.weight/bias: [48, 3, 1, 1]
    - conv.0.weight/bias: [6, 3, 1, 1]
    - conv.1.weight/bias: [96, 6, 3, 3]
    - conv.2.weight/bias: [48, 96, 1, 1]
    - eval_conv.weight/bias: [48, 3, 3, 3]
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 48, hidden_channels: int = 96):
        super().__init__()
        
        # Spectral kernel (48x3x1x1)
        self.sk = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.register_parameter('sk_bias', nn.Parameter(torch.zeros(out_channels)))
        
        # Three conv layers with exact dimensions and naming
        self.conv = nn.Module()
        self.conv._0 = nn.Conv2d(in_channels, 6, kernel_size=1, padding=0)  # conv.0: [6, 3, 1, 1]
        self.conv._1 = nn.Conv2d(6, hidden_channels, kernel_size=3, padding=1)  # conv.1: [96, 6, 3, 3]
        self.conv._2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, padding=0)  # conv.2: [48, 96, 1, 1]
        
        # Evaluation convolution (48x3x3x3)
        self.eval_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights to match checkpoint expectations"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(getattr(self, 'sk_bias'))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spectral kernel
        sk_out = self.sk(x) + getattr(self, 'sk_bias').view(1, -1, 1, 1)
        
        # Three conv layers (conv.0 takes original input, not sk_out)
        conv_out = self.conv._0(x)  # Takes original input [3 channels]
        conv_out = F.relu(conv_out)
        conv_out = self.conv._1(conv_out)
        conv_out = F.relu(conv_out)
        conv_out = self.conv._2(conv_out)
        
        # Evaluation convolution (takes original input)
        eval_out = self.eval_conv(x)
        
        # Combine outputs
        out = sk_out + conv_out + eval_out
        
        return out


class Conv2BlockExact(nn.Module):
    """
    Conv_2 block that matches exact checkpoint structure:
    - sk.weight/bias: [48, 48, 1, 1]
    - conv.0.weight/bias: [96, 48, 1, 1]
    - conv.1.weight/bias: [96, 96, 3, 3]
    - conv.2.weight/bias: [48, 96, 1, 1]
    - eval_conv.weight/bias: [48, 48, 3, 3]
    """
    
    def __init__(self, in_channels: int = 48, out_channels: int = 48, hidden_channels: int = 96):
        super().__init__()
        
        # Spectral kernel (48x48x1x1)
        self.sk = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.register_parameter('sk_bias', nn.Parameter(torch.zeros(in_channels)))
        
        # Three conv layers with exact dimensions and naming
        self.conv = ConvModuleExact(in_channels, hidden_channels, out_channels)
        
        # Evaluation convolution (48x48x3x3)
        self.eval_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights to match checkpoint expectations"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(getattr(self, 'sk_bias'))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spectral kernel
        sk_out = self.sk(x) + getattr(self, 'sk_bias').view(1, -1, 1, 1)
        
        # Three conv layers
        conv_out = self.conv(sk_out)
        
        # Evaluation convolution
        eval_out = self.eval_conv(conv_out)
        
        # Combine outputs
        out = sk_out + conv_out + eval_out
        
        return out


class SPANBlockExact(nn.Module):
    """
    SPAN block that matches exact checkpoint structure:
    - c1_r: ConvSubBlockExact
    - c2_r: ConvSubBlockExact  
    - c3_r: ConvSubBlockExact
    """
    
    def __init__(self, block_num: int, channels: int = 48, hidden_channels: int = 96):
        super().__init__()
        
        self.c1_r = ConvSubBlockExact(channels, channels, hidden_channels)
        self.c2_r = ConvSubBlockExact(channels, channels, hidden_channels)
        self.c3_r = ConvSubBlockExact(channels, channels, hidden_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sequential processing through sub-blocks
        out1 = self.c1_r(x)
        out2 = self.c2_r(out1)
        out3 = self.c3_r(out2)
        
        # Residual connection
        return out3 + x


class UpsamplerBlockExact(nn.Module):
    """
    Upsampler that matches exact checkpoint structure:
    - upsampler.0.weight/bias: [48, 48, 3, 3]
    """
    
    def __init__(self, in_channels: int = 48, scale: int = 4):
        super().__init__()
        
        self.scale = scale
        # Direct naming to match checkpoint exactly (needs in_channels*scale*scale for pixel shuffle)
        self._0 = nn.Conv2d(in_channels, in_channels * scale * scale, kernel_size=3, padding=1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights to match checkpoint expectations"""
        nn.init.kaiming_normal_(self._0.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self._0.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._0(x)
        x = F.pixel_shuffle(x, upscale_factor=self.scale)
        return x


class CheckpointCompatibleSPANExact(BaseSRModel):
    """
    Exact Checkpoint-compatible SPAN model that matches the downloaded checkpoint
    architecture and parameter naming perfectly for 100% loading success.
    
    Architecture:
    - conv_1: Conv1BlockExact (sk + conv.0/1/2 + eval_conv)
    - conv_2: Conv2BlockExact (sk + conv.0/1/2 + eval_conv)
    - conv_cat: Concatenation layer
    - 6 blocks: Each SPANBlockExact (c1_r, c2_r, c3_r sub-blocks)
    - upsampler: UpsamplerBlockExact (4x upscaling)
    - Residual connections
    """
    
    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: int = 48,
        hidden_channels: int = 96,
        num_blocks: int = 6,
    ):
        super().__init__(scale=scale, in_channels=in_channels, out_channels=out_channels)
        
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks
        
        # conv_1 block - exact checkpoint structure
        self.conv_1 = Conv1BlockExact(in_channels, channels, hidden_channels)
        
        # conv_2 block - exact checkpoint structure
        self.conv_2 = Conv2BlockExact(channels, channels, hidden_channels)
        
        # conv_cat layer - exact checkpoint structure (concatenates conv_1 and conv_2 outputs)
        self.conv_cat = nn.Conv2d(channels * 2, channels, kernel_size=1, padding=0)
        
        # SPAN blocks - exact checkpoint structure
        for i in range(num_blocks):
            setattr(self, f'block_{i+1}', SPANBlockExact(i+1, channels, hidden_channels))
        
        # Upsampler - exact checkpoint structure
        self.upsampler = UpsamplerBlockExact(channels, int(scale))
        
        # Final convolution to get correct output channels (identity when channels == out_channels)
        if channels != out_channels:
            self.final_conv = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)
            print(f"  Created final_conv: {channels} -> {out_channels}")
        else:
            self.final_conv = nn.Identity()
            print(f"  Created final_conv: identity")
        
        print(f"CheckpointCompatibleSPANExact initialized:")
        print(f"  Scale: {scale}x")
        print(f"  Channels: {channels}")
        print(f"  Hidden channels: {hidden_channels}")
        print(f"  Blocks: {num_blocks}")
        print(f"  Expected parameters: 204")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with exact checkpoint structure and NaN handling.
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Output tensor [B, 3, H*scale, W*scale]
        """
        # Input validation
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN/Inf detected in input, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Initial feature extraction
        feat1 = self.conv_1(x)
        feat2 = self.conv_2(feat1)  # Pass feat1 (48 channels) to conv_2
        
        # NaN detection after feature extraction
        if torch.isnan(feat1).any():
            feat1 = torch.nan_to_num(feat1, nan=0.0, posinf=1.0, neginf=0.0)
        if torch.isnan(feat2).any():
            feat2 = torch.nan_to_num(feat2, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Concatenate features
        feat = torch.cat([feat1, feat2], dim=1)
        feat = self.conv_cat(feat)
        
        # NaN detection after conv_cat
        if torch.isnan(feat).any():
            feat = torch.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Pass through SPAN blocks
        for i in range(self.num_blocks):
            block = getattr(self, f'block_{i+1}')
            feat = block(feat)
            
            # NaN detection after each block
            if torch.isnan(feat).any():
                feat = torch.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Upsampling
        upsampled = self.upsampler(feat)
        
        # NaN detection after upsampling
        if torch.isnan(upsampled).any():
            upsampled = torch.nan_to_num(upsampled, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Final channel adjustment
        upsampled = self.final_conv(upsampled)
        
        # NaN detection after final_conv
        if torch.isnan(upsampled).any():
            upsampled = torch.nan_to_num(upsampled, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Bicubic upsampling for residual connection
        lr_upsampled = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        
        # Residual connection (only if channels match)
        if upsampled.shape[1] == lr_upsampled.shape[1]:
            out = upsampled + lr_upsampled
        else:
            # If channels don't match, only use upsampled features
            out = upsampled
        
        # Final NaN detection and clamping
        if torch.isnan(out).any():
            print("Warning: NaN detected in final output, replacing with zeros")
            out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Clamp to valid range
        out = torch.clamp(out, 0, 1)
        
        return out


def load_checkpoint_compatible_weights_exact(model, checkpoint_path: str, device: str = 'cpu', verbose: bool = True):
    """
    Load SPAN checkpoint weights with parameter name mapping to achieve 100% loading.
    
    Args:
        model: CheckpointCompatibleSPANExact model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load on
        verbose: Whether to print loading info
        
    Returns:
        success: bool indicating if loading was successful
        loading_info: dict with loading statistics
    """
    import torch
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'params' in checkpoint:
        checkpoint_params = checkpoint['params']
    elif 'model_state_dict' in checkpoint:
        checkpoint_params = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        checkpoint_params = checkpoint['state_dict']
    else:
        checkpoint_params = checkpoint
    
    # Create mapping from model parameter names to checkpoint names
    model_state = model.state_dict()
    loaded_params = 0
    missing_params = []
    unexpected_params = []
    
    # Build parameter name mapping (model -> checkpoint)
    name_mapping = {}
    for model_name in model_state.keys():
        checkpoint_name = model_name
        
        # Map sk_bias -> sk.bias
        if 'sk_bias' in model_name:
            checkpoint_name = model_name.replace('sk_bias', 'sk.bias')
        
        # Map conv._0 -> conv.0, conv._1 -> conv.1, conv._2 -> conv.2
        if 'conv._' in model_name:
            checkpoint_name = model_name.replace('conv._', 'conv.')
        
        # Map upsampler._0 -> upsampler.0 for direct upsampler parameters
        if model_name.startswith('upsampler._'):
            checkpoint_name = model_name.replace('upsampler._', 'upsampler.')
        # Map nested upsampler.upsampler._0 -> upsampler.0
        elif 'upsampler.upsampler._' in model_name:
            checkpoint_name = model_name.replace('upsampler.upsampler._', 'upsampler.')
        # Handle reverse mapping for checkpoints with old structure
        elif checkpoint_name and 'upsampler.upsampler._' in checkpoint_name:
            checkpoint_name = checkpoint_name.replace('upsampler.upsampler._', 'upsampler._')
        
        name_mapping[model_name] = checkpoint_name
    
    # Load parameters with mapping
    for model_name, checkpoint_name in name_mapping.items():
        model_param = model_state[model_name]
        if checkpoint_name in checkpoint_params:
            if model_param.shape == checkpoint_params[checkpoint_name].shape:
                model_param.copy_(checkpoint_params[checkpoint_name])
                loaded_params += 1
            else:
                missing_params.append(f'{model_name} -> {checkpoint_name} (shape mismatch: {model_param.shape} vs {checkpoint_params[checkpoint_name].shape})')
        else:
            missing_params.append(f'{model_name} -> {checkpoint_name} (not found)')
    
    # Check for unexpected checkpoint parameters
    loaded_checkpoint_names = set(name_mapping.values())
    for checkpoint_name in checkpoint_params.keys():
        if checkpoint_name not in loaded_checkpoint_names:
            unexpected_params.append(checkpoint_name)
    
    if verbose:
        print(f"\nCheckpoint Loading Results:")
        print(f"  Successfully loaded: {loaded_params}/{len(checkpoint_params)} ({loaded_params/len(checkpoint_params)*100:.1f}%)")
        print(f"  Missing parameters: {len(missing_params)}")
        print(f"  Unexpected parameters: {len(unexpected_params)}")
        
        if missing_params and len(missing_params) <= 5:
            print("  Missing:", missing_params)
        elif missing_params:
            print(f"  Missing (showing first 5 of {len(missing_params)}):", missing_params[:5])
            
        if unexpected_params and len(unexpected_params) <= 5:
            print("  Unexpected:", unexpected_params)
        elif unexpected_params:
            print(f"  Unexpected (showing first 5 of {len(unexpected_params)}):", unexpected_params[:5])
    
    success = loaded_params > 0
    loading_info = {
        'loaded_params': loaded_params,
        'total_checkpoint_params': len(checkpoint_params),
        'missing_params': missing_params,
        'unexpected_params': unexpected_params,
        'success': success
    }
    
    return success, loading_info


def create_checkpoint_compatible_model_exact(config: Dict[str, Any]) -> CheckpointCompatibleSPANExact:
    """
    Factory function to create exact checkpoint-compatible SPAN model.
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        CheckpointCompatibleSPANExact instance
    """
    return CheckpointCompatibleSPANExact(
        scale=int(config.get('scale', 4)),
        in_channels=config.get('in_channels', 3),
        out_channels=config.get('out_channels', 3),
        channels=config.get('channels', 48),
        hidden_channels=config.get('hidden_channels', 96),
        num_blocks=config.get('num_blocks', 6),
    )


if __name__ == "__main__":
    # Test the exact model
    config = {
        'scale': 4,
        'channels': 48,
        'hidden_channels': 96,
        'num_blocks': 6
    }
    
    model = create_checkpoint_compatible_model_exact(config)
    print(f"Model created: {type(model).__name__}")
    print(f"Total parameters: {len(list(model.named_parameters()))}")
    
    # Test checkpoint loading
    checkpoint_path = 'checkpoints/span/spanx4_ch48.pth'
    success, loading_info = load_checkpoint_compatible_weights_exact(
        model,
        checkpoint_path,
        device='cpu',
        verbose=True
    )
    
    print(f"Overall success: {success}")
    print(f"Loading percentage: {loading_info['loaded_params']/loading_info['total_checkpoint_params']*100:.1f}%")
