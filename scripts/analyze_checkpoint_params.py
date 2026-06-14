#!/usr/bin/env python3
"""
Analyze checkpoint parameter details to understand exact structure
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def analyze_checkpoint_params():
    """Analyze exact parameter counts and shapes"""
    checkpoint_path = "checkpoints/span/spanx4_ch48.pth"
    
    print("CHECKPOINT PARAMETER ANALYSIS")
    print("="*50)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    params = checkpoint['params']
    
    print(f"Total parameters: {len(params)}")
    
    # Analyze by group
    conv_1_params = {k: v for k, v in params.items() if 'conv_1' in k}
    block_params = {k: v for k, v in params.items() if 'block_' in k}
    upsample_params = {k: v for k, v in params.items() if 'upsampler' in k}
    
    print(f"\nconv_1 parameters: {len(conv_1_params)}")
    total_conv_1 = sum(v.numel() for v in conv_1_params.values())
    print(f"conv_1 total elements: {total_conv_1:,}")
    
    print(f"\nblock parameters: {len(block_params)}")
    total_blocks = sum(v.numel() for v in block_params.values())
    print(f"blocks total elements: {total_blocks:,}")
    
    print(f"\nupsample parameters: {len(upsample_params)}")
    total_upsample = sum(v.numel() for v in upsample_params.values())
    print(f"upsample total elements: {total_upsample:,}")
    
    total_elements = total_conv_1 + total_blocks + total_upsample
    print(f"\nTotal elements: {total_elements:,}")
    
    # Show detailed shapes for conv_1
    print(f"\nconv_1 detailed shapes:")
    for name, param in conv_1_params.items():
        print(f"  {name}: {param.shape} = {param.numel():,}")
    
    # Show one block as example
    print(f"\nblock_1 detailed shapes:")
    block_1_params = {k: v for k, v in params.items() if 'block_1.' in k}
    for name, param in block_1_params.items():
        print(f"  {name}: {param.shape} = {param.numel():,}")
    
    # Show upsampler
    print(f"\nupsampler detailed shapes:")
    for name, param in upsample_params.items():
        print(f"  {name}: {param.shape} = {param.numel():,}")

if __name__ == "__main__":
    analyze_checkpoint_params()
