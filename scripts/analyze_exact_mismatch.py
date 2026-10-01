#!/usr/bin/env python3
"""
Analyze the exact parameter naming mismatch between checkpoint and model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

import torch

def analyze_exact_mismatch():
    """Analyze the exact parameter naming mismatch."""
    print("ANALYZING EXACT PARAMETER MISMATCH")
    print("=" * 50)
    
    # Load checkpoint
    try:
        checkpoint = torch.load('checkpoints/span/spanx4_ch48.pth', map_location='cpu', weights_only=True)
    except Exception:
        checkpoint = torch.load('checkpoints/span/spanx4_ch48.pth', map_location='cpu', weights_only=False)
    checkpoint_params = checkpoint['params']
    
    # Create model
    from anime_sr.models.span.checkpoint_compatible_model_v3 import CheckpointCompatibleSPANv3
    model = CheckpointCompatibleSPANv3()
    model_params = model.state_dict()
    
    print(f"Checkpoint parameters: {len(checkpoint_params)}")
    print(f"Model parameters: {len(model_params)}")
    
    # Show exact parameter names
    print("\nCHECKPOINT PARAMETER NAMES (first 20):")
    for i, name in enumerate(list(checkpoint_params.keys())[:20]):
        print(f"  {i+1:2d}. {name}")
    
    print("\nMODEL PARAMETER NAMES (first 20):")
    for i, name in enumerate(list(model_params.keys())[:20]):
        print(f"  {i+1:2d}. {name}")
    
    # Find patterns
    print("\nPATTERN ANALYSIS:")
    print("Checkpoint patterns:")
    for name in list(checkpoint_params.keys())[:10]:
        if 'block_' in name:
            parts = name.split('.')
            print(f"  {name} -> block_{parts[0].split('_')[1]}, {parts[1]}, {parts[2]}")
        elif 'conv_1' in name:
            parts = name.split('.')
            print(f"  {name} -> conv_1, {parts[1]}")
    
    print("\nModel patterns:")
    for name in list(model_params.keys())[:10]:
        if 'blocks.' in name:
            parts = name.split('.')
            print(f"  {name} -> blocks.{parts[1]}, {parts[2]}, {parts[3]}")
        elif 'conv_1' in name:
            parts = name.split('.')
            print(f"  {name} -> conv_1, {parts[1]}")
    
    # Create the correct mapping
    print("\nCREATING CORRECT MAPPING:")
    mapping = {}
    
    # Map conv_1 parameters
    conv_1_mapping = {
        'conv_1.sk.weight': 'conv_1.sk.weight',
        'conv_1.sk.bias': 'conv_1.sk_bias',
        'conv_1.conv.0.weight': 'conv_1.conv_0.weight',
        'conv_1.conv.0.bias': 'conv_1.conv_0.bias',
        'conv_1.conv.1.weight': 'conv_1.conv_1.weight',
        'conv_1.conv.1.bias': 'conv_1.conv_1.bias',
        'conv_1.conv.2.weight': 'conv_1.conv_2.weight',
        'conv_1.conv.2.bias': 'conv_1.conv_2.bias',
        'conv_1.eval_conv.weight': 'conv_1.eval_conv.weight',
        'conv_1.eval_conv.bias': 'conv_1.eval_conv.bias',
    }
    
    # Map block parameters
    for block_idx in range(1, 7):  # blocks 1-6
        for sub_block in ['c1_r', 'c2_r', 'c3_r']:
            for param in ['sk.weight', 'sk.bias', 'conv.0.weight', 'conv.0.bias', 
                         'conv.1.weight', 'conv.1.bias', 'conv.2.weight', 'conv.2.bias',
                         'eval_conv.weight', 'eval_conv.bias']:
                ckpt_name = f'block_{block_idx}.{sub_block}.{param}'
                model_name = f'blocks.{block_idx-1}.{sub_block}.{param.replace(".", "_")}'
                mapping[ckpt_name] = model_name
    
    # Map upsampler parameters
    mapping['upsampler.0.weight'] = 'upsampler.conv.weight'
    mapping['upsampler.0.bias'] = 'upsampler.conv.bias'
    
    # Test the mapping
    print(f"\nMapping has {len(mapping)} entries")
    
    # Apply mapping
    mapped_params = {}
    successful_mappings = 0
    
    for ckpt_name, ckpt_param in checkpoint_params.items():
        if ckpt_name in mapping:
            model_name = mapping[ckpt_name]
            if model_name in model_params:
                if ckpt_param.shape == model_params[model_name].shape:
                    mapped_params[model_name] = ckpt_param
                    successful_mappings += 1
                else:
                    print(f"Shape mismatch: {ckpt_name} {ckpt_param.shape} vs {model_params[model_name].shape}")
    
    print(f"Successful mappings: {successful_mappings}")
    print(f"Success rate: {successful_mappings / len(checkpoint_params) * 100:.1f}%")
    
    return mapping, successful_mappings / len(checkpoint_params)

if __name__ == "__main__":
    mapping, success_rate = analyze_exact_mismatch()
    print(f"\nFinal success rate: {success_rate:.1f}%")
