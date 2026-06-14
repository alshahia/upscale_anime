#!/usr/bin/env python3
"""
SPAN-F Checkpoint Analysis Tool

This script analyzes the downloaded SPAN-F checkpoint files to:
1. Extract model architecture information
2. Check key compatibility with current implementation
3. Identify configuration issues for fine-tuning
"""

import os
import sys
import torch
import zipfile
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def analyze_checkpoint_structure(checkpoint_path):
    """Analyze PyTorch checkpoint structure and keys"""
    print(f"\n=== Analyzing {os.path.basename(checkpoint_path)} ===")
    
    try:
        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except Exception:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Check checkpoint structure
        if isinstance(checkpoint, dict):
            print(f"Checkpoint type: dict with {len(checkpoint)} keys")
            
            # Look for common keys
            common_keys = ['state_dict', 'model', 'params', 'epoch', 'optimizer_state_dict']
            found_keys = []
            for key in common_keys:
                if key in checkpoint:
                    found_keys.append(key)
                    print(f"  Found key: {key}")
            
            # Analyze state_dict if present
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"  State dict has {len(state_dict)} parameters")
                
                # Analyze parameter names and shapes
                param_analysis = analyze_parameter_names(state_dict)
                return param_analysis
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                print(f"  Model dict has {len(state_dict)} parameters")
                param_analysis = analyze_parameter_names(state_dict)
                return param_analysis
            else:
                # Assume checkpoint itself is the state dict
                print(f"  Checkpoint appears to be raw state dict with {len(checkpoint)} parameters")
                param_analysis = analyze_parameter_names(checkpoint)
                return param_analysis
                
        else:
            print(f"Checkpoint type: {type(checkpoint)}")
            return None
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def analyze_parameter_names(state_dict):
    """Analyze parameter names to infer architecture"""
    print("\n  Parameter Analysis:")
    
    # Collect parameter info
    param_info = {}
    for key, tensor in state_dict.items():
        param_info[key] = {
            'shape': list(tensor.shape),
            'num_params': tensor.numel()
        }
    
    # Look for architecture indicators
    channels = None
    num_blocks = 0
    has_lora = False
    
    # Find channels from first conv
    for key, info in param_info.items():
        if 'shallow_conv' in key and 'weight' in key:
            if len(info['shape']) == 4:
                channels = info['shape'][0]
                print(f"    Detected channels: {channels}")
                break
    
    # Count blocks
    block_keys = [k for k in param_info.keys() if 'blocks' in k]
    if block_keys:
        # Extract block numbers
        block_numbers = set()
        for key in block_keys:
            parts = key.split('.')
            for part in parts:
                if part.isdigit():
                    block_numbers.add(int(part))
        
        num_blocks = max(block_numbers) + 1 if block_numbers else 0
        print(f"    Detected blocks: {num_blocks}")
    
    # Check for LoRA
    lora_keys = [k for k in param_info.keys() if 'lora' in k.lower()]
    if lora_keys:
        has_lora = True
        print(f"    Has LoRA: {len(lora_keys)} LoRA parameters")
    
    # Check for attention mechanisms
    attention_keys = [k for k in param_info.keys() if 'attention' in k.lower() or 'attn' in k.lower()]
    if attention_keys:
        print(f"    Attention mechanisms: {len(set(k.split('.')[0] for k in attention_keys))} modules")
    
    return {
        'channels': channels,
        'num_blocks': num_blocks,
        'has_lora': has_lora,
        'param_count': sum(info['num_params'] for info in param_info.values()),
        'key_structure': list(param_info.keys())[:10]  # First 10 keys for reference
    }

def check_compatibility(analysis, expected_config):
    """Check compatibility between checkpoint and expected config"""
    print("\n  Compatibility Check:")
    
    issues = []
    
    if analysis['channels'] != expected_config['channels']:
        issues.append(f"Channel mismatch: expected {expected_config['channels']}, got {analysis['channels']}")
    
    if analysis['num_blocks'] != expected_config['num_blocks']:
        issues.append(f"Block count mismatch: expected {expected_config['num_blocks']}, got {analysis['num_blocks']}")
    
    if expected_config.get('use_lora', False) != analysis['has_lora']:
        issues.append(f"LoRA mismatch: expected {expected_config.get('use_lora')}, got {analysis['has_lora']}")
    
    if issues:
        print("    [ERROR] Issues found:")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print("    [OK] No compatibility issues detected")
    
    return issues

def main():
    """Main analysis function"""
    checkpoint_dir = Path("checkpoints/span")
    
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # Find all .pth files
    checkpoints = list(checkpoint_dir.glob("*.pth"))
    
    if not checkpoints:
        print("No .pth files found in checkpoint directory")
        return
    
    print(f"Found {len(checkpoints)} checkpoint files")
    
    # Expected SPAN-F configuration based on NTIRE 2025 paper
    expected_config = {
        'channels': 32,
        'num_blocks': 10,
        'use_lora': True
    }
    
    print(f"\nExpected SPAN-F Configuration:")
    print(f"  Channels: {expected_config['channels']}")
    print(f"  Blocks: {expected_config['num_blocks']}")
    print(f"  LoRA: {expected_config['use_lora']}")
    
    # Analyze each checkpoint
    analyses = {}
    for checkpoint_path in sorted(checkpoints):
        analysis = analyze_checkpoint_structure(checkpoint_path)
        if analysis:
            analyses[checkpoint_path.name] = analysis
            check_compatibility(analysis, expected_config)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, analysis in analyses.items():
        print(f"\n{name}:")
        print(f"  Channels: {analysis['channels']}")
        print(f"  Blocks: {analysis['num_blocks']}")
        print(f"  LoRA: {analysis['has_lora']}")
        print(f"  Parameters: {analysis['param_count']:,}")
        
        # Check if this matches SPAN-F
        is_spanf = (analysis['channels'] == 32 and 
                   analysis['num_blocks'] == 10)
        print(f"  SPAN-F Compatible: {'[OK]' if is_spanf else '[ERROR]'}")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR FINE-TUNING")
    print("="*60)
    
    compatible_checkpoints = [name for name, analysis in analyses.items()
                            if analysis['channels'] == 32 and analysis['num_blocks'] == 10]
    
    if compatible_checkpoints:
        print(f"\n[OK] Compatible SPAN-F checkpoints found:")
        for name in compatible_checkpoints:
            print(f"  - {name}")
        
        print(f"\nRecommended configuration for fine-tuning:")
        print(f"  Use: {compatible_checkpoints[0]}")
        print(f"  Update config file: pretrained_checkpoint: checkpoints/span/{compatible_checkpoints[0]}")
        print(f"  Verify model architecture matches before training")
        
        # Check if current config path is correct
        current_config_path = "pretrained/SPANF_x4/spanx4_ch48.pth"
        actual_path = f"checkpoints/span/{compatible_checkpoints[0]}"
        
        if current_config_path != actual_path:
            print(f"\n[WARN]  Config path mismatch:")
            print(f"  Current in config: {current_config_path}")
            print(f"  Should be: {actual_path}")
            print(f"  Update configs/finetune_spanf.yaml line 13")
    else:
        print(f"\n[ERROR] No SPAN-F compatible checkpoints found")
        print(f"  Expected: 32 channels, 10 blocks")
        print(f"  Found checkpoints have different architectures")
        print(f"  Consider using SPAN-Tiny configuration or different checkpoints")

if __name__ == "__main__":
    main()
