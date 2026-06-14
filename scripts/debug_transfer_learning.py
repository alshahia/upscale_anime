#!/usr/bin/env python3
"""
Transfer Learning Debug Tool

This script diagnoses transfer learning issues by:
1. Checking checkpoint compatibility
2. Analyzing model architecture mismatches
3. Identifying gradient/NaN sources
4. Validating teacher model loading
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def load_config(config_path):
    """Load training configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def analyze_checkpoint_loading(config):
    """Analyze checkpoint loading issues"""
    print("="*60)
    print("CHECKPOINT LOADING ANALYSIS")
    print("="*60)
    
    checkpoint_path = config['training']['pretrained_checkpoint']
    print(f"Checkpoint path: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint file not found: {checkpoint_path}")
        return False
    
    try:
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except Exception:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"[OK] Checkpoint loaded successfully")
        
        # Check structure
        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            # Find state dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"Using 'state_dict' key with {len(state_dict)} parameters")
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                print(f"Using 'model' key with {len(state_dict)} parameters")
            else:
                state_dict = checkpoint
                print(f"Using checkpoint as state_dict with {len(state_dict)} parameters")
            
            # Check if state_dict is the actual parameters or a container
            if 'params' in state_dict:
                actual_params = state_dict['params']
                print(f"Using 'params' key with {len(actual_params)} parameters")
            elif 'params_ema' in state_dict:
                actual_params = state_dict['params_ema']
                print(f"Using 'params_ema' key with {len(actual_params)} parameters")
            else:
                actual_params = state_dict
                print(f"Using checkpoint as direct parameter dict with {len(actual_params)} parameters")
            
            # Analyze parameter names
            param_names = list(actual_params.keys())
            print(f"\nFirst 10 parameter names:")
            for i, name in enumerate(param_names[:10]):
                param = actual_params[name]
                if hasattr(param, 'shape'):
                    shape = param.shape
                    print(f"  {i+1}. {name}: {shape}")
                else:
                    print(f"  {i+1}. {name}: {type(param)}")
            
            # Look for architecture indicators
            shallow_conv_keys = [k for k in param_names if 'shallow_conv' in k]
            block_keys = [k for k in param_names if 'blocks' in k]
            
            print(f"\nArchitecture indicators:")
            print(f"  Shallow conv keys: {len(shallow_conv_keys)}")
            print(f"  Block keys: {len(block_keys)}")
            
            if shallow_conv_keys:
                # Extract channels from shallow conv
                for key in shallow_conv_keys:
                    if 'weight' in key:
                        param = actual_params[key]
                        if hasattr(param, 'shape'):
                            channels = param.shape[0]
                            print(f"  Detected channels: {channels}")
                            break
            
            # Count blocks
            if block_keys:
                block_numbers = set()
                for key in block_keys:
                    parts = key.split('.')
                    for part in parts:
                        if part.isdigit():
                            block_numbers.add(int(part))
                if block_numbers:
                    num_blocks = max(block_numbers) + 1
                    print(f"  Detected blocks: {num_blocks}")
            
            return True, actual_params
        else:
            print(f"[ERROR] Unexpected checkpoint format: {type(checkpoint)}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error loading checkpoint: {e}")
        return False

def analyze_model_creation(config):
    """Analyze model creation and architecture"""
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE ANALYSIS")
    print("="*60)
    
    try:
        from models.span import create_span_model
        
        # Create model as configured
        model_config = config['model']
        print(f"Creating model with config:")
        for key, value in model_config.items():
            print(f"  {key}: {value}")
        
        model = create_span_model(model_config)
        print(f"[OK] Model created successfully")
        
        # Get model state dict
        model_state_dict = model.state_dict()
        print(f"Model parameters: {len(model_state_dict)}")
        
        # Show first few parameter names
        print(f"\nFirst 10 model parameter names:")
        for i, name in enumerate(list(model_state_dict.keys())[:10]):
            shape = model_state_dict[name].shape
            print(f"  {i+1}. {name}: {shape}")
        
        return True, model_state_dict
        
    except Exception as e:
        print(f"[ERROR] Error creating model: {e}")
        return False, None

def check_compatibility(checkpoint_state_dict, model_state_dict):
    """Check compatibility between checkpoint and model"""
    print("\n" + "="*60)
    print("COMPATIBILITY ANALYSIS")
    print("="*60)
    
    if not checkpoint_state_dict or not model_state_dict:
        print("[ERROR] Cannot analyze compatibility - missing state dicts")
        return
    
    checkpoint_keys = set(checkpoint_state_dict.keys())
    model_keys = set(model_state_dict.keys())
    
    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys
    matching_keys = checkpoint_keys & model_keys
    
    print(f"Checkpoint parameters: {len(checkpoint_keys)}")
    print(f"Model parameters: {len(model_keys)}")
    print(f"Matching keys: {len(matching_keys)}")
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    
    # Show some examples
    if missing_keys:
        print(f"\nMissing keys (first 10):")
        for i, key in enumerate(list(missing_keys)[:10]):
            print(f"  - {key}")
    
    if unexpected_keys:
        print(f"\nUnexpected keys (first 10):")
        for i, key in enumerate(list(unexpected_keys)[:10]):
            print(f"  - {key}")
    
    # Check shape compatibility for matching keys
    shape_mismatches = []
    for key in matching_keys:
        ckpt_shape = checkpoint_state_dict[key].shape
        model_shape = model_state_dict[key].shape
        if ckpt_shape != model_shape:
            shape_mismatches.append((key, ckpt_shape, model_shape))
    
    if shape_mismatches:
        print(f"\nShape mismatches ({len(shape_mismatches)}):")
        for key, ckpt_shape, model_shape in shape_mismatches[:5]:
            print(f"  - {key}: checkpoint {ckpt_shape} vs model {model_shape}")
    
    # Compatibility assessment
    total_params = len(model_keys)
    loaded_params = len(matching_keys)
    load_percentage = (loaded_params / total_params) * 100
    
    print(f"\nCompatibility Assessment:")
    print(f"  Parameters loaded: {loaded_params:,} / {total_params:,} ({load_percentage:.1f}%)")
    
    if load_percentage > 90:
        print(f"  [OK] Excellent compatibility")
    elif load_percentage > 70:
        print(f"  [WARN]  Good compatibility with some missing parameters")
    elif load_percentage > 50:
        print(f"  [WARN]  Moderate compatibility - may need configuration adjustment")
    else:
        print(f"  [ERROR] Poor compatibility - significant configuration mismatch")

def simulate_loading(config):
    """Simulate the actual loading process during training"""
    print("\n" + "="*60)
    print("SIMULATED TRAINING LOADING")
    print("="*60)
    
    try:
        from models.span import create_span_model
        
        # Create model
        model = create_span_model(config['model'])
        
        # Load checkpoint
        checkpoint_path = config['training']['pretrained_checkpoint']
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except Exception:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Attempt loading with strict=False (as done in training)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        print(f"Loading simulation results:")
        print(f"  Missing keys: {len(missing_keys)}")
        print(f"  Unexpected keys: {len(unexpected_keys)}")
        
        # Count loaded parameters
        total_params = sum(p.numel() for p in model.parameters())
        loaded_params = total_params - sum(p.numel() for k, p in model.named_parameters() if k in missing_keys)
        load_percentage = (loaded_params / total_params) * 100
        
        print(f"  Parameters loaded: {loaded_params:,} / {total_params:,} ({load_percentage:.1f}%)")
        
        # Check for NaN in model
        has_nan = any(torch.isnan(p).any() for p in model.parameters())
        has_inf = any(torch.isinf(p).any() for p in model.parameters())
        
        print(f"  Model has NaN: {has_nan}")
        print(f"  Model has Inf: {has_inf}")
        
        return load_percentage
        
    except Exception as e:
        print(f"[ERROR] Error during loading simulation: {e}")
        return 0

def analyze_nan_sources():
    """Analyze potential sources of NaN gradients"""
    print("\n" + "="*60)
    print("NAN GRADIENT ANALYSIS")
    print("="*60)
    
    print("Common causes of NaN gradients in transfer learning:")
    print("1. Learning rate too high for pretrained weights")
    print("2. Improper weight initialization")
    print("3. Gradient explosion from teacher models")
    print("4. Loss function instability")
    print("5. Data preprocessing issues")
    
    # Check learning rate
    print(f"\nLearning rate analysis:")
    print(f"  Current LR: {config['training']['lr']}")
    print(f"  Recommended for fine-tuning: 1e-5 to 1e-6")
    
    if config['training']['lr'] > 1e-4:
        print(f"  [WARN]  Learning rate may be too high for fine-tuning")
    
    # Check loss configuration
    loss_config = config.get('loss', {})
    print(f"\nLoss configuration:")
    for loss_type, loss_settings in loss_config.items():
        if isinstance(loss_settings, dict) and loss_settings.get('enabled'):
            weight = loss_settings.get('weight', 1.0)
            print(f"  {loss_type}: weight={weight}")

def main():
    """Main analysis function"""
    config_path = "configs/finetune_spanf.yaml"
    
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        return
    
    print("TRANSFER LEARNING DIAGNOSTIC TOOL")
    print("="*60)
    
    # Load config
    global config
    config = load_config(config_path)
    print(f"[OK] Config loaded: {config_path}")
    
    # Analyze checkpoint loading
    checkpoint_result = analyze_checkpoint_loading(config)
    
    if checkpoint_result and len(checkpoint_result) == 2:
        checkpoint_success, checkpoint_state_dict = checkpoint_result
        
        # Analyze model creation
        model_success, model_state_dict = analyze_model_creation(config)
        
        if model_success:
            # Check compatibility
            check_compatibility(checkpoint_state_dict, model_state_dict)
            
            # Simulate loading
            load_percentage = simulate_loading(config)
            
            # Analyze NaN sources
            analyze_nan_sources()
            
            # Summary
            print("\n" + "="*60)
            print("DIAGNOSTIC SUMMARY")
            print("="*60)
            
            print(f"1. Checkpoint Loading: {'[OK]' if checkpoint_success else '[ERROR]'}")
            print(f"2. Model Creation: {'[OK]' if model_success else '[ERROR]'}")
            print(f"3. Parameter Loading: {load_percentage:.1f}%")
            
            if load_percentage < 50:
                print(f"\n[CONFIG] PRIMARY ISSUE: Poor checkpoint compatibility")
                print(f"   - Expected: 48 channels, 12 blocks")
                print(f"   - Check if checkpoint matches model architecture")
            elif load_percentage < 90:
                print(f"\n[WARN]  SECONDARY ISSUE: Partial compatibility")
                print(f"   - Some parameters not loaded")
                print(f"   - May cause training instability")
            
            print(f"\n[CONFIG] NaN GRADIENT FIXES:")
            print(f"1. Reduce learning rate to 1e-6")
            print(f"2. Enable gradient clipping")
            print(f"3. Check teacher model outputs")
            print(f"4. Use mixed precision training")

if __name__ == "__main__":
    main()
