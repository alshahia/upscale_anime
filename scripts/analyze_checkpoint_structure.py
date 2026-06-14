#!/usr/bin/env python3
"""
Detailed Checkpoint Structure Analysis

This script performs comprehensive analysis of the SPAN checkpoint to extract:
- Exact layer architecture and naming patterns
- Parameter shapes and counts
- Block structure and connectivity
- Upsampling and residual connections
"""

import os
import sys
import torch
import json
from pathlib import Path
from collections import defaultdict, OrderedDict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def analyze_checkpoint_structure(checkpoint_path):
    """Perform detailed analysis of checkpoint structure"""
    print("="*80)
    print("DETAILED CHECKPOINT STRUCTURE ANALYSIS")
    print("="*80)
    
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except Exception:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"[OK] Checkpoint loaded: {checkpoint_path}")
        
        # Extract parameters
        if 'params' in checkpoint:
            params = checkpoint['params']
            print(f"Using 'params' key with {len(params)} parameters")
        elif 'params_ema' in checkpoint:
            params = checkpoint['params_ema']
            print(f"Using 'params_ema' key with {len(params)} parameters")
        else:
            params = checkpoint
            print(f"Using checkpoint as direct parameter dict with {len(params)} parameters")
        
        # Analyze structure
        analysis = analyze_parameter_structure(params)
        
        # Generate architecture map
        architecture_map = generate_architecture_map(params)
        
        # Save analysis results
        save_analysis_results(analysis, architecture_map, checkpoint_path)
        
        return analysis, architecture_map
        
    except Exception as e:
        print(f"[ERROR] Error analyzing checkpoint: {e}")
        return None, None

def analyze_parameter_structure(params):
    """Analyze parameter structure and patterns"""
    print("\n" + "="*60)
    print("PARAMETER STRUCTURE ANALYSIS")
    print("="*60)
    
    analysis = {
        'total_parameters': len(params),
        'parameter_groups': defaultdict(list),
        'layer_types': defaultdict(list),
        'block_structure': defaultdict(dict),
        'shape_analysis': defaultdict(list),
        'naming_patterns': {}
    }
    
    # Group parameters by type and structure
    for name, param in params.items():
        parts = name.split('.')
        
        # Identify main components
        if 'conv_1' in name:
            analysis['parameter_groups']['conv_1'].append((name, param.shape))
            if 'weight' in name:
                analysis['layer_types']['conv_weight'].append((name, param.shape))
            elif 'bias' in name:
                analysis['layer_types']['conv_bias'].append((name, param.shape))
        
        elif 'block_' in name:
            block_num = name.split('_')[1]
            analysis['parameter_groups']['blocks'].append((name, param.shape))
            if block_num not in analysis['block_structure']:
                analysis['block_structure'][block_num] = []
            analysis['block_structure'][block_num].append((name, param.shape))
        
        elif 'upsample' in name or 'up' in name:
            analysis['parameter_groups']['upsampling'].append((name, param.shape))
        
        elif 'final' in name or 'output' in name:
            analysis['parameter_groups']['output'].append((name, param.shape))
        
        # Shape analysis
        if len(param.shape) == 4:  # Convolution
            analysis['shape_analysis']['conv4d'].append((name, param.shape))
        elif len(param.shape) == 2:  # Linear/BatchNorm
            analysis['shape_analysis']['linear2d'].append((name, param.shape))
        elif len(param.shape) == 1:  # Bias
            analysis['shape_analysis']['bias1d'].append((name, param.shape))
    
    # Print analysis
    print(f"Total parameters: {analysis['total_parameters']}")
    print(f"Parameter groups:")
    for group, params_list in analysis['parameter_groups'].items():
        print(f"  {group}: {len(params_list)} parameters")
    
    print(f"\nLayer types:")
    for layer_type, params_list in analysis['layer_types'].items():
        print(f"  {layer_type}: {len(params_list)} parameters")
    
    print(f"\nBlock structure:")
    for block_num, params_list in sorted(analysis['block_structure'].items()):
        print(f"  Block {block_num}: {len(params_list)} parameters")
    
    print(f"\nShape distribution:")
    for shape_type, params_list in analysis['shape_analysis'].items():
        print(f"  {shape_type}: {len(params_list)} parameters")
    
    return analysis

def generate_architecture_map(params):
    """Generate detailed architecture map"""
    print("\n" + "="*60)
    print("ARCHITECTURE MAP GENERATION")
    print("="*60)
    
    architecture = {
        'input_channels': 3,
        'output_channels': 3,
        'scale_factor': 4,
        'conv_1': {},
        'blocks': {},
        'upsampling': {},
        'output': {},
        'connections': []
    }
    
    # Analyze conv_1 structure
    conv_1_params = {name: param for name, param in params.items() if 'conv_1' in name}
    if conv_1_params:
        architecture['conv_1'] = analyze_conv_1_structure(conv_1_params)
    
    # Analyze block structure
    block_params = {name: param for name, param in params.items() if 'block_' in name}
    if block_params:
        architecture['blocks'] = analyze_block_structure(block_params)
    
    # Analyze upsampling structure
    upsample_params = {name: param for name, param in params.items() 
                      if any(key in name for key in ['upsample', 'up', 'pixel_shuffle'])}
    if upsample_params:
        architecture['upsampling'] = analyze_upsampling_structure(upsample_params)
    
    # Analyze output structure
    output_params = {name: param for name, param in params.items() 
                    if any(key in name for key in ['final', 'output', 'conv_last'])}
    if output_params:
        architecture['output'] = analyze_output_structure(output_params)
    
    # Print architecture summary
    print(f"Architecture Summary:")
    print(f"  Input channels: {architecture['input_channels']}")
    print(f"  Output channels: {architecture['output_channels']}")
    print(f"  Scale factor: {architecture['scale_factor']}")
    print(f"  Conv_1 layers: {len(architecture['conv_1'])}")
    print(f"  Blocks: {len(architecture['blocks'])}")
    print(f"  Upsampling layers: {len(architecture['upsampling'])}")
    print(f"  Output layers: {len(architecture['output'])}")
    
    return architecture

def analyze_conv_1_structure(conv_1_params):
    """Analyze conv_1 specific structure"""
    structure = {}
    
    # Look for specific patterns
    for name, shape in conv_1_params.items():
        if 'sk' in name:  # Spectral kernel?
            structure['spectral_kernel'] = {'name': name, 'shape': shape}
        elif 'eval_conv' in name:
            structure['evaluation_conv'] = {'name': name, 'shape': shape}
        elif 'conv.' in name:
            if 'conv.' not in structure:
                structure['conv_layers'] = []
            structure['conv_layers'].append({'name': name, 'shape': shape})
    
    return structure

def analyze_block_structure(block_params):
    """Analyze block structure and patterns"""
    blocks = {}
    
    # Group by block number
    block_groups = defaultdict(list)
    for name, shape in block_params.items():
        parts = name.split('_')
        if len(parts) >= 2 and parts[0] == 'block':
            block_num = parts[1]
            block_groups[block_num].append({'name': name, 'shape': shape})
    
    # Analyze each block
    for block_num, params_list in sorted(block_groups.items()):
        block_structure = {
            'parameters': params_list,
            'sub_blocks': {},
            'total_params': len(params_list)
        }
        
        # Look for sub-block patterns (c1_r, c2_r, c3_r)
        for param_info in params_list:
            name = param_info['name']
            if 'c1_r' in name:
                if 'c1_r' not in block_structure['sub_blocks']:
                    block_structure['sub_blocks']['c1_r'] = []
                block_structure['sub_blocks']['c1_r'].append(param_info)
            elif 'c2_r' in name:
                if 'c2_r' not in block_structure['sub_blocks']:
                    block_structure['sub_blocks']['c2_r'] = []
                block_structure['sub_blocks']['c2_r'].append(param_info)
            elif 'c3_r' in name:
                if 'c3_r' not in block_structure['sub_blocks']:
                    block_structure['sub_blocks']['c3_r'] = []
                block_structure['sub_blocks']['c3_r'].append(param_info)
        
        blocks[block_num] = block_structure
    
    return blocks

def analyze_upsampling_structure(upsample_params):
    """Analyze upsampling structure"""
    structure = {'layers': []}
    
    for name, shape in upsample_params.items():
        structure['layers'].append({'name': name, 'shape': shape})
    
    return structure

def analyze_output_structure(output_params):
    """Analyze output layer structure"""
    structure = {'layers': []}
    
    for name, shape in output_params.items():
        structure['layers'].append({'name': name, 'shape': shape})
    
    return structure

def save_analysis_results(analysis, architecture, checkpoint_path):
    """Save analysis results to files"""
    output_dir = Path(checkpoint_path).parent / 'analysis'
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed analysis
    analysis_file = output_dir / 'checkpoint_analysis.json'
    with open(analysis_file, 'w') as f:
        # Convert non-serializable objects
        serializable_analysis = convert_to_serializable(analysis)
        json.dump(serializable_analysis, f, indent=2)
    
    # Save architecture map
    arch_file = output_dir / 'architecture_map.json'
    with open(arch_file, 'w') as f:
        serializable_architecture = convert_to_serializable(architecture)
        json.dump(serializable_architecture, f, indent=2)
    
    # Save parameter list
    params_file = output_dir / 'parameter_list.txt'
    with open(params_file, 'w') as f:
        for group, params_list in analysis['parameter_groups'].items():
            f.write(f"\n=== {group.upper()} ===\n")
            for name, shape in params_list:
                f.write(f"{name}: {shape}\n")
    
    print(f"\n[OK] Analysis results saved to: {output_dir}")
    print(f"  - checkpoint_analysis.json")
    print(f"  - architecture_map.json")
    print(f"  - parameter_list.txt")

def convert_to_serializable(obj):
    """Convert non-serializable objects to serializable format"""
    if isinstance(obj, defaultdict):
        return dict(obj)
    elif isinstance(obj, torch.Size):
        return list(obj)
    elif isinstance(obj, torch.Tensor):
        return str(obj.shape)  # Convert tensor shapes to strings
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj

def main():
    """Main analysis function"""
    checkpoint_path = "checkpoints/span/spanx4_ch48.pth"
    
    print("SPAN CHECKPOINT STRUCTURE ANALYZER")
    print("="*80)
    
    # Perform analysis
    analysis, architecture = analyze_checkpoint_structure(checkpoint_path)
    
    if analysis and architecture:
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"Total parameters analyzed: {analysis['total_parameters']}")
        print(f"Blocks identified: {len(architecture['blocks'])}")
        print(f"Architecture mapping saved successfully")
        
        # Provide key insights for model creation
        print("\n" + "="*60)
        print("KEY INSIGHTS FOR MODEL CREATION")
        print("="*60)
        
        if architecture['conv_1']:
            print("[OK] conv_1 structure identified")
        
        if architecture['blocks']:
            block_count = len(architecture['blocks'])
            print(f"[OK] {block_count} blocks identified")
            
            # Check block consistency
            first_block = list(architecture['blocks'].values())[0]
            sub_blocks = first_block.get('sub_blocks', {})
            if sub_blocks:
                print(f"[OK] Block sub-structure: {list(sub_blocks.keys())}")
        
        print("[OK] Architecture map ready for model implementation")
    else:
        print("[ERROR] Analysis failed")

if __name__ == "__main__":
    main()
