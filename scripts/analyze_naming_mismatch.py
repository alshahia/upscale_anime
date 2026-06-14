#!/usr/bin/env python3
"""
Analyze naming mismatch between checkpoint and model
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def analyze_naming_mismatch():
    """Analyze the naming mismatch between checkpoint and model"""
    print("NAMING MISMATCH ANALYSIS")
    print("="*50)
    
    # Load checkpoint
    checkpoint_path = "checkpoints/span/spanx4_ch48.pth"
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    checkpoint_params = checkpoint['params']
    
    # Create model
    from models.span import CheckpointCompatibleSPAN
    model = CheckpointCompatibleSPAN()
    model_params = model.state_dict()
    
    print(f"Checkpoint parameters: {len(checkpoint_params)}")
    print(f"Model parameters: {len(model_params)}")
    
    # Show naming patterns
    print(f"\nCheckpoint naming pattern (first 10):")
    for i, (name, shape) in enumerate(list(checkpoint_params.items())[:10]):
        print(f"  {i+1}. {name}: {shape}")
    
    print(f"\nModel naming pattern (first 10):")
    for i, (name, shape) in enumerate(list(model_params.items())[:10]):
        print(f"  {i+1}. {name}: {shape}")
    
    # Find matches
    matches = set(checkpoint_params.keys()) & set(model_params.keys())
    print(f"\nDirect matches: {len(matches)}")
    if matches:
        for match in list(matches)[:5]:
            print(f"  - {match}")
    
    # Show the pattern differences
    print(f"\nPattern differences:")
    print(f"  Checkpoint: block_1.c1_r.sk.weight")
    print(f"  Model:     blocks.0.c1_r.sk.weight")
    print(f"  Checkpoint: conv_1.sk.weight")
    print(f"  Model:     conv_1.sk.weight")

if __name__ == "__main__":
    analyze_naming_mismatch()
