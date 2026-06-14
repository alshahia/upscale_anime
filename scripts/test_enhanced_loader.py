#!/usr/bin/env python3
"""
Test the enhanced checkpoint loader to verify improved parameter loading.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from utils.enhanced_checkpoint_loader import load_checkpoint_compatible_span_enhanced
from models.span.checkpoint_compatible_model_v3 import CheckpointCompatibleSPANv3

def test_enhanced_loader():
    """Test the enhanced checkpoint loader."""
    print("TESTING ENHANCED CHECKPOINT LOADER")
    print("=" * 50)
    
    # Create model
    model = CheckpointCompatibleSPANv3()
    
    # Test enhanced loading
    success, info = load_checkpoint_compatible_span_enhanced(
        model, 
        'checkpoints/span/spanx4_ch48.pth'
    )
    
    print(f'Enhanced loading success: {success}')
    print(f'Load percentage: {info["load_percentage"]:.1f}%')
    print(f'Loaded parameters: {info["loaded_params"]}')
    print(f'Missing parameters: {len(info["missing_params"])}')
    
    if info["load_percentage"] >= 95:
        print("[OK] Excellent loading success!")
    elif info["load_percentage"] >= 80:
        print("[WARN]  Good loading success")
    else:
        print("[ERROR] Poor loading success")
    
    return success, info["load_percentage"]

if __name__ == "__main__":
    success, load_percentage = test_enhanced_loader()
    print(f"\nFinal result: {success}, {load_percentage:.1f}%")
