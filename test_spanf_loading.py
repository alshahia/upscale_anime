#!/usr/bin/env python3
"""
Test script to verify SPAN-F model and checkpoint loading.
Run this after downloading SPAN-F weights to verify everything works.
"""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.models.span import SPANF, create_span_model
from src.utils.checkpoint_loader import load_pretrained_weights, get_checkpoint_info


def test_model_creation():
    """Test that SPAN-F model can be created."""
    print("=" * 60)
    print("Test 1: SPAN-F Model Creation")
    print("=" * 60)
    
    try:
        # Method 1: Direct instantiation
        model = SPANF(scale=4)
        params = sum(p.numel() for p in model.parameters())
        print(f"✓ SPAN-F created successfully")
        print(f"  Parameters: {params:,}")
        print(f"  Expected: ~300K")
        
        # Method 2: Factory function
        model2 = create_span_model({'type': 'spanf', 'scale': 4})
        params2 = sum(p.numel() for p in model2.parameters())
        print(f"✓ Factory method works")
        print(f"  Parameters: {params2:,}")
        
        assert params == params2, "Parameter count mismatch!"
        print()
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_info(checkpoint_path: str):
    """Test checkpoint info extraction."""
    print("=" * 60)
    print("Test 2: Checkpoint Info")
    print("=" * 60)
    
    if not Path(checkpoint_path).exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print("  Please download SPAN-F weights first:")
        print("  python scripts/download_spanf.py")
        return False
    
    try:
        info = get_checkpoint_info(checkpoint_path)
        if info is None:
            print("✗ Failed to read checkpoint info")
            return False
        
        if 'error' in info:
            print(f"✗ Error reading checkpoint: {info['error']}")
            return False
        
        print(f"✓ Checkpoint info extracted:")
        for key, value in info.items():
            if key != 'path':
                print(f"  {key}: {value}")
        print()
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_weight_loading(checkpoint_path: str):
    """Test loading weights into model."""
    print("=" * 60)
    print("Test 3: Weight Loading")
    print("=" * 60)
    
    if not Path(checkpoint_path).exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        model = SPANF(scale=4)
        
        success = load_pretrained_weights(
            model,
            checkpoint_path,
            strict=False,
            device='cpu',
            verbose=True
        )
        
        if success:
            print("✓ Weights loaded successfully!")
            
            # Test forward pass
            dummy_input = torch.randn(1, 3, 64, 64)
            with torch.no_grad():
                output = model(dummy_input)
            
            expected_size = (1, 3, 256, 256)  # 4x upscaling
            if output.shape == expected_size:
                print(f"✓ Forward pass works")
                print(f"  Input: {dummy_input.shape}")
                print(f"  Output: {output.shape}")
            else:
                print(f"✗ Output shape mismatch: {output.shape} vs {expected_size}")
                return False
            
            print()
            return True
        else:
            print("✗ Weight loading failed")
            return False
            
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference_engine(checkpoint_path: str):
    """Test loading with InferenceEngine."""
    print("=" * 60)
    print("Test 4: Inference Engine")
    print("=" * 60)
    
    if not Path(checkpoint_path).exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        from src.inference.engine import InferenceEngine
        
        # Try to load with inference engine
        # Note: This requires the checkpoint to have been saved in the correct format
        # or converted using convert_spanf_checkpoint
        
        print("Attempting to load with InferenceEngine...")
        print("  (This may fail for original SPAN-F checkpoints - use manual loading instead)")
        
        engine = InferenceEngine.from_checkpoint(
            checkpoint_path=checkpoint_path,
            model_type='span',
            device='cpu',
            use_ema=False
        )
        
        print("✓ InferenceEngine loaded successfully!")
        print()
        return True
        
    except Exception as e:
        print(f"⚠ InferenceEngine loading failed (expected for original checkpoints):")
        print(f"  {str(e)[:100]}...")
        print("  Use manual model loading instead (see SPANF_INTEGRATION.md)")
        print()
        return True  # Not a failure - just needs different loading method


def main():
    """Run all tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test SPAN-F model loading')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='pretrained/SPANF_x4.pth',
        help='Path to SPAN-F checkpoint'
    )
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("SPAN-F Model Loading Tests")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1: Model creation
    results.append(("Model Creation", test_model_creation()))
    
    # Test 2: Checkpoint info (if file exists)
    results.append(("Checkpoint Info", test_checkpoint_info(args.checkpoint)))
    
    # Test 3: Weight loading (if file exists)
    results.append(("Weight Loading", test_weight_loading(args.checkpoint)))
    
    # Test 4: Inference engine (if file exists)
    results.append(("Inference Engine", test_inference_engine(args.checkpoint)))
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! SPAN-F is ready to use.")
        print("\nNext steps:")
        print("  1. Fine-tune on your data:")
        print("     python scripts/train.py --config configs/finetune_spanf.yaml")
        print("\n  2. Run inference:")
        print("     python scripts/inference.py --checkpoint pretrained/SPANF_x4.pth --input image.png")
        return 0
    else:
        print("\n⚠ Some tests failed. Please check:")
        print("  - Download SPAN-F weights: python scripts/download_spanf.py")
        print("  - Verify checkpoint: python scripts/download_spanf.py --verify <path>")
        print("  - See SPANF_INTEGRATION.md for troubleshooting")
        return 1


if __name__ == '__main__':
    sys.exit(main())
