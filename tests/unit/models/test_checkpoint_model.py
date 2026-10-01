#!/usr/bin/env python3
"""
Test suite for checkpoint-compatible SPAN model

Tests model creation, checkpoint loading, and forward pass functionality.
"""

import sys
import os
import torch
import yaml
from pathlib import Path

# Add src to path
def test_model_creation():
    """Test checkpoint-compatible model creation"""
    print("="*60)
    print("TEST 1: Model Creation")
    print("="*60)
    
    try:
        from anime_sr.models.span import CheckpointCompatibleSPAN
        
        # Test default model
        model = CheckpointCompatibleSPAN()
        print(f"✅ Default model created successfully")
        print(f"   Parameters: {model.get_parameter_count():,}")
        
        # Test custom model
        model_custom = CheckpointCompatibleSPAN(
            scale=4,
            channels=48,
            hidden_channels=96,
            num_blocks=6
        )
        print(f"✅ Custom model created successfully")
        print(f"   Parameters: {model_custom.get_parameter_count():,}")
        
        # Test parameter names
        param_names = model.get_parameter_names()
        print(f"✅ Parameter names generated: {len(param_names)}")
        
        return True, model
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False, None

def test_model_factory():
    """Test model factory integration"""
    print("\n" + "="*60)
    print("TEST 2: Model Factory Integration")
    print("="*60)
    
    try:
        from anime_sr.models.span import create_span_model
        
        # Test checkpoint-compatible model creation via factory
        config = {
            'type': 'checkpoint_compatible',
            'scale': 4,
            'channels': 48,
            'hidden_channels': 96,
            'num_blocks': 6
        }
        
        model = create_span_model(config)
        print(f"✅ Factory created checkpoint-compatible model")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True, model
        
    except Exception as e:
        print(f"❌ Factory integration failed: {e}")
        return False, None

def test_checkpoint_loading():
    """Test checkpoint loading functionality"""
    print("\n" + "="*60)
    print("TEST 3: Checkpoint Loading")
    print("="*60)
    
    try:
        from anime_sr.models.span import CheckpointCompatibleSPAN
        from anime_sr.utils.checkpoint_loader import load_checkpoint_compatible_span
        
        # Create model
        model = CheckpointCompatibleSPAN()
        
        # Load checkpoint
        checkpoint_path = "checkpoints/span/spanx4_ch48.pth"
        success, loading_info = load_checkpoint_compatible_span(
            model, checkpoint_path, verbose=True
        )
        
        print(f"✅ Checkpoint loading completed")
        print(f"   Success: {success}")
        print(f"   Load percentage: {loading_info['load_percentage']:.1f}%")
        print(f"   Loaded parameters: {loading_info['loaded_params']}")
        print(f"   Missing parameters: {len(loading_info['missing_params'])}")
        print(f"   Unexpected parameters: {len(loading_info['unexpected_params'])}")
        
        return success, model, loading_info
        
    except Exception as e:
        print(f"❌ Checkpoint loading failed: {e}")
        return False, None, None

def test_forward_pass():
    """Test forward pass functionality"""
    print("\n" + "="*60)
    print("TEST 4: Forward Pass")
    print("="*60)
    
    try:
        from anime_sr.models.span import CheckpointCompatibleSPAN
        from anime_sr.utils.checkpoint_loader import load_checkpoint_compatible_span
        
        # Create and load model
        model = CheckpointCompatibleSPAN()
        checkpoint_path = "checkpoints/span/spanx4_ch48.pth"
        success, _, _ = load_checkpoint_compatible_span(model, checkpoint_path, verbose=False)
        
        if not success:
            print(f"❌ Cannot test forward pass - checkpoint loading failed")
            return False
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            # Test different input sizes
            test_sizes = [
                (1, 3, 64, 64),
                (1, 3, 128, 128),
                (2, 3, 96, 96),
            ]
            
            for i, (batch, ch, h, w) in enumerate(test_sizes):
                test_input = torch.randn(batch, ch, h, w)
                
                # Forward pass
                output = model(test_input)
                
                expected_h, expected_w = h * 4, w * 4
                expected_shape = (batch, ch, expected_h, expected_w)
                
                if output.shape == expected_shape:
                    print(f"✅ Test {i+1}: {test_input.shape} -> {output.shape}")
                else:
                    print(f"❌ Test {i+1}: Expected {expected_shape}, got {output.shape}")
                    return False
                
                # Check output range
                if torch.all(output >= 0) and torch.all(output <= 1):
                    print(f"   Output range valid: [{output.min():.3f}, {output.max():.3f}]")
                else:
                    print(f"   ⚠️  Output range issue: [{output.min():.3f}, {output.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("\n" + "="*60)
    print("TEST 5: Configuration Loading")
    print("="*60)
    
    try:
        config_path = "configs/finetune_checkpoint_compatible.yaml"
        
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            return False
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Config loaded successfully")
        print(f"   Model type: {config['model']['type']}")
        print(f"   Model scale: {config['model']['scale']}")
        print(f"   Model channels: {config['model']['channels']}")
        print(f"   Checkpoint path: {config['training']['pretrained_checkpoint']}")
        
        # Test model creation from config
        from anime_sr.models.span import create_span_model
        model = create_span_model(config['model'])
        
        print(f"✅ Model created from config")
        print(f"   Model type: {type(model).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

def test_checkpoint_compatibility():
    """Test checkpoint compatibility verification"""
    print("\n" + "="*60)
    print("TEST 6: Checkpoint Compatibility Verification")
    print("="*60)
    
    try:
        from anime_sr.utils.checkpoint_loader import verify_checkpoint_compatibility
        from anime_sr.models.span import CheckpointCompatibleSPAN
        
        checkpoint_path = "checkpoints/span/spanx4_ch48.pth"
        
        # Verify compatibility
        analysis = verify_checkpoint_compatibility(
            checkpoint_path, 
            model_class=CheckpointCompatibleSPAN,
            verbose=True
        )
        
        if analysis['compatible_models']:
            compatibility = analysis['compatible_models'][0]['compatibility']
            print(f"✅ Compatibility analysis completed")
            print(f"   Compatibility: {compatibility:.1f}%")
            
            if compatibility >= 95:
                print(f"   ✅ Excellent compatibility")
            elif compatibility >= 80:
                print(f"   ⚠️  Good compatibility")
            else:
                print(f"   ❌ Poor compatibility")
                return False
        else:
            print(f"❌ No compatibility data found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Compatibility verification failed: {e}")
        return False

def main():
    """Run all tests"""
    print("CHECKPOINT-COMPATIBLE SPAN MODEL TEST SUITE")
    print("="*60)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Model Factory", test_model_factory),
        ("Checkpoint Loading", test_checkpoint_loading),
        ("Forward Pass", test_forward_pass),
        ("Config Loading", test_config_loading),
        ("Compatibility Verification", test_checkpoint_compatibility),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if isinstance(result, tuple):
                success = result[0]
            else:
                success = result
            
            results.append((test_name, success))
            
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Checkpoint-compatible model is ready for training.")
    elif passed >= total * 0.8:
        print("⚠️  Most tests passed. Model should work with some limitations.")
    else:
        print("❌ Multiple test failures. Model needs fixes before training.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
