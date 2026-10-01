#!/usr/bin/env python3
"""
Training Readiness Validation Script

Validates that the checkpoint-compatible model is ready for training:
- Model creation from config
- Checkpoint loading
- Forward pass with different inputs
- Gradient computation
- Loss calculation
- Optimizer setup
"""

import sys
import os
import torch
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_model_creation_from_config():
    """Test model creation from configuration"""
    print("="*60)
    print("TEST: Model Creation from Config")
    print("="*60)
    
    try:
        # Load config
        config_path = "configs/finetune_checkpoint_compatible.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"[OK] Config loaded: {config_path}")
        
        # Create model using factory
        from anime_sr.models.span import create_span_model
        model = create_span_model(config['model'])
        
        print(f"[OK] Model created via factory")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True, model, config
        
    except Exception as e:
        print(f"[ERROR] Model creation failed: {e}")
        return False, None, None

def test_checkpoint_loading(model):
    """Test checkpoint loading with detailed reporting"""
    print("\n" + "="*60)
    print("TEST: Checkpoint Loading")
    print("="*60)
    
    try:
        from anime_sr.utils.checkpoint_loader import load_checkpoint_compatible_span
        
        checkpoint_path = "checkpoints/span/spanx4_ch48.pth"
        success, loading_info = load_checkpoint_compatible_span(
            model, checkpoint_path, verbose=True
        )
        
        print(f"Loading Results:")
        print(f"  Success: {success}")
        print(f"  Load percentage: {loading_info['load_percentage']:.1f}%")
        print(f"  Loaded parameters: {loading_info['loaded_params']}")
        print(f"  Missing parameters: {len(loading_info['missing_params'])}")
        print(f"  Unexpected parameters: {len(loading_info['unexpected_params'])}")
        
        return success, loading_info
        
    except Exception as e:
        print(f"[ERROR] Checkpoint loading failed: {e}")
        return False, {}

def test_forward_pass_variations(model):
    """Test forward pass with different input variations"""
    print("\n" + "="*60)
    print("TEST: Forward Pass Variations")
    print("="*60)
    
    try:
        model.eval()
        
        test_cases = [
            {"name": "Small input", "shape": (1, 3, 32, 32)},
            {"name": "Medium input", "shape": (1, 3, 64, 64)},
            {"name": "Large input", "shape": (1, 3, 128, 128)},
            {"name": "Batch input", "shape": (4, 3, 64, 64)},
            {"name": "Different aspect ratio", "shape": (1, 3, 48, 96)},
        ]
        
        for test_case in test_cases:
            test_input = torch.randn(*test_case["shape"])
            
            with torch.no_grad():
                output = model(test_input)
            
            expected_h, expected_w = test_case["shape"][2] * 4, test_case["shape"][3] * 4
            expected_shape = (test_case["shape"][0], 3, expected_h, expected_w)
            
            if output.shape == expected_shape:
                print(f"[OK] {test_case['name']}: {test_input.shape} -> {output.shape}")
            else:
                print(f"[ERROR] {test_case['name']}: Expected {expected_shape}, got {output.shape}")
                return False
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Forward pass test failed: {e}")
        return False

def test_gradient_computation(model):
    """Test gradient computation"""
    print("\n" + "="*60)
    print("TEST: Gradient Computation")
    print("="*60)
    
    try:
        model.train()
        
        # Forward pass
        test_input = torch.randn(2, 3, 64, 64, requires_grad=True)
        output = model(test_input)
        
        # Compute loss
        target = torch.randn_like(output)
        loss = torch.nn.functional.mse_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        grad_count = 0
        nan_count = 0
        inf_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_count += 1
                
                if torch.isnan(param.grad).any():
                    nan_count += 1
                    print(f"  [WARN]  NaN gradient in: {name}")
                
                if torch.isinf(param.grad).any():
                    inf_count += 1
                    print(f"  [WARN]  Inf gradient in: {name}")
        
        print(f"[OK] Gradient computation completed")
        print(f"   Parameters with gradients: {grad_count}")
        print(f"   NaN gradients: {nan_count}")
        print(f"   Inf gradients: {inf_count}")
        print(f"   Loss value: {loss.item():.6f}")
        
        return nan_count == 0 and inf_count == 0
        
    except Exception as e:
        print(f"[ERROR] Gradient computation failed: {e}")
        return False

def test_optimizer_setup(model, config):
    """Test optimizer setup"""
    print("\n" + "="*60)
    print("TEST: Optimizer Setup")
    print("="*60)
    
    try:
        # Extract optimizer config
        training_config = config.get('training', {})
        lr = float(training_config.get('lr', 1e-4))
        weight_decay = float(training_config.get('weight_decay', 0.001))
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        print(f"[OK] Optimizer created")
        print(f"   Type: AdamW")
        print(f"   LR: {lr}")
        print(f"   Weight decay: {weight_decay}")
        print(f"   Parameter groups: {len(optimizer.param_groups)}")
        
        # Test optimizer step
        test_input = torch.randn(1, 3, 64, 64)
        output = model(test_input)
        target = torch.randn_like(output)
        loss = torch.nn.functional.l1_loss(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"[OK] Optimizer step completed")
        print(f"   Loss before step: {loss.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Optimizer setup failed: {e}")
        return False

def test_training_step(model, config):
    """Test complete training step"""
    print("\n" + "="*60)
    print("TEST: Training Step")
    print("="*60)
    
    try:
        model.train()
        
        # Extract training config
        training_config = config.get('training', {})
        lr = float(training_config.get('lr', 1e-4))
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
        # Simulate training step
        batch_size = 2
        test_input = torch.randn(batch_size, 3, 64, 64)
        target = torch.randn(batch_size, 3, 256, 256)
        
        # Forward pass
        output = model(test_input)
        
        # Compute loss
        pixel_loss = torch.nn.functional.l1_loss(output, target)
        perceptual_loss = torch.nn.functional.mse_loss(output, target)
        total_loss = pixel_loss + 0.1 * perceptual_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        print(f"[OK] Training step completed")
        print(f"   Batch size: {batch_size}")
        print(f"   Pixel loss: {pixel_loss.item():.6f}")
        print(f"   Perceptual loss: {perceptual_loss.item():.6f}")
        print(f"   Total loss: {total_loss.item():.6f}")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Check for NaN/Inf in loss
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"[ERROR] Invalid loss value: {total_loss.item()}")
            return False
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Training step failed: {e}")
        return False

def test_memory_usage(model):
    """Test memory usage"""
    print("\n" + "="*60)
    print("TEST: Memory Usage")
    print("="*60)
    
    try:
        model.eval()
        
        # Test different input sizes
        test_sizes = [
            (1, 3, 64, 64),
            (1, 3, 128, 128),
            (2, 3, 64, 64),
        ]
        
        for batch, ch, h, w in test_sizes:
            test_input = torch.randn(batch, ch, h, w)
            
            # Measure memory before forward pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                mem_before = torch.cuda.memory_allocated()
            
            # Forward pass
            with torch.no_grad():
                output = model(test_input)
            
            # Measure memory after forward pass
            if torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated()
                mem_peak = torch.cuda.max_memory_allocated()
                
                print(f"[OK] {test_input.shape}:")
                print(f"   Memory used: {(mem_after - mem_before) / 1024**2:.1f} MB")
                print(f"   Peak memory: {mem_peak / 1024**2:.1f} MB")
            else:
                print(f"[OK] {test_input.shape}: CPU only (no CUDA memory tracking)")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Memory usage test failed: {e}")
        return False

def main():
    """Run all training readiness tests"""
    print("TRAINING READINESS VALIDATION")
    print("="*60)
    
    tests = [
        ("Model Creation from Config", test_model_creation_from_config),
        ("Forward Pass Variations", test_forward_pass_variations),
        ("Gradient Computation", test_gradient_computation),
        ("Optimizer Setup", test_optimizer_setup),
        ("Training Step", test_training_step),
        ("Memory Usage", test_memory_usage),
    ]
    
    results = []
    model = None
    config = None
    
    for test_name, test_func in tests:
        try:
            if test_name == "Model Creation from Config":
                success, model, config = test_func()
                results.append((test_name, success))
            elif test_name == "Forward Pass Variations":
                success = test_func(model) if model else False
                results.append((test_name, success))
            elif test_name == "Gradient Computation":
                success = test_func(model) if model else False
                results.append((test_name, success))
            elif test_name == "Optimizer Setup":
                success = test_func(model, config) if model and config else False
                results.append((test_name, success))
            elif test_name == "Training Step":
                success = test_func(model, config) if model and config else False
                results.append((test_name, success))
            elif test_name == "Memory Usage":
                success = test_func(model) if model else False
                results.append((test_name, success))
            
        except Exception as e:
            print(f"[ERROR] {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Test checkpoint loading separately
    if model:
        success, _ = test_checkpoint_loading(model)
        results.append(("Checkpoint Loading", success))
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "[OK] PASS" if success else "[ERROR] FAIL"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("[CELEBRATE] All tests passed! Model is ready for training.")
    elif passed >= total * 0.8:
        print("[WARN]  Most tests passed. Model should work with some limitations.")
    else:
        print("[ERROR] Multiple test failures. Model needs fixes before training.")
    
    # Training readiness assessment
    print(f"\nTRAINING READINESS ASSESSMENT:")
    print(f"  Model creation: {'[OK]' if any('Model Creation' in name and success for name, success in results) else '[ERROR]'}")
    print(f"  Checkpoint loading: {'[OK]' if any('Checkpoint Loading' in name and success for name, success in results) else '[ERROR]'}")
    print(f"  Forward pass: {'[OK]' if any('Forward Pass' in name and success for name, success in results) else '[ERROR]'}")
    print(f"  Gradient computation: {'[OK]' if any('Gradient' in name and success for name, success in results) else '[ERROR]'}")
    print(f"  Optimizer setup: {'[OK]' if any('Optimizer' in name and success for name, success in results) else '[ERROR]'}")
    print(f"  Training step: {'[OK]' if any('Training Step' in name and success for name, success in results) else '[ERROR]'}")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
