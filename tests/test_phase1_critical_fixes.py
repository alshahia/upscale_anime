"""
Regression tests for Phase 1 Critical Fixes.

Tests:
1. PyTorch AMP import compatibility (PyTorch 1.x and 2.0+)
2. torch.load() weights_only fallback for custom objects
"""
import sys
from pathlib import Path
import tempfile
import shutil

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import pytest


def test_pytorch_amp_import_compatibility():
    """Test that AMP imports work for both PyTorch 1.x and 2.0+"""
    # This test verifies the import pattern works
    try:
        # Try the new PyTorch 2.0+ import pattern
        from torch.amp import autocast
        from torch.cuda.amp import GradScaler
    except ImportError:
        # Fall back to old import pattern
        from torch.cuda.amp import autocast, GradScaler
    
    # Verify we can create a GradScaler (works on both CPU and GPU)
    scaler = GradScaler()
    assert scaler is not None, "GradScaler should be created successfully"
    
    print("✓ PyTorch AMP import compatibility test passed")


def test_torch_load_weights_only_fallback():
    """Test that checkpoint loading falls back gracefully for custom objects."""
    from src.training.base_trainer import BaseTrainer
    
    # Create a minimal model and trainer
    config = {
        'training': {
            'device': 'cpu',
            'lr': 0.001,
            'use_ema': False,
            'mixed_precision': False,
        },
        'model': {'scale': 4},
        'paths': {'checkpoint_dir': 'test_checkpoints_phase1'},
        'checkpoint': {'keep_last_n': 5, 'keep_best': True},
    }
    
    model = nn.Conv2d(3, 3, 3, padding=1)
    trainer = BaseTrainer(config, model)
    
    # Test 1: Create and save a checkpoint with only tensors (weights_only should work)
    checkpoint_dir = Path('test_checkpoints_phase1')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    simple_checkpoint = {
        'epoch': 5,
        'model_state_dict': model.state_dict(),
        'best_loss': 0.123,
    }
    
    simple_path = checkpoint_dir / 'simple.pth'
    torch.save(simple_checkpoint, simple_path)
    
    # Load with the new code (should work with weights_only=True)
    model2 = nn.Conv2d(3, 3, 3, padding=1)
    trainer2 = BaseTrainer(config, model2)
    
    try:
        epoch = trainer2.load_checkpoint(str(simple_path))
        assert epoch == 6, f"Expected epoch 6, got {epoch}"
        print("✓ Simple checkpoint (tensors only) loaded successfully with weights_only=True")
    except Exception as e:
        pytest.fail(f"Failed to load simple checkpoint: {e}")
    
    # Test 2: Create checkpoint with custom objects (should trigger fallback)
    class CustomConfig:
        def __init__(self):
            self.value = 42
    
    custom_checkpoint = {
        'epoch': 10,
        'model_state_dict': model.state_dict(),
        'best_loss': 0.456,
        'custom_config': CustomConfig(),  # Custom object that weights_only can't handle
    }
    
    custom_path = checkpoint_dir / 'custom.pth'
    torch.save(custom_checkpoint, custom_path)
    
    # Load with the new code (should fall back to weights_only=False)
    model3 = nn.Conv2d(3, 3, 3, padding=1)
    trainer3 = BaseTrainer(config, model3)
    
    try:
        epoch = trainer3.load_checkpoint(str(custom_path))
        assert epoch == 11, f"Expected epoch 11, got {epoch}"
        print("✓ Custom checkpoint (with objects) loaded successfully with fallback")
    except Exception as e:
        pytest.fail(f"Failed to load custom checkpoint with fallback: {e}")
    
    # Cleanup
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    
    print("✓ torch.load() weights_only fallback test passed")


def test_base_trainer_imports_successfully():
    """Test that BaseTrainer can be imported after AMP import changes."""
    try:
        from src.training.base_trainer import BaseTrainer
        assert BaseTrainer is not None
        print("✓ BaseTrainer imports successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import BaseTrainer: {e}")


if __name__ == '__main__':
    print("Running Phase 1 Critical Fixes regression tests...\n")
    
    test_pytorch_amp_import_compatibility()
    test_base_trainer_imports_successfully()
    test_torch_load_weights_only_fallback()
    
    print("\n✅ All Phase 1 tests passed!")
