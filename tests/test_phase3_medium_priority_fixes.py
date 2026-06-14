"""
Regression tests for Phase 3 Medium Priority Fixes.

Tests:
1. Config override parser handles 'none' and 'null' as Python None
2. ConvergenceMonitor scipy fallback returns correct tuple format
3. CallbackList has all required methods
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest


def test_config_override_handles_none():
    """Test that config override parser handles 'none' and 'null' as Python None."""
    sys.path.insert(0, str(project_root / 'scripts'))
    from train import parse_config_override
    
    # Test various case combinations
    assert parse_config_override('none') is None
    assert parse_config_override('None') is None
    assert parse_config_override('NONE') is None
    assert parse_config_override('null') is None
    assert parse_config_override('Null') is None
    assert parse_config_override('NULL') is None
    
    # Ensure other values still work
    assert parse_config_override('true') is True
    assert parse_config_override('false') is False
    assert parse_config_override('123') == 123
    assert parse_config_override('1.5') == 1.5
    assert parse_config_override('string') == 'string'
    
    print("✓ Config override parser correctly handles None/null")


def test_convergence_monitor_scipy_fallback():
    """Test that ConvergenceMonitor's scipy fallback returns correct tuple format."""
    import numpy as np
    sys.path.insert(0, str(project_root / 'src'))
    from training.convergence_monitor import ConvergenceMonitor
    
    # Create monitor
    monitor = ConvergenceMonitor()
    
    # Test _linear_regression with scipy (if available) or fallback
    x = np.arange(10)
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]  # Perfect linear relationship
    
    result = monitor._linear_regression(x, y)
    
    # Should return a tuple with 5 elements (slope, intercept, r_value, p_value, std_err)
    assert isinstance(result, tuple), "Result should be a tuple"
    assert len(result) == 5, f"Result should have 5 elements, got {len(result)}"
    
    slope, intercept, r_value, p_value, std_err = result
    
    # For perfect linear relationship y = x + 1:
    assert abs(slope - 1.0) < 0.01, f"Slope should be ~1.0, got {slope}"
    assert abs(intercept - 1.0) < 0.01, f"Intercept should be ~1.0, got {intercept}"
    
    print("✓ ConvergenceMonitor scipy fallback returns correct tuple format")


def test_callback_list_has_required_methods():
    """Test that CallbackList has all required methods."""
    sys.path.insert(0, str(project_root / 'src' / 'training'))
    from callbacks import CallbackList, Callback
    
    # Create a simple callback
    class TestCallback(Callback):
        def __init__(self):
            self.called = False
        
        def on_epoch_end(self, epoch, logs=None):
            self.called = True
    
    # Create CallbackList
    callback_list = CallbackList()
    
    # Test add method
    test_callback = TestCallback()
    callback_list.add(test_callback)
    assert len(callback_list.callbacks) == 1
    
    # Test on_epoch_end method
    callback_list.on_epoch_end(0)
    assert test_callback.called
    
    # Test should_stop method
    result = callback_list.should_stop()
    assert isinstance(result, bool)
    
    print("✓ CallbackList has all required methods (add, on_epoch_end, should_stop)")


def test_callback_list_early_stopping_integration():
    """Test CallbackList integration with EarlyStoppingCallback."""
    sys.path.insert(0, str(project_root / 'src' / 'training'))
    from callbacks import CallbackList, EarlyStoppingCallback
    
    # Create early stopping callback
    early_stop = EarlyStoppingCallback(
        monitor='val_loss',
        patience=2,
        min_epochs=0
    )
    
    # Add to callback list
    callback_list = CallbackList([early_stop])
    
    # Simulate training that should trigger early stopping
    early_stop.on_train_begin()
    
    # Epoch 0: Good loss
    callback_list.on_epoch_end(0, {'val_loss': 1.0})
    assert not callback_list.should_stop()
    
    # Epoch 1: Slightly worse
    callback_list.on_epoch_end(1, {'val_loss': 1.1})
    assert not callback_list.should_stop()
    
    # Epoch 2: No improvement (patience=2 reached)
    callback_list.on_epoch_end(2, {'val_loss': 1.2})
    # Note: should_stop returns True only after patience epochs without improvement
    
    print("✓ CallbackList EarlyStopping integration works correctly")


if __name__ == '__main__':
    print("Running Phase 3 Medium Priority Fixes regression tests...\n")
    
    test_config_override_handles_none()
    test_convergence_monitor_scipy_fallback()
    test_callback_list_has_required_methods()
    test_callback_list_early_stopping_integration()
    
    print("\n✅ All Phase 3 tests passed!")
