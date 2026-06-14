"""
Tests for early stopping and convergence monitoring functionality.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import directly from modules to avoid utils dependencies
import importlib.util
spec = importlib.util.spec_from_file_location("convergence_monitor", Path(__file__).parent.parent / 'src' / 'training' / 'convergence_monitor.py')
convergence_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convergence_module)
ConvergenceMonitor = convergence_module.ConvergenceMonitor
MultiMetricConvergenceMonitor = convergence_module.MultiMetricConvergenceMonitor

spec2 = importlib.util.spec_from_file_location("callbacks", Path(__file__).parent.parent / 'src' / 'training' / 'callbacks.py')
callbacks_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(callbacks_module)
EarlyStoppingCallback = callbacks_module.EarlyStoppingCallback
CallbackList = callbacks_module.CallbackList


class TestConvergenceMonitor:
    """Test ConvergenceMonitor utility."""
    
    def test_initialization(self):
        monitor = ConvergenceMonitor(patience=10, min_delta=0.001)
        assert monitor.patience == 10
        assert monitor.min_delta == 0.001
        assert monitor.best_loss == float('inf')
    
    def test_improvement_detection(self):
        monitor = ConvergenceMonitor(patience=3, min_delta=0.01)
        
        # First update
        status = monitor.update(1.0, epoch=0)
        assert status['best_loss'] == 1.0
        assert status['counter'] == 0
        
        # Significant improvement
        status = monitor.update(0.9, epoch=1)
        assert status['best_loss'] == 0.9
        assert status['counter'] == 0
        
        # Small improvement (below min_delta)
        status = monitor.update(0.895, epoch=2)
        assert status['best_loss'] == 0.9  # No change
        assert status['counter'] == 1  # Incremented
    
    def test_convergence_detection(self):
        monitor = ConvergenceMonitor(patience=3, min_delta=0.01)
        
        # Train for a few epochs
        monitor.update(1.0, epoch=0)
        monitor.update(0.9, epoch=1)  # Significant improvement
        # Use values that are definitely less than 0.01 improvement
        # 0.9 - 0.895 = 0.005 < 0.01
        monitor.update(0.895, epoch=2)  # No significant improvement
        # 0.9 - 0.894 = 0.006 < 0.01
        monitor.update(0.894, epoch=3)  # No significant improvement
        
        assert not monitor.is_converged()  # counter=2, patience=3
        
        # 0.9 - 0.893 = 0.007 < 0.01
        monitor.update(0.893, epoch=4)  # No significant improvement
        assert monitor.is_converged()  # counter=3, patience=3
    
    def test_plateau_detection(self):
        monitor = ConvergenceMonitor(patience=10, window_size=5)
        
        # Add values that form a plateau
        for i in range(7):
            monitor.update(0.5 + np.random.normal(0, 0.001), epoch=i)
        
        assert monitor.is_plateaued(slope_threshold=1e-3)
    
    def test_divergence_detection(self):
        monitor = ConvergenceMonitor(divergence_patience=3)
        
        # Decreasing loss
        monitor.update(1.0, epoch=0)
        monitor.update(0.9, epoch=1)
        assert not monitor.is_diverging()
        
        # Increasing loss for 3 epochs
        monitor.update(0.95, epoch=2)
        monitor.update(1.0, epoch=3)
        monitor.update(1.1, epoch=4)
        assert monitor.is_diverging()
    
    def test_state_dict(self):
        monitor = ConvergenceMonitor(patience=5)
        
        # Update some values
        for i in range(10):
            monitor.update(1.0 - i * 0.05, epoch=i)
        
        # Save state
        state = monitor.state_dict()
        
        # Create new monitor and load state
        new_monitor = ConvergenceMonitor(patience=5)
        new_monitor.load_state_dict(state)
        
        assert new_monitor.best_loss == monitor.best_loss
        assert new_monitor.best_epoch == monitor.best_epoch
        assert new_monitor.counter == monitor.counter
        assert len(new_monitor.losses) == len(monitor.losses)
    
    def test_reset(self):
        monitor = ConvergenceMonitor()
        
        for i in range(5):
            monitor.update(1.0 - i * 0.1, epoch=i)
        
        monitor.reset()
        
        assert monitor.best_loss == float('inf')
        assert len(monitor.losses) == 0
        assert monitor.counter == 0


class TestMultiMetricConvergenceMonitor:
    """Test multi-metric convergence monitoring."""
    
    def test_multi_metric_any_mode(self):
        config = {
            'loss': {'patience': 3, 'min_delta': 0.01, 'mode': 'min'},
            'psnr': {'patience': 3, 'min_delta': 0.1, 'mode': 'max'},
        }
        monitor = MultiMetricConvergenceMonitor(config, mode='any')
        
        # Update with no improvement in either metric
        for i in range(4):
            monitor.update(epoch=i, loss=1.0, psnr=30.0)
        
        assert monitor.should_stop()  # Both should trigger
    
    def test_multi_metric_all_mode(self):
        config = {
            'loss': {'patience': 2, 'min_delta': 0.01, 'mode': 'min'},
            'psnr': {'patience': 5, 'min_delta': 0.1, 'mode': 'max'},
        }
        monitor = MultiMetricConvergenceMonitor(config, mode='all')
        
        # Loss converges but PSNR doesn't (keep improving)
        for i in range(3):
            monitor.update(epoch=i, loss=1.0, psnr=30.0 + i * 0.05)  # Small improvements
        
        assert not monitor.should_stop()  # PSNR still improving
        
        # Now PSNR also plateaus (no more improvements)
        # Need 5 epochs of no improvement for PSNR to converge
        for i in range(3, 10):
            monitor.update(epoch=i, loss=1.0, psnr=30.15)  # Same value, no improvement
        
        assert monitor.should_stop()  # Both converged


class TestEarlyStoppingCallback:
    """Test EarlyStoppingCallback."""
    
    def test_initialization(self):
        callback = EarlyStoppingCallback(
            monitor='val_loss',
            patience=10,
            min_delta=0.001
        )
        assert callback.monitor == 'val_loss'
        assert callback.patience == 10
        assert not callback.should_stop
    
    def test_patience_trigger(self):
        callback = EarlyStoppingCallback(patience=3, min_epochs=0)
        callback.on_train_begin()
        
        # Improving
        callback.on_epoch_end(0, {'val_loss': 1.0})
        callback.on_epoch_end(1, {'val_loss': 0.9})
        assert not callback.should_stop
        
        # Plateau for 3 epochs
        callback.on_epoch_end(2, {'val_loss': 0.9})
        callback.on_epoch_end(3, {'val_loss': 0.9})
        callback.on_epoch_end(4, {'val_loss': 0.9})
        
        assert callback.should_stop
        assert callback.stopped_epoch == 4
    
    def test_min_epochs(self):
        callback = EarlyStoppingCallback(patience=1, min_epochs=5)
        callback.on_train_begin()
        
        # Plateau before min_epochs
        for i in range(3):
            callback.on_epoch_end(i, {'val_loss': 1.0})
        
        assert not callback.should_stop  # Too early
        
        # Continue past min_epochs
        callback.on_epoch_end(3, {'val_loss': 1.0})
        callback.on_epoch_end(4, {'val_loss': 1.0})
        callback.on_epoch_end(5, {'val_loss': 1.0})
        
        assert callback.should_stop  # Now can trigger
    
    def test_divergence_trigger(self):
        callback = EarlyStoppingCallback(divergence_patience=3, min_epochs=0)
        callback.on_train_begin()
        
        # Decreasing loss
        callback.on_epoch_end(0, {'val_loss': 1.0})
        callback.on_epoch_end(1, {'val_loss': 0.9})
        callback.on_epoch_end(2, {'val_loss': 0.8})
        
        assert not callback.should_stop
        
        # Diverging
        callback.on_epoch_end(3, {'val_loss': 0.85})
        callback.on_epoch_end(4, {'val_loss': 0.9})
        callback.on_epoch_end(5, {'val_loss': 1.0})
        
        assert callback.should_stop
    
    def test_max_mode(self):
        callback = EarlyStoppingCallback(monitor='psnr', mode='max', patience=2, min_epochs=0)
        callback.on_train_begin()
        
        # Improving PSNR (higher is better)
        callback.on_epoch_end(0, {'psnr': 30.0})
        callback.on_epoch_end(1, {'psnr': 31.0})
        assert not callback.should_stop
        
        # Plateau
        callback.on_epoch_end(2, {'psnr': 31.0})
        callback.on_epoch_end(3, {'psnr': 31.0})
        
        assert callback.should_stop
    
    def test_plateau_detection(self):
        callback = EarlyStoppingCallback(
            patience=100,  # Long patience
            plateau_slope_threshold=1e-3,
            min_epochs=5
        )
        callback.on_train_begin()
        
        # Add values forming a flat plateau
        for i in range(10):
            callback.on_epoch_end(i, {'val_loss': 0.5 + np.random.normal(0, 0.001)})
        
        assert callback.should_stop  # Should trigger on plateau
    
    def test_state_dict(self):
        callback = EarlyStoppingCallback(patience=5)
        callback.on_train_begin()
        
        for i in range(10):
            callback.on_epoch_end(i, {'val_loss': 1.0 - i * 0.05})
        
        state = callback.state_dict()
        
        new_callback = EarlyStoppingCallback(patience=5)
        new_callback.load_state_dict(state)
        
        assert new_callback.best_value == callback.best_value
        assert new_callback.best_epoch == callback.best_epoch
        assert len(new_callback.values) == len(callback.values)
    
    def test_train_end_logs(self):
        callback = EarlyStoppingCallback()
        callback.on_train_begin()
        
        for i in range(5):
            callback.on_epoch_end(i, {'val_loss': 1.0 - i * 0.1})
        
        logs = {}
        callback.on_train_end(logs)
        
        assert 'best_epoch' in logs
        assert 'best_value' in logs
        assert logs['early_stopped'] == False


class TestIntegration:
    """Integration tests with callbacks and training."""
    
    def test_callback_list_should_stop(self):
        early_stopping = EarlyStoppingCallback(patience=2, min_epochs=0)
        callback_list = CallbackList([early_stopping])
        
        early_stopping.on_train_begin()
        
        assert not callback_list.should_stop()
        
        # Trigger early stopping
        early_stopping.on_epoch_end(0, {'val_loss': 1.0})
        early_stopping.on_epoch_end(1, {'val_loss': 1.0})
        early_stopping.on_epoch_end(2, {'val_loss': 1.0})
        
        assert callback_list.should_stop()
    
    def test_get_early_stopping_status(self):
        early_stopping = EarlyStoppingCallback(patience=3)
        callback_list = CallbackList([early_stopping])
        
        status = callback_list.get_early_stopping_status()
        assert status is not None
        assert 'patience' in status


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
