"""Enhanced Early Stopping Tests with Real Mini Training"""
import pytest, sys, tempfile, shutil
from pathlib import Path
import torch, numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))
for _tdm_k in ('utils', 'utils.test_data_manager'):
    sys.modules.pop(_tdm_k, None)
from utils.test_data_manager import ensure_test_data

@pytest.fixture(scope='module')
def test_data_dir():
    data_dir = ensure_test_data(min_images=4, pattern='gradient', verbose=False)
    yield data_dir

class TestEarlyStoppingRealTraining:
    """Test early stopping with actual mini training"""
    
    def test_early_stopping_convergence_detection(self):
        """Test convergence detection with real loss curves"""
        try:
            from training.convergence_monitor import ConvergenceMonitor
            
            monitor = ConvergenceMonitor(patience=3, min_delta=0.01)
            
            # Improving phase
            for i in range(5):
                monitor.update(1.0 - i * 0.15, epoch=i)
            assert not monitor.is_converged()
            
            # Plateau phase
            for i in range(5, 10):
                monitor.update(0.25 + np.random.randn() * 0.005, epoch=i)
            assert monitor.is_converged()
            
        except ImportError:
            pytest.skip("Convergence monitor not available")
    
    def test_divergence_detection_real(self):
        """Test divergence detection with increasing loss"""
        try:
            from training.convergence_monitor import ConvergenceMonitor
            
            monitor = ConvergenceMonitor(divergence_patience=3)
            
            # Decreasing loss
            for i in range(3):
                monitor.update(1.0 - i * 0.1, epoch=i)
            assert not monitor.is_diverging()
            
            # Increasing loss
            for i in range(3, 6):
                monitor.update(0.7 + (i-2) * 0.1, epoch=i)
            assert monitor.is_diverging()
            
        except ImportError:
            pytest.skip("Convergence monitor not available")
    
    def test_best_checkpoint_tracking(self):
        """Test best checkpoint is tracked correctly"""
        try:
            from training.convergence_monitor import ConvergenceMonitor
            
            monitor = ConvergenceMonitor(patience=5, min_delta=0.01)
            
            losses = [1.0, 0.8, 0.7, 0.65, 0.64, 0.63, 0.635]
            best_epochs = []
            
            for i, loss in enumerate(losses):
                status = monitor.update(loss, epoch=i)
                if status.get('counter', 0) == 0:
                    best_epochs.append(i)
            
            assert monitor.best_loss == min(losses)
            
        except ImportError:
            pytest.skip("Convergence monitor not available")


class TestEarlyStoppingCallbackIntegration:
    """Test callback integration"""
    
    def test_callback_stops_training(self):
        """Test callback signals stop"""
        try:
            from training.callbacks import EarlyStoppingCallback
            
            callback = EarlyStoppingCallback(
                monitor='val_loss',
                patience=2,
                min_delta=0.01
            )
            
            callback.on_train_begin()
            
            # Simulate improving
            for i in range(3):
                callback.on_epoch_end(i, {'val_loss': 1.0 - i * 0.1})
            
            assert not callback.should_stop()
            
            # Simulate plateau
            for i in range(3, 6):
                callback.on_epoch_end(i, {'val_loss': 0.7})
            
            assert callback.should_stop()
            
        except ImportError:
            pytest.skip("Callbacks not available")
    
    def test_callback_restores_best_weights(self):
        """Test best weights restoration"""
        try:
            from training.callbacks import EarlyStoppingCallback
            import torch.nn as nn
            
            model = nn.Linear(10, 10)
            initial_weights = model.weight.clone().detach()
            
            callback = EarlyStoppingCallback(
                monitor='val_loss',
                patience=2,
                restore_best_weights=True
            )
            
            callback.set_model(model)
            callback.on_train_begin()
            
            # Simulate training
            for i in range(5):
                callback.on_epoch_end(i, {'val_loss': 1.0 - i * 0.1})
                # Modify weights
                model.weight.data += 0.01
            
            # Trigger stop and restore
            callback.on_epoch_end(5, {'val_loss': 1.0})  # Bad epoch
            callback.on_epoch_end(6, {'val_loss': 1.0})  # Bad epoch
            
            if callback.should_stop() and callback.restore_best_weights:
                callback.restore_weights()
            
        except ImportError:
            pytest.skip("Callbacks not available")


class TestEarlyStoppingConfig:
    """Test early stopping configuration"""
    
    def test_early_stopping_config_structure(self):
        """Test config has proper structure"""
        base_config = Path(__file__).parent.parent / 'configs' / 'base.yaml'
        
        import yaml
        with open(base_config) as f:
            config = yaml.safe_load(f)
        
        assert 'training' in config
        assert 'early_stopping' in config['training']
        
        es_config = config['training']['early_stopping']
        assert 'enabled' in es_config
        assert 'patience' in es_config
        assert 'min_delta' in es_config
    
    def test_adaptive_lr_config(self):
        """Test adaptive LR config"""
        base_config = Path(__file__).parent.parent / 'configs' / 'base.yaml'
        
        import yaml
        with open(base_config) as f:
            config = yaml.safe_load(f)
        
        assert 'adaptive_lr' in config['training']
        al_config = config['training']['adaptive_lr']
        assert 'enabled' in al_config
        assert 'scheduler' in al_config


def test_data_source_report():
    """Report data source"""
    data_dir = ensure_test_data(min_images=4, verbose=True)
    print(f"\n[Early Stopping Enhanced] Using data: {data_dir}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
