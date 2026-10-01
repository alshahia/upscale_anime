"""
Tests for adaptive learning rate functionality.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path

# Import directly from modules to avoid utils dependencies
import importlib.util

spec = importlib.util.spec_from_file_location("schedulers", Path(__file__).parent.parent.parent.parent / 'src' / 'anime_sr' / 'training' / 'schedulers.py')
schedulers_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schedulers_module)
WarmupCosinePlateauScheduler = schedulers_module.WarmupCosinePlateauScheduler
PlateauLRScheduler = schedulers_module.PlateauLRScheduler

spec2 = importlib.util.spec_from_file_location("lr_history", Path(__file__).parent.parent.parent.parent / 'src' / 'anime_sr' / 'training' / 'lr_history.py')
lr_history_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(lr_history_module)
LRHistory = lr_history_module.LRHistory

class MockOptimizer:
    """Mock optimizer for testing."""
    
    def __init__(self, lr=0.001):
        self.param_groups = [{'lr': lr}]
    
    def step(self):
        pass

class TestLRHistory:
    """Test LRHistory tracking."""
    
    def test_initialization(self):
        history = LRHistory()
        assert len(history.epochs) == 0
        assert history.best_loss == float('inf')
        assert history.best_lr is None
    
    def test_record(self):
        history = LRHistory()
        history.record(epoch=0, lr=0.001, loss=1.0)
        history.record(epoch=1, lr=0.001, loss=0.9)
        
        assert len(history.epochs) == 2
        assert history.lr_values == [0.001, 0.001]
        assert history.best_loss == 0.9
        assert history.best_lr == 0.001
    
    def test_record_reduction(self):
        history = LRHistory()
        history.record_reduction(epoch=5, old_lr=0.001, new_lr=0.0005, reason="plateau")
        
        assert len(history.reduction_events) == 1
        event = history.reduction_events[0]
        assert event['epoch'] == 5
        assert event['factor'] == 0.5
    
    def test_get_stats(self):
        history = LRHistory()
        history.record(epoch=0, lr=0.001, loss=1.0)
        history.record(epoch=1, lr=0.0005, loss=0.9)
        history.record_reduction(epoch=1, old_lr=0.001, new_lr=0.0005)
        
        stats = history.get_stats()
        assert stats['total_epochs'] == 2
        assert stats['num_reductions'] == 1
        assert stats['final_lr'] == 0.0005
    
    def test_recommend_lr(self):
        history = LRHistory()
        history.record(epoch=0, lr=0.001, loss=1.0)
        history.record(epoch=1, lr=0.0005, loss=0.9)  # Best loss
        history.record(epoch=2, lr=0.00025, loss=0.95)
        
        # Best mode should return LR with best loss
        assert history.recommend_lr(mode="best") == 0.0005
        
        # Final mode should return last LR
        assert history.recommend_lr(mode="final") == 0.00025
    
    def test_get_lr_efficiency(self):
        history = LRHistory()
        history.record(epoch=0, lr=0.001, loss=1.0)
        history.record(epoch=1, lr=0.0005, loss=0.95)  # LR changed, 0.05 improvement
        history.record(epoch=2, lr=0.0005, loss=0.90)  # Same LR, 0.05 improvement
        
        efficiency = history.get_lr_efficiency()
        assert 'per_lr_improvements' in efficiency
        # We get 2 unique LRs: 0.001 (from epoch 0->1 transition) and 0.0005 (from epoch 1->2)
        assert len(efficiency['per_lr_improvements']) == 2

class TestWarmupCosinePlateauScheduler:
    """Test WarmupCosinePlateauScheduler."""
    
    def test_initialization(self):
        pytest.skip("Requires real PyTorch optimizer")
    
    def test_warmup_phase(self):
        pytest.skip("Requires real PyTorch optimizer")
    
    def test_plateau_detection_and_reduction(self):
        pytest.skip("Requires real PyTorch optimizer")
    
    def test_max_reductions_limit(self):
        pytest.skip("Requires real PyTorch optimizer")
    
    def test_state_dict(self):
        pytest.skip("Requires real PyTorch optimizer")

class TestPlateauLRScheduler:
    """Test PlateauLRScheduler (wrapper around ReduceLROnPlateau)."""
    
    def test_initialization(self):
        pytest.skip("Requires real PyTorch optimizer")
    
    def test_warmup_phase(self):
        pytest.skip("Requires real PyTorch optimizer")
    
    def test_plateau_phase(self):
        pytest.skip("Requires real PyTorch optimizer")

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
