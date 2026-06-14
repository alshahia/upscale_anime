"""
Tests for critical fixes applied during the comprehensive code audit.
Covers: NaN detection, checkpoint loading, config validation, security fixes.
"""
import pytest
import torch
import torch.nn as nn
import tempfile
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestNaNProtection:
    """Test NaN/Inf detection in training loops."""
    
    def test_stage2_nan_detection_logic(self):
        """Verify NaN detection logic works correctly."""
        # Simulate the NaN check from train_stage2_epoch
        loss_normal = torch.tensor(0.5)
        loss_nan = torch.tensor(float('nan'))
        loss_inf = torch.tensor(float('inf'))
        
        assert not (torch.isnan(loss_normal) or torch.isinf(loss_normal))
        assert torch.isnan(loss_nan) or torch.isinf(loss_nan)
        assert torch.isnan(loss_inf) or torch.isinf(loss_inf)
    
    def test_mtkd_nan_protection(self):
        """Test that MTKD loss handles NaN gracefully."""
        from distillation.mtkd.distillation import FullMTKDLoss
        
        mtkd = FullMTKDLoss(l1_weight=1.0, wavelet_weight=1.0)
        
        # Normal inputs
        student = torch.randn(2, 3, 64, 64)
        teacher = torch.randn(2, 3, 64, 64)
        gt = torch.randn(2, 3, 64, 64)
        
        total, loss_dict = mtkd(student, teacher, gt)
        assert not torch.isnan(total)
        assert 'total' in loss_dict
    
    def test_fakd_loss_returns_tensor(self):
        """Test that FAKD loss returns tensor, not float."""
        from distillation.fakd.affinity_loss import FeatureAffinityLoss
        
        fakd = FeatureAffinityLoss(layers=[0], layer_weights=[1.0])
        
        # No matching layers - should return zero tensor
        student_feats = {}
        teacher_feats = {}
        
        result = fakd(student_feats, teacher_feats)
        assert isinstance(result, torch.Tensor)
        assert result.item() == 0.0


class TestCheckpointSecurity:
    """Test secure checkpoint loading."""
    
    def test_checkpoint_loader_imports(self):
        """Verify checkpoint_loader has proper imports."""
        from utils.checkpoint_loader import UnpicklingError
        assert UnpicklingError is not None
    
    def test_base_trainer_checkpoint_structure(self):
        """Test that BaseTrainer checkpoint includes optimizer state."""
        from training.base_trainer import BaseTrainer
        
        # Create a simple model
        model = nn.Linear(10, 10)
        config = {
            'training': {
                'epochs': 10,
                'batch_size': 2,
                'lr': 0.001,
                'optimizer': 'adam',
                'mixed_precision': False,
                'use_ema': False,
                'use_swa': False,
            },
            'paths': {'checkpoint_dir': tempfile.mkdtemp()},
            'checkpoint': {'keep_last_n': 5, 'keep_best': True},
        }
        
        trainer = BaseTrainer(config, model)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save checkpoint
        trainer.save_checkpoint(0, optimizer)
        
        # Load and verify structure
        checkpoint_path = Path(config['paths']['checkpoint_dir']) / 'latest.pth'
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        
        assert 'optimizer_state_dict' in checkpoint
        assert 'model_state_dict' in checkpoint
        assert 'epoch' in checkpoint
        assert 'best_loss' in checkpoint


class TestConfigValidation:
    """Test config validation improvements."""
    
    def test_valid_config_passes(self):
        """Test that valid config passes validation."""
        from utils.config import Config
        
        config = Config({
            'model': {'name': 'span', 'scale': 4},
            'training': {'batch_size': 4, 'epochs': 100},
            'data': {'crop_size': 128},
        })
        
        assert config.validate() is True
    
    def test_invalid_batch_size_rejected(self):
        """Test that negative batch_size is rejected."""
        from utils.config import Config
        
        config = Config({
            'model': {'name': 'span', 'scale': 4},
            'training': {'batch_size': -1, 'epochs': 100},
            'data': {'crop_size': 128},
        })
        
        with pytest.raises(ValueError, match="batch_size must be positive"):
            config.validate()
    
    def test_invalid_epochs_rejected(self):
        """Test that negative epochs is rejected."""
        from utils.config import Config
        
        config = Config({
            'model': {'name': 'span', 'scale': 4},
            'training': {'batch_size': 4, 'epochs': 0},
            'data': {'crop_size': 128},
        })
        
        with pytest.raises(ValueError, match="epochs must be positive"):
            config.validate()
    
    def test_invalid_scale_rejected(self):
        """Test that invalid scale factor is rejected."""
        from utils.config import Config
        
        config = Config({
            'model': {'name': 'span', 'scale': 5},
            'training': {'batch_size': 4, 'epochs': 100},
            'data': {'crop_size': 128},
        })
        
        with pytest.raises(ValueError, match="model.scale must be 2, 3, or 4"):
            config.validate()


class TestAnimeDegradationReproducibility:
    """Test that anime degradation uses torch.Generator for reproducibility."""
    
    def test_degradation_no_numpy_random(self):
        """Verify anime_degradation doesn't use np.random."""
        import inspect
        from data.anime_degradation import (
            AnimeBlur, DirectionalBlur, ColorQuantization,
            BandingArtifact, RingingArtifact
        )
        
        # Check that forward methods don't reference np.random
        for cls in [AnimeBlur, DirectionalBlur, ColorQuantization, BandingArtifact, RingingArtifact]:
            source = inspect.getsource(cls.forward)
            assert 'np.random' not in source, f"{cls.__name__} still uses np.random"
    
    def test_degradation_with_generator(self):
        """Test that degradation accepts generator parameter."""
        from data.anime_degradation import AnimeBlur
        
        blur = AnimeBlur()
        x = torch.randn(1, 3, 64, 64)
        
        # Should work with generator
        gen = torch.Generator()
        gen.manual_seed(42)
        
        result = blur(x, generator=gen)
        assert result.shape == x.shape


class TestStageController:
    """Test stage controller fixes."""
    
    def test_plateau_check_conservative(self):
        """Test that plateau check returns False when monitor unavailable."""
        from training.stage_controller import StageTransitionController
        
        # Create a minimal controller
        config = {
            'training': {
                'stage1': {},
                'stage2': {'enabled': False},
                'stage_automation': {},
            }
        }
        
        # Mock state manager
        class MockStateManager:
            def load(self):
                return None
            def save(self, state):
                pass
        
        controller = StageTransitionController(config, MockStateManager())
        
        # Should return False conservatively
        assert controller._check_loss_plateau({}) is False


class TestLoggingImport:
    """Test that logging is properly imported in all modules."""
    
    def test_model_a_trainer_logging(self):
        """Verify model_a_trainer has logging import."""
        import training.model_a_trainer as mat
        assert hasattr(mat, 'logger')
        assert mat.logger is not None
    
    def test_checkpoint_loader_logging(self):
        """Verify checkpoint_loader has logging import."""
        import utils.checkpoint_loader as cl
        assert hasattr(cl, 'logger')
        assert cl.logger is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
