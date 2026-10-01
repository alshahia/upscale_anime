"""
Integration tests for Stage 1 improvements.
Tests end-to-end workflows and component integration.
"""
import sys
import os
import torch
import pytest
from pathlib import Path
import tempfile
import yaml


class TestConfigIntegration:
    """Test configuration loading and validation."""
    
    def test_stage1_fast_config_loads(self):
        """Test that stage1_fast.yaml loads correctly."""
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'stage1_fast.yaml'
        assert config_path.exists()
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert config is not None
        assert 'training' in config
        assert 'stage1' in config['training']
    
    def test_stage1_balanced_config_loads(self):
        """Test that stage1_balanced.yaml loads correctly."""
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'stage1_balanced.yaml'
        assert config_path.exists()
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert config['training']['stage1']['aggregation_type'] == 'adaptive'
        assert config['training']['stage1']['ohem_enabled'] == True
    
    def test_stage1_anime_config_loads(self):
        """Test that stage1_anime.yaml loads correctly."""
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'stage1_anime.yaml'
        assert config_path.exists()
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert config['training']['stage1']['anime']['line_art_preservation'] == True
        assert config['data']['degradation']['anime_degradation'] == True
    
    def test_base_config_updated(self):
        """Test that base.yaml has new Stage 1 parameters."""
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'base.yaml'
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert 'stage1' in config['training']
        assert 'aggregation_type' in config['training']['stage1']
        assert 'warmup_epochs' in config['training']


class TestAggregationFactoryIntegration:
    """Test aggregation factory with different configs."""
    
    def test_factory_creates_all_types(self):
        """Test factory creates all aggregation types from configs."""
        from anime_sr.distillation.mtkd import create_aggregation_network
        
        base_config = {
            'training': {
                'stage1': {
                    'num_blocks': 4,
                    'embed_dim': 64,
                    'teachers': [{'name': 't1'}, {'name': 't2'}],
                }
            },
            'model': {'scale': 4}
        }
        
        for agg_type in ['simple', 'adaptive', 'multiscale']:
            config = base_config.copy()
            config['training'] = base_config['training'].copy()
            config['training']['stage1'] = base_config['training']['stage1'].copy()
            config['training']['stage1']['aggregation_type'] = agg_type
            
            model = create_aggregation_network(config)
            assert model is not None
            
            # Test forward pass
            teacher_outputs = [
                torch.randn(1, 3, 64, 64)
                for _ in range(2)
            ]
            
            with torch.no_grad():
                output = model(teacher_outputs)
            
            assert output.shape == (1, 3, 64, 64)


class TestSchedulerIntegration:
    """Test scheduler integration."""
    
    def test_warmup_cosine_scheduler(self):
        """Test WarmupCosineScheduler with optimizer."""
        from anime_sr.training.schedulers import WarmupCosineScheduler
        
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=5,
            total_epochs=20,
            min_lr=1e-7,
        )
        
        # Test warmup phase
        initial_lr = optimizer.param_groups[0]['lr']
        assert initial_lr < 0.001  # Should start lower due to warmup
        
        # Simulate training
        for epoch in range(10):
            scheduler.step()
        
        # After warmup, should be at or near target LR
        lr_after_warmup = optimizer.param_groups[0]['lr']
        assert lr_after_warmup > initial_lr
    
    def test_scheduler_state_dict(self):
        """Test scheduler state dict save/load."""
        from anime_sr.training.schedulers import WarmupCosineScheduler
        
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=5,
            total_epochs=20,
        )
        
        # Step a few times
        for _ in range(5):
            scheduler.step()
        
        # Save state
        state = scheduler.state_dict()
        
        # Create new scheduler and load
        optimizer2 = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler2 = WarmupCosineScheduler(
            optimizer2,
            warmup_epochs=5,
            total_epochs=20,
        )
        scheduler2.load_state_dict(state)
        
        assert scheduler2.last_epoch == scheduler.last_epoch


class TestLossIntegration:
    """Test loss function integration."""
    
    def test_combined_loss_with_model(self):
        """Test combined loss with a simple model."""
        from anime_sr.losses.combined_loss import Stage1CombinedLoss
        
        loss_fn = Stage1CombinedLoss()
        
        # Simple model
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 3, 3, padding=1),
        )
        
        # Forward pass
        x = torch.randn(2, 3, 32, 32)
        target = torch.randn(2, 3, 32, 32)
        
        output = model(x)
        
        # Compute loss
        total_loss, losses = loss_fn(output, target)
        
        # Backward
        total_loss.backward()
        
        # Check gradients
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


class TestOHEMIntegration:
    """Test OHEM integration with training loop."""
    
    def test_ohem_with_model_training(self):
        """Test OHEM in a training loop scenario."""
        from anime_sr.training.ohem import OHEMLoss
        
        model = torch.nn.Conv2d(3, 3, 3, padding=1)
        optimizer = torch.optim.Adam(model.parameters())
        ohem = OHEMLoss(torch.nn.L1Loss(), ratio=0.7)
        
        # Simulate training step
        x = torch.randn(4, 3, 32, 32)
        target = torch.randn(4, 3, 32, 32)
        
        optimizer.zero_grad()
        output = model(x)
        loss = ohem(output, target)
        loss.backward()
        optimizer.step()
        
        # Check stats
        stats = ohem.get_stats()
        assert stats['selected_samples'] > 0
        assert stats['total_samples'] == 4


class TestAnimeDegradationIntegration:
    """Test anime degradation with data pipeline."""
    
    def test_anime_degradation_with_dataloader(self):
        """Test anime degradation produces valid outputs."""
        from anime_sr.data.anime_degradation import apply_anime_degradation
        
        # Simulate HR batch
        hr_batch = torch.rand(2, 3, 128, 128)
        
        # Apply degradation
        lr_batch, hr_degraded = apply_anime_degradation(
            hr_batch,
            scale=4,
            enable_all=True,
        )
        
        # Check shapes
        assert lr_batch.shape == (2, 3, 32, 32)  # 4x downsampled
        assert hr_degraded.shape == hr_batch.shape
        
        # Check valid range
        assert lr_batch.min() >= 0 and lr_batch.max() <= 1
        assert hr_degraded.min() >= 0 and hr_degraded.max() <= 1


class TestCheckpointIntegration:
    """Test checkpoint save/load with new features."""
    
    def test_save_load_adaptive_aggregation(self):
        """Test saving and loading adaptive aggregation."""
        from anime_sr.distillation.mtkd import AdaptiveTeacherAggregation
        
        model = AdaptiveTeacherAggregation(
            num_teachers=3,
            embed_dim=64,
            num_blocks=4,
            scale=4,
        )
        
        # Save state
        state_dict = model.state_dict()
        
        # Create new model and load
        model2 = AdaptiveTeacherAggregation(
            num_teachers=3,
            embed_dim=64,
            num_blocks=4,
            scale=4,
        )
        model2.load_state_dict(state_dict)
        
        # Test both produce same output
        teacher_outputs = [
            torch.randn(1, 3, 64, 64)
            for _ in range(3)
        ]
        
        with torch.no_grad():
            output1 = model(teacher_outputs)
            output2 = model2(teacher_outputs)
        
        assert torch.allclose(output1, output2)


def test_end_to_end_stage1_forward():
    """Test complete Stage 1 forward pass."""
    from anime_sr.distillation.mtkd import create_aggregation_network
    from anime_sr.losses.combined_loss import Stage1CombinedLoss
    from anime_sr.training.ohem import compute_ohem_loss
    
    # Create aggregation network
    config = {
        'training': {
            'stage1': {
                'aggregation_type': 'simple',
                'num_blocks': 4,
                'embed_dim': 64,
                'teachers': [{'name': 't1'}, {'name': 't2'}, {'name': 't3'}],
                'loss_weights': {'l1': 0.4, 'wavelet': 0.3, 'gradient': 0.2, 'diversity': 0.1},
            }
        },
        'model': {'scale': 4}
    }
    
    model = create_aggregation_network(config)
    loss_fn = Stage1CombinedLoss()
    
    # Simulate training batch
    batch_size = 2
    hr_size = 64
    
    teacher_outputs = [
        torch.randn(batch_size, 3, hr_size, hr_size)
        for _ in range(3)
    ]
    target = torch.randn(batch_size, 3, hr_size, hr_size)
    
    # Forward
    output = model(teacher_outputs)
    
    # Loss
    total_loss, losses = loss_fn(output, target, teacher_outputs)
    
    # Backward
    total_loss.backward()
    
    # Verify
    assert output.shape == target.shape
    assert total_loss.item() >= 0
    assert not torch.isnan(total_loss)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
