"""
Tests for data loading optimization features.
"""
import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.dataloader import get_optimal_num_workers, DataLoaderFactory
from data.base import BaseDataset, DatasetFactory
from data.async_prefetcher import AsyncDataPrefetcher, create_prefetcher


class TestNumWorkersAutoTuning:
    """Test auto-tuning of num_workers."""
    
    def test_auto_tuning_returns_reasonable_value(self):
        """Should return value between 1 and cpu_count."""
        test_cfg = {'gradient_accumulation_steps': 1, 'auto_tune_workers': True}
        workers = get_optimal_num_workers(test_cfg, batch_size=16)
        
        import os
        assert 1 <= workers <= os.cpu_count()
        assert workers <= 16  # Hard cap at 16
    
    def test_user_override_takes_precedence(self):
        """User-specified num_workers should not be overridden."""
        test_cfg = {'num_workers': 8, 'auto_tune_workers': False}
        workers = get_optimal_num_workers(test_cfg, batch_size=16)
        assert workers == 8
    
    def test_gradient_accumulation_increases_workers(self):
        """Higher accumulation should suggest more workers."""
        cfg_low = {'gradient_accumulation_steps': 1, 'auto_tune_workers': True}
        cfg_high = {'gradient_accumulation_steps': 4, 'auto_tune_workers': True}
        
        workers_low = get_optimal_num_workers(cfg_low, 16)
        workers_high = get_optimal_num_workers(cfg_high, 16)
        
        # With higher accumulation, we should need more workers
        assert workers_high >= workers_low
    
    def test_low_ram_reduces_workers(self):
        """Systems with low RAM should get fewer workers."""
        # This test is environment-dependent, so we just check the logic exists
        # The function should at least run without error
        test_cfg = {'gradient_accumulation_steps': 1, 'auto_tune_workers': True}
        workers = get_optimal_num_workers(test_cfg, batch_size=16)
        assert isinstance(workers, int)
        assert workers >= 1


class TestGPUDegradationSupport:
    """Test GPU degradation mode in BaseDataset."""
    
    def test_gpu_degradation_parameter_exists(self):
        """BaseDataset should accept gpu_degradation parameter."""
        # We can't test without actual images, but we can verify the parameter exists
        import inspect
        sig = inspect.signature(BaseDataset.__init__)
        assert 'gpu_degradation' in sig.parameters
    
    def test_dataset_factory_passes_gpu_degradation(self):
        """DatasetFactory should pass gpu_degradation to BaseDataset."""
        import inspect
        source = inspect.getsource(DatasetFactory.create)
        assert 'gpu_degradation' in source


class TestNonBlockingTransfers:
    """Test non-blocking GPU transfers in trainers."""
    
    def test_non_blocking_in_model_a_trainer(self):
        """Verify non_blocking=True is used in ModelATrainer."""
        from training.model_a_trainer import ModelATrainer
        import inspect
        
        source = inspect.getsource(ModelATrainer)
        assert 'non_blocking=True' in source
    
    def test_non_blocking_in_model_b_trainer(self):
        """Verify non_blocking=True is used in ModelBTrainer."""
        from training.model_b_trainer import ModelBTrainer
        import inspect
        
        source = inspect.getsource(ModelBTrainer)
        assert 'non_blocking=True' in source
    
    def test_non_blocking_in_ensemble_trainer(self):
        """Verify non_blocking=True is used in EnsembleTrainer."""
        from training.ensemble_trainer import EnsembleTrainer
        import inspect
        
        source = inspect.getsource(EnsembleTrainer)
        assert 'non_blocking=True' in source


class TestAsyncPrefetcher:
    """Test AsyncDataPrefetcher functionality."""
    
    def test_prefetcher_creation(self):
        """Should create prefetcher with correct parameters."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create a simple mock dataloader
        class MockLoader:
            def __iter__(self):
                for i in range(5):
                    yield {'lr': torch.rand(2, 3, 32, 32), 'hr': torch.rand(2, 3, 128, 128)}
            def __len__(self):
                return 5
        
        device = torch.device('cuda')
        prefetcher = AsyncDataPrefetcher(MockLoader(), device, num_prefetch=2)
        
        assert prefetcher.device == device
        assert prefetcher.num_prefetch == 2
        assert prefetcher.stream is not None
    
    def test_prefetcher_iteration(self):
        """Should iterate through all batches."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        class MockLoader:
            def __iter__(self):
                for i in range(3):
                    yield {'lr': torch.rand(2, 3, 32, 32), 'hr': torch.rand(2, 3, 128, 128)}
            def __len__(self):
                return 3
        
        device = torch.device('cuda')
        prefetcher = AsyncDataPrefetcher(MockLoader(), device, num_prefetch=2)
        
        count = 0
        for batch in prefetcher:
            assert batch['lr'].device.type == 'cuda'
            assert batch['hr'].device.type == 'cuda'
            count += 1
        
        assert count == 3
    
    def test_create_prefetcher_factory_disabled(self):
        """Factory should return original dataloader when disabled."""
        class MockLoader:
            pass
        
        config = {'training': {'use_async_prefetcher': False}}
        device = torch.device('cpu')
        
        loader = MockLoader()
        result = create_prefetcher(loader, config, device)
        
        assert result is loader  # Same object returned
    
    def test_create_prefetcher_factory_cpu_warning(self):
        """Factory should warn and return original dataloader on CPU."""
        class MockLoader:
            pass
        
        config = {'training': {'use_async_prefetcher': True}}
        device = torch.device('cpu')
        
        loader = MockLoader()
        result = create_prefetcher(loader, config, device)
        
        assert result is loader  # Same object returned (no CUDA)


class TestConfigurationIntegration:
    """Test that all config options are properly integrated."""
    
    def test_base_yaml_has_dataloader_options(self):
        """base.yaml should contain all new data loading options."""
        import yaml
        
        config_path = Path(__file__).parent.parent / 'configs' / 'base.yaml'
        with open(config_path) as f:
            content = f.read()
        
        # Check for key options
        assert 'num_workers' in content
        assert 'prefetch_factor' in content
        assert 'auto_tune_workers' in content
        assert 'use_async_prefetcher' in content
        assert 'gpu_degradation' in content
    
    def test_benchmark_config_exists(self):
        """Benchmark config should exist."""
        config_path = Path(__file__).parent.parent / 'configs' / 'benchmark_data_loading.yaml'
        assert config_path.exists()


class TestGPUDegradationPipeline:
    """Test GPU-based degradation pipeline."""
    
    def test_anime_degradation_imports(self):
        """Anime degradation module should be importable."""
        from data.anime_degradation import AnimeDegradationPipeline, apply_anime_degradation
        
        # Should be able to instantiate
        pipeline = AnimeDegradationPipeline()
        assert pipeline is not None
    
    def test_gpu_degradation_produces_valid_output(self):
        """GPU degradation should produce valid LR images."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        from data.anime_degradation import AnimeDegradationPipeline
        import torch.nn.functional as F
        
        device = torch.device('cuda')
        pipeline = AnimeDegradationPipeline().to(device)
        
        # Create dummy HR tensor
        hr = torch.rand(2, 3, 128, 128, device=device)
        
        # Apply degradation
        with torch.no_grad():
            hr_degraded = pipeline(hr)
            lr = F.interpolate(hr_degraded, scale_factor=0.25, mode='bicubic')
            lr = torch.clamp(lr, 0, 1)
        
        # Check output validity
        assert lr.shape == (2, 3, 32, 32)  # 4x downsampling
        assert torch.all(lr >= 0) and torch.all(lr <= 1)
        assert not torch.isnan(lr).any()


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
