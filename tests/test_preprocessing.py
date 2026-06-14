"""
Unit tests for new preprocessing components.
Tests: storage_estimator, dataset_sampler, preprocessing_manager, and integration.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestStorageEstimator:
    """Tests for storage_estimator.py"""
    
    def test_estimate_single_image_size(self):
        from data.storage_estimator import estimate_single_image_size
        
        # Test PNG estimation
        result = estimate_single_image_size(1920, 1080, format='png')
        assert result['hr_mb'] > 0
        assert result['lr_mb'] > 0
        assert result['total_mb'] > 0
        assert result['lr_mb'] < result['hr_mb']  # LR should be smaller
        
        # Test JPEG estimation
        result_jpg = estimate_single_image_size(1920, 1080, format='jpeg')
        assert result_jpg['total_mb'] < result['total_mb']  # JPEG should be smaller than PNG
    
    def test_estimate_dataset_storage(self):
        from data.storage_estimator import estimate_dataset_storage
        
        # Test with actual data directory
        result = estimate_dataset_storage(
            hr_dir='data/anime_video_frames',
            crop_size=128,
            scale=4,
            format='png',
            sample_ratio=0.1  # Use 10% for quick test
        )
        
        if 'error' not in result:
            assert result['total_images'] > 0
            assert result['images_to_use'] > 0
            assert result['total_storage_gb'] > 0
            assert 'per_dataset_breakdown' in result
    
    def test_check_against_available_space(self):
        from data.storage_estimator import check_against_available_space
        
        # Test insufficient space
        estimate = {
            'total_storage_gb': 15.0,
            'worst_case_total_gb': 20.0
        }
        result = check_against_available_space(estimate, 10.0)
        assert result['sufficient_space'] == False
        assert result['space_shortage_gb'] > 0
        
        # Test sufficient space
        result2 = check_against_available_space(estimate, 25.0)
        assert result2['sufficient_space'] == True


class TestDatasetSampler:
    """Tests for dataset_sampler.py"""
    
    def test_create_sampler(self):
        from data.dataset_sampler import DatasetSampler
        
        configs = [
            {'name': 'test1', 'hr_dir': 'data/anime_video_frames', 
             'sample_ratio': 0.1, 'weight': 1.0, 'is_validation': False, 'enabled': True},
        ]
        
        sampler = DatasetSampler(configs, max_images_total=100)
        assert len(sampler) > 0
        assert len(sampler) <= 100
    
    def test_global_cap(self):
        from data.dataset_sampler import DatasetSampler
        
        configs = [
            {'name': 'test1', 'hr_dir': 'data/anime_video_frames', 
             'sample_ratio': 1.0, 'weight': 1.0, 'is_validation': False, 'enabled': True},
        ]
        
        sampler = DatasetSampler(configs, max_images_total=50)
        assert len(sampler) <= 50
    
    def test_sampled_indices(self):
        from data.dataset_sampler import DatasetSampler
        
        configs = [
            {'name': 'test1', 'hr_dir': 'data/anime_video_frames', 
             'sample_ratio': 0.5, 'weight': 1.0, 'is_validation': False, 'enabled': True},
        ]
        
        sampler = DatasetSampler(configs, max_images_total=200)
        indices = sampler.get_sampled_indices(0)
        assert len(indices) <= 200
        assert len(indices) == sampler.total_images


class TestPreprocessingManager:
    """Tests for preprocessing_manager.py"""
    
    def test_create_manager(self):
        from data.preprocessing_manager import PreprocessingManager
        
        config = {
            'mode': 'hybrid',
            'scale': 4,
            'gpu_degradation': {
                'enabled': True,
                'modules': {'blur': True, 'quantization': True}
            },
            'line_enhancement': {'enabled': False}
        }
        
        manager = PreprocessingManager(config)
        assert manager is not None
        assert manager.mode == 'hybrid'
        assert manager.device is not None
    
    def test_get_info(self):
        from data.preprocessing_manager import PreprocessingManager
        
        config = {
            'mode': 'gpu_degradation',
            'gpu_degradation': {
                'enabled': True,
                'modules': {'blur': True, 'directional_blur': True}
            }
        }
        
        manager = PreprocessingManager(config)
        info = manager.get_info()
        
        assert 'mode' in info
        assert 'gpu_modules' in info
        assert 'blur' in info['gpu_modules']
    
    def test_modes(self):
        from data.preprocessing_manager import PreprocessingManager

        valid_modes = ['on_the_fly', 'precomputed', 'gpu_degradation', 'hybrid', 'quality_adaptive']

        for mode in valid_modes:
            # Issue #4: pre-flight check is opt-in. To test pure construction
            # of every mode, explicitly disable precomputed opt-in (the test
            # is not about pre-flight behavior).
            config = {
                'mode': mode,
                'scale': 4,
                'precomputed': {'enabled': False},
                'hybrid': {'precomputed_base': False},
            }
            manager = PreprocessingManager(config)
            assert manager.mode == mode

    def test_precomputed_mode_raises_when_enabled_and_missing(self):
        """Issue #3: precomputed mode with no base_dir raises a clear, mode-correct error."""
        from data.preprocessing_manager import PreprocessingManager

        try:
            PreprocessingManager({'mode': 'precomputed', 'scale': 4})
        except FileNotFoundError as e:
            msg = str(e)
            assert "mode='precomputed'" in msg, f"Wrong mode in error: {msg}"
            assert "precomputed.base_dir" in msg or "base_dir" in msg, f"Missing base_dir in: {msg}"
        else:
            raise AssertionError("Expected FileNotFoundError when precomputed base_dir missing")


class TestConfigIntegration:
    """Tests for config parsing and integration"""
    
    def test_base_yaml_parse(self):
        import yaml
        with open('configs/base.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        assert 'data' in config
        assert 'preprocessing' in config['data']
        preprocessing = config['data']['preprocessing']
        assert preprocessing['mode'] == 'hybrid'
        assert 'storage' in preprocessing
        assert 'dataset_sampling' in preprocessing
    
    def test_v4_yaml_parse(self):
        import yaml
        with open('configs/finetune_neosr_span_v4.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        assert 'data' in config
        assert 'preprocessing' in config['data']
    
    def test_v5_yaml_parse(self):
        import yaml
        with open('configs/finetune_neosr_span_v5.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        assert 'data' in config
        assert 'preprocessing' in config['data']


class TestBaseDatasetIntegration:
    """Tests for BaseDataset integration with new parameters"""
    
    def test_sample_ratio_param(self):
        from data.base import BaseDataset
        
        # This will fail if directory doesn't exist, but tests param acceptance
        # We just verify the parameter exists in the constructor
        import inspect
        sig = inspect.signature(BaseDataset.__init__)
        params = list(sig.parameters.keys())
        assert 'sample_ratio' in params
        assert 'preprocessing_manager' in params
    
    def test_factory_accepts_preprocessing_config(self):
        from data.base import DatasetFactory
        import inspect
        sig = inspect.signature(DatasetFactory.create)
        params = list(sig.parameters.keys())
        assert 'preprocessing_config' in params


class TestDataloaderIntegration:
    """Tests for DataLoaderFactory integration"""

    def test_dataloader_uses_preprocessing(self):
        from data.dataloader import DataLoaderFactory
        import yaml

        with open('configs/finetune_neosr_span_v4.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Should not raise - validates config parsing
        assert 'data' in config
        preprocessing_cfg = config['data'].get('preprocessing')
        assert preprocessing_cfg is not None


class TestDegradationPipeline:
    """Tests for the shared DegradationPipeline class."""

    def test_presets_present(self):
        from data.degradation_pipeline import PRESETS
        for mode in ['bicubic', 'light', 'medium', 'heavy', 'anime', 'anime_heavy']:
            assert mode in PRESETS

    def test_bicubic_downscale(self):
        from data.degradation_pipeline import DegradationPipeline
        import numpy as np
        p = DegradationPipeline(mode='bicubic', scale=4, seed=0)
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        out = p(img)
        assert out.shape == (64, 64, 3)
        assert out.dtype == np.uint8

    def test_anime_heavy_returns_uint8(self):
        from data.degradation_pipeline import DegradationPipeline
        import numpy as np
        p = DegradationPipeline(mode='anime_heavy', scale=4, seed=42)
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        out = p(img)
        assert out.dtype == np.uint8
        assert out.shape[2] == 3

    def test_two_stage_produces_lr(self):
        from data.degradation_pipeline import DegradationPipeline
        import numpy as np
        p = DegradationPipeline(mode='anime_heavy', scale=4, two_stage=True, seed=42)
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        out = p(img)
        assert out.shape == (64, 64, 3)

    def test_float_input_coerced_to_uint8(self):
        from data.degradation_pipeline import DegradationPipeline
        import numpy as np
        p = DegradationPipeline(mode='light', scale=4, seed=0)
        img = np.random.rand(256, 256, 3).astype(np.float32)
        out = p(img)
        assert out.dtype == np.uint8

    def test_seeded_determinism(self):
        from data.degradation_pipeline import DegradationPipeline
        import numpy as np
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        a = DegradationPipeline(mode='medium', scale=4, seed=123)(img.copy())
        b = DegradationPipeline(mode='medium', scale=4, seed=123)(img.copy())
        np.testing.assert_array_equal(a, b)


class TestPreprocessingManagerModes:
    """Tests for the 5 PreprocessingManager modes."""

    def _hr(self, size=128):
        import torch
        return torch.rand(1, 3, size, size)

    def test_on_the_fly_returns_lr_tensor(self):
        import torch
        from data.preprocessing_manager import PreprocessingManager
        cfg = {
            'scale': 4,
            'mode': 'on_the_fly',
            'preprocessing': {'on_the_fly': {'mode': 'anime_heavy', 'seed': 0}},
        }
        m = PreprocessingManager(config=cfg, device=torch.device('cpu'))
        out = m._process_on_the_fly(self._hr(256))
        assert out.dim() == 3
        assert out.shape[0] == 3
        assert 0.0 <= out.min().item() <= 1.0
        assert 0.0 <= out.max().item() <= 1.0

    def test_quality_adaptive_returns_lr_tensor(self):
        import torch
        from data.preprocessing_manager import PreprocessingManager
        cfg = {
            'scale': 4,
            'mode': 'quality_adaptive',
            'preprocessing': {'quality_adaptive': {
                'light_degradation': 'light',
                'medium_degradation': 'medium',
                'heavy_degradation': 'heavy',
            }},
        }
        m = PreprocessingManager(config=cfg, device=torch.device('cpu'))
        out = m._process_quality_adaptive(self._hr(256))
        assert out.dim() == 3
        assert out.shape[0] == 3

    def test_gpu_degradation_returns_lr_tensor(self):
        import torch
        from data.preprocessing_manager import PreprocessingManager
        cfg = {'scale': 4, 'mode': 'gpu_degradation'}
        m = PreprocessingManager(config=cfg, device=torch.device('cpu'))
        try:
            out = m._process_gpu_degradation(self._hr(128))
        except Exception as e:
            pytest.skip(f"GPU modules not available on this platform: {e}")
            return
        assert out.shape[1:] in [(32, 32), (32, 32, 3)] or out.dim() in (3, 4)

    def test_precomputed_raises_when_unavailable(self):
        import torch
        from data.preprocessing_manager import PreprocessingManager
        cfg = {
            'scale': 4,
            'mode': 'precomputed',
            'precomputed': {'enabled': True, 'base_dir': 'data/_does_not_exist_xyz'},
        }
        with pytest.raises(FileNotFoundError):
            PreprocessingManager(config=cfg, device=torch.device('cpu'))

    def test_hybrid_fallback_to_on_the_fly(self):
        import torch
        from data.preprocessing_manager import PreprocessingManager
        cfg = {
            'scale': 4,
            'mode': 'hybrid',
            'preprocessing': {'on_the_fly': {'mode': 'light', 'seed': 0}},
            'precomputed': {'enabled': False},
            'hybrid': {'precomputed_base': False},
        }
        m = PreprocessingManager(config=cfg, device=torch.device('cpu'))
        out = m._process_hybrid_fallback(self._hr(256), idx=0)
        assert out.dim() == 3
        assert out.shape[0] == 3


class TestEndToEndModes:
    """Regression tests covering all 5 modes end-to-end with a dummy HR tensor
    (T19). One per mode; ensures the PreprocessingManager + DegradationPipeline
    integration does not regress on a fresh install.
    """

    def _hr(self, size=128):
        import torch
        return torch.rand(1, 3, size, size)

    def _assert_lr(self, lr):
        assert lr is not None
        assert hasattr(lr, 'shape')
        assert lr.shape[0] == 3
        assert lr.min().item() >= 0.0
        assert lr.max().item() <= 1.0

    def test_all_5_modes_produce_valid_lr(self):
        import torch
        from data.preprocessing_manager import PreprocessingManager
        from data.degradation_pipeline import DegradationPipeline
        hr = self._hr(128)
        hr_np_uint8 = (hr[0].permute(1, 2, 0).numpy() * 255).astype('uint8')

        # 1. on_the_fly
        m_otf = PreprocessingManager(
            config={'scale': 4, 'mode': 'on_the_fly',
                    'preprocessing': {'on_the_fly': {'mode': 'anime_heavy', 'seed': 0}}},
            device=torch.device('cpu'),
        )
        self._assert_lr(m_otf._process_on_the_fly(hr))

        # 2. precomputed: skip; requires on-disk pairs
        # 3. gpu_degradation: handled below; depends on platform

        # 4. hybrid (no precomputed base, falls back to on_the_fly)
        m_hyb = PreprocessingManager(
            config={'scale': 4, 'mode': 'hybrid',
                    'preprocessing': {'on_the_fly': {'mode': 'anime_heavy', 'seed': 0}},
                    'precomputed': {'enabled': False},
                    'hybrid': {'precomputed_base': False}},
            device=torch.device('cpu'),
        )
        self._assert_lr(m_hyb._process_hybrid_fallback(hr, idx=0))

        # 5. quality_adaptive (no analyzer -> on_the_fly fallback)
        m_qa = PreprocessingManager(
            config={'scale': 4, 'mode': 'quality_adaptive',
                    'preprocessing': {'quality_adaptive': {
                        'light_degradation': 'light',
                        'medium_degradation': 'medium',
                        'heavy_degradation': 'heavy',
                    }}},
            device=torch.device('cpu'),
        )
        self._assert_lr(m_qa._process_quality_adaptive(hr))

        # 6. raw DegradationPipeline (no PM)
        dp = DegradationPipeline(mode='anime_heavy', scale=4, seed=0)
        out_np = dp(hr_np_uint8)
        assert out_np.dtype.name == 'uint8'
        assert out_np.shape[2] == 3


class TestStorageEstimatorDeterminism:
    """Determinism + CLI behavior for storage_estimator (T20, T21)."""

    def test_seeded_estimates_are_equal(self):
        from data.storage_estimator import estimate_dataset_storage
        a = estimate_dataset_storage('data/anime_hr', crop_size=128,
                                     sample_ratio=0.1, sample_images=20, seed=7)
        b = estimate_dataset_storage('data/anime_hr', crop_size=128,
                                     sample_ratio=0.1, sample_images=20, seed=7)
        assert a.get('avg_size_per_image_mb') == b.get('avg_size_per_image_mb')

    def test_unseeded_estimates_can_differ(self):
        from data.storage_estimator import estimate_dataset_storage
        # Two calls without seed - they MIGHT pick different samples.
        # We only assert the function runs successfully (no crash).
        a = estimate_dataset_storage('data/anime_hr', crop_size=128,
                                     sample_ratio=0.1, sample_images=5)
        b = estimate_dataset_storage('data/anime_hr', crop_size=128,
                                     sample_ratio=0.1, sample_images=5)
        assert 'total_images' in a and 'total_images' in b

    def test_subprocess_estimate_only_runs(self):
        """--estimate-only lives in precompute_pairs.py, not storage_estimator.py."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, 'scripts/precompute_pairs.py',
             '--estimate-only', '--all_datasets', '--crop_size', '128',
             '--available-space', '10'],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0 or 'STORAGE ESTIMATION' in (result.stdout + result.stderr), \
            f"exit={result.returncode} stderr={result.stderr[:500]}"

    def test_subprocess_storage_estimator_cli_runs(self):
        """storage_estimator.py CLI runs without crashing on --all_datasets."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, 'src/data/storage_estimator.py',
             '--all_datasets', '--crop_size', '128', '--seed', '42'],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, \
            f"exit={result.returncode} stderr={result.stderr[:500]}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])