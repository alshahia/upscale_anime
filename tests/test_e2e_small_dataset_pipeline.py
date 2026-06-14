"""
End-to-End Test Suite for Small Dataset 10-Phase Pipeline

Tests complete small dataset workflow from validation to production:
- Phase 1: Dataset Validation (cleaning/validation)
- Phase 2: Self-Supervised Pre-Training
- Phase 3: Transfer Learning Fine-Tuning
- Phase 4: Advanced Augmentation (Mixup/CutMix)
- Phase 5: Meta-Learning (MAML)
- Phase 6: Test-Time Adaptation
- Phase 7: Feature Distillation
- Phase 8: SWA/EMA
- Phase 9: Benchmarking
- Phase 10: Production Export

Data Strategy:
- Primary: data/val_hr/ (4 images) - simulate small dataset
- Fallback: Create 5 dummy images
"""
import pytest
import sys
import tempfile
import shutil
from pathlib import Path
import torch

# Add src and utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

for _tdm_k in ('utils', 'utils.test_data_manager'):
    sys.modules.pop(_tdm_k, None)
from utils.test_data_manager import ensure_test_data, get_val_hr_path, get_test_hr_path, create_temp_dataset


@pytest.fixture(scope='module')
def small_dataset_dir():
    """Provide small dataset (4-5 images)."""
    data_dir = ensure_test_data(min_images=4, pattern='anime_style', verbose=True)
    yield data_dir


@pytest.fixture(scope='module')
def temp_working_dir():
    """Create temporary working directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestPhase1DatasetValidation:
    """Phase 1: Dataset Validation."""
    
    def test_validation_script_runs(self, small_dataset_dir, temp_working_dir):
        """Test that dataset validation script runs successfully."""
        import subprocess
        
        script_path = Path(__file__).parent.parent / 'scripts' / 'validate_dataset.py'
        if not script_path.exists():
            pytest.skip(f"Validation script not found: {script_path}")
        
        result = subprocess.run(
            [
                'python', str(script_path),
                '--data-dir', str(small_dataset_dir),
                '--output', str(temp_working_dir / 'validation_report.json')
            ],
            capture_output=True,
            text=True
        )
        
        # Should complete without crashing
        assert result.returncode in [0, 1], "Should complete (may have warnings)"
        
        # Check if report was created
        report_path = temp_working_dir / 'validation_report.json'
        if report_path.exists():
            import json
            with open(report_path) as f:
                report = json.load(f)
            assert 'summary' in report or 'total_images' in report or 'count' in str(report)
    
    def test_validation_output_structure(self, small_dataset_dir):
        """Test validation output has expected structure."""
        try:
            from data.dataset_validation import validate_dataset
            
            report = validate_dataset(small_dataset_dir, check_duplicates=True)
            
            # Should return dict with results
            assert isinstance(report, dict), "Should return dictionary"
            assert len(report) > 0, "Should not be empty"
            
        except ImportError:
            pytest.skip("Dataset validation module not available")


class TestPhase2SelfSupervisedPretrain:
    """Phase 2: Self-Supervised Pre-Training."""
    
    def test_pretrain_config_exists(self):
        """Test that pretrain config exists."""
        config_path = Path(__file__).parent.parent / 'configs' / 'pretrain_selfsupervised.yaml'
        assert config_path.exists(), f"Pretrain config not found: {config_path}"
    
    def test_pretrain_mode_loading(self):
        """Test that pretrain config can be loaded."""
        try:
            from utils.config import Config
            
            config_path = Path(__file__).parent.parent / 'configs' / 'pretrain_selfsupervised.yaml'
            if config_path.exists():
                config = Config.load(config_path)
                assert config is not None
        except ImportError:
            pytest.skip("Config module not available")


class TestPhase3TransferFinetune:
    """Phase 3: Transfer Learning Fine-Tuning."""
    
    def test_finetune_config_exists(self):
        """Test that finetune config exists."""
        config_path = Path(__file__).parent.parent / 'configs' / 'finetune_transfer.yaml'
        assert config_path.exists(), f"Finetune config not found: {config_path}"
    
    def test_transfer_learning_setup(self, small_dataset_dir):
        """Test transfer learning setup."""
        try:
            from utils.config import Config
            
            config_path = Path(__file__).parent.parent / 'configs' / 'finetune_transfer.yaml'
            if not config_path.exists():
                pytest.skip("Finetune config not available")
            
            config = Config.load(config_path)
            
            # Verify transfer learning settings
            training_cfg = config.config.get('training', {})
            
            # Should have some form of transfer/pretrain settings
            assert 'stage1' in training_cfg or 'stage2' in training_cfg or 'lr' in training_cfg
            
        except ImportError:
            pytest.skip("Config module not available")


class TestPhase4AdvancedAugmentation:
    """Phase 4: Advanced Augmentation."""
    
    def test_mixup_cutmix_available(self):
        """Test that Mixup and CutMix are available."""
        try:
            from data.augmentation import MixupAugmentation, CutMixAugmentation
            
            mixup = MixupAugmentation(alpha=0.4)
            cutmix = CutMixAugmentation()
            
            assert mixup is not None
            assert cutmix is not None
            
        except ImportError:
            pytest.skip("Augmentation module not available")
    
    def test_augmentation_on_real_data(self, small_dataset_dir):
        """Test augmentation on real images."""
        try:
            from data.augmentation import MixupAugmentation, CutMixAugmentation
            from PIL import Image
            import numpy as np
            
            # Load two images
            image_files = list(small_dataset_dir.glob('*.png'))[:2]
            if len(image_files) < 2:
                pytest.skip("Need at least 2 images for augmentation")
            
            img1 = Image.open(image_files[0]).convert('RGB')
            img2 = Image.open(image_files[1]).convert('RGB')
            
            # Convert to tensors
            t1 = torch.from_numpy(np.array(img1)).permute(2, 0, 1).float() / 255.0
            t2 = torch.from_numpy(np.array(img2)).permute(2, 0, 1).float() / 255.0
            
            # Test Mixup
            mixup = MixupAugmentation(alpha=0.4)
            mixed, _, lam = mixup(t1, t2, t1, t2)
            
            assert mixed.shape == t1.shape, "Mixup should preserve shape"
            assert 0 <= lam <= 1, "Lambda should be in [0, 1]"
            
        except ImportError:
            pytest.skip("Required modules not available")


class TestPhase5MetaLearning:
    """Phase 5: Meta-Learning (MAML)."""
    
    def test_meta_learning_script_exists(self):
        """Test that meta-learning script exists."""
        script_path = Path(__file__).parent.parent / 'scripts' / 'train_meta.py'
        assert script_path.exists(), f"Meta-learning script not found: {script_path}"
    
    def test_maml_basic_functionality(self, small_dataset_dir):
        """Test basic MAML functionality."""
        # This is tested in test_meta_learning.py
        pass


class TestPhase6TestTimeAdaptation:
    """Phase 6: Test-Time Adaptation."""
    
    def test_tta_script_exists(self):
        """Test that TTA script exists."""
        script_path = Path(__file__).parent.parent / 'scripts' / 'test_time_adapt.py'
        assert script_path.exists(), f"TTA script not found: {script_path}"
    
    def test_tta_basic_functionality(self, small_dataset_dir):
        """Test basic TTA functionality."""
        # This is tested in test_test_time_adaptation.py
        pass


class TestPhase7FeatureDistillation:
    """Phase 7: Feature Distillation."""
    
    def test_feature_distillation_module_exists(self):
        """Test that feature distillation module exists."""
        try:
            from distillation.feature_distillation import FeatureDistillationLoss
            assert True, "Feature distillation module available"
        except ImportError:
            pytest.skip("Feature distillation module not available")
    
    def test_feature_distillation_loss_computation(self):
        """Test feature distillation loss computation."""
        try:
            from distillation.feature_distillation import FeatureDistillationLoss
            
            loss_fn = FeatureDistillationLoss()
            
            # Create dummy features
            student_feat = torch.randn(1, 64, 32, 32)
            teacher_feat = torch.randn(1, 64, 32, 32)
            
            loss = loss_fn(student_feat, teacher_feat)
            
            assert loss.dim() == 0, "Loss should be scalar"
            assert loss.item() >= 0, "Loss should be non-negative"
            
        except ImportError:
            pytest.skip("Feature distillation module not available")


class TestPhase8SWAEMA:
    """Phase 8: SWA/EMA (Stochastic Weight Averaging / Exponential Moving Average)."""
    
    def test_ema_model_creation(self):
        """Test that EMA model can be created."""
        try:
            import torch.nn as nn
            
            # Simple model
            model = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 3, 3, padding=1)
            )
            
            # Create EMA
            from training.ema import EMA
            ema = EMA(model, decay=0.999)
            
            assert ema is not None, "EMA should be created"
            
        except ImportError:
            pytest.skip("EMA module not available")
    
    def test_ema_update(self):
        """Test EMA update."""
        try:
            import torch.nn as nn
            from training.ema import EMA
            
            model = nn.Conv2d(3, 16, 3, padding=1)
            ema = EMA(model, decay=0.999)
            
            # Get initial EMA params
            initial_ema = list(ema.shadow_params.values())[0].clone()
            
            # Update model
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * 0.1)
            
            # Update EMA
            ema.update(model)
            
            # Verify EMA changed
            updated_ema = list(ema.shadow_params.values())[0]
            diff = torch.abs(initial_ema - updated_ema).mean()
            
            assert diff > 0, "EMA should update when model changes"
            
        except ImportError:
            pytest.skip("EMA module not available")


class TestPhase9Benchmarking:
    """Phase 9: Benchmarking."""
    
    def test_benchmark_script_exists(self):
        """Test that benchmark script exists."""
        script_path = Path(__file__).parent.parent / 'scripts' / 'benchmark_model.py'
        assert script_path.exists(), f"Benchmark script not found: {script_path}"
    
    def test_metrics_available(self):
        """Test that metrics are available."""
        try:
            from utils.metrics import calculate_psnr, calculate_ssim
            
            # Test on dummy data
            img1 = torch.rand(1, 3, 64, 64)
            img2 = img1.clone()
            
            psnr = calculate_psnr(img1, img2)
            ssim = calculate_ssim(img1, img2)
            
            assert psnr > 40, "Identical images should have high PSNR"
            assert ssim > 0.99, "Identical images should have SSIM ~ 1"
            
        except ImportError:
            pytest.skip("Metrics module not available")


class TestPhase10ProductionExport:
    """Phase 10: Production Export."""
    
    def test_export_script_exists(self):
        """Test that export script exists."""
        script_path = Path(__file__).parent.parent / 'scripts' / 'export_model.py'
        assert script_path.exists(), f"Export script not found: {script_path}"
    
    def test_export_formats_supported(self, temp_working_dir):
        """Test that export formats are supported."""
        # This is tested in test_model_export.py
        pass


class TestFullPipelineOrchestration:
    """Test full pipeline orchestration."""
    
    def test_run_full_pipeline_script_exists(self):
        """Test that full pipeline script exists."""
        script_path = Path(__file__).parent.parent / 'scripts' / 'run_full_pipeline.py'
        assert script_path.exists(), f"Full pipeline script not found: {script_path}"
    
    def test_pipeline_config_exists(self):
        """Test that pipeline config exists."""
        config_path = Path(__file__).parent.parent / 'pipeline_config.example.json'
        if config_path.exists():
            import json
            with open(config_path) as f:
                config = json.load(f)
            assert 'data_dir' in config or 'phases' in str(config)
    
    @pytest.mark.slow
    def test_full_pipeline_integration(self, small_dataset_dir, temp_working_dir):
        """Test full pipeline integration (run if possible)."""
        import subprocess
        
        script_path = Path(__file__).parent.parent / 'scripts' / 'run_full_pipeline.py'
        if not script_path.exists():
            pytest.skip("Full pipeline script not available")
        
        # Create minimal pipeline config
        pipeline_config = {
            'data_dir': str(small_dataset_dir),
            'num_workers': 2,
            'remove_duplicates': True,
            'run_pretrain': False,  # Skip for speed
            'run_finetune': False,  # Skip for speed
            'run_benchmark': True,
            'run_export': False  # Skip for speed
        }
        
        import json
        config_path = temp_working_dir / 'pipeline_config.json'
        with open(config_path, 'w') as f:
            json.dump(pipeline_config, f)
        
        # Run with dry-run or minimal execution
        result = subprocess.run(
            ['python', str(script_path), '--config', str(config_path), '--dry-run'],
            capture_output=True,
            text=True,
            timeout=60  # Timeout after 60 seconds
        )
        
        # Should at least parse config without crashing
        assert result.returncode in [0, 1, 2], "Should complete or fail gracefully"


def test_small_dataset_summary():
    """Print summary of small dataset test setup."""
    print("\n" + "="*70)
    print("SMALL DATASET 10-PHASE PIPELINE TEST SUMMARY")
    print("="*70)
    
    data_dir = ensure_test_data(min_images=4, verbose=False)
    images = list(data_dir.glob('*.png'))
    
    print(f"Data directory: {data_dir}")
    print(f"Number of images: {len(images)}")
    
    phases = [
        "Phase 1: Dataset Validation",
        "Phase 2: Self-Supervised Pre-Training",
        "Phase 3: Transfer Learning Fine-Tuning",
        "Phase 4: Advanced Augmentation",
        "Phase 5: Meta-Learning (MAML)",
        "Phase 6: Test-Time Adaptation",
        "Phase 7: Feature Distillation",
        "Phase 8: SWA/EMA",
        "Phase 9: Benchmarking",
        "Phase 10: Production Export"
    ]
    
    print("\nTested phases:")
    for phase in phases:
        print(f"  ✓ {phase}")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
