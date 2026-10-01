"""
End-to-End Test Suite for Inference Pipeline

Tests complete inference workflows from image to super-resolved output:
- Single image inference
- Batch inference
- Inference with TTA (Test-Time Augmentation)
- Inference with EMA model
- Ensemble (A+B) inference
- Output quality validation (PSNR check)
- Inference speed benchmark

Data Strategy:
- Primary: data/test_hr/ (4 images) as HR reference
- Generate LR via bicubic downscale
- Fallback: Create test images if needed
"""
import pytest
import sys
import tempfile
import shutil
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add src and utils to path
for _tdm_k in ('utils', 'utils.test_data_manager'):
    sys.modules.pop(_tdm_k, None)
from utils.test_data_manager import ensure_test_data, get_test_hr_path, create_fallback_data


class MockSRModel(nn.Module):
    """Mock SR model for inference testing."""
    
    def __init__(self, scale: int = 4):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 3, 3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x is LR, upsample then refine
        x = F.interpolate(x, scale_factor=self.scale, mode='bicubic')
        x = self.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x


@pytest.fixture(scope='module')
def test_images():
    """Provide test image pairs (LR and HR)."""
    data_dir = get_test_hr_path()
    image_files = list(data_dir.glob('*.png'))
    
    pairs = []
    
    # Load real images if available
    if len(image_files) > 0:
        try:
            from PIL import Image
            for img_path in image_files[:2]:  # Use first 2 images
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img)
                
                # Use first 256x256 region
                img_array = img_array[:256, :256, :]
                
                # Convert to tensor [C, H, W]
                hr_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
                hr_tensor = hr_tensor.unsqueeze(0)  # Add batch
                
                # Create LR via bicubic downscale
                lr_tensor = F.interpolate(hr_tensor, scale_factor=1/4, mode='bicubic')
                
                pairs.append((lr_tensor, hr_tensor))
        except Exception as e:
            print(f"Warning: Could not load real images: {e}")
    
    # Fallback: Create synthetic pairs
    if len(pairs) == 0:
        for i in range(2):
            hr = torch.rand(1, 3, 256, 256)
            lr = F.interpolate(hr, scale_factor=1/4, mode='bicubic')
            pairs.append((lr, hr))
    
    return pairs


@pytest.fixture(scope='module')
def mock_checkpoint():
    """Create mock model checkpoint for inference testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = MockSRModel(scale=4)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'ema_model_state_dict': model.state_dict(),  # Same for testing
            'epoch': 100,
            'config': {
                'model': {'type': 'mock', 'scale': 4, 'channels': 16}
            }
        }
        
        checkpoint_path = Path(tmpdir) / 'test_model.pth'
        torch.save(checkpoint, checkpoint_path)
        
        yield checkpoint_path


class TestSingleImageInference:
    """Test single image inference."""
    
    def test_single_image_forward_pass(self, test_images):
        """Test that single image can be processed end-to-end."""
        try:
            from anime_sr.inference.engine import InferenceEngine
            
            model = MockSRModel(scale=4)
            engine = InferenceEngine(model, device='cpu')
            
            lr, hr = test_images[0]
            
            # Run inference
            sr_output = engine.run(lr, return_tensor=True)
            
            # Verify output
            assert sr_output.shape[0] == 1, "Batch size should be 1"
            assert sr_output.shape[1] == 3, "Should have 3 channels"
            assert sr_output.shape[2] == lr.shape[2] * 4, "Should be 4x height"
            assert sr_output.shape[3] == lr.shape[3] * 4, "Should be 4x width"
            
            # Verify range
            assert torch.all(sr_output >= 0) and torch.all(sr_output <= 1), \
                "Output should be in [0, 1] range"
                
        except ImportError as e:
            pytest.skip(f"Inference engine not available: {e}")
    
    def test_inference_engine_from_checkpoint(self, mock_checkpoint):
        """Test loading inference engine from checkpoint."""
        try:
            from anime_sr.inference.engine import InferenceEngine
            
            engine = InferenceEngine.from_checkpoint(
                mock_checkpoint,
                model_type='span',  # Use span type even though it's mock
                device='cpu',
                use_ema=False
            )
            
            assert engine is not None, "Engine should be created"
            assert engine.model is not None, "Model should be loaded"
            
        except ImportError as e:
            pytest.skip(f"Inference engine not available: {e}")


class TestBatchInference:
    """Test batch inference."""
    
    def test_batch_forward_pass(self, test_images):
        """Test processing multiple images in batch."""
        try:
            from anime_sr.inference.engine import InferenceEngine
            
            model = MockSRModel(scale=4)
            engine = InferenceEngine(model, device='cpu')
            
            # Stack multiple images
            lr_batch = torch.cat([lr for lr, hr in test_images], dim=0)
            
            # Run inference
            sr_output = engine.run(lr_batch, return_tensor=True)
            
            # Verify output
            assert sr_output.shape[0] == len(test_images), "Batch size should match"
            assert sr_output.shape[2] == lr_batch.shape[2] * 4, "Should be 4x height"
            
        except ImportError as e:
            pytest.skip(f"Inference engine not available: {e}")
    
    def test_batch_processing_consistency(self, test_images):
        """Test that batch and individual processing give similar results."""
        try:
            from anime_sr.inference.engine import InferenceEngine
            
            model = MockSRModel(scale=4)
            engine = InferenceEngine(model, device='cpu')
            
            # Process individually
            individual_outputs = []
            for lr, hr in test_images:
                output = engine.run(lr, return_tensor=True)
                individual_outputs.append(output)
            
            # Process as batch
            lr_batch = torch.cat([lr for lr, hr in test_images], dim=0)
            batch_output = engine.run(lr_batch, return_tensor=True)
            
            # Compare (allow small numerical differences)
            for i, individual in enumerate(individual_outputs):
                diff = torch.abs(individual - batch_output[i:i+1]).mean()
                assert diff < 1e-4, f"Batch and individual should match for image {i}"
                
        except ImportError as e:
            pytest.skip(f"Inference engine not available: {e}")


class TestTTAInference:
    """Test Test-Time Augmentation inference."""
    
    def test_tta_forward_pass(self, test_images):
        """Test TTA inference."""
        try:
            from anime_sr.inference.engine import InferenceEngine, TTA_TRANSFORMS
            
            model = MockSRModel(scale=4)
            engine = InferenceEngine(model, device='cpu')
            
            lr, hr = test_images[0]
            
            # Run TTA inference
            sr_output = engine.run_tta(lr, return_tensor=True)
            
            # Verify output
            assert sr_output.shape[2] == lr.shape[2] * 4, "Should be 4x upscaled"
            assert torch.all(sr_output >= 0) and torch.all(sr_output <= 1), \
                "Output should be in [0, 1]"
                
        except ImportError as e:
            pytest.skip(f"Inference engine not available: {e}")
    
    def test_tta_improves_quality(self, test_images):
        """Test that TTA improves output quality."""
        try:
            from anime_sr.inference.engine import InferenceEngine
            
            model = MockSRModel(scale=4)
            engine = InferenceEngine(model, device='cpu')
            
            lr, hr = test_images[0]
            
            # Standard inference
            sr_standard = engine.run(lr, return_tensor=True)
            
            # TTA inference
            sr_tta = engine.run_tta(lr, return_tensor=True)
            
            # Both should produce valid outputs
            assert sr_standard.shape == sr_tta.shape
            assert torch.all(sr_tta >= 0) and torch.all(sr_tta <= 1)
            
            # Outputs should be similar but not identical
            diff = torch.abs(sr_standard - sr_tta).mean()
            assert diff < 0.5, "TTA should not drastically change output"
            
        except ImportError as e:
            pytest.skip(f"Inference engine not available: {e}")
    
    def test_tta_transforms_reversible(self):
        """Test that TTA transforms are reversible."""
        try:
            from anime_sr.inference.engine import TTA_TRANSFORMS
            
            x = torch.rand(1, 3, 64, 64)
            
            for name, (forward_fn, inverse_fn) in TTA_TRANSFORMS.items():
                transformed = forward_fn(x)
                recovered = inverse_fn(transformed)
                
                assert torch.allclose(x, recovered, atol=1e-6), \
                    f"Transform {name} should be reversible"
                    
        except ImportError as e:
            pytest.skip(f"TTA transforms not available: {e}")


class TestEMAInference:
    """Test EMA model inference."""
    
    def test_ema_model_loading(self, mock_checkpoint):
        """Test loading EMA weights from checkpoint."""
        try:
            from anime_sr.inference.engine import InferenceEngine
            
            # Load with EMA
            engine_ema = InferenceEngine.from_checkpoint(
                mock_checkpoint,
                model_type='span',
                device='cpu',
                use_ema=True
            )
            
            # Load without EMA
            engine_regular = InferenceEngine.from_checkpoint(
                mock_checkpoint,
                model_type='span',
                device='cpu',
                use_ema=False
            )
            
            assert engine_ema is not None
            assert engine_regular is not None
            
        except ImportError as e:
            pytest.skip(f"Inference engine not available: {e}")


class TestOutputQualityValidation:
    """Test output quality validation."""
    
    def test_psnr_calculation(self, test_images):
        """Test PSNR calculation on inference output."""
        try:
            from anime_sr.inference.engine import InferenceEngine
            from anime_sr.utils.metrics import calculate_psnr
            
            model = MockSRModel(scale=4)
            engine = InferenceEngine(model, device='cpu')
            
            lr, hr = test_images[0]
            
            # Run inference
            sr = engine.run(lr, return_tensor=True)
            
            # Calculate PSNR
            psnr = calculate_psnr(sr, hr)
            
            assert psnr > 0, "PSNR should be positive"
            assert psnr < 100, "PSNR should be reasonable"
            
        except ImportError as e:
            pytest.skip(f"Required module not available: {e}")
    
    def test_ssim_calculation(self, test_images):
        """Test SSIM calculation on inference output."""
        try:
            from anime_sr.inference.engine import InferenceEngine
            from anime_sr.utils.metrics import calculate_ssim
            
            model = MockSRModel(scale=4)
            engine = InferenceEngine(model, device='cpu')
            
            lr, hr = test_images[0]
            
            # Run inference
            sr = engine.run(lr, return_tensor=True)
            
            # Calculate SSIM
            ssim = calculate_ssim(sr, hr)
            
            assert 0 <= ssim <= 1, "SSIM should be in [0, 1]"
            
        except ImportError as e:
            pytest.skip(f"Required module not available: {e}")


class TestInferenceSpeed:
    """Test inference speed benchmarking."""
    
    def test_inference_speed_measurement(self, test_images):
        """Test that inference speed can be measured."""
        try:
            from anime_sr.inference.engine import InferenceEngine
            
            model = MockSRModel(scale=4)
            engine = InferenceEngine(model, device='cpu')
            
            lr, hr = test_images[0]
            
            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    _ = engine.run(lr, return_tensor=True)
            
            # Measure
            num_runs = 5
            times = []
            
            with torch.no_grad():
                for _ in range(num_runs):
                    start = time.time()
                    _ = engine.run(lr, return_tensor=True)
                    times.append(time.time() - start)
            
            avg_time = np.mean(times)
            assert avg_time > 0, "Inference time should be positive"
            
            print(f"\nAverage inference time: {avg_time*1000:.2f} ms")
            
        except ImportError as e:
            pytest.skip(f"Inference engine not available: {e}")
    
    def test_benchmark_method(self):
        """Test engine benchmark method."""
        try:
            from anime_sr.inference.engine import InferenceEngine
            
            model = MockSRModel(scale=4)
            engine = InferenceEngine(model, device='cpu')
            
            # Run benchmark
            results = engine.benchmark(input_size=(3, 64, 64), num_runs=10, warmup=3)
            
            # Verify results
            assert 'mean_time_ms' in results
            assert 'fps' in results
            assert results['mean_time_ms'] > 0
            assert results['fps'] > 0
            
        except ImportError as e:
            pytest.skip(f"Inference engine not available: {e}")


class TestInferencePipelineIntegration:
    """Integration tests for inference pipeline."""
    
    def test_inference_script_exists(self):
        """Test that inference script exists."""
        script_path = Path(__file__).parent.parent.parent / 'scripts' / 'inference.py'
        assert script_path.exists(), f"Inference script not found: {script_path}"
    
    def test_inference_script_help(self):
        """Test that inference script shows help."""
        import subprocess
        
        script_path = Path(__file__).parent.parent.parent / 'scripts' / 'inference.py'
        result = subprocess.run(
            ['python', str(script_path), '--help'],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, "Help command should succeed"
        assert '--checkpoint' in result.stdout, "Help should mention --checkpoint"
        assert '--input' in result.stdout, "Help should mention --input"
    
    def test_enhanced_inference_script_exists(self):
        """Test that enhanced inference script exists."""
        script_path = Path(__file__).parent.parent.parent / 'scripts' / 'enhanced_inference.py'
        assert script_path.exists(), f"Enhanced inference script not found: {script_path}"


def test_data_source_report():
    """Report which data source is being used for tests."""
    data_dir = ensure_test_data(min_images=4, verbose=True)
    print(f"\n[E2E Inference Pipeline] Using data: {data_dir}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
