"""
Test Suite for Model Benchmarking (Phase 9 of Small Dataset Techniques)

Tests comprehensive model evaluation including:
- PSNR (Peak Signal-to-Noise Ratio) calculation
- SSIM (Structural Similarity Index) calculation
- LPIPS (Learned Perceptual Image Patch Similarity)
- DISTS (Deep Image Structure and Texture Similarity)
- Inference speed benchmarking (FPS)
- Memory usage tracking (VRAM)
- Benchmark report generation (JSON/HTML)

Data Strategy:
- Primary: data/test_hr/ (4 real images) as HR reference
- Generate LR versions via bicubic downscale for testing
- Fallback: Use dummy data if real data missing
"""
import pytest
import sys
import tempfile
import time
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# Add src and utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

for _tdm_k in ('utils', 'utils.test_data_manager'):
    sys.modules.pop(_tdm_k, None)
from utils.test_data_manager import ensure_test_data, get_test_hr_path, create_fallback_data


class MockSRModel(nn.Module):
    """Mock SR model for benchmarking tests."""
    
    def __init__(self, scale: int = 4):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(16, 3, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=scale, mode='bicubic', align_corners=False)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.conv2(x)
        return torch.clamp(x, 0, 1)


def create_lr_from_hr(hr_tensor: torch.Tensor, scale: int = 4) -> torch.Tensor:
    """Create LR version via bicubic downscale."""
    import torch.nn.functional as F
    
    # Downscale
    lr = F.interpolate(
        hr_tensor,
        scale_factor=1/scale,
        mode='bicubic',
        align_corners=False,
        antialias=True
    )
    return lr


@pytest.fixture(scope='module')
def test_images():
    """Provide test image pairs (LR and HR)."""
    # Get or create test data
    data_dir = get_test_hr_path()
    
    # Load real images if available
    image_files = list(data_dir.glob('*.png'))
    
    if len(image_files) >= 2:
        try:
            from PIL import Image
            images = []
            for img_path in image_files[:2]:  # Use first 2 images
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img)
                # Convert to tensor [C, H, W]
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
                images.append(img_tensor)
            
            # Create LR/HR pairs
            pairs = []
            for hr in images:
                if hr.dim() == 3:
                    hr = hr.unsqueeze(0)  # Add batch dimension
                lr = create_lr_from_hr(hr, scale=4)
                pairs.append((lr, hr))
            
            return pairs
        except Exception as e:
            print(f"Warning: Could not load real images: {e}")
    
    # Fallback: Create synthetic images
    pairs = []
    for i in range(2):
        hr = torch.rand(1, 3, 256, 256)
        lr = create_lr_from_hr(hr, scale=4)
        pairs.append((lr, hr))
    
    return pairs


@pytest.fixture(scope='module')
def mock_model():
    """Provide mock SR model."""
    return MockSRModel(scale=4)


class TestPSNRCalculation:
    """Test PSNR metric calculation."""
    
    def test_psnr_perfect_reconstruction(self, test_images):
        """Test PSNR with identical images (should be infinite)."""
        try:
            from utils.metrics import calculate_psnr
            
            for lr, hr in test_images:
                psnr = calculate_psnr(hr, hr)  # Same image
                assert psnr > 50, "PSNR of identical images should be very high"
        except ImportError:
            pytest.skip("PSNR calculation module not available")
    
    def test_psnr_different_images(self, test_images):
        """Test PSNR with different images."""
        try:
            from utils.metrics import calculate_psnr
            
            for lr, hr in test_images[:1]:
                # Create slightly different version
                hr_noisy = torch.clamp(hr + torch.randn_like(hr) * 0.01, 0, 1)
                
                psnr_clean = calculate_psnr(hr, hr)
                psnr_noisy = calculate_psnr(hr, hr_noisy)
                
                assert psnr_noisy < psnr_clean, "Noisy image should have lower PSNR"
                assert psnr_noisy > 20, "Slightly noisy image should still have reasonable PSNR"
        except ImportError:
            pytest.skip("PSNR calculation module not available")
    
    def test_psnr_output_shape(self, test_images):
        """Test PSNR handles various tensor shapes."""
        try:
            from utils.metrics import calculate_psnr
            
            lr, hr = test_images[0]
            
            # Test with batch dimension
            psnr_batch = calculate_psnr(hr, hr)
            
            # Test without batch dimension
            psnr_no_batch = calculate_psnr(hr.squeeze(0), hr.squeeze(0))
            
            assert abs(psnr_batch - psnr_no_batch) < 0.01, \
                "PSNR should be same regardless of batch dimension"
        except ImportError:
            pytest.skip("PSNR calculation module not available")


class TestSSIMCalculation:
    """Test SSIM metric calculation."""
    
    def test_ssim_perfect_reconstruction(self, test_images):
        """Test SSIM with identical images (should be 1.0)."""
        try:
            from utils.metrics import calculate_ssim
            
            for lr, hr in test_images:
                ssim = calculate_ssim(hr, hr)
                assert abs(ssim - 1.0) < 0.01, "SSIM of identical images should be ~1.0"
        except ImportError:
            pytest.skip("SSIM calculation module not available")
    
    def test_ssim_different_images(self, test_images):
        """Test SSIM with different images."""
        try:
            from utils.metrics import calculate_ssim
            
            for lr, hr in test_images[:1]:
                # Create different version
                hr_modified = torch.clamp(hr * 0.9 + 0.05, 0, 1)
                
                ssim_same = calculate_ssim(hr, hr)
                ssim_diff = calculate_ssim(hr, hr_modified)
                
                assert ssim_diff < ssim_same, "Different images should have lower SSIM"
                assert ssim_diff > 0.5, "Similar images should have SSIM > 0.5"
        except ImportError:
            pytest.skip("SSIM calculation module not available")


class TestLPIPSCalculation:
    """Test LPIPS metric calculation."""
    
    def test_lpips_import(self):
        """Test that LPIPS can be imported."""
        try:
            import lpips
            assert True, "LPIPS module available"
        except ImportError:
            pytest.skip("LPIPS module not available")
    
    def test_lpips_perceptual_distance(self, test_images):
        """Test LPIPS perceptual distance."""
        try:
            from utils.metrics import calculate_lpips
            
            for lr, hr in test_images[:1]:
                # Identical images should have LPIPS ~ 0
                lpips_same = calculate_lpips(hr, hr)
                
                # Different images should have higher LPIPS
                hr_modified = torch.clamp(hr + torch.randn_like(hr) * 0.1, 0, 1)
                lpips_diff = calculate_lpips(hr, hr_modified)
                
                assert lpips_same < 0.01, "Identical images should have LPIPS ~ 0"
                assert lpips_diff > lpips_same, "Different images should have higher LPIPS"
        except ImportError:
            pytest.skip("LPIPS calculation module not available")


class TestDISTSMetric:
    """Test DISTS metric calculation."""
    
    def test_dists_import(self):
        """Test that DISTS can be imported."""
        try:
            from utils.dists_loss import DISTS
            assert True, "DISTS module available"
        except ImportError:
            pytest.skip("DISTS module not available")
    
    def test_dists_calculation(self, test_images):
        """Test DISTS metric calculation."""
        try:
            from utils.metrics import calculate_dists
            
            for lr, hr in test_images[:1]:
                # Identical images should have DISTS ~ 0
                dists_same = calculate_dists(hr, hr)
                
                assert dists_same < 0.1, "Identical images should have low DISTS"
        except ImportError:
            pytest.skip("DISTS calculation module not available")


class TestInferenceSpeedBenchmark:
    """Test inference speed benchmarking."""
    
    def test_inference_speed_measurement(self, mock_model):
        """Test that inference speed can be measured."""
        device = torch.device('cpu')
        model = mock_model.to(device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 64, 64).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_input)
        
        # Measure
        num_runs = 10
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = model(dummy_input)
                times.append(time.time() - start)
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time
        
        assert avg_time > 0, "Inference time should be positive"
        assert fps > 0, "FPS should be positive"
        
        print(f"\nInference speed: {avg_time*1000:.2f} ms/image ({fps:.2f} FPS)")
    
    def test_benchmark_results_structure(self, mock_model):
        """Test that benchmark returns proper structure."""
        try:
            from utils.benchmark import benchmark_model
            
            results = benchmark_model(
                mock_model,
                input_size=(3, 64, 64),
                num_runs=10,
                device='cpu'
            )
            
            # Verify structure
            assert 'mean_time_ms' in results, "Should have mean_time_ms"
            assert 'fps' in results, "Should have fps"
            assert 'std_time_ms' in results, "Should have std_time_ms"
            
            # Verify values
            assert results['mean_time_ms'] > 0, "Mean time should be positive"
            assert results['fps'] > 0, "FPS should be positive"
            
        except ImportError:
            pytest.skip("Benchmark module not available")


class TestMemoryUsageTracking:
    """Test VRAM/memory usage tracking."""
    
    def test_memory_tracking_import(self):
        """Test that memory tracking can be imported."""
        try:
            from utils.memory import get_gpu_memory, track_memory
            assert True, "Memory tracking available"
        except ImportError:
            pytest.skip("Memory tracking module not available")
    
    def test_peak_memory_measurement(self, mock_model):
        """Test peak memory usage during inference."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            from utils.memory import track_memory
            
            device = torch.device('cuda')
            model = mock_model.to(device)
            model.eval()
            
            dummy_input = torch.randn(1, 3, 256, 256).to(device)
            
            # Track memory
            with track_memory() as mem_stats:
                with torch.no_grad():
                    _ = model(dummy_input)
            
            assert 'peak_allocated_mb' in mem_stats, "Should track peak memory"
            assert mem_stats['peak_allocated_mb'] > 0, "Peak memory should be positive"
            
        except ImportError:
            pytest.skip("Memory tracking module not available")


class TestBenchmarkReportGeneration:
    """Test benchmark report generation."""
    
    def test_json_report_generation(self, mock_model, test_images):
        """Test JSON report generation."""
        try:
            from utils.benchmark import generate_benchmark_report
            
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / 'benchmark_report.json'
                
                # Generate report
                report = generate_benchmark_report(
                    model=mock_model,
                    test_images=[lr for lr, hr in test_images],
                    reference_images=[hr for lr, hr in test_images],
                    output_path=output_path
                )
                
                # Verify file was created
                assert output_path.exists(), "Report file should be created"
                
                # Verify structure
                assert 'metrics' in report, "Report should have metrics"
                assert 'inference_speed' in report, "Report should have inference_speed"
                assert 'timestamp' in report, "Report should have timestamp"
                
                # Load and verify JSON
                import json
                with open(output_path) as f:
                    loaded = json.load(f)
                assert loaded['metrics']['psnr'] == report['metrics']['psnr']
                
        except ImportError:
            pytest.skip("Benchmark report module not available")


class TestBenchmarkScriptIntegration:
    """Integration tests for benchmark script."""
    
    def test_benchmark_script_exists(self):
        """Test that benchmark script exists."""
        script_path = Path(__file__).parent.parent / 'scripts' / 'benchmark_model.py'
        assert script_path.exists(), f"Benchmark script not found: {script_path}"
    
    def test_benchmark_script_help(self):
        """Test that benchmark script shows help."""
        import subprocess
        
        script_path = Path(__file__).parent.parent / 'scripts' / 'benchmark_model.py'
        result = subprocess.run(
            ['python', str(script_path), '--help'],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, "Help command should succeed"
        assert '--checkpoint' in result.stdout, "Help should mention --checkpoint option"


def test_data_source_report():
    """Report which data source is being used for tests."""
    data_dir = ensure_test_data(min_images=4, verbose=True)
    print(f"\n[Benchmarking] Using data: {data_dir}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
