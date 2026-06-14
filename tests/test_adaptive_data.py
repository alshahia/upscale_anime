"""
Test adaptive data pipeline features:
- Auto crop_size detection
- Smart quality-based degradation
"""
import pytest
import numpy as np
from pathlib import Path
import sys
import warnings

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Direct imports to avoid __init__.py issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "quality_analyzer", Path(__file__).parent.parent / "src" / "data" / "quality_analyzer.py"
)
qa_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa_module)

spec2 = importlib.util.spec_from_file_location(
    "base", Path(__file__).parent.parent / "src" / "data" / "base.py"
)
base_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(base_module)

QualityAnalyzer = qa_module.QualityAnalyzer
compute_optimal_crop_size = qa_module.compute_optimal_crop_size
get_degradation_preset_config = qa_module.get_degradation_preset_config
calculate_image_quality = qa_module.calculate_image_quality
BaseDataset = base_module.BaseDataset


class TestAutoCropSize:
    """Test automatic crop size detection."""
    
    def test_compute_optimal_crop_size_auto(self):
        """Test auto crop size computation."""
        # Create dummy image paths
        from PIL import Image
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # Create test images of different sizes
            for i, size in enumerate([(1280, 720), (1920, 1080)]):
                img = Image.new('RGB', size, color=(i*50, i*50, i*50))
                img.save(tmpdir / f"test_{i}.png")
            
            image_paths = list(tmpdir.glob("*.png"))
            
            # Test auto mode - should pick size appropriate for smallest images (720p)
            crop_size = compute_optimal_crop_size(
                image_paths, mode='auto', max_size=960, min_size=64
            )
            
            # Should be <= 720 * 0.9 (with margin)
            assert crop_size <= 720 * 0.9, f"crop_size {crop_size} too large for 720p images"
            assert crop_size >= 64, f"crop_size {crop_size} below minimum"
    
    def test_compute_optimal_crop_size_manual(self):
        """Test manual crop size."""
        image_paths = [Path("dummy.png")]
        
        crop_size = compute_optimal_crop_size(
            image_paths, mode='manual', manual_size=256, max_size=960, min_size=64
        )
        
        assert crop_size == 256, "Manual crop_size should be used"
    
    def test_dataset_auto_crop(self):
        """Test BaseDataset with auto crop_size."""
        frames_dir = Path("data/anime_video_frames")
        
        if not frames_dir.exists() or not list(frames_dir.glob("*.png")):
            pytest.skip("Test data not available")
        
        # Initialize with auto mode
        ds = BaseDataset(
            hr_dir=str(frames_dir),
            scale=4,
            crop_size=1080,  # Wrong size
            crop_size_mode='auto',
            degradation={'enabled': False}
        )
        
        # Crop size should be adjusted to fit dataset
        assert ds.crop_size <= 720, f"crop_size {ds.crop_size} should be <= 720 for 720p images"
    
    def test_dataset_manual_crop_warning(self):
        """Test that manual mode shows warning for wrong crop_size."""
        frames_dir = Path("data/anime_video_frames")
        
        if not frames_dir.exists() or not list(frames_dir.glob("*.png")):
            pytest.skip("Test data not available")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            ds = BaseDataset(
                hr_dir=str(frames_dir),
                scale=4,
                crop_size=1080,  # Wrong size
                crop_size_mode='manual',
                degradation={'enabled': False}
            )
            
            # Should have warned about inappropriate crop_size
            assert len(w) > 0, "Should warn about crop_size > image dimensions"


class TestSmartDegradation:
    """Test smart quality-based degradation."""
    
    def test_quality_analyzer_init(self):
        """Test QualityAnalyzer initialization."""
        analyzer = QualityAnalyzer()
        assert analyzer is not None
    
    def test_get_degradation_preset_config(self):
        """Test preset configuration retrieval."""
        # Test light preset
        light = get_degradation_preset_config('light')
        assert light['enabled'] is True
        assert light['blur_prob'] < 0.5  # Light has low blur prob
        
        # Test heavy preset
        heavy = get_degradation_preset_config('heavy')
        assert heavy['enabled'] is True
        assert heavy['blur_prob'] > 0.8  # Heavy has high blur prob
        
        # Test disabled preset
        disabled = get_degradation_preset_config('disabled')
        assert disabled['enabled'] is False
    
    def test_calculate_image_quality_synthetic(self):
        """Test quality calculation on synthetic images."""
        # Create clean high-quality image
        clean_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        quality_clean = calculate_image_quality(clean_img)
        
        # Quality should be in valid range
        assert 0 <= quality_clean <= 1
        
        # Create blurry low-quality image
        blurry_img = np.full((256, 256, 3), 128, dtype=np.uint8)
        quality_blurry = calculate_image_quality(blurry_img)
        
        # Blurry should have lower quality than clean
        assert quality_blurry < quality_clean
    
    def test_quality_tier_mapping(self):
        """Test quality score to tier mapping."""
        analyzer = QualityAnalyzer(
            auto_quality_thresholds={'hd_threshold': 0.8, 'standard_threshold': 0.5}
        )
        
        # HD quality
        tier = analyzer.get_degradation_tier(0.9)
        assert tier == 'light'
        
        # Standard quality
        tier = analyzer.get_degradation_tier(0.6)
        assert tier == 'medium'
        
        # Low quality
        tier = analyzer.get_degradation_tier(0.3)
        assert tier == 'heavy'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
