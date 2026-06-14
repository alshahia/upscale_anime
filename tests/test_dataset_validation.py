"""
Test Suite for Dataset Validation Pipeline (Phase 1 of Small Dataset Techniques)

Tests dataset validation and cleaning functionality including:
- Perceptual hashing for duplicate detection
- Blur detection (Laplacian variance)
- Resolution validation
- Compression artifact estimation
- Validation report generation

Data Strategy:
- Primary: data/val_hr/ (4 real images)
- Fallback: Create controlled test images with known duplicates/blur
"""
import pytest
import sys
import json
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

# Add src and utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

for _tdm_k in ('utils', 'utils.test_data_manager'):
    sys.modules.pop(_tdm_k, None)
from utils.test_data_manager import ensure_test_data, create_fallback_data, cleanup_temp_data


class TestDatasetValidation:
    """Test dataset validation functionality."""
    
    @pytest.fixture(scope='class')
    def test_data_dir(self):
        """Provide test data directory (real or fallback)."""
        data_dir = ensure_test_data(min_images=4, pattern='gradient', verbose=False)
        yield data_dir
        # Cleanup only temp data, not real data
        if 'temp_test_data' in str(data_dir):
            cleanup_temp_data()
    
    @pytest.fixture
    def temp_validation_dir(self):
        """Create temporary directory for validation testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_perceptual_hash_duplicate_detection(self, temp_validation_dir):
        """Test that identical/near-identical images are detected as duplicates."""
        # Create controlled test images
        img1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img2 = img1.copy()  # Exact duplicate
        img3 = img1 + np.random.randint(0, 5, img1.shape, dtype=np.uint8)  # Near duplicate
        img4 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)  # Different
        
        # Save images
        Image.fromarray(img1).save(temp_validation_dir / 'img1.png')
        Image.fromarray(img2).save(temp_validation_dir / 'img2.png')  # Duplicate
        Image.fromarray(img3).save(temp_validation_dir / 'img3.png')  # Near-duplicate
        Image.fromarray(img4).save(temp_validation_dir / 'img4.png')  # Different
        
        # Try to import validation functions
        try:
            from data.dataset_validation import find_duplicates, compute_perceptual_hash
            
            # Test perceptual hash
            hash1 = compute_perceptual_hash(temp_validation_dir / 'img1.png')
            hash2 = compute_perceptual_hash(temp_validation_dir / 'img2.png')
            hash4 = compute_perceptual_hash(temp_validation_dir / 'img4.png')
            
            # Exact duplicates should have identical hashes
            assert hash1 == hash2, "Exact duplicates should have identical hashes"
            
            # Different images should have different hashes
            assert hash1 != hash4, "Different images should have different hashes"
            
            # Test duplicate detection
            duplicates = find_duplicates(temp_validation_dir, threshold=10)
            
            # Should find at least 2 duplicates (img1 and img2)
            assert len(duplicates) >= 1, "Should detect at least one duplicate pair"
            
        except ImportError:
            pytest.skip("Dataset validation module not available")
    
    def test_blur_detection(self, temp_validation_dir):
        """Test blur detection using Laplacian variance."""
        # Create sharp image
        sharp_img = np.zeros((100, 100, 3), dtype=np.uint8)
        sharp_img[::10, :] = 255  # Sharp edges
        
        # Create blurred version
        from scipy.ndimage import gaussian_filter
        blurred_img = gaussian_filter(sharp_img.astype(float), sigma=3).astype(np.uint8)
        
        # Save images
        Image.fromarray(sharp_img).save(temp_validation_dir / 'sharp.png')
        Image.fromarray(blurred_img).save(temp_validation_dir / 'blurred.png')
        
        try:
            from data.dataset_validation import detect_blur, compute_laplacian_variance
            
            # Test Laplacian variance computation
            sharp_var = compute_laplacian_variance(temp_validation_dir / 'sharp.png')
            blurred_var = compute_laplacian_variance(temp_validation_dir / 'blurred.png')
            
            # Sharp image should have higher variance
            assert sharp_var > blurred_var, "Sharp image should have higher Laplacian variance"
            
            # Test blur detection
            is_sharp_blurry = detect_blur(temp_validation_dir / 'sharp.png', threshold=100)
            is_blurred_blurry = detect_blur(temp_validation_dir / 'blurred.png', threshold=100)
            
            assert not is_sharp_blurry, "Sharp image should not be flagged as blurry"
            assert is_blurred_blurry, "Blurred image should be flagged as blurry"
            
        except ImportError:
            pytest.skip("Dataset validation module not available")
    
    def test_resolution_validation(self, temp_validation_dir):
        """Test resolution-based filtering."""
        # Create images with different resolutions
        small_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        medium_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        large_img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        
        Image.fromarray(small_img).save(temp_validation_dir / 'small.png')
        Image.fromarray(medium_img).save(temp_validation_dir / 'medium.png')
        Image.fromarray(large_img).save(temp_validation_dir / 'large.png')
        
        try:
            from data.dataset_validation import validate_resolution, get_image_resolution
            
            # Test resolution extraction
            res_small = get_image_resolution(temp_validation_dir / 'small.png')
            res_medium = get_image_resolution(temp_validation_dir / 'medium.png')
            res_large = get_image_resolution(temp_validation_dir / 'large.png')
            
            assert res_small == (100, 100), f"Expected (100, 100), got {res_small}"
            assert res_medium == (512, 512), f"Expected (512, 512), got {res_medium}"
            assert res_large == (1024, 1024), f"Expected (1024, 1024), got {res_large}"
            
            # Test validation with minimum resolution
            valid_small = validate_resolution(temp_validation_dir / 'small.png', min_size=256)
            valid_medium = validate_resolution(temp_validation_dir / 'medium.png', min_size=256)
            valid_large = validate_resolution(temp_validation_dir / 'large.png', min_size=256)
            
            assert not valid_small, "Small image should fail min resolution check"
            assert valid_medium, "Medium image should pass min resolution check"
            assert valid_large, "Large image should pass min resolution check"
            
        except ImportError:
            pytest.skip("Dataset validation module not available")
    
    def test_compression_artifact_estimation(self, temp_validation_dir):
        """Test JPEG quality/compression artifact estimation."""
        # Create test image
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Save with different JPEG qualities
        Image.fromarray(img).save(temp_validation_dir / 'high_quality.jpg', quality=95)
        Image.fromarray(img).save(temp_validation_dir / 'medium_quality.jpg', quality=70)
        Image.fromarray(img).save(temp_validation_dir / 'low_quality.jpg', quality=40)
        
        try:
            from data.dataset_validation import estimate_compression_quality
            
            # Test quality estimation
            high_q = estimate_compression_quality(temp_validation_dir / 'high_quality.jpg')
            medium_q = estimate_compression_quality(temp_validation_dir / 'medium_quality.jpg')
            low_q = estimate_compression_quality(temp_validation_dir / 'low_quality.jpg')
            
            # Higher quality should have better score
            assert high_q > low_q, "High quality should have better score than low quality"
            assert medium_q > low_q, "Medium quality should have better score than low quality"
            
        except ImportError:
            pytest.skip("Dataset validation module not available")
    
    def test_validation_report_generation(self, test_data_dir, temp_validation_dir):
        """Test JSON report generation from validation."""
        try:
            from data.dataset_validation import validate_dataset
            
            # Run validation
            report = validate_dataset(
                test_data_dir,
                check_duplicates=True,
                check_blur=True,
                check_resolution=True,
                min_resolution=128,
                output_report=temp_validation_dir / 'validation_report.json'
            )
            
            # Verify report structure
            assert 'summary' in report, "Report should have summary section"
            assert 'total_images' in report['summary'], "Summary should have total_images"
            assert 'valid_images' in report['summary'], "Summary should have valid_images"
            assert 'issues' in report, "Report should have issues section"
            
            # Verify report was saved
            report_path = temp_validation_dir / 'validation_report.json'
            assert report_path.exists(), "Report file should be created"
            
            # Load and verify JSON
            with open(report_path) as f:
                saved_report = json.load(f)
            assert saved_report['summary']['total_images'] == report['summary']['total_images']
            
        except ImportError:
            pytest.skip("Dataset validation module not available")
    
    def test_clean_dataset_output(self, test_data_dir, temp_validation_dir):
        """Test that cleaning produces valid output dataset."""
        try:
            from data.dataset_validation import validate_dataset, clean_dataset
            
            # Run validation and cleaning
            report = validate_dataset(test_data_dir, remove_duplicates=True, remove_blurry=True)
            
            if report['summary']['valid_images'] > 0:
                clean_dir = temp_validation_dir / 'cleaned'
                cleaned = clean_dataset(
                    test_data_dir,
                    clean_dir,
                    remove_duplicates=True,
                    remove_blurry=True,
                    min_resolution=128
                )
                
                # Verify cleaned dataset
                assert clean_dir.exists(), "Cleaned dataset directory should exist"
                cleaned_images = list(clean_dir.glob('*.png'))
                assert len(cleaned_images) > 0, "Cleaned dataset should have images"
                assert len(cleaned_images) <= report['summary']['valid_images'], \
                    "Cleaned dataset should not have more images than valid count"
            else:
                pytest.skip("No valid images to test cleaning")
                
        except ImportError:
            pytest.skip("Dataset validation module not available")


class TestValidationIntegration:
    """Integration tests for the validation script."""
    
    def test_validate_dataset_script_exists(self):
        """Test that the validation script exists and is runnable."""
        script_path = Path(__file__).parent / 'scripts' / 'validate_dataset.py'
        assert script_path.exists(), f"Validation script not found: {script_path}"
    
    def test_validate_dataset_cli_help(self):
        """Test that validation script shows help."""
        import subprocess
        
        script_path = Path(__file__).parent / 'scripts' / 'validate_dataset.py'
        result = subprocess.run(
            ['python', str(script_path), '--help'],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, "Help command should succeed"
        assert '--data-dir' in result.stdout or '-h' in result.stdout, \
            "Help should mention --data-dir option"


def test_data_source_report():
    """Report which data source is being used for tests."""
    for _tdm_k in ('utils', 'utils.test_data_manager'):
        sys.modules.pop(_tdm_k, None)
    from utils.test_data_manager import print_data_summary
    print_data_summary()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
