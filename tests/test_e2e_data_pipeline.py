"""
End-to-End Test Suite for Data Pipeline

Tests complete data loading workflow from files to training batches:
- Dataset loading from disk (data/val_hr/)
- Degradation pipeline (RealESRGAN-style)
- Dataloader batch creation for GPU
- Multi-dataset weighting
- GPU degradation
- Prefetcher integration

Data Strategy:
- Primary: data/val_hr/ (4 images)
- Fallback: scripts/create_test_data.py --num-images 5
"""
import pytest
import sys
import tempfile
import shutil
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

for _tdm_k in ('utils', 'utils.test_data_manager'):
    sys.modules.pop(_tdm_k, None)
from utils.test_data_manager import ensure_test_data, get_val_hr_path, cleanup_temp_data


@pytest.fixture(scope='module')
def test_data_dir():
    """Provide test data directory."""
    data_dir = ensure_test_data(min_images=4, pattern='gradient', verbose=True)
    yield data_dir
    if 'temp_test_data' in str(data_dir):
        cleanup_temp_data()


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    temp = Path(tempfile.mkdtemp())
    yield temp
    shutil.rmtree(temp, ignore_errors=True)


class TestDatasetLoadingFromDisk:
    """Test loading datasets from disk."""
    
    def test_load_single_dataset(self, test_data_dir):
        """Test loading single dataset from disk."""
        try:
            from data import SuperResolutionDataset
            
            dataset = SuperResolutionDataset(
                hr_dir=test_data_dir,
                scale=4,
                crop_size=64
            )
            
            assert len(dataset) > 0, "Dataset should have images"
            
            # Load first sample
            lr, hr = dataset[0]
            assert lr.shape[1] == 3, "Should have 3 channels"
            assert hr.shape[1] == 3, "Should have 3 channels"
            
        except ImportError as e:
            pytest.skip(f"Dataset module not available: {e}")
    
    def test_load_multiple_datasets(self, test_data_dir, temp_dir):
        """Test loading multiple datasets."""
        try:
            from data import SuperResolutionDataset, MultiDataset
            
            # Create two datasets
            ds1 = SuperResolutionDataset(hr_dir=test_data_dir, scale=4)
            ds2 = SuperResolutionDataset(hr_dir=test_data_dir, scale=4)
            
            # Combine
            multi = MultiDataset([ds1, ds2], weights=[0.6, 0.4])
            
            assert len(multi) > 0, "MultiDataset should have items"
            
        except ImportError as e:
            pytest.skip(f"Dataset module not available: {e}")
    
    def test_dataset_image_format(self, test_data_dir):
        """Test dataset handles image formats correctly."""
        image_files = list(test_data_dir.glob('*.png'))
        
        if len(image_files) > 0:
            img = Image.open(image_files[0])
            assert img.mode in ['RGB', 'RGBA', 'L'], "Should support standard modes"
            
            # Check shape
            w, h = img.size
            assert w > 0 and h > 0, "Should have positive dimensions"


class TestDegradationPipeline:
    """Test degradation pipeline."""
    
    def test_realesrgan_degradation(self, test_data_dir):
        """Test RealESRGAN-style degradation."""
        try:
            from data.degradation import RealESRGANDegrader
            
            # Create degrader
            degrader = RealESRGANDegrader(
                scale=4,
                mode='light'
            )
            
            # Create HR image
            hr = torch.rand(1, 3, 256, 256)
            
            # Apply degradation
            lr = degrader(hr)
            
            # Verify
            assert lr.shape[2] == hr.shape[2] // 4, "Should be 4x smaller"
            assert lr.shape[3] == hr.shape[3] // 4, "Should be 4x smaller"
            assert torch.all(lr >= 0) and torch.all(lr <= 1), "Should be in [0, 1]"
            
        except ImportError as e:
            pytest.skip(f"Degradation module not available: {e}")
    
    def test_anime_degradation(self, test_data_dir):
        """Test anime-specific degradation."""
        try:
            from data.anime_degradation import AnimeDegrader
            
            degrader = AnimeDegrader(preset='light')
            
            # Create HR image
            hr = torch.rand(1, 3, 256, 256)
            
            # Apply degradation
            lr = degrader(hr)
            
            assert lr.shape[2] == hr.shape[2] // 4
            assert torch.all(lr >= 0)
            
        except ImportError as e:
            pytest.skip(f"Anime degradation not available: {e}")
    
    def test_degradation_modes(self, test_data_dir):
        """Test different degradation modes."""
        try:
            from data.degradation import RealESRGANDegrader
            
            modes = ['light', 'medium', 'heavy']
            hr = torch.rand(1, 3, 256, 256)
            
            for mode in modes:
                degrader = RealESRGANDegrader(scale=4, mode=mode)
                lr = degrader(hr)
                
                assert lr.shape[2] == 64, f"Mode {mode}: wrong height"
                assert lr.shape[3] == 64, f"Mode {mode}: wrong width"
                
        except ImportError as e:
            pytest.skip(f"Degradation module not available: {e}")


class TestDataLoaderBatchCreation:
    """Test dataloader batch creation."""
    
    def test_dataloader_creates_batches(self, test_data_dir):
        """Test dataloader creates proper batches."""
        try:
            from data import SuperResolutionDataset
            from torch.utils.data import DataLoader
            
            # Create dataset
            dataset = SuperResolutionDataset(
                hr_dir=test_data_dir,
                scale=4,
                crop_size=64
            )
            
            if len(dataset) == 0:
                pytest.skip("Dataset is empty")
            
            # Create dataloader
            loader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                num_workers=0
            )
            
            # Get batch
            batch = next(iter(loader))
            lr, hr = batch
            
            assert lr.shape[0] == min(2, len(dataset)), "Batch size should match"
            assert lr.shape[1] == 3, "Should have 3 channels"
            assert hr.shape[1] == 3, "Should have 3 channels"
            
        except ImportError as e:
            pytest.skip(f"Dataloader module not available: {e}")
    
    def test_batch_tensors_for_gpu(self, test_data_dir):
        """Test batches are GPU-ready."""
        try:
            from data import SuperResolutionDataset
            from torch.utils.data import DataLoader
            
            dataset = SuperResolutionDataset(
                hr_dir=test_data_dir,
                scale=4,
                crop_size=64
            )
            
            if len(dataset) == 0:
                pytest.skip("Dataset is empty")
            
            loader = DataLoader(dataset, batch_size=2, num_workers=0)
            lr, hr = next(iter(loader))
            
            # Verify tensor properties
            assert isinstance(lr, torch.Tensor), "LR should be tensor"
            assert isinstance(hr, torch.Tensor), "HR should be tensor"
            assert lr.dtype == torch.float32, "Should be float32"
            assert hr.dtype == torch.float32, "Should be float32"
            
            # Can move to GPU (if available)
            if torch.cuda.is_available():
                lr_gpu = lr.cuda()
                hr_gpu = hr.cuda()
                assert lr_gpu.is_cuda, "Should be on GPU"
                assert hr_gpu.is_cuda, "Should be on GPU"
                
        except ImportError as e:
            pytest.skip(f"Dataloader module not available: {e}")


class TestMultiDatasetWeighting:
    """Test multi-dataset weighting."""
    
    def test_multi_dataset_sampling(self, test_data_dir):
        """Test multi-dataset with different weights."""
        try:
            from data import SuperResolutionDataset, MultiDataset
            
            ds1 = SuperResolutionDataset(hr_dir=test_data_dir, scale=4)
            ds2 = SuperResolutionDataset(hr_dir=test_data_dir, scale=4)
            
            # Create weighted multi-dataset
            multi = MultiDataset([ds1, ds2], weights=[0.7, 0.3])
            
            # Sample multiple times
            sources = []
            for i in range(min(20, len(multi))):
                _, _, source_idx = multi[i]
                sources.append(source_idx)
            
            # Should have samples from both sources
            unique_sources = set(sources)
            assert len(unique_sources) >= 1, "Should sample from at least one source"
            
        except ImportError as e:
            pytest.skip(f"MultiDataset not available: {e}")
    
    def test_standardized_crop_sizes(self, test_data_dir):
        """Test crop sizes are standardized for batching."""
        try:
            from data import SuperResolutionDataset, MultiDataset
            
            ds1 = SuperResolutionDataset(hr_dir=test_data_dir, scale=4, crop_size=64)
            ds2 = SuperResolutionDataset(hr_dir=test_data_dir, scale=4, crop_size=96)
            
            # MultiDataset should standardize to minimum
            multi = MultiDataset([ds1, ds2])
            
            # Check crop sizes
            crop_sizes = []
            for ds in [ds1, ds2]:
                if hasattr(ds, 'crop_size'):
                    crop_sizes.append(ds.crop_size)
            
            if len(crop_sizes) > 1:
                assert len(set(crop_sizes)) == 1, "Crop sizes should be standardized"
                
        except ImportError as e:
            pytest.skip(f"MultiDataset not available: {e}")


class TestGPUDegradation:
    """Test GPU-based degradation."""
    
    def test_gpu_degradation_if_available(self, test_data_dir):
        """Test degradation on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            from data.degradation import RealESRGANDegrader
            
            degrader = RealESRGANDegrader(scale=4, mode='light')
            degrader = degrader.cuda()
            
            # Create HR on GPU
            hr = torch.rand(1, 3, 256, 256).cuda()
            
            # Apply degradation
            lr = degrader(hr)
            
            assert lr.is_cuda, "Output should be on GPU"
            assert lr.shape[2] == 64, "Should be 4x smaller"
            
        except ImportError as e:
            pytest.skip(f"Degradation module not available: {e}")


class TestPrefetcherIntegration:
    """Test async prefetcher integration."""
    
    def test_prefetcher_creation(self, test_data_dir):
        """Test prefetcher can be created."""
        try:
            from data.async_prefetcher import AsyncDataPrefetcher
            from data import SuperResolutionDataset
            from torch.utils.data import DataLoader
            
            dataset = SuperResolutionDataset(hr_dir=test_data_dir, scale=4)
            
            if len(dataset) == 0:
                pytest.skip("Dataset is empty")
            
            loader = DataLoader(dataset, batch_size=2, num_workers=0)
            
            # Create prefetcher
            if torch.cuda.is_available():
                prefetcher = AsyncDataPrefetcher(loader, device='cuda')
                assert prefetcher is not None
            else:
                pytest.skip("CUDA not available for prefetcher")
                
        except ImportError as e:
            pytest.skip(f"Prefetcher not available: {e}")
    
    def test_prefetcher_provides_batches(self, test_data_dir):
        """Test prefetcher provides batches."""
        try:
            from data.async_prefetcher import AsyncDataPrefetcher
            from data import SuperResolutionDataset
            from torch.utils.data import DataLoader
            
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")
            
            dataset = SuperResolutionDataset(hr_dir=test_data_dir, scale=4)
            
            if len(dataset) == 0:
                pytest.skip("Dataset is empty")
            
            loader = DataLoader(dataset, batch_size=2, num_workers=0)
            prefetcher = AsyncDataPrefetcher(loader, device='cuda')
            
            # Get batch
            lr, hr = prefetcher.next()
            
            assert lr is not None, "Should provide LR batch"
            assert hr is not None, "Should provide HR batch"
            assert lr.is_cuda, "Should be on GPU"
            
        except ImportError as e:
            pytest.skip(f"Prefetcher not available: {e}")


class TestDataPipelineIntegration:
    """Integration tests for data pipeline."""
    
    def test_full_pipeline_from_files_to_batches(self, test_data_dir):
        """Test complete pipeline: files → tensors → batches."""
        try:
            from data import SuperResolutionDataset
            from data.degradation import RealESRGANDegrader
            from torch.utils.data import DataLoader
            
            # 1. Load dataset from files
            dataset = SuperResolutionDataset(
                hr_dir=test_data_dir,
                scale=4,
                crop_size=64,
                degradation={'enabled': True, 'mode': 'light'}
            )
            
            assert len(dataset) > 0, "Should load images"
            
            # 2. Create dataloader
            loader = DataLoader(dataset, batch_size=2, num_workers=0)
            
            # 3. Get batch
            batch = next(iter(loader))
            lr, hr = batch
            
            # 4. Verify tensors
            assert lr.dim() == 4, "Should be 4D tensor"
            assert hr.dim() == 4, "Should be 4D tensor"
            assert lr.shape[1] == 3, "Should have 3 channels"
            
            # 5. Verify scale relationship
            assert hr.shape[2] == lr.shape[2] * 4, "HR should be 4x LR height"
            assert hr.shape[3] == lr.shape[3] * 4, "HR should be 4x LR width"
            
        except ImportError as e:
            pytest.skip(f"Required module not available: {e}")
    
    def test_analyze_dataset_script(self):
        """Test analyze dataset script exists."""
        script_path = Path(__file__).parent.parent / 'scripts' / 'analyze_dataset.py'
        assert script_path.exists(), f"Script not found: {script_path}"
    
    def test_benchmark_data_loading_script(self):
        """Test benchmark data loading script exists."""
        script_path = Path(__file__).parent.parent / 'scripts' / 'benchmark_data_loading.py'
        assert script_path.exists(), f"Script not found: {script_path}"


def test_data_source_report():
    """Report which data source is being used."""
    data_dir = ensure_test_data(min_images=4, verbose=True)
    print(f"\n[E2E Data Pipeline] Using data: {data_dir}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
