"""Enhanced Augmentation Tests with Real Images"""
import pytest, sys
from pathlib import Path
import torch, numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))
for _tdm_k in ('utils', 'utils.test_data_manager'):
    sys.modules.pop(_tdm_k, None)
from utils.test_data_manager import ensure_test_data

@pytest.fixture(scope='module')
def real_image_tensors():
    """Load real images as tensors"""
    data_dir = ensure_test_data(min_images=2, verbose=False)
    image_files = list(data_dir.glob('*.png'))[:2]
    
    tensors = []
    for img_path in image_files:
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)[:256, :256, :]  # Crop to 256x256
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        tensors.append(tensor)
    
    return tensors


class TestMixupRealImages:
    """Test Mixup on real images"""
    
    def test_mixup_output_valid(self, real_image_tensors):
        """Test Mixup produces valid output"""
        try:
            from data.augmentation import MixupAugmentation
            
            mixup = MixupAugmentation(alpha=0.4)
            hr1, hr2 = real_image_tensors[0], real_image_tensors[1]
            
            # Create dummy LR
            lr1 = torch.nn.functional.interpolate(hr1.unsqueeze(0), scale_factor=1/4, mode='bicubic').squeeze(0)
            lr2 = torch.nn.functional.interpolate(hr2.unsqueeze(0), scale_factor=1/4, mode='bicubic').squeeze(0)
            
            hr_mixed, lr_mixed, lam = mixup(hr1, hr2, lr1, lr2)
            
            assert hr_mixed.shape == hr1.shape
            assert lr_mixed.shape == lr1.shape
            assert 0 <= lam <= 1
            assert torch.all(hr_mixed >= 0) and torch.all(hr_mixed <= 1)
            
        except ImportError:
            pytest.skip("Augmentation module not available")


class TestCutMixRealImages:
    """Test CutMix on real images"""
    
    def test_cutmix_output_valid(self, real_image_tensors):
        """Test CutMix produces valid output"""
        try:
            from data.augmentation import CutMixAugmentation
            
            cutmix = CutMixAugmentation()
            hr1, hr2 = real_image_tensors[0], real_image_tensors[1]
            
            lr1 = torch.nn.functional.interpolate(hr1.unsqueeze(0), scale_factor=1/4, mode='bicubic').squeeze(0)
            lr2 = torch.nn.functional.interpolate(hr2.unsqueeze(0), scale_factor=1/4, mode='bicubic').squeeze(0)
            
            hr_mixed, lr_mixed, ratio = cutmix(hr1, hr2, lr1, lr2, scale=4)
            
            assert hr_mixed.shape == hr1.shape
            assert lr_mixed.shape == lr1.shape
            assert 0 <= ratio <= 1
            
        except ImportError:
            pytest.skip("Augmentation module not available")


class TestProgressiveCropSizing:
    """Test progressive crop sizing"""
    
    def test_crop_size_progression(self):
        """Test 64→96→128→160 progression"""
        try:
            from data.augmentation import ProgressiveCropScheduler
            
            scheduler = ProgressiveCropScheduler(
                initial_size=64,
                max_size=160,
                epochs_per_stage=10
            )
            
            sizes = []
            for epoch in range(0, 50, 5):
                size = scheduler.get_crop_size(epoch)
                sizes.append((epoch, size))
            
            # Verify progression
            assert sizes[0][1] == 64
            assert sizes[-1][1] == 160
            
            # Verify monotonic increase
            for i in range(len(sizes) - 1):
                assert sizes[i+1][1] >= sizes[i][1]
                
        except ImportError:
            pytest.skip("Progressive crop scheduler not available")


class TestAugmentationPipelineIntegration:
    """Test full augmentation pipeline"""
    
    def test_pipeline_applies_augmentations(self, real_image_tensors):
        """Test pipeline applies augmentations"""
        try:
            from data.augmentation import AugmentationPipeline
            
            pipeline = AugmentationPipeline(
                use_mixup=True,
                use_cutmix=True,
                use_color_jitter=True
            )
            
            hr = real_image_tensors[0]
            lr = torch.nn.functional.interpolate(hr.unsqueeze(0), scale_factor=1/4, mode='bicubic').squeeze(0)
            
            # Apply pipeline
            hr_aug, lr_aug = pipeline(hr, lr)
            
            assert hr_aug.shape == hr.shape
            assert lr_aug.shape == lr.shape
            assert torch.all(hr_aug >= 0) and torch.all(hr_aug <= 1)
            
        except ImportError:
            pytest.skip("Augmentation pipeline not available")


def test_data_source_report():
    """Report data source"""
    data_dir = ensure_test_data(min_images=4, verbose=True)
    print(f"\n[Augmentation Enhanced] Using data: {data_dir}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
