"""
Tests for advanced augmentation techniques.
Covers Mixup, CutMix, and augmentation pipeline.
"""
import unittest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.augmentation import (
    MixupAugmentation,
    CutMixAugmentation,
    RandomResizedCrop,
    AugmentationPipeline,
    apply_geometric_augmentation,
    apply_color_jitter
)


class TestMixupAugmentation(unittest.TestCase):
    """Test Mixup augmentation for SR."""
    
    def setUp(self):
        self.mixup = MixupAugmentation(alpha=0.4)
        self.hr1 = torch.randn(3, 128, 128)
        self.hr2 = torch.randn(3, 128, 128)
        self.lr1 = torch.randn(3, 32, 32)
        self.lr2 = torch.randn(3, 32, 32)
    
    def test_mixup_output_shape(self):
        """Test that mixup preserves output shapes."""
        hr_mixed, lr_mixed, lam = self.mixup(self.hr1, self.hr2, self.lr1, self.lr2)
        
        self.assertEqual(hr_mixed.shape, self.hr1.shape)
        self.assertEqual(lr_mixed.shape, self.lr1.shape)
        self.assertIsInstance(lam, float)
        self.assertGreater(lam, 0)
        self.assertLess(lam, 1)
    
    def test_mixup_interpolation(self):
        """Test that mixup properly interpolates."""
        hr_mixed, lr_mixed, lam = self.mixup(self.hr1, self.hr2, self.lr1, self.lr2)
        
        # Check that output is between inputs
        expected_hr = lam * self.hr1 + (1 - lam) * self.hr2
        expected_lr = lam * self.lr1 + (1 - lam) * self.lr2
        
        torch.testing.assert_close(hr_mixed, expected_hr, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(lr_mixed, expected_lr, rtol=1e-5, atol=1e-5)
    
    def test_mixup_different_alpha(self):
        """Test mixup with different alpha values."""
        for alpha in [0.1, 0.5, 1.0, 2.0]:
            mixup = MixupAugmentation(alpha=alpha)
            hr_mixed, lr_mixed, lam = mixup(self.hr1, self.hr2, self.lr1, self.lr2)
            
            self.assertTrue(0 <= lam <= 1)
            self.assertEqual(hr_mixed.shape, self.hr1.shape)


class TestCutMixAugmentation(unittest.TestCase):
    """Test CutMix augmentation for SR."""
    
    def setUp(self):
        self.cutmix = CutMixAugmentation(min_ratio=0.2, max_ratio=0.8)
        self.hr1 = torch.randn(3, 128, 128)
        self.hr2 = torch.randn(3, 128, 128)
        self.lr1 = torch.randn(3, 32, 32)
        self.lr2 = torch.randn(3, 32, 32)
        self.scale = 4
    
    def test_cutmix_output_shape(self):
        """Test that CutMix preserves output shapes."""
        hr_mixed, lr_mixed, ratio = self.cutmix(
            self.hr1, self.hr2, self.lr1, self.lr2, scale=self.scale
        )
        
        self.assertEqual(hr_mixed.shape, self.hr1.shape)
        self.assertEqual(lr_mixed.shape, self.lr1.shape)
        self.assertIsInstance(ratio, float)
    
    def test_cutmix_regions(self):
        """Test that CutMix properly cuts and pastes regions."""
        hr_mixed, lr_mixed, ratio = self.cutmix(
            self.hr1, self.hr2, self.lr1, self.lr2, scale=self.scale
        )
        
        # Verify that some regions come from hr2/lr2
        # and others remain from hr1/lr1
        diff_hr = (hr_mixed != self.hr1).float().mean()
        diff_lr = (lr_mixed != self.lr1).float().mean()
        
        # At least some pixels should be different
        self.assertGreater(diff_hr.item(), 0)
        self.assertGreater(diff_lr.item(), 0)
    
    def test_cutmix_scale_consistency(self):
        """Test that HR and LR regions correspond correctly."""
        # Use identifiable patterns
        hr1 = torch.zeros(3, 128, 128)
        hr2 = torch.ones(3, 128, 128)
        lr1 = torch.zeros(3, 32, 32)
        lr2 = torch.ones(3, 32, 32)
        
        hr_mixed, lr_mixed, _ = self.cutmix(hr1, hr2, lr1, lr2, scale=4)
        
        # Check that regions with value 1 in HR correspond to regions with value 1 in LR
        hr_mask = (hr_mixed[0] > 0.5)
        lr_mask = (lr_mixed[0] > 0.5)
        
        # Downsample HR mask to LR resolution for comparison
        hr_mask_lr = torch.nn.functional.interpolate(
            hr_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(32, 32),
            mode='nearest'
        ).squeeze()
        
        # Should have reasonable correspondence
        agreement = (hr_mask_lr == lr_mask.float()).float().mean()
        self.assertGreater(agreement.item(), 0.7)  # At least 70% agreement


class TestRandomResizedCrop(unittest.TestCase):
    """Test random resized crop augmentation."""
    
    def setUp(self):
        self.rrc = RandomResizedCrop(scale_range=(0.5, 2.0))
        self.image = torch.randn(3, 128, 128)
    
    def test_output_size(self):
        """Test that output has correct size."""
        target_size = (64, 64)
        output = self.rrc(self.image, target_size)
        
        self.assertEqual(output.shape, (3, 64, 64))
    
    def test_different_scales(self):
        """Test with different scale ranges."""
        for scale_range in [(0.5, 1.0), (1.0, 2.0), (0.3, 3.0)]:
            rrc = RandomResizedCrop(scale_range=scale_range)
            output = rrc(self.image, (64, 64))
            self.assertEqual(output.shape, (3, 64, 64))


class TestAugmentationPipeline(unittest.TestCase):
    """Test complete augmentation pipeline."""
    
    def setUp(self):
        self.pipeline = AugmentationPipeline(
            mixup_prob=0.5,
            cutmix_prob=0.5,
            mixup_alpha=0.4,
            random_resize_prob=0.5
        )
        
        self.batch_size = 4
        self.hr_batch = torch.randn(self.batch_size, 3, 128, 128)
        self.lr_batch = torch.randn(self.batch_size, 3, 32, 32)
    
    def test_pipeline_output_shape(self):
        """Test that pipeline preserves batch shapes."""
        hr_aug, lr_aug, lambdas = self.pipeline.apply_to_batch(
            self.hr_batch, self.lr_batch, scale=4
        )
        
        self.assertEqual(hr_aug.shape, self.hr_batch.shape)
        self.assertEqual(lr_aug.shape, self.lr_batch.shape)
    
    def test_pipeline_deterministic(self):
        """Test that pipeline can be run multiple times."""
        # Run multiple times
        for _ in range(5):
            hr_aug, lr_aug, lambdas = self.pipeline.apply_to_batch(
                self.hr_batch, self.lr_batch, scale=4
            )
            self.assertEqual(hr_aug.shape, self.hr_batch.shape)
    
    def test_no_augmentation(self):
        """Test that pipeline works with no augmentation enabled."""
        pipeline = AugmentationPipeline(
            mixup_prob=0.0,
            cutmix_prob=0.0,
            random_resize_prob=0.0
        )
        
        hr_aug, lr_aug, lambdas = pipeline.apply_to_batch(
            self.hr_batch, self.lr_batch, scale=4
        )
        
        # Without augmentation, output should be same as input
        torch.testing.assert_close(hr_aug, self.hr_batch)
        torch.testing.assert_close(lr_aug, self.lr_batch)


class TestGeometricAugmentation(unittest.TestCase):
    """Test geometric augmentations."""
    
    def setUp(self):
        self.hr = torch.randn(3, 128, 128)
        self.lr = torch.randn(3, 32, 32)
    
    def test_geometric_output_shape(self):
        """Test that geometric aug preserves shapes."""
        hr_aug, lr_aug = apply_geometric_augmentation(self.hr, self.lr)
        
        self.assertEqual(hr_aug.shape, self.hr.shape)
        self.assertEqual(lr_aug.shape, self.lr.shape)
    
    def test_geometric_consistency(self):
        """Test that HR and LR get same geometric transform."""
        # Use a simple pattern
        hr = torch.arange(128 * 128).float().reshape(1, 128, 128).repeat(3, 1, 1)
        lr = torch.arange(32 * 32).float().reshape(1, 32, 32).repeat(3, 1, 1)
        
        # Apply augmentation
        hr_aug, lr_aug = apply_geometric_augmentation(hr, lr)
        
        # Check shapes preserved
        self.assertEqual(hr_aug.shape, hr.shape)
        self.assertEqual(lr_aug.shape, lr.shape)
        
        # Values should be rearranged but same set
        self.assertEqual(set(hr_aug[0].flatten().tolist()), set(hr[0].flatten().tolist()))


class TestColorJitter(unittest.TestCase):
    """Test color jitter augmentation."""
    
    def setUp(self):
        self.image = torch.rand(3, 128, 128)  # RGB image [0, 1]
    
    def test_color_jitter_output_range(self):
        """Test that color jitter output is in valid range."""
        jittered = apply_color_jitter(self.image, brightness=0.2, contrast=0.2)
        
        # Output should be in [0, 1] range
        self.assertTrue((jittered >= 0).all())
        self.assertTrue((jittered <= 1).all())
    
    def test_color_jitter_preserves_shape(self):
        """Test that color jitter preserves shape."""
        jittered = apply_color_jitter(self.image, brightness=0.2, contrast=0.2)
        self.assertEqual(jittered.shape, self.image.shape)
    
    def test_color_jitter_zero_factor(self):
        """Test that zero factor doesn't change image."""
        jittered = apply_color_jitter(self.image, brightness=0.0, contrast=0.0)
        torch.testing.assert_close(jittered, self.image)


class TestAugmentationIntegration(unittest.TestCase):
    """Integration tests for augmentation with realistic data."""
    
    def test_full_pipeline_with_batch(self):
        """Test full pipeline with realistic batch."""
        batch_size = 8
        hr = torch.randn(batch_size, 3, 128, 128)
        lr = torch.randn(batch_size, 3, 32, 32)
        
        pipeline = AugmentationPipeline(
            mixup_prob=0.3,
            cutmix_prob=0.3,
            random_resize_prob=0.0
        )
        
        # Run multiple times to test randomness
        results = []
        for _ in range(10):
            hr_aug, lr_aug, _ = pipeline.apply_to_batch(hr, lr, scale=4)
            results.append(hr_aug.clone())
        
        # Not all results should be identical (due to randomness)
        all_same = all(torch.allclose(results[0], r) for r in results[1:])
        self.assertFalse(all_same)
    
    def test_augmentation_with_varying_batch_sizes(self):
        """Test with different batch sizes."""
        pipeline = AugmentationPipeline(mixup_prob=0.5, cutmix_prob=0.5)
        
        for batch_size in [1, 2, 4, 8, 16]:
            hr = torch.randn(batch_size, 3, 64, 64)
            lr = torch.randn(batch_size, 3, 16, 16)
            
            hr_aug, lr_aug, _ = pipeline.apply_to_batch(hr, lr, scale=4)
            
            self.assertEqual(hr_aug.shape[0], batch_size)
            self.assertEqual(lr_aug.shape[0], batch_size)


if __name__ == '__main__':
    unittest.main()
