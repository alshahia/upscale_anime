"""
Unit tests for anime-specific features.
Tests degradation models and anime losses.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import pytest
from data.anime_degradation import (
    AnimeBlur,
    DirectionalBlur,
    ColorQuantization,
    BandingArtifact,
    RingingArtifact,
    AnimeDegradationPipeline,
    apply_anime_degradation,
)
from losses.anime_losses import (
    LineArtPreservationLoss,
    ColorConsistencyLoss,
    FlatRegionPreservationLoss,
)


class TestAnimeBlur:
    """Test AnimeBlur degradation."""
    
    def test_anime_blur_init(self):
        """Test AnimeBlur initialization."""
        blur = AnimeBlur()
        assert blur.kernel_sizes == [3, 5, 7, 9]
    
    def test_anime_blur_forward(self):
        """Test AnimeBlur forward."""
        blur = AnimeBlur()
        x = torch.randn(2, 3, 64, 64)
        
        output = blur(x)
        
        assert output.shape == x.shape
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)
    
    def test_anime_blur_preserves_range(self):
        """Test that blur preserves valid pixel range."""
        blur = AnimeBlur()
        x = torch.rand(2, 3, 64, 64)
        
        output = blur(x)
        
        assert output.min() >= 0
        assert output.max() <= 1


class TestDirectionalBlur:
    """Test DirectionalBlur degradation."""
    
    def test_directional_blur_forward(self):
        """Test DirectionalBlur forward."""
        blur = DirectionalBlur()
        x = torch.randn(2, 3, 64, 64)
        
        output = blur(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()


class TestColorQuantization:
    """Test ColorQuantization degradation."""
    
    def test_color_quantization_forward(self):
        """Test ColorQuantization forward."""
        quant = ColorQuantization()
        x = torch.rand(2, 3, 64, 64)
        
        output = quant(x)
        
        assert output.shape == x.shape
        assert output.min() >= 0
        assert output.max() <= 1
    
    def test_color_quantization_band_increases(self):
        """Test that quantization creates visible banding."""
        quant = ColorQuantization(levels_range=[8])
        
        # Create smooth gradient
        x = torch.linspace(0, 1, 256).view(1, 1, 1, 256).expand(1, 3, 256, 256)
        
        output = quant(x)
        
        # Should have fewer unique values than input
        input_unique = torch.unique(x).numel()
        output_unique = torch.unique(output).numel()
        
        assert output_unique < input_unique


class TestBandingArtifact:
    """Test BandingArtifact degradation."""
    
    def test_banding_forward(self):
        """Test BandingArtifact forward."""
        banding = BandingArtifact()
        
        # Create smooth region
        x = torch.ones(2, 3, 64, 64) * 0.5
        x += torch.linspace(0, 0.1, 64).view(1, 1, 1, 64).expand(2, 3, 64, 64)
        
        output = banding(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()


class TestRingingArtifact:
    """Test RingingArtifact degradation."""
    
    def test_ringing_forward(self):
        """Test RingingArtifact forward."""
        ringing = RingingArtifact()
        
        # Create image with sharp edge
        x = torch.zeros(2, 3, 64, 64)
        x[:, :, :, 32:] = 1.0  # Sharp edge at x=32
        
        output = ringing(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert output.min() >= 0
        assert output.max() <= 1


class TestAnimeDegradationPipeline:
    """Test AnimeDegradationPipeline."""
    
    def test_pipeline_init(self):
        """Test pipeline initialization."""
        pipeline = AnimeDegradationPipeline(
            enable_blur=True,
            enable_quantization=True,
        )
        
        assert pipeline.enable_blur
        assert pipeline.enable_quantization
        assert hasattr(pipeline, 'blur')
        assert hasattr(pipeline, 'quantization')
    
    def test_pipeline_forward(self):
        """Test pipeline forward."""
        pipeline = AnimeDegradationPipeline(enable_all=True)
        x = torch.rand(2, 3, 64, 64)
        
        output = pipeline(x)
        
        assert output.shape == x.shape
        assert output.min() >= 0
        assert output.max() <= 1
    
    def test_apply_anime_degradation(self):
        """Test apply_anime_degradation function."""
        hr = torch.rand(2, 3, 64, 64)
        
        lr, hr_degraded = apply_anime_degradation(hr, scale=4, enable_all=True)
        
        # LR should be smaller
        assert lr.shape[2] == hr.shape[2] // 4
        assert lr.shape[3] == hr.shape[3] // 4
        
        # Both should be in valid range
        assert lr.min() >= 0 and lr.max() <= 1
        assert hr_degraded.min() >= 0 and hr_degraded.max() <= 1


class TestAnimeLosses:
    """Test anime-specific losses."""
    
    def test_line_art_preservation_with_edges(self):
        """Test line art loss detects edges."""
        loss_fn = LineArtPreservationLoss()
        
        # Create image with sharp edge (high gradient)
        pred = torch.zeros(2, 3, 64, 64)
        target = torch.zeros(2, 3, 64, 64)
        target[:, :, :, 32:] = 1.0  # Sharp edge
        
        losses = loss_fn(pred, target)
        
        assert losses['edge_coverage'] > 0  # Should detect edges
        assert losses['line_art'] > losses['base_l1']  # Weighted loss > base
    
    def test_color_consistency_flat_regions(self):
        """Test color consistency on flat regions."""
        loss_fn = ColorConsistencyLoss()
        
        # Create flat color regions
        pred = torch.ones(2, 3, 64, 64) * 0.5
        target = torch.ones(2, 3, 64, 64) * 0.5
        
        losses = loss_fn(pred, target)
        
        assert losses['variance'] >= 0  # Variance should be low
    
    def test_flat_region_detection(self):
        """Test flat region preservation detects flat areas."""
        loss_fn = FlatRegionPreservationLoss()
        
        # Create image with flat region
        target = torch.ones(2, 3, 64, 64) * 0.5
        pred = target + torch.randn(2, 3, 64, 64) * 0.1  # Add some noise
        
        losses = loss_fn(pred, target)
        
        assert losses['flat_coverage'] > 0  # Should detect flat regions
        assert losses['flat_preservation'] >= 0


def test_anime_degradation_realistic():
    """Test that anime degradation produces realistic artifacts."""
    pipeline = AnimeDegradationPipeline(enable_all=True)
    
    # Create synthetic anime-like image
    x = torch.ones(1, 3, 128, 128)
    # Add flat color regions
    x[:, 0, :64, :64] = 0.2
    x[:, 1, :64, :64] = 0.4
    x[:, 2, :64, :64] = 0.8
    # Add sharp edge
    x[:, :, :, 64:] = x[:, :, :, :64] + 0.3
    x = torch.clamp(x, 0, 1)
    
    degraded = pipeline(x)
    
    # Should still be valid
    assert degraded.min() >= 0 and degraded.max() <= 1
    
    # Should be different from input (degradation occurred)
    assert not torch.allclose(degraded, x, atol=1e-4)


def test_gradient_preservation():
    """Test that gradients flow through anime degradations."""
    pipeline = AnimeDegradationPipeline(
        enable_blur=True,
        enable_quantization=False,  # Quantization not differentiable
    )
    
    x = torch.rand(2, 3, 64, 64, requires_grad=True)
    
    output = pipeline(x)
    loss = output.mean()
    loss.backward()
    
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
