"""
Unit tests for new loss functions.
Tests gradient loss, diversity loss, combined loss, anime losses, temporal losses.
"""
import sys
import os
import torch
import pytest
from anime_sr.losses.gradient_loss import GradientLoss, LaplacianLoss, CombinedGradientLoss
from anime_sr.losses.diversity_loss import TeacherDiversityLoss, TeacherDisagreementLoss
from anime_sr.losses.combined_loss import Stage1CombinedLoss, AdaptiveCombinedLoss
from anime_sr.losses.anime_losses import (
    LineArtPreservationLoss,
    ColorConsistencyLoss,
    FlatRegionPreservationLoss,
    AnimeCombinedLoss,
)
from anime_sr.losses.temporal_loss import TemporalConsistencyLoss, FlowGuidedTemporalLoss


class TestGradientLoss:
    """Test gradient-based losses."""
    
    def test_gradient_loss_init(self):
        """Test GradientLoss initialization."""
        loss = GradientLoss()
        assert hasattr(loss, 'sobel_x')
        assert hasattr(loss, 'sobel_y')
    
    def test_gradient_loss_forward(self):
        """Test GradientLoss forward."""
        loss_fn = GradientLoss()
        
        pred = torch.randn(2, 3, 32, 32)
        target = torch.randn(2, 3, 32, 32)
        
        loss = loss_fn(pred, target)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_laplacian_loss(self):
        """Test LaplacianLoss."""
        loss_fn = LaplacianLoss()
        
        pred = torch.randn(2, 3, 32, 32)
        target = torch.randn(2, 3, 32, 32)
        
        loss = loss_fn(pred, target)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_combined_gradient_loss(self):
        """Test CombinedGradientLoss."""
        loss_fn = CombinedGradientLoss(gradient_weight=1.0, laplacian_weight=0.5)
        
        pred = torch.randn(2, 3, 32, 32)
        target = torch.randn(2, 3, 32, 32)
        
        losses = loss_fn(pred, target)
        
        assert 'total' in losses
        assert 'gradient' in losses
        assert 'laplacian' in losses
        assert losses['total'].item() >= 0


class TestDiversityLoss:
    """Test teacher diversity losses."""
    
    def test_teacher_diversity_loss(self):
        """Test TeacherDiversityLoss."""
        loss_fn = TeacherDiversityLoss(diversity_weight=0.1)
        
        student_output = torch.randn(2, 3, 32, 32)
        teacher_outputs = [
            torch.randn(2, 3, 32, 32)
            for _ in range(3)
        ]
        
        losses = loss_fn(student_output, teacher_outputs)
        
        assert 'diversity' in losses
        assert 'teacher_variance' in losses
    
    def test_teacher_disagreement_loss(self):
        """Test TeacherDisagreementLoss."""
        loss_fn = TeacherDisagreementLoss()
        
        student_output = torch.randn(2, 3, 32, 32)
        teacher_outputs = [
            torch.randn(2, 3, 32, 32)
            for _ in range(3)
        ]
        target = torch.randn(2, 3, 32, 32)
        
        losses = loss_fn(student_output, teacher_outputs, target)
        
        assert 'disagreement_weighted' in losses
        assert 'base_loss' in losses


class TestCombinedLoss:
    """Test combined Stage 1 loss."""
    
    def test_stage1_combined_loss(self):
        """Test Stage1CombinedLoss."""
        loss_fn = Stage1CombinedLoss(
            l1_weight=0.4,
            wavelet_weight=0.3,
            gradient_weight=0.2,
            diversity_weight=0.1,
            wavelet_levels=3,
        )
        
        student_output = torch.randn(2, 3, 32, 32)
        target = torch.randn(2, 3, 32, 32)
        teacher_outputs = [
            torch.randn(2, 3, 32, 32)
            for _ in range(3)
        ]
        
        total_loss, losses = loss_fn(student_output, target, teacher_outputs)
        
        assert 'total' in losses
        assert 'l1' in losses
        assert 'wavelet' in losses
        assert 'gradient' in losses
        assert 'diversity' in losses
        assert 'weights' in losses
        assert total_loss.item() >= 0
        assert not torch.isnan(total_loss)
    
    def test_stage1_combined_without_teachers(self):
        """Test Stage1CombinedLoss without teachers."""
        loss_fn = Stage1CombinedLoss()
        
        student_output = torch.randn(2, 3, 32, 32)
        target = torch.randn(2, 3, 32, 32)
        
        total_loss, losses = loss_fn(student_output, target, None)
        
        assert total_loss.item() >= 0
        assert losses['diversity'].item() == 0.0  # Should be zero without teachers


class TestAnimeLosses:
    """Test anime-specific losses."""
    
    def test_line_art_preservation_loss(self):
        """Test LineArtPreservationLoss."""
        loss_fn = LineArtPreservationLoss()
        
        pred = torch.randn(2, 3, 64, 64)
        target = torch.randn(2, 3, 64, 64)
        
        losses = loss_fn(pred, target)
        
        assert 'line_art' in losses
        assert 'base_l1' in losses
        assert 'edge_coverage' in losses
        assert losses['line_art'].item() >= 0
    
    def test_color_consistency_loss(self):
        """Test ColorConsistencyLoss."""
        loss_fn = ColorConsistencyLoss()
        
        pred = torch.randn(2, 3, 32, 32)
        target = torch.randn(2, 3, 32, 32)
        
        losses = loss_fn(pred, target)
        
        assert 'color_consistency' in losses
        assert 'variance' in losses
    
    def test_flat_region_preservation_loss(self):
        """Test FlatRegionPreservationLoss."""
        loss_fn = FlatRegionPreservationLoss()
        
        # Create a flat region target
        target = torch.ones(2, 3, 32, 32) * 0.5
        pred = torch.randn(2, 3, 32, 32)
        
        losses = loss_fn(pred, target)
        
        assert 'flat_preservation' in losses
        assert 'flat_coverage' in losses
    
    def test_anime_combined_loss(self):
        """Test AnimeCombinedLoss."""
        loss_fn = AnimeCombinedLoss(
            line_art_weight=1.0,
            color_weight=0.5,
            flat_weight=0.5,
        )
        
        pred = torch.randn(2, 3, 64, 64)
        target = torch.randn(2, 3, 64, 64)
        
        losses = loss_fn(pred, target)
        
        assert 'total' in losses
        assert 'line_art' in losses
        assert 'color_consistency' in losses
        assert 'flat_preservation' in losses
        assert losses['total'].item() >= 0


class TestTemporalLosses:
    """Test temporal consistency losses."""
    
    def test_temporal_consistency_loss(self):
        """Test TemporalConsistencyLoss."""
        loss_fn = TemporalConsistencyLoss()
        
        pred_curr = torch.randn(2, 3, 32, 32)
        pred_prev = torch.randn(2, 3, 32, 32)
        hr_curr = torch.randn(2, 3, 32, 32)
        hr_prev = torch.randn(2, 3, 32, 32)
        
        losses = loss_fn(pred_curr, pred_prev, hr_curr, hr_prev)
        
        assert 'temporal' in losses
        assert 'consistency' in losses
    
    def test_flow_guided_temporal_loss(self):
        """Test FlowGuidedTemporalLoss."""
        loss_fn = FlowGuidedTemporalLoss()
        
        pred_curr = torch.randn(2, 3, 32, 32)
        pred_prev = torch.randn(2, 3, 32, 32)
        hr_curr = torch.randn(2, 3, 32, 32)
        hr_prev = torch.randn(2, 3, 32, 32)
        
        losses = loss_fn(pred_curr, pred_prev, hr_curr, hr_prev)
        
        assert 'flow_temporal' in losses
        assert 'pred_warp_error' in losses


def test_loss_gradient_flow():
    """Test that gradients flow properly through all losses."""
    losses_to_test = [
        GradientLoss(),
        LaplacianLoss(),
        LineArtPreservationLoss(),
    ]
    
    for loss_fn in losses_to_test:
        pred = torch.randn(2, 3, 32, 32, requires_grad=True)
        target = torch.randn(2, 3, 32, 32)
        
        if isinstance(loss_fn, (LineArtPreservationLoss)):
            loss_dict = loss_fn(pred, target)
            loss = loss_dict['line_art'] if 'line_art' in loss_dict else loss_dict[list(loss_dict.keys())[0]]
        else:
            loss = loss_fn(pred, target)
        
        loss.backward()
        
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()


def test_loss_weights_sum():
    """Test that combined loss weights sum to 1."""
    loss_fn = Stage1CombinedLoss(
        l1_weight=0.4,
        wavelet_weight=0.3,
        gradient_weight=0.2,
        diversity_weight=0.1,
    )
    
    # Check weights are normalized
    assert abs(loss_fn.l1_weight + loss_fn.wavelet_weight + 
               loss_fn.gradient_weight + loss_fn.diversity_weight - 1.0) < 1e-6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
