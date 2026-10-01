"""
Unit tests for Online Hard Example Mining (OHEM).
"""
import sys
import os
import torch
import pytest
from anime_sr.training.ohem import OHEMLoss, OHEMSampler, compute_ohem_loss


class TestOHEMLoss:
    """Test OHEMLoss wrapper."""
    
    def test_ohem_loss_init(self):
        """Test OHEMLoss initialization."""
        base_loss = torch.nn.L1Loss()
        ohem = OHEMLoss(base_loss, ratio=0.7)
        
        assert ohem.ratio == 0.7
        assert ohem.loss_fn == base_loss
    
    def test_ohem_loss_forward(self):
        """Test OHEMLoss forward pass."""
        base_loss = torch.nn.L1Loss()
        ohem = OHEMLoss(base_loss, ratio=0.5)
        
        pred = torch.randn(4, 3, 32, 32)
        target = torch.randn(4, 3, 32, 32)
        
        loss = ohem(pred, target)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_ohem_selects_hard_examples(self):
        """Test that OHEM selects harder examples."""
        base_loss = torch.nn.L1Loss(reduction='none')
        ohem = OHEMLoss(base_loss, ratio=0.5)
        
        # Create prediction with varying errors
        pred = torch.zeros(4, 3, 32, 32)
        target = torch.zeros(4, 3, 32, 32)
        
        # Make some examples have larger errors
        target[0] = 1.0  # Large error
        target[1] = 0.9  # Large error
        target[2] = 0.1  # Small error
        target[3] = 0.0  # No error
        
        loss = ohem(pred, target)
        stats = ohem.get_stats()
        
        # Should select 50% of 4 = 2 examples
        assert stats['selected_samples'] == 2
        assert stats['total_samples'] == 4
    
    def test_ohem_stats_tracking(self):
        """Test that OHEM tracks statistics correctly."""
        base_loss = torch.nn.L1Loss()
        ohem = OHEMLoss(base_loss, ratio=0.7)
        
        pred = torch.randn(8, 3, 32, 32)
        target = torch.randn(8, 3, 32, 32)
        
        loss = ohem(pred, target)
        stats = ohem.get_stats()
        
        assert 'total_samples' in stats
        assert 'selected_samples' in stats
        assert 'threshold' in stats
        
        # Should have selected ~70% of 8 = 5-6 examples
        assert stats['selected_samples'] == 5 or stats['selected_samples'] == 6


class TestOHEMSampler:
    """Test OHEMSampler."""
    
    def test_sampler_selects_correct_ratio(self):
        """Test that sampler selects correct ratio."""
        sampler = OHEMSampler(ratio=0.6)
        
        # Create losses
        losses = torch.tensor([0.1, 0.5, 0.3, 0.8, 0.2, 0.9, 0.4, 0.6, 0.0, 1.0])
        
        hard_indices = sampler.select_hard_examples(losses)
        
        # Should select 60% of 10 = 6 examples
        assert len(hard_indices) == 6
        
        # Should select the largest losses
        expected_indices = torch.topk(losses, 6, largest=True)[1].sort()[0]
        actual_indices = hard_indices.sort()[0]
        
        assert torch.allclose(expected_indices, actual_indices)
    
    def test_sampler_hard_ratio(self):
        """Test get_hard_ratio method."""
        sampler = OHEMSampler(ratio=0.5)
        
        losses = torch.randn(10)
        sampler.select_hard_examples(losses)
        
        ratio = sampler.get_hard_ratio()
        assert ratio == 0.5


class TestComputeOHEMLoss:
    """Test compute_ohem_loss function."""
    
    def test_compute_ohem_loss(self):
        """Test compute_ohem_loss function."""
        loss_fn = torch.nn.L1Loss()
        
        pred = torch.randn(4, 3, 32, 32)
        target = torch.randn(4, 3, 32, 32)
        
        loss, info = compute_ohem_loss(loss_fn, pred, target, ratio=0.7)
        
        assert loss.item() >= 0
        assert 'threshold' in info
        assert 'max_loss' in info
        assert 'num_hard' in info
        assert 'hard_indices' in info
        
        # Should have selected ~70% of 4 = 2-3 examples
        assert info['num_hard'] in [2, 3]
    
    def test_ohem_with_perfect_prediction(self):
        """Test OHEM when prediction is perfect."""
        loss_fn = torch.nn.L1Loss()
        
        target = torch.randn(4, 3, 32, 32)
        pred = target.clone()  # Perfect prediction
        
        loss, info = compute_ohem_loss(loss_fn, pred, target, ratio=0.5)
        
        # Loss should be near zero
        assert loss.item() < 1e-5
        assert info['max_loss'] < 1e-5


def test_ohem_gradient_flow():
    """Test that gradients flow through OHEM correctly."""
    base_loss = torch.nn.L1Loss()
    ohem = OHEMLoss(base_loss, ratio=0.7)
    
    pred = torch.randn(4, 3, 32, 32, requires_grad=True)
    target = torch.randn(4, 3, 32, 32)
    
    loss = ohem(pred, target)
    loss.backward()
    
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()
    assert not torch.isinf(pred.grad).any()


def test_ohem_vs_standard_loss():
    """Test that OHEM loss is higher than standard for hard examples."""
    base_loss = torch.nn.L1Loss()
    ohem = OHEMLoss(base_loss, ratio=0.5)
    
    # Create prediction with some very wrong examples
    pred = torch.zeros(4, 3, 32, 32)
    target = torch.zeros(4, 3, 32, 32)
    
    # Make half have large errors
    target[:2] = 1.0
    
    # Standard loss (all examples)
    standard_loss = base_loss(pred, target)
    
    # OHEM loss (only hard examples)
    ohem_loss = ohem(pred, target)
    
    # OHEM should focus on hard examples, so loss should be higher
    assert ohem_loss > standard_loss


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
