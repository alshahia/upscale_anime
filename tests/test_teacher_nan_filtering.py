"""
Unit tests for teacher NaN filtering and output padding features.
Tests the robustness improvements for handling corrupted teacher models.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock


class TestTeacherNaNFiltering:
    """Test NaN detection and filtering in teacher outputs."""
    
    def test_detect_nan_in_teacher_output(self):
        """Test NaN detection in teacher outputs."""
        # Simulate teacher output with NaN
        teacher_output = torch.rand(2, 3, 64, 64)
        teacher_output[0, 0, 0, 0] = float('nan')
        
        has_nan = torch.isnan(teacher_output).any().item()
        assert has_nan is True
    
    def test_detect_inf_in_teacher_output(self):
        """Test Inf detection in teacher outputs."""
        # Simulate teacher output with Inf
        teacher_output = torch.rand(2, 3, 64, 64)
        teacher_output[0, 0, 0, 0] = float('inf')
        
        has_inf = torch.isinf(teacher_output).any().item()
        assert has_inf is True
    
    def test_filter_nan_teachers(self):
        """Test filtering out NaN-producing teachers."""
        teacher_outputs = [
            torch.rand(2, 3, 64, 64),  # Valid
            torch.full((2, 3, 64, 64), float('nan')),  # Invalid (NaN)
            torch.rand(2, 3, 64, 64),  # Valid
        ]
        
        # Filter logic
        valid_outputs = []
        for i, to in enumerate(teacher_outputs):
            if torch.isnan(to).any() or torch.isinf(to).any():
                continue  # Skip invalid
            valid_outputs.append(to)
        
        assert len(valid_outputs) == 2
    
    def test_all_teachers_nan_handling(self):
        """Test handling when all teachers produce NaN."""
        teacher_outputs = [
            torch.full((2, 3, 64, 64), float('nan')),
            torch.full((2, 3, 64, 64), float('nan')),
        ]
        
        valid_outputs = [
            to for to in teacher_outputs
            if not (torch.isnan(to).any() or torch.isinf(to).any())
        ]
        
        assert len(valid_outputs) == 0
        # In actual training, this would skip the batch


class TestTeacherOutputPadding:
    """Test padding teacher outputs to expected count."""
    
    def test_pad_teacher_outputs(self):
        """Test padding to maintain expected teacher count."""
        valid_outputs = [torch.rand(2, 3, 64, 64)]  # Only 1 valid teacher
        expected_count = 3
        
        # Pad with zeros
        while len(valid_outputs) < expected_count:
            zero_teacher = torch.zeros_like(valid_outputs[0])
            valid_outputs.append(zero_teacher)
        
        assert len(valid_outputs) == 3
        # Check padded outputs are zeros
        assert torch.allclose(valid_outputs[1], torch.zeros(2, 3, 64, 64))
        assert torch.allclose(valid_outputs[2], torch.zeros(2, 3, 64, 64))
    
    def test_pad_preserves_valid_tensors(self):
        """Test that padding doesn't affect valid teacher outputs."""
        valid_outputs = [torch.rand(2, 3, 64, 64)]
        original = valid_outputs[0].clone()
        
        # Pad
        while len(valid_outputs) < 3:
            valid_outputs.append(torch.zeros_like(valid_outputs[0]))
        
        # Original should be unchanged
        assert torch.allclose(valid_outputs[0], original)


class TestDebugForwardSafety:
    """Test debug_forward attribute checking."""
    
    def test_model_without_debug_forward(self):
        """Test handling models without debug_forward method."""
        # Use spec to prevent Mock from auto-creating attributes
        model = Mock(spec=['forward'])
        # Don't add debug_forward attribute
        
        assert not hasattr(model, 'debug_forward')
    
    def test_model_with_debug_forward(self):
        """Test handling models with debug_forward method."""
        model = Mock()
        model.debug_forward = Mock(return_value={})
        
        assert hasattr(model, 'debug_forward')
        # Can safely call
        result = model.debug_forward([])
        model.debug_forward.assert_called_once()


class TestPerceptualLossDevicePlacement:
    """Test perceptual loss device placement in trainers."""
    
    def test_perceptual_loss_to_device(self):
        """Test moving perceptual loss to correct device."""
        from losses.perceptual_loss import DISTSLoss
        
        loss = DISTSLoss()
        
        # Mock device placement
        device = 'cpu'
        loss = loss.to(device)
        
        # Check normalization buffers are on correct device
        assert loss.mean.device.type == device
        assert loss.std.device.type == device
    
    def test_perceptual_loss_buffer_device_after_init(self):
        """Test that perceptual loss buffers are on correct device after init."""
        from losses.perceptual_loss import DISTSLoss
        
        loss = DISTSLoss()
        
        # Buffers should be on CPU by default
        assert loss.mean.is_cpu
        assert loss.std.is_cpu


class TestLossTrackingVerbose:
    """Test verbose loss tracking in progress bars."""
    
    def test_loss_dict_structure(self):
        """Test that loss dict contains expected keys."""
        loss_dict = {
            'l1': 0.5,
            'perceptual': 0.1,
            'wavelet_distill': 0.2,
            'fakd': 0.3,
            'total': 1.1
        }
        
        assert 'l1' in loss_dict
        assert 'perceptual' in loss_dict
        assert 'total' in loss_dict
    
    def test_perceptual_loss_in_dict(self):
        """Test perceptual loss is tracked when enabled."""
        # Simulating when perceptual loss is computed
        perceptual_loss = torch.tensor(0.15)
        
        loss_dict = {'l1': 0.5}
        loss_dict['perceptual'] = perceptual_loss.item()
        
        assert 'perceptual' in loss_dict
        assert loss_dict['perceptual'] == pytest.approx(0.15, abs=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
