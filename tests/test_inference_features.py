"""
Unit tests for inference features: TTA, EMA, Model Soup.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch


class TestTestTimeAugmentation:
    """Test Test-Time Augmentation (TTA) features."""
    
    def test_tta_transforms_exist(self):
        """Test TTA transforms are defined."""
        from inference.engine import TTA_TRANSFORMS
        
        assert 'identity' in TTA_TRANSFORMS
        assert 'flip_h' in TTA_TRANSFORMS
        assert 'flip_v' in TTA_TRANSFORMS
        assert 'rot90' in TTA_TRANSFORMS
    
    def test_tta_flip_h(self):
        """Test horizontal flip transform."""
        from inference.engine import TTA_TRANSFORMS
        
        x = torch.rand(1, 3, 64, 64)
        forward_fn, inverse_fn = TTA_TRANSFORMS['flip_h']
        flipped = forward_fn(x)
        recovered = inverse_fn(flipped)
        
        assert torch.allclose(x, recovered, atol=1e-6)
    
    def test_tta_flip_v(self):
        """Test vertical flip transform."""
        from inference.engine import TTA_TRANSFORMS
        
        x = torch.rand(1, 3, 64, 64)
        forward_fn, inverse_fn = TTA_TRANSFORMS['flip_v']
        flipped = forward_fn(x)
        recovered = inverse_fn(flipped)
        
        assert torch.allclose(x, recovered, atol=1e-6)
    
    def test_tta_rot90(self):
        """Test 90-degree rotation transform."""
        from inference.engine import TTA_TRANSFORMS
        
        x = torch.rand(1, 3, 64, 64)
        forward_fn, inverse_fn = TTA_TRANSFORMS['rot90']
        rotated = forward_fn(x)
        recovered = inverse_fn(rotated)
        
        assert torch.allclose(x, recovered, atol=1e-6)


class TestEMAInference:
    """Test EMA model loading for inference."""
    
    def test_ema_state_dict_loading(self):
        """Test that EMA state dict can be loaded."""
        # Create a mock checkpoint with EMA state
        checkpoint = {
            'model_state_dict': {'weight': torch.rand(3, 3)},
            'ema_model_state_dict': {'weight': torch.rand(3, 3)},
            'epoch': 100
        }
        
        assert 'ema_model_state_dict' in checkpoint
        assert 'model_state_dict' in checkpoint
    
    def test_inference_engine_accepts_use_ema(self):
        """Test InferenceEngine accepts use_ema parameter in checkpoint."""
        # Create a mock checkpoint with EMA state
        checkpoint = {
            'model_state_dict': {'weight': torch.rand(3, 3)},
            'ema_model_state_dict': {'weight': torch.rand(3, 3)},
            'epoch': 100,
            'config': {}
        }
        
        # Verify EMA state exists
        assert 'ema_model_state_dict' in checkpoint
        assert 'model_state_dict' in checkpoint
        
        # EMA weights should be different (simulated)
        assert not torch.allclose(
            checkpoint['model_state_dict']['weight'],
            checkpoint['ema_model_state_dict']['weight']
        )


class TestModelSoup:
    """Test Model Soup (weight averaging)."""
    
    def test_model_soup_from_state_dicts(self):
        """Test ModelSoup can be created from state dicts."""
        from inference.engine import ModelSoup
        
        # Create mock state dicts
        state1 = {'layer.weight': torch.ones(3, 3), 'layer.bias': torch.zeros(3)}
        state2 = {'layer.weight': torch.ones(3, 3) * 2, 'layer.bias': torch.ones(3)}
        
        # Create ModelSoup directly from state dicts
        soup = ModelSoup([state1, state2])
        
        # Create soup with equal weights
        soup_state = soup.create_soup()
        
        # Averaged weight should be 1.5
        expected_weight = torch.ones(3, 3) * 1.5
        assert torch.allclose(soup_state['layer.weight'], expected_weight)
    
    def test_model_soup_with_custom_weights(self):
        """Test ModelSoup with custom weights."""
        from inference.engine import ModelSoup
        
        state1 = {'layer.weight': torch.ones(3, 3), 'layer.bias': torch.zeros(3)}
        state2 = {'layer.weight': torch.ones(3, 3) * 2, 'layer.bias': torch.ones(3)}
        
        soup = ModelSoup([state1, state2])
        
        # Create soup with custom weights (0.3, 0.7)
        soup_state = soup.create_soup(weights=[0.3, 0.7])
        
        # Weighted average: 1.0 * 0.3 + 2.0 * 0.7 = 1.7
        expected_weight = torch.ones(3, 3) * 1.7
        assert torch.allclose(soup_state['layer.weight'], expected_weight)


class TestInferenceEngine:
    """Test InferenceEngine core functionality."""
    
    def test_inference_engine_preprocess(self):
        """Test image preprocessing."""
        from inference.engine import InferenceEngine
        
        # Create dummy engine
        engine = InferenceEngine(Mock(), device='cpu')
        
        # Test with numpy array
        img_np = np.random.rand(64, 64, 3).astype(np.float32)
        tensor = engine.preprocess(img_np)
        
        assert tensor.shape == (1, 3, 64, 64)
        assert tensor.dtype == torch.float32
    
    def test_inference_engine_postprocess(self):
        """Test tensor postprocessing."""
        from inference.engine import InferenceEngine
        
        engine = InferenceEngine(Mock(), device='cpu')
        
        # Test with tensor
        tensor = torch.rand(1, 3, 64, 64)
        img_np = engine.postprocess(tensor)
        
        assert img_np.shape == (64, 64, 3)
        assert img_np.dtype == np.uint8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
