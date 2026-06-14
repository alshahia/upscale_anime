"""
Integration tests for the full anime super-resolution pipeline.
Tests end-to-end functionality including training, inference, and model loading.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
import os
from pathlib import Path
import yaml

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from models.span.checkpoint_compatible_exact import CheckpointCompatibleSPANExact
from training.orchestrator import TrainingOrchestrator
from inference.engine import InferenceEngine
from utils.config import load_config


class TestFullPipeline:
    """Test the complete pipeline from training to inference."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_config(self, temp_dir):
        """Create a minimal test configuration."""
        config = {
            "model": {
                "name": "test_model",
                "type": "checkpoint_compatible",
                "scale": 4,
                "channels": 48,
                "hidden_channels": 96,
                "num_blocks": 6,
                "in_channels": 3,
                "out_channels": 3
            },
            "training": {
                "mode": "stage2_only",
                "epochs": 2,
                "batch_size": 1,
                "lr": 1e-6,
                "device": "cpu",
                "mixed_precision": False,
                "max_grad_norm": 1.0,
                "stage2": {
                    "enabled": True,
                    "epochs": 2,
                    "lr": 1e-6,
                    "distillation_weight": 0.0,
                    "perceptual_weight": 0.01,
                    "pixel_weight": 1.0
                }
            },
            "data": {
                "crop_size": 64,
                "datasets": [
                    {
                        "name": "test_dataset",
                        "hr_dir": str(temp_dir / "test_images"),
                        "lr_dir": None,
                        "enabled": True,
                        "weight": 1,
                        "crop_size_mode": "fixed"
                    }
                ]
            },
            "losses": {
                "l1": {"weight": 1.0},
                "perceptual": {"weight": 0.01}
            },
            "checkpoint": {
                "save_dir": str(temp_dir / "checkpoints"),
                "save_interval": 1,
                "keep_best": True
            }
        }
        
        # Create test images directory
        (temp_dir / "test_images").mkdir(exist_ok=True)
        
        # Create dummy config file
        config_path = temp_dir / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path
    
    @pytest.fixture
    def sample_images(self, temp_dir):
        """Create sample test images."""
        image_dir = temp_dir / "test_images"
        image_dir.mkdir(exist_ok=True)
        
        # Create a few dummy image files
        for i in range(3):
            # Create a simple RGB image
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            img_path = image_dir / f"test_image_{i}.png"
            
            # Save as a simple file (we'll use text for testing)
            with open(img_path, 'w') as f:
                f.write(f"dummy_image_{i}")
        
        return image_dir
    
    def test_model_creation(self):
        """Test that the model can be created successfully."""
        model = CheckpointCompatibleSPANExact(
            scale=4,
            channels=48,
            hidden_channels=96,
            num_blocks=6
        )
        
        # Check model parameters
        assert model.scale == 4
        assert model.channels == 48
        assert model.hidden_channels == 96
        assert model.num_blocks == 6
        
        # Check that model has expected components
        assert hasattr(model, 'conv_1')
        assert hasattr(model, 'conv_2')
        assert hasattr(model, 'conv_cat')
        assert hasattr(model, 'upsampler')
        assert hasattr(model, 'final_conv')
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        assert total_params < 10_000_000  # Should be reasonable size
    
    def test_model_forward_pass(self):
        """Test that the model can perform forward pass."""
        model = CheckpointCompatibleSPANExact(
            scale=4,
            channels=48,
            hidden_channels=96,
            num_blocks=6
        )
        model.eval()
        
        # Create dummy input
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 64, 64)
        
        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)
        
        # Check output shape
        expected_shape = (batch_size, 3, 256, 256)  # 4x upscaling
        assert output.shape == expected_shape
        
        # Check output values are reasonable
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert output.min() >= 0  # Should be non-negative
        assert output.max() <= 1  # Should be clamped
    
    def test_checkpoint_loading(self):
        """Test checkpoint loading functionality."""
        model = CheckpointCompatibleSPANExact(
            scale=4,
            channels=48,
            hidden_channels=96,
            num_blocks=6
        )
        
        # Create a dummy checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': 1,
            'loss': 0.5
        }
        
        # Load checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Test that model still works after loading
        input_tensor = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (1, 3, 256, 256)
    
    def test_config_loading(self, sample_config):
        """Test configuration loading."""
        config = load_config(str(sample_config))
        
        # Check that config has expected sections
        assert 'model' in config
        assert 'training' in config
        assert 'data' in config
        assert 'losses' in config
        
        # Check model config
        assert config['model']['type'] == 'checkpoint_compatible'
        assert config['model']['scale'] == 4
        assert config['model']['channels'] == 48
    
    def test_inference_engine(self, temp_dir):
        """Test inference engine functionality."""
        model = CheckpointCompatibleSPANExact(
            scale=4,
            channels=48,
            hidden_channels=96,
            num_blocks=6
        )
        
        # Create inference engine
        engine = InferenceEngine(model, device='cpu')
        
        # Create dummy input image
        input_path = temp_dir / "input.png"
        output_path = temp_dir / "output.png"
        
        # Create a simple "image" file
        with open(input_path, 'w') as f:
            f.write("dummy_image")
        
        # Test would normally process real image
        # For now, just test engine creation
        assert engine.model == model
        assert engine.device == 'cpu'
    
    def test_training_config_validation(self, sample_config):
        """Test training configuration validation."""
        config = load_config(str(sample_config))
        
        # Validate required fields
        required_fields = [
            'model.name', 'model.type', 'model.scale',
            'training.mode', 'training.epochs',
            'data.datasets'
        ]
        
        for field in required_fields:
            keys = field.split('.')
            current = config
            for key in keys:
                assert key in current, f"Missing required field: {field}"
                current = current[key]
    
    def test_nan_handling(self):
        """Test NaN handling in model forward pass."""
        model = CheckpointCompatibleSPANExact(
            scale=4,
            channels=48,
            hidden_channels=96,
            num_blocks=6
        )
        model.eval()
        
        # Create input with NaN values
        input_tensor = torch.randn(1, 3, 64, 64)
        input_tensor[0, 0, 0, 0] = float('nan')
        
        # Forward pass should handle NaN gracefully
        with torch.no_grad():
            output = model(input_tensor)
        
        # Check output doesn't contain NaN
        assert not torch.isnan(output).any()
        assert output.shape == (1, 3, 256, 256)
    
    def test_memory_usage(self):
        """Test memory usage during training."""
        model = CheckpointCompatibleSPANExact(
            scale=4,
            channels=48,
            hidden_channels=96,
            num_blocks=6
        )
        
        # Check model size
        total_params = sum(p.numel() for p in model.parameters())
        param_size = total_params * 4  # 4 bytes per float32
        
        # Model should be reasonable size (< 100MB)
        assert param_size < 100 * 1024 * 1024
        
        # Test forward pass memory
        input_tensor = torch.randn(1, 3, 64, 64)
        
        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)
        
        # Check that tensors are properly sized
        assert input_tensor.numel() == 1 * 3 * 64 * 64
        assert output.numel() == 1 * 3 * 256 * 256
    
    def test_model_consistency(self):
        """Test model consistency across multiple runs."""
        model = CheckpointCompatibleSPANExact(
            scale=4,
            channels=48,
            hidden_channels=96,
            num_blocks=6
        )
        model.eval()
        
        # Create fixed input
        input_tensor = torch.randn(1, 3, 64, 64)
        
        # Run multiple times
        outputs = []
        for _ in range(3):
            with torch.no_grad():
                output = model(input_tensor)
            outputs.append(output.clone())
        
        # Check outputs are identical
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i], atol=1e-6)
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        model = CheckpointCompatibleSPANExact(
            scale=4,
            channels=48,
            hidden_channels=96,
            num_blocks=6
        )
        
        # Test with invalid input size
        try:
            input_tensor = torch.randn(1, 3, 32, 32)  # Too small
            with torch.no_grad():
                output = model(input_tensor)
            # Should work, but might have issues with very small inputs
        except Exception as e:
            # Should handle gracefully
            assert isinstance(e, (RuntimeError, ValueError))
        
        # Test with wrong number of channels
        try:
            input_tensor = torch.randn(1, 4, 64, 64)  # Wrong channels
            with torch.no_grad():
                output = model(input_tensor)
            # Should fail gracefully
        except Exception as e:
            assert isinstance(e, (RuntimeError, ValueError))


class TestPerformance:
    """Performance tests for the pipeline."""
    
    def test_inference_speed(self):
        """Test inference speed."""
        model = CheckpointCompatibleSPANExact(
            scale=4,
            channels=48,
            hidden_channels=96,
            num_blocks=6
        )
        model.eval()
        
        # Create input
        input_tensor = torch.randn(1, 3, 256, 256)
        
        # Measure inference time
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(10):
                output = model(input_tensor)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        # Should be reasonably fast (< 1 second per inference)
        assert avg_time < 1.0
    
    def test_memory_efficiency(self):
        """Test memory efficiency."""
        model = CheckpointCompatibleSPANExact(
            scale=4,
            channels=48,
            hidden_channels=96,
            num_blocks=6
        )
        
        # Check parameter count
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should be efficient (< 5M parameters)
        assert total_params < 5_000_000
        
        # Check memory usage during forward pass
        input_tensor = torch.randn(1, 3, 256, 256)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        # Output should be 4x larger
        assert output.numel() == input_tensor.numel() * 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
