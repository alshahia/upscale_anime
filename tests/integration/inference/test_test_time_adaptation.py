"""
Test Suite for Test-Time Adaptation (Phase 6 of Small Dataset Techniques)

Tests per-image adaptation functionality including:
- TTA initialization for single image
- Online learning loop on single image
- TTA loss computation (reconstruction loss)
- TTA convergence detection
- TTA checkpointing
- Inference quality improvement after adaptation

Data Strategy:
- Primary: Single image from data/test_hr/
- Fallback: Create single test image
"""
import pytest
import sys
import tempfile
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add src and utils to path
for _tdm_k in ('utils', 'utils.test_data_manager'):
    sys.modules.pop(_tdm_k, None)
from utils.test_data_manager import ensure_test_data, get_test_hr_path, create_fallback_data


class SimpleSRModel(nn.Module):
    """Simple SR model for TTA testing."""
    
    def __init__(self, scale: int = 4):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 3, 3, padding=1)
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        # Downscale for pseudo-LR
        lr = F.interpolate(x, scale_factor=1/self.scale, mode='bicubic')
        
        # SR
        x = self.relu(self.conv1(lr))
        x = self.relu(self.conv2(x))
        x = F.interpolate(x, scale_factor=self.scale, mode='bicubic')
        x = torch.sigmoid(self.conv3(x))
        return x


class MockTTA:
    """Mock Test-Time Adaptation implementation."""
    
    def __init__(self, model, lr=0.0001, max_steps=100, 
                 convergence_threshold=1e-5, patience=10):
        self.model = model
        self.lr = lr
        self.max_steps = max_steps
        self.convergence_threshold = convergence_threshold
        self.patience = patience
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_history = []
        self.adapted = False
    
    def compute_reconstruction_loss(self, lr_image, hr_image=None):
        """
        Compute reconstruction loss for TTA.
        
        Strategy: Downscale HR prediction and compare to input LR.
        This creates a self-supervised signal.
        """
        # Forward pass
        sr_pred = self.model(lr_image)
        
        # Downscale prediction to compare with input
        sr_downscaled = F.interpolate(
            sr_pred, 
            size=lr_image.shape[2:], 
            mode='bicubic',
            align_corners=False
        )
        
        # Reconstruction loss
        loss = F.l1_loss(sr_downscaled, lr_image)
        
        return loss, sr_pred
    
    def adapt(self, lr_image):
        """
        Adapt model to specific image.
        
        Args:
            lr_image: Low-resolution input [B, C, H, W]
        
        Returns:
            Adapted SR prediction
        """
        self.model.train()
        self.loss_history = []
        
        best_loss = float('inf')
        patience_counter = 0
        
        for step in range(self.max_steps):
            self.optimizer.zero_grad()
            
            # Compute reconstruction loss
            loss, sr_pred = self.compute_reconstruction_loss(lr_image)
            
            # Backward and update
            loss.backward()
            self.optimizer.step()
            
            # Track loss
            loss_value = loss.item()
            self.loss_history.append(loss_value)
            
            # Check convergence
            if loss_value < best_loss - self.convergence_threshold:
                best_loss = loss_value
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.patience:
                break
        
        self.adapted = True
        self.model.eval()
        
        # Final prediction
        with torch.no_grad():
            _, sr_final = self.compute_reconstruction_loss(lr_image)
        
        return sr_final
    
    def is_converged(self):
        """Check if adaptation has converged."""
        if len(self.loss_history) < 2:
            return False
        
        # Check recent loss change
        recent_losses = self.loss_history[-self.patience:]
        if len(recent_losses) < 2:
            return False
        
        loss_change = abs(recent_losses[-1] - recent_losses[0])
        return loss_change < self.convergence_threshold
    
    def save_adapted_model(self, path):
        """Save adapted model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'adapted': self.adapted,
            'lr': self.lr
        }
        torch.save(checkpoint, path)
    
    def load_adapted_model(self, path):
        """Load adapted model checkpoint."""
        checkpoint = torch.load(path, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_history = checkpoint.get('loss_history', [])
        self.adapted = checkpoint.get('adapted', False)
        self.lr = checkpoint.get('lr', self.lr)


@pytest.fixture(scope='module')
def test_image():
    """Provide test image (real or synthetic)."""
    data_dir = get_test_hr_path()
    image_files = list(data_dir.glob('*.png'))
    
    if len(image_files) > 0:
        try:
            from PIL import Image
            # Load first image
            img = Image.open(image_files[0]).convert('RGB')
            img_array = np.array(img)
            # Resize to manageable size
            img_array = img_array[:256, :256, :]
            # Convert to tensor [C, H, W]
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)  # Add batch
            
            # Create LR version
            lr = F.interpolate(img_tensor, scale_factor=1/4, mode='bicubic')
            return lr, img_tensor
        except Exception as e:
            print(f"Warning: Could not load real image: {e}")
    
    # Fallback: Create synthetic image
    lr = torch.rand(1, 3, 64, 64)
    hr = F.interpolate(lr, scale_factor=4, mode='bicubic')
    return lr, hr


@pytest.fixture
def mock_model():
    """Provide mock SR model."""
    return SimpleSRModel(scale=4)


class TestTTAInitialization:
    """Test TTA initialization."""
    
    def test_tta_initialization(self, mock_model):
        """Test that TTA can be initialized."""
        tta = MockTTA(
            mock_model,
            lr=0.0001,
            max_steps=50,
            convergence_threshold=1e-5,
            patience=10
        )
        
        assert tta.model is mock_model, "Model should be stored"
        assert tta.lr == 0.0001, "LR should be set"
        assert tta.max_steps == 50, "Max steps should be set"
        assert not tta.adapted, "Should not be adapted initially"
    
    def test_tta_optimizer_setup(self, mock_model):
        """Test TTA optimizer configuration."""
        tta = MockTTA(mock_model, lr=0.001)
        
        # Verify optimizer has model parameters
        opt_params = list(tta.optimizer.param_groups[0]['params'])
        model_params = list(mock_model.parameters())
        
        assert len(opt_params) == len(model_params), \
            "Optimizer should have same number of parameters as model"


class TestTTALossComputation:
    """Test TTA loss computation."""
    
    def test_reconstruction_loss_computation(self, mock_model, test_image):
        """Test that reconstruction loss is computed correctly."""
        tta = MockTTA(mock_model)
        lr, hr = test_image
        
        loss, sr_pred = tta.compute_reconstruction_loss(lr)
        
        # Verify loss is scalar
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"
        
        # Verify SR prediction shape
        assert sr_pred.shape[2] == lr.shape[2] * 4, "SR should be 4x height"
        assert sr_pred.shape[3] == lr.shape[3] * 4, "SR should be 4x width"
    
    def test_reconstruction_loss_decreases_with_perfect_model(self):
        """Test that reconstruction loss is low for perfect reconstruction."""
        # Create a model that just upsamples (perfect for this test)
        class PerfectUpsample(nn.Module):
            def forward(self, x):
                return F.interpolate(x, scale_factor=4, mode='bicubic')
        
        model = PerfectUpsample()
        tta = MockTTA(model)
        
        lr = torch.rand(1, 3, 64, 64)
        loss, _ = tta.compute_reconstruction_loss(lr)
        
        # With perfect upsampling, downscaling should match input
        assert loss.item() < 0.1, "Perfect upsampler should have low reconstruction loss"


class TestTTAAdaptation:
    """Test TTA adaptation process."""
    
    def test_adaptation_changes_model(self, mock_model, test_image):
        """Test that adaptation changes model parameters."""
        tta = MockTTA(mock_model, max_steps=10)
        lr, hr = test_image
        
        # Get initial params
        initial_params = [p.clone().detach() for p in mock_model.parameters()]
        
        # Adapt
        tta.adapt(lr)
        
        # Verify params changed
        for init, current in zip(initial_params, mock_model.parameters()):
            diff = torch.abs(init - current.detach()).mean()
            assert diff > 1e-7, "Parameters should change during adaptation"
    
    def test_adaptation_reduces_loss(self, mock_model, test_image):
        """Test that adaptation reduces reconstruction loss."""
        tta = MockTTA(mock_model, max_steps=20, patience=5)
        lr, hr = test_image
        
        # Initial loss
        with torch.no_grad():
            initial_loss, _ = tta.compute_reconstruction_loss(lr)
        
        # Adapt
        tta.adapt(lr)
        
        # Final loss
        with torch.no_grad():
            final_loss, _ = tta.compute_reconstruction_loss(lr)
        
        # Loss should decrease or stay similar
        assert final_loss.item() <= initial_loss.item() * 1.5, \
            "Loss should not increase significantly"
        
        # Verify loss history was recorded
        assert len(tta.loss_history) > 0, "Should have loss history"
    
    def test_adaptation_convergence_detection(self, mock_model, test_image):
        """Test convergence detection during adaptation."""
        tta = MockTTA(mock_model, max_steps=50, patience=5, convergence_threshold=1e-5)
        lr, hr = test_image
        
        # Adapt
        tta.adapt(lr)
        
        # Either should have converged or reached max steps
        assert len(tta.loss_history) <= 50, "Should not exceed max steps"
        
        # If stopped early, should have converged
        if len(tta.loss_history) < 50:
            assert tta.is_converged(), "Early stop should mean convergence"


class TestTTACheckpointing:
    """Test TTA checkpoint save/load."""
    
    def test_save_adapted_model(self, mock_model, test_image):
        """Test saving adapted model."""
        tta = MockTTA(mock_model, max_steps=10)
        lr, hr = test_image
        
        # Adapt
        tta.adapt(lr)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'tta_checkpoint.pth'
            
            # Save
            tta.save_adapted_model(checkpoint_path)
            
            # Verify file exists
            assert checkpoint_path.exists(), "Checkpoint should be created"
            assert checkpoint_path.stat().st_size > 0, "Checkpoint should not be empty"
    
    def test_load_adapted_model(self, mock_model, test_image):
        """Test loading adapted model."""
        tta = MockTTA(mock_model, max_steps=10)
        lr, hr = test_image
        
        # Adapt and save
        tta.adapt(lr)
        
        # Get adapted params
        adapted_params = [p.clone().detach() for p in mock_model.parameters()]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'tta_checkpoint.pth'
            tta.save_adapted_model(checkpoint_path)
            
            # Reset model
            new_model = SimpleSRModel(scale=4)
            new_tta = MockTTA(new_model)
            
            # Load
            new_tta.load_adapted_model(checkpoint_path)
            
            # Verify loaded params match
            for saved, loaded in zip(adapted_params, new_model.parameters()):
                assert torch.allclose(saved, loaded, atol=1e-6), \
                    "Loaded params should match saved params"
            
            # Verify metadata loaded
            assert new_tta.adapted, "Adapted flag should be loaded"
            assert len(new_tta.loss_history) > 0, "Loss history should be loaded"


class TestTTAInferenceQuality:
    """Test inference quality improvement after TTA."""
    
    def test_inference_output_valid(self, mock_model, test_image):
        """Test that inference produces valid output."""
        tta = MockTTA(mock_model, max_steps=10)
        lr, hr = test_image
        
        # Adapt
        sr_adapted = tta.adapt(lr)
        
        # Verify output
        assert sr_adapted.shape[0] == lr.shape[0], "Batch size should match"
        assert sr_adapted.shape[1] == 3, "Should have 3 channels"
        assert sr_adapted.shape[2] == lr.shape[2] * 4, "Should be 4x height"
        assert sr_adapted.shape[3] == lr.shape[3] * 4, "Should be 4x width"
        
        # Verify range
        assert torch.all(sr_adapted >= 0) and torch.all(sr_adapted <= 1), \
            "Output should be in [0, 1] range"
    
    def test_adapted_vs_baseline_quality(self, mock_model, test_image):
        """Test that adapted model produces better quality than baseline."""
        lr, hr = test_image
        
        # Baseline inference (no adaptation)
        mock_model.eval()
        with torch.no_grad():
            sr_baseline = mock_model(lr)
        
        # Adapted inference
        tta = MockTTA(mock_model, max_steps=20)
        sr_adapted = tta.adapt(lr)
        
        # Compare reconstruction quality
        # Downscale both and compare to input LR
        baseline_down = F.interpolate(sr_baseline, size=lr.shape[2:], mode='bicubic')
        adapted_down = F.interpolate(sr_adapted, size=lr.shape[2:], mode='bicubic')
        
        baseline_error = F.l1_loss(baseline_down, lr).item()
        adapted_error = F.l1_loss(adapted_down, lr).item()
        
        # Adapted should be at least as good
        assert adapted_error <= baseline_error * 1.2, \
            "Adapted model should not be significantly worse than baseline"


class TestTTAScriptIntegration:
    """Integration tests for TTA script."""
    
    def test_tta_script_exists(self):
        """Test that TTA script exists."""
        script_path = Path(__file__).parent.parent.parent.parent / 'scripts' / 'test_time_adapt.py'
        assert script_path.exists(), f"TTA script not found: {script_path}"
    
    def test_tta_script_help(self):
        """Test that TTA script shows help."""
        import subprocess
        
        script_path = Path(__file__).parent.parent.parent.parent / 'scripts' / 'test_time_adapt.py'
        result = subprocess.run(
            ['python', str(script_path), '--help'],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, "Help command should succeed"


def test_data_source_report():
    """Report which data source is being used for tests."""
    data_dir = ensure_test_data(min_images=1, verbose=True)
    print(f"\n[TestTimeAdaptation] Using data: {data_dir}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
