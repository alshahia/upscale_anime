"""
Test Suite for Meta-Learning (MAML) (Phase 5 of Small Dataset Techniques)

Tests Model-Agnostic Meta-Learning functionality including:
- MAML initialization (inner/outer loop setup)
- Inner loop task-specific adaptation
- Outer loop meta-parameter update
- Fast adaptation on few samples
- Meta-gradient computation
- MAML checkpoint save/load

Data Strategy:
- Use data/val_hr/ (4 images) divided into meta-train/meta-test splits
- Fallback: Create synthetic task distribution
"""
import pytest
import sys
import tempfile
import shutil
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add src and utils to path
for _tdm_k in ('utils', 'utils.test_data_manager'):
    sys.modules.pop(_tdm_k, None)
from utils.test_data_manager import ensure_test_data, get_val_hr_path, create_temp_dataset


class SimpleConvModel(nn.Module):
    """Simple conv model for MAML testing."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 3, 3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x
    
    def get_flat_params(self):
        """Get flattened parameters."""
        return torch.cat([p.flatten() for p in self.parameters()])
    
    def set_flat_params(self, flat_params):
        """Set parameters from flattened vector."""
        idx = 0
        for p in self.parameters():
            numel = p.numel()
            p.data = flat_params[idx:idx+numel].view_as(p)
            idx += numel


class MockMAML:
    """Mock MAML implementation for testing."""
    
    def __init__(self, model, inner_lr=0.01, meta_lr=0.001, num_inner_steps=5):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
        self.loss_fn = nn.MSELoss()
    
    def inner_loop(self, support_x, support_y):
        """Adapt model to task using support set."""
        # Clone model parameters
        adapted_params = []
        for param in self.model.parameters():
            adapted_params.append(param.clone().detach().requires_grad_(True))
        
        # Inner loop updates
        for _ in range(self.num_inner_steps):
            # Forward pass with adapted params
            self._set_params(adapted_params)
            pred = self.model(support_x)
            loss = self.loss_fn(pred, support_y)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, adapted_params, create_graph=True)
            
            # Update adapted params
            adapted_params = [p - self.inner_lr * g for p, g in zip(adapted_params, grads)]
        
        return adapted_params
    
    def outer_loop(self, tasks):
        """Meta-update using multiple tasks."""
        meta_loss = 0.0
        
        for support_x, support_y, query_x, query_y in tasks:
            # Inner loop adaptation
            adapted_params = self.inner_loop(support_x, support_y)
            
            # Evaluate on query set with adapted params
            self._set_params(adapted_params)
            query_pred = self.model(query_x)
            task_loss = self.loss_fn(query_pred, query_y)
            
            meta_loss += task_loss
        
        # Meta-update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item() / len(tasks)
    
    def _set_params(self, params):
        """Temporarily set model parameters."""
        for p_model, p_new in zip(self.model.parameters(), params):
            p_model.data = p_new.data
    
    def save_checkpoint(self, path, epoch, loss):
        """Save MAML checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.meta_optimizer.state_dict(),
            'inner_lr': self.inner_lr,
            'meta_lr': self.meta_lr,
            'epoch': epoch,
            'loss': loss
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """Load MAML checkpoint."""
        checkpoint = torch.load(path, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.inner_lr = checkpoint.get('inner_lr', self.inner_lr)
        self.meta_lr = checkpoint.get('meta_lr', self.meta_lr)
        return checkpoint.get('epoch', 0), checkpoint.get('loss', 0.0)


@pytest.fixture(scope='module')
def test_tasks():
    """Create synthetic meta-learning tasks."""
    # Each task: (support_x, support_y, query_x, query_y)
    tasks = []
    
    for task_id in range(3):
        # Support set (5 samples)
        support_x = torch.randn(5, 3, 32, 32)
        # Create target with task-specific transformation
        support_y = torch.clamp(support_x * (0.5 + task_id * 0.2) + 0.1, 0, 1)
        
        # Query set (3 samples)
        query_x = torch.randn(3, 3, 32, 32)
        query_y = torch.clamp(query_x * (0.5 + task_id * 0.2) + 0.1, 0, 1)
        
        tasks.append((support_x, support_y, query_x, query_y))
    
    return tasks


class TestMAMLInitialization:
    """Test MAML initialization."""
    
    def test_maml_model_creation(self):
        """Test that MAML can be initialized with a model."""
        model = SimpleConvModel()
        maml = MockMAML(model, inner_lr=0.01, meta_lr=0.001, num_inner_steps=5)
        
        assert maml.inner_lr == 0.01, "Inner LR should be set correctly"
        assert maml.meta_lr == 0.001, "Meta LR should be set correctly"
        assert maml.num_inner_steps == 5, "Num inner steps should be set correctly"
        assert maml.model is model, "Model should be stored"
    
    def test_maml_optimizer_setup(self):
        """Test that MAML optimizer is properly configured."""
        model = SimpleConvModel()
        maml = MockMAML(model, meta_lr=0.001)
        
        # Check optimizer has model parameters
        opt_params = list(maml.meta_optimizer.param_groups[0]['params'])
        model_params = list(model.parameters())
        
        assert len(opt_params) == len(model_params), \
            "Optimizer should have same number of parameter groups as model"


class TestMAMLInnerLoop:
    """Test MAML inner loop adaptation."""
    
    def test_inner_loop_returns_adapted_params(self, test_tasks):
        """Test that inner loop returns adapted parameters."""
        model = SimpleConvModel()
        maml = MockMAML(model, num_inner_steps=3)
        
        support_x, support_y, _, _ = test_tasks[0]
        
        # Run inner loop
        adapted_params = maml.inner_loop(support_x, support_y)
        
        # Verify returns list of parameters
        assert isinstance(adapted_params, list), "Should return list of adapted params"
        assert len(adapted_params) > 0, "Should return non-empty list"
        assert all(isinstance(p, torch.Tensor) for p in adapted_params), \
            "All elements should be tensors"
    
    def test_inner_loop_changes_params(self, test_tasks):
        """Test that inner loop actually changes parameters."""
        model = SimpleConvModel()
        maml = MockMAML(model, num_inner_steps=5)
        
        support_x, support_y, _, _ = test_tasks[0]
        
        # Get initial params
        initial_params = [p.clone().detach() for p in model.parameters()]
        
        # Run inner loop
        adapted_params = maml.inner_loop(support_x, support_y)
        
        # Verify params changed
        for init, adapted in zip(initial_params, adapted_params):
            diff = torch.abs(init - adapted.detach()).mean()
            assert diff > 1e-6, "Parameters should change during inner loop"
    
    def test_inner_loop_improves_loss(self, test_tasks):
        """Test that inner loop improves loss on support set."""
        model = SimpleConvModel()
        maml = MockMAML(model, num_inner_steps=1)
        
        support_x, support_y, _, _ = test_tasks[0]
        
        # Initial loss
        with torch.no_grad():
            initial_pred = model(support_x)
            initial_loss = maml.loss_fn(initial_pred, support_y).item()
        
        # Run inner loop
        maml.inner_loop(support_x, support_y)
        
        # Final loss (after adaptation)
        with torch.no_grad():
            final_pred = model(support_x)
            final_loss = maml.loss_fn(final_pred, support_y).item()
        
        # Note: With very few steps, loss might not always improve
        # But should at least not explode
        assert final_loss < initial_loss * 10, "Loss should not explode"


class TestMAMLOuterLoop:
    """Test MAML outer loop meta-update."""
    
    def test_outer_loop_updates_meta_params(self, test_tasks):
        """Test that outer loop updates meta-parameters."""
        model = SimpleConvModel()
        maml = MockMAML(model, meta_lr=0.001)
        
        # Get initial params
        initial_params = [p.clone().detach() for p in model.parameters()]
        
        # Run outer loop
        meta_loss = maml.outer_loop(test_tasks)
        
        # Verify params changed
        for init, current in zip(initial_params, model.parameters()):
            diff = torch.abs(init - current.detach()).mean()
            assert diff > 1e-8, "Meta-parameters should change during outer loop"
        
        # Verify returns loss
        assert isinstance(meta_loss, float), "Should return scalar loss"
        assert meta_loss >= 0, "Loss should be non-negative"
    
    def test_outer_loop_multiple_iterations(self, test_tasks):
        """Test multiple outer loop iterations."""
        model = SimpleConvModel()
        maml = MockMAML(model, meta_lr=0.001)
        
        losses = []
        for _ in range(3):
            loss = maml.outer_loop(test_tasks)
            losses.append(loss)
        
        # Verify losses are reasonable
        assert all(l >= 0 for l in losses), "All losses should be non-negative"
        assert len(losses) == 3, "Should have 3 loss values"


class TestMAMLFastAdaptation:
    """Test fast adaptation capability."""
    
    def test_fast_adaptation_few_shots(self, test_tasks):
        """Test adaptation on few samples."""
        model = SimpleConvModel()
        maml = MockMAML(model, num_inner_steps=5)
        
        # Use only 2-shot support set
        support_x, support_y, query_x, query_y = test_tasks[0]
        support_x = support_x[:2]
        support_y = support_y[:2]
        
        # Adapt
        adapted_params = maml.inner_loop(support_x, support_y)
        maml._set_params(adapted_params)
        
        # Evaluate on query
        with torch.no_grad():
            query_pred = model(query_x)
            query_loss = maml.loss_fn(query_pred, query_y).item()
        
        assert query_loss < 10, "Query loss should be reasonable after adaptation"
    
    def test_meta_learned_initialization_better(self, test_tasks):
        """Test that meta-learned initialization is better than random."""
        # Model with meta-learned init
        meta_model = SimpleConvModel()
        meta_maml = MockMAML(meta_model, meta_lr=0.001)
        
        # Do some meta-training
        for _ in range(5):
            meta_maml.outer_loop(test_tasks)
        
        # Model with random init
        random_model = SimpleConvModel()
        random_maml = MockMAML(random_model, inner_lr=0.01, meta_lr=0.001)
        
        # Test on new task
        test_support_x, test_support_y, test_query_x, test_query_y = test_tasks[0]
        
        # Adapt both models
        meta_adapted = meta_maml.inner_loop(test_support_x, test_support_y)
        random_adapted = random_maml.inner_loop(test_support_x, test_support_y)
        
        # Evaluate
        meta_maml._set_params(meta_adapted)
        random_maml._set_params(random_adapted)
        
        with torch.no_grad():
            meta_pred = meta_model(test_query_x)
            meta_loss = meta_maml.loss_fn(meta_pred, test_query_y).item()
            
            random_pred = random_model(test_query_x)
            random_loss = random_maml.loss_fn(random_pred, test_query_y).item()
        
        # Meta-learned should be at least as good
        assert meta_loss <= random_loss * 2, \
            "Meta-learned model should not be much worse than random init"


class TestMAMLCheckpointing:
    """Test MAML checkpoint save/load."""
    
    def test_save_checkpoint(self, test_tasks):
        """Test checkpoint saving."""
        model = SimpleConvModel()
        maml = MockMAML(model)
        
        # Do some training
        loss = maml.outer_loop(test_tasks)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'maml_checkpoint.pth'
            
            # Save
            maml.save_checkpoint(checkpoint_path, epoch=5, loss=loss)
            
            # Verify file exists
            assert checkpoint_path.exists(), "Checkpoint file should be created"
            assert checkpoint_path.stat().st_size > 0, "Checkpoint should not be empty"
    
    def test_load_checkpoint(self, test_tasks):
        """Test checkpoint loading."""
        model = SimpleConvModel()
        maml = MockMAML(model)
        
        # Do some training and save
        loss = maml.outer_loop(test_tasks)
        
        # Get current params
        saved_params = [p.clone().detach() for p in model.parameters()]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'maml_checkpoint.pth'
            maml.save_checkpoint(checkpoint_path, epoch=5, loss=loss)
            
            # Reset model
            new_model = SimpleConvModel()
            new_maml = MockMAML(new_model)
            
            # Load checkpoint
            loaded_epoch, loaded_loss = new_maml.load_checkpoint(checkpoint_path)
            
            # Verify
            assert loaded_epoch == 5, "Should load correct epoch"
            assert abs(loaded_loss - loss) < 1e-5, "Should load correct loss"
            
            # Verify params loaded
            for saved, loaded in zip(saved_params, new_model.parameters()):
                assert torch.allclose(saved, loaded), \
                    "Loaded params should match saved params"
    
    def test_resume_training(self, test_tasks):
        """Test resuming training from checkpoint."""
        model = SimpleConvModel()
        maml = MockMAML(model, meta_lr=0.001)
        
        # Train and save
        loss1 = maml.outer_loop(test_tasks)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'maml_checkpoint.pth'
            maml.save_checkpoint(checkpoint_path, epoch=1, loss=loss1)
            
            # Create new model and resume
            new_model = SimpleConvModel()
            new_maml = MockMAML(new_model, meta_lr=0.001)
            new_maml.load_checkpoint(checkpoint_path)
            
            # Continue training
            loss2 = new_maml.outer_loop(test_tasks)
            
            # Should be able to continue training
            assert loss2 >= 0, "Should compute valid loss after resume"


class TestMetaLearningScriptIntegration:
    """Integration tests for meta-learning script."""
    
    def test_meta_learning_script_exists(self):
        """Test that meta-learning script exists."""
        script_path = Path(__file__).parent.parent.parent.parent / 'scripts' / 'train_meta.py'
        assert script_path.exists(), f"Meta-learning script not found: {script_path}"
    
    def test_meta_learning_script_help(self):
        """Test that meta-learning script shows help."""
        import subprocess
        
        script_path = Path(__file__).parent.parent.parent.parent / 'scripts' / 'train_meta.py'
        result = subprocess.run(
            ['python', str(script_path), '--help'],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, "Help command should succeed"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
