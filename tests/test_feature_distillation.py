"""
Tests for feature distillation module.
Covers feature-level knowledge distillation from teachers.
"""
import unittest
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from training.feature_distillation import FeatureDistillationLoss, MultiTeacherFeatureDistillation


class SimpleTestModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, num_blocks=4):
        super().__init__()
        # First block: 3 -> 16 channels
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU()
            )
        ])
        
        # Remaining blocks: 16 -> 16 channels
        for _ in range(num_blocks - 1):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU()
            ))
        
        self.upsample = nn.Conv2d(16, 3, 3, padding=1)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.upsample(x)


class TestFeatureDistillationLoss(unittest.TestCase):
    """Test feature distillation loss."""
    
    def setUp(self):
        self.teacher = SimpleTestModel(num_blocks=4)
        self.student = SimpleTestModel(num_blocks=4)
        
        # Set to eval mode
        self.teacher.eval()
        self.student.eval()
        
        self.layer_indices = [0, 1, 2, 3]
        self.fd_loss = FeatureDistillationLoss(
            layer_indices=self.layer_indices,
            weights=None,
            loss_type='l2'
        )
    
    def test_initialization(self):
        """Test that loss initializes correctly."""
        self.assertEqual(self.fd_loss.layer_indices, self.layer_indices)
        self.assertEqual(self.fd_loss.loss_type, 'l2')
        self.assertEqual(len(self.fd_loss.weights), len(self.layer_indices))
    
    def test_forward_pass(self):
        """Test forward pass computes loss."""
        input_tensor = torch.randn(2, 3, 32, 32)
        
        # Register hooks
        self.fd_loss.register_hooks(self.teacher, self.student)
        
        # Compute loss
        loss = self.fd_loss(self.student, self.teacher, input_tensor)
        
        # Check loss is a scalar tensor
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertTrue(loss.item() >= 0)  # L2 loss should be non-negative
    
    def test_different_loss_types(self):
        """Test different loss types."""
        input_tensor = torch.randn(2, 3, 32, 32)
        
        for loss_type in ['l2', 'l1', 'cosine']:
            fd_loss = FeatureDistillationLoss(
                layer_indices=[0, 1],
                loss_type=loss_type
            )
            fd_loss.register_hooks(self.teacher, self.student)
            
            loss = fd_loss(self.student, self.teacher, input_tensor)
            
            self.assertIsInstance(loss, torch.Tensor)
            self.assertTrue(loss.item() >= 0)
    
    def test_hooks_capture_features(self):
        """Test that hooks capture intermediate features."""
        input_tensor = torch.randn(1, 3, 32, 32)
        
        self.fd_loss.register_hooks(self.teacher, self.student)
        
        # Clear any previous features
        self.fd_loss.teacher_features.clear()
        self.fd_loss.student_features.clear()
        
        # Forward pass
        with torch.no_grad():
            _ = self.teacher(input_tensor)
        _ = self.student(input_tensor)
        
        # Check that features were captured
        self.assertGreater(len(self.fd_loss.teacher_features), 0)
        self.assertGreater(len(self.fd_loss.student_features), 0)
    
    def test_loss_decreases_with_similar_models(self):
        """Test that loss is lower when models are similar."""
        input_tensor = torch.randn(2, 3, 32, 32)
        
        # Create two identical models
        model1 = SimpleTestModel(num_blocks=4)
        model2 = SimpleTestModel(num_blocks=4)
        model2.load_state_dict(model1.state_dict())
        
        model1.eval()
        model2.eval()
        
        fd_loss = FeatureDistillationLoss(layer_indices=[0, 1])
        fd_loss.register_hooks(model1, model2)
        
        loss = fd_loss(model2, model1, input_tensor)
        
        # Loss should be very small for identical models
        self.assertLess(loss.item(), 0.01)
    
    def test_cleanup_hooks(self):
        """Test that hooks are properly cleaned up."""
        self.fd_loss.register_hooks(self.teacher, self.student)
        
        # Should have hooks registered
        self.assertGreater(len(self.fd_loss.teacher_hooks), 0)
        self.assertGreater(len(self.fd_loss.student_hooks), 0)
        
        # Remove hooks
        self.fd_loss._remove_hooks()
        
        # Should be empty
        self.assertEqual(len(self.fd_loss.teacher_hooks), 0)
        self.assertEqual(len(self.fd_loss.student_hooks), 0)


class TestMultiTeacherFeatureDistillation(unittest.TestCase):
    """Test multi-teacher feature distillation."""
    
    def setUp(self):
        self.teacher1 = SimpleTestModel(num_blocks=4)
        self.teacher2 = SimpleTestModel(num_blocks=4)
        self.student = SimpleTestModel(num_blocks=4)
        
        self.teachers = [self.teacher1, self.teacher2]
        
        self.multi_fd = MultiTeacherFeatureDistillation(
            teachers=self.teachers,
            teacher_weights=[0.5, 0.5],
            layer_indices=[0, 1, 2],
            loss_weight=0.1
        )
    
    def test_initialization(self):
        """Test multi-teacher FD initializes correctly."""
        self.assertEqual(len(self.multi_fd.teachers), 2)
        self.assertEqual(len(self.multi_fd.teacher_weights), 2)
        self.assertEqual(len(self.multi_fd.distillation_losses), 2)
        self.assertEqual(self.multi_fd.loss_weight, 0.1)
    
    def test_teachers_frozen(self):
        """Test that teachers are frozen."""
        for teacher in self.multi_fd.teachers:
            for param in teacher.parameters():
                self.assertFalse(param.requires_grad)
    
    def test_teachers_in_eval_mode(self):
        """Test that teachers are in eval mode."""
        for teacher in self.multi_fd.teachers:
            self.assertFalse(teacher.training)
    
    def test_forward_pass(self):
        """Test forward pass with multiple teachers."""
        input_tensor = torch.randn(2, 3, 32, 32)
        
        loss = self.multi_fd(self.student, input_tensor)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertTrue(loss.item() >= 0)
    
    def test_weighted_loss(self):
        """Test that teacher weights are applied."""
        input_tensor = torch.randn(2, 3, 32, 32)
        
        # Equal weights
        multi_fd_equal = MultiTeacherFeatureDistillation(
            teachers=self.teachers,
            teacher_weights=[0.5, 0.5],
            layer_indices=[0],
            loss_weight=1.0
        )
        
        # Unequal weights
        multi_fd_unequal = MultiTeacherFeatureDistillation(
            teachers=self.teachers,
            teacher_weights=[0.8, 0.2],
            layer_indices=[0],
            loss_weight=1.0
        )
        
        loss_equal = multi_fd_equal(self.student, input_tensor)
        loss_unequal = multi_fd_unequal(self.student, input_tensor)
        
        # Both should be valid losses
        self.assertTrue(loss_equal.item() >= 0)
        self.assertTrue(loss_unequal.item() >= 0)
    
    def test_different_layer_indices(self):
        """Test with different layer configurations."""
        for layers in [[0], [0, 1], [0, 1, 2, 3]]:
            multi_fd = MultiTeacherFeatureDistillation(
                teachers=self.teachers[:1],
                layer_indices=layers
            )
            
            input_tensor = torch.randn(1, 3, 32, 32)
            loss = multi_fd(self.student, input_tensor)
            
            self.assertIsInstance(loss, torch.Tensor)


class TestFeatureDistillationIntegration(unittest.TestCase):
    """Integration tests with realistic scenarios."""
    
    def test_with_batch_sizes(self):
        """Test with different batch sizes."""
        teacher = SimpleTestModel(num_blocks=4)
        student = SimpleTestModel(num_blocks=4)
        
        fd_loss = FeatureDistillationLoss(layer_indices=[0, 1])
        fd_loss.register_hooks(teacher, student)
        
        for batch_size in [1, 2, 4, 8]:
            input_tensor = torch.randn(batch_size, 3, 32, 32)
            loss = fd_loss(student, teacher, input_tensor)
            
            self.assertIsInstance(loss, torch.Tensor)
            self.assertTrue(loss.item() >= 0)
    
    def test_with_different_resolutions(self):
        """Test with different input resolutions."""
        teacher = SimpleTestModel(num_blocks=4)
        student = SimpleTestModel(num_blocks=4)
        
        fd_loss = FeatureDistillationLoss(layer_indices=[0, 1])
        fd_loss.register_hooks(teacher, student)
        
        for size in [16, 32, 64, 128]:
            input_tensor = torch.randn(1, 3, size, size)
            loss = fd_loss(student, teacher, input_tensor)
            
            self.assertIsInstance(loss, torch.Tensor)
    
    def test_gradient_flow(self):
        """Test that gradients flow through student."""
        teacher = SimpleTestModel(num_blocks=4)
        student = SimpleTestModel(num_blocks=4)
        
        fd_loss = FeatureDistillationLoss(layer_indices=[0, 1])
        fd_loss.register_hooks(teacher, student)
        
        # Set student to train mode
        student.train()
        
        # Ensure some parameters require grad
        for param in student.parameters():
            param.requires_grad = True
        
        input_tensor = torch.randn(2, 3, 32, 32)
        
        # Forward and backward
        loss = fd_loss(student, teacher, input_tensor)
        loss.backward()
        
        # Check gradients exist
        has_gradients = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in student.parameters()
        )
        self.assertTrue(has_gradients)


if __name__ == '__main__':
    unittest.main()
