"""
Tests for new callbacks: ProgressiveCropCallback and LayerFreezeCallback.
"""
import unittest
import torch
import torch.nn as nn
from pathlib import Path
import sys
from anime_sr.training.callbacks import ProgressiveCropCallback, LayerFreezeCallback


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, num_blocks=6):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Linear(10, 10) for _ in range(num_blocks)
        ])
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class TestProgressiveCropCallback(unittest.TestCase):
    """Test progressive crop sizing callback."""
    
    def setUp(self):
        self.sizes = [64, 96, 128, 160]
        self.epochs_per_size = 25
        self.callback = ProgressiveCropCallback(
            sizes=self.sizes,
            epochs_per_size=self.epochs_per_size
        )
    
    def test_initialization(self):
        """Test callback initializes correctly."""
        self.assertEqual(self.callback.sizes, self.sizes)
        self.assertEqual(self.callback.epochs_per_size, 25)
        self.assertEqual(self.callback.current_size_idx, 0)
    
    def test_train_begin(self):
        """Test on_train_begin resets state."""
        self.callback.current_size_idx = 2
        self.callback.on_train_begin()
        self.assertEqual(self.callback.current_size_idx, 0)
    
    def test_size_progression(self):
        """Test that size progresses correctly over epochs."""
        logs = {}
        
        # Epoch 0: should use first size
        self.callback.on_epoch_begin(0, logs)
        self.assertEqual(logs['crop_size'], 64)
        
        # Epoch 24: still first size
        logs = {}
        self.callback.on_epoch_begin(24, logs)
        self.assertEqual(logs['crop_size'], 64)
        
        # Epoch 25: should advance to second size
        logs = {}
        self.callback.on_epoch_begin(25, logs)
        self.assertEqual(logs['crop_size'], 96)
        
        # Epoch 50: should advance to third size
        logs = {}
        self.callback.on_epoch_begin(50, logs)
        self.assertEqual(logs['crop_size'], 128)
        
        # Epoch 75: should advance to fourth size
        logs = {}
        self.callback.on_epoch_begin(75, logs)
        self.assertEqual(logs['crop_size'], 160)
    
    def test_size_capping(self):
        """Test that size doesn't go beyond available sizes."""
        logs = {}
        
        # Epoch 100: well beyond available sizes
        self.callback.on_epoch_begin(100, logs)
        
        # Should still use last size
        self.assertEqual(logs['crop_size'], 160)
    
    def test_get_current_size(self):
        """Test get_current_size method."""
        self.assertEqual(self.callback.get_current_size(), 64)
        
        # Progress to next size
        self.callback.on_epoch_begin(25, {})
        self.assertEqual(self.callback.get_current_size(), 96)


class TestLayerFreezeCallback(unittest.TestCase):
    """Test layer freezing callback."""
    
    def setUp(self):
        self.model = SimpleModel(num_blocks=6)
        
        self.freeze_schedule = {
            'freeze_layers': [{'type': 'blocks', 'count': 4}],
            'unfreeze_schedule': [
                {'epoch': 10, 'unfreeze_blocks': 1, 'lr_multiplier': 0.5},
                {'epoch': 20, 'unfreeze_blocks': 1, 'lr_multiplier': 0.4},
                {'epoch': 30, 'unfreeze_all': True, 'lr_multiplier': 0.3}
            ]
        }
        
        self.callback = LayerFreezeCallback(self.model, self.freeze_schedule)
    
    def test_initialization(self):
        """Test callback initializes correctly."""
        self.assertIs(self.callback.model, self.model)
        self.assertEqual(self.callback.freeze_schedule, self.freeze_schedule)
        self.assertEqual(len(self.callback.frozen_blocks), 0)
    
    def test_freeze_initial_blocks(self):
        """Test freezing initial blocks."""
        self.callback._freeze_initial_blocks(3)
        
        # Check first 3 blocks are frozen
        for i in range(3):
            for param in self.model.blocks[i].parameters():
                self.assertFalse(param.requires_grad)
        
        # Check remaining blocks are not frozen
        for i in range(3, 6):
            for param in self.model.blocks[i].parameters():
                self.assertTrue(param.requires_grad)
    
    def test_unfreeze_blocks(self):
        """Test unfreezing blocks."""
        # First freeze
        self.callback._freeze_initial_blocks(4)
        self.assertEqual(len(self.callback.frozen_blocks), 4)
        
        # Then unfreeze 2
        self.callback._unfreeze_blocks(2)
        self.assertEqual(len(self.callback.frozen_blocks), 2)
        
        # Check first 2 are unfrozen
        for i in range(2):
            for param in self.model.blocks[i].parameters():
                self.assertTrue(param.requires_grad)
        
        # Check remaining 2 are still frozen
        for i in range(2, 4):
            for param in self.model.blocks[i].parameters():
                self.assertFalse(param.requires_grad)
    
    def test_unfreeze_all(self):
        """Test unfreezing all blocks."""
        # Freeze some blocks
        self.callback._freeze_initial_blocks(4)
        self.assertEqual(len(self.callback.frozen_blocks), 4)
        
        # Unfreeze all
        self.callback._unfreeze_all()
        self.assertEqual(len(self.callback.frozen_blocks), 0)
        
        # All parameters should be trainable
        for param in self.model.parameters():
            self.assertTrue(param.requires_grad)
    
    def test_schedule_execution(self):
        """Test that unfreeze schedule is executed correctly."""
        # Setup: freeze first 4 blocks
        self.callback._freeze_initial_blocks(4)
        
        # Epoch 10: should unfreeze 1 block
        logs = {}
        self.callback.on_epoch_begin(10, logs)
        self.assertEqual(len(self.callback.frozen_blocks), 3)
        self.assertEqual(logs.get('lr_multiplier'), 0.5)
        
        # Epoch 20: should unfreeze another block
        logs = {}
        self.callback.on_epoch_begin(20, logs)
        self.assertEqual(len(self.callback.frozen_blocks), 2)
        self.assertEqual(logs.get('lr_multiplier'), 0.4)
        
        # Epoch 30: should unfreeze all
        logs = {}
        self.callback.on_epoch_begin(30, logs)
        self.assertEqual(len(self.callback.frozen_blocks), 0)
        self.assertEqual(logs.get('lr_multiplier'), 0.3)
    
    def test_no_execution_on_other_epochs(self):
        """Test that nothing happens on epochs not in schedule."""
        self.callback._freeze_initial_blocks(4)
        initial_frozen = self.callback.frozen_blocks.copy()
        
        # Epoch 5: not in schedule
        self.callback.on_epoch_begin(5, {})
        
        # Should remain the same
        self.assertEqual(self.callback.frozen_blocks, initial_frozen)


class TestCallbackIntegration(unittest.TestCase):
    """Integration tests for callbacks."""
    
    def test_progressive_crop_with_training(self):
        """Test progressive crop in simulated training."""
        callback = ProgressiveCropCallback(
            sizes=[64, 96, 128],
            epochs_per_size=10
        )
        
        # Simulate training
        sizes_over_time = []
        for epoch in range(30):
            logs = {}
            callback.on_epoch_begin(epoch, logs)
            sizes_over_time.append(logs['crop_size'])
        
        # Should progress: 64x10, 96x10, 128x10
        expected = [64]*10 + [96]*10 + [128]*10
        self.assertEqual(sizes_over_time, expected)
    
    def test_layer_freeze_with_model(self):
        """Test layer freeze with actual model training simulation."""
        model = SimpleModel(num_blocks=4)
        
        schedule = {
            'unfreeze_schedule': [
                {'epoch': 5, 'unfreeze_blocks': 1, 'lr_multiplier': 0.5}
            ]
        }
        
        callback = LayerFreezeCallback(model, schedule)
        callback._freeze_initial_blocks(3)
        
        # Check initial state
        for i in range(3):
            for p in model.blocks[i].parameters():
                self.assertFalse(p.requires_grad)
        for i in range(3, 4):
            for p in model.blocks[i].parameters():
                self.assertTrue(p.requires_grad)
        
        # Unfreeze at epoch 5
        callback.on_epoch_begin(5, {})
        
        # Check unfrozen state - block 0 should be unfrozen (first of frozen)
        for p in model.blocks[0].parameters():
            self.assertTrue(p.requires_grad)
        
        # Blocks 1 and 2 should still be frozen
        for i in range(1, 3):
            for p in model.blocks[i].parameters():
                self.assertFalse(p.requires_grad)


if __name__ == '__main__':
    unittest.main()
