"""
Tests for stage automation functionality.
"""
import pytest
import sys
import json
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import directly from modules
import importlib.util

spec = importlib.util.spec_from_file_location("stage_state", Path(__file__).parent.parent / 'src' / 'training' / 'stage_state.py')
stage_state_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage_state_module)
StageState = stage_state_module.StageState
StageStateManager = stage_state_module.StageStateManager


class TestStageState:
    """Test StageState dataclass."""
    
    def test_initialization(self):
        state = StageState()
        assert state.current_stage == 1
        assert not state.stage1_completed
        assert not state.stage2_enabled
        assert state.stage1_best_metric == float('inf')
    
    def test_mark_stage_complete(self):
        state = StageState()
        state.mark_stage_complete(1, "checkpoint.pth", 0.045)
        
        assert state.stage1_completed
        assert state.stage1_best_checkpoint == "checkpoint.pth"
        assert state.stage1_best_metric == 0.045
        assert state.stage1_converged
    
    def test_can_advance_to_stage2(self):
        state = StageState()
        state.stage2_enabled = True
        
        # Can't advance if not complete
        assert not state.can_advance_to_stage2()
        
        # Mark complete but not enough epochs
        state.stage1_completed = True
        state.stage1_epochs_completed = 20
        state.min_epochs_stage1 = 30
        assert not state.can_advance_to_stage2()
        
        # Now can advance
        state.stage1_epochs_completed = 35
        assert state.can_advance_to_stage2()
    
    def test_to_dict(self):
        state = StageState(
            current_stage=2,
            stage1_completed=True,
            stage1_best_checkpoint="best.pth",
        )
        data = state.to_dict()
        
        assert data['current_stage'] == 2
        assert data['stage1_completed'] == True
        assert data['stage1_best_checkpoint'] == "best.pth"
    
    def test_from_dict(self):
        data = {
            'current_stage': 2,
            'stage1_completed': True,
            'stage1_best_checkpoint': 'best.pth',
            'stage1_best_metric': 0.04,
        }
        state = StageState.from_dict(data)
        
        assert state.current_stage == 2
        assert state.stage1_completed
        assert state.stage1_best_checkpoint == 'best.pth'
        assert state.stage1_best_metric == 0.04


class TestStageStateManager:
    """Test StageStateManager persistence."""
    
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StageStateManager(state_dir=tmpdir, training_id="test")
            
            # Create and save state
            state = StageState(
                current_stage=2,
                stage1_completed=True,
                stage1_best_checkpoint="test.pth",
            )
            manager.save(state)
            
            # Load state
            loaded = manager.load()
            assert loaded is not None
            assert loaded.current_stage == 2
            assert loaded.stage1_completed
            assert loaded.stage1_best_checkpoint == "test.pth"
    
    def test_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StageStateManager(state_dir=tmpdir, training_id="test")
            
            # Initially doesn't exist
            assert not manager.exists()
            
            # Save and now exists
            state = StageState()
            manager.save(state)
            assert manager.exists()
    
    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StageStateManager(state_dir=tmpdir, training_id="test")
            
            # Save state
            state = StageState()
            manager.save(state)
            assert manager.exists()
            
            # Clear
            manager.clear()
            assert not manager.exists()
    
    def test_get_all_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StageStateManager(state_dir=tmpdir, training_id="test")
            
            state = StageState(
                stage1_best_checkpoint="stage1.pth",
                stage2_enabled=True,
                stage2_best_checkpoint="stage2.pth",
            )
            manager.save(state)
            
            checkpoints = manager.get_all_checkpoints()
            assert checkpoints[1] == "stage1.pth"
            assert checkpoints[2] == "stage2.pth"


class TestStageTransitionController:
    """Test StageTransitionController."""
    
    def test_should_advance_from_stage1_convergence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = importlib.util.spec_from_file_location(
                "stage_controller", 
                Path(__file__).parent.parent / 'src' / 'training' / 'stage_controller.py'
            )
            controller_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(controller_module)
            StageTransitionController = controller_module.StageTransitionController
            
            config = {
                'training': {
                    'stage1': {'enabled': True, 'epochs': 100},
                    'stage2': {'enabled': True},
                    'stage_automation': {
                        'enabled': True,
                        'auto_advance': True,
                        'min_epochs_before_advance': 30,
                        'advance_on_convergence': True,
                    }
                }
            }
            
            manager = StageStateManager(state_dir=tmpdir, training_id="test")
            controller = StageTransitionController(config, manager)
            
            # Set up converged state
            controller.state.stage1_completed = True
            controller.state.stage1_epochs_completed = 35
            controller.state.stage2_enabled = True
            
            # Should be able to advance
            metrics = {'val_loss': 0.04, 'psnr': 35.5}
            assert controller.should_advance_stage(1, 35, metrics)
    
    def test_should_not_advance_before_min_epochs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = importlib.util.spec_from_file_location(
                "stage_controller", 
                Path(__file__).parent.parent / 'src' / 'training' / 'stage_controller.py'
            )
            controller_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(controller_module)
            StageTransitionController = controller_module.StageTransitionController
            
            config = {
                'training': {
                    'stage_automation': {
                        'enabled': True,
                        'min_epochs_before_advance': 30,
                    }
                }
            }
            
            manager = StageStateManager(state_dir=tmpdir, training_id="test")
            controller = StageTransitionController(config, manager)
            controller.state.stage2_enabled = True
            controller.state.stage1_completed = True
            
            # Too early - only 20 epochs
            controller.state.stage1_epochs_completed = 20
            metrics = {'val_loss': 0.04}
            assert not controller.should_advance_stage(1, 20, metrics)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
