"""
Stage transition controller for automated multi-stage training.
Orchestrates advancement from Stage 1 to Stage 2 based on metrics.
"""
import logging
import traceback
from typing import Dict, Optional, Callable, Any
from pathlib import Path

from anime_sr.training.stage_state import StageState, StageStateManager
from anime_sr.training.convergence_monitor import ConvergenceMonitor

logger = logging.getLogger(__name__)


class StageTransitionController:
    """
    Controls automated transitions between training stages.
    
    Monitors training progress and triggers stage advancement when criteria met.
    
    Args:
        config: Training configuration
        state_manager: StageStateManager for persistence
        convergence_monitor: Optional ConvergenceMonitor for plateau detection
    """
    
    def __init__(
        self,
        config: Dict,
        state_manager: StageStateManager,
        convergence_monitor: Optional[ConvergenceMonitor] = None,
    ):
        self.config = config
        self.state_manager = state_manager
        self.convergence_monitor = convergence_monitor
        
        # Load or create state
        self.state = state_manager.load()
        if self.state is None:
            self.state = self._create_initial_state()
        
        # Transition criteria from config
        self.stage1_cfg = config.get('training', {}).get('stage1', {})
        self.stage2_cfg = config.get('training', {}).get('stage2', {})
        self.automation_cfg = config.get('training', {}).get('stage_automation', {})
    
    def _create_initial_state(self) -> StageState:
        """Create initial stage state from config."""
        stage1 = self.config.get('training', {}).get('stage1', {})
        stage2 = self.config.get('training', {}).get('stage2', {})
        automation = self.config.get('training', {}).get('stage_automation', {})
        
        return StageState(
            current_stage=1,
            stage2_enabled=stage2.get('enabled', False),
            min_epochs_stage1=automation.get('min_epochs_before_advance', 30),
            min_epochs_stage2=automation.get('stage2', {}).get('min_epochs', 50),
            target_metric_stage1=automation.get('stage1', {}).get('target_metric'),
            target_metric_stage2=automation.get('stage2', {}).get('target_metric'),
        )
    
    def should_advance_stage(self, current_stage: int, epoch: int, 
                            metrics: Dict[str, float]) -> bool:
        """
        Check if should advance from current stage to next.
        
        Args:
            current_stage: Current training stage (1 or 2)
            epoch: Current epoch number
            metrics: Dictionary of training metrics (loss, psnr, etc.)
            
        Returns:
            True if should advance to next stage
        """
        if current_stage == 1:
            return self._should_advance_from_stage1(epoch, metrics)
        return False  # No stage after 2
    
    def _should_advance_from_stage1(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """Check if should advance from Stage 1 to Stage 2."""
        # Check if Stage 2 is enabled
        if not self.state.stage2_enabled:
            return False
        
        # Check minimum epochs
        min_epochs = self.automation_cfg.get('min_epochs_before_advance', 30)
        if epoch < min_epochs:
            return False
        
        # Check if auto-advance on convergence is enabled
        if not self.automation_cfg.get('advance_on_convergence', True):
            return False
        
        # Check convergence using ConvergenceMonitor if available
        if self.convergence_monitor is not None:
            if not self.convergence_monitor.is_converged():
                # Also check plateau if configured
                if self.automation_cfg.get('advance_on_plateau', False):
                    if not self.convergence_monitor.is_plateaued():
                        return False
                else:
                    return False
        else:
            # Fallback: check if loss has plateaued manually
            if not self._check_loss_plateau(metrics):
                return False
        
        # Check target metric if specified
        target_metric_name = self.automation_cfg.get('stage1', {}).get('target_metric')
        target_threshold = self.automation_cfg.get('stage1', {}).get('target_threshold')
        if target_metric_name is not None and target_threshold is not None:
            metric_value = metrics.get(target_metric_name)
            if metric_value is not None and metric_value >= float(target_threshold):
                return True  # Target met, advance
        
        return True
    
    def _check_loss_plateau(self, metrics: Dict[str, float]) -> bool:
        """
        Simple manual check for loss plateau.
        Used when ConvergenceMonitor is not available.
        """
        # Conservative: do not advance without proper convergence monitoring
        # In practice, use ConvergenceMonitor for proper plateau detection
        return False
    
    def execute_transition(self, from_stage: int, to_stage: int,
                        checkpoint_path: str, metrics: Dict[str, float]) -> bool:
        """
        Execute transition from one stage to next.
        
        Args:
            from_stage: Stage transitioning from
            to_stage: Stage transitioning to
            checkpoint_path: Path to best checkpoint from previous stage
            metrics: Final metrics from previous stage
            
        Returns:
            True if transition successful
        """
        try:
            # Mark previous stage complete
            if from_stage == 1:
                self.state.mark_stage_complete(
                    stage=1,
                    checkpoint=checkpoint_path,
                    metric=metrics.get('val_loss', metrics.get('loss', float('inf')))
                )
                self.state.stage2_started = True
            
            # Update current stage
            self.state.current_stage = to_stage
            
            # Save state
            self.state_manager.save(self.state)
            
            print(f"\n{'='*60}")
            print(f"Stage Transition: {from_stage} -> {to_stage}")
            print(f"Checkpoint: {checkpoint_path}")
            print(f"Best metric: {metrics.get('val_loss', metrics.get('loss', 'N/A'))}")
            print(f"{'='*60}\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during stage transition: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def prepare_next_stage(self, stage: int) -> Optional[str]:
        """
        Prepare for the next stage.
        
        Returns:
            Path to checkpoint to load, or None if not available
        """
        if stage == 2:
            # Check if Stage 1 checkpoint available
            if self.state.stage1_best_checkpoint:
                return self.state.stage1_best_checkpoint
            
            # Check config for stage1_checkpoint
            stage2_cfg = self.config.get('training', {}).get('stage2', {})
            if stage2_cfg.get('use_stage1_checkpoint', False):
                checkpoint = stage2_cfg.get('stage1_checkpoint')
                if checkpoint and Path(checkpoint).exists():
                    return checkpoint
        
        return None
    
    def get_stage_status(self) -> Dict[str, Any]:
        """Get current stage status."""
        return {
            'current_stage': self.state.current_stage,
            'stage1_completed': self.state.stage1_completed,
            'stage1_best_metric': self.state.stage1_best_metric,
            'stage2_enabled': self.state.stage2_enabled,
            'stage2_started': self.state.stage2_started,
            'stage2_completed': self.state.stage2_completed,
            'can_advance_to_stage2': self.state.can_advance_to_stage2(),
        }
    
    def update_stage_progress(self, stage: int, epoch: int, metrics: Dict[str, float]):
        """Update progress for current stage."""
        if stage == 1:
            self.state.stage1_epochs_completed = epoch + 1
            # Update best metric if improved
            val_loss = metrics.get('val_loss', metrics.get('loss', float('inf')))
            if val_loss < self.state.stage1_best_metric:
                self.state.stage1_best_metric = val_loss
        elif stage == 2:
            self.state.stage2_epochs_completed = epoch + 1
            val_loss = metrics.get('val_loss', metrics.get('loss', float('inf')))
            if val_loss < self.state.stage2_best_metric:
                self.state.stage2_best_metric = val_loss
        
        # Save state periodically
        self.state_manager.save(self.state)
    
    def reset(self):
        """Reset stage state."""
        self.state = self._create_initial_state()
        self.state_manager.clear()
        print("Stage state reset. Starting from Stage 1.")
    
    def print_status(self):
        """Print current stage status."""
        print(self.state.get_summary())
