"""
Persistent state management for multi-stage training.
Tracks progress across Stage 1, Stage 2, and beyond.
"""
import json
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class StageState:
    """
    Persistent state for multi-stage training.
    
    Tracks:
    - Current stage
    - Stage completion status
    - Best checkpoints per stage
    - Metrics history
    - Transition criteria
    """
    
    # Current state
    current_stage: int = 1
    
    # Stage 1 (Knowledge Aggregation)
    stage1_completed: bool = False
    stage1_best_checkpoint: Optional[str] = None
    stage1_best_metric: float = float('inf')
    stage1_epochs_completed: int = 0
    stage1_converged: bool = False
    
    # Stage 2 (Student Distillation)
    stage2_enabled: bool = False
    stage2_started: bool = False
    stage2_completed: bool = False
    stage2_best_checkpoint: Optional[str] = None
    stage2_best_metric: float = float('inf')
    stage2_epochs_completed: int = 0
    stage2_converged: bool = False
    
    # Transition configuration
    min_epochs_stage1: int = 30
    min_epochs_stage2: int = 50
    target_metric_stage1: Optional[float] = None
    target_metric_stage2: Optional[float] = None
    
    # Metadata
    training_id: Optional[str] = None
    config_hash: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StageState':
        """Create StageState from dictionary."""
        return cls(**data)
    
    def mark_stage_complete(self, stage: int, checkpoint: str, metric: float):
        """Mark a stage as complete."""
        if stage == 1:
            self.stage1_completed = True
            self.stage1_best_checkpoint = checkpoint
            self.stage1_best_metric = metric
            self.stage1_converged = True
        elif stage == 2:
            self.stage2_completed = True
            self.stage2_best_checkpoint = checkpoint
            self.stage2_best_metric = metric
            self.stage2_converged = True
    
    def can_advance_to_stage2(self) -> bool:
        """Check if can advance from Stage 1 to Stage 2."""
        if not self.stage1_completed:
            return False
        if self.stage1_epochs_completed < self.min_epochs_stage1:
            return False
        return True
    
    def get_summary(self) -> str:
        """Get text summary of stage state."""
        lines = [
            f"Stage State Summary:",
            f"  Current Stage: {self.current_stage}",
            f"  Stage 1:",
            f"    Completed: {self.stage1_completed}",
            f"    Epochs: {self.stage1_epochs_completed}",
            f"    Best Metric: {self.stage1_best_metric:.6f}" if self.stage1_best_metric != float('inf') else "    Best Metric: N/A",
            f"    Checkpoint: {self.stage1_best_checkpoint}" if self.stage1_best_checkpoint else "    Checkpoint: N/A",
        ]
        
        if self.stage2_enabled:
            lines.extend([
                f"  Stage 2:",
                f"    Enabled: {self.stage2_enabled}",
                f"    Started: {self.stage2_started}",
                f"    Completed: {self.stage2_completed}",
                f"    Epochs: {self.stage2_epochs_completed}",
            ])
        
        return "\n".join(lines)


class StageStateManager:
    """
    Manages persistence of stage state.
    
    Handles:
    - Save/load stage state to JSON
    - State versioning
    - Recovery from interruptions
    """
    
    def __init__(self, state_dir: str = "checkpoints", training_id: Optional[str] = None):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.training_id = training_id or "default"
        self.state_file = self.state_dir / f"stage_state_{self.training_id}.json"
    
    def save(self, state: StageState):
        """Save stage state to file."""
        with open(self.state_file, 'w') as f:
            json.dump(state.to_dict(), f, indent=2)
    
    def load(self) -> Optional[StageState]:
        """Load stage state from file."""
        if not self.state_file.exists():
            return None
        
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
            return StageState.from_dict(data)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Warning: Could not load stage state: {e}")
            return None
    
    def exists(self) -> bool:
        """Check if state file exists."""
        return self.state_file.exists()
    
    def clear(self):
        """Clear stage state file."""
        if self.state_file.exists():
            self.state_file.unlink()
    
    def get_all_checkpoints(self) -> Dict[int, Optional[str]]:
        """Get all stage checkpoint paths."""
        state = self.load()
        if state is None:
            return {}
        
        return {
            1: state.stage1_best_checkpoint,
            2: state.stage2_best_checkpoint if state.stage2_enabled else None,
        }
