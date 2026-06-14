"""
Shared state for progressive training features.
Allows dataloader and finetuner to share dynamic configuration during training.
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ProgressiveCropState:
    """Shared state for progressive crop training."""
    enabled: bool = False
    current_crop_size: int = 128
    stages: List[dict] = field(default_factory=list)

    def get_crop_size(self, epoch: int) -> int:
        """Get crop size for given epoch."""
        if not self.enabled or not self.stages:
            return self.current_crop_size

        cumulative = 0
        for stage in self.stages:
            stage_epochs = stage.get('epochs', 10)
            if epoch < cumulative + stage_epochs:
                return stage.get('crop_size', 128)
            cumulative += stage_epochs

        return self.stages[-1].get('crop_size', 128) if self.stages else self.current_crop_size

    def update_crop_size(self, new_size: int):
        """Update current crop size."""
        self.current_crop_size = new_size


class TrainingState:
    """Global training state for sharing between components."""

    def __init__(self):
        self._progressive_crop = ProgressiveCropState()
        self._phase = 1
        self._global_step = 0

    @property
    def progressive_crop(self) -> ProgressiveCropState:
        return self._progressive_crop

    @progressive_crop.setter
    def progressive_crop(self, state: ProgressiveCropState):
        self._progressive_crop = state

    @property
    def phase(self) -> int:
        return self._phase

    @phase.setter
    def phase(self, value: int):
        self._phase = value

    @property
    def global_step(self) -> int:
        return self._global_step

    @global_step.setter
    def global_step(self, value: int):
        self._global_step = value


_training_state = TrainingState()


def get_training_state() -> TrainingState:
    """Get the global training state instance."""
    return _training_state


def set_progressive_crop(enabled: bool, stages: List[dict] = None, default_crop: int = 128):
    """Set progressive crop configuration in global state."""
    state = _training_state.progressive_crop
    state.enabled = enabled
    state.stages = stages or []
    state.current_crop_size = stages[0].get('crop_size', default_crop) if stages else default_crop


def get_current_crop_size(epoch: int) -> int:
    """Get current crop size for given epoch from global state."""
    return _training_state.progressive_crop.get_crop_size(epoch)