"""Typed state layer for the GUI.

Three types:
- `JobStatus`: str-based enum for queue states (pending/running/done/error/skipped/cancelled).
- `Job`: dataclass replacing the legacy `List[dict]` queue item.
- `FormState`: numeric form fields with range validation, extended by `_Defaults`.

`JobStatus` inherits from `str` so existing string comparisons continue to work:
`JobStatus.PENDING == "pending"` is True.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"

    def __str__(self) -> str:
        # Default Enum.__str__ returns "JobStatus.PENDING"; override so format strings
        # and str() calls render the lowercase value used by the legacy dict layer.
        return self.value


@dataclass
class Job:
    """A single upscaling task in the batch queue.

    Replaces the legacy dict literal used in `UpscaleGUI._jobs`. All fields except
    `id`, `input`, `output` have defaults so a job can be constructed incrementally
    (single-mode path leaves model_filename/kind/scale unset until `_on_start`).
    """
    id: int
    input: Path
    output: Path
    status: JobStatus = JobStatus.PENDING
    error: str = ""
    fps: float = 0.0
    infer_ms: float = 0.0
    model_filename: str = ""
    kind: str = ""
    scale: int = 4
    is_video: bool = False
    cut_start: float = 0.0
    cut_end: float = 0.0


@dataclass
class FormState:
    """Numeric form fields with range validation.

    `_Defaults` extends this so saved settings flow through the same validators.
    Invalid values raise `ValueError` at construction time. `_coerce` bypasses this
    by setting attributes directly; that path is for backward compat with old
    settings.json files and is tightened in Phase 4.
    """
    outscale: float = 2.0
    batch_size: int = 1
    tile_size: int = 256

    def __post_init__(self):
        if not 1.0 <= self.outscale <= 8.0:
            raise ValueError(f"outscale must be in [1.0, 8.0], got {self.outscale}")
        if not 1 <= self.batch_size <= 16:
            raise ValueError(f"batch_size must be in [1, 16], got {self.batch_size}")
        if not 64 <= self.tile_size <= 1024:
            raise ValueError(f"tile_size must be in [64, 1024], got {self.tile_size}")
