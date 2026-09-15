"""anime_upscaler_gui -- standalone Tk GUI for vendored anime SR models.

This package is the GUI application layer for the upscale_anime project.
It is self-contained: it vendors the model architectures it needs (archs.py)
so it can be moved to its own repo without dragging the rest of
upscale_anime's training pipeline.

Public API (Phase A1: stable names; underscore aliases kept for back-compat):

  Pipeline (apps.anime_upscaler_gui.pipeline):
    RunJob            -- one input file with all settings needed to process it
    JobEvent          -- progress / error / frame-error event for the GUI queue
    PipelineWorker    -- threaded worker that processes jobs from a queue
    SKIP_FRAME / SKIP_REST / ABORT_JOB / RETRY_FRAME -- resume actions

  Settings (apps.anime_upscaler_gui.settings):
    Settings          -- JSON-backed user settings
    AppPaths          -- resolved paths for settings + cache
    Defaults          -- dataclass with every user-configurable default
    QueueController   -- in-memory queue + atomic persistence

  Registry (apps.anime_upscaler_gui.registry):
    ModelRegistry     -- on-disk scan + preset catalog
    is_supported_kind -- True if the arch registry knows how to build a kind

  Architectures (apps.anime_upscaler_gui.archs):
    ArchSpec          -- (kind, detect, loader, default_scale) dataclass
    Capability        -- arch capabilities (tta, pad_multiple, ...)
    build(kind, ckpt) -- instantiate the right model class for a kind
    register_arch     -- register a new ArchSpec
    registered_kinds  -- list of all registered kinds

Deprecated aliases (kept for back-compat with existing scripts):
    _RunJob  -> RunJob
    _JobEvent -> JobEvent
    _PipelineWorker -> PipelineWorker
    _Defaults -> Defaults
    _AppPaths -> AppPaths
    _Settings -> Settings
    _QueueController -> QueueController
    _ModelRegistry -> ModelRegistry
    _is_supported_kind -> is_supported_kind
    _ModelDownloader -> ModelDownloader (in apps.anime_upscaler_gui.downloader)
"""

__version__ = "0.1.0"


# Public API re-exports. These are the names external scripts should use.
# The modules themselves keep the underscore aliases for back-compat.
from .pipeline import (  # noqa: E402,F401
    RunJob,
    JobEvent,
    PipelineWorker,
    SKIP_FRAME,
    SKIP_REST,
    ABORT_JOB,
    RETRY_FRAME,
)
from .settings import (  # noqa: E402,F401
    Settings,
    AppPaths,
    Defaults,
    QueueController,
)
from .registry import (  # noqa: E402,F401
    ModelRegistry,
    is_supported_kind,
)
from .archs import (  # noqa: E402,F401
    ArchSpec,
    Capability,
    build,
    register_arch,
    registered_kinds,
    spec_of,
)
