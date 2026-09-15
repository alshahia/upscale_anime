# Anime Upscaler GUI — Public API Reference

This is the stable, public surface of `apps/anime_upscaler_gui`. Scripts,
tests, and downstream tooling should import these names. Underscore-
prefixed names (`_RunJob`, `_JobEvent`, etc.) are kept as aliases for
back-compat with older scripts (Phase A1); new code should use the public
names listed here.

All public names are also re-exported from the top-level package:

```python
import apps.anime_upscaler_gui.anime_upscaler_gui as g

g.RunJob          # pipeline.jobs.RunJob
g.PipelineWorker  # pipeline.worker.PipelineWorker
g.Settings        # settings.Settings
g.ModelRegistry   # registry.ModelRegistry
g.build           # archs.build
```

---

## 1. Pipeline (`apps.anime_upscaler_gui.pipeline`)

### `RunJob`

Immutable description of one job to process. Built by `build_run_job` (see
controllers below); the worker consumes these from its input queue.

```python
from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline import RunJob

job = RunJob(
    job_id=1,
    input_path=Path("frames/0001.png"),
    output_path=Path("out/0001_x4.png"),
    is_video=False,
    model_filename="RFDN_distill_v1_4x_student.pth",
    kind="rfdn_student",
    scale=4,
    outscale=4.0,
    fp16=True,
    device="cuda",
    batch_size=4,
    decode="cv2",
    prefetch="async",
    pin_memory="auto",
    downscale_max_edge=0,
    gpu_guard_mode="auto_downscale",
    tile_size=256,
    tile_overlap=8,
    tta=False,
    use_tensorrt=False,
    use_nvenc=True,
    nvenc_preset="p4",
    nvenc_qp=18,
    cascade_mode=None,
    cut_start_seconds=0.0,
    cut_end_seconds=0.0,
    on_frame_error=lambda idx, msg: "abort",
)
```

### `JobEvent`

An event the worker emits onto its output queue. The GUI consumes these
to update the status bar, progress bar, and preview.

```python
from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline import JobEvent

# JobEvent(kind="progress", job_id=1, ratio=0.42, fps=8.3, eta=12.5)
# JobEvent(kind="finished", job_id=1, output_path=Path("out/0001_x4.png"))
# JobEvent(kind="error", job_id=1, message="...")
# JobEvent(kind="frame_error", job_id=1, frame_idx=42, message="...")
```

### `PipelineWorker`

Threaded worker. Put `RunJob` instances on `in_queue`, read `JobEvent`s
from `out_queue`, and call `.start()`/`.stop()` to control lifecycle.

```python
import queue
import threading
from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline import PipelineWorker, RunJob

in_q: queue.Queue[RunJob] = queue.Queue()
out_q: queue.Queue[JobEvent] = queue.Queue()
worker = PipelineWorker(in_q, out_q, device_pref="cuda", fp16=True)
worker.start()

in_q.put(build_run_job(...))   # push a job

while True:
    ev = out_q.get()
    if ev.kind == "finished":
        break
    print(ev)
```

### Frame-error actions

Constants the frame-error callback should return:

```python
from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline import (
    SKIP_FRAME, SKIP_REST, ABORT_JOB, RETRY_FRAME,
)
```

### Backends registry (Phase B3)

The pipeline selects a backend (PyTorch / TensorRT / future ONNX) via a
priority-ordered registry. `select_backend()` walks the registry and
returns the first backend whose `predicate` is True; factory failures
log a WARNING and fall through.

```python
from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline import (
    REGISTRY, BackendRegistry, BackendSelectionCtx,
    register_backend, select_backend,
)
import torch

# REGISTRY[0] is "TensorRT" (priority 1, when available),
# REGISTRY[1] is "PyTorch" (always-on fallback).

# Add a new backend:
register_backend(
    "MyBackend",
    predicate=lambda c: c.device.type == "cuda",  # only on GPU
    factory=lambda c: MyBackend(c.ckpt_path, c.kind, c.device, c.fp16),
)
```

---

## 2. Settings (`apps.anime_upscaler_gui.settings`)

### `Settings`, `AppPaths`, `Defaults`, `QueueController`

`Settings` is the JSON-backed user settings file (auto-loads, validates,
and persists). `AppPaths` resolves the on-disk locations for settings,
queue, and presets. `Defaults` is the dataclass of every user-
configurable default. `QueueController` is the in-memory job queue with
atomic on-disk persistence.

```python
from apps.anime_upscaler_gui.anime_upscaler_gui.settings import (
    Settings, AppPaths, Defaults, QueueController, Job, JobStatus,
)

paths = AppPaths.from_default()
settings = Settings(paths)
print(settings.data.outscale)   # Defaults.outscale (float)
print(settings.data.fp16)       # Defaults.fp16 (bool)

qc = QueueController(paths)
qc.add(Job(input_path=..., output_path=..., model_filename=...))
qc.persist()                     # atomic write to disk
```

---

## 3. Registry (`apps.anime_upscaler_gui.registry`)

### `ModelRegistry`

Scans the configured pretrained folder for checkpoints and matches them
against known architectures.

```python
from apps.anime_upscaler_gui.anime_upscaler_gui.registry import ModelRegistry

reg = ModelRegistry(pretrained_dir=Path("pretrained"))
models = reg.scan_installed()      # list[InstalledModel]
presets = reg.presets               # list[PresetEntry] from data/registry.json
```

### `is_supported_kind`

Returns True when the architecture registry knows how to build a given
kind string.

```python
from apps.anime_upscaler_gui.anime_upscaler_gui.registry import is_supported_kind

is_supported_kind("rfdn_student")   # True
is_supported_kind("made_up_arch")  # False
```

### `InstalledModel`, `PresetEntry`

Dataclasses returned by `ModelRegistry.scan_installed()` and
`ModelRegistry.presets`. See `apps.anime_upscaler_gui.registry` for full
field lists.

---

## 4. Architectures (`apps.anime_upscaler_gui.archs`)

### `ArchSpec`, `Capability`

Dataclass describing one architecture: kind string, detector callable,
loader callable, default scale, and capabilities.

### `build(kind, ckpt)`

Instantiate the right model class for a kind. Raises ValueError if the
kind is not registered.

```python
from apps.anime_upscaler_gui.anime_upscaler_gui.archs import build, register_arch, ArchSpec
import torch

model = build("rfdn_student", "pretrained/RFDN_distill_v1_4x_student.pth")
model.eval().to("cuda")
```

### `register_arch(spec)`

Add a new architecture. After registration, `is_supported_kind` returns
True for the new kind and `build(...)` can instantiate it.

```python
from apps.anime_upscaler_gui.anime_upscaler_gui.archs import register_arch, ArchSpec

def _detect(state_dict):
    return "first_key" in state_dict and state_dict["first_key"].shape[0] == 64

def _load(path):
    return MyArch(), torch.load(path, map_location="cpu")

register_arch(ArchSpec(
    kind="my_arch",
    detect=_detect,
    loader=_load,
    default_scale=4,
    # capabilities are optional; defaults are permissive
))
```

### `registered_kinds`, `spec_of`

Enumerators for all registered architectures and per-kind lookups.

---

## 5. Controllers (`apps.anime_upscaler_gui.controllers`)

Pure-logic helpers extracted from the Tk orchestrator. Phase B2's
`build_run_job` is the canonical RunJob builder.

### `build_run_job`

Given a queued `Job`, a `value_provider` (live settings panel or a
test fake), a `Defaults` dataclass, and a few orchestrator-supplied
callables, returns a fully-built `RunJob`.

```python
from apps.anime_upscaler_gui.anime_upscaler_gui.controllers import (
    build_run_job, make_panel_value_provider, compute_output_path,
)

get_value = make_panel_value_provider(settings_panel)
run_job = build_run_job(
    job=queued_job,
    value_provider=get_value,
    data=settings.data,
    resolve_model_path=lambda: dropdown_model_path(),
    resolve_kind=lambda: dropdown_kind(),
    on_frame_error=on_frame_error_modal,
)
in_queue.put(run_job)
```

### `compute_output_path(input_path, output_mode, ...)`

Compute where the upscaled file is written: same-folder batch dir or a
user-chosen custom directory. Pure function, easy to test.

### Model-resolver helpers

`parse_dropdown_filename`, `resolve_from_dropdown`,
`resolve_kind_from_dropdown`, `resolve_path_from_dropdown` — all pure,
all unit-tested.

---

## 6. Services (`apps.anime_upscaler_gui.services`)

### `sort_trained_first(models)`

Sort installed models so trained (in-house) entries come first.

### `format_dropdown_label(model, supported_kind=True)`

Render one InstalledModel as a single dropdown label with metadata and
tag suffix.

### `pick_default_index(items, last_model="")`

Pick which dropdown row should be selected by default. Returns -1 when
the list is empty.

### Tag constants

`TAG_TAINTED`, `TAG_TRAINED`, `TAG_UNSUPPORTED`, `TAG_NO_ARCH` — the
suffixes appended to dropdown labels by `format_dropdown_label`.

---

## 7. Extension recipes

For recipes that walk through adding a model, setting, or backend, see
`apps/anime_upscaler_gui/docs/EXTENDING.md`.

---

## 8. Backwards compatibility

### 8.1 The underscore aliases

The following underscore-prefixed names are aliases kept for older
scripts (Phase A1):

```
_RunJob          -> RunJob
_JobEvent        -> JobEvent
_PipelineWorker  -> PipelineWorker
_Defaults        -> Defaults
_AppPaths        -> AppPaths
_Settings        -> Settings
_QueueController -> QueueController
_ModelRegistry   -> ModelRegistry
_is_supported_kind -> is_supported_kind
_make_backend    -> select_backend
```

### 8.2 Soft deprecation (Phase E)

Every alias in section 8.1 is wrapped by `DeprecatedAlias` from
`apps/anime_upscaler_gui/anime_upscaler_gui/_deprecation.py`. The proxy
behaves exactly like the underlying public name for every Python
operation (construction, `isinstance`, `pickle`, dataclass introspection,
attribute access, `setattr`/`delattr` for monkeypatching, equality,
`repr`, etc.). The only side effect is a `DeprecationWarning`.

The warning fires **on first use, not at import time**:

- `_RunJob(...)` raises a `DeprecationWarning` whose message names the
  public replacement and the removal milestone.
- Subsequent uses from the same call site are silent (one-shot dedup
  keyed by `(public_name, removal_version, caller_filename)`); this
  prevents log spam even in long-lived sessions.
- Importing the module that declares the alias is silent: only actual
  attribute access / calls trigger the warning.
- Dunder introspection (`__doc__`, `__module__`, `__hash__`, etc.) does
  NOT warn so `isinstance(_RunJob(...), RunJob)`, `pickle`, dataclass
  machinery, and debuggers continue to work.

Code that needs the unwrapped object without warning can read
`proxy.__wrapped__` (this is the documented escape hatch).

### 8.3 Removal timeline (single source of truth)

The removal milestone is declared once, in
`apps/anime_upscaler_gui/anime_upscaler_gui/_deprecation.py`, and copied
by each alias site:

| Version    | Behavior                                                        |
|------------|-----------------------------------------------------------------|
| `0.2.x`    | `DeprecationWarning` emitted on first use (this release)        |
| `0.3.x`    | same warning, escalated to `always` filter in CI (noisy build)   |
| `0.4.0`    | aliases removed                                                 |

New code should always use the public names from section 2. When
`0.4.0` ships, every alias above will raise `AttributeError`. Migrate
by replacing the underscore name with its public equivalent.

To grep for the removal milestone (CI policy check):

```bash
grep -R 'removal_version="0.4.0"' apps/anime_upscaler_gui/anime_upscaler_gui
```

To grep for all alias sites:

```bash
grep -R '_make_deprecated_alias\|make_alias(' apps/anime_upscaler_gui/anime_upscaler_gui
```