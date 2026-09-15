# Extending Anime Upscaler GUI

How to add a model, a setting, or a backend without editing the
orchestrator. Each recipe ends with the one-line commit message you
should use.

These recipes assume you have read `docs/gui_audit.md` and
`docs/PUBLIC_API.md`.

---

## 1. Add a model kind (architecture)

Use `register_arch(...)` once at import time. The architecture becomes
available everywhere (GUI dropdown, preset catalog, build_run_job,
tests) without further wiring.

### Steps

1. Add the model class to `apps/anime_upscaler_gui/anime_upscaler_gui/archs.py`,
   alongside `RFDN`, `SRVGGNetCompact`, `ERANet`, etc.
2. Write two callables:
   - `detect(state_dict) -> bool` — matches your checkpoint's keys/shapes.
   - `loader(path) -> (nn.Module, state_dict)` — instantiates the model and
     loads the weights.
3. Add one line to `apps/anime_upscaler_gui/data/registry.json` under
   `presets` (so users see it in the preset catalog even before they
   drop a local checkpoint).
4. In the package init, register it. The cleanest place is the bottom of
   `archs.py` (next to the existing `register_arch(...)` calls), so the
   registry is fully populated by the time anything else imports it.

```python
# apps/anime_upscaler_gui/anime_upscaler_gui/archs.py
from . import register_arch, ArchSpec

def _my_detect(state_dict):
    return "first_layer.weight" in state_dict and \
           state_dict["first_layer.weight"].shape == (64, 3, 3, 3)

def _my_load(path):
    import torch
    from . import MyModel
    sd = torch.load(path, map_location="cpu")
    model = MyModel()
    model.load_state_dict(sd, strict=True)
    return model, sd

register_arch(ArchSpec(
    kind="my_arch",
    detect=_my_detect,
    loader=_my_load,
    default_scale=4,
    # capability flags: defaults are permissive; override only when needed
    # e.g. recurrent models need tta=False to avoid crashes:
))
```

After this, `is_supported_kind("my_arch")` returns True, the GUI dropdown
shows the preset, and `build("my_arch", path)` works.

### Tests

Add a fixture + tests to `tests/test_arch_registry.py` following the
`dummy_arch_registered` pattern.

### Commit message

`archs: register my_arch (kind=my_arch, scale=4)`

---

## 2. Add a setting to the panel

Adding a panel setting is a one-row addition to
`widgets/settings_spec.py::SETTING_SPECS`. The orchestrator does not
need to change. If the new setting also needs to flow into `RunJob`,
add one more line inside `build_run_job`.

### Steps

1. Add the field to `settings.Defaults` (the dataclass).
2. Add one entry to `SETTING_SPECS` in `widgets/settings_spec.py`:

```python
SettingSpec(
    key="my_setting",       # must match the Defaults field name
    label="My setting:",    # shown before the widget
    widget="spinbox",       # radio | checkbox | spinbox | combobox
    var_name="my_setting_var",  # tk var attribute name on the panel
    spin=(0, 100, 1),        # only for spinbox: (from, to, increment)
    coerce=_int,             # how to convert tk var -> Python value
    help_text="Hint shown after the widget.",
)
```

3. If the setting also affects `RunJob`:
   - In `controllers/job_builder.py::build_run_job`, add the new
     keyword to the `RunJob(...)` constructor, reading it via
     `value_provider("my_setting")`.
   - Add the new field to the `RunJob` dataclass in `pipeline/jobs.py`.
4. Add the field to the `RunJob` call site (only if the worker needs it
   to make a decision at runtime; most settings just flow into the queue).

If you only need the setting for advanced users (e.g. `nvenc_qp`,
`cascade_mode`), skip `SETTING_SPECS` and read it directly from
`Defaults` inside `build_run_job` — `_PANEL_KEYS` already partitions the
two groups.

### Tests

Append a SettingSpec entry to your local fork? Don't forget to add a
matching test to `tests/test_settings_spec.py` — it asserts every entry
has a valid widget kind, a non-empty var_name, and (for radio/combobox)
a non-empty choices list.

### Commit message

`settings: add my_setting to defaults + panel (via SETTING_SPECS)`

---

## 3. Add a backend

Backends (PyTorch, TensorRT, future ONNX) are selected via a
priority-ordered registry. Adding a backend is one call to
`register_backend(name, predicate, factory)`.

### Steps

1. Implement your backend class — anything callable that takes a single
   frame tensor and returns one. The minimum surface is `__call__(x) -> y`.
2. Write a `predicate(ctx) -> bool` that decides whether the backend is
   eligible for a given job. Use fields on `BackendSelectionCtx`
   (`device`, `kind`, `fp16`, `use_tensorrt`, `tta`, `batch_size`).
3. Write a `factory(ctx) -> backend_instance` that builds the backend.
4. Register it. The cleanest place is at the bottom of
   `pipeline/backends.py`, after the existing
`register_backend("PyTorch", ...)` call:

```python
from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline import (
    BackendSelectionCtx, register_backend,
)

def _my_predicate(c: BackendSelectionCtx) -> bool:
    # Only run on CUDA, fp16, not TTA, and only for the kinds we support.
    return (
        c.device.type == "cuda"
        and c.fp16
        and not c.tta
        and c.kind in {"rfdn_student", "srvgg_student"}
    )

def _my_factory(c: BackendSelectionCtx):
    return MyBackend(c.ckpt_path, c.kind, c.device, c.batch_size)

register_backend("MyBackend", _my_predicate, _my_factory)
```

After this, `select_backend(...)` will try your backend before falling
through to PyTorch. If your `factory` raises (e.g. CUDA OOM), the worker
logs a WARNING and tries the next entry; you do not need to handle the
fallback yourself.

### Tests

Append a fixture + tests to `tests/test_backend_registry.py`. The
existing tests cover priority order, factory exception fall-through,
predicate exception skipping, and the `make_backend` alias contract.

### Commit message

`pipeline: register MyBackend (priority=2, requires cuda + fp16)`

---

## 4. Add a model without code changes (preset catalog only)

If the architecture already exists in `archs.py`, you can register a
new preset via the user-extensible catalog. No code edit required.

1. Edit `apps/anime_upscaler_gui/data/registry.json` and append a
   `presets` entry:

```json
{
  "kind": "rfdn_student",
  "name": "My fine-tuned RFDN v2",
  "url": "https://example.com/my_rfdn_v2.pth",
  "scale": 4,
  "trained": true,
  "tainted": false,
  "size_mb": 18.4
}
```

2. Restart the app. The Models tab will list the new preset. Users can
download it with the existing **Custom URL** flow or by clicking the
preset row.

### Commit message

`registry: add My fine-tuned RFDN v2 preset (rfdn_student, 4x)`

---

## 5. Where things live

Quick map for when you only remember half the path:

```
apps/anime_upscaler_gui/
  anime_upscaler_gui/
    archs.py                          # register_arch, build, ArchSpec
    pipeline/
      backends.py                     # register_backend, select_backend
      jobs.py                         # RunJob dataclass
      worker.py                       # PipelineWorker
    controllers/
      job_builder.py                  # build_run_job, make_panel_value_provider
      model_resolver.py               # resolve_*_from_dropdown
    services/
      model_service.py                # sort_trained_first, format_dropdown_label,
                                       # pick_default_index
    widgets/
      settings_spec.py                # SETTING_SPECS (data-driven settings panel)
      settings_panel.py               # SettingsPanel (renders SETTING_SPECS)
    data/
      registry.json                   # user-extensible preset catalog
  tests/
    test_arch_registry.py             # register_arch round-trip
    test_backend_registry.py          # register_backend round-trip
    test_build_run_job.py             # build_run_job from Job + Settings
    test_settings_spec.py             # every SettingSpec has a valid widget
    test_model_service.py             # trained-first sort + default pick
    test_app_uses_build_run_job.py    # app.py -> build_run_job (no RunJob literal)
```