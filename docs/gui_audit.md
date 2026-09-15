# GUI Audit: apps/anime_upscaler_gui

Scope: the Tk-based GUI package (apps/anime_upscaler_gui/).
Goal: identify the structural problems that force hardcoding every new
model/setting/backend, and propose a refactor that turns each into a
one-line addition.

## TL;DR

1. Adding a model kind touches 5 files. No central registry.
2. Adding a setting touches 4 files. Settings panel is hand-coded row by row.
3. app.py is a 878-line god class with ~10 mixed responsibilities.
4. pipeline.py is a 1002-line god module mixing tensor helpers, pinned
   pool, model loaders, backend factory, video encoder, frame-error
   protocol, progress events.
5. Job construction is a 30-arg inline literal in app.py:_enqueue_next.
6. Public API is unclear. Underscore-prefixed classes are imported
   externally (tmp/mid2s_job.py, tmp/srvgg_distill_*.py, etc.).
7. Backend selection is hardcoded string switches.
8. Model dropdown formatting is hardcoded in app.py:_refresh_model_dropdown.
9. No tests for the GUI package itself.
10. _CutWindow hardcodes HH:MM:SS format.

## Detailed findings

### 1. app.py (878 LOC, single class)

UpscaleGUI does everything:
  - Builds the entire UI (_build_ui, _build_preview_section,
    _build_actions_section, _make_scrollable).
  - Owns the worker thread (self._in_queue, self._evt_queue,
    self._worker, self._running).
  - Owns the queue controller (self.queue_ctrl).
  - Owns the model registry (self.registry).
  - Owns the settings (self.settings, self.settings.data).
  - Builds the menu bar (self.menubar = _MenuBar(self, self)).
  - Owns keyboard shortcuts (_bind_shortcuts).
  - Owns frame-error modals (_ask_frame_error_action, _wait_resume_action).
  - Owns preflight checks (_preflight_check).
  - Owns job lifecycle (_on_start, _enqueue_next, _job_finished,
    _poll_events).
  - Owns the cut-window integration.
  - Owns theme application.
  - Owns log-panel setup.
  - Owns window-geometry settings persistence.

Every new feature (model kind, setting, output option, new tab) requires
adding methods to this class.

### 2. pipeline.py (1002 LOC)

Single file containing:
  - Job / event dataclasses (_RunJob, _JobEvent).
  - Resume-action constants (SKIP_FRAME, SKIP_REST, ABORT_JOB, RETRY_FRAME).
  - TTA forward helper (_tta_forward).
  - Cascade counter (_cascade_count).
  - Pinned-memory pool (_PinnedPool).
  - Tensor helpers (_to_tensor, _to_tensor_batch, _tensor_to_bgr,
    _tensor_to_bgr_batch).
  - Resize / downscale helpers (_resize_keep_ar, _downscale_if_needed).
  - Backend factory (_make_backend, _PyTorchBackend, _OnnxBackend).
  - Video encoder glue (F_ENC_DIED, _encode_write).
  - The pipeline worker thread (_PipelineWorker with _run_one, _run_image,
    _run_video).
  - Batched-iteration helpers (_batched, _find_ckpt).

Adding a new backend or a new video output format requires touching several
of these sections in lockstep.

### 3. archs.py (1034 LOC)

The architectures are correctly vendored and the ArchSpec registry is
well-designed (register_arch, unregister_arch, registered_kinds, spec_of,
capabilities, is_supported_kind, detect_kind_from_state, build). However,
every kind requires hand-editing four registration spots: model class,
_detect_*, _load_*, _register_builtin_archs.

Mitigation: add a register.py companion file with
register_external(kind, model_class, detector, loader) so external packages
can register new kinds without editing archs.py.

### 4. widgets/settings_panel.py (134 LOC, 6 manual rows)

Every setting is hand-coded:
  - ttk.Label(...) + ttk.Radiobutton(...).pack(side="left", padx=...).
  - ttk.Spinbox(...) with manual from_/to/increment/width.
  - ttk.Combobox(...) with hardcoded values=[...].
  - Manual theme-swatch Canvas creation inside the loop.
  - Manual bind_return walking the widget tree to find spinboxes.

Adding a setting = ~10 lines of new widget code here, plus adding the field
to _Defaults, plus wiring it through _enqueue_next in app.py, plus adding
a _RunJob field in pipeline.py.

### 5. widgets/model_panel.py (19 LOC) and app.py:_refresh_model_dropdown

_refresh_model_dropdown builds the dropdown items by hand-formatting a
f-string inside app.py (f"{m.filename}  ({m.kind or '?'}, {m.scale}x,
{m.size_mb:.1f} MB){tag}"). Trained/tainted/unsupported tags are hardcoded
in this method. Adding a new tag (e.g. "[TILED]", "[GAUSSIAN]") requires
editing this method.

### 6. Public API surface

The underscore-prefixed classes are imported externally:

  # from tmp/mid2s_job.py and the SRVGG distill jobs I just wrote:
  from apps.anime_upscaler_gui.anime_upscaler_gui import pipeline as P
  job = P._RunJob(...)
  worker = P._PipelineWorker(in_q, out_q)

Naive rename will silently break these scripts. Phase A1 keeps aliases.

### 7. _CutWindow

Hardcoded HH:MM:SS. No audio cut, no timeline slider without rewriting it.

### 8. Tests

apps/anime_upscaler_gui/tests/ exists but covers older anime_upscaler
training modules (per the grep earlier), not the GUI itself. Phase C will
add GUI-level tests.

## Refactor plan

### Phase Audit (this document)

Findings + plan only. No code changes.

### Phase A: Pure code reorganization (no behavior change)

A1. Rename _RunJob to RunJob, _JobEvent to JobEvent, _PipelineWorker to
    PipelineWorker, _Defaults to Defaults, _Settings to Settings,
    _AppPaths to AppPaths. Keep underscore aliases (RunJob = _RunJob etc.)
    and emit a DeprecationWarning. Update internal call sites in app.py,
    widgets/*.py, registry.py, pipeline.py. Update external call sites in
    tmp/mid2s_job.py, tmp/srvgg_distill_*.py, scripts/step2_video_test.py,
    scripts/step2_visual_pair.py, scripts/step3_*.py.

A2. Split pipeline.py into pipeline/jobs.py, pipeline/backends.py,
    pipeline/tensors.py, pipeline/encoder.py, pipeline/worker.py, plus a
    pipeline/__init__.py that re-exports for backwards compatibility.
    Pure file reorganization; same symbols.

A3. Extract controllers/queue_controller.py,
    controllers/settings_controller.py, controllers/job_builder.py,
    controllers/preflight.py, controllers/model_resolver.py from app.py.
    In Phase A3 we move the existing methods verbatim; in Phase B we
    rewrite them as pure functions.

A4. Extract services/model_service.py with the dropdown format,
    trained-first sort, and default pick. app.py:_refresh_model_dropdown
    becomes a 3-line wrapper.

After Phase A: imports succeed, behavior unchanged, external scripts keep
working (via aliases), existing tests still pass.

### Phase B: Data-driven settings + job building

B1. settings_spec.py. Introduce SETTING_SPECS: list[SettingSpec]. Each
    spec is (key, label, kind, default, choices, min, max, increment,
    group, depends_on, help_text). _Defaults stays the dataclass for
    serialization; the spec drives UI rendering + validation. Rewrite
    widgets/settings_panel.py to iterate specs.

B2. controllers/job_builder.py:build_run_job(job, settings, model).
    Pure function. Reads a RUN_JOB_FIELDS mapping table
    (key, source="settings"|"job"|"literal"). New setting with a
    SETTING_SPEC row + a RUN_JOB_FIELDS row = automatically picked up.
    app.py:_enqueue_next becomes self._in_queue.put(build_run_job(...)).

B3. pipeline/backends.py:BackendRegistry. register_backend(name, factory_fn)
    + select_backend(model, kind, device, opts). Adding DirectML / OpenVINO
    = one register call.

After Phase B: adding a model = add a row to data/registry.json (or
PRESET_CATALOG in registry.py). Adding a setting = add a dataclass field
+ a SETTING_SPEC entry + (optionally) a RUN_JOB_FIELDS row. Adding a
backend = one register_backend(name, factory) call.

### Phase C: Documentation + tests

C1. apps/anime_upscaler_gui/__init__.py. Document and re-export the public
    surface: RunJob, JobEvent, PipelineWorker, Settings, AppPaths,
    InstalledModel, PresetEntry, ArchSpec, Capability, build, register_arch.

C2. docs/PUBLIC_API.md. Same list, with usage examples.

C3. New tests:
    - tests/test_job_builder.py: given a Job + Settings + InstalledModel,
      build_run_job returns a RunJob with every specd field.
    - tests/test_settings_spec.py: every SETTING_SPEC has a valid widget.
    - tests/test_backend_registry.py: register / select round-trip.
    - tests/test_model_service.py: trained-first sort, default pick.

C4. docs/USER_GUIDE.md update. New "How to add a model / setting / backend"
    recipes.

## Risk + effort

  Phase    Risk              Effort
  Audit    None              30 min
  A        Low (mechanical)  ~3-4 h equivalent
  B        Medium (real refactor) ~6-8 h equivalent
  C        Low (docs + tests) ~2 h equivalent

Phase A alone cleans up naming and splits modules; Phase B is what
actually makes future additions one-line changes.

## What you may have missed

  - External API surface (tmp/mid2s_job.py, tmp/srvgg_distill_*.py,
    scripts/step2_video_test.py, scripts/step2_visual_pair.py,
    scripts/step3_*.py all import the underscore-prefixed names).
  - apps/anime_upscaler_gui/data/registry.json is already the
    user-extensible preset catalog. Adding a community preset shouldn't
    require editing code at all.
  - apps/anime_upscaler_gui/scripts/ has older scripts that vendored
    architectures; they may reference the old names and need updating in
    lockstep with Phase A1.
  - docs/USER_GUIDE.md has a "How to add a model" section that will need
    updating for Phase C, not just docs/comments.
  - apps/anime_upscaler_gui/tests/ exists. Phase A should be validated
    against it before moving to Phase B.
