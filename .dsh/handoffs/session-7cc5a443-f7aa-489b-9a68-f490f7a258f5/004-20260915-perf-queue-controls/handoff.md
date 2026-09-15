# Handoff: GUI Queue Controls + GPU Codec (branch perf/queue-controls-gpu-codec)

**Date**: 2026-09-15
**Branch**: `perf/queue-controls-gpu-codec`
**Cut from**: `refactor/gui-extensibility` at commit `6f7c1f6`
**Active TODO**: Q1 documentation (DONE this turn). Q2-Q5 PLANNED.

## What this branch is for

Two orthogonal gaps surfaced from a hardware review of the GUI on RTX 4000:
1. No user-visible Pause / Cancel / Resume for in-flight jobs (current
   resume modal is for per-frame errors, not "pause and come back").
2. GPU-accelerated video codec paths exist in `RunJob` fields
   (`use_nvenc`, `nvenc_preset`, `nvenc_qp`, `use_tensorrt`,
   `cascade_mode`) but are not exposed in the Settings panel and not
   exercised in the worker.

Full plan: `docs/ROADMAP_QUEUE_CONTROLS_GPU_CODEC.md`.

## State after Q1 (this commit)

- Q1 commit: documentation only — `PLAN.md` (new section), `docs/ROADMAP_QUEUE_CONTROLS_GPU_CODEC.md` (new), `PROJECT_STATUS.md` (new entry).
- Q2-Q5 not started. Branch is clean and ready.

## Phases planned (each = one commit)

- **Q2 Pause/Cancel/Resume**: `JobControlEvent` dataclass, `_ctl_queue`,
  per-job `threading.Event`, three buttons in `input_panel`, status
  state `"paused"`. Tests: pause/resume video, cancel video, pause image.
- **Q3 GPU decode + GPU encode**: `_NvDecReader` (PyAV hwaccel=cuda with
  auto-fallback), `open_encoder()` honors `use_nvenc`/`nvenc_preset`/
  `nvenc_qp` explicitly, four new settings panel rows, job builder
  wiring. Tests: encoder argv, decoder fallback, panel render.
- **Q4 TF32 + async default + GPU util**: enable TF32 in `tensors.py`,
  flip `prefetch` default to `async`, fill in `widgets/gpu_monitor.py`
  (file already exists, empty), `JobEvent(kind="gpu_util", ...)` once
  per second. Tests: flags, default, widget poll cycle.
- **Q5 Tests + smoke + commit**: end-to-end smoke on `tmp/mid2s_job.py`,
  full GUI suite green, one commit per phase as above.

## Critical context

- Workspace policy: danger-full-access for file ops.
- Approval prompts disabled: do not set `sandbox_permissions`.
- Branch protection: never delete branches or rewrite history.
- Public API back-compat must be preserved (Phase E `DeprecatedAlias` proxies
  warn on underscore aliases; removal at 0.4.0).
- 6C/12T i7-9750H + Quadro RTX 4000 8 GB; CUDA 12.6; PyTorch 2.12.0+cu126.
- Smoke command: `.venv\Scripts\python.exe tmp/mid2s_job.py` from repo root.
- Test runner: `.venv\Scripts\python.exe -m pytest apps/anime_upscaler_gui/tests --ignore=.../test_video_end_to_end.py --ignore=.../test_assets.py`.
- Known environmental flakes: `test_video_end_to_end.py` (needs init.tcl),
  `test_assets.py` (concurrent permission errors). Not in scope.

## Next step for whoever continues

Implement Q2: the data model (`JobControlEvent`, cancel exception,
per-job event) and the worker plumbing first; the buttons in
`input_panel` come last. Make sure the existing thread safety tests
(`tests/test_threading.py`) still pass — `monkeypatch.setattr` on the
deprecated `_PipelineWorker` alias must continue to work (Phase E
contract).
