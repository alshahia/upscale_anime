# Anime Upscaler GUI

Standalone Tkinter desktop app for running the vendored anime super-resolution models over images and videos. It can be copied out of the main repository and keeps settings outside the source tree.

## Documentation

- [`docs/EXTENDING.md`](docs/EXTENDING.md) -- recipes for adding a new model kind, setting, backend, or preset (no orchestrator edits required).
- [`../../docs/PUBLIC_API.md`](../../docs/PUBLIC_API.md) -- stable public surface (`RunJob`, `JobEvent`, `PipelineWorker`, `Settings`, `ModelRegistry`, `build`, `register_arch`, `build_run_job`, `select_backend`, ...) for downstream scripts.
- [`docs/onboarding.md`](docs/onboarding.md) -- first-time user walkthrough.
- [`docs/themes.md`](docs/themes.md) -- theme tokens and accessibility.
- [`docs/dev/widgets.md`](docs/dev/widgets.md) -- widget internals for contributors.

## Quick start

From the repository root:

```powershell
apps\anime_upscaler_gui\.venv\Scripts\python.exe -m apps.anime_upscaler_gui
```

From a standalone copy, install `requirements.txt` into a local virtual environment and run:

```bash
python -m anime_upscaler_gui
```

Double-click `LAUNCHER.bat` on Windows or run `./LAUNCHER.sh` on Linux/macOS.

## First run

1. Open the **Models** tab and use **Re-scan** after placing checkpoints in the configured pretrained folder.
2. If no local checkpoint is available, choose a catalog preset or use **Custom URL**.
3. Return to **Upscale**, add image or video files, select a model, and choose the output folder.
4. Adjust scale, batch, tiling, decoding, and GPU guard settings if needed.
5. Press **Start**. The queue, preview, log panel, and GPU monitor update while the worker runs.

The default pretrained folder resolves to the repository's `pretrained/` directory when the app is used in-tree. The settings panel can move the app data folder and the Models tab can import a local checkpoint.

## Keyboard shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl+O` | Add files |
| `Ctrl+S` | Save settings |
| `Ctrl+Enter` | Start processing |
| `Delete` | Remove selected queue item |
| `F5` | Re-scan models |
| `Ctrl+R` | Reset settings |
| `Ctrl+Shift+L` | Switch language (English / Arabic) |
| `Ctrl+Q` | Quit |

The File, Edit, Run, Tools, and Help menus expose the same actions with accelerator labels.

## Supported models

| Kind | Checkpoints | Scale | License |
|---|---|---:|---|
| `srvgg` | realesr-animevideov3, 4xLSDIRCompactv2 | 4x | BSD-3 / CC-BY |
| `span` | span_pix_pretrain_4x, span_mssim_pretrain_4x | 4x | CC-BY-4.0 |
| `era` | eranet_N12_pretrain_325k | 2x | MIT |
| `animesr` | AnimeSR_v1, AnimeSR_v2 | 4x | Apache-2.0 |
| `rfdn_student` | RFDN_distill_v1_4x_student | 4x | Project-internal |
| `srvgg_student` | SRVGG_distill_v1_4x_student | 4x | Project-internal |

The catalog also lists unsupported architectures so they are visible rather than silently ignored. RRDB/APISR/MambaIRv2 are not vendored in this GUI build.

`rfdn_student` is the in-house teacher-distilled RFDN student (~315K params, KD from SPAN V7, 40 epochs, 29.32 dB test PSNR). It is shipped as `pretrained/RFDN_distill_v1_4x_student.pth` and shows up automatically in the Models tab when the app opens. Best for fast, real-time, low-VRAM upscaling of anime frames.

`srvgg_student` is the **current best** in-house model (Step-2 rebase of the pretrained-teacher plan): TinySRVGG student, 317K params, distilled from the `4xHFA2k_ludvae_realplksr_dysample` teacher for 50 epochs — 33.55 dB / 0.9146 SSIM on the held-out test split (bicubic 33.05), NIQE 7.38 on degraded real frames, deployment-benchmarked at **51.8 fps on a real 25 fps episode at 2560x1440 output** (versus 42.3 fps for `realesr-animevideov3`). Shipped as `pretrained/SRVGG_distill_v1_4x_student.pth` under the display name "SRVGG Distill v1 * Step-2 (HFA teacher)" in the Trained group of the Models tab; auto-detected from its state dict (no manual kind selection needed).

> Deployment note: run it with **TensorRT off** (ONNX-Runtime fp16 backend) — TensorRT 11's compiler backend mis-compiles this graph family on Turing/Windows (see `docs/ROADMAP_REALTIME_QUALITY.md`, section 7). The GPU also needs AC power; on DC the driver caps clocks ~6x below full speed.


### Adding a new model (without code changes)

Two extension paths, both additive -- nothing existing is modified:

1. **New preset for an already-supported architecture.** Add an entry to
   `apps/anime_upscaler_gui/data/registry.json` under `models` (same shape as
   the catalog table above; a `kind` of `srvgg`, `span`, `era`, `animesr`,
   `rfdn_student` or `srvgg_student` is already loadable). Entries with an id
   matching a built-in preset override it; malformed entries are skipped with a
   logged warning. The app-data folder can also carry its own registry.json via
   `_ModelRegistry(pretrained_dir, presets_path=...)`.
2. **New checkpoint family.** Register an `ArchSpec` (a `detect` + `loader`
   pair + a `Capability`) in `anime_upscaler_gui/archs.py` (or from any
   module at import time via `archs.register_arch()`). Per-kind runtime
   behavior (tiling, TensorRT, video batching, recurrent frames, padding) is
   declared on the spec, so `pipeline.py`, `trt_engine.py`, `registry.py`
   and the UI pick it up with no extra kind-string checks anywhere.

For the full set of extension paths (model kind, settings panel entry,
backend registry, and the preset-catalog shortcut), see
[`docs/EXTENDING.md`](docs/EXTENDING.md). Each recipe ends with the
one-line commit-message convention used in this repo.


## Settings and storage

Settings are stored at:

- Windows: `%APPDATA%\anime_upscaler_gui\settings.json`
- Linux/macOS: `$XDG_CONFIG_HOME/anime_upscaler_gui/settings.json`, or the platform user config directory

Queue state is persisted separately and restored on startup. Writes use a temporary file followed by an atomic replace. A corrupt settings file is replaced with defaults and reported in the UI.

Important defaults:

```json
{
  "device": "cuda",
  "fp16": true,
  "outscale": 2.0,
  "batch_size": 1,
  "decode": "cv2",
  "prefetch": "sync",
  "downscale_max_edge": 2560,
  "gpu_guard_mode": "warn",
  "tile_size": 256,
  "tile_overlap": 32,
  "queue_mode": "single",
  "theme": "light"
}
```

## Themes and accessibility

The app includes **Light**, **Dark**, and **High contrast** themes. Theme tokens live in `anime_upscaler_gui/theme.py`; changing a theme rebuilds the UI so all widgets use the active palette. Status rows expose text, glyph, and color together. Focus outlines, dynamic wrapping, keyboard activation, and an 800x600 minimum window support keyboard and low-vision use.

See [`docs/onboarding.md`](docs/onboarding.md), [`docs/themes.md`](docs/themes.md), and [`docs/dev/widgets.md`](docs/dev/widgets.md) for user and contributor guidance. For extension recipes and the public API surface, see [`docs/EXTENDING.md`](docs/EXTENDING.md) and [`../../docs/PUBLIC_API.md`](../../docs/PUBLIC_API.md).

## GPU memory guard

| Mode | Behavior |
|---|---|
| Warn | Estimate memory and ask before proceeding when the budget is exceeded |
| Auto-downscale | Reduce the input to fit the estimated budget |
| Tiled | Process tiles with overlap and raised-cosine blending |
| Off | Skip the guard for users who manage their own VRAM |

The AnimeSR recurrent model does not support spatial tiling and falls back to whole-frame processing.

## Standalone deployment

```bash
cp -r apps/anime_upscaler_gui ~/projects/anime-upscaler
cp -r pretrained ~/projects/anime-upscaler/pretrained
cd ~/projects/anime-upscaler
python -m venv .venv
.venv/bin/pip install -r requirements.txt
./LAUNCHER.sh
```

The app data folder is independent of the copied source, so settings survive moves and upgrades.

## Tests and checks

Run from the repository root so the `apps` package is importable:

```powershell
apps\anime_upscaler_gui\.venv\Scripts\python.exe -m pytest apps\anime_upscaler_gui\tests\ -q
```

The test suite covers state validation, settings and queue persistence, model registry behavior, threading, themes, accessibility, telemetry, empty states, tiling, VRAM estimation, and documentation links.

## Current limitations

- Drag-and-drop requires optional `tkinterdnd2` integration and is not enabled.
- ONNX is wired in the backend but is not exposed by the GUI.
- Live video scrubbing re-decodes frames; the thumbnail strip is faster for long videos.
- APISR, RRDB, and MambaIRv2 architecture classes are not included in this standalone GUI.
- The Arabic catalog has 35 hand-translated strings; native-speaker review is pending before shipping to Arabic users.

See the [UI/UX overhaul plan](docs/plans/ui_ux_overhaul.md) and [resume point](docs/RESUME.md) for the implementation roadmap.

## Building the asset bundle

Icon and splash assets are generated once from JPEG masters in `apps/anime_upscaler_gui/assets/`:

```powershell
apps\anime_upscaler_gui\.venv\Scripts\python.exe apps\anime_upscaler_gui\scripts\build_assets.py
```

This writes the duotone `app.ico` for Windows, the 16/24/32 outline set, the 16-512 app-icon PNGs, and the 800x600 splash hero PNG. Re-run after replacing masters.

Skip the splash for a faster launch:

```powershell
apps\anime_upscaler_gui\.venv\Scripts\python.exe -m apps.anime_upscaler_gui --no-splash
```
