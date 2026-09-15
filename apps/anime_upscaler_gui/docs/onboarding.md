# Onboarding

## Before launching

Use Python 3.10 or newer and create the GUI's local virtual environment:

```powershell
cd apps\anime_upscaler_gui
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For an in-tree checkout, the main repository environment may be used instead. The GUI does not require the training pipeline to be installed separately.

## First launch

Start from the repository root:

```powershell
apps\anime_upscaler_gui\.venv\Scripts\python.exe -m apps.anime_upscaler_gui
```

The application opens on the **Upscale** tab. The **Models** tab is the first place to go when the model dropdown is empty.

## Add a model

1. Place a supported checkpoint in the pretrained folder.
2. Select **Models → Re-scan** or press `F5`.
3. Confirm the detected kind and scale in the installed-model list.
4. If the checkpoint is elsewhere, use **Import local**.
5. If a catalog preset is available, select it and start the download.

Only supported architectures can be selected for processing. Unsupported catalog entries remain visible with an explanatory label.

## Process an image

1. Select **Add files** or press `Ctrl+O`.
2. Choose an installed model.
3. Set the output folder.
4. Pick output scale and optional tiling.
5. Press **Start** or `Ctrl+Enter`.

The queue remains available while processing. Completed outputs are written beside the configured output directory and the preview updates when a result is available.

## Process a video

Add a video file in the same way as an image. Use the cut controls to select a time range before starting. The thumbnail strip is preferable to repeated live scrubbing for long videos. AnimeSR uses whole-frame processing because its recurrent three-frame model cannot be spatially tiled.

## Recommended GPU settings

For an 8 GB RTX 4000 Mobile, start with `device=cuda`, `fp16=true`, synchronous decoding, batch size 1, and GPU guard mode **Warn**. Use **Tiled** mode when full-frame memory estimates are too high. Increase batch size only after a short test run.

If CUDA is selected but unavailable, the app warns and permits CPU processing. CPU mode is suitable for validation, not long production jobs.

## Recovering from common problems

- **Empty model dropdown:** rescan after placing checkpoints, or import a local file.
- **Invalid checkpoint:** verify the file is complete and matches a supported architecture.
- **Disk full or permission denied:** select another output folder and retry.
- **VRAM warning:** choose Auto-downscale or Tiled, reduce tile size, or lower batch size.
- **Settings reset notice:** the previous JSON was unreadable; review the reported app-data path and re-enter settings.

## Settings and queue recovery

Settings and queue state are stored in the platform app-data directory, not beside the source code. Closing the window during a job leaves unfinished items as pending on the next launch. The app uses atomic writes so a process interruption does not intentionally expose half-written JSON.

## Keyboard-first workflow

`Ctrl+O` adds files, `Ctrl+S` saves settings, `Ctrl+Enter` starts, `Delete` removes the selected queue row, `F5` rescans models, `Ctrl+R` resets settings, and `Ctrl+Q` quits. Use `Tab` and `Shift+Tab` to move through controls; `Return` commits spinbox values.
