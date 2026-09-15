# Anime Upscaler GUI — Resume Point

> **Read this first** if you're picking up work in a fresh
> conversation. The phases below are described in the master plan; this
> file is the **state of play + how to continue + critical gotchas**.

## Session update — architecture + extensibility + video fixes

Decisions and contracts to remember before touching these files again:

1. **Arch registry (`archs.py`).** Kinds live in `ArchSpec`s: a `detect`+
   `loader` pair plus a `Capability` (tiling/TensorRT/video-batch/recurrent
   frames/pad+crop multiples). Built-ins register via `_register_builtin_archs()`.
   `pipeline.py`, `trt_engine.py`, `registry.py` query `capabilities(kind)` —
   do NOT reintroduce `kind == "animesr"` string checks.
2. **Preset catalog is data-extendable.** `apps/anime_upscaler_gui/data/registry.json`
   is merged with built-ins (override by `id`); the app-data folder can carry its
   own `registry.json` (see `_AppPaths.registry_path`, moved with app data).
   Malformed entries warn and are skipped.
3. **app.py mixins.** Settings I/O lives in `app_settings_io.py`
   (`SettingsIOMixin`), menu/window ceremony in `app_window_chrome.py`
   (`WindowChromeMixin`). Doc-consistency tests scan all `app*.py`, so new
   shortcut bindings must stay mirrored with README + those tests.
4. **NVENC detection is functional now** (`ffmpeg.py`): encodes 3 black frames
   through h264_nvenc; "-encoders listing" alone lies on mixed-GPU machines.
   Probe timeout is 15s.
5. **Unknown-length videos** (PyAV `stream.frames` is None → total 0) now run
   to EOF with an indeterminate progress display instead of failing with
   "cut window is empty". `decode=pyav` is honored for real (the old code
   silently re-opened with cv2); cut-start skip uses `_SkipFirstFrames`.
6. **Known flake:** `test_empty_states.py::..._batch_mode` (and friends) can
   fail once per session due to the Windows pytest tempfile race; re-run and
   it passes (verified twice, 229/229).

### Remaining structural debt (next session candidates)

- `app.py` still owns queue/model/download/event-polling logic (~840 lines).
  Follow the same mixin-or-controller pattern as Defaults: the event-poll
  section and the model/download actions are the next extraction targets.
- `preset_path` for arbitrary user preset dirs in Settings UI (plumbing done).
- Arabic strings still need native-speaker review; drag-and-drop, ONNX not wired.

## Where we are

- **All 14 phases DONE.** Asset bundle built from supplied JPEG masters; iconography wired into action row + window title + taskbar icon; splash shown on launch with `--no-splash` opt-out.
- **Test count: 229/229 passing** (~5s).
- **app.py: ~1070 → ~845 lines** (settings/menu extracted into mixins)
- **Test count: 198/198 passing** (~5s).
- **app.py: 692 → 734 lines** (locale toggle, window icon, action-row icon helpers).
- **`apps/anime_upscaler_gui/assets/`**: 24 outline PNGs (8 icons x 16/24/32), 7 duotone app-icon PNGs + `app.ico`, splash hero PNG.

## Critical files (read these first)

| File | Why |
|---|---|
| `docs/plans/ui_ux_overhaul.md` | The 14-phase master plan with full per-phase detail |
| `docs/plans/ui_ux_overhaul_checklist.md` | Per-phase task list with current status (10/14 done) |
| `docs/audit/ui_ux_audit.md` | The initial audit that informed the plan |
| `tests/conftest.py` | **MUST READ** — session-scoped `shared_tk_root` (Tk only allows one root per process) |
| `apps/anime_upscaler_gui/docs/RESUME.md` | This file |

## How to run

```bash
# From the repo root, NOT the app folder (apps/ must be on sys.path):
apps/anime_upscaler_gui/.venv/Scripts/python.exe -m pytest apps/anime_upscaler_gui/tests/ -q
```

The `pytest tests/` from inside `apps/anime_upscaler_gui/` breaks with
`ModuleNotFoundError: No module named 'apps'`. The atexit
`PermissionError: ...\pytest-of-<user>\pytest-current` after the run is a
known Windows tempfile race — **ignore it**.

## Critical gotchas (learned the hard way)

### 1. Tk is one-per-process

`tk.Tk()` can only be called ONCE per process. After a root is destroyed
on Windows, no new root can be created (the tcl library enters a bad
state — `Can't find a usable init.tcl`).

**Solution:** `tests/conftest.py` provides `shared_tk_root` (session-scoped,
never destroyed). All Tk-using test files depend on it via `@pytest.fixture(shared_tk_root)`.
When adding a new test file that needs Tk, **use the shared root** —
do not create a new one.

### 2. Toplevel fallback for fake `UpscaleGUI`

`UpscaleGUI(tk.Tk)` IS the root, so the second test in a session can't
construct another one. The fixtures in `test_async_blocking.py` and
`test_shortcuts.py` work around this: the FIRST call creates a real
`UpscaleGUI` Tk root; subsequent calls construct a `Toplevel` and bind
the needed methods via `getattr(UpscaleGUI, name).__get__(a, type(a))`.

When adding methods that tests call, extend the bound list in both
fixtures.

### 3. `winfo_ismapped()` unreliable on withdrawn roots

`widget.winfo_ismapped()` returns 0 even after `pack()` if the root is
withdrawn. Use `widget.pack_info()` (raises TclError if not packed)
instead — see `test_empty_states.py:test_input_panel_shows_empty_state_in_batch_mode`.

### 4. `event_generate("<Return>")` doesn't dispatch on withdrawn roots

Use `widget.bind("<Return>")` to verify the binding is installed, and
call the lambda directly to verify the captured state. See
`test_focus.py:test_bind_return_invokes_command`.

### 5. `STATUS_COLORS` is mutated at runtime by `theme.apply()`

`ui_constants.STATUS_COLORS` and the `PENDING_FG` / `RUNNING_FG` /
`CANCELLED_FG` module attributes don't exist until `theme.apply()` runs
at startup. Code that needs them must be resilient: `a11y.py` has a
`_DEFAULT_COLORS` fallback for this reason.

### 6. `_coerce()` was hardened in Phase 4

Original code raised `ValueError` on unparseable int/float strings
(e.g. `"wat"`). Phase 4 changed it to silently skip — the field keeps
its default. This made `import_file()` more robust.

## Phase-by-phase status

| # | Phase | Status | Key files | Test count |
|---|---|---|---|---|
| 0 | Token sweep | DONE | `ui_constants.py` | 18/18 |
| 1 | Data layer | DONE | `state.py` | 29/29 |
| 2 | Widget extraction | DONE | `widgets/*.py` (7 widgets) | 38/38 |
| 3 | Tabbed layout | DONE | `app.py:_build_ui` (Notebook) | 38/38 |
| 4 | Settings & persistence | DONE | `settings.py` (atomic + QueueController) | 53/53 |
| 5 | Theme system | DONE | `theme.py` (3 themes, WCAG AA) | 75/75 |
| 6 | Threading fixes | DONE | `pipeline.py` (fatal_error), `app.py` (non-blocking modal) | 87/87 |
| 7 | Interaction improvements | DONE | `widgets/menu_bar.py`, `errors.py`, `messages.py` | 103/103 |
| 8 | Empty/error states | DONE | `widgets/empty_state.py`, `eta.py`, pre-flight in `app.py` | 117/117 |
| 9 | Telemetry & monitoring | DONE | `widgets/gpu_monitor.py`, `log_panel.py`, `toast.py`, `logging_setup.py` | 131/131 |
| 10 | Accessibility | DONE | `a11y.py`, conftest.py (shared_tk_root) | 144/144 |
| 11 | Visual polish | DONE | icons.py + action-row icons + window title + 256/512 app icon | 198/198 |
| 12 | Documentation | DONE | README, onboarding, themes, dev/widgets, test_docs.py | 159/159 |
| 13 | Splash screen | DONE | widgets/splash.py + --no-splash + fade | 198/198 |
| 14 | i18n (EN + AR) | DONE | messages.py dicts + RTL toggle + 20 tests | 179/179 |

## Next steps (resume order)

1. **Phase 11 — Visual polish.** Skip if no Phosphor PNG assets; otherwise
   the plan is in `docs/plans/ui_ux_overhaul.md` §11.
2. **Phase 12 — Documentation.** DONE (README + onboarding + themes + dev/widgets + test_docs).
3. **Phase 13 — Splash screen.** New `widgets/splash.py` (Toplevel,
   borderless, centered, 1.5s display + 200ms fade).
4. **Phase 14 — i18n.** DONE (EN + AR dicts + `Ctrl+Shift+L` + 20 tests). Skipped
   babel/gettext + native-speaker Arabic review + bundled fonts. Native reviewer
   for Arabic strings is the one outstanding item before shipping to AR users.

## Quick commands

```bash
# Full test suite
apps/anime_upscaler_gui/.venv/Scripts/python.exe -m pytest apps/anime_upscaler_gui/tests/ -q

# Just the new a11y tests
apps/anime_upscaler_gui/.venv/Scripts/python.exe -m pytest apps/anime_upscaler_gui/tests/test_focus.py apps/anime_upscaler_gui/tests/test_screen_reader.py -v

# Compile-check all sources
apps/anime_upscaler_gui/.venv/Scripts/python.exe -m py_compile $(Get-ChildItem -Recurse -Filter "*.py" apps/anime_upscaler_gui/anime_upscaler_gui/*.py)

# Smoke launch the GUI
apps/anime_upscaler_gui/.venv/Scripts/python.exe -m apps.anime_upscaler_gui
```
