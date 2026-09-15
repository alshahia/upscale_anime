# Widget development

The GUI uses Tkinter/ttk. `UpscaleGUI` composes small internal widgets from `anime_upscaler_gui/widgets/`; widgets own their controls and variables, while the app owns cross-widget coordination, jobs, worker events, settings, and persistence.

## Widget contract

A panel normally follows this shape:

```python
class _ExamplePanel(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._build()
```

Keep widgets internal (`_Name`) unless a public API is actually needed. Use `self.app` for callbacks into the orchestrator. Do not read another widget directly; route coordination through `app.py`.

Use relative imports inside the package and existing tokens from `ui_constants.py`. Do not add new dependencies for simple Tkinter behavior. Optional integrations must degrade cleanly when unavailable.

## Building a panel

1. Read the neighboring widget and the relevant `app.py` call sites.
2. Keep construction safe for the shared test root and avoid starting threads in `__init__` unless the widget owns a clear lifecycle.
3. Use `ttk` controls and accessible text labels. Tooltips and glyphs are supplemental, not replacements for text.
4. Schedule polling with `after()` and cancel it in `destroy()` when needed.
5. Keep worker callbacks off the Tk thread. Send events to the app's event queue and update widgets from the polling loop.
6. Use `errors.humanize()` for user-facing exceptions and the package logger for diagnostics.

## Wiring a widget

Import the widget from `widgets/__init__.py`, create it from `_build_ui`, and keep the app callback surface explicit. If a widget needs a new callback method in tests, update the Toplevel fallback bindings in `tests/test_async_blocking.py` and `tests/test_shortcuts.py` when applicable.

## Tests

Headless Tk tests use the session-scoped `shared_tk_root` fixture from `tests/conftest.py`. Tk only supports one root reliably in this Windows test setup; never create and destroy a fresh root per test file.

A minimal widget test should construct the panel, assert its key controls exist, and destroy only child widgets that the fixture owns. `winfo_ismapped()` is unreliable on withdrawn roots; use `pack_info()` or `grid_info()` to assert geometry.

Run the GUI suite from the repository root:

```powershell
apps\anime_upscaler_gui\.venv\Scripts\python.exe -m pytest apps\anime_upscaler_gui\tests\ -q
```

Compile modified modules before handing off:

```powershell
apps\anime_upscaler_gui\.venv\Scripts\python.exe -m py_compile apps\anime_upscaler_gui\anime_upscaler_gui\widgets\example.py
```

## Checklist for a new panel

- [ ] Uses existing tokens and `ttk` patterns.
- [ ] Has readable labels and keyboard focus behavior.
- [ ] Handles missing optional dependencies.
- [ ] Does not block the GUI thread.
- [ ] Has a headless construction test using `shared_tk_root`.
- [ ] Is exported from `widgets/__init__.py` if app composition imports it there.
- [ ] Is documented here only if it introduces a new lifecycle or callback convention.
