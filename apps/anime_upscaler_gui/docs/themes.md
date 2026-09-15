# Themes

The GUI has three built-in themes:

| Theme | Use |
|---|---|
| Light | Default palette for normal lighting |
| Dark | Reduced-brightness palette for dark rooms |
| High contrast | Strong borders and contrast for low-vision use |

Choose a theme in the Settings panel. The selection is saved and applied before the next UI build on startup. A live change rebuilds the UI so existing widgets receive the new token values.

## Token ownership

Theme definitions are in `anime_upscaler_gui/theme.py`. The `Theme` dataclass owns surfaces, text, borders, accent colors, status colors, status background tints, preview colors, and listbox colors. `theme.apply()` synchronizes those values into `ui_constants.py` for existing widget imports.

Do not add color literals to widget modules. Add a field to `Theme`, define it in all three theme constants, map it in `_FIELD_TO_CONST`, and use the corresponding `ui_constants` token from widgets.

## Adding a theme

1. Create a frozen `Theme` value with every field populated.
2. Add it to `THEMES` with a stable settings key.
3. Check all required foreground/background pairs with `assert_wcag(theme)`.
4. Add it to the theme picker in `widgets/settings_panel.py`.
5. Add a round-trip and contrast test in `tests/test_theme.py`.
6. Rebuild the UI after applying it, matching `app._on_theme_change`.

A new theme must retain at least 4.5:1 contrast for body text and status labels. Accessibility tests are the gate; do not weaken them for palette convenience.

## Spacing and typography

Use the shared spacing and font tokens from `ui_constants.py` rather than widget-local padding or font tuples. This keeps theme changes and future visual polish localized.

## Status colors

Status presentation must not rely on color alone. Queue rows use the `JobStatus` value, a glyph, and readable text. Background tints are supplementary and must maintain readable foreground contrast.

## Verification

```powershell
apps\anime_upscaler_gui\.venv\Scripts\python.exe -m pytest apps\anime_upscaler_gui\tests\test_theme.py apps\anime_upscaler_gui\tests\test_accessibility_contrast.py -q
```
