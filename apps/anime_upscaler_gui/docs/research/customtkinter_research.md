# CustomTkinter Fit Assessment for Anime Upscaler GUI

**Research date:** 2026-07-25
**Project reviewed:** `apps/anime_upscaler_gui`
**Framework version reviewed:** CustomTkinter 6.0.0
**Research method:** Official website, official documentation, GitHub repository/wiki/issues, PyPI metadata, and read-only inspection of the current GUI.

> **Executive conclusion:** CustomTkinter is a strong visual-modernization option for the existing Tkinter application, but it is not a complete UI platform. It provides useful themed replacements for common controls, dark/light appearance handling, custom JSON themes, and DPI scaling. It does not provide first-class RTL, localization, drag-and-drop, splash-screen, or accessibility infrastructure. Migration would therefore modernize appearance while leaving several planned overhaul phases as application responsibilities.

## 1. Overview

CustomTkinter is an open-source Python desktop UI library built on top of Tkinter. It supplies modern-looking, configurable widgets while retaining Tkinter's event loop, geometry managers, variables, and general programming model.

It is intended to solve several weaknesses of vanilla Tkinter and `ttk`:

- Vanilla Tk widgets can look dated and vary noticeably across operating systems.
- `ttk.Style` theming is powerful but can be verbose and platform-sensitive.
- Dark-mode support normally requires application-specific styling.
- Rounded controls, modern switches, segmented buttons, and unified color palettes require custom drawing in ordinary Tkinter.
- High-DPI behavior can require additional handling.

CustomTkinter provides:

- A themed `CTk` root window.
- Modern replacements for common Tkinter controls.
- Light, dark, and system appearance modes.
- Light/dark color pairs.
- JSON-based color themes.
- Runtime appearance-mode switching.
- Widget and window scaling.
- A reasonably consistent visual design across Windows, macOS, and Linux.
- Interoperability with ordinary Tkinter widgets.

The official project describes it as a modern, fully customizable Tkinter-based UI library whose widgets can be used like normal Tkinter widgets and mixed with ordinary Tkinter elements.

It is better described as a **Tkinter widget/theme layer** than a drop-in replacement for every Tk or `ttk` feature. Existing application logic can remain, but unsupported controls must continue using Tkinter/`ttk` or be implemented separately.

## 2. License

CustomTkinter's repository identifies its license as the **MIT License**.

Commercial use is permitted. The MIT License generally permits:

- Commercial use
- Modification
- Distribution
- Private use

The copyright and license notice must be retained in copies or substantial portions of the software.

There is a metadata inconsistency worth recording:

- The GitHub repository and PyPI classifier identify the project as MIT.
- The current `setup.cfg` contains `license = Creative Commons Zero v1.0 Universal`.
- The same metadata file also declares the MIT classifier and includes the repository `LICENSE` file.
- The changelog says the project changed to MIT in version 5.1.0.

For practical compliance, the repository's actual `LICENSE` file should be treated as authoritative and bundled with any vendored or redistributed copy. If the application will be commercially distributed, record the exact CustomTkinter version and preserve its license text.

**Commercial-use assessment:** Yes, subject to normal MIT notice retention.

## 3. Maturity

### Current release

PyPI reported:

- **Latest version:** 6.0.0
- **PyPI release date:** June 24, 2026
- **Minimum Python:** 3.7
- **Distribution:** Pure-Python, platform-independent wheel

The repository changelog labels version 6.0.0 as dated January 21, 2026, which does not match PyPI's June 24 upload date. This may reflect a delayed package publication or stale changelog date.

### Repository adoption

At research time, the GitHub repository displayed approximately:

- **13.5k stars**
- **1.2k forks**
- **640 commits**
- **292 open issues**
- **19 open pull requests**

These figures indicate substantial community adoption for a Tkinter extension.

### Release and maintenance pattern

Release history is uneven:

- CustomTkinter had frequent releases during 2021-2023.
- Version 5.2.2 was released January 10, 2024.
- Version 6.0.0 was then uploaded June 24, 2026.

The gap between 5.2.2 and 6.0.0 was long. Version 6.0.0 nevertheless provides a recent maintenance signal and contains bug fixes and incremental features, including:

- A showroom application
- A gold theme
- Additional widget methods
- Vertical segmented buttons
- Tabview font configuration
- Label borders
- Improved slider/scrollbar mouse handling
- Improved focus behavior for entry/text widgets
- Improved configuration APIs
- CTkToplevel custom-icon fixes
- Scrollable-frame fixes

The repository README calls the library actively developed, but the release cadence does not support assuming rapid responses or frequent package releases. The 292 open issues and open accessibility request also suggest limited maintainer bandwidth.

GitHub's commits API could not be queried due to an HTTP 403 response, so a reliable recent commit-per-month measurement could not be produced. The visible repository total was 640 commits, and the recent 6.0.0 release is the strongest current maintenance signal.

### Maturity judgment

CustomTkinter is:

- Mature enough for ordinary desktop applications.
- Widely used and stable in its core widget model.
- Not comparable to a large, accessibility-focused GUI toolkit such as Qt.
- Maintained, but with historically irregular releases.
- Best adopted with an exact version pin and application-level regression testing.

## 4. Installation

Install from PyPI:

```bash
pip install customtkinter
```

Upgrade:

```bash
pip install --upgrade customtkinter
```

For this standalone application, a safer dependency entry would be an exact or compatible release pin after validation, for example:

```
customtkinter==6.0.0
```

### Dependencies

Current package metadata lists:

- `darkdetect`
- `packaging`
- `typing_extensions` only for Python 3.7 and older

Tkinter itself is part of the CPython standard library but may be packaged separately on some Linux distributions. Linux deployment may therefore require the operating-system Tk package, such as `python3-tk`.

CustomTkinter does not add a GPU dependency and should have negligible effect on the application's approximately 8 GB VRAM target. It remains CPU-side UI code.

### Package size

PyPI reports:

- Wheel: approximately **302.2 kB**
- Source distribution: approximately **267.2 kB**

Installed size will be larger because the package includes Python files, JSON themes, fonts/assets, and dependencies. The official sources do not publish a canonical total installed footprint. It should still be small relative to PyTorch, Torchvision, OpenCV, model checkpoints, and video dependencies.

### Standalone packaging

The official packaging page warns that CustomTkinter includes non-Python data such as JSON and OTF files.

Its documented PyInstaller guidance says:

- Use `--onedir`, not `--onefile`.
- Explicitly include the `customtkinter` package data with `--add-data`.

Example from the official documentation:

```bash
pyinstaller --noconfirm --onedir --windowed \
  --add-data "<CustomTkinter Location>/customtkinter;customtkinter/" \
  "<Path to Python Script>"
```

This is an important concern for the "vendored standalone" requirement. CustomTkinter can be included with the app, but packaging must preserve its assets and license. If a single executable is a hard requirement, the official guidance is a poor fit and a custom PyInstaller hook/extraction strategy would need testing.

The current GUI requirements file contains Torch, Torchvision, OpenCV, NumPy, Pillow, and PyAV but does not currently list CustomTkinter, `darkdetect`, or `packaging`.

## 5. Features

### Confirmed built-in widgets

CustomTkinter 6.0.0 exports:

- `CTk`
- `CTkToplevel`
- `CTkInputDialog`
- `CTkButton`
- `CTkCheckBox`
- `CTkComboBox`
- `CTkEntry`
- `CTkFrame`
- `CTkLabel`
- `CTkOptionMenu`
- `CTkProgressBar`
- `CTkRadioButton`
- `CTkScrollbar`
- `CTkScrollableFrame`
- `CTkSegmentedButton`
- `CTkSlider`
- `CTkSwitch`
- `CTkTabview`
- `CTkTextbox`
- `CTkCanvas`
- `CTkFont`
- `CTkImage`

It also re-exports common Tkinter variables such as `StringVar`, `IntVar`, `DoubleVar`, and `BooleanVar`.

### Requested-widget matrix

| Requested widget | Built in? | Assessment |
|---|---:|---|
| `CTkTabview` | Yes | CustomTkinter's themed notebook-like control. Tabs are `CTkFrame` instances. |
| `CTkListbox` | No | Continue using `tk.Listbox`, style it manually, or implement a list in `CTkScrollableFrame`. Third-party controls exist but are not part of CustomTkinter. |
| `CTkSpinbox` | No | The official wiki has an old tutorial for creating a custom Spinbox, but no built-in `CTkSpinbox` is exported in 6.0.0. Continue using `ttk.Spinbox` or build a small composite control. |
| `CTkComboBox` | Yes | Editable or read-only combo control with themed entry and dropdown behavior. |
| `CTkProgressBar` | Yes | Supports determinate and indeterminate modes, orientation, `.set()`, `.start()`, and `.stop()`. |
| `CTkSlider` | Yes | Supports range, discrete steps, orientation, and command callbacks. |
| `CTkLabel` | Yes | Supports text, images, fonts, colors, anchoring, and a border in 6.0.0. |
| `CTkButton` | Yes | Supports commands, hover state, images, text/image composition, colors, and rounded corners. |
| `CTkFrame` | Yes | Primary themed container. |
| `CTkLabelFrame` | No | No direct equivalent to `ttk.LabelFrame`; use `CTkFrame` plus a `CTkLabel`, or retain `ttk.LabelFrame`. |
| `CTkCanvas` | Partially/publicly exported | Primarily part of CustomTkinter's rendering internals. It is not a modern high-level replacement for all `tk.Canvas` usage. Existing image/preview canvases should generally remain `tk.Canvas`. |

### Other useful features

- Rounded corners and border configuration.
- Widget-specific foreground, hover, border, text, and disabled colors.
- Light/dark tuple colors.
- Image support through `CTkImage`.
- Custom font handling through `CTkFont`.
- High-DPI scaling on Windows and macOS.
- Scrollable frame support.
- Runtime appearance switching.
- Interoperability with Tkinter dialogs, variables, geometry managers, and event bindings.
- Showroom application via:

```python
import customtkinter as ctk

ctk.run_showroom()
```

### Relevance to the current app

The existing `app.py` uses many controls that map directly:

- `tk.Tk` → `CTk`
- `ttk.Frame` → `CTkFrame`
- `ttk.Label` → `CTkLabel`
- `ttk.Button` → `CTkButton`
- `ttk.Entry` → `CTkEntry`
- `ttk.Combobox` → `CTkComboBox`
- `ttk.Progressbar` → `CTkProgressBar`
- `ttk.Scrollbar` → `CTkScrollbar`
- `ttk.Radiobutton` → `CTkRadioButton`
- `ttk.Checkbutton` → `CTkCheckBox`
- Future `ttk.Notebook` → `CTkTabview`

But several heavily used controls have no direct replacement:

- `tk.Listbox`
- `ttk.Spinbox`
- `ttk.LabelFrame`
- `ttk.Separator`
- Application image canvases
- Standard message boxes and file dialogs

A mixed CustomTkinter/Tkinter application is officially supported and is the realistic migration path.

## 6. Theme system

CustomTkinter distinguishes between **appearance mode** and **color theme**.

### Appearance modes

Built-in appearance modes are:

```python
customtkinter.set_appearance_mode("system")
customtkinter.set_appearance_mode("dark")
customtkinter.set_appearance_mode("light")
```

Appearance mode can be changed at runtime.

If `"system"` is selected:

- Windows and macOS can follow operating-system appearance changes.
- Linux currently falls back to light mode because system appearance detection is not supported by the official implementation.

On macOS, the project README states that Python 3.10 or newer, or a compatible Anaconda Python with Tcl/Tk 8.6.9 or newer, may be required to get a dark window header.

### Built-in color themes

The official color/theme documentation lists:

- `blue`
- `dark-blue`
- `green`

The 6.0.0 changelog also says a **gold** theme was added. This indicates the website documentation may lag behind the package.

Select a built-in theme before constructing widgets:

```python
customtkinter.set_default_color_theme("dark-blue")
```

### Light and dark colors

A widget color can be:

- A Tk color name
- A hex color
- A two-item tuple containing light and dark colors

Example:

```python
button = customtkinter.CTkButton(
    app,
    fg_color=("#DB3E39", "#821D1A"),
)
```

The appropriate tuple entry is selected automatically according to appearance mode.

### Custom JSON themes

A theme is represented by a JSON file. The official recommendation is to copy one of the bundled theme files, modify its widget values, and load the file:

```python
customtkinter.set_default_color_theme(
    "assets/themes/anime_upscaler.json"
)
```

This is suitable for an anime-oriented palette and token sweep because common colors, borders, fonts, and control defaults can be centralized rather than repeated across the UI.

For a vendored standalone app:

- Store the custom JSON under the app's own asset directory.
- Resolve it relative to the installed package/executable, not the working directory.
- Ensure it is included in macOS bundles, Linux packages, and Windows PyInstaller output.

### Live switching

Appearance mode is explicitly supported at runtime:

```python
customtkinter.set_appearance_mode("dark")
```

Existing widgets react to appearance-mode changes.

Full color-theme switching is more limited. `set_default_color_theme()` loads theme defaults; existing widgets should not be assumed to rebuild all of their colors dynamically from a newly loaded theme. The 6.0.0 showroom's theme-change handler destroys and recreates the application window after calling `set_default_color_theme()`. That is a strong signal that reliable live color-theme switching should be implemented by rebuilding the themed widget tree or restarting the UI.

Therefore:

- **Light/dark switching:** genuinely live.
- **Changing blue/green/custom palettes:** plan for UI reconstruction or restart and test every control.

## 7. i18n / RTL support

CustomTkinter does not provide a built-in localization framework.

It does not supply:

- Translation catalogs
- Locale negotiation
- Resource-string loading
- Pluralization
- Date/number formatting
- Mirrored layouts
- Automatic RTL direction
- Arabic shaping or bidi processing
- Language-specific fallback fonts

English and Arabic localization must be implemented in the application, for example with standard `gettext` or a small app-owned translation resource layer.

### Arabic and Hebrew text

CustomTkinter ultimately relies on Tk/Tcl and platform fonts for rendering text. Unicode strings can be assigned to widget `text` values, but that does not guarantee correct:

- Bidirectional ordering
- Arabic glyph shaping
- Cursor movement
- Selection behavior
- Mixed Arabic/Latin/numeric display
- Placeholder alignment
- Dropdown alignment

An issue titled **"RTL text is reversed"** existed in the repository and was closed in 2023. A more recent open request asks for dropdown-list justification in `CTkComboBox`. These are warning signs that complete RTL behavior is not a designed, documented feature.

### Layout mirroring

The current app heavily uses directional packing such as:

```python
widget.pack(side="left")
widget.pack(side="right")
```

CustomTkinter will not automatically reverse those for Arabic. The application must explicitly mirror:

- Row order
- Labels and controls
- Button groups
- Tab alignment where possible
- Padding tuples
- Status placement
- Text anchors
- Entry and combo justification
- Canvas annotations

`CTkLabel` and some other controls expose `anchor` or `justify`, but these are local presentation options, not an application-wide RTL layout engine.

### Recommended interpretation

CustomTkinter is **neutral rather than supportive** for i18n:

- It generally accepts translated Unicode text.
- It does not solve localization or RTL.
- Arabic must be validated on Windows, macOS, and Linux with the actual bundled font.
- Native Tk/`ttk` widgets may sometimes behave better than canvas-drawn custom controls for keyboard input and assistive technology.

RTL is the most important factor preventing a higher fit score.

## 8. Drag-and-drop

CustomTkinter does not provide built-in operating-system file drag-and-drop.

The application would still need:

- `tkinterdnd2` / TkDND, or
- Platform-specific integration, or
- A separate drag-and-drop package compatible with Tkinter

Because CustomTkinter is built on Tkinter, it can often coexist with `tkinterdnd2`, but root-window integration must be tested.

Potential approaches include:

- Use the DnD-capable Tk root class as the CustomTkinter parent/root.
- Use CustomTkinter's `set_ctk_parent_class()` hook if appropriate for the installed version.
- Register DnD targets on the underlying Tk widgets.
- Retain ordinary Tkinter frames/widgets specifically at the drop target.

This is not a zero-risk drop-in feature. Test:

- Windows Explorer file drops
- macOS Finder drops
- Linux file-manager drops
- Paths containing spaces and non-ASCII characters
- Multiple files
- Folder drops
- Drops over nested CTk controls and canvas-backed widgets

If drag-and-drop is a planned phase, `tkinterdnd2` remains a separate dependency and packaging concern.

## 9. Splash screen / window branding

### Window title

CustomTkinter's root and top-level windows retain normal Tk window APIs:

```python
app.title("Anime Upscaler")
```

The current app already calls `.title()` and that code maps directly.

### Window icon

Normal Tk icon methods remain relevant:

```python
app.iconphoto(True, tk.PhotoImage(file="assets/icon.png"))
```

On Windows, `.iconbitmap()` can be used with an ICO file:

```python
app.iconbitmap("assets/anime_upscaler.ico")
```

Version 6.0.0 specifically reports a fix for custom icons on `CTkToplevel`.

Icon behavior is platform-specific:

- Windows commonly uses `.ico`.
- Linux often accepts `PhotoImage`/PNG.
- macOS application branding is primarily determined by the `.app` bundle and packaging configuration rather than only Tk's runtime icon API.

### Splash screen

CustomTkinter has no documented dedicated splash-screen class.

A splash remains straightforward to implement with:

- `CTkToplevel` or a temporary `CTk` window
- `overrideredirect(True)` if a borderless window is desired
- `CTkImage` and `CTkLabel`
- `after()` to close the splash
- Careful sequencing so only one Tk root is created

The splash must not block model discovery or expensive imports on the Tk event thread. For perceived startup improvement, show the window before importing or initializing heavy Torch/model components where architecture permits.

A splash is possible, but CustomTkinter does not materially provide splash lifecycle management beyond themed windows and image display.

## 10. Accessibility

Accessibility is a weak area.

An open GitHub issue titled **"Accessibility and keyboard navigation?"** was filed in January 2026. A pull request to make `CTkButton` focusable was abandoned in 2023. These signals suggest accessibility is not comprehensively supported.

### Keyboard navigation

Do not assume full native keyboard behavior for canvas-drawn widgets.

Each migrated control should be tested for:

- Tab and Shift+Tab traversal
- Visible focus
- Space/Enter activation
- Arrow-key operation
- Escape behavior in dropdowns and dialogs
- Home/End/Page Up/Page Down where applicable
- Focus return after modal dialogs
- Keyboard-only tab changes
- Disabled-widget traversal

Some CustomTkinter widgets implement `.bind()` and `.focus()`, but that is not equivalent to a complete accessible keyboard-navigation model.

### Focus rings

CustomTkinter does not document a global accessible focus-ring system or a theme token dedicated to keyboard focus.

A custom focus color can potentially be simulated by changing a widget's border color on `<FocusIn>` and restoring it on `<FocusOut>`, but:

- It is application code.
- Not all controls expose equivalent focus/border behavior.
- Canvas-backed internals may not expose focus semantics like native controls.
- A visual border does not create accessibility-tree metadata.

### Screen readers

The official documentation does not claim screen-reader compatibility.

Custom-drawn controls can be less visible to platform accessibility APIs than native controls. Screen-reader behavior should be treated as unknown until tested with:

- NVDA and/or Narrator on Windows
- VoiceOver on macOS
- Orca on Linux

Where accessibility is important, retaining native `ttk` controls for text input, list selection, and complex navigation may be safer than replacing every widget.

### Contrast and WCAG

CustomTkinter allows full color customization, but it does not appear to:

- Validate contrast ratios
- Enforce WCAG thresholds
- Warn about low-contrast disabled text
- Generate accessible palettes automatically

The project must calculate and test contrast itself. At minimum, check normal text, secondary text, disabled text, progress state colors, focus indications, destructive buttons, and status colors in both light and dark modes.

### Accessibility conclusion

CustomTkinter helps visual design but does not provide an accessibility compliance layer. Accessibility must be a dedicated project phase, not assumed as a side effect of using a modern theme.

## 11. Migration effort from current Tkinter app

### Current application characteristics

Read-only inspection found:

- `app.py` is approximately 909 lines.
- The root class currently extends `tk.Tk`.
- The UI uses a mixture of `tk` and `ttk`.
- The app has a custom scrollable canvas/frame arrangement.
- It contains two `tk.Listbox` controls.
- It uses multiple `ttk.Spinbox` controls.
- It uses many `ttk.LabelFrame`, `ttk.Frame`, `ttk.Label`, `ttk.Button`, `ttk.Combobox`, `ttk.Progressbar`, `ttk.Scrollbar`, `ttk.Radiobutton`, `ttk.Checkbutton`, and `ttk.Separator` widgets.
- The code comments anticipate eventually wrapping sections in a `ttk.Notebook`.
- UI constants already centralize colors, fonts, padding, and status colors.
- Preview and scrolling code use `tk.Canvas`.
- Standard Tk dialogs and message boxes are used.

### Likely code paths to change

#### Root window

```python
class UpscaleGUI(tk.Tk):
```

would become a `CTk` subclass or equivalent composition. Appearance mode and theme should be configured before widget construction.

#### Common controls

Direct replacements are available for most frames, labels, buttons, entries, comboboxes, progress bars, scrollbars, radio buttons, and checkboxes.

However, constructor and configuration names are not perfectly identical. Typical differences include:

- `background`/`bg` → `fg_color` or `bg_color`
- `foreground` → `text_color`
- `ttk` style names → direct widget arguments or JSON theme values
- `orient` → `orientation` for some controls
- Progress values may be normalized through `.set()` rather than relying only on a Tk variable/maximum combination
- Some widget width and height values are pixels rather than character units
- Fonts should be tuples or `CTkFont`
- Transparent backgrounds use `"transparent"`

Mechanical search-and-replace is therefore unsafe.

#### Label frames

There is no `CTkLabelFrame`. The app uses label frames for major sections such as:

- Upscale
- Models
- Input
- Settings
- Output

Options:

1. Keep `ttk.LabelFrame`.
2. Replace each with `CTkFrame` plus a heading `CTkLabel`.
3. Create one small reusable labeled-section component.

For a restrained migration, option 1 or one minimal shared component is preferable to duplicating custom header markup across every section.

#### Listboxes

The model and queue lists are `tk.Listbox`. There is no built-in `CTkListbox`.

The lowest-risk approach is to retain `tk.Listbox` and explicitly style its:

- Background
- Foreground
- Selection background
- Selection foreground
- Border
- Highlight
- Font

Building a custom list out of buttons or labels would risk worse keyboard navigation, selection semantics, and performance.

#### Spinboxes

The settings area uses several `ttk.Spinbox` controls. There is no built-in `CTkSpinbox`.

Retain `ttk.Spinbox` initially. A custom composite spinbox should only be added if the mixed appearance is unacceptable.

#### Separators

There is no documented `CTkSeparator`. Use a one- or two-pixel `CTkFrame` or retain `ttk.Separator`.

#### Canvas and preview

The existing scroll area and preview should not be blindly changed to `CTkCanvas`.

The current `tk.Canvas` code provides:

- Scroll-region management
- Embedded frames
- Mouse-wheel bindings
- Width synchronization
- Image/preview rendering elsewhere in the app

`CTkScrollableFrame` could replace the scroll-wrapper code, but only after checking:

- Dynamic section hiding/showing
- Width behavior
- Mouse-wheel behavior on nested controls
- Linux wheel events
- Keyboard scrolling
- Preview and bottom-panel sizing

The simplest migration is to keep the canvas behavior first and replace it only in a later focused phase.

#### Tab migration

The future `ttk.Notebook` plan maps naturally to `CTkTabview`.

Differences include:

- Tabs are selected through a segmented-button-style header.
- Tabs are addressed by unique names.
- Existing notebook keyboard expectations may differ.
- RTL positioning and tab-order behavior must be tested.
- Programmatic APIs use `.add()`, `.tab()`, `.set()`, `.get()`, `.move()`, `.rename()`, and `.delete()`.

#### Dialogs

Continue using:

- `tkinter.filedialog`
- `tkinter.messagebox`

`CTkInputDialog` exists, but it does not replace all standard file, warning, error, and confirmation dialogs.

### Font and geometry differences

CustomTkinter dimensions often behave as pixel dimensions rather than traditional Tk character units. Existing values such as combobox widths and spinbox widths will not necessarily produce the same layout.

Expected migration work includes:

- Retuning control widths.
- Increasing row height.
- Checking clipping at 100%, 125%, 150%, and 200% scaling.
- Checking long translated strings.
- Avoiding fixed geometry where Arabic or German-like expansion could overflow.
- Reassessing the current `1200x900` geometry and `1000x700` minimum.
- Rechecking the fixed 220-pixel preview wrapper.
- Rechecking bottom controls at smaller displays.

Current padding constants provide a useful central migration point, but CustomTkinter's rounded widgets and larger default control heights will change visual density.

### Estimated effort

For visual migration only:

- **Moderate effort**, not a true one-line drop-in conversion.
- Approximately 2-4 focused development days for the primary window and child components, plus cross-platform testing.

For the complete planned overhaul including themes, splash, EN/AR, RTL, DnD, accessibility, and packaging:

- CustomTkinter addresses only part of the work.
- A realistic schedule is multiple phases and substantially more than the base widget conversion.

### Lowest-risk migration sequence

1. Pin CustomTkinter and build a tiny packaging proof.
2. Convert the root plus one isolated section.
3. Establish the custom JSON palette.
4. Convert straightforward buttons, labels, frames, entries, and progress bars.
5. Keep listboxes, spinboxes, canvases, dialogs, and separators initially.
6. Add `CTkTabview` only when the actual navigation overhaul begins.
7. Add runtime light/dark switching.
8. Test packaging on all three platforms.
9. Implement i18n independently.
10. Validate Arabic shaping, mirroring, keyboard navigation, and screen-reader behavior before full rollout.

## 12. Code sample

### Minimal Hello World

The official documentation provides the following basic structure:

```python
import customtkinter


def button_callback():
    print("button clicked")


app = customtkinter.CTk()
app.geometry("400x150")

button = customtkinter.CTkButton(
    app,
    text="my button",
    command=button_callback,
)
button.pack(padx=20, pady=20)

app.mainloop()
```

The repository README additionally demonstrates configuring appearance mode and theme before constructing the window:

```python
import customtkinter

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")

app = customtkinter.CTk()
app.geometry("400x240")


def button_function():
    print("button pressed")


button = customtkinter.CTkButton(
    master=app,
    text="CTkButton",
    command=button_function,
)
button.place(
    relx=0.5,
    rely=0.5,
    anchor=customtkinter.CENTER,
)

app.mainloop()
```

### Notebook with status bar and dark-mode toggle

The official docs provide separate examples for `CTkTabview` and runtime appearance switching, rather than a single combined status-bar application. The following minimal sample combines those documented APIs:

```python
import customtkinter as ctk

ctk.set_appearance_mode("system")
ctk.set_default_color_theme("dark-blue")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Anime Upscaler")
        self.geometry("720x480")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.tabs = ctk.CTkTabview(self)
        self.tabs.grid(
            row=0,
            column=0,
            padx=16,
            pady=(16, 8),
            sticky="nsew",
        )

        upscale_tab = self.tabs.add("Upscale")
        models_tab = self.tabs.add("Models")

        ctk.CTkLabel(
            upscale_tab,
            text="Upscale controls",
        ).pack(padx=20, pady=20)

        ctk.CTkLabel(
            models_tab,
            text="Installed models",
        ).pack(padx=20, pady=20)

        self.status = ctk.CTkLabel(
            self,
            text="Ready",
            anchor="w",
        )
        self.status.grid(
            row=1,
            column=0,
            padx=16,
            pady=(0, 8),
            sticky="ew",
        )

        self.dark_mode = ctk.CTkSwitch(
            self,
            text="Dark mode",
            command=self.toggle_dark_mode,
        )
        self.dark_mode.grid(
            row=2,
            column=0,
            padx=16,
            pady=(0, 16),
            sticky="w",
        )

    def toggle_dark_mode(self):
        mode = "dark" if self.dark_mode.get() else "light"
        ctk.set_appearance_mode(mode)
        self.status.configure(text=f"Appearance: {mode}")


app = App()
app.mainloop()
```

This uses documented APIs:

- `CTk`
- `CTkTabview`
- `CTkLabel`
- `CTkSwitch`
- `set_appearance_mode()`
- `set_default_color_theme()`

This is an adapted integration sample, not a verbatim single example published on docs site.

## 13. Pros and cons

### Pros

- Fits the existing Tkinter architecture without replacing the event loop.
- Existing Tk variables, geometry managers, callbacks, bindings, dialogs, and worker queues remain usable.
- Provides an immediate visual improvement over default Tkinter.
- Built-in light, dark, and system appearance modes.
- Appearance mode can switch live.
- Custom colors can specify separate light and dark values.
- JSON themes provide a practical centralized palette.
- `CTkTabview` fits the planned notebook/navigation overhaul.
- `CTkProgressBar`, buttons, labels, entries, frames, comboboxes, switches, and scrollable frames cover most common controls.
- High-DPI support is valuable for artists and video editors using high-resolution displays.
- Cross-platform support targets Windows, macOS, and Linux.
- MIT license is commercially permissive.
- Small package compared with the application's ML dependencies.
- Pure Python and no effect on GPU/VRAM use.
- Can be mixed incrementally with existing Tk and `ttk` widgets.
- Active enough to have a recent 6.0.0 release and a large user community.
- `CTkImage` and themed controls suit an anime-oriented visual identity.

### Cons

- No built-in localization framework.
- No first-class RTL layout or Arabic support.
- No built-in drag-and-drop.
- No built-in splash-screen abstraction.
- No built-in `CTkListbox`.
- No built-in `CTkSpinbox`.
- No built-in `CTkLabelFrame`.
- No built-in `CTkSeparator`.
- Theme documentation appears behind version 6.0.0 in places.
- Full palette switching may require rebuilding the UI.
- Linux `"system"` appearance mode falls back to light.
- Accessibility and keyboard navigation are incomplete or insufficiently documented.
- Screen-reader support is unknown.
- Canvas-drawn widgets may be less native and less accessible.
- Custom themes do not guarantee WCAG contrast.
- Official PyInstaller instructions require `--onedir` and manual asset inclusion.
- Mixing CTk and `ttk` controls can produce inconsistent appearance.
- Migration involves more than renaming imports because option names and sizing semantics differ.
- Large open-issue count and uneven release cadence create maintenance risk.

## 14. Known limitations

### Missing controls

CustomTkinter does not cover all of Tkinter/`ttk`. In particular, the current application needs to account for absent direct replacements for:

- Listbox
- Spinbox
- LabelFrame
- Separator
- Treeview/table
- Menubar/menu system
- Native file dialogs
- Rich text
- Full canvas functionality

### Linux system-theme detection

The official appearance-mode documentation states that `"system"` always resolves to light on Linux because OS appearance detection is not currently available.

The application should therefore expose an explicit Light/Dark setting on Linux rather than relying only on System.

### macOS title-bar appearance

The project README says a sufficiently recent Tcl/Tk is needed for a dark macOS window header. The packaged Python/Tk runtime matters, not only CustomTkinter.

### Packaging constraints

The official Windows PyInstaller guide says:

- `--onefile` cannot be used under its documented approach.
- `--onedir` is required.
- The CustomTkinter package directory must be manually added to include JSON and OTF assets.

This is a significant deployment gotcha for a vendored standalone application.

### Runtime theme switching

Appearance modes switch live, but complete color-theme changes are not equally seamless. The bundled showroom destroys and recreates its window after changing the color theme. Treat arbitrary palette switching as requiring a UI rebuild or restart.

### Accessibility gaps

Known signals include:

- An open accessibility/keyboard-navigation issue.
- An abandoned button-focus pull request.
- No documented screen-reader contract.
- No automatic focus-ring theme.
- No WCAG contrast tooling.

### RTL gaps

- No automatic mirrored layout.
- No bidi or Arabic shaping layer.
- Historical issue reporting reversed RTL text.
- Open request for combo dropdown justification.
- Mixed-script editing and cursor behavior depend on Tk/platform behavior.
- Tabs, segmented controls, and pack/grid direction need explicit testing.

### Canvas-based rendering

CustomTkinter draws many modern controls through canvas-based internals. Possible consequences include:

- Non-native appearance and behavior in edge cases.
- Different focus handling from native widgets.
- Potential screen-reader limitations.
- More redraw overhead than simple native controls.
- Visual artifacts during scaling or resizing in some environments.
- Different behavior for event bindings because internal child canvas/text items may receive events.

For this app, training/inference workload will dominate performance. CustomTkinter itself should not meaningfully affect GPU throughput, but huge dynamically updated widget lists should still be avoided.

### Threading

CustomTkinter does not change Tkinter's thread-safety rules. UI changes must remain on the Tk main thread. The current worker/event-queue polling design is compatible with this model and should be retained.

### Standard Tk incompatibilities

Potential migration mistakes include:

- Passing `bg`/`background` rather than CTk color options.
- Treating width as character count when CTk expects pixels.
- Assuming every `ttk.Style` configuration applies to CTk widgets.
- Assuming all Tk widget options are accepted.
- Assuming internal canvas components have the same binding behavior as native widgets.
- Assuming CTk progress variables and ranges behave identically to `ttk.Progressbar`.

### Release metadata inconsistency

The package metadata simultaneously references CC0 and MIT, although the repository license and classifier identify MIT. Version 6.0.0 also has different dates in the changelog and PyPI.

These inconsistencies are not necessarily functional defects, but they reduce confidence in release-process polish.

### Documentation freshness

The GitHub wiki explicitly says it is outdated and redirects users to the official website. The website itself lists only three color themes while the 6.0.0 changelog adds a fourth, gold theme. Source and changelog inspection may be required when documentation and current behavior differ.

### Windows-specific considerations

- PyInstaller requires explicit package-data handling under the official recipe.
- ICO handling differs from PNG icon handling on other platforms.
- DPI scaling should be tested at multiple Windows display scale factors.
- TkDND/`tkinterdnd2` packaging adds native files and architecture-specific concerns.
- Avoid calling DPI-awareness APIs independently without checking CustomTkinter's own DPI handling.
- Native dialogs may not match CustomTkinter's dark theme.
- Custom window chrome is not supplied; the OS title bar remains platform-controlled.

## 15. Fit assessment

### Score: 7/10

CustomTkinter is a **good but incomplete fit** for this anime upscaler GUI.

### Why it fits

1. **The application is already Tkinter.**
   This is CustomTkinter's strongest advantage. The current event loop, queue polling, background worker, settings, model registry, preview logic, file dialogs, and pipeline integration can remain.

2. **The planned theme overhaul maps well to CustomTkinter.**
   Dark/light appearance, light/dark color pairs, JSON palettes, rounded widgets, custom fonts, progress bars, and tab views address much of the planned visual modernization.

3. **It is lightweight relative to the ML application.**
   It adds no GPU workload and negligible package size compared with PyTorch and model files. The 8 GB VRAM target is unaffected.

4. **Incremental migration is supported.**
   Existing listboxes, spinboxes, canvases, and dialogs can remain while high-visibility controls migrate first.

5. **Cross-platform scope matches the target audience.**
   Windows, macOS, and Linux are supported, with known caveats.

6. **The license allows redistribution.**
   MIT is appropriate for a standalone commercial or open-source desktop app.

### Why it is not an 8-10

1. **RTL is a major planned requirement, not an edge case.**
   CustomTkinter provides no RTL engine, layout mirroring, localization system, or documented Arabic guarantee. This project must build and test that work independently.

2. **Accessibility is insufficiently mature.**
   The target users may depend on keyboard navigation and assistive technologies, but the framework has unresolved accessibility questions and uses canvas-drawn controls.

3. **Important current widgets have no CTk replacement.**
   The application relies on listboxes, spinboxes, label frames, separators, and canvases. A mixed UI is unavoidable unless custom controls are built.

4. **Standalone packaging is not frictionless.**
   The official PyInstaller instructions require an onedir build and manual package-data inclusion. This must be reconciled with the app's distribution format.

5. **CustomTkinter does not address several overhaul phases.**
   Splash lifecycle, drag-and-drop, i18n, RTL, accessibility, and platform branding remain application or packaging work.

### Project-specific recommendation

Adopt CustomTkinter only after a small cross-platform proof validates these four points:

1. A packaged `CTk` application starts correctly with bundled themes and fonts.
2. Existing `tk.Listbox`, `ttk.Spinbox`, dialogs, and preview canvases look acceptable inside the CTk window.
3. Arabic labels, entries, combo selections, and mixed Arabic/Latin paths render and edit correctly.
4. Keyboard-only navigation works for the actual upscaling workflow.

If the proof succeeds, use CustomTkinter as the **visual shell and theme layer**, not as a mandate to replace every native Tk widget.

Recommended architecture:

- CustomTkinter for root, frames, headings, buttons, entries, progress, switches, tabs, and scroll containers.
- Existing Tk/`ttk` for listboxes, spinboxes, native dialogs, and canvas-based previews where they are functionally superior.
- Application-owned `gettext` or equivalent i18n.
- Application-owned RTL layout policy.
- `tkinterdnd2` only if file drag-and-drop is approved.
- A bundled app theme JSON with verified light/dark contrast.
- Exact dependency pinning and per-platform packaging tests.

This approach captures most of CustomTkinter's value without creating unnecessary custom widgets.

## 16. Documentation links

### Primary sources

- GitHub repository: https://github.com/TomSchimansky/CustomTkinter
- Official website: https://customtkinter.tomschimansky.com/
- Official documentation: https://customtkinter.tomschimansky.com/documentation/
- Official tutorial: https://customtkinter.tomschimansky.com/tutorial/
- Official showcase: https://customtkinter.tomschimansky.com/showcase/
- PyPI: https://pypi.org/project/customtkinter/
- GitHub wiki (marked outdated): https://github.com/TomSchimansky/CustomTkinter/wiki

### Feature documentation

- Colors and themes: https://customtkinter.tomschimansky.com/documentation/color/
- Appearance mode: https://customtkinter.tomschimansky.com/documentation/appearancemode/
- Scaling: https://customtkinter.tomschimansky.com/documentation/scaling/
- Packaging: https://customtkinter.tomschimansky.com/documentation/packaging/
- Widget index: https://customtkinter.tomschimansky.com/documentation/widgets/
- CTkTabview: https://customtkinter.tomschimansky.com/documentation/widgets/tabview/
- Window documentation: https://customtkinter.tomschimansky.com/documentation/windows/
- Utility classes: https://customtkinter.tomschimansky.com/documentation/utility-classes/

### Code and support

- Examples directory: https://github.com/TomSchimansky/CustomTkinter/tree/master/examples
- Changelog: https://github.com/TomSchimansky/CustomTkinter/blob/master/CHANGELOG.md
- License: https://github.com/TomSchimansky/CustomTkinter/blob/master/LICENSE
- Built-in theme example: https://github.com/TomSchimansky/CustomTkinter/blob/master/customtkinter/assets/themes/dark-blue.json
- GitHub issues: https://github.com/TomSchimansky/CustomTkinter/issues
- GitHub discussions: https://github.com/TomSchimansky/CustomTkinter/discussions
- Stack Overflow tag: https://stackoverflow.com/questions/tagged/customtkinter
- Accessibility issue: https://github.com/TomSchimansky/CustomTkinter/issues/2786
- RTL text issue: https://github.com/TomSchimansky/CustomTkinter/issues/1540
- Combo dropdown justification: https://github.com/TomSchimansky/CustomTkinter/issues/2759

### Video resources

The official repository README embeds demonstration videos for:

- Windows appearance-mode switching
- macOS appearance-mode and scaling switching
- TkinterMapView integration

The most reliable entry point is the repository README:
https://github.com/TomSchimansky/CustomTkinter#more-examples-and-showcase

No separate current official video-course URL was identified.

## Top 3 fit-for-this-project findings

1. **Strong incremental migration path:** Most visible controls can move to CTk without changing the worker queues, inference pipeline, settings, dialogs, or Tk event model.
2. **Theme overhaul alignment:** Runtime light/dark switching, paired colors, JSON palettes, `CTkTabview`, and DPI scaling directly support the planned visual overhaul.
3. **Low runtime cost:** CustomTkinter is small, pure Python, and does not affect the app's approximately 8 GB VRAM budget.

## Top 3 risks or concerns

1. **No first-class Arabic/RTL support:** Localization, bidi correctness, layout mirroring, translated sizing, and cross-platform Arabic input remain application responsibilities.
2. **Accessibility gaps:** Keyboard navigation, focus indication, and screen-reader behavior are incomplete or undocumented, especially for canvas-drawn controls.
3. **Incomplete widget and packaging coverage:** Listbox, Spinbox, LabelFrame, Separator, DnD, and splash support are absent; official PyInstaller guidance requires an onedir build and explicit data inclusion.

## Research limitations

- GitHub's unauthenticated repository and commits API returned HTTP 403, so recent commit frequency could not be quantified reliably.
- GitHub showed current aggregate counts, but those counts can change after this report.
- No framework code was installed or executed; behavior was assessed from official web sources and source metadata.
- Arabic shaping, bidi editing, screen-reader behavior, DnD integration, DPI behavior, and packaging were not empirically tested on Windows, macOS, or Linux.
- PyPI reports version 6.0.0 as released June 24, 2026, while the changelog dates it January 21, 2026; the reason for that discrepancy was not documented.
- The official site does not provide a canonical installed-size figure.
- The requested "Notebook with status bar + dark mode toggle" was not found as one verbatim official example; the report's sample combines separately documented official APIs.
