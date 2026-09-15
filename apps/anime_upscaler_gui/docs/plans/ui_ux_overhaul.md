# UI/UX Overhaul Plan — `anime_upscaler_gui`

**Status:** Approved 2026-07-25
**Scope:** `apps/anime_upscaler_gui/` (Tkinter desktop app)
**Audit:** [`audit/ui_ux_audit.md`](audit/ui_ux_audit.md)

---

## Locked Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Framework** | Tkinter + ttk (current) | Lowest risk; phases scale; CustomTkinter reserved as future upgrade |
| **Future upgrade path** | CustomTkinter (post-Phase 14) | 7/10 fit; `widgets/*.py` extraction makes migration mechanical |
| **Not this** | Reflex | Solves wrong problem; 4/10 |
| **Color palette** | Indigo Slate (Light + Dark + High-contrast) | All WCAG AA pass; slate + indigo, anime-friendly |
| **Icon style** | Phosphor (outline + solid + duotone) | MIT; 21 weights; reads at 16px; used by GitHub/Linear |
| **Splash screen** | Every launch, 1.5s display + 200ms fade | Confirmed |
| **Persistence** | Every change (incremental JSON patch) | Confirmed |
| **i18n** | EN + AR, gettext + Babel | Confirmed |
| **RTL** | Code-level mirroring + `<TkRightToLeft>` if available | Confirmed |

---

## Framework Decision (final)

| Criterion | Tkinter (current) | CustomTkinter | Reflex |
|---|---|---|---|
| Mode | Native desktop | Native desktop | Web (needs wrapper) |
| Migration effort | — | 2-4 days | 2-4 weeks (rewrite) |
| Visual ceiling | Low | Medium-high | High |
| Theme system | Manual | Built-in JSON + 2 modes | Radix + CSS |
| Dark mode | Manual | First-class | First-class |
| RTL / i18n | Manual | Manual | CSS capable |
| DnD | tkinterdnd2 | tkinterdnd2 | Enterprise only |
| Splash | Manual Toplevel | Manual Toplevel | Manual in-page |
| Screen reader | Tk + platform | Canvas-drawn, gaps | Browser mature |
| PyInstaller | `--onefile` OK | `--onedir` + add-data | Tauri/Electron |
| Bundle size | 0 KB | ~300 KB | MB-scale |
| Vendored standalone | Yes | Yes | Difficult |
| ML/VRAM impact | None | None | Extra JavaScript RAM |
| Worker / threading | Tk | Tk | WebSocket + async |
| **Best for** | Tool that works | Tool that looks good | Browser app |

**Verdict:** Keep Tkinter. Plan delivers 90% of wins. Re-evaluate CustomTkinter after Phase 14.

Full research: [`research/customtkinter_research.md`](research/customtkinter_research.md) · [`research/reflex_research.md`](research/reflex_research.md)

---

## Phase Sequencing Rationale

The order below minimizes risk and rework. Each phase ends with green tests + a working GUI; nothing leaves a phase broken.

| Phase | Title | Why this order |
|---|---|---|
| 0 | Token sweep | Cheap, no-risk, unblocks every visual change later |
| 1 | Data layer | Types before refactor; removes stringly-typed bugs |
| 2 | Widget extraction | Smaller classes before tabs; each testable in isolation |
| 3 | Tabbed layout | Refactor first, then add tabs |
| 4 | Settings & persistence | Needed before theme picker (theme is a setting) |
| 5 | Theme system | Tokens already centralized |
| 6 | Threading fixes | Must precede modal-blocking fixes |
| 7 | Interaction improvements | Shortcuts, DnD, confirmations |
| 8 | Empty/error states | Builds on tabbed layout |
| 9 | Telemetry & monitoring | Status bar / GPU monitor |
| 10 | Accessibility | Adds focus rings, high-contrast, status icons |
| 11 | Visual polish | Icons, spacing, typography, status tints |
| 12 | Documentation | Last, when story is complete |
| 13 | Splash screen | Every launch (1.5s + fade) |
| 14 | i18n (EN + AR) | gettext + Babel; RTL + Arabic fonts |

---

## Phase 0 — Token Sweep [S]

**Goal:** every color, font, padding lives in `ui_constants.py`. No literals in widget code.

### Tasks
- [ ] 0.1 Add new tokens to `ui_constants.py`:
  - `surface`, `surface_alt`, `border`, `border_strong`, `text_muted`
  - `accent_hover`, `accent_pressed`
  - `preview_bg`, `preview_label`, `listbox_bg`, `listbox_fg`
  - `status_bg_pending`, `status_bg_running`, `status_bg_done`, `status_bg_error` (tints)
- [ ] 0.2 Define spacing scale `space.xs=4, sm=8, md=12, lg=16, xl=24` — keep `PAD_X/PAD_Y/GROUP_PAD` as aliases for now
- [ ] 0.3 Define typography scale `FONT_CAPTION=9pt`, `FONT_BODY=10pt`, `FONT_TITLE=14pt bold`, `FONT_DISPLAY=20pt bold` — keep `FONT_BASE/HEADING` as aliases
- [ ] 0.4 Replace violations per `app.py:154, 185`, `preview.py:37, 39, 119, 132`, `cut_window.py:52, 53, 56, 59`
- [ ] 0.5 Add `grep -rn '#[0-9a-fA-F]\{3,6\}' anime_upscaler_gui/` to CI scripts (or pre-commit) — fails if a literal appears outside `ui_constants.py`
- [ ] 0.6 Keep `"#222"` style hex inside `ui_constants.py` only

### Acceptance
- [ ] `grep -rn '#[0-9a-fA-F]\{3,6\}' anime_upscaler_gui/` shows only `ui_constants.py`
- [ ] `pytest tests/` → 18/18 pass
- [ ] `python -m anime_upscaler_gui` launches with identical look

### Risk
- **None** — pure rename. Revert = restore literals.

---

## Phase 1 — Data Layer [M]

**Goal:** replace `List[dict]` job data and `tk.StringVar` business logic with typed dataclasses.

### Tasks
- [ ] 1.1 Create `state.py` with:
  ```python
  class JobStatus(str, Enum):
      PENDING = "pending"
      RUNNING = "running"
      DONE = "done"
      ERROR = "error"
      SKIPPED = "skipped"
      CANCELLED = "cancelled"

  @dataclass
  class Job:
      id: int
      input: Path
      output: Path
      status: JobStatus = JobStatus.PENDING
      error: str = ""
      fps: float = 0.0
      infer_ms: float = 0.0
      model_filename: str = ""
      kind: str = ""
      scale: int = 4
      is_video: bool = False
      cut_start: float = 0.0
      cut_end: float = 0.0
  ```
- [ ] 1.2 Create `FormState` dataclass mirroring `_Defaults` but with `__post_init__` validators (`outscale ∈ [1,8]`, `batch_size ∈ [1,16]`, `tile_size ∈ [64,1024]`)
- [ ] 1.3 Add `state.py` to `from .state import Job, JobStatus, FormState` in `app.py`
- [ ] 1.4 Replace `self._jobs: List[dict]` with `self._jobs: List[Job]`; rewrite `_add_files`, `_refresh_queue_listbox`, `_retry_selected`, `_remove_selected`, `_enqueue_next`, `_on_start`, `_handle_event` to use dataclass
- [ ] 1.5 Replace `STATUS_COLOR_MAP = STATUS_COLORS` (dead alias) with `JobStatus → color` map
- [ ] 1.6 Update `test_registry.py` to use new `InstalledModel` (no dict leak through)
- [ ] 1.7 Update `settings.py` `_Defaults` to extend `FormState`; add `version: int = 1` field

### Acceptance
- [ ] `pytest tests/` → still 18/18 pass
- [ ] App launches, single + batch modes work, errors display correctly
- [ ] No `j["status"] == "..."` string compares remain

### Risk
- **Medium** — typing changes ripple through `app.py`. Mitigate by introducing `Job` parallel to dict, then flipping all readers, then removing dict.

---

## Phase 2 — Widget Extraction [M]

**Goal:** split `app.py` into a thin shell + per-tab widget classes. Each widget testable in isolation.

### Tasks
- [ ] 2.1 Create `widgets/__init__.py` (empty)
- [ ] 2.2 Create `widgets/input_panel.py` — extract `_build_input_section`, `_on_mode_change`, `_add_files`, `_clear_jobs`, `_retry_selected`, `_remove_selected`, `_pick_input` (delete), `_refresh_queue_listbox`, `_visible_jobs` into `_InputPanel(ttk.Frame)`
- [ ] 2.3 Create `widgets/model_panel.py` — extract `_build_model_section`, `_refresh_model_dropdown`
- [ ] 2.4 Create `widgets/settings_panel.py` — extract `_build_settings_section` (4 rows)
- [ ] 2.5 Create `widgets/output_panel.py` — extract `_build_output_section`, `_pick_output_dir`
- [ ] 2.6 Create `widgets/models_panel.py` — extract `_build_tab2_models`, `_build_models_list`, `_build_models_actions`, `_refresh_models_listbox`, `_refresh_preset_dropdown`, `_download_preset`, `_download_custom_url`, `_start_download`, `_import_local`, `_after_import`
- [ ] 2.7 Create `widgets/status_bar.py` — extract `_build_status_section`, `_build_actions_section` into `_StatusBar`
- [ ] 2.8 Wire preview + cut_window to be passed in via constructor (not auto-built)
- [ ] 2.9 `app.py` becomes ~150 lines: imports + `__init__` + `_build_ui` + `_on_close` + `_poll_events` + `_handle_event`
- [ ] 2.10 Add `tests/test_widgets.py` — instantiate each panel headlessly; assert no exception

### Acceptance
- [ ] `app.py` < 200 lines
- [ ] Each `widgets/*.py` < 250 lines
- [ ] `pytest tests/` → 18 + 4 panels = 22+ pass
- [ ] GUI behavior identical to baseline

### Risk
- **Medium** — large mechanical refactor. Mitigate by extracting one panel at a time, running tests after each.

---

## Phase 3 — Tabbed Layout [M]

**Goal:** wrap existing widgets in a `ttk.Notebook`. Smaller, more focused screens.

### Tasks
- [ ] 3.1 Convert `_build_ui` to use `ttk.Notebook` with 5 tabs: **Upscale**, **Batch Queue**, **Models**, **Settings**, **Log**
- [ ] 3.2 Upscale tab: Input + Model + Settings + Output + Cut + Preview + Status
- [ ] 3.3 Batch Queue tab: input listbox + filter + queue listbox + status per row
- [ ] 3.4 Models tab: preset catalog + installed list + custom URL + import + progress
- [ ] 3.5 Settings tab: settings panel + storage folder + theme picker (placeholder for Phase 5)
- [ ] 3.6 Log tab: `ScrolledText` widget bound to log handler
- [ ] 3.7 Add `tabs.py` if needed for IconTab definitions
- [ ] 3.8 Add `tests/test_tabs.py` — verify each tab exists with the right title

### Acceptance
- [ ] All 5 tabs accessible via `Ctrl+Tab` and tabs are clickable
- [ ] Tab state preserved when switching (e.g., scroll position)
- [ ] `pytest tests/` → still pass

### Risk
- **Low** — additive; old layout kept as fallback initially.

---

## Phase 4 — Settings & Persistence [M]

**Goal:** schema versioning, queue persistence, storage folder picker.

### Tasks
- [ ] 4.1 Add `"version": 1` to `_Defaults`; `_coerce` migrates older files
- [ ] 4.2 Implement `SettingsController.save()` with atomic write (`tempfile + os.replace`)
- [ ] 4.3 Add `SettingsController.export(path)` / `import(path)` methods
- [ ] 4.4 Add `paths_app_dir_picker` — "Settings → Storage folder..." button (uses `_AppPaths.move_to`)
- [ ] 4.5 Add `QueueController.persist(jobs)` — writes JSON **incremental patch** (RFC 6902) to `cache_dir/queue.json` on every change
- [ ] 4.6 Add `QueueController.load()` — restores on startup; cap at 1000 items
- [ ] 4.7 Add `tests/test_settings_migration.py` — version 0 → 1 migration
- [ ] 4.8 Add `tests/test_queue_persistence.py` — save+load round-trip

### Acceptance
- [ ] Old `settings.json` (no `version`) loads as v1
- [ ] Queue survives close + reopen
- [ ] Storage folder move preserves all data
- [ ] Atomic write is visible via watcher (no half-written file)

### Risk
- **Medium** — backward compat. Mitigate with versioned migration.

---

## Phase 5 — Theme System [M]

**Goal:** Theme dataclass + light (default) / dark / high-contrast themes; theme picker in Settings.

### Tasks
- [ ] 5.1 Create `theme.py` with `Theme` dataclass + `to_ctk_theme_dict()` hook for future CustomTkinter migration
- [ ] 5.2 Define `LIGHT`, `DARK`, `HIGH_CONTRAST` constants per **Indigo Slate** palette
- [ ] 5.3 Move every token from `ui_constants.py` into `Theme`; keep module-level constants as `LIGHT.<token>` aliases
- [ ] 5.4 Add `ThemeController.apply(theme_name)` — rebuilds `ttk.Style` with new colors
- [ ] 5.5 Add `Settings → Theme` radio: Light / Dark / High contrast
- [ ] 5.6 Add theme preview swatch (16x16 strip per theme) in the Settings picker
- [ ] 5.7 Add `tests/test_theme.py` — apply all 3 themes, assert no exception
- [ ] 5.8 Add `WCAG` contrast check in `tests/test_theme.py` — assert 4.5:1 body, 3:1 large/UI
- [ ] 5.9 Add `tests/test_accessibility_contrast.py` — verify each theme's `accent-on-bg`, `text-on-bg`, `error-on-bg` ratios
- [ ] 5.10 Add **24 tokens** (4.1 expand): `surface`, `surface_alt`, `border`, `border_strong`, `text_muted`, `accent_hover`, `accent_pressed`, `preview_bg`, `preview_label`, `listbox_bg`, `listbox_fg`, `status_bg_pending/running/done/error/skip/cancel`
- [ ] 5.11 Implement status-tinted row backgrounds
- [ ] 5.12 Add **3 theme radios** in Settings → Theme

### Acceptance
- [ ] Switching theme is instant (no restart)
- [ ] All 3 themes pass WCAG AA
- [ ] No flicker during theme switch (only affected widgets repaint)

### Risk
- **Low** — additive; default = light keeps current look.

### Indigo Slate reference

| Token | Light | Dark | High-contrast |
|---|---|---|---|
| `bg` | #FAFAFA | #0F172A | #FFFFFF |
| `surface` | #FFFFFF | #1E293B | #FFFFFF |
| `surface_alt` | #F3F4F6 | #334155 | #FFFFFF |
| `border` | #E5E7EB | #334155 | #000000 |
| `border_strong` | #D1D5DB | #475569 | #000000 |
| `text` | #111827 | #F1F5F9 | #000000 |
| `text_muted` | #6B7280 | #94A3B8 | #000000 |
| `accent` | #4F46E5 | #818CF8 | #0000EE |
| `accent_hover` | #4338CA | #A5B4FC | #0000AA |
| `accent_pressed` | #3730A3 | #C7D2FE | #000088 |
| `success` | #10B981 | #34D399 | #006400 |
| `warn` | #F59E0B | #FBBF24 | #B8860B |
| `error` | #EF4444 | #F87171 | #8B0000 |
| `status_bg_pending` | #F3F4F6 | #1E293B | #FFFFFF |
| `status_bg_running` | #DBEAFE | #1E3A8A | #E0E0FF |
| `status_bg_done` | #D1FAE5 | #064E3B | #C0FFC0 |
| `status_bg_error` | #FEE2E2 | #7F1D1D | #FFC0C0 |
| `preview_bg` | #1F2937 | #0F172A | #000000 |
| `preview_label` | #F9FAFB | #F1F5F9 | #FFFFFF |
| `listbox_bg` | #FFFFFF | #1E293B | #FFFFFF |
| `listbox_fg` | #111827 | #F1F5F9 | #000000 |

---

## Phase 6 — Threading Fixes [M]

**Goal:** unblock GUI thread on frame errors; recover from worker crashes.

### Tasks
- [ ] 6.1 Replace `_current_resume_box: dict` with `self._resume_boxes: Dict[int, dict]` keyed by `job_id`
- [ ] 6.2 Refactor `_ask_frame_error_action` to use `Toplevel` only (no `wait_window()`); close via callback
- [ ] 6.3 Wrap `_PipelineWorker` lifecycle in `try/except`; on crash, post fatal-error event
- [ ] 6.4 Add `_handle_event` case for `kind == "fatal_error"` — show modal, offer to save logs
- [ ] 6.5 Add tests for `async_blocking.py` — verify modal doesn't block mainloop
- [ ] 6.6 Add `tests/test_threading.py` — concurrent jobs don't share resume boxes

### Acceptance
- [ ] Frame error modal does **not** freeze the rest of the UI
- [ ] Worker crash surfaces a clear modal with a "Copy log" button
- [ ] Concurrent jobs each get their own resume box

### Risk
- **Medium** — touchy threading code. Mitigate with extensive tests.

---

## Phase 7 — Interaction Improvements [L]

**Goal:** keyboard shortcuts, drag-and-drop, confirmation dialogs, error copy localization.

### Tasks
- [ ] 7.1 Add `tkinterdnd2` dependency to `requirements.txt`
- [ ] 7.2 Enable DnD on the input listbox (`drop_target_register`, `dnd_bind('<<Drop>>', ...)`)
- [ ] 7.3 Add `widgets/menu_bar.py` — File / Edit / Tools / Help menu
- [ ] 7.4 Bind accelerators:
  - `Ctrl+O` → Add files
  - `Ctrl+S` → Save settings
  - `Ctrl+Enter` → Start
  - `Delete` → Remove selected
  - `F5` → Re-scan models
  - `Ctrl+Z` / `Ctrl+Y` → Undo/Redo queue action
  - `Ctrl+1..5` → Switch tabs
  - `Ctrl+Shift+L` → Toggle language
- [ ] 7.5 Add `_confirm_destructive` helper — wraps `messagebox.askyesno` with consistent copy
- [ ] 7.6 Replace `_pick_input` with delete (dead code)
- [ ] 7.7 Localize `_on_start` warnings: "Tainted checkpoint" → "This model may produce flat-blue output. Continue?"
- [ ] 7.8 Localize model tags: `[unsupported]` → `(unsupported - this build lacks X architecture)`
- [ ] 7.9 Map user-facing errors: `URLError` → "Network unreachable", `OSError 28` → "Disk full", `RuntimeError` invalid checkpoint → "This file is not a valid model"
- [ ] 7.10 Add `tests/test_shortcuts.py` — verify each binding fires the right handler
- [ ] 7.11 Add `tests/test_dnd.py` — mock drop event, verify file added
- [ ] 7.12 Bundle `assets/icons/phosphor/outline/` (16x16 PNG)
- [ ] 7.13 Bundle `assets/icons/phosphor/solid/` (16x16 PNG)

### Acceptance
- [ ] All shortcuts work; file menu shows hints
- [ ] Drag from Explorer/Finder adds files to input listbox
- [ ] All destructive actions confirm
- [ ] No "bias_mean ~0.43" jargon in user-facing strings

### Risk
- **Low** — additive.

---

## Phase 8 — Empty / Error States [M]

**Goal:** every empty state has a CTA; every error has a clear next action.

### Tasks
- [ ] 8.1 Add `widgets/empty_state.py` — illustration + title + subtitle + CTA button
- [ ] 8.2 Models empty state: "Get started by downloading a model" + featured preset CTA
- [ ] 8.3 Input empty state: "+ Add files" + drag-drop hint
- [ ] 8.4 Job errored: inline "Retry" button + copy error button
- [ ] 8.5 Pre-flight check: at Start, verify output folder writable; if not, show banner "Will create X" or "Permission denied at Y"
- [ ] 8.6 CUDA detection: on startup, show banner "GPU not detected. Running on CPU (slower)." if `not torch.cuda.is_available()`
- [ ] 8.7 VRAM overflow UX: "This image needs ~6 GB; you have 4 GB. Tile it?" with explicit options
- [ ] 8.8 Download progress: show speed (MB/s) + ETA + Cancel button
- [ ] 8.9 Settings corruption: notify "Settings file was corrupt; reset to defaults. Old file: .bak"
- [ ] 8.10 Add `tests/test_empty_states.py` — each empty state has the right CTA

### Acceptance
- [ ] No empty area is just blank space
- [ ] Every error has a clear next action

### Risk
- **Low** — additive.

---

## Phase 9 — Telemetry & Monitoring [M]

**Goal:** GPU monitor widget, status bar, log panel, completion toasts.

### Tasks
- [ ] 9.1 Add `widgets/gpu_monitor.py` — polls `torch.cuda` every 1s; shows util%, VRAM used/total, temp
- [ ] 9.2 Add `widgets/status_bar.py` improvements — three segments: GPU monitor, current job, global ETA
- [ ] 9.3 Add `widgets/log_panel.py` — `ScrolledText` bound to Python logging handler
- [ ] 9.4 Add `widgets/toast.py` — transient notification (`AfterJobSave`, `AfterDownload`, `AfterError`)
- [ ] 9.5 Wire `logging` to all modules (`app.py`, `pipeline.py`, `registry.py`, etc.) with `RotatingFileHandler` in `logs_dir`
- [ ] 9.6 Add `tests/test_gpu_monitor.py` — mock `torch.cuda`; verify timer fires
- [ ] 9.7 Add `tests/test_log_panel.py` — verify log lines appear

### Acceptance
- [ ] GPU monitor updates every 1s, no flickering
- [ ] Log panel captures all `logger.info/warning/error` calls
- [ ] Toasts auto-dismiss after 3s

### Risk
- **Low** — additive.

---

## Phase 10 — Accessibility [M]

**Goal:** focus indicators, color+icon status, high-contrast theme, smaller min size.

### Tasks
- [ ] 10.1 Configure `ttk.Style` to show 2px accent outline on focused widgets
- [ ] 10.2 Replace color-only status with icon + text + color triplet
- [ ] 10.3 Generate status icons (16x16 PNG): `pending`, `running`, `done`, `error`, `skipped`, `cancelled`
- [ ] 10.4 Reduce `minsize` from 1000x700 to 800x600
- [ ] 10.5 Calculate `wraplength` dynamically from `winfo_width()` on `<Configure>`
- [ ] 10.6 Add `wm_attributes("-toolwindow", True)` to all modal dialogs
- [ ] 10.7 Bind `<Return>` on Spinbox/Entry to commit values
- [ ] 10.8 Add `tests/test_focus.py` — verify Tab navigates in order
- [ ] 10.9 Add `tests/test_screen_reader.py` — verify all icon images have `text` fallback

### Acceptance
- [ ] All status indicators have icon + text + color
- [ ] Tab key navigates all controls in logical order
- [ ] 800x600 screens render the full Upscale tab without clipping

### Risk
- **Low**.

---

## Phase 11 — Visual Polish [L]

**Goal:** icons, spacing scale, typography scale, status tints, window branding.

### Tasks
- [ ] 11.1 Generate 16x16 PNG icons (Phosphor outline): folder-open, play, save, refresh, download, trash, plus, minus, settings, help
- [ ] 11.2 Bundle icons in `assets/icons/phosphor/outline/`
- [ ] 11.3 Replace text-only buttons with icon+text where space allows
- [ ] 11.4 Replace `ttk.Separator` heavy dividers with whitespace + alignment
- [ ] 11.5 Apply 4-pt spacing scale throughout (`space.xs/sm/md/lg/xl`)
- [ ] 11.6 Apply typography scale (caption/body/title/display)
- [ ] 11.7 Add status background tints to listbox rows (matching Phase 5 colors)
- [ ] 11.8 Update window title to "Anime Upscaler" (proper case)
- [ ] 11.9 Generate 256x256 app icon (Phosphor duotone), set via `iconphoto`
- [ ] 11.10 Add splash screen (covered by Phase 13)
- [ ] 11.11 Add `tests/test_icons.py` — verify all icons load + render at 16x16
- [ ] 11.12 Phosphor solid icons for primary CTAs
- [ ] 11.13 Phosphor duotone app icon (256x256)

### Acceptance
- [ ] Every button has an icon (text optional)
- [ ] Spacing follows 4-pt scale everywhere
- [ ] Window title = "Anime Upscaler"
- [ ] App icon visible in taskbar/dock

### Risk
- **Low** — visual only.

---

## Phase 12 — Documentation [S]

**Goal:** README + AGENTS.md + onboarding docs reflect the new state.

### Tasks
- [ ] 12.1 Update `README.md` — remove "Known limitations" items that are now fixed (DnD, wizard, dark mode)
- [ ] 12.2 Add `docs/onboarding.md` — first-run experience walkthrough
- [ ] 12.3 Add `docs/themes.md` — theme tokens + how to add a new theme
- [ ] 12.4 Add `docs/dev/widgets.md` — widget contract + how to add a new panel
- [ ] 12.5 Update `AGENTS.md` Quick Commands with `--theme`, `--locale` flags if added
- [ ] 12.6 Add `tests/test_docs.py` — verify all CLI flags in README exist in `scripts/` or `app.py`

### Acceptance
- [ ] README + AGENTS.md cross-link correct
- [ ] Every CLI/setting referenced in docs actually exists

### Risk
- **None**.

---

## Phase 13 — Splash Screen [S]

**Goal:** every launch shows a 1.5s splash screen with the brand icon, then fades to the main window.

### Tasks
- [ ] 13.1 Create `widgets/splash.py` — `Splash` class extending `tk.Toplevel`, borderless, centered, fixed size
- [ ] 13.2 Show app icon (256x256 Phosphor duotone) + app name "Anime Upscaler" + tagline "Open-source anime super-resolution"
- [ ] 13.3 Background: `surface` token from active theme
- [ ] 13.4 On `__main__`: instantiate Splash, schedule fade-out after 1500ms, then create `UpscaleGUI`
- [ ] 13.5 Fade-out: 200ms `wm_attributes('-alpha', ...)`, every 20ms reduce 0.05
- [ ] 13.6 Add `--no-splash` CLI flag for tests / fast launch
- [ ] 13.7 Add `tests/test_splash.py` — verify duration + fade + dismissal

### Acceptance
- [ ] Splash appears on every launch
- [ ] Dismissable via `--no-splash`
- [ ] 1.5s display + 200ms fade = ~1.7s total
- [ ] No flicker when main window takes over

### Risk
- **Low** — additive.

---

## Phase 14 — i18n (English + Arabic) [M]

**Goal:** full localization for English + Arabic, with RTL mirroring for Arabic.

### Tasks

#### 14.1 — i18n framework
- [ ] 14.1.1 Add `babel` to `requirements.txt`
- [ ] 14.1.2 Create `i18n/` directory with `en/LC_MESSAGES/messages.po`, `ar/LC_MESSAGES/messages.po`
- [ ] 14.1.3 Create `i18n.py` with `t(key, **kwargs)` function supporting `_("...")` namespace
- [ ] 14.1.4 Add `pybabel extract` config + `pybabel update` / `pybabel compile` to Makefile
- [ ] 14.1.5 Add `app.locale` field to `_Defaults` (`"en"` or `"ar"`)
- [ ] 14.1.6 Add `Ctrl+Shift+L` shortcut to toggle locale

#### 14.2 — Font selection
- [ ] 14.2.1 Latin font: `"Segoe UI"` (Win) / `"Helvetica"` (mac) / `"DejaVu Sans"` (Linux) — current
- [ ] 14.2.2 Arabic font: `"Segoe UI Arabic"` (Win) / `"SF Arabic"` (mac) / `"Noto Sans Arabic"` (Linux, fallback)
- [ ] 14.2.3 Bundle `NotoSansArabic-Regular.ttf` and `NotoSansArabic-Bold.ttf` in `assets/fonts/` (Linux fallback)
- [ ] 14.2.4 In `Theme`, add `font_family_latin`, `font_family_arabic` fields
- [ ] 14.2.5 `FONT_BASE` tuple = `(arabic_family if locale=="ar" else latin_family, size)`

#### 14.3 — String extraction
- [ ] 14.3.1 Extract every user-facing string into `messages.pot` via `pybabel extract`
- [ ] 14.3.2 Translate to Arabic (right-to-left native speaker review needed)
- [ ] 14.3.3 Compile `.mo` files
- [ ] 14.3.4 Replace every literal with `_("...")` in:
  - `app.py` (titles, labels, buttons, status messages)
  - `preview.py` (canvas labels)
  - `cut_window.py` (labels, buttons)
  - `widgets/*.py` (new panels)
  - `dialogs` (error messages, confirmations)

#### 14.4 — RTL support
- [ ] 14.4.1 Detect locale; if `ar`, set layout direction via `tk.call('ttk::style', 'configure', ...)`
- [ ] 14.4.2 Mirror layout: replace `side="left"` with side based on locale. Use helper:
  ```python
  def side_for(widget):
      return "right" if locale == "ar" else "left"
  ```
- [ ] 14.4.3 Render Arabic text: connect to `tkextrafont` or use `Text` widget with bidi-aware tags
- [ ] 14.4.4 Test every screen with `ar` locale

#### 14.5 — Date / number / time formatting
- [ ] 14.5.1 Use `babel.dates.format_date`, `format_time`, `format_number`
- [ ] 14.5.2 Use `format_timedelta` for duration display
- [ ] 14.5.3 Spinbox `HH:MM:SS` parses via `strptime` per locale

#### 14.6 — Tests
- [ ] 14.6.1 `tests/test_i18n.py` — `t("greeting")` returns Arabic when locale="ar"
- [ ] 14.6.2 `tests/test_rtl.py` — instantiate every widget with `locale="ar"`, verify no exception
- [ ] 14.6.3 `tests/test_rtl_layout.py` — assert `side` of widget under Arabic is mirrored
- [ ] 14.6.4 Snapshot test: render same screen in EN + AR, compare image similarity > 0.85

### Acceptance
- [ ] Every user-visible string is in `messages.po`
- [ ] `Ctrl+Shift+L` switches between EN and AR live
- [ ] Arabic layout is mirrored (sections read right-to-left)
- [ ] All buttons/icons work in Arabic locale
- [ ] Numbers and dates formatted per locale

### Risk
- **HIGH** — Arabic is the first RTL language. Tk has limited native RTL. Shape glyphs, bidi text in mixed strings, and layout mirroring are well-known issues. **Strongly recommend Phase 14 take 1-2 days for testing + native Arabic speaker review.**

---

## Cross-Cutting Concerns

### Pre-Flight Checklist (every phase)
Per AGENTS.md:
- [ ] `pytest tests/` — same or higher pass count
- [ ] `python -m py_compile <each modified file>` — clean
- [ ] `python -m anime_upscaler_gui` — manual smoke test (10s)
- [ ] No unicode in print statements (Windows cp1252)
- [ ] Branch lives on `feature/ui-ux-overhaul` until Phase 12

### Test Strategy
- **Unit:** every widget in `tests/test_widgets.py`, every error path in `tests/test_<feature>.py`
- **Integration:** `tests/test_app.py` — instantiate `UpscaleGUI`, mock worker, verify event flow
- **Visual:** `tests/test_theme.py` — assert `tk.PhotoImage` of each screen renders without exception
- **Accessibility:** `tests/test_accessibility.py` — assert focus order, contrast ratios

### Risk Mitigation
- After every phase: `git commit -m "phase X: <scope>"` — single rollback point
- Keep `app.py` working at all times; new widgets can be additive
- Major refactors (Phase 1, 2) ship behind feature flag if possible

### Estimated Total Effort (single developer)
| Phase | Effort |
|---|---|
| 0 — Token sweep | 0.5 day |
| 1 — Data layer | 1 day |
| 2 — Widget extraction | 1.5 days |
| 3 — Tabbed layout | 0.5 day |
| 4 — Settings & persistence | 1 day |
| 5 — Theme system | 1.5 days |
| 6 — Threading fixes | 1 day |
| 7 — Interaction improvements | 2 days |
| 8 — Empty / error states | 1 day |
| 9 — Telemetry & monitoring | 1.5 days |
| 10 — Accessibility | 1 day |
| 11 — Visual polish | 2 days |
| 12 — Documentation | 0.5 day |
| 13 — Splash screen | 0.5 day |
| 14 — i18n (EN + AR) | 2 days |
| **Total** | **~17.5 days** |

---

## Open Questions

Awaiting user confirmation before Phase 1:

1. **Arabic translation source:** native speaker review, or generate machine-translated drafts for you to verify?
2. **Arabic font on Windows:** `"Segoe UI Arabic"` (bundled, may be missing on older Win 10) or bundle `"Noto Sans Arabic"` (cross-platform)?
3. **Persistence granularity:** "every change" — incremental JSON patch (RFC 6902) for cheap I/O, or full snapshot?
4. **Phase 5 default theme on first run:** light (current), or detect OS via `darkdetect`?
5. **Skip splash flag:** `--no-splash` only, or `--splash-duration=N` for testing?
6. **Translation memory:** EN + AR only (final scope), or scaffold for future languages?

---

## File Layout

```
apps/anime_upscaler_gui/
├── docs/
│   ├── audit/
│   │   └── ui_ux_audit.md           ← audit findings
│   ├── plans/
│   │   ├── ui_ux_overhaul.md        ← this document
│   │   └── ui_ux_overhaul_checklist.md  ← phase-by-phase todos
│   └── research/
│       ├── customtkinter_research.md  ← CustomTkinter fit assessment
│       └── reflex_research.md         ← Reflex fit assessment
├── anime_upscaler_gui/
│   ├── app.py                       (slimmed through phases)
│   ├── widgets/                     (created in Phase 2)
│   ├── controllers/                 (created in Phase 2)
│   ├── theme.py                     (created in Phase 5)
│   ├── state.py                     (created in Phase 1)
│   ├── i18n.py                      (created in Phase 14)
│   ├── splash.py                    (created in Phase 13)
│   ├── ui_constants.py              (Phase 0 expands, Phase 5 replaces)
│   └── ...
├── assets/
│   ├── icons/
│   │   └── phosphor/
│   │       ├── outline/             (Phase 7, 10, 11)
│   │       ├── solid/               (Phase 11)
│   │       └── duotone/             (Phase 11 — app icon)
│   └── fonts/
│       ├── NotoSansArabic-Regular.ttf  (Phase 14)
│       └── NotoSansArabic-Bold.ttf     (Phase 14)
├── i18n/
│   ├── en/LC_MESSAGES/messages.po   (Phase 14)
│   └── ar/LC_MESSAGES/messages.po   (Phase 14)
└── tests/
    └── ... (existing 18 + new per phase)
```
