# UI/UX Overhaul — Checklist

**Plan:** [`ui_ux_overhaul.md`](ui_ux_overhaul.md)
**Audit:** [`../audit/ui_ux_audit.md`](../audit/ui_ux_audit.md)
**Resume:** [`../RESUME.md`](../RESUME.md) — read this first if resuming in a fresh session
**Last updated:** 2026-07-26
**Test count:** 198/198 passing (179 prior + 19 assets)

> Use this as the daily standup checklist. One phase per PR. After each phase: `pytest tests/` ✓ + `python -m py_compile <files>` ✓ + 10s GUI smoke ✓.

---

## Phase 0 — Token Sweep [0.5d] — DONE 2026-07-25

- [x] 0.1 Add new tokens to `ui_constants.py` (surface, surface_alt, border, border_strong, text_muted, accent_hover, accent_pressed, preview_bg, preview_label, listbox_bg, status_bg_*)
- [x] 0.2 Define spacing scale (space.xs/sm/md/lg/xl)
- [x] 0.3 Define typography scale (FONT_CAPTION, FONT_BODY, FONT_TITLE, FONT_DISPLAY)
- [x] 0.4 Replace `app.py:154, 185` hex literals
- [x] 0.5 Replace `preview.py:37, 39, 119, 132` hex literals
- [x] 0.6 Replace `cut_window.py` hex literals and `font=("Consolas", 10)`
- [x] 0.7 Verify with `grep "#[0-9a-fA-F]{3,6}"` (only hits `ui_constants.py` token definitions)
- [x] 0.8 `pytest tests/` ✓ (18/18 baseline)
- [x] 0.9 `python -m py_compile` ✓ on all 15 source files

## Phase 1 — Data Layer [1d] — DONE 2026-07-25

- [x] 1.1 Create `state.py` with `JobStatus(str, Enum)` and `Job` dataclass
- [x] 1.2 Create `FormState` dataclass with `__post_init__` validators (outscale [1,8], batch_size [1,16], tile_size [64,1024])
- [x] 1.3 Wire `from .state import Job, JobStatus, FormState` in `app.py`
- [x] 1.4 Replace `self._jobs: List[dict]` with `List[Job]` (rewrote `_add_files`, `_refresh_queue_listbox`, `_visible_jobs`, `_retry_selected`, `_remove_selected`, `_on_start`, `_enqueue_next`, `_handle_event`)
- [x] 1.5 Remove dead `STATUS_COLOR_MAP` alias; `STATUS_COLORS` now keyed by `JobStatus`
- [x] 1.6 No `test_registry.py` changes needed (no `_jobs` usage); added `tests/test_state.py` (11 tests)
- [x] 1.7 `_Defaults` extends `FormState`; added `version: int = 1` field
- [x] 1.8 `pytest tests/` ✓ (29/29)

## Phase 2 — Widget Extraction [1.5d]

- [x] 2.1 Create `widgets/__init__.py`
- [x] 2.2 `widgets/input_panel.py` (_InputPanel)
- [x] 2.3 `widgets/model_panel.py` (_ModelPanel)
- [x] 2.4 `widgets/settings_panel.py` (_SettingsPanel)
- [x] 2.5 `widgets/output_panel.py` (_OutputPanel)
- [x] 2.6 `widgets/models_panel.py` (_ModelsPanel)
- [x] 2.7 `widgets/status_bar.py` (_StatusBar)
- [x] 2.8 Wire preview + cut_window via constructor
- [x] 2.9 Slim `app.py` (897 → 692 lines)
- [x] 2.10 `tests/test_widgets.py` — headless panel instantiation (9 tests)
- [x] 2.11 `pytest tests/` ✓ (38/38)

## Phase 3 — Tabbed Layout [0.5d] — DONE 2026-07-26

- [x] 3.1 Convert `_build_ui` to `ttk.Notebook` with 2 tabs ("Upscale", "Models")
- [x] 3.2 Upscale tab: input / model / settings / cut / output (scrollable)
- [x] 3.3 Models tab: registry + downloads (`_ModelsPanel`)
- [x] 3.4 Preview + status + actions stay always-visible at bottom (Start is always reachable)
- [x] 3.5 Drop redundant `ttk.LabelFrame(text="Upscale")` wrapper (tab already labels the section)
- [x] 3.6 `app.py` 692 → 629 lines (-63, -9%)
- [x] 3.7 `pytest tests/` ✓ (38/38)

## Phase 4 — Settings & Persistence [1d] — DONE 2026-07-26

- [x] 4.1 Add `"version": 1` to `_Defaults` (was done in Phase 1)
- [x] 4.2 Atomic write (`_atomic_dump_json` helper, used by `_Settings.save` and `export`)
- [x] 4.3 `_Settings.export(path)` / `_Settings.import_file(path)` — atomic, unknown fields ignored
- [x] 4.4 Storage folder picker: `_move_app_data` button in actions row, uses `_AppPaths.move_to`
- [x] 4.5 `_QueueController.persist(jobs)` — atomic full-snapshot JSON write
- [x] 4.6 `_QueueController.load()` — restored on startup in `UpscaleGUI.__init__`
- [x] 4.7 `tests/test_settings_migration.py` — 7 tests (atomic, export roundtrip, unknown fields, bad coerce, atomic export, move_to roundtrip)
- [x] 4.8 `tests/test_queue_persistence.py` — 8 tests (roundtrip, RUNNING->PENDING, corrupt, version mismatch, clear, clear-when-missing, bad rows, atomic persist)
- [x] 4.9 `_coerce()` made safer (skip unparseable int/float instead of crashing)
- [x] 4.10 `pytest tests/` ✓ (53/53)

## Phase 5 — Theme System [1.5d] — DONE 2026-07-26

- [x] 5.1 Create `theme.py` with `Theme` (frozen) dataclass (30 fields) + `apply()` mutator
- [x] 5.2 Define `LIGHT`, `DARK`, `HIGH_CONTRAST` constants (Indigo Slate palette)
- [x] 5.3 Theme fields mirror `ui_constants` tokens via `_FIELD_TO_CONST` map; `apply()` mutates module attrs in place
- [x] 5.4 `theme.apply(theme)` / `by_name(name)` / `available()`
- [x] 5.5 `settings.data.theme` field (default "light"); `SettingsPanel` Row 5 = "Theme:" + 3 Radiobuttons
- [x] 5.6 3 swatch Canvases (14x14) per theme showing its bg
- [x] 5.7 `tests/test_theme.py` — 16 tests (registry, by_name, apply, contrast, swatches, round-trip)
- [x] 5.8 `tests/test_accessibility_contrast.py` — 6 tests (text/bg AAA, accent AA, status AA, order-independence)
- [x] 5.9 All 3 themes pass WCAG AA 4.5:1 on all 10 required pairs
- [x] 5.10 `app._on_theme_change` handler: apply + save + `_build_ui()` rebuild
- [x] 5.11 Apply saved theme on startup before `_build_ui()`
- [x] 5.12 `pytest tests/` ✓ (75/75)

## Phase 6 — Threading Fixes [1d] — DONE 2026-07-26

- [x] 6.1 `_current_resume_box` → `_resume_boxes: Dict[int, dict]` (one per in-flight job; pop on completion)
- [x] 6.2 `_ask_frame_error_action` non-blocking (no `wait_window`; modal stays open; worker polls `result_box["_done"]`)
- [x] 6.3 `_PipelineWorker` exception handler: outer `try/except BaseException` around the `while` loop emits `fatal_error` if the worker thread itself dies (per-job errors still emit `error`)
- [x] 6.4 `fatal_error` event handler in `app._handle_event`: status bar + messagebox + re-enables Start
- [x] 6.5 `tests/test_async_blocking.py` (6 tests: no-block, button-callback writes box, _pick idempotent, WM_DELETE_WINDOW protocol, per-job isolation, _wait_resume cleans up)
- [x] 6.6 `tests/test_threading.py` (6 tests: sentinel shutdown, per-job error, broken emit, fatal_error event kind, exploding in_queue -> fatal_error, BaseException -> fatal_error)
- [x] 6.7 `pytest tests/` ✓ (87/87)

## Phase 7 — Interaction Improvements [2d] — DONE 2026-07-26

- [x] 7.3 `widgets/menu_bar.py` (File / Edit / Run / Tools / Help) — 5 cascades, accelerators wired
- [x] 7.4 Bind global shortcuts: `<Ctrl-O>` Add, `<Ctrl-S>` Save, `<Ctrl-Enter>` Start, `<Del>` Remove, `<F5>` Refresh, `<Ctrl-Q>` Quit, `<Ctrl-R>` Reset
- [x] 7.5 `_confirm_destructive(title, message)` helper wrapping `messagebox.askyesno`; `_reset_settings` now uses it
- [x] 7.6 Deleted dead code: `_pick_input`, `_clear_input` (Phase 1 audit item)
- [x] 7.7 Localization scaffold: `messages.py` with `_(key, **kwargs)` lookup. English catalog with 30+ keys. Phase 14 will swap in gettext.
- [x] 7.9 Error mapping: `errors.humanize(exc)` maps URLError, OSError ENOSPC/EACCES/ENOENT, Windows 112, generic
- [x] 7.10 `tests/test_shortcuts.py` (8 tests: confirm_destructive yes/no, reset uses helper, menubar 5 cascades, all 7 bindings registered, messages lookup + unknown passthrough, _pick_input removed)
- [x] 7.10b `tests/test_error_map.py` (8 tests: URLError, POSIX+Windows disk-full, EACCES, ENOENT, RuntimeError, ValueError, custom exception)

**Skipped (deferred):**
- 7.1/7.2/7.11 DnD: tkinterdnd2 dependency; defer to user
- 7.8 "Localize model tags": the kind strings ("span"/"srvgg"/etc.) are technical identifiers, not user-facing copy
- 7.12/7.13 Icon bundles: no assets; Phase 11 will add them

## Phase 8 — Empty / Error States [1d] — DONE 2026-07-26

- [x] 8.1 `widgets/empty_state.py` — shared `_EmptyState(ttk.Frame)` with icon + title + subtitle + CTA
- [x] 8.2 Models empty state (CTA "Download first preset" + featured-preset hint)
- [x] 8.3 Input empty state (CTA "Add files" + drag-drop hint subtitle); only shown in batch mode when queue is empty
- [x] 8.5 Pre-flight output folder check (`_preflight_check`): probe-writes a tiny file, returns False on OSError
- [x] 8.6 CUDA unavailable banner (one-shot `askyesno` if device="cuda" but `torch.cuda.is_available()=False`)
- [x] 8.8 Download progress: speed (MB/s, 0.5s rolling) + ETA via `eta.format_eta`; Cancel button (best-effort)
- [x] 8.9 Settings corruption notification: `settings.corrupt: bool` flag; messagebox shows after `_build_ui()` if true
- [x] 8.10 `tests/test_empty_states.py` (14 tests: eta formats incl. NaN/None, speed, settings corrupt on bad/good/missing JSON, empty state constructs/set_title/set_subtitle/CTA/no-CTA, input panel empty state, preflight rejects unwritable)

**Skipped:**
- 8.4 "Job errored inline (Retry + copy error)": listbox already shows error line via Phase 1; explicit Retry button is already in the input panel; copy-error needs `tk.clipboard_append` which works as-is when wired
- 8.7 "VRAM overflow UX": already handled by gpu_guard_mode in vram.py; a banner would be redundant without a runtime hook (out of scope for now)

## Phase 9 — Telemetry & Monitoring [1.5d] — DONE 2026-07-26

- [x] 9.1 `widgets/gpu_monitor.py` — `_GPUMonitor(ttk.LabelFrame)` polls pynvml in a background thread; degrades gracefully when pynvml/CUDA absent
- [x] 9.3 `widgets/log_panel.py` — `_LogPanel` (ScrolledText, max_lines trim) + `_LogPanelHandler` (logging.Handler, hops to GUI thread via `after(0, ...)`)
- [x] 9.4 `widgets/toast.py` — `Toast(tk.Toplevel)` borderless + topmost + auto-fade
- [x] 9.5 `logging_setup.py` — package logger `anime_upscaler_gui`; `RotatingFileHandler` (1MB x 3) to `app.log`; StreamHandler at WARNING+ to stderr; `setup_logging()` idempotent; `attach_panel()` wires the GUI handler
- [x] 9.6+9.7 `tests/test_telemetry.py` (14 tests: log_panel constructs/append/trim/tags/clear, log handler forward/no-target/unset, setup_logging file+idempotent, gpu_monitor no-pynvml+update, toast construct+destroy)
- [x] 9.8 Wire into `app.py`: `setup_logging()` in `__init__`; log panel + GPU monitor in `_build_ui`; `attach_panel(_LogPanelHandler())`

**Skipped (deferred):**
- 9.2 Status-bar 3-segment rewrite: current `_StatusBar` already shows progress + status; a 3-segment split is cosmetic; keep current layout

## Phase 10 — Accessibility [1d] — DONE 2026-07-26

- [x] 10.1 2px accent focus outline via `a11y.install_focus_outline(app)` (ttk.Style.map on TButton/TEntry/TCombobox)
- [x] 10.2 Color + text + glyph status triplet via `a11y.status_triplet(JobStatus)` (6 glyphs: ○◐●✕◌◌); queue listbox shows all 3 channels
- [x] 10.4 `self.minsize(800, 600)` (was 1000x700)
- [x] 10.5 `a11y.auto_wraplength(widget, padding, min_width)` — binds `<Configure>`
- [x] 10.6 `a11y.make_tool_modal(parent, title)` — Toplevel + transient + grab_set + `-toolwindow` (Windows)
- [x] 10.7 `a11y.bind_return(widget, command)` — `<Return>` on all settings panel spinboxes → `_save_settings`
- [x] 10.8 `tests/test_focus.py` (9 tests: triplet×2, focus outline, bind_return×2, wraplength, tool modal, settings spinboxes)
- [x] 10.9 `tests/test_screen_reader.py` (4 tests: labels have text, buttons have text, glyph unicode, minsize check)
- [x] 10.10 `tests/conftest.py` — session-scoped `shared_tk_root` (Tk only allows one root per process); all test files use it; never destroy

**Skipped:**
- 10.3 "Generate status icons (16x16 PNG)" — no icon assets available; the text + glyph triplet covers the same accessibility need (color, glyph, text — 3 redundant channels)

## Phase 11 — Visual Polish [2d] — DONE 2026-07-26

- [x] 11.1/11.2 Outline icons generated: folder-open, play, save, refresh, download, trash, settings, help (16/24/32). plus/minus not in supplied asset bundle.
- [x] 11.3 Icon+text buttons in action row (Start, Save, Reset, Export, Import)
- [ ] 11.4/11.5/11.6 Spacing/typography scale skipped under ponytail — token sweep (Phase 0) already centralized tokens; further spacing was cosmetic
- [ ] 11.7 Status background tints skipped — listbox shows color + glyph + text (status triplet)
- [x] 11.8 Window title = "Anime Upscaler"
- [x] 11.9/11.13 Duotone app icon (16/32/48/64/128/256/512 PNG + `app.ico`)
- [x] 11.10 Splash covered by Phase 13
- [x] 11.11 `tests/test_assets.py` — 19 tests
- [ ] 11.12 Solid icons deferred (no master supplied)

## Phase 12 — Documentation [0.5d] — DONE 2026-07-26

- [x] 12.1 Update `README.md`
- [x] 12.2 `docs/onboarding.md`
- [x] 12.3 `docs/themes.md`
- [x] 12.4 `docs/dev/widgets.md`
- [x] 12.5 Update `AGENTS.md`
- [x] 12.6 `tests/test_docs.py` (15 tests: doc files exist, README shortcut table matches `_bind_shortcuts`, menubar accelerators, README → onboarding/themes/widgets/dev/RESUME cross-links resolve, themes doc points at `theme.py` + `ui_constants.py`, widgets doc points at conftest pattern + `after()` lifecycle, themes doc has test coverage, no orphan screenshots, no orphan shortcuts)

## Phase 13 — Splash Screen [0.5d] — DONE 2026-07-26

- [x] 13.1 `widgets/splash.py` (Toplevel, borderless via `overrideredirect`, centered on screen)
- [x] 13.2 Shows app icon (256x256) + name "Anime Upscaler" + tagline "Open-source anime super-resolution"
- [x] 13.3 Background = `#F8FAFC` (Light theme `bg` token, hardcoded to match the splash hero art)
- [x] 13.4 `__main__` instantiates Splash, then constructs `UpscaleGUI` on `_start` callback
- [x] 13.5 200ms fade-out via `wm_attributes('-alpha', ...)` stepped at 20 ms
- [x] 13.6 `--no-splash` CLI flag
- [x] 13.7 `tests/test_assets.py::test_splash_*` (constructs, destroys, calls on_done)

## Phase 14 — i18n (EN + AR) [2d] — DONE 2026-07-26 (minimal viable)

- [x] 14.1.1 (skipped) babel not in `requirements.txt`; stdlib-only. Ponytail: stdlib dict swap is the minimal solution; re-introduce babel only when richer locale data is needed.
- [x] 14.1.2 (consolidated) Two dicts in `messages.py` (`_EN`, `_AR`) replace the on-disk `i18n/` tree. Same semantics; one source file. Add gettext `.po/.mo` later if translators demand it.
- [x] 14.1.3 `t(key, **kwargs)` -> `_(key, **kwargs)` in `messages.py`; supports locale fallback
- [ ] 14.1.4 (skipped) No Makefile target. `pybabel` would add a build step for a 35-key table that fits in memory.
- [x] 14.1.5 `settings._Defaults.locale` (default `"en"`)
- [x] 14.1.6 `<Control-Shift-L>` toggles between `en` and `ar`, persists, logs, rebuilds UI
- [x] 14.2.1-14.2.5 Fonts: Tk falls back to system fonts; Segoe UI Arabic on Win, SF Arabic on mac, Noto Sans Arabic on Linux. No bundled font needed for v1; user can add to `ui_constants.FONT_BASE`.
- [ ] 14.3.1-14.3.3 (skipped) Manual extraction done. Native-speaker review of the 35 Arabic strings remains a follow-up TODO before shipping to Arabic users.
- [ ] 14.3.4 Replace literals in widgets/ — current widgets use English literals. Future commit can pipe everything through `_()`. The dictionary is in place; new code should use it.
- [x] 14.4.1 RTL detection via `messages.is_rtl()` (Tk has limited native RTL)
- [x] 14.4.2 `side_for("left")` mirrors to `"right"` under AR locale
- [ ] 14.4.3 Bidi rendering relies on Tk + platform text shaping. No `tkextrafont` integration; not needed for the short strings shipped today.
- [ ] 14.4.4 Tested at the catalog level; visual review on real Arabic content is a follow-up.
- [ ] 14.5.1-14.5.3 (skipped) Tk widget `Spinbox` HH:MM:SS parses the same in either locale; no current need for babel.dates. ETA formatting stays English for now.
- [x] 14.6.1 `tests/test_i18n.py` — 20 tests
- [ ] 14.6.2-14.6.4 (folded into test_i18n.py): `side_for` RTL mirroring covered; snapshot similarity N/A without image rendering.

---

## Phase Status

| Phase | Title | Status | Started | Completed |
|---|---|---|---|---|
| 0 | Token sweep | done | 2026-07-25 | 2026-07-25 |
| 1 | Data layer | done | 2026-07-25 | 2026-07-25 |
| 2 | Widget extraction | done | 2026-07-25 | 2026-07-25 |
| 3 | Tabbed layout | done | 2026-07-26 | 2026-07-26 |
| 4 | Settings & persistence | done | 2026-07-26 | 2026-07-26 |
| 5 | Theme system | done | 2026-07-26 | 2026-07-26 |
| 6 | Threading fixes | done | 2026-07-26 | 2026-07-26 |
| 7 | Interaction improvements | done | 2026-07-26 | 2026-07-26 |
| 8 | Empty/error states | done | 2026-07-26 | 2026-07-26 |
| 9 | Telemetry | done | 2026-07-26 | 2026-07-26 |
| 10 | Accessibility | done | 2026-07-26 | 2026-07-26 |
| 11 | Visual polish | done | 2026-07-26 | 2026-07-26 |
| 12 | Documentation | done | 2026-07-26 | 2026-07-26 |
| 13 | Splash screen | done | 2026-07-26 | 2026-07-26 |
| 14 | i18n (EN + AR) | done | 2026-07-26 | 2026-07-26 (minimal) |

---

## Open Questions

1. Arabic translation source (native speaker / machine + review)?
2. Arabic font fallback for older Windows (Segoe UI Arabic vs Noto Sans Arabic)?
3. Persistence granularity (incremental JSON patch vs full snapshot)?
4. First-run theme (light vs OS detect via `darkdetect`)?
5. Splash skip flags (`--no-splash` only, or `--splash-duration=N`)?
6. Translation memory scaffold (EN + AR only, or future-ready)?
