# UI/UX Audit — `anime_upscaler_gui`

**Date:** 2026-07-25
**Scope:** `apps/anime_upscaler_gui/` (Tkinter desktop app, ~1200 LOC)
**Method:** Read-only inspection of `app.py` (909 LOC), `ui_constants.py`, `settings.py`, `registry.py`, `preview.py`, `cut_window.py`, `__main__.py`, `README.md`

Severity: **[H]** high impact · **[M]** medium · **[L]** polish. Effort: S = <2h · M = half-day · L = 1-2d.

---

## 1. Layout / Information Architecture [H]

| # | Issue | Fix |
|---|---|---|
| 1.1 | Everything scrolls in one column — no tabs despite comment "later we wrap in ttk.Notebook" (app.py:142) | Wrap in `ttk.Notebook` with **Upscale**, **Batch Queue**, **Models**, **Settings**, **Log** |
| 1.2 | "Models" section bolted to bottom (app.py:144) — feels like an afterthought | Move to its own tab with header + list + actions split |
| 1.3 | Settings panel packs 4 rows of dense controls (device/precision/outscale/batch/decode/prefetch/downscale/gpu_guard/tile) — overload | Split into **Inference** (device, fp16, outscale, batch), **I/O** (decode, prefetch, downscale), **VRAM** (guard mode, tile, overlap) |
| 1.4 | Video cut window sits between Settings and Output — breaks the model+output flow | Move cut window to a sub-section of the Input tab when input is video, or its own **Video** tab |
| 1.5 | Output "Same folder" mode uses `batch_folder_name` from a separate row — implicit coupling | Show the suffix/folder fields inline next to the radio button they affect |
| 1.6 | Single-mode vs batch-mode toggles via `pack_forget` (app.py:316-320) — flickers and is fragile | Use `ttk.Notebook` with two tabs: Single / Batch |
| 1.7 | Preview height fixed at 220px (app.py:113) — actions always visible but preview unusable for 4K outputs | Make preview resizable (drag handle) or auto-size to a percentage of the window |

---

## 2. Interaction / UX [H]

| # | Issue | Fix |
|---|---|---|
| 2.1 | **No drag-and-drop** for files onto the input listbox (README:135) | Add `tkinterdnd2` |
| 2.2 | **No keyboard shortcuts** | Add: `Ctrl+O`=add files, `Ctrl+S`=save settings, `Ctrl+Enter`=start, `Delete`=remove selected, `F5`=re-scan models, `Ctrl+Shift+L`=toggle language |
| 2.3 | `_reset_settings` calls `_build_ui()` (app.py:816) — silently destroys the entire job queue and any in-progress work | Confirm with a modal listing what will be lost; only reset the form, not the entire widget tree |
| 2.4 | `_ask_frame_error_action` uses `wait_window()` on the GUI thread (app.py:752) — **UI freezes hard** on frame errors | Move to a non-blocking `Toplevel`; use `result_box` polling only (already wired) — but **don't block mainloop** |
| 2.5 | **No undo/redo** for batch queue | Persist queue to `cache_dir/queue.json` on every change; restore on launch |
| 2.6 | "Tainted checkpoint" warning shows `bias_mean ~0.43` (app.py:640) — only devs understand | Localize: "This model is known to produce flat-blue output. Continue?" |
| 2.7 | "unsupported" / "no arch" tags in model dropdown (app.py:602-604) — cryptic | Use icons + explanations + a "Why?" link to README; disable the item instead of just tagging |
| 2.8 | No confirmation on **Clear** / **Remove selected** (app.py:562, 579) — one click nukes work | Add `messagebox.askyesno` for both |
| 2.9 | No ETA / time-remaining estimate during runs | Track job start time, emit it in `_JobEvent.progress`; compute `frames_left / fps` |
| 2.10 | No GPU monitor (utilization, VRAM, temp) | Add a small status-bar widget polling `torch.cuda` every 1-2s |
| 2.11 | Status strip is one-line text — no log panel | Add a tab with a scrolling `ScrolledText` for full history; status strip keeps only the latest |
| 2.12 | Add-files dialog only filters by listed extensions (app.py:475) | Add a "Recursive folder" toggle that walks + filters |
| 2.13 | `_pick_input` is dead code (app.py:455) — superseded by `_add_files` | Delete it |
| 2.14 | `_refresh_queue_listbox` clears+rebuilds the listbox every 100ms (app.py:537) — flickers, slow on big batches | Refactor to in-place row updates; only rebuild when membership changes |
| 2.15 | No **first-run wizard** (README says "first-run wizard scaffold" — it doesn't exist) | On first launch: detect CUDA, point at `pretrained/`, offer to download a default model |
| 2.16 | No "About" / "Help" dialog | Add a menu bar with **File / Edit / Tools / Help** and standard menu items |
| 2.17 | No **app icon** | Generate a 32x32 PNG, set via `iconphoto` |

---

## 3. Visual Design / Restyle [H]

### 3.1 Tokens (cheapest, biggest win)

The codebase already has `ui_constants.py` but violations are pervasive:

| File | Line | Violation |
|---|---|---|
| `app.py` | 154 | `foreground="#555"` — should use `DISABLED` |
| `app.py` | 185 | `foreground="#555"` — should use `DISABLED` |
| `preview.py` | 37, 39 | `bg="#222"` — should be a `PREVIEW_BG` token |
| `preview.py` | 119 | `fill="#888"` — should use `DISABLED` |
| `preview.py` | 132 | `fill="#fff"` — should use a `PREVIEW_LABEL` token |
| `cut_window.py` | 52, 56, 59 | `font=("Consolas", 10)` — should use `FONT_MONO` |
| `cut_window.py` | 53 | `foreground="#444"` — should use `DISABLED` |

**Fix:** Add tokens to `ui_constants.py`, replace all literals. Est: 30 minutes.

### 3.2 Theme system

| # | Issue | Fix |
|---|---|---|
| 3.2.1 | **Light theme only** — README:139 mentions "swapping ui_constants.py colors" as the future dark-mode path | Refactor `ui_constants.py` into a `Theme` dataclass with light (default) + dark; add a `Settings → Theme` radio |
| 3.2.2 | Only 6 colors defined; no semantic surface/border hierarchy | Add `surface`, `surface_alt`, `border`, `border_strong`, `text_muted`, `accent_pressed`, `accent_hover` |
| 3.2.3 | No accent_hover / accent_pressed states for buttons | Define disabled/hover/pressed for each control group (mouseover for ttk requires `style.map`) |
| 3.2.4 | Flat Tk look — no visual hierarchy beyond LabelFrame borders | Use `ttk.Style` with a custom theme; add subtle elevation (background tint) for active panels |
| 3.2.5 | Padding is bare magic numbers (PAD_X=6, PAD_Y=4) | Adopt a 4-pt scale: 4/8/12/16/24 — rename to `space.xs/sm/md/lg/xl` |
| 3.2.6 | Title shows lowercase "anime upscaler gui" (app.py:34) | Use "Anime Upscaler" — match the class name casing |
| 3.2.7 | All fonts are 10pt with one 11pt bold heading — under-differentiated | Add: caption (9pt), body (10pt), title (14pt bold), display (20pt) |
| 3.2.8 | Status colors are text-only — no background fills | Add background tints: pending=light gray, running=blue tint, done=green tint, error=red tint, with **both color + icon** for color-blind users |
| 3.2.9 | Use of `ttk.Separator` between inline radios (app.py:343, 360) — visually heavy | Replace with whitespace + label alignment |
| 3.2.10 | Models listbox height=5 (app.py:156) — too tall for 1-2 models, too short for 10+ | Dynamic: `min(8, len(models))` |
| 3.2.11 | Window title bar has no app branding | Set `wm_iconphoto` + frame title |

### 3.3 Iconography

| # | Issue | Fix |
|---|---|---|
| 3.3.1 | No icons in buttons (text-only) | Add 16x16 PNG icons: folder-open, play, save, refresh, download, trash, plus, minus |
| 3.3.2 | No icons in status (only text + color) | Add symbols: `✓` done, `✗` error, `⟳` running, `⏸` paused — but unicode above U+00FF breaks on Windows (AGENTS.md); use PIL-drawn icons |
| 3.3.3 | No icon for the app itself | Bundle a 256x256 PNG scaled to 16/32/48 |

---

## 4. Accessibility [M]

| # | Issue | Fix |
|---|---|---|
| 4.1 | Color-only status indication | Pair every status color with an icon + text label |
| 4.2 | No visible focus indicator | Configure `ttk.Style` to show a 2px accent outline on focused widgets |
| 4.3 | Minimum window size 1000×700 (app.py:36) — too large for 1366×768 displays | Reduce to 800×600; let preview adapt |
| 4.4 | `wraplength=900` (app.py:438) — hard-coded for the current window | Tie to `winfo_width()` on resize |
| 4.5 | Modals have no role descriptions | Add `modal.wm_attributes("-toolwindow", True)` so they don't appear in taskbar |
| 4.6 | Spinbox keyboard stepping not tested | Add `bind("<Return>", ...)` to commit values |
| 4.7 | No high-contrast theme | Add a third "High contrast" theme option |

---

## 5. Refactor (Code-level) [H]

### 5.1 Split `app.py` (909 lines, monolithic)

```
app.py                  →  app.py          (main shell + lifecycle only, ~150 lines)
widgets/
  input_panel.py        →  _InputPanel(ttk.Frame)
  model_panel.py        →  _ModelPanel(ttk.Frame)
  settings_panel.py     →  _SettingsPanel(ttk.Frame)
  output_panel.py       →  _OutputPanel(ttk.Frame)
  preview_panel.py      →  _PreviewPane (already exists)
  cut_window.py         →  (already exists)
  models_panel.py       →  _ModelsPanel(ttk.Frame)
  status_bar.py         →  _StatusBar (toolbar + log)
theme.py                →  ui_constants.py → theme.py (Theme dataclass + apply)
controllers/
  job_controller.py     →  _JobController (queue + worker + events)
  settings_controller.py → _SettingsController
```

**Why:** every panel will be testable in isolation; themes apply per top-level; new panels (e.g., "Compare" for batch before/after) drop in cleanly.

### 5.2 Threading model

| # | Issue | Fix |
|---|---|---|
| 5.2.1 | `_current_resume_box` is a single shared dict (app.py:53, 734, 759) — race between concurrent jobs | Map `job_id → result_box` |
| 5.2.2 | Worker dies silently if `_PipelineWorker` raises — no UI feedback | Catch worker exceptions in `_poll_events` and surface a fatal-error modal |
| 5.2.3 | `_poll_events` runs every 100ms regardless of activity | Drive off a `_notify` from the worker (event-based, not polled) |

### 5.3 State management

| # | Issue | Fix |
|---|---|---|
| 5.3.1 | All jobs state lives in `self._jobs: List[dict]` — no type safety, stringly-typed status | Replace with `@dataclass class Job` + enum `JobStatus(Enum)` |
| 5.3.2 | `tk.StringVar` / `BooleanVar` / `IntVar` used directly in business logic | Wrap in a `FormState` dataclass with `bind(var, field)` for one-way → dataclass sync |
| 5.3.3 | `_Defaults` (settings.py:29) is a flat dataclass; no validation | Add `__post_init__` validators for `outscale ∈ [1, 8]`, `batch_size ∈ [1, 16]`, etc. |

### 5.4 Error handling

| # | Issue | Fix |
|---|---|---|
| 5.4.1 | 18+ bare `except Exception:` blocks (see AGENTS.md) | Add `logger.warning(f"...")` everywhere; reserve bare `except` for fallback UI |
| 5.4.2 | Downloads show `FAIL: {exc}` (app.py:248) — raw exception text | Map known errors to user copy: "Network unreachable", "Disk full", "Invalid checkpoint" |
| 5.4.3 | No global crash handler | Wrap `mainloop` in `try/except`; on crash, write `crash.log` and show a "Report" dialog |

### 5.5 Persistence

| # | Issue | Fix |
|---|---|---|
| 5.5.1 | `_AppPaths.move_to` (settings.py:98) is never called from the UI | Add a "Settings → Storage folder" picker that calls it |
| 5.5.2 | Queue is lost on close | Persist on every change; restore on launch |
| 5.5.3 | No schema versioning for `settings.json` | Add `"version": 1`; migrate on load |

---

## 6. Empty / Error States [M]

| # | State | Current | Better |
|---|---|---|---|
| 6.1 | No models installed | Just "Refresh" button | Illustration + "Get started by downloading a model" + featured preset CTA |
| 6.2 | No input selected | Empty listbox | "+ Add files" CTA + drag-drop hint |
| 6.3 | Job errored | Red text in listbox | Inline error row + "Retry" button next to it |
| 6.4 | Output folder missing | Crashes at `Path.mkdir` (app.py:683) | Pre-flight check on Start; show banner "Output folder will be created at X" |
| 6.5 | CUDA not available | Silent — runs on CPU | Detect at startup; show "GPU not detected. Running on CPU (slower)." |
| 6.6 | VRAM overflow | Modal "auto-downscale / downscale-before / cancel" | Pre-flight estimate in plain English: "This image needs ~6 GB; you have 4 GB. Tile it?" |
| 6.7 | Download in progress | Plain progress bar | Show speed (MB/s), ETA, and a Cancel button |
| 6.8 | Settings corrupt | Backup to `.bak` (settings.py:127) — silent | Notify: "Settings file was corrupt; reset to defaults. Old file: .bak" |

---

## 7. Telemetry / Feedback [M]

| # | Issue | Fix |
|---|---|---|
| 7.1 | No way to report a bug | Add **Help → Report issue** that copies system info (GPU, Python, app version) to clipboard |
| 7.2 | No way to share a config | Add `Export settings...` / `Import settings...` buttons |
| 7.3 | No metric on success | After each job, show "Saved X.png in 12.3s (45 fps) — 1080×1920 → 4320×7680" in a completion toast |
| 7.4 | No usage stats | Optional opt-in "Send anonymous usage stats" toggle |

---

## Top 10 Priority Queue (recommended order)

1. **[H, S]** Token sweep — fix 6-7 hardcoded hex/font literals listed in 3.1. 30 min, no risk.
2. **[H, M]** Wrap in `ttk.Notebook` — 5 tabs (Upscale, Batch, Models, Settings, Log). Half day. Big UX win.
3. **[H, S]** Drag-and-drop with `tkinterdnd2` — 1-2h. Removes the biggest friction.
4. **[H, S]** Keyboard shortcuts (Ctrl+O, Ctrl+Enter, etc.) — 1h.
5. **[H, M]** Theme system: `Theme` dataclass + dark mode toggle. Half day. Unblocks all visual work.
6. **[H, M]** Replace `_build_ui()` reset with a clean `reset_form()` and a confirm dialog. 2h. Prevents silent data loss.
7. **[H, M]** Split `app.py` into `widgets/` + `controllers/`. 1 day. Foundations for everything else.
8. **[M, S]** Replace `List[dict]` with `Job` dataclass + `JobStatus` enum. 1h. Type safety.
9. **[M, S]** Drag-resizable preview pane. 2h.
10. **[M, S]** Empty states with CTAs (6.1, 6.2). 2h. Big first-run impression.

---

## Refusals

- **Custom-drawn anything in Tk.** If you want a modern look, **switch to PyQt/PySide/Reflex/webview**. Tk is fine for an MVP but its styling is a dead end.
- **A "design system"** in `ui_constants.py`. Tokens are correct; a full design system at this scope is over-engineering.
- **Ponytail:** none of the above is gold-plating; every item has a concrete user-visible or developer-experience payoff.
