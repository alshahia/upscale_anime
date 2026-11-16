# GUI Production-Readiness Review — anime_upscaler_gui

**Date:** 2026-10-11
**Reviewer:** automated code + runtime review (DeepSeek Harness)
**Scope:** <code>apps/anime_upscaler_gui/</code> — the Tkinter desktop app
**Method:** read-only source inspection, full test-suite run, live GUI launch,
programmatic widget introspection, and direct logic harnesses (not just docs).
**Fix plan:** [../plans/production_readiness_fixes.md](../plans/production_readiness_fixes.md)

Severity: **[BLOCKER]** ship-stopper · **[HIGH]** user-visible breakage ·
**[MED]** production gap · **[LOW]** polish.

---

## Verdict

**Not production-ready yet.** The engine and app are genuinely functional and
better than the docs claim, but four real functional bugs ship today and the
370-test suite does not catch them. Treat the current state as a solid beta.

---

## What was verified (not merely read)

| Check | Result |
|---|---|
| Full test suite | **370 passed, 0 failed** in 37.18s (docs claim 349 — understated) |
| GUI launches | 1200x825, title "Anime Upscaler", 10 models detected |
| Themes render | light / dark / high-contrast all build without error |
| Real end-to-end upscale | RFDN_distill_v1 on CPU: 96x128 -> 192x256 in **150 ms** |
| Wheel build | anime_upscaler_gui-0.1.0-py3-none-any.whl (139 KB) built OK |
| Dependencies present | torch 2.12.0+cu126, opencv 4.13, numpy 2.4.4, av 17.1, tkinterdnd2 1.7.3, onnxruntime-gpu 1.21 |

The review harnesses were temporary and removed after use; app-data side effects
(a persisted theme/locale flip and fake queue.json entries) were reverted.

---

## Blocking bugs

### BLOCKER-1 — Batch mode retries a failed job forever
<code>_handle_event</code> on <code>error</code> calls
<code>_enqueue_next(self._jobs)</code>, and <code>_enqueue_next</code> selects
jobs with <code>status in (PENDING, ERROR)</code> (<code>app.py:664</code>).
The just-failed job is <code>ERROR</code>, so it is immediately re-run before
any later file.

- **Evidence:** harness — after job #1 errored it was re-enqueued 5/5 times and
  **job #2 never started** (<code>in_queue</code> sequence <code>[1,1,1,1,1]</code>).
- **Impact:** one bad file wedges the entire batch in an infinite failure loop.
- **Fix:** auto-continue only to <code>PENDING</code>; keep <code>ERROR</code>
  retry manual.
- **Refs:** <code>app.py:567</code>, <code>app.py:664</code>,
  <code>app.py:1005-1007</code>

### BLOCKER-2 — Drag-and-drop crashes on every drop
<code>_on_drop</code> calls <code>app._add_files(paths=accepted)</code>, but
<code>_add_files</code> takes no arguments (<code>app.py:324</code>).

- **Evidence:** <code>TypeError: UpscaleGUI._add_files() got an unexpected
  keyword argument 'paths'</code>.
- **Impact:** tkinterdnd2 **is** installed, so DnD is wired up and actively
  broken on every drop. README simultaneously claims DnD "is not enabled".
- **Fix:** <code>def _add_files(self, paths=None)</code> sharing the picker code
  path.
- **Refs:** <code>widgets/input_panel.py:153</code>, <code>app.py:324</code>

### BLOCKER-3 — Queue visually vanishes after a theme change / locale toggle
<code>_on_theme_change</code>, <code>_toggle_locale</code>,
<code>_reset_settings</code>, and <code>_import_settings</code> all call
<code>_build_ui()</code>, which builds a fresh <code>_InputPanel</code> but
never calls <code>_refresh_queue_listbox()</code>.

- **Evidence:** <code>_jobs</code> still held 4 entries, listbox rows went
  **4 -> 0**, and neither the listbox nor the empty state was packed. The queue
  mode also reverted to the last *saved* value rather than the live selection.
- **Impact:** running jobs become invisible and unmanageable (can't pause,
  cancel, or remove).
- **Fix:** after any rebuild, re-apply the live queue mode and call
  <code>_refresh_queue_listbox()</code>.
- **Refs:** <code>app_settings_io.py</code> (<code>_on_theme_change</code>,
  <code>_reset_settings</code>, <code>_import_settings</code>),
  <code>app_window_chrome.py</code> (<code>_toggle_locale</code>)

### BLOCKER-4 — Pause / Cancel / Resume target the wrong job under a filter
<code>_selected_job_id</code> and <code>_refresh_controls</code> index
<code>app._jobs[idxs[0]]</code>, but the listbox is filtered
(<code>_visible_jobs()</code>).

- **Evidence:** jobs #10 (pending) / #11 (error), filter = "Error", row 0
  selected -> returned **10** instead of **11**.
- **Impact:** cancel/pause/resume can hit the wrong file.
- **Fix:** map the selected row through <code>_visible_jobs()</code>.
- **Refs:** <code>widgets/input_panel.py:219</code>,
  <code>widgets/input_panel.py:243</code>, <code>app.py:419</code>

---

## Packaging / distribution gaps

### MED-1 — The wheel ships no assets
<code>pyproject.toml</code> declares only <code>data/*.json</code> as package
data. The built wheel contains **zero PNG/ICO files and no assets/ directory**.
<code>icons.py</code> resolves <code>../assets</code> and silently returns
<code>None</code>.

- **Impact:** a pip install gives no button icons, no splash artwork, and no
  window icon. Splash hero (hero-800x600.png) is also missing.
- **Refs:** <code>pyproject.toml</code>
  (<code>[tool.setuptools.package-data]</code>), <code>icons.py:21</code>

### MED-2 — GPU monitor is dead by default
<code>pynvml</code> is neither declared nor in <code>requirements.txt</code>, so
the util/VRAM/temp panel always shows an em dash.

- **Fix:** add <code>nvidia-ml-py</code> (or document the optional install and
  show an explicit "install pynvml" hint).
- **Refs:** <code>widgets/gpu_monitor.py:20</code>, <code>requirements.txt</code>

### MED-3 — requirements.txt and pyproject.toml disagree
<code>torch>=2.12</code> vs <code>torch>=2.0</code>, <code>numpy>=2.4</code> vs
<code>>=1.24</code>, and several more. <code>pip install .</code> can resolve
versions the app was never tested against.

- **Fix:** single source of truth (derive one from the other, or pin both).
- **Refs:** <code>requirements.txt</code>, <code>pyproject.toml</code>

### MED-4 — License / manifest
No <code>LICENSE</code> in the app directory (repo root has one) and no
<code>MANIFEST.in</code>, despite <code>license = { text = "MIT" }</code>.

---

## UX / accessibility / polish

### LOW-1 — auto_wraplength() is dead code
Defined in <code>a11y.py:99</code>, never called. The status label keeps
<code>wraplength=900</code> (<code>status_bar.py:21</code>) while the window
minimum is 800px wide, so long status text clips horizontally instead of
wrapping.

### LOW-2 — Global mousewheel binding steals scrolling
<code>_make_scrollable</code> uses <code>canvas.bind_all("&lt;MouseWheel&gt;")</code>
(<code>app.py:284-287</code>); scrolling the queue listbox or log panel also
scrolls the Upscale canvas. Bind on the canvas with enter/leave, or route by
widget.

### LOW-3 — Documentation drift
README says 349 tests (actual 370) and DnD "not enabled" (it is). RESUME says
<code>app.py ~845</code> lines (actual **1054**).
<code>ui_ux_overhaul_checklist.md</code> says "10/14 done" while RESUME says all
14 are done. RESUME also linked to this audit file, which did not exist before
this review.

### LOW-4 — Splash asset in installed builds
The splash reads <code>hero-800x600.png</code>, among the assets missing from a
wheel install (same root cause as MED-1).

### LOW-5 — Minor cleanups
<code>_iconbtn</code>'s <code>primary</code> parameter is unused;
<code>status_triplet</code> is imported inside <code>__init__</code> but only
used elsewhere; single-mode jobs are never added to <code>_jobs</code>, so they
never appear in the queue list.

---

## Coverage gap (why 370 green still misses these)

There is **no test** for: DnD (<code>_on_drop</code>), batch error continuation,
queue re-population after <code>_build_ui</code>, or filtered-listbox job
selection. Those blind spots coincide exactly with BLOCKER-1..4. Adding targeted
tests is the cheapest way to stop regressions.

---

## Recommended path to "production-ready"

1. Fix BLOCKER-1..4 (each is a 1-10 line change).
2. Add regression tests for those paths.
3. Fix packaging (MED-1..3) and verify a wheel install launches with icons.
4. Reconcile README/RESUME/checklist with reality.
5. Then the LOW polish items.

See [../plans/production_readiness_fixes.md](../plans/production_readiness_fixes.md)
for the tracked milestone/task breakdown.

---

## Appendix — commands used

~~~powershell
# full suite (from repo root; apps/ must be importable)
apps\anime_upscaler_gui\.venv\Scripts\python.exe -m pytest apps\anime_upscaler_gui\tests\ -q

# live launch + widget introspection and DnD/batch/filter harnesses were
# temporary scripts run with the same interpreter and deleted afterwards.
~~~
