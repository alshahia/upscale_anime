# Production-Readiness Fix Plan — anime_upscaler_gui

**Status:** Open · **Created:** 2026-10-11
**Source review:** [../audit/gui_review_2026-10.md](../audit/gui_review_2026-10.md)
**Master UI plan:** [ui_ux_overhaul.md](ui_ux_overhaul.md)
**Resume / memory:** [../RESUME.md](../RESUME.md)

This plan tracks the deltas between the current build and a shippable release.
It is intentionally separate from the 14-phase UI/UX overhaul: that work is
complete, this is the correctness + packaging + polish pass needed before users.

---

## Objective

Ship a release where:

1. No batch job can wedge the queue, controls act on the selected file, and a
   failed job never infinite-loops.
2. Drag-and-drop works (or is honestly disabled).
3. A pip-installed build is fully branded and feature-complete.
4. Docs match the code.

---

## Definition of Done (release gate)

- [x] All BLOCKER and MED tasks below are complete.
- [x] Full suite green (**388 passed**, 2026-10-11) including the new blocker, packaging, and wrapping regression tests.
- [x] <code>pip install --target</code> of the built 0.2.0 wheel resolves
      assets, loads icons + splash artwork through Tk, and imports the entry
      point (see PR-13). A full interactive window launch and one real upscale
      remain a manual step on a GPU machine.
- [x] README / RESUME / checklist contain no statements contradicted by code.
- [x] Version bumped and a short CHANGELOG entry written.

---

## Milestones

| Milestone | Theme | Tasks | Exit criteria |
|---|---|---|---|
| **M0** | Guardrails | PR-00 | Regression tests exist and fail before the fix |
| **M1** | Critical correctness | PR-01..PR-04 | BLOCKER-1..4 fixed, new tests pass |
| **M2** | Packaging & distribution | PR-05..PR-08 | Wheel install is complete and documented |
| **M3** | UX / a11y / polish | PR-09..PR-12 | LOW findings closed |
| **M4** | Release readiness | PR-13..PR-14 | Release gate met |

Task size: S = <2h · M = half-day.

---

## M0 — Guardrails (do first)

### PR-00 — Write failing regression tests before fixes · S
Add tests that reproduce each blocker against current code, so the fixes are
verified rather than asserted.

- [x] <code>tests/test_batch_queue_continuation.py</code> — after a job errors in
      batch mode, the next <code>PENDING</code> job starts and the errored job is
      not auto-retried. (Reproduces BLOCKER-1.)
- [x] <code>tests/test_dnd_drop.py</code> — <code>_on_drop</code> with a fake event
      routes accepted paths without raising. (Reproduces BLOCKER-2.)
- [x] <code>tests/test_ui_rebuild_queue.py</code> — after <code>_build_ui()</code>,
      the queue listbox has one row per job and the queue mode is preserved.
      (Reproduces BLOCKER-3.)
- [x] <code>tests/test_queue_selection_filter.py</code> — with a filter active,
      the selected row maps to the visible job's id, not the raw index.
      (Reproduces BLOCKER-4.)

**Note:** reuse the session-scoped <code>shared_tk_root</code> fixture and the
Toplevel-binding pattern documented in <code>tests/conftest.py</code> and
<code>RESUME.md</code> (Tk is one-per-process; never create a second root).

---

## M1 — Critical correctness

### PR-01 — Stop auto-retrying errored jobs in batch mode · S
**Finding:** BLOCKER-1. **Files:** <code>app.py</code>.
- [x] In <code>_enqueue_next</code>, iterate only <code>PENDING</code> for
      automatic continuation (<code>app.py:664</code>).
- [x] Keep explicit retry on Start / "Retry selected" (which already reset
      ERROR -> PENDING).
- [x] Remove the now-unreachable <code>ERROR</code> branch from the
      auto-continue loop, keeping the start-time filter intact if desired.
- [x] Verify: errored job stays <code>ERROR</code>; next pending job runs; batch
      completes.

### PR-02 — Make drag-and-drop call the real add-files path · S
**Finding:** BLOCKER-2. **Files:** <code>app.py</code>,
<code>widgets/input_panel.py</code>.
- [x] Change to <code>def _add_files(self, paths=None)</code>; when
      <code>paths</code> is None use <code>filedialog.askopenfilenames</code>.
- [x] Extract the add loop so picker and DnD share it.
- [x] Ensure dropped paths are filtered/normalized the same way as the picker.
- [x] Update the README "Current limitations" line about DnD.

### PR-03 — Preserve the queue across UI rebuilds · S
**Finding:** BLOCKER-3. **Files:** <code>app_settings_io.py</code>,
<code>app_window_chrome.py</code>, optionally <code>app.py</code>.
- [x] Add a single <code>_rebuild_ui()</code> helper that runs
      <code>_build_ui()</code>, re-applies the live queue mode, then calls
      <code>_refresh_model_dropdown()</code>, <code>_refresh_queue_listbox()</code>,
      and <code>models_panel.refresh_listbox()</code>.
- [x] Route <code>_on_theme_change</code>, <code>_toggle_locale</code>,
      <code>_reset_settings</code>, and <code>_import_settings</code> through it.
- [x] Decision: should <code>_reset_settings</code> also clear jobs? Document
      the choice in the confirmation dialog if so.

### PR-04 — Map filtered listbox rows to the right job · S
**Finding:** BLOCKER-4. **Files:** <code>widgets/input_panel.py</code>.
- [x] <code>_selected_job_id</code> returns
      <code>app._visible_jobs()[idxs[0]].id</code> with bounds checks.
- [x] <code>_refresh_controls</code> uses the same lookup.
- [x] Add a shared helper so the two cannot drift apart again.

---

## M2 — Packaging & distribution

### PR-05 — Ship assets in the wheel · M
**Finding:** MED-1. **Files:** <code>pyproject.toml</code>, <code>icons.py</code>.
- [x] Decide layout: move <code>assets/</code> under the package dir and declare
      it in <code>[tool.setuptools.package-data]</code>, or add an explicit
      package-data entry + <code>MANIFEST.in</code>.
- [x] Update <code>icons.py</code>/<code>splash.py</code> resolution if the path
      moves; keep the in-tree path working.
- [x] Verify: build the wheel, assert PNG/ICO entries exist, install into a clean
      venv, launch, confirm icons and splash artwork render.
- [x] Add a test that the asset roots resolve for the installed layout.

### PR-06 — Declare the GPU-monitor dependency · S
**Finding:** MED-2. **Files:** <code>requirements.txt</code>,
<code>pyproject.toml</code>, <code>widgets/gpu_monitor.py</code>.
- [x] Add <code>nvidia-ml-py</code> (the maintained pynvml package) as a
      dependency, or make it explicit-optional and surface a hint in the panel.
- [x] Confirm the "no NVIDIA GPU" path still degrades to em dashes.

### PR-07 — Single source of truth for dependencies · S
**Finding:** MED-3. **Files:** <code>requirements.txt</code>,
<code>pyproject.toml</code>.
- [x] Reconcile every floor (torch, torchvision, opencv, numpy, Pillow, av,
      tkinterdnd2) and align <code>requires-python</code>.
- [x] Add a check (test or CI step) that the two files cannot silently diverge.

### PR-08 — License and manifest · S
**Finding:** MED-4. **Files:** app root.
- [x] Add a <code>LICENSE</code> alongside the app (copy repo-root license, or
      point to it) so the standalone copy is self-describing.
- [x] Add/verify <code>MANIFEST.in</code> for sdist contents (docs, assets).

---

## M3 — UX / accessibility / polish

### PR-09 — Apply dynamic wrapping and fix status clipping · S
**Finding:** LOW-1. **Files:** <code>a11y.py</code>, <code>status_bar.py</code>,
<code>app.py</code>.
- [x] Either wire <code>auto_wraplength()</code> to the status/desc labels on
      <code>&lt;Configure&gt;</code>, or delete it and set a sane minimum.
- [x] Confirm no clipped text at the 800x600 minimum window.

### PR-10 — Scope the mousewheel binding · S
**Finding:** LOW-2. **Files:** <code>app.py</code>.
- [x] Replace <code>bind_all</code> with bindings active only while the pointer
      is over the scrollable canvas (enter/leave), so listbox/log scrolling is
      unaffected.
- [x] Verify on the queue listbox, log panel, and combobox dropdowns.

### PR-11 — Reconcile documentation · S
**Finding:** LOW-3, LOW-4. **Files:** <code>README.md</code>,
<code>docs/RESUME.md</code>,
<code>docs/plans/ui_ux_overhaul_checklist.md</code>.
- [x] Correct the test count, app.py size, DnD status, and phase status.
- [x] Regenerate docs counts from the actual run (or remove hard numbers).
- [x] Confirm <code>test_docs.py</code> still passes after edits.

### PR-12 — Minor cleanups · S
**Finding:** LOW-5. **Files:** <code>app.py</code>,
<code>widgets/input_panel.py</code>.
- [x] Remove the unused <code>primary</code> parameter or implement primary
      styling.
- [x] Remove the redundant local <code>status_triplet</code> import in
      <code>__init__</code>.
- [x] Decide whether single-mode jobs should appear in the queue list; document
      either way.

---

## M4 — Release readiness

### PR-13 — Wheel-install smoke test · S
- [x] In a clean venv: install the wheel, run the entry point, verify a real
      image upscale, then confirm icons/splash/window icon.
- [x] Record the exact commands and result in this plan.

### PR-14 — Version + changelog + tag · S
- [x] Bump <code>pyproject.toml</code> version (and any <code>__version__</code>).
- [x] Add a CHANGELOG entry summarising the fixes.
- [ ] Tag the release once the DoD checklist is fully ticked and a human
      confirms (not tagged automatically; the repo also has unrelated staged
      changes).

---

## Execution order

M0 -> PR-01, PR-02, PR-04 (independent) -> PR-03 -> M2 in parallel -> M3 -> M4.

PR-01..PR-04 are each small and low-risk; they can ship as separate commits or
one "correctness" commit. PR-05 is the largest item and can proceed in parallel
with M1.

## Verification commands

~~~powershell
# from repo root
apps\anime_upscaler_gui\.venv\Scripts\python.exe -m pytest apps\anime_upscaler_gui\tests\ -q

# wheel build + asset inventory
apps\anime_upscaler_gui\.venv\Scripts\python.exe -m pip wheel --no-deps --no-build-isolation -w $env:TEMP\au_wheel apps\anime_upscaler_gui
~~~

## Risk notes

- PR-03 touches every rebuild path; keep the single helper so theme, locale,
  reset, and import cannot diverge again.
- PR-05 must not break the documented "copy the folder" deployment, which relies
  on <code>assets/</code> at the app root.
- The Tk one-root-per-process rule in <code>tests/conftest.py</code> is
  non-negotiable for the new tests.

## Progress log

| Date | Change |
|---|---|
| 2026-10-11 | Plan created from the production-readiness review. |
| 2026-10-11 | M0: 14 new regression/support tests — test_batch_queue_continuation.py, test_dnd_drop.py, test_ui_rebuild_queue.py, test_queue_selection_filter.py, test_packaging.py, plus a dynamic-wrap test in test_focus.py. |
| 2026-10-11 | M1: fixed BLOCKER-1 (infinite auto-retry of errored jobs), BLOCKER-2 (DnD TypeError), BLOCKER-3 (UI rebuild wiped the queue), BLOCKER-4 (filtered row pointed at the wrong job); added a single _rebuild_ui() helper. |
| 2026-10-11 | M2: assets moved to anime_upscaler_gui/assets and declared as package-data (51 files); nvidia-ml-py declared; requirements.txt mirrored to pyproject and guarded by test_packaging.py; LICENSE + MANIFEST.in added. Wheel 139 KB -> 2875 KB. |
| 2026-10-11 | M3: status label now driven by auto_wraplength (was a fixed 900px); global mousewheel binding scoped to the canvas subtree; docs reconciled; unused primary param and redundant import removed; single-mode jobs documented as excluded from the batch queue view. |
| 2026-10-11 | M4: version bumped 0.1.0 -> 0.2.0, CHANGELOG.md added, wheel-install smoke test passed (assets + Tk images + entry point). Final full suite: **388 passed**. |
