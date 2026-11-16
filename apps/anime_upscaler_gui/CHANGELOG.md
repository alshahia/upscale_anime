# Changelog

All notable changes to the Anime Upscaler GUI. The format loosely follows
Keep a Changelog; versions follow SemVer.

## [0.2.0] - 2026-10-11

Production-readiness pass. See `docs/audit/gui_review_2026-10.md` and
`docs/plans/production_readiness_fixes.md`.

### Fixed

- Batch mode no longer retries a failed job forever: auto-continuation
  advances PENDING only, while Start resets ERROR -> PENDING for an explicit
  retry (PR-01).
- Drag-and-drop no longer raises TypeError: `_add_files` accepts dropped
  paths and shares one code path with the file picker (PR-02).
- Theme change / locale toggle / settings reset / import no longer wipe the
  queue listbox or revert the queue mode; a single `_rebuild_ui()` helper
  restores the widget state (PR-03).
- Pause / Resume / Cancel now act on the selected row even when a status
  filter is active (PR-04).
- The status label wraps to its real width instead of a fixed 900px, so long
  messages no longer clip at the 800px minimum window (PR-09).
- The mousewheel binding no longer steals scrolling from the queue listbox
  and the log panel (PR-10).

### Changed

- Icon and splash assets moved inside the package and declared as
  package-data; the wheel now ships 51 asset files (~2.9 MB) instead of none
  (PR-05). Asset resolution keeps a fallback for older source trees.
- `nvidia-ml-py` is a declared dependency, so the GPU monitor works out of
  the box (PR-06).
- `requirements.txt` and `[project].dependencies` are reconciled and guarded
  by `tests/test_packaging.py` (PR-07).
- Removed the unused `primary` button parameter and a redundant import
  (PR-12).
- Documentation counts and drag-and-drop status corrected (PR-11).

### Added

- LICENSE in the app directory and MANIFEST.in for sdist contents (PR-08).
- Regression tests covering the four blockers, packaging, and dynamic
  wrapping.
- This CHANGELOG.

## [0.1.0] - prior

Initial standalone Tk GUI: 14-phase UI/UX overhaul, 3 themes, EN/AR locales,
batch queue, video cut window, model downloader, splash screen.
