# PR bundle: GUI Queue Controls + GPU Codec

This directory contains everything you need to open a PR for the
`perf/queue-controls-gpu-codec` branch without me needing a configured
remote.

The branch has already been merged into `refactor/gui-extensibility`
locally (merge commit `3590a4b`). To push to a forge, you have two
paths:

## Path A: push the merged branch directly (recommended)

```powershell
# Add your remote (one-time)
git remote add origin <your-fork-or-upstream-URL>

# Push the merged branch
git push -u origin refactor/gui-extensibility

# Open the PR in your web UI:
#   base: main
#   head: refactor/gui-extensibility
#   body: paste PR_BODY.md contents
```

If you want the PR to come from a feature branch instead, create one:

```powershell
git checkout -b pr/gui-queue-controls-gpu-codec
git push -u origin pr/gui-queue-controls-gpu-codec
```

## Path B: apply the patches somewhere else

If you want to land these changes on a different machine or in a
freshly-cloned repo without the merge commit:

```powershell
# Inside a clean repo at the same parent commit (6f7c1f6):
git am < docs/PRs/0002-...Q1-docs...patch
git am < docs/PRs/0003-...Q2-Pause...patch
git am < docs/PRs/0004-...Q3-NVDEC...patch
git am < docs/PRs/0005-...Q4-TF32...patch
git am < docs/PRs/0006-...Q5-docs...patch
# No merge commit is created — history is linear.
```

Or apply everything in one go:

```powershell
# Apply Q1-Q5 atomically; creates the 5 commits.
git am < docs/PRs/0000-MERGE.patch
```

## Files in this bundle

| File | Purpose |
|------|---------|
| `PR_BODY.md` | Paste into the PR description in your forge. |
| `0000-MERGE.patch` | All 5 commits as one patch (apply with `git am`). |
| `0002-...Q1-...patch` | Just Q1 (docs only). |
| `0003-...Q2-...patch` | Just Q2 (Pause/Resume/Cancel). |
| `0004-...Q3-...patch` | Just Q3 (NVDEC + NVENC). |
| `0005-...Q4-...patch` | Just Q4 (TF32 + async + telemetry). |
| `0006-...Q5-...patch` | Just Q5 (final docs). |

## Sanity check before opening the PR

```powershell
# 1. Make sure tests still pass on the merged branch
git checkout refactor/gui-extensibility
.venv\Scripts\python.exe -m pytest apps/anime_upscaler_gui/tests `
  --ignore=apps/anime_upscaler_gui/tests/test_video_end_to_end.py `
  --ignore=apps/anime_upscaler_gui/tests/test_assets.py `
  --basetemp=E:/Temp/pytest_pr

# 2. Smoke test
.venv\Scripts\python.exe tmp/mid2s_job.py
# Expect: 'FINAL: finished size: ~575701' in ~95s wall-clock.
```

## Known to NOT be a regression

- `test_empty_states.py::test_input_panel_shows_empty_state_in_batch_mode`
  fails on Windows when Tk can't find `init.tcl`. This is an
  environmental flake that pre-dates this branch.
- NVENC output byte counts vary ±0.2% run-to-run at qp=18. Benign
  hardware rate-control non-determinism.

## After the PR is merged

- Delete the local feature branch (optional):
  ```
  git branch -d perf/queue-controls-gpu-codec
  git branch -d pr/gui-queue-controls-gpu-codec
  ```
- Keep the handoff at
  `.dsh/handoffs/session-7cc5a443-f7aa-489b-9a68-f490f7a258f5/004-20260915-perf-queue-controls/handoff.md`
  as the session-history record. Move it to a permanent location
  (e.g. `docs/handoffs/`) if your repo convention requires it.
