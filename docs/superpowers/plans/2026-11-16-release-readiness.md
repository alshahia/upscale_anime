# upscale_anime Release Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the project at E:/python projects/upscale_anime into something a stranger can install and use: one self-contained GUI wheel that upscales video with its audio intact and never loses the source, committed behind a tag, validated by three outside testers by 2026-12-14 - and only afterwards restore the frozen research core.

**Architecture:** Two tracks over one repository. Track 1 (Tasks 0-20, release-critical, 2026-11-16 to 2026-12-14) works on branch `release/gui-preview` off the snapshot commit: commit the truth, fix the GUI data-integrity defects, bundle the measured torch-only model closure into the GUI wheel, fix the manifests, make checkpoint loading safe, rehearse a clean-room install, publish a GitHub pre-release, and run the stranger test. Track 2 (Tasks 21-28, post-release, after 2026-12-14) restores the research core: training imports, preprocessing semantics, data/test integrity, the legacy packager, repository history, PyPI. No Track 1 task edits training code; no Track 2 task blocks the release.

**Tech Stack:** Python 3.12.7 (repo `.venv`), PyTorch 2.12 (CPU tested; CUDA only for manual checks), Tk GUI (`apps/anime_upscaler_gui`, package `anime_upscaler_gui`), ffmpeg/ffprobe CLI, pytest 9, setuptools 77, GitHub Actions, PowerShell 7 on Windows 11.

---

## How to use this plan

- Work top to bottom inside a track. Every task ends with a commit; do not batch tasks.
- Commands are Windows PowerShell run from the repository root. Forward slashes are used inside paths (PowerShell and Python both accept them). Set these once per shell:

```powershell
Set-Location "E:/python projects/upscale_anime"
$py = ".venv/Scripts/python.exe"
$basetemp = Join-Path $env:TEMP "pytest-anime-sr"
$env:PYTHONIOENCODING = "utf-8"
```

- GUI suite command used throughout (the explicit basetemp avoids a local temp-directory permission quirk and is equivalent to plain pytest on CI):

```powershell
& $py -m pytest apps/anime_upscaler_gui/tests -q -p no:cacheprovider "--basetemp=$basetemp"
```

- Baseline at plan time: 388 GUI tests pass in ~38.8 s; HEAD 51da496; 1,792 uncommitted paths. Record the baseline once (Task 0) so every regression is attributable.
- If an expected output does not match, stop and fix the step; never continue on a red baseline.
- `review_room/` is locally excluded from git and holds the review record. This plan is saved under `docs/superpowers/plans/` and is committed in Task 2.

## Task 0: Baseline snapshot and branch

**Files:** none (measurement only).

- [ ] **Step 0.1: Record the baseline**

```powershell
git status --porcelain | Measure-Object -Line
git rev-parse --short HEAD
& $py -m pytest apps/anime_upscaler_gui/tests -q -p no:cacheprovider "--basetemp=$basetemp"
& $py -c "import anime_sr; print(anime_sr.__version__)"
```

Expected: 1,777 status entries, 51da496, 388 passed, 0.1.0. Record the real numbers in the task log.

- [ ] **Step 0.2: Confirm the review artefacts are excluded**

```powershell
git status --porcelain | Select-String "review_room" | Measure-Object -Line
```

Expected: 0. If non-zero, stop: the review record must not enter repository history.

---

# Track 1 - Release-critical (2026-11-16 to 2026-12-14)

## Task 1: Repair the training syntax error (unblocks commit and test collection)

**Files:**
- Modify: `src/anime_sr/training/stage_state.py:83-104`
- Test: `tests/e2e/test_e2e_training_pipeline.py -k stage_state` plus the import check below.

- [ ] **Step 1.1: Confirm the failure**

```powershell
& $py -c "import anime_sr.training"
```

Expected: non-zero exit with SyntaxError: unterminated string literal (detected at line 86).

- [ ] **Step 1.2: Replace `get_summary` with the corrected body**

Replace the method currently spanning lines 83-104 with exactly:

```python
    def get_summary(self) -> str:
        """Get text summary of stage state."""
        lines = [
            "Stage State Summary:",
            f"  Current Stage: {self.current_stage}",
            "  Stage 1:",
            f"    Completed: {self.stage1_completed}",
            f"    Epochs: {self.stage1_epochs_completed}",
            f"    Best Metric: {self.stage1_best_metric:.6f}"
            if self.stage1_best_metric != float("inf")
            else "    Best Metric: N/A",
            f"    Checkpoint: {self.stage1_best_checkpoint}"
            if self.stage1_best_checkpoint
            else "    Checkpoint: N/A",
        ]

        if self.stage2_enabled:
            lines.extend([
                "  Stage 2:",
                f"    Enabled: {self.stage2_enabled}",
                f"    Started: {self.stage2_started}",
                f"    Completed: {self.stage2_completed}",
                f"    Epochs: {self.stage2_epochs_completed}",
            ])

        return "\n".join(lines)
```

- [ ] **Step 1.3: Verify**

```powershell
& $py -m compileall -q src
& $py -c "import anime_sr.training; print('training import OK')"
& $py -m pytest tests/e2e/test_e2e_training_pipeline.py -k stage_state -q -p no:cacheprovider "--basetemp=$basetemp"
```

Expected: training import OK and the stage-state test passes.

- [ ] **Step 1.4: Commit**

```powershell
git add src/anime_sr/training/stage_state.py
git commit -m "fix(training): restore three f-strings in StageState.get_summary"
```

## Task 2: Commit the 46-day backlog and tag the snapshot

**Files:** all 1,792 changed paths; no edits beyond the Task 1 repair and this plan document.

- [ ] **Step 2.1: Inspect what is staged (the repository purge)**

```powershell
git diff --cached --stat | Select-Object -Last 3
git diff --cached --name-only | Select-String "Python/|windsurf|claude" | Measure-Object -Line
git status --porcelain | Select-String "docs/superpowers" | Measure-Object -Line
```

Expected: about 1,628 staged paths dominated by deletions; the vendored Python interpreter and agent-state folders among them; the plan document untracked. If the staged set does not look like the reviewed purge, stop and ask before committing.

- [ ] **Step 2.2: Commit the purge, then the readiness batch and the plan**

```powershell
git commit -m "chore(repo): remove vendored Python runtime, agent state, and stale 2025 artefacts"
git add -A
git status --porcelain | Measure-Object -Line
git commit -m "feat: production-readiness batch (GUI fixes, packaging guards, docs, tests); training core frozen"
git log --oneline -3
```

Expected: status empty after the add, three commits visible.

- [ ] **Step 2.3: Tag and branch**

```powershell
git tag -a snapshot-2026-11-16 -m "Review-room snapshot: clean tree, measured 2026-11-16"
git switch -c release/gui-preview
git rev-parse --abbrev-ref HEAD
```

Expected: release/gui-preview.

- [ ] **Step 2.4: Record the starting build failure**

```powershell
& $py -m pip wheel . --no-deps -w dist
```

Expected: **fails** with InvalidConfigError about superseded License classifiers (PEP 639). That is the known blocker Task 13 removes; paste the message in the task log.

## Task 3: Truth pass - make documentation match the tree

**Files:**
- Modify: `docs/PROJECT_STATUS.md:4-6` and `:344`
- Modify: `docs/PROJECT_MEMORY.md:46`, `:68`
- Modify: `docs/AGENTS.md:39`, `:118`, `:121`
- Modify: `README.md:40`, `:54`, `:57`
- Modify: `apps/anime_upscaler_gui/README.md:143-152`, `:171`
- Create: `docs/status/MEASURED_STATUS_2026-11-16.md`

- [ ] **Step 3.1: Create the measured-status source of truth**

Create `docs/status/MEASURED_STATUS_2026-11-16.md` with:

```markdown
# Measured status - 2026-11-16

| Component | State | Evidence |
|---|---|---|
| GUI test suite | green | 388 passed in 38.84 s |
| GUI wheel build | green | 3.1 MB wheel, 51 images, data/registry.json |
| GUI installability | broken | imports anime_sr, declares no dependency; pyav not on PyPI |
| GUI video output | broken | video only, no audio |
| GUI failed-encode handling | broken | failed encode reported as finished |
| GUI output path guard | broken | output path can equal input path |
| Training package import | broken | SyntaxError at stage_state.py:86 (fixed 2026-11-16) |
| Root wheel build (worktree) | broken | PEP 639 classifier conflict |
| Root wheel build (HEAD) | green | built in rehearsal |
| CI on this branch | never run | triggers main/develop; branch is refactor/restructure |
| Uncommitted work | 1,792 paths | committed 2026-11-16, tag snapshot-2026-11-16 |
```

- [ ] **Step 3.2: Replace the false status lines**

In `docs/PROJECT_STATUS.md` replace lines 4-6 with:

```markdown
**Status:** Pre-release (GUI preview v0.2.0-preview, pending stranger test)
**Completion:** GUI path under active repair; research core frozen until 2026-12-14
**Last verified:** 2026-11-16 (see docs/status/MEASURED_STATUS_2026-11-16.md)
```

At line 344 replace the "95% complete and production ready" sentence with: status is generated from the measured table; do not quote historical completion percentages.

- [ ] **Step 3.3: Fix the stale architecture docs**

`docs/PROJECT_MEMORY.md` line 46: branch becomes `release/gui-preview`; line 68: Updated: 2026-11-16.

`docs/AGENTS.md` line 39: replace "No pyproject.toml" with "Packaging: root pyproject.toml (anime-sr 0.1.0) and apps/anime_upscaler_gui/pyproject.toml (anime-upscaler-gui 0.2.0)." Line 118: replace "no conftest.py, no pytest.ini" with "Tests: root tests/ and apps/anime_upscaler_gui/tests/; conftest.py exists in both trees." Line 121: replace bare-import convention with "Imports are package-qualified (anime_sr.*, anime_upscaler_gui.*)."

- [ ] **Step 3.4: Fix the install commands**

`README.md` line 40: `pip install -e .` becomes `pip install -e ".[dev]"`. Line 54: `configs/train_config.yaml` becomes `configs/base.yaml`. Line 57: `./checkpoints/model_best.pt` becomes `checkpoints/finetune_best.pth` plus the sentence "Demonstration checkpoint, not a released model; see docs/status/MEASURED_STATUS_2026-11-16.md."

`apps/anime_upscaler_gui/README.md` lines 143-152: replace the broken standalone recipe with the wheel flow Task 17 rehearses:

```powershell
python -m venv .venv-anime-gui
.venv-anime-gui/Scripts/python.exe -m pip install anime_upscaler_gui-0.2.0-py3-none-any.whl
.venv-anime-gui/Scripts/anime-upscaler-gui.exe --self-check
```

State that the wheel is attached to the GitHub pre-release and that `--self-check` is added in Task 17. At line 171 replace "not included in this standalone GUI" with "bundled inside the wheel under anime_upscaler_gui/_models/; see scripts/sync_vendored_models.py."

- [ ] **Step 3.5: Verify and commit**

```powershell
Select-String -Path docs/PROJECT_STATUS.md -Pattern "Production Ready" | Measure-Object -Line
Select-String -Path docs/PROJECT_MEMORY.md -Pattern "phase-1-realtime-4k" | Measure-Object -Line
Select-String -Path docs/AGENTS.md -Pattern "No pyproject.toml" | Measure-Object -Line
git add docs README.md apps/anime_upscaler_gui/README.md
git commit -m "docs: replace 95%-production-ready claim with measured 2026-11-16 status"
```

Expected counts: 0, 0, 0.

## Task 4: Add the release regression harness

**Files:**
- Create: `tests/release/test_artifacts.py`

- [ ] **Step 4.1: Write the failing tests**

```python
"""Release-branch guards: must fail while the release blockers exist."""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _run(args):
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True)


def test_gui_wheel_builds():
    out = REPO / "dist"
    out.mkdir(exist_ok=True)
    r = _run([sys.executable, "-m", "pip", "wheel",
              str(REPO / "apps" / "anime_upscaler_gui"),
              "--no-deps", "-w", str(out)])
    assert r.returncode == 0, r.stderr[-2000:]


def test_clean_room_import_does_not_need_anime_sr():
    code = "\n".join([
        "import sys",
        "class Block:",
        "    def find_spec(self, name, path=None, target=None):",
        "        if name == 'anime_sr' or name.startswith('anime_sr.'):",
        "            raise ModuleNotFoundError('blocked for clean-room test')",
        "sys.meta_path.insert(0, Block())",
        "import anime_upscaler_gui.arch_registry",
        "print('clean-room import OK')",
    ])
    r = _run([sys.executable, "-c", code])
    assert r.returncode == 0, r.stderr[-2000:]
```

- [ ] **Step 4.2: Run to record the starting state**

```powershell
& $py -m pytest tests/release -q -p no:cacheprovider "--basetemp=$basetemp"
```

Expected: both tests fail at this point (wheel may pass; clean-room import raises ModuleNotFoundError). Record which fail.

- [ ] **Step 4.3: Commit the red guards**

```powershell
git add tests/release/test_artifacts.py
git commit -m "test(release): add wheel-build and clean-room import guards (currently red)"
```

---

# Track 1A - GUI safety: protect the user's files (Tasks 5-11)

## Task 5: Audio remux helpers in ffmpeg.py

**Files:**
- Modify: `apps/anime_upscaler_gui/anime_upscaler_gui/ffmpeg.py` (append after `probe_video_meta`)
- Create: `apps/anime_upscaler_gui/tests/test_ffmpeg_audio.py`

- [ ] **Step 5.1: Write the failing tests**

```python
"""Audio-preservation helpers: encoder output must keep the source audio."""
from __future__ import annotations

import shutil
import subprocess

import pytest

from anime_upscaler_gui.ffmpeg import has_audio_stream, mux_audio

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe required",
)


def _make_clip(tmp_path, *, with_audio: bool):
    out = tmp_path / ("av.mp4" if with_audio else "video.mp4")
    args = ["ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=s=64x64:d=1:r=10"]
    if with_audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                 "-c:a", "aac", "-shortest"]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(args, check=True)
    return out


def test_has_audio_stream_detects_audio_and_video_only(tmp_path):
    assert has_audio_stream(_make_clip(tmp_path, with_audio=True)) is True
    assert has_audio_stream(_make_clip(tmp_path, with_audio=False)) is False


def test_mux_audio_adds_the_source_track(tmp_path):
    src = _make_clip(tmp_path, with_audio=True)
    video_only = tmp_path / "video_only.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                    "-an", "-c:v", "copy", str(video_only)], check=True)
    dest = tmp_path / "muxed.mp4"
    assert mux_audio(video_only, src, dest) is True
    assert dest.is_file()
    assert has_audio_stream(dest) is True
    assert not (tmp_path / "muxed.muxing.mp4").exists()
```

- [ ] **Step 5.2: Run to verify failure**

```powershell
& $py -m pytest apps/anime_upscaler_gui/tests/test_ffmpeg_audio.py -q -p no:cacheprovider "--basetemp=$basetemp"
```

Expected: ImportError: cannot import name 'has_audio_stream'.

- [ ] **Step 5.3: Implement the helpers**

Append to `apps/anime_upscaler_gui/anime_upscaler_gui/ffmpeg.py`:

```python
def has_audio_stream(path) -> bool:
    """True when ffprobe reports at least one audio stream."""
    if not ffprobe_available():
        return False
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return r.returncode == 0 and r.stdout.strip() != ""


def mux_audio(video_only, audio_source, dest, *, start_seconds: float = 0.0,
              duration_seconds: float | None = None) -> bool:
    """Remux the source audio into the encoded video without re-encoding.

    Writes a sibling '.muxing' temp file and os.replace()s it over dest, so a
    crash never leaves a half-written destination. Returns True when dest now
    exists and carries audio. start_seconds/duration_seconds mirror the job's
    cut window so the sound matches exactly the frames that were encoded.
    """
    if not ffmpeg_available():
        return False
    tmp = dest.with_name(dest.stem + ".muxing" + dest.suffix)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_only)]
    if start_seconds > 0:
        cmd += ["-ss", f"{start_seconds:.3f}"]
    if duration_seconds is not None and duration_seconds > 0:
        cmd += ["-t", f"{duration_seconds:.3f}"]
    cmd += ["-i", str(audio_source),
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c", "copy", "-shortest", str(tmp)]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=3600)
    except subprocess.TimeoutExpired:
        return False
    if r.returncode != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        return False
    os.replace(tmp, dest)
    return True
```

Note: `ffmpeg.py` already imports `subprocess`; add `import os` next to the existing imports if missing.

- [ ] **Step 5.4: Verify**

```powershell
& $py -m pytest apps/anime_upscaler_gui/tests/test_ffmpeg_audio.py -q -p no:cacheprovider "--basetemp=$basetemp"
```

Expected: 2 passed.

- [ ] **Step 5.5: Commit**

```powershell
git add apps/anime_upscaler_gui/anime_upscaler_gui/ffmpeg.py apps/anime_upscaler_gui/tests/test_ffmpeg_audio.py
git commit -m "feat(ffmpeg): audio probe + atomic audio remux helpers"
```

## Task 6: Wire audio preservation into both video paths

**Files:**
- Modify: `apps/anime_upscaler_gui/anime_upscaler_gui/pipeline/worker.py` (per-frame video: encode open ~575, finally ~724, finished ~740; recurrent path: encode open ~863, finished ~930, finally ~945)
- Create: `apps/anime_upscaler_gui/tests/test_video_audio.py`

- [ ] **Step 6.1: Write the failing tests (no model needed)**

```python
"""A finished video job must carry the source audio when the source has it."""
from __future__ import annotations

import shutil
import subprocess

import pytest

from anime_upscaler_gui.ffmpeg import has_audio_stream
from anime_upscaler_gui.pipeline.worker import _mux_job_audio

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe required",
)


def _make_clip(tmp_path, with_audio=True):
    out = tmp_path / "src.mp4"
    args = ["ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=s=64x64:d=1:r=10"]
    if with_audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                 "-c:a", "aac", "-shortest"]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(args, check=True)
    return out


def test_mux_job_audio_keeps_sound(tmp_path):
    src = _make_clip(tmp_path, with_audio=True)
    silent = tmp_path / "out.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                    "-an", "-c:v", "copy", str(silent)], check=True)
    ok = _mux_job_audio(silent, src, start_seconds=0.0, duration_seconds=1.0)
    assert ok is True
    assert has_audio_stream(silent) is True


def test_mux_job_audio_returns_false_without_audio(tmp_path):
    src = _make_clip(tmp_path, with_audio=False)
    silent = tmp_path / "out.mp4"
    shutil.copyfile(src, silent)
    assert _mux_job_audio(silent, src, start_seconds=0.0,
                          duration_seconds=1.0) is False
```

- [ ] **Step 6.2: Run to verify failure**

```powershell
& $py -m pytest apps/anime_upscaler_gui/tests/test_video_audio.py -q -p no:cacheprovider "--basetemp=$basetemp"
```

Expected: ImportError: cannot import name '_mux_job_audio'.

- [ ] **Step 6.3: Add the helper to worker.py after `_unlink_partial_output`**

```python
def _mux_job_audio(out_path, input_path, *, start_seconds: float = 0.0,
                   duration_seconds: float | None = None) -> bool:
    """Remux the source audio into the encoded output.

    Returns False when the source has no audio (video-only output stays
    valid); the caller treats a False on an audio-bearing source as a warning
    it must show, never as silence shipped quietly.
    """
    from ..ffmpeg import has_audio_stream, mux_audio

    if not has_audio_stream(input_path):
        return False
    return mux_audio(out_path, input_path, out_path,
                     start_seconds=start_seconds,
                     duration_seconds=duration_seconds)
```

- [ ] **Step 6.4: Call it in `_run_video` before the finished emit**

Insert immediately before the existing finished emit:

```python
        # Keep the sound: remux the source audio into the encoded frames.
        audio_ok = True
        if use_pipe:
            cut = getattr(job, "cut_end_seconds", 0) - getattr(job, "cut_start_seconds", 0)
            if cut > 0:
                audio_ok = _mux_job_audio(out_path, job.input_path,
                                          start_seconds=job.cut_start_seconds,
                                          duration_seconds=cut)
            else:
                audio_ok = _mux_job_audio(out_path, job.input_path)
        else:
            audio_ok = False  # cv2 fallback cannot mux audio

        message = f"-> {out_path.name} ({proc} frames, {avg_fps:.2f} fps)"
        if not audio_ok and has_audio_stream(job.input_path):
            message += " [warning: audio not preserved]"
```

Change the existing finished emit to pass `message=message` instead of re-formatting inline (import `has_audio_stream` at module import time only if the import cycle allows; otherwise move all three imports into the helper and have the helper return a tri-state).

- [ ] **Step 6.5: Repeat in `_run_video_recurrent` before its finished emit**

Same block, using `n_frames` in the message; the cut-window arguments are identical.

- [ ] **Step 6.6: Verify**

```powershell
& $py -m pytest apps/anime_upscaler_gui/tests/test_video_audio.py apps/anime_upscaler_gui/tests/test_ffmpeg_audio.py -q -p no:cacheprovider "--basetemp=$basetemp"
& $py -m pytest apps/anime_upscaler_gui/tests -q -p no:cacheprovider "--basetemp=$basetemp"
```

Expected: new audio tests pass; suite count is baseline + 4 (388 -> 392), no new failures.

- [ ] **Step 6.7: Commit**

```powershell
git add apps/anime_upscaler_gui/anime_upscaler_gui/pipeline/worker.py apps/anime_upscaler_gui/tests/test_video_audio.py
git commit -m "fix(gui): remux source audio into every encoded video; warn instead of shipping silence"
```

## Task 7: Never report a failed encode as finished

**Files:**
- Modify: `apps/anime_upscaler_gui/anime_upscaler_gui/pipeline/worker.py` (per-frame finally ~724-736 and finished emit ~740; recurrent finally ~945-966)
- Create: `apps/anime_upscaler_gui/tests/test_encode_failure.py`

- [ ] **Step 7.1: Write the failing test against the encoder contract**

```python
"""A failed ffmpeg encode must never produce a 'finished' event."""
from __future__ import annotations

import shutil

import pytest

from anime_upscaler_gui.ffmpeg import open_encoder

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                reason="ffmpeg required")


def test_ffmpeg_rejects_mismatched_container(tmp_path):
    # .webm cannot take libx264; ffmpeg exits non-zero. This pins the
    # behaviour that the worker must inspect instead of ignoring.
    pipe = open_encoder(tmp_path / "bad.webm", fps=10.0, w=64, h=64)
    assert pipe is not None
    import numpy as np
    for _ in range(3):
        try:
            pipe.stdin.write(np.zeros((64, 64, 3), dtype=np.uint8).tobytes())
        except (BrokenPipeError, OSError):
            break
    pipe.stdin.close()
    rc = pipe.wait(timeout=30)
    assert rc != 0
```

- [ ] **Step 7.2: Run it**

```powershell
& $py -m pytest apps/anime_upscaler_gui/tests/test_encode_failure.py -q -p no:cacheprovider "--basetemp=$basetemp"
```

Expected: 1 passed (pins ffmpeg behaviour).

- [ ] **Step 7.3: Add the shared post-encode gate to worker.py**

Insert next to `_unlink_partial_output`:

```python
def _handle_encode_result(worker, job, out_path, encode_rc, frames) -> bool:
    """Shared post-encode gate: True only when the output may be reported."""
    if encode_rc != 0:
        _unlink_partial_output(out_path)
        worker.emit(JobEvent(kind="error", job_id=job.job_id,
                             message=f"encoder failed (exit {encode_rc}); output discarded"))
        return False
    if frames == 0 or not out_path.exists() or out_path.stat().st_size == 0:
        _unlink_partial_output(out_path)
        worker.emit(JobEvent(kind="error", job_id=job.job_id,
                             message="encoder produced no usable output"))
        return False
    return True
```

- [ ] **Step 7.4: Capture the encoder exit code in `_run_video`**

Replace the current finally block that closes stdin/waits/kills with:

```python
        encode_rc = 0
        try:
            if use_pipe:
                pipe.stdin.close()
                try:
                    encode_rc = pipe.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pipe.kill()
                    encode_rc = pipe.wait()
            else:
                writer.release()
        finally:
            sync_reader.release()
```

Immediately after the try/finally (before the audio block) add:

```python
        if not _handle_encode_result(self, job, out_path, encode_rc, proc):
            return
```

- [ ] **Step 7.5: Mirror the check in `_run_video_recurrent`**

In its cleanup block capture `encode_rc` the same way; after cleanup call `_handle_encode_result(self, job, out_path, encode_rc, n_frames)` and return when it is False, before the finished emit.

- [ ] **Step 7.6: Add the decision regression test**

Append to `apps/anime_upscaler_gui/tests/test_encode_failure.py`:

```python
def test_failed_encode_emits_error_not_finished(tmp_path):
    from anime_upscaler_gui.pipeline import worker as worker_mod

    events = []

    class FakeWorker:
        def emit(self, evt):
            events.append(evt)

    job = type("J", (), {"job_id": 1})()
    worker_mod._handle_encode_result(FakeWorker(), job, tmp_path / "x.mp4", 0, 1)
    assert [e.kind for e in events] == ["error"]
    assert not (tmp_path / "x.mp4").exists()
```

- [ ] **Step 7.7: Verify**

```powershell
& $py -m pytest apps/anime_upscaler_gui/tests/test_encode_failure.py apps/anime_upscaler_gui/tests -q -p no:cacheprovider "--basetemp=$basetemp"
```

Expected: all pass; suite count grows by 2.

- [ ] **Step 7.8: Commit**

```powershell
git add apps/anime_upscaler_gui/anime_upscaler_gui/pipeline/worker.py apps/anime_upscaler_gui/tests/test_encode_failure.py
git commit -m "fix(gui): check ffmpeg exit code; failed encodes become ERROR with no partial file"
```

## Task 8: Never write over the source

**Files:**
- Modify: `apps/anime_upscaler_gui/anime_upscaler_gui/controllers/job_builder.py:32-58`
- Modify: `apps/anime_upscaler_gui/anime_upscaler_gui/app.py:406-420` and the enqueue call site ~678
- Modify: `apps/anime_upscaler_gui/anime_upscaler_gui/pipeline/worker.py` (`_run_one`, after the input-exists check ~326)
- Create: `apps/anime_upscaler_gui/tests/test_output_guard.py`

- [ ] **Step 8.1: Write the failing tests**

```python
"""The output path must never resolve to the input file."""
from __future__ import annotations

from pathlib import Path

import pytest

from anime_upscaler_gui.controllers.job_builder import (
    compute_output_path,
    validate_output_path,
)


def test_same_folder_and_empty_suffix_is_rejected(tmp_path):
    src = tmp_path / "ep01.mp4"
    out = compute_output_path(src, "custom", "", "batch", str(tmp_path))
    assert out == src
    with pytest.raises(ValueError, match="equals the input"):
        validate_output_path(src, out)


def test_normal_custom_output_is_allowed(tmp_path):
    src = tmp_path / "ep01.mp4"
    out = compute_output_path(src, "custom", "_x4", "batch", str(tmp_path / "out"))
    validate_output_path(src, out)  # must not raise


def test_empty_custom_folder_is_rejected(tmp_path):
    src = tmp_path / "ep01.mp4"
    with pytest.raises(ValueError, match="output folder"):
        validate_output_path(src, Path(""))
```

- [ ] **Step 8.2: Run to verify failure**

```powershell
& $py -m pytest apps/anime_upscaler_gui/tests/test_output_guard.py -q -p no:cacheprovider "--basetemp=$basetemp"
```

Expected: ImportError: cannot import name 'validate_output_path'.

- [ ] **Step 8.3: Implement the guard in job_builder.py**

Append after `compute_output_path`:

```python
def validate_output_path(input_path: Path, output_path: Path) -> None:
    """Raise ValueError when the destination would overwrite the input.

    Raises for an empty output folder (Path('') is the process CWD, never
    what the user meant) and for any destination that resolves to the input
    file. resolve() is used so '..', mixed separators and symlinks cannot
    sneak past the check.
    """
    if str(output_path).strip() in ("", "."):
        raise ValueError("output folder is empty; choose a folder before starting")
    try:
        same = output_path.resolve() == input_path.resolve()
    except OSError:
        same = str(output_path.absolute()).lower() == str(input_path.absolute()).lower()
    if same:
        raise ValueError(
            f"output path equals the input file ({input_path}); "
            "change the suffix or choose a different output folder")
```

- [ ] **Step 8.4: Read live widget values and validate in app.py**

Replace `_compute_output_path` (lines 406-420) with:

```python
    def _compute_output_path(self, input_path: Path) -> Path:
        """Build the destination from *live* widget state.

        The output-mode radios and the suffix entry have no trace, so
        settings.data only changes on Save; building from settings sends the
        next file to the previous destination.
        """
        return _compute_output_path_ctrl(
            input_path,
            output_mode=self.output_panel.output_mode_var.get(),
            single_file_suffix=self.output_panel.suffix_var.get(),
            batch_folder_name=self.output_panel.batch_folder_var.get(),
            custom_output_dir=self.output_panel.output_dir_var.get(),
        )
```

Add one helper next to it:

```python
    def _compute_output_path_or_none(self, input_path: Path):
        from .controllers.job_builder import validate_output_path
        try:
            out = self._compute_output_path(input_path)
            validate_output_path(input_path, out)
            return out
        except ValueError as e:
            messagebox.showerror("Output folder", str(e))
            return None
```

At the enqueue site (~678) use `out = self._compute_output_path_or_none(self.input_path)` and `return` when it is None. Queue persistence/reload sites keep the unvalidated builder; the worker re-validates at run time.

- [ ] **Step 8.5: Add the worker defence in `_run_one`**

After the input-exists check insert:

```python
        try:
            same = job.output_path.resolve() == job.input_path.resolve()
        except OSError:
            same = str(job.output_path.absolute()).lower() == str(job.input_path.absolute()).lower()
        if same:
            self.emit(JobEvent(kind="error", job_id=job.job_id,
                               message="output path equals input path; refusing to overwrite the source"))
            return
```

- [ ] **Step 8.6: Verify**

```powershell
& $py -m pytest apps/anime_upscaler_gui/tests/test_output_guard.py apps/anime_upscaler_gui/tests -q -p no:cacheprovider "--basetemp=$basetemp"
```

Expected: all pass; suite count grows by 3.

- [ ] **Step 8.7: Commit**

```powershell
git add apps/anime_upscaler_gui/anime_upscaler_gui/controllers/job_builder.py apps/anime_upscaler_gui/anime_upscaler_gui/app.py apps/anime_upscaler_gui/anime_upscaler_gui/pipeline/worker.py apps/anime_upscaler_gui/tests/test_output_guard.py
git commit -m "fix(gui): refuse output paths that overwrite the input; read live output widgets"
```

## Task 9: Make "Abort job" report a cancel, not a crash

**Files:**
- Modify: `apps/anime_upscaler_gui/anime_upscaler_gui/pipeline/worker.py:687-693`
- Test: `apps/anime_upscaler_gui/tests/test_video_pipeline_fixes.py` (extend)

- [ ] **Step 9.1: Read the existing abort/frame-error tests**

```powershell
Select-String -Path apps/anime_upscaler_gui/tests/test_video_pipeline_fixes.py -Pattern "ABORT_JOB|on_frame_error|_JobCancelled" | Select-Object -First 20
```

- [ ] **Step 9.2: Replace the re-raise with the cancellation exception**

At the ABORT_JOB branch replace the body with:

```python
                    if action == ABORT_JOB:
                        # "Abort job" is a user cancel. Drain queued controls
                        # first (the GUI may re-queue a cancel behind the
                        # modal), then raise the cancellation type so _run_one
                        # emits a clean cancelled and unlinks the partial.
                        self._drain_ctl_for_job(job.job_id)
                        raise _JobCancelled()
```

- [ ] **Step 9.3: Add a regression test**

Append to `apps/anime_upscaler_gui/tests/test_video_pipeline_fixes.py`:

```python
def test_abort_job_is_a_cancellation_not_a_crash():
    """ABORT_JOB must map to _JobCancelled so the output is removed cleanly."""
    import inspect
    from anime_upscaler_gui.pipeline import worker as worker_mod
    src = inspect.getsource(worker_mod.PipelineWorker._run_video)
    assert "raise _JobCancelled()" in src
    assert "raise e" not in src
```

Prefer a behavioural test if the existing fake-backend fixture makes it cheap: queue a job whose on_frame_error returns ABORT_JOB, assert the terminal event kind is cancelled and the output file is gone.

- [ ] **Step 9.4: Verify**

```powershell
& $py -m pytest apps/anime_upscaler_gui/tests/test_video_pipeline_fixes.py apps/anime_upscaler_gui/tests/test_queue_controls.py -q -p no:cacheprovider "--basetemp=$basetemp"
& $py -m pytest apps/anime_upscaler_gui/tests -q -p no:cacheprovider "--basetemp=$basetemp"
```

Expected: all pass, no regressions.

- [ ] **Step 9.5: Commit**

```powershell
git add apps/anime_upscaler_gui/anime_upscaler_gui/pipeline/worker.py apps/anime_upscaler_gui/tests/test_video_pipeline_fixes.py
git commit -m "fix(gui): Abort job raises _JobCancelled (clean cancel + partial cleanup)"
```

## Task 10: Pause must work after the first resume; output settings must be live

**Files:**
- Modify: `apps/anime_upscaler_gui/anime_upscaler_gui/pipeline/worker.py:252-262`
- Modify: `apps/anime_upscaler_gui/anime_upscaler_gui/widgets/output_panel.py:19-43`
- Modify: `apps/anime_upscaler_gui/anime_upscaler_gui/app.py`
- Create: `apps/anime_upscaler_gui/tests/test_pause_state.py`

- [ ] **Step 10.1: Write the failing test**

```python
"""Pause state machine: pause -> resume -> pause must block again."""
from __future__ import annotations

import queue

from anime_upscaler_gui.pipeline.worker import PipelineWorker


def _worker_stub():
    w = object.__new__(PipelineWorker)
    w._pause_events = {}
    w.ctl_queue = queue.Queue()
    w.emit = lambda evt: None
    return w


def test_second_pause_clears_the_event():
    from anime_upscaler_gui.pipeline import worker as worker_mod
    # Use whatever control-event type _drain_ctl_for_job consumes; grep
    # "class .*Control" in pipeline/jobs.py if JobControl is not the name.
    from anime_upscaler_gui.pipeline.jobs import JobControl
    w = _worker_stub()
    for kind in ("pause", "resume", "pause"):
        w.ctl_queue.put(JobControl(job_id=7, kind=kind))
        w._drain_ctl_for_job(7)
    assert w._pause_events[7].is_set() is False
```

- [ ] **Step 10.2: Run to verify failure**

```powershell
& $py -m pytest apps/anime_upscaler_gui/tests/test_pause_state.py -q -p no:cacheprovider "--basetemp=$basetemp"
```

Expected: AssertionError: assert True is False.

- [ ] **Step 10.3: Clear the event on pause**

Replace the pause branch with:

```python
            if evt.kind == "pause":
                # A pause must block: clear the event set by the last resume.
                ev = self._pause_events.setdefault(job_id, threading.Event())
                ev.clear()
                self.emit(JobEvent(kind="log", job_id=job_id,
                                   message="paused by user"))
```

- [ ] **Step 10.4: Make the output widgets live**

In `output_panel.py` bind traces after the widgets exist:

```python
        for var in (self.output_mode_var, self.suffix_var,
                    self.batch_folder_var, self.output_dir_var):
            var.trace_add("write", lambda *_: app._on_output_setting_changed())
```

Add to `app.py` next to `_compute_output_path`:

```python
    def _on_output_setting_changed(self):
        """Persist output widgets immediately so Start never uses stale state."""
        if getattr(self, "_restoring_settings", False):
            return
        data = self.settings.data
        data.output_mode = self.output_panel.output_mode_var.get()
        data.single_file_suffix = self.output_panel.suffix_var.get()
        data.batch_folder_name = self.output_panel.batch_folder_var.get()
        data.output_dir = self.output_panel.output_dir_var.get()
```

If `_restoring_settings` does not exist, add `self._restoring_settings = False` in __init__ and set it around the settings-restore calls.

- [ ] **Step 10.5: Verify**

```powershell
& $py -m pytest apps/anime_upscaler_gui/tests/test_pause_state.py apps/anime_upscaler_gui/tests -q -p no:cacheprovider "--basetemp=$basetemp"
```

Expected: all pass; count grows by 1.

- [ ] **Step 10.6: Commit**

```powershell
git add apps/anime_upscaler_gui/anime_upscaler_gui/pipeline/worker.py apps/anime_upscaler_gui/anime_upscaler_gui/widgets/output_panel.py apps/anime_upscaler_gui/anime_upscaler_gui/app.py apps/anime_upscaler_gui/tests/test_pause_state.py
git commit -m "fix(gui): pause works after resume; output settings persist live"
```

## Task 11: Tiling must not crash on small images

**Files:**
- Modify: `apps/anime_upscaler_gui/anime_upscaler_gui/tiling.py:40-81`
- Test: `apps/anime_upscaler_gui/tests/test_tiling.py` (extend)

- [ ] **Step 11.1: Add the failing tests**

```python
import torch
import torch.nn as nn

from anime_upscaler_gui.tiling import _TileProcessor


class _Identity(nn.Module):
    def forward(self, x):
        return x


def test_image_smaller_than_tile_does_not_crash():
    proc = _TileProcessor(_Identity(), tile_size=64, overlap=16)
    out = proc.forward(torch.rand(1, 3, 32, 24))
    assert out.shape == (1, 3, 32, 24)


def test_non_multiple_size_is_exact():
    proc = _TileProcessor(_Identity(), tile_size=64, overlap=16)
    out = proc.forward(torch.rand(1, 3, 100, 130))
    assert out.shape == (1, 3, 100, 130)
```

- [ ] **Step 11.2: Run to verify failure**

```powershell
& $py -m pytest apps/anime_upscaler_gui/tests/test_tiling.py -q -p no:cacheprovider "--basetemp=$basetemp"
```

Expected: one or both new tests fail (pad error or shape mismatch).

- [ ] **Step 11.3: Replace the tile loop**

Replace `_TileProcessor.forward` with:

```python
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (1, 3, H, W) -> (1, 3, H*scale, W*scale)."""
        _, c, h, w = x.shape
        scale = self._infer_scale(x)
        out_h, out_w = h * scale, w * scale
        device, dtype = x.device, x.dtype
        out = torch.zeros(1, c, out_h, out_w, device=device, dtype=dtype)
        weight = torch.zeros(1, 1, out_h, out_w, device=device, dtype=dtype)

        stride = self.tile - 2 * self.overlap
        if stride <= 0:
            raise ValueError("overlap too large for tile_size")

        win = _raised_cosine_window(self.tile * scale, self.overlap * scale,
                                    device, dtype)

        def positions(size: int) -> list[int]:
            if size <= self.tile:
                return [0]
            pos = list(range(0, size - self.tile + 1, stride))
            if pos[-1] != size - self.tile:
                pos.append(size - self.tile)
            return pos

        for ty in positions(h):
            for tx in positions(w):
                tile = x[:, :, ty:ty + self.tile, tx:tx + self.tile]
                pad_h = self.tile - tile.shape[-2]
                pad_w = self.tile - tile.shape[-1]
                if pad_h or pad_w:
                    # replicate is defined even when pad >= dimension;
                    # reflect raises in torch for small images.
                    tile = F.pad(tile, (0, pad_w, 0, pad_h), mode="replicate")
                y_tile = self.model(tile) * win
                out[:, :, ty * scale:ty * scale + self.tile * scale,
                    tx * scale:tx * scale + self.tile * scale] += y_tile
                weight[:, :, ty * scale:ty * scale + self.tile * scale,
                       tx * scale:tx * scale + self.tile * scale] += win
        return out / weight.clamp(min=1e-6)
```

Replace `_infer_scale` so it fails loudly instead of silently returning 4:

```python
    def _infer_scale(self, x: torch.Tensor) -> int:
        """Probe the model to learn its upscaling factor."""
        probe_h = min(x.shape[-2], self.tile)
        probe_w = min(x.shape[-1], self.tile)
        with torch.no_grad():
            probe = self.model(x[:, :, :probe_h, :probe_w])
        ratio = probe.shape[-2] // probe_h
        if ratio <= 0:
            raise ValueError(
                f"model returned a smaller image ({tuple(probe.shape)}) than its input")
        return int(ratio)
```

- [ ] **Step 11.4: Verify**

```powershell
& $py -m pytest apps/anime_upscaler_gui/tests/test_tiling.py apps/anime_upscaler_gui/tests -q -p no:cacheprovider "--basetemp=$basetemp"
```

Expected: all tiling tests pass, no regressions.

- [ ] **Step 11.5: Commit**

```powershell
git add apps/anime_upscaler_gui/anime_upscaler_gui/tiling.py apps/anime_upscaler_gui/tests/test_tiling.py
git commit -m "fix(tiling): clamp tile positions and replicate-pad; fail loudly on scale probe"
```

---

# Track 1B - Packaging and release truth (Tasks 12-20)

## Task 12: Bundle the model closure into the GUI package

**Files:**
- Create: `scripts/sync_vendored_models.py`
- Create: `apps/anime_upscaler_gui/anime_upscaler_gui/_models/` (generated tree)
- Modify: `apps/anime_upscaler_gui/anime_upscaler_gui/arch_registry.py:22-25`
- Create: `apps/anime_upscaler_gui/tests/test_vendored_models.py`

The measured closure is exactly seven files, torch-only: `students/__init__.py`, `students/mambair.py`, `span/neosr_span.py`, `span/mambair_v2.py` (with pure-PyTorch fallback), `eranet.py`, `msrsr.py`, `teachers/spandrel.py`.

- [ ] **Step 12.1: Write the sync script**

Create `scripts/sync_vendored_models.py`:

```python
"""Regenerate the GUI's bundled model code from src/anime_sr/models.

The GUI wheel must not import anime_sr. These files are copied with the
import rewrites listed in REWRITES; the GUI test suite compares the vendored
bytes against this transform, so upstream drift fails CI until re-synced.

Run: python scripts/sync_vendored_models.py
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "anime_sr" / "models"
DST = REPO / "apps" / "anime_upscaler_gui" / "anime_upscaler_gui" / "_models"

FILES = {
    "students/__init__.py": "students/__init__.py",
    "students/mambair.py": "students/mambair.py",
    "span/neosr_span.py": "span/neosr_span.py",
    "span/mambair_v2.py": "span/mambair_v2.py",
    "eranet.py": "eranet.py",
    "msrsr.py": "msrsr.py",
    "teachers/spandrel.py": "teachers/spandrel.py",
}

REWRITES = [
    ("from anime_sr.models.span.mambair_v2", "from .mambair_v2"),
    ("from anime_sr.models.span.neosr_span", "from ..span.neosr_span"),
]


def transform(text: str) -> str:
    for old, new in REWRITES:
        text = text.replace(old, new)
    return text


def main() -> int:
    for rel in FILES:
        (DST / rel).parent.mkdir(parents=True, exist_ok=True)
    for pkg in ("", "students", "span", "teachers"):
        init = (DST / pkg / "__init__.py") if pkg else (DST / "__init__.py")
        init.parent.mkdir(parents=True, exist_ok=True)
        if not init.exists():
            init.write_text("", encoding="utf-8", newline="\n")
    for dst_rel, src_rel in FILES.items():
        src = SRC / src_rel
        text = transform(src.read_text(encoding="utf-8"))
        (DST / dst_rel).write_text(text, encoding="utf-8", newline="\n")
    print(f"synced {len(FILES)} model files into {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 12.2: Run it and rewrite the registry imports**

```powershell
& $py scripts/sync_vendored_models.py
Get-ChildItem apps/anime_upscaler_gui/anime_upscaler_gui/_models -Recurse -Filter *.py | Measure-Object -Line
```

Expected: 7 files synced; the listing shows the 7 modules plus package init files.

In `arch_registry.py` replace the four top-level imports (lines 22-25) with:

```python
from ._models.students import RFDN, TinySRVGGStudent
from ._models.span.neosr_span import NeosrSPAN
from ._models.eranet import ERANet
from ._models.msrsr import MSRSWVSR
```

Then find any other anime_sr import in the GUI package:

```powershell
Get-ChildItem apps/anime_upscaler_gui/anime_upscaler_gui -Recurse -Filter *.py |
    Select-String -Pattern "^from anime_sr|^import anime_sr" | Select-Object -First 50
```

Rewrite each to the matching `._models` path; the packaged closure is complete, so no other anime_sr code may be imported.

- [ ] **Step 12.3: Write the drift and clean-import tests**

Create `apps/anime_upscaler_gui/tests/test_vendored_models.py`:

```python
"""The bundled model tree must be generated, in sync, and self-contained."""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
GUI = Path(__file__).resolve().parents[1] / "anime_upscaler_gui"
SCRIPT = REPO / "scripts" / "sync_vendored_models.py"

spec = importlib.util.spec_from_file_location("sync_vendored_models", SCRIPT)
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


def test_vendored_models_match_upstream_transform():
    problems = []
    for dst_rel, src_rel in sync.FILES.items():
        expected = sync.transform((sync.SRC / src_rel).read_text(encoding="utf-8"))
        actual = (sync.DST / dst_rel).read_text(encoding="utf-8")
        if expected != actual:
            problems.append(f"{dst_rel} differs from src/anime_sr/models/{src_rel}")
    assert not problems, (
        "vendored models drifted; run python scripts/sync_vendored_models.py\n"
        + "\n".join(problems))


def test_gui_package_does_not_import_anime_sr():
    offenders = []
    for path in GUI.rglob("*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("import anime_sr", "from anime_sr")):
                offenders.append(f"{path.relative_to(GUI)}:{number}: {stripped}")
    assert not offenders, "GUI must be standalone:\n" + "\n".join(offenders)
```

- [ ] **Step 12.4: Verify, including the clean-room guard from Task 4**

```powershell
& $py -m pytest apps/anime_upscaler_gui/tests/test_vendored_models.py -q -p no:cacheprovider "--basetemp=$basetemp"
& $py -m pytest tests/release -q -p no:cacheprovider "--basetemp=$basetemp"
& $py -m pytest apps/anime_upscaler_gui/tests -q -p no:cacheprovider "--basetemp=$basetemp"
```

Expected: vendored tests pass; the clean-room import guard from Task 4 now passes (the wheel-build guard stays green); GUI suite unchanged.

- [ ] **Step 12.5: Commit**

```powershell
git add scripts/sync_vendored_models.py apps/anime_upscaler_gui/anime_upscaler_gui/_models apps/anime_upscaler_gui/anime_upscaler_gui/arch_registry.py apps/anime_upscaler_gui/tests/test_vendored_models.py
git commit -m "feat(gui): bundle the torch-only model closure; sync script + drift test"
```

## Task 13: Fix the manifests so both wheels build and install

**Files:**
- Modify: `pyproject.toml:19` (delete superseded classifier)
- Modify: `pyproject.toml:26-47` (relax pinning)
- Modify: `apps/anime_upscaler_gui/pyproject.toml` (pyav to av)
- Modify: `apps/anime_upscaler_gui/requirements.txt:11` (same)
- Verify: `tests/release/test_artifacts.py`, GUI packaging parity test

- [ ] **Step 13.1: Fix the root build blocker**

Delete the line `"License :: OSI Approved :: MIT License",` from `pyproject.toml` (line 19). PEP 639 supersedes classifiers whenever `license-files` is present; setuptools 77 errors otherwise.

- [ ] **Step 13.2: Relax the exact pins**

In `[project.dependencies]` replace each compatible-release pin with its lower bound, keeping the same number. Examples:

```toml
torch>=2.4.0
torchvision>=0.19.0
numpy>=1.26.0
av>=17.0
```

Do this for every entry that uses a tilde-equals pin. Rationale (measured): the installed environment has torch 2.12, numpy 2.4 and av 17.1, so the cap numpy~=1.26.0 makes the root package uninstallable next to the GUI floor numpy>=2.4; pip fails with ResolutionImpossible. After relaxing, all 20 requirements are satisfied by the installed interpreter.

- [ ] **Step 13.3: Fix the GUI dependency name**

In `apps/anime_upscaler_gui/pyproject.toml` and `apps/anime_upscaler_gui/requirements.txt` replace `pyav>=17.1` with `av>=17.1`. pyav does not exist on PyPI (404); the actual distribution is av (17.1.0 installed). The parity test compares the two files, so edit both in the same commit.

- [ ] **Step 13.4: Re-verify the manifests end to end**

```powershell
Remove-Item dist -Recurse -Force -ErrorAction SilentlyContinue
& $py -m pip wheel . --no-deps -w dist
& $py -m pip wheel apps/anime_upscaler_gui --no-deps -w dist
Get-ChildItem dist | Select-Object Name,Length
& $py -m pip install -e . --no-deps
& $py -m pip install -e apps/anime_upscaler_gui --no-deps
& $py -m pip check
```

Expected: both wheels built (root ~0.4 MB, GUI ~3.2 MB after bundling); pip check exits 0. If pip check reports a remaining conflict, adjust the floor of the offending requirement (never re-add a cap).

```powershell
& $py -m pytest tests/release apps/anime_upscaler_gui/tests/test_packaging.py -q -p no:cacheprovider "--basetemp=$basetemp"
```

Expected: release guards pass (2 passed), packaging parity passes.

- [ ] **Step 13.5: Commit**

```powershell
git add pyproject.toml apps/anime_upscaler_gui/pyproject.toml apps/anime_upscaler_gui/requirements.txt
git commit -m "build: PEP 639 license, relaxed dependency floors, av instead of pyav"
```

## Task 14: Load checkpoints safely

**Files:**
- Modify: `apps/anime_upscaler_gui/anime_upscaler_gui/registry.py:395-457` (scan) and `:515-549` (download hash)
- Modify: `apps/anime_upscaler_gui/anime_upscaler_gui/arch_registry.py:31-44`
- Create: `apps/anime_upscaler_gui/tests/test_safe_load.py`

- [ ] **Step 14.1: Write the failing tests**

```python
"""Checkpoint loading: scanning must never execute pickled code."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch


def test_scan_does_not_execute_a_pickled_payload(tmp_path):
    marker = tmp_path / "pwned.txt"

    class Evil:
        def __reduce__(self):
            return (Path.write_text, (marker, "pwned"))

    ckpt = tmp_path / "evil.pth"
    torch.save(Evil(), ckpt)

    from anime_upscaler_gui import registry as registry_mod
    # Build the registry the same way the existing registry tests do
    # (grep "ModelRegistry(" in apps/anime_upscaler_gui/tests), then point
    # its scan roots at tmp_path.
    reg = registry_mod.ModelRegistry.__new__(registry_mod.ModelRegistry)
    reg._roots = [ckpt.parent]
    reg.scan_installed()

    assert not marker.exists(), "torch.load executed pickled code during scan"


def test_arch_load_rejects_unsafe_state_dicts(tmp_path):
    from anime_upscaler_gui import arch_registry as arch

    bad = tmp_path / "bad.pth"
    torch.save({"definitely": "not a state dict"}, bad)
    with pytest.raises(Exception):
        arch.load_model_weights(bad, architecture="rfdn", scale=4)
```

Adapt names to the real public functions (scan_installed is a method; the arch loader name is whatever arch_registry.py exposes). The assertions are the requirement: marker absent, unsafe payload rejected.

- [ ] **Step 14.2: Run to observe the vulnerability**

```powershell
& $py -m pytest apps/anime_upscaler_gui/tests/test_safe_load.py -q -p no:cacheprovider "--basetemp=$basetemp"
```

Expected before the fix: the first test fails because the marker exists (the scan fell back to weights_only=False).

- [ ] **Step 14.3: Make scanning conservative**

In registry.py line 404 change the load to:

```python
            ck = torch.load(p, map_location="cpu", weights_only=True)
```

In the except handler at lines 446-447 replace the bare pass with a recorded, visible skip:

```python
        except Exception as e:  # unreadable or unsafe checkpoint: skip, never execute
            skipped = getattr(self, "_skipped", None)
            if skipped is not None:
                skipped.append((str(p), str(e)[:200]))
            continue
```

Add `self._skipped = []` at the start of `scan_installed` so the GUI can display unloadable files instead of hiding them.

- [ ] **Step 14.4: Remove the silent unsafe fallback from the architecture loader**

In arch_registry.py replace the _load_state_dict fallback with an explicit opt-in:

```python
def _load_state_dict(path, *, allow_unsafe_pickle: bool | None = None):
    if allow_unsafe_pickle is None:
        allow_unsafe_pickle = os.environ.get("ANIME_UPSCALER_ALLOW_UNSAFE_PICKLE") == "1"
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as safe_error:
        if not allow_unsafe_pickle:
            raise RuntimeError(
                f"refusing to load {path}: not a plain tensor checkpoint "
                f"(set ANIME_UPSCALER_ALLOW_UNSAFE_PICKLE=1 only for files you trust): "
                f"{safe_error}"
            ) from safe_error
        return torch.load(path, map_location="cpu", weights_only=False)
```

Also require a declared sha256 for every preset that lacks one (the shipped RealESRGAN_x4plus_anime_6B preset declares none): when no hash is declared the download may proceed but the model must be marked unverified in the UI.

- [ ] **Step 14.5: Verify**

```powershell
& $py -m pytest apps/anime_upscaler_gui/tests/test_safe_load.py apps/anime_upscaler_gui/tests -q -p no:cacheprovider "--basetemp=$basetemp"
```

Expected: both new tests pass, no regressions; the marker file is never created.

- [ ] **Step 14.6: Commit**

```powershell
git add apps/anime_upscaler_gui/anime_upscaler_gui/registry.py apps/anime_upscaler_gui/anime_upscaler_gui/arch_registry.py apps/anime_upscaler_gui/tests/test_safe_load.py
git commit -m "fix(gui): weights_only checkpoint loading; unsafe pickle requires explicit opt-in"
```

## Task 15: Retire the broken legacy packager from the release path

**Files:**
- Modify: `scripts/package_project.py` (deprecation banner)
- Create: `scripts/legacy/README.md`
- Create: `tests/release/test_no_legacy_packager.py`

- [ ] **Step 15.1: Add the deprecation marker**

At the top of scripts/package_project.py insert:

```python
"""DEPRECATED - do not use for releases.

This script pre-dates the standard build path and is known broken: it does
not stage README/LICENSE into the wheel tree and its generated setup.py
declares entry points for modules that are not installed. Releases use
"python -m build" against the two pyproject.toml files instead.
Full repair or deletion is Task 27 (post-release).
"""
import sys as _sys
print("WARNING: scripts/package_project.py is deprecated; use python -m build",
      file=_sys.stderr)
```

- [ ] **Step 15.2: Add the guard test**

Create `tests/release/test_no_legacy_packager.py`:

```python
"""Release docs and CI must not point at the deprecated packager."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BAD = "package_project.py"


def _texts():
    for pattern in ("*.md", "docs/**/*.md", ".github/workflows/*.yml"):
        for path in REPO.glob(pattern):
            yield path, path.read_text(encoding="utf-8", errors="replace")


def test_no_release_docs_recommend_the_legacy_packager():
    offenders = []
    for path, text in _texts():
        for number, line in enumerate(text.splitlines(), 1):
            if BAD in line and "DEPRECATED" not in line:
                offenders.append(str(path.relative_to(REPO)) + ":" + str(number))
    assert not offenders, "remove legacy packager references:\n" + "\n".join(offenders)
```

- [ ] **Step 15.3: Verify and commit**

```powershell
& $py -m pytest tests/release/test_no_legacy_packager.py -q -p no:cacheprovider "--basetemp=$basetemp"
git add scripts/package_project.py scripts/legacy/README.md tests/release/test_no_legacy_packager.py
git commit -m "chore: deprecate scripts/package_project.py; release builds use python -m build"
```

Expected: 1 passed (if docs still reference the script, fix those docs in this step rather than relaxing the test).

## Task 16: Make CI tell the truth about this branch

**Files:**
- Modify: `.github/workflows/ci.yml` (429 lines)

- [ ] **Step 16.1: Fix the triggers**

Replace the `on:` block so push events cover the working branches (currently only main and develop, which never matched refactor/restructure or release/gui-preview):

```yaml
on:
  push:
    branches: [main, develop, "release/**", "refactor/**"]
    tags: ["v*"]
  pull_request:
    branches: [main]
  workflow_dispatch:
```

- [ ] **Step 16.2: Split blocking from advisory checks**

In the test job:
- The blocking lint gate becomes `flake8 . --select=E9 --exclude=.venv,data,runs` (syntax errors only; the 17 F821 undefined names and the style backlog are Track 2 Task 27 work).
- Mark the full `flake8 .`, `mypy src/anime_sr` and `pytest tests` steps `continue-on-error: true` with a comment naming Task 27. Do not delete them; they stay visible advisory signal.
- Python versions: 3.12 only (requires-python is >=3.12; a 3.10 leg can never pass).

- [ ] **Step 16.3: Repair the GUI job**

Replace the install line `pip install -e ".[dev,gui]"` (no such extra exists) with, in order:
- `pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu`
- `pip install -e ".[dev]"`
- `pip install -r apps/anime_upscaler_gui/requirements.txt`

Remove the stale torch==2.4.0 pins; the GUI declares torch>=2.12. Keep the existing xvfb wrapper for the Tk suite and add `tests/release` to the pytest invocation so the clean-room and drift guards run in CI.

- [ ] **Step 16.4: Verify the workflow parses**

```powershell
& $py -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml', encoding='utf-8')); print('ci.yml parses')"
```

Expected: `ci.yml parses`. (Actionlint is not installed locally; the YAML parse plus a real push is the evidence.)

- [ ] **Step 16.5: Commit and push the branch so CI actually runs**

```powershell
git add .github/workflows/ci.yml
git commit -m "ci: trigger working branches, block only syntax errors, fix GUI install line"
git push -u origin release/gui-preview
gh run list --branch release/gui-preview --limit 5
```

Expected: a run appears for the branch. Record the run URL and job outcomes; if the GUI job fails for environment reasons, fix the job (not the tests) before Task 17.

## Task 17: Clean-room rehearsal and --self-check

**Files:**
- Modify: `apps/anime_upscaler_gui/anime_upscaler_gui/__main__.py`
- Create: `scripts/release/rehearse_windows.ps1`
- Create: `apps/anime_upscaler_gui/tests/test_self_check.py`

- [ ] **Step 17.1: Add --self-check to the GUI entry point**

In `__main__.py`, before Tk is created, handle the flag:

```python
import json
import shutil
import sys


def _self_check() -> int:
    from . import __version__
    from .arch_registry import available_architectures

    report = {
        "version": __version__,
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ffprobe": shutil.which("ffprobe") is not None,
        "architectures": sorted(available_architectures()),
    }
    print(json.dumps(report, indent=2))
    return 0 if report["ffmpeg"] and report["ffprobe"] else 1


def main() -> int:
    if "--self-check" in sys.argv[1:]:
        return _self_check()
    ...  # existing Tk start-up
```

Adapt the name to the existing main() signature; the requirement is: the console script exits 0 on a machine with ffmpeg and prints the JSON report, including when no anime_sr is installed (Task 12).

- [ ] **Step 17.2: Write the test**

Create `apps/anime_upscaler_gui/tests/test_self_check.py`:

```python
"""The installed console script can prove its own environment."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_self_check_reports_environment():
    r = subprocess.run(
        [sys.executable, "-m", "anime_upscaler_gui", "--self-check"],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    report = json.loads(r.stdout)
    assert report["version"]
    assert isinstance(report["architectures"], list) and report["architectures"]
    assert report["ffmpeg"] is True
```

- [ ] **Step 17.3: Write the rehearsal script**

Create `scripts/release/rehearse_windows.ps1`:

```powershell
param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../")).Path
)
$ErrorActionPreference = "Stop"
$work = Join-Path $env:TEMP ("anime-sr-rehearsal-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $work | Out-Null
$wheel = Get-ChildItem (Join-Path $RepoRoot "dist/anime_upscaler_gui-*.whl") |
    Sort-Object LastWriteTime | Select-Object -Last 1
if (-not $wheel) { throw "no GUI wheel in dist/ - build it first" }
$venv = Join-Path $work "venv"
& python -m venv $venv
$vpy = Join-Path $venv "Scripts/python.exe"
& $vpy -m pip install --upgrade pip | Out-Null
& $vpy -m pip install $wheel.FullName
if ($LASTEXITCODE -ne 0) { throw "wheel install failed in clean venv" }
$exe = Join-Path $venv "Scripts/anime-upscaler-gui.exe"
& $exe --self-check
if ($LASTEXITCODE -ne 0) { throw "self-check failed in clean venv" }
& $vpy -c "from anime_upscaler_gui.arch_registry import available_architectures as a; print(sorted(a()))"
Write-Host "REHEARSAL OK: $work"
```

- [ ] **Step 17.4: Rehearse for real**

```powershell
Remove-Item dist -Recurse -Force -ErrorAction SilentlyContinue
& $py -m pip wheel apps/anime_upscaler_gui --no-deps -w dist
& $py -m pip wheel . --no-deps -w dist
& pwsh -NoProfile -File scripts/release/rehearse_windows.ps1
```

Expected: `REHEARSAL OK` plus a JSON report; this proves the wheel installs with no repo checkout and no anime_sr.

- [ ] **Step 17.5: Commit**

```powershell
git add apps/anime_upscaler_gui/anime_upscaler_gui/__main__.py apps/anime_upscaler_gui/tests/test_self_check.py scripts/release/rehearse_windows.ps1
git commit -m "feat(release): --self-check flag and clean-room Windows rehearsal script"
```

## Task 18: Publish the GitHub pre-release

**Files:**
- Create: `docs/release/v0.2.0-preview-notes.md`
- GitHub: release v0.2.0-preview (pre-release, not PyPI)

- [ ] **Step 18.1: Write the release notes**

Create `docs/release/v0.2.0-preview-notes.md`:

```markdown
# anime-upscaler-gui 0.2.0-preview

A standalone Windows GUI for upscaling anime images and video with PyTorch.
This is a preview, not a supported release.

## Install

1. Install ffmpeg and put ffmpeg.exe + ffprobe.exe on PATH.
2. Download the wheel from this release.
3. python -m venv anime-gui
4. anime-gui/Scripts/python.exe -m pip install anime_upscaler_gui-0.2.0-py3-none-any.whl
5. anime-gui/Scripts/anime-upscaler-gui.exe --self-check
6. anime-gui/Scripts/anime-upscaler-gui.exe

## What works

- Image upscaling to PNG/JPEG, CPU and CUDA.
- Video upscaling with the source audio preserved (MP4/MKV/MOV sources).
- Batch queue, pause/resume, cancel, tiled inference for small GPU memory.

## Known limitations

- Windows only; tested on one machine with Python 3.12.
- Training and the anime-sr research package are NOT part of this preview.
- No model weights are bundled; download a compatible checkpoint or follow docs.
- WEBM output may warn that audio could not be preserved (container limitation).
- Custom URL downloads are unverified unless a sha256 is declared upstream.

## Verify the download

Compare Get-FileHash anime_upscaler_gui-0.2.0-py3-none-any.whl -Algorithm SHA256
against SHA256SUMS.txt attached to this release.
```

- [ ] **Step 18.2: Build, checksum, and create the release**

```powershell
Remove-Item dist -Recurse -Force -ErrorAction SilentlyContinue
& $py -m pip wheel apps/anime_upscaler_gui --no-deps -w dist
$wheel = Get-ChildItem dist/anime_upscaler_gui-*.whl | Select-Object -Last 1
((Get-FileHash $wheel.FullName -Algorithm SHA256).Hash.ToLower() + "  " + $wheel.Name) |
    Set-Content -Path dist/SHA256SUMS.txt -Encoding ascii
git add docs/release/v0.2.0-preview-notes.md
git commit -m "docs(release): v0.2.0-preview notes and limitations"
git tag -a v0.2.0-preview -m "anime-upscaler-gui 0.2.0 preview"
git push origin release/gui-preview --tags
gh release create v0.2.0-preview $wheel.FullName dist/SHA256SUMS.txt --prerelease --title "anime-upscaler-gui 0.2.0-preview" --notes-file docs/release/v0.2.0-preview-notes.md
gh release view v0.2.0-preview
```

Expected: the release page lists two assets, marked pre-release. Do not run twine upload; PyPI is Track 2.

- [ ] **Step 18.3: Verify the published artefact by downloading it back**

```powershell
$dl = Join-Path $env:TEMP "anime-gui-download"
Remove-Item $dl -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory $dl | Out-Null
gh release download v0.2.0-preview --dir $dl
Get-FileHash (Join-Path $dl $wheel.Name) -Algorithm SHA256
Get-Content (Join-Path $dl "SHA256SUMS.txt")
```

Expected: hashes match. Record both hashes in the task log.

## Task 19: Run the stranger test (2026-11-24 to 2026-12-14)

**Files:**
- Create: `docs/validation/STRANGER_TEST.md`
- Create: `docs/validation/logs/TEMPLATE.md`

- [ ] **Step 19.1: Write the protocol**

Create `docs/validation/STRANGER_TEST.md`:

```markdown
# Stranger test protocol (v0.2.0-preview)

Goal: prove an outside user, given only the release page, can install and
upscale one video without the author touching the keyboard.

## Recruitment

- 3 testers, none of whom wrote any part of this project.
- Give each only the release URL + the wheel + one source MP4 (with audio).
- Do not answer questions during the first 15 minutes; note them instead.

## Tasks

1. Install the wheel into a clean venv (follow only the release notes).
2. Run --self-check and paste its JSON output.
3. Launch the GUI and load the sample video.
4. Run one upscale and report wall-clock time to first finished file.
5. Verify the output plays with sound; the source file is byte-identical
   to a copy taken before the run.

## Pass criteria (need 2 of 3 testers)

- First upscale completes unassisted in under 15 minutes.
- Output carries an audio stream when the source has one.
- Source file sha256 unchanged.
- No traceback dialog; no report of finished for a missing or 0-byte file.

## Record

Copy logs/TEMPLATE.md to logs/YYYY-MM-DD-tester-N.md and fill every field,
including verbatim questions and the commands the tester typed.
```

Create `docs/validation/logs/TEMPLATE.md` with fields: tester name/handle, date, machine, OS/Python, start/end time, questions asked (verbatim), time-to-first-output, ffprobe audio result, source sha256 before/after, observed failures, verdict PASS/FAIL.

- [ ] **Step 19.2: Execute the sessions and record evidence**

For each tester run this before and after their upscale (replace the paths):

```powershell
Get-FileHash "E:/path/to/source.mp4" -Algorithm SHA256
ffprobe -v error -select_streams a -show_entries stream=index -of csv=p=0 "E:/path/to/output.mp4"
```

Expected: identical source hashes; non-empty audio index in the output. File the log.

- [ ] **Step 19.3: Commit the protocol and logs**

```powershell
git add docs/validation
git commit -m "docs(validation): stranger-test protocol, template, and session logs"
```

## Task 20: Decide on 2026-12-14

**Files:**
- Create: `docs/validation/DECISION_2026-12-14.md`
- Modify: `docs/status/MEASURED_STATUS_2026-11-16.md` (append outcome)
- Modify: `README.md` (only if reclassifying)

- [ ] **Step 20.1: Write the decision record**

Create `docs/validation/DECISION_2026-12-14.md` with a table of the three sessions (tester, time-to-first-output, audio, source hash, verdict) and one of three outcomes:

- **PASS (2 of 3 or better):** keep the GitHub preview, merge release/gui-preview to main, schedule Track 2; PyPI publication is allowed after Track 2.
- **PARTIAL (1 of 3):** keep the preview labelled experimental, fix the top three observed failures, schedule a second round for 2027-01-15, and reclassify the README as author tooling until a round passes.
- **FAIL (0 of 3):** stop releases, archive the preview, keep the repository private, and re-run the readiness review before another attempt.

- [ ] **Step 20.2: Execute the follow-up and commit**

If PASS, open the PR and merge:

```powershell
git switch main
git merge --no-ff release/gui-preview -m "merge: v0.2.0-preview validated by stranger test"
git push origin main
git add docs/validation/DECISION_2026-12-14.md
git commit -m "docs: release decision after stranger test (2026-12-14)"
```

If PARTIAL or FAIL, do the edits named by the chosen outcome instead, then commit with the message "docs: record the 2026-12-14 release decision and follow-up".

---

# Track 2 - Restore the research core (Tasks 21-28, start after 2026-12-14)

Nothing in this track ships to the GUI preview. It returns the training and inference core to a working, honest state. Each task is independently committable.

## Task 21: Restore the training package and its tests

**Files:**
- Verify: `src/anime_sr/training/` (Task 1 repaired stage_state)
- Modify: whatever the suite proves broken (likely annotation imports in training modules)
- Test: `tests -k "training or stage_state"`

- [ ] **Step 21.1: Collect the full error list**

```powershell
& $py -m pytest tests -q -p no:cacheprovider "--basetemp=$basetemp" --collect-only 2>&1 | Select-Object -Last 40
& $py -m flake8 src/anime_sr/training --select=F821,F82,F811
```

Expected: whatever remains after Task 1 (historically 17 F821 undefined names across the package plus test imports of a removed data API). Write the exact list into the task log.

- [ ] **Step 21.2: Fix undefined names one file at a time**

For each F821, add the missing import (usually `from typing import ...`, a model class, or a constant) and run the module import plus its unit tests before moving to the next file:

```powershell
& $py -c "import anime_sr.training; print('ok')"
& $py -m pytest tests/unit/training -q -p no:cacheprovider "--basetemp=$basetemp"
```

- [ ] **Step 21.3: Prove the training path starts**

```powershell
& $py -m scripts.train --help
& $py -c "from anime_sr.training import AnimeSRTrainer; print(AnimeSRTrainer)"
```

Expected: help renders and the trainer imports. Do not run a full training job here; one epoch on the tiny dataset is covered by the e2e test in Task 26.

- [ ] **Step 21.4: Commit**

```powershell
git add src/anime_sr/training
git commit -m "fix(training): resolve undefined names; training package imports cleanly"
```

## Task 22: Make preprocessing configuration actually take effect

**Files:**
- Modify: `src/anime_sr/data/preprocessing_manager.py:325-330` and `:366-369`
- Modify: `src/anime_sr/data/degradation_pipeline.py:159-171`
- Test: `tests/unit/data/test_preprocessing_manager.py` (create if absent)

- [ ] **Step 22.1: Write the failing tests**

```python
"""Preprocessing config must change the produced samples, not be ignored."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from anime_sr.data.preprocessing_manager import PreprocessingManager


def _write_image(path, size=(64, 64)):
    rng = np.random.default_rng(0)
    data = (rng.random((size[1], size[0], 3)) * 255).astype("uint8")
    Image.fromarray(data).save(path)
    return path


def test_hybrid_mode_is_not_a_plain_bicubic_noop(tmp_path):
    src = _write_image(tmp_path / "img.png")
    mgr = PreprocessingManager({"preprocessing": {"mode": "hybrid", "scale": 1}})
    sample = mgr.process_image(src)
    lr = np.asarray(Image.open(src).resize((16, 16), Image.BICUBIC))
    assert not np.array_equal(np.asarray(sample["lr"]), lr)


def test_unknown_mode_fails_loudly():
    with pytest.raises(ValueError):
        PreprocessingManager({"preprocessing": {"mode": "definitely-not-a-mode"}})
```

Adapt process_image to the real API (grep for the manager's public method names in its existing tests); the two requirements are fixed: a mode switch changes pixels, an unknown mode raises.

- [ ] **Step 22.2: Run to confirm the no-op and the missing validation**

```powershell
& $py -m pytest tests/unit/data/test_preprocessing_manager.py -q -p no:cacheprovider "--basetemp=$basetemp"
```

Expected: the hybrid comparison fails (pixel-identical to bicubic) and the unknown mode does not raise.

- [ ] **Step 22.3: Read the config one way**

In preprocessing_manager.py replace the config unpacking with:

```python
        cfg = self.config.get("preprocessing") or {}
        if not cfg:
            # Some callers construct the manager with the preprocessing block
            # itself; accept both shapes instead of silently using defaults.
            cfg = self.config if isinstance(self.config, dict) else {}
        mode = cfg.get("mode") or "hybrid"
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"unknown preprocessing mode {mode!r}; supported: {sorted(self.SUPPORTED_MODES)}")
```

Define `SUPPORTED_MODES` from the modes the pipeline actually implements (read the branches below it) and store the resolved `cfg` on `self`.

- [ ] **Step 22.4: Make the preset selection real**

In degradation_pipeline.py line 159 replace the "cfg is None" test with "no cfgs configured", and make an unknown/empty mode fall back to the named preset instead of to an empty pipeline:

```python
        if not cfgs and mode in PRESETS:
            cfgs = PRESETS[mode]
```

Ensure the `hybrid` and `on_the_fly` branches do not collapse to `F.interpolate(..., mode="bicubic")` only; each must add at least the configured blur/noise/jpeg step. Add a comment naming the steps each mode runs.

- [ ] **Step 22.5: Fix the CUDA generator crash**

At preprocessing_manager.py lines 366-369 the random draw uses a CUDA generator against a CPU tensor, which raises `RuntimeError: Expected a 'cpu' device type for generator but found 'cuda'`. Replace it with a device-matched generator:

```python
        generator = torch.Generator(device=self.device)
        noise = torch.rand(1, 1, h, w, generator=generator, device=self.device)
```

- [ ] **Step 22.6: Verify**

```powershell
& $py -m pytest tests/unit/data/test_preprocessing_manager.py -q -p no:cacheprovider "--basetemp=$basetemp"
```

Expected: both tests pass. When CUDA is present, run the gpu_degradation branch once and record the output.

- [ ] **Step 22.7: Commit**

```powershell
git add src/anime_sr/data/preprocessing_manager.py src/anime_sr/data/degradation_pipeline.py tests/unit/data/test_preprocessing_manager.py
git commit -m "fix(data): preprocessing config actually applies; validate mode; fix CUDA generator device"
```

## Task 23: Forward preprocessing through separate dataloaders

**Files:**
- Modify: `src/anime_sr/data/dataloader.py:455`, `:479`, `:536`
- Test: `tests/unit/data/test_dataloader.py` (create if absent)

- [ ] **Step 23.1: Write the failing test**

```python
"""--separate-loaders must not drop the preprocessing configuration."""
from anime_sr.data.dataloader import DatasetFactory

def test_separate_loaders_forwards_preprocessing(tmp_path):
    (tmp_path / "train" / "hr").mkdir(parents=True)
    cfg = {
        "datasets": {
            "train": {
                "type": "paired",
                "hr_dir": str(tmp_path / "train" / "hr"),
                "lr_dir": str(tmp_path / "train" / "lr"),
            }
        },
        "separate_loaders": True,
        "preprocessing": {"mode": "hybrid"},
    }
    ds = DatasetFactory.create(cfg["datasets"]["train"], preprocessing_config=cfg["preprocessing"])
    assert getattr(ds, "preprocessing_config", None) == {"mode": "hybrid"}
```

Adapt constructor names to the real DatasetFactory signature (read lines 400-540 first); the requirement is that the preprocessing dict reaches the dataset in the separate-loaders branch.

- [ ] **Step 23.2: Run, fix, verify**

```powershell
& $py -m pytest tests/unit/data/test_dataloader.py -q -p no:cacheprovider "--basetemp=$basetemp"
```

Then in dataloader.py pass `preprocessing_config=preprocessing_config` in every branch (line 455 and the two list comprehensions at 479 and 536 that currently construct datasets without it). Re-run until green.

- [ ] **Step 23.3: Commit**

```powershell
git add src/anime_sr/data/dataloader.py tests/unit/data/test_dataloader.py
git commit -m "fix(data): separate loaders keep preprocessing config"
```

## Task 24: Repair the data package import surface

**Files:**
- Modify: `src/anime_sr/data/__init__.py:48`, `:96-142`
- Modify: `tests/e2e/test_e2e_data_pipeline.py`, `tests/e2e/test_e2e_small_dataset_pipeline.py`
- Optionally create thin compatibility modules: `src/anime_sr/data/degradation.py`, `dataset_validation.py`, `base.py`

- [ ] **Step 24.1: Map the intended public API**

```powershell
& $py -c "import anime_sr.data as d; print(sorted(n for n in dir(d) if not n.startswith('_')))"
Select-String -Path src/anime_sr/data/__init__.py -Pattern "except ImportError|from \.|import \." | Select-Object -First 30
Get-ChildItem src/anime_sr/data -Filter *.py | Select-Object Name
```

Compare with what the e2e tests import. Decide per name: it exists (re-export), it was renamed (update the test), or it never existed (delete the test or build a compat shim). Record the mapping table in the task log.

- [ ] **Step 24.2: Remove the silent whole-namespace fallback**

In `src/anime_sr/data/__init__.py` the bare `except ImportError` retry hides real breakage. Replace it with explicit module re-exports:

```python
from .dataset import PairedDataset, UnpairedDataset  # adjust to real names
from .preprocessing_manager import PreprocessingManager
from .degradation_pipeline import DegradationPipeline

__all__ = ["PairedDataset", "UnpairedDataset", "PreprocessingManager", "DegradationPipeline"]
```

If a module genuinely cannot import without an optional dependency, guard that single symbol (not the whole namespace) and document it.

- [ ] **Step 24.3: Fix or add the modules the tests reference**

For each of `anime_sr.data.degradation`, `anime_sr.data.dataset_validation` and `anime_sr.data.base.SuperResolutionDataset`: if the implementation exists under another name, add a one-screen compatibility module that re-exports it; if it does not exist, update the e2e test to the real API. Do not leave an ImportError in a test module.

- [ ] **Step 24.4: Verify**

```powershell
& $py -c "import anime_sr.data; print('data import OK')"
& $py -m pytest tests/e2e/test_e2e_data_pipeline.py tests/e2e/test_e2e_small_dataset_pipeline.py -q -p no:cacheprovider "--basetemp=$basetemp"
```

Expected: import OK; both e2e files collect and run (no collection errors).

- [ ] **Step 24.5: Commit**

```powershell
git add src/anime_sr/data tests/e2e
git commit -m "fix(data): explicit public imports; repair e2e data tests and compat modules"
```

## Task 25: Restore test integrity

**Files:**
- Modify: `tests/e2e/test_e2e_training_pipeline.py:321-324`
- Modify: `tests/unit/models/test_checkpoint_model.py:15-47`
- Modify: `tests/unit/performance/test_benchmarking.py` (the bare `assert True` occurrences at 204, 234, 319)
- Modify: `tests/integration/test_evaluate_quality_smoke.py:167-171`
- Sweep: every `except ImportError:` guard that silently skips a test body

- [ ] **Step 25.1: Enumerate the fake tests**

```powershell
& $py -m pip install radon --quiet 2>$null
Select-String -Path tests/**/*.py -Pattern "assert True" | Select-Object Path,LineNumber
Select-String -Path tests/**/*.py -Pattern "^\s+pass\s*$" | Select-Object Path,LineNumber
Select-String -Path tests/**/*.py -Pattern "except ImportError" -Context 0,6 | Select-Object -First 60
```

Expected: the known list (pass-only test, tests that return booleans instead of asserting, three assert True placeholders, the nonexistent metrics path, and about 90 import-guarded test bodies). Record the count.

- [ ] **Step 25.2: Rewrite the worst offenders with real assertions**

- `tests/e2e/test_e2e_training_pipeline.py:321-324`: the body is just `pass`. Either implement the checkpoint round-trip (save -> load -> compare one tensor) using the trainer already under test, or mark it `pytest.mark.skip(reason="covered by tests/unit/training/test_checkpoint_io.py")` and open a follow-up. A bare pass is not allowed.
- `tests/unit/models/test_checkpoint_model.py:15-47`: replace every `return True/False` with `assert` or `pytest.raises` so a failure fails the test.
- `tests/unit/performance/test_benchmarking.py`: replace `assert True` with the measured property (for example `assert elapsed < limit` or `assert result["fps"] > 0`).
- `tests/integration/test_evaluate_quality_smoke.py:167-171`: the referenced `src/utils/metrics.py` does not exist; point it at the real module under `src/anime_sr/` or delete the dead test.

- [ ] **Step 25.3: Replace silent import skips with explicit ones**

For each import-guarded body, keep the guard only when the dependency is genuinely optional; otherwise remove the try/except so a broken import fails the test. When keeping a skip, it must be `pytest.mark.skip(reason="optional dependency X not installed")` at module level, never a swallowed ImportError around the assertions.

- [ ] **Step 25.4: Verify**

```powershell
& $py -m pytest tests -q -p no:cacheprovider "--basetemp=$basetemp" 2>&1 | Select-Object -Last 30
```

Expected: collection errors are gone; tests fail only for genuine product bugs. Record the pass/fail/skip counts as the new baseline.

- [ ] **Step 25.5: Commit**

```powershell
git add tests
git commit -m "test: replace placeholder assertions and silent import skips with real checks"
```

## Task 26: Repair or delete the legacy packager; clean history; unblock strict CI

**Files:**
- Modify or delete: `scripts/package_project.py`
- Modify: `tests/unit/performance/test_benchmarking.py` (only if Task 25 left debt)
- Modify: `.github/workflows/ci.yml` (turn advisory lint/mypy blocking)
- History: repository rewrite before going public

- [ ] **Step 26.1: Decide and execute the packager outcome**

If keeping it: replace its wheel assembly with a call to the standard build (`python -m build --wheel`) and delete the generated-setup.py path and the bogus entry points (`anime-sr-train=scripts.train:main`, `anime-sr-api=src.api.inference_api:main`; src/api does not exist). If deleting: remove `scripts/package_project.py` and `scripts/legacy/` and update the guard test from Task 15 to require absence. Either way add `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` wherever it prints non-ASCII.

- [ ] **Step 26.2: Remove the compiled-Python history before publication**

```powershell
git count-objects -vH
git rev-list --objects --all | Select-String "^[0-9a-f]+ Python/" | Measure-Object -Line
```

If the vendored interpreter is still in history (it inflates .git to ~308 MiB), install git-filter-repo, take a full backup bundle, then rewrite:

```powershell
git bundle create ../upscale_anime-backup.bundle --all
git filter-repo --path Python/ --invert-paths
git count-objects -vH
```

Coordinate before force-pushing: this rewrites every commit hash; tags made in Track 1 must be re-created on the rewritten commits.

- [ ] **Step 26.3: Make strict checks blocking**

Once `flake8 .`, `mypy src/anime_sr` and `pytest tests` are clean locally, remove the `continue-on-error` flags from Task 16 and add the F821/F82 gates back to the blocking lint step.

```powershell
& $py -m flake8 . --select=E9,F63,F7,F82 --exclude=.venv,data,runs
& $py -m pip install mypy --quiet
& $py -m mypy src/anime_sr
```

Expected: zero blocking errors. If mypy has a long tail, fix the src/anime_sr/{training,data,models} tree only and scope the CI step to that tree, noting the narrowed scope in the workflow comment with the date it was added.

- [ ] **Step 26.4: Commit and push**

```powershell
git add scripts tests .github/workflows/ci.yml
git commit -m "chore: retrench legacy packager; strict lint and type gates back on"
git push --force-with-lease origin main
```

## Task 27: Publish to PyPI (only after the stranger test passes)

**Files:**
- Modify: `pyproject.toml` (version 0.1.0 -> 1.0.0 only if the core is actually restored; otherwise keep pre-1.0)
- Modify: `apps/anime_upscaler_gui/pyproject.toml` (version 0.2.0 final)
- Create: `docs/release/PYPI_CHECKLIST.md`

- [ ] **Step 27.1: Rehearse the upload without uploading**

```powershell
& $py -m pip install build twine --quiet
Remove-Item dist -Recurse -Force -ErrorAction SilentlyContinue
& $py -m build
& $py -m twine check dist/*
```

Expected: two sdists and two wheels; twine check passes both. Confirm both wheels are self-contained: unzip the GUI wheel and check `anime_upscaler_gui/_models/` and `data/registry.json` are present.

- [ ] **Step 27.2: Upload only if the gate is green**

Gate: stranger test PASS (Task 20), CI green on main, no open release-blocking issue. Then:

```powershell
& $py -m twine upload dist/*
```

- [ ] **Step 27.3: Verify from a fresh environment**

```powershell
$v = Join-Path $env:TEMP ("pypi-check-" + [guid]::NewGuid().ToString("N"))
& python -m venv $v
& (Join-Path $v "Scripts/python.exe") -m pip install anime-sr anime-upscaler-gui
& (Join-Path $v "Scripts/anime-upscaler-gui.exe") --self-check
```

Expected: self-check exits 0 from the published packages with no repository present.

- [ ] **Step 27.4: Commit and tag**

```powershell
git add pyproject.toml apps/anime_upscaler_gui/pyproject.toml docs/release/PYPI_CHECKLIST.md
git commit -m "release: PyPI publication checklist and final version metadata"
git tag -a anime-sr-1.0.0 -m "first supported release of the restored core"
git push origin main --tags
```

## Task 28: Close the loop

**Files:**
- Modify: `docs/status/MEASURED_STATUS_2026-11-16.md` (rename to the release date and update)
- Modify: `docs/PROJECT_STATUS.md`
- Modify: `README.md`

- [ ] **Step 28.1: Re-measure everything and write the final status**

Re-run the complete evidence set: root and GUI wheel builds, GUI suite, root suite, clean-room rehearsal, PyPI fresh install, and the stranger-test logs. Update the measured table with real dates and copy the stranger-test verdict into `docs/PROJECT_STATUS.md`.

- [ ] **Step 28.2: Delete the stale claims**

Update README badges/claims to point at the measured table. Remove any remaining "production ready", "95%", or benchmark numbers that were never reproduced with the committed code.

- [ ] **Step 28.3: Commit**

```powershell
git add docs README.md
git commit -m "docs: final measured status after the 2026-12-14 release gate"
```

---

# Risks, stop rules, and definition of done

## Stop rules

- If the GUI wheel cannot be made self-contained by 2026-11-23 (Task 12 shows imports reaching beyond the seven-file closure), stop Track 1A and ship the source-only preview instead: mark the GitHub release as "source archive, install from checkout", and postpone the wheel. Do not ship a wheel that imports undeclared packages.
- If any fix would require touching the training core before 2026-12-14, defer it to Track 2 and record the deferral in docs/status.
- If the stranger test fails, do not publish to PyPI regardless of how green CI looks.
- If a Task's expected output is not reached within a day, stop and request a review room; do not lower the test.

## Rollback

- Every task is one commit. To revert a bad fix: `git revert <sha>`; the GUI suite and `tests/release` are the mechanical check.
- The snapshot commit is the safety net: `git reset --hard snapshot-2026-11-16` restores the reviewed tree (uncommitted review artefacts live outside git).
- Repository history rewrite in Task 26 is destructive: it requires a bundle backup first and is never done after PyPI publication of the same version.

## Definition of done

- [ ] Both wheels build from a clean checkout with `python -m pip wheel` and `python -m build`.
- [ ] GUI wheel installs into a fresh venv and `anime-upscaler-gui --self-check` exits 0 with no repository present.
- [ ] A video upscale keeps the source audio; the source file sha256 is unchanged; a failed encode reports ERROR and leaves no partial file.
- [ ] Checkpoint scanning never executes pickle code; unsafe loading requires `ANIME_UPSCALER_ALLOW_UNSAFE_PICKLE=1`.
- [ ] CI runs on the working branch, blocks syntax errors, and reports full lint/type/test status honestly.
- [ ] A GitHub pre-release v0.2.0-preview exists with a wheel, SHA256SUMS.txt, and limitations; the download hash verifies.
- [ ] Three stranger sessions are logged; 2 of 3 pass the criteria; the 2026-12-14 decision record exists and has been executed.
- [ ] Track 2: training package imports and its tests collect; preprocessing config changes pixels; separate loaders keep preprocessing; CUDA degradation runs; test integrity sweep complete; repository history no longer ships the vendored interpreter; PyPI publication only after the gate.

---

# Self-review of this plan

- **Review coverage:** every fatal finding from the review room maps to a task: audio loss (Tasks 5-6), failed-encode-as-finished (Task 7), in-place truncation (Task 8), abort semantics (Task 9), pause/stale settings (Task 10), tiling crash (Task 11), undeclared anime_sr import (Task 12), PEP 639 build break and ResolutionImpossible pins (Task 13), unsafe pickle loading (Task 14), broken legacy packager (Tasks 15 and 26), CI never running (Task 16), clean-room install (Task 17), no published artefact (Task 18), no outsider evidence (Tasks 19-20), frozen core (Tasks 21-24), fake tests (Task 25), history bloat and PyPI (Tasks 26-27), dishonest docs (Tasks 3 and 28).
- **Placeholders:** none. Every step names exact files, commands, and expected output. Where a real function/dataclass name is unknown, the step says how to find it and states the invariant the code must satisfy.
- **Dates:** absolute throughout; the 2026-12-14 gate is the only calendar decision that matters.
- **Write scopes:** Track 1 edits `apps/anime_upscaler_gui/`, `pyproject.toml`, docs, CI, and `tests/release/` only; Track 2 edits `src/anime_sr/` and root tests. The two tracks never share a file, which is what allows Track 2 to wait.

# Execution handoff

Plan saved to `docs/superpowers/plans/2026-11-16-release-readiness.md`.

Two execution options:

1. **Subagent-driven (recommended):** one fresh subagent per task, review between tasks, fast iteration. Use `subagent-driven-development`.
2. **Inline execution:** batch execution with checkpoints for review. Use `executing-plans`.

Start with Task 0 in this session, then Task 1 (the three-f-string repair) because it unblocks both the commit and test collection.
