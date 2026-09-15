"""Settings persistence: load/save JSON to a user-configurable directory.

OS defaults:
  Windows: %APPDATA%/anime_upscaler_gui/settings.json
  Linux/macOS: $XDG_CONFIG_HOME/anime_upscaler_gui/settings.json
            (falls back to ~/.config/anime_upscaler_gui/settings.json)

Phase 4 additions:
  - Atomic write (`tempfile + os.replace`) so a crash mid-write can't truncate
    the file to half its content.
  - Export/import settings to/from an arbitrary user-chosen JSON file.
  - `QueueController` for incremental persistence of the batch queue.
"""
import json
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import List, Optional

from .state import FormState, Job, JobStatus


APP_DIR_NAME = "anime_upscaler_gui"
SETTINGS_FILE = "settings.json"
# Optional preset-catalog extension in the app-data folder (same schema as the
# bundled apps/anime_upscaler_gui/data/registry.json; entries override by id).
REGISTRY_FILE = "registry.json"


def _default_app_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / APP_DIR_NAME


def _atomic_dump_json(path: Path, payload: dict) -> None:
    """Atomically write JSON to `path` via temp file + os.replace.

    Worst-case the original file is untouched if the write fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


@dataclass
class _Defaults(FormState):
    # Schema version for future JSON migrations.
    version: int = 1
    # Paths
    app_dir: str = ""
    pretrained_dir: str = ""
    output_dir: str = ""
    # Output mode: "system_default" | "same_folder" | "custom"
    output_mode: str = "system_default"
    single_file_suffix: str = "_upscaled"
    batch_folder_name: str = APP_DIR_NAME  # created inside input folder for multi-file
    # Inference
    device: str = "cuda"   # cuda | cpu
    fp16: bool = True
    # outscale, batch_size, tile_size: inherited from FormState with validators
    decode: str = "cv2"    # cv2 | pyav
    prefetch: str = "sync" # sync | async
    pin_memory: str = "auto"  # auto | on | off
    downscale_max_edge: int = 2560  # 0 = disabled
    # GPU guard
    gpu_guard_mode: str = "warn"  # warn | auto_downscale | tiled | off
    tile_overlap: int = 32
    # Inference TTA (Phase 5 ship): default ON for the recommended default
    # model (rfdn_distill_v1_4x + TTA = 29.46 dB test PSNR). The user can disable
    # in the Settings panel if inference speed matters more than peak quality.
    tta: bool = True
    # Phase 1 (Real-time 4K): TensorRT engine acceleration.
    # Auto-disabled when TRT not importable, on CPU, or with kind=animesr.
    use_tensorrt: bool = True
    # Output video
    video_crf: int = 18
    video_preset: str = "medium"
    # Phase 1 (Real-time 4K): NVIDIA hardware encoder (NVENC).
    # Auto-disabled when ffmpeg's h264_nvenc is not present.
    use_nvenc: bool = True
    nvenc_preset: str = "p1"
    nvenc_qp: int = 18         # constant-QP target; lower = better quality, larger file
    # Phase 2 (Real-time 4K): Cascade depth. None = auto from the loaded model's
    # scale (2x model -> cascade 2x+2x = 4 calls; 4x model -> single shot).
    # Set to an explicit positive int to force a depth (advanced / debugging).
    cascade_mode: Optional[int] = None
    # Model
    last_model: str = ""
    # Queue
    queue_mode: str = "single"  # single | batch
    # Window
    geometry: str = "1100x800"
    # Theme: "light" | "dark" | "high_contrast"
    theme: str = "light"
    # Locale: "en" | "ar" (Phase 14)
    locale: str = "en"


DEFAULTS = _Defaults()


def _coerce(d: dict) -> _Defaults:
    """Merge saved dict onto defaults; coerce types so an old file doesn't crash.

    Values that can't be coerced (e.g. an int field with the string "wat") are
    silently dropped — the field keeps its default. This makes _Settings robust
    to hand-edited or partially-corrupt settings files.
    """
    out = _Defaults()
    for k, v in d.items():
        if hasattr(out, k):
            cur = getattr(out, k)
            if isinstance(cur, bool):
                setattr(out, k, bool(v))
            elif isinstance(cur, int) and not isinstance(cur, bool):
                try:
                    setattr(out, k, int(v))
                except (TypeError, ValueError):
                    pass
            elif isinstance(cur, float):
                try:
                    setattr(out, k, float(v))
                except (TypeError, ValueError):
                    pass
            else:
                setattr(out, k, v)
    return out


class _AppPaths:
    """Resolved paths for settings + cache."""

    def __init__(self, app_dir: Optional[Path] = None):
        self.app_dir = Path(app_dir) if app_dir else _default_app_dir()
        self.app_dir.mkdir(parents=True, exist_ok=True)
        self.settings_path = self.app_dir / SETTINGS_FILE
        self.registry_path = self.app_dir / REGISTRY_FILE
        self.cache_dir = self.app_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.logs_dir = self.app_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

    def move_to(self, new_dir: Path) -> None:
        """Move existing app data to a new directory, preserving files."""
        new_dir = Path(new_dir)
        new_dir.mkdir(parents=True, exist_ok=True)
        if new_dir.resolve() == self.app_dir.resolve():
            return
        for sub in ("cache", "logs"):
            src = self.app_dir / sub
            if src.exists():
                shutil.move(str(src), str(new_dir / sub))
        if self.settings_path.exists():
            shutil.move(str(self.settings_path), str(new_dir / SETTINGS_FILE))
        if self.registry_path.exists():
            shutil.move(str(self.registry_path), str(new_dir / REGISTRY_FILE))
        self.app_dir = new_dir
        self.settings_path = new_dir / SETTINGS_FILE
        self.registry_path = new_dir / REGISTRY_FILE


class _Settings:
    """JSON-backed user settings with on-disk layout above."""

    def __init__(self, paths: _AppPaths):
        self.paths = paths
        self.data: _Defaults = _Defaults()
        self.corrupt: bool = False  # set True when settings file was unparseable
        if paths.settings_path.exists():
            try:
                raw = json.loads(paths.settings_path.read_text(encoding="utf-8"))
                self.data = _coerce(raw)
            except (OSError, json.JSONDecodeError):
                # Corrupt file -> back it up and start fresh.
                self.corrupt = True
                try:
                    shutil.copy(paths.settings_path, paths.settings_path.with_suffix(".bak"))
                except OSError:
                    pass
        # Fill in path defaults if empty.
        if not self.data.app_dir:
            self.data.app_dir = str(paths.app_dir)
        if not self.data.pretrained_dir:
            self.data.pretrained_dir = str(_default_pretrained_dir())
        if not self.data.output_dir:
            self.data.output_dir = str(_default_output_dir())

    def save(self) -> None:
        _atomic_dump_json(self.paths.settings_path, asdict(self.data))

    def update(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self.data, k):
                setattr(self.data, k, v)
        self.save()

    def export(self, path: Path) -> None:
        """Write current settings to a user-chosen JSON file (atomic)."""
        _atomic_dump_json(Path(path), asdict(self.data))

    def import_file(self, path: Path) -> None:
        """Load settings from `path`; merge onto defaults; save back atomically.

        Unknown fields in the import file are silently ignored (forward compat).
        Missing fields fall back to defaults. Invalid types are coerced.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        new = _coerce(raw)
        for f in fields(_Defaults):
            if hasattr(new, f.name):
                setattr(self.data, f.name, getattr(new, f.name))
        self.save()

    # ---- helpers used by the GUI ----
    def pretrained_dir_path(self) -> Path:
        p = Path(self.data.pretrained_dir) if self.data.pretrained_dir else _default_pretrained_dir()
        return p

    def output_dir_path(self) -> Path:
        if self.data.output_mode == "same_folder":
            # For multi-file, the actual folder is computed per-job; this is the fallback.
            return Path(self.data.output_dir) if self.data.output_dir else _default_output_dir()
        return Path(self.data.output_dir) if self.data.output_dir else _default_output_dir()


def _default_pretrained_dir() -> Path:
    """Where to look for downloaded checkpoints. Defaults to a sibling of this package.

    apps/anime_upscaler_gui/anime_upscaler_gui/settings.py  ->  3 .parent to reach repo root,
    then /pretrained. When moved standalone, this default is harmless; the GUI shows the
    resolved path on first run and lets the user change it.
    """
    here = Path(__file__).resolve().parent
    return here.parent.parent.parent / "pretrained"


def _default_output_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("USERPROFILE", Path.home())) / "Documents"
    else:
        base = Path.home() / "Pictures"
    return base / APP_DIR_NAME


# ---------------------------------------------------------------------------- #
# Queue persistence
# ---------------------------------------------------------------------------- #
QUEUE_FILE = "queue.json"
QUEUE_VERSION = 1


def _job_to_dict(j: Job) -> dict:
    return {
        "id": j.id,
        "input": str(j.input),
        "output": str(j.output),
        "status": j.status.value,
        "error": j.error,
        "fps": j.fps,
        "infer_ms": j.infer_ms,
        "model_filename": j.model_filename,
        "kind": j.kind,
        "scale": j.scale,
        "is_video": j.is_video,
        "cut_start": j.cut_start,
        "cut_end": j.cut_end,
    }


def _dict_to_job(d: dict) -> Optional[Job]:
    try:
        status = JobStatus(d.get("status", "pending"))
    except ValueError:
        status = JobStatus.PENDING
    # The worker thread that owned a RUNNING job is gone across restarts;
    # reset to PENDING so the user can re-Start.
    if status == JobStatus.RUNNING:
        status = JobStatus.PENDING
    try:
        return Job(
            id=int(d["id"]),
            input=Path(d["input"]),
            output=Path(d["output"]),
            status=status,
            error=str(d.get("error", "")),
            fps=float(d.get("fps", 0.0)),
            infer_ms=float(d.get("infer_ms", 0.0)),
            model_filename=str(d.get("model_filename", "")),
            kind=str(d.get("kind", "")),
            scale=int(d.get("scale", 4)),
            is_video=bool(d.get("is_video", False)),
            cut_start=float(d.get("cut_start", 0.0)),
            cut_end=float(d.get("cut_end", 0.0)),
        )
    except (KeyError, TypeError, ValueError):
        return None


class _QueueController:
    """In-memory queue + atomic persistence to `queue.json`.

    `persist(jobs)` rewrites the full queue atomically. The file is small
    (a few KB for 100 jobs) so an append-only delta log is unnecessary.
    `load()` restores the queue on startup; unknown fields and bad rows
    are silently dropped so a half-corrupt file doesn't crash the GUI.
    """

    def __init__(self, paths: _AppPaths):
        self.paths = paths
        self.queue_path = paths.app_dir / QUEUE_FILE

    def load(self) -> List[Job]:
        if not self.queue_path.exists():
            return []
        try:
            data = json.loads(self.queue_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(data, dict) or data.get("version") != QUEUE_VERSION:
            return []
        out: List[Job] = []
        for raw in data.get("jobs", []):
            j = _dict_to_job(raw)
            if j is not None:
                out.append(j)
        return out

    def persist(self, jobs: List[Job]) -> None:
        payload = {
            "version": QUEUE_VERSION,
            "jobs": [_job_to_dict(j) for j in jobs],
        }
        _atomic_dump_json(self.queue_path, payload)

    def clear(self) -> None:
        try:
            self.queue_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass