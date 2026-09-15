"""Model registry: scan pretrained/, parse state_dict to detect kind, expose preset catalog.

Phase 5 additions:
  - `source` field on PresetEntry: "trained" (in-house, shipped) vs
    "community" (downloadable). Models panel groups the dropdown by this.
  - `display_name` field on PresetEntry and InstalledModel: human-readable
    label used in the GUI's installed-list and preset-dropdown.
  - `is_trained` field on InstalledModel: True for files matching the
    in-house training patterns (see is_trained_filename).
  - `is_trained_filename` helper: classifies a filename as in-house by
    prefix. Add new patterns when you train new variants.
"""
import json
import logging
import math
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from . import archs

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Baseline catalog (shipped with the package). Extendable WITHOUT code edits:
# _ModelRegistry also loads extra presets from a registry.json file -- see
# _load_preset_catalog below. Each entry may set:
# Each entry may set:
#   source        "trained" | "community" (defaults to "community")
#   display_name  short human label (defaults to id)
PRESET_CATALOG = [
    {
        "id": "realesr-animevideov3",
        "filename": "realesr-animevideov3.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        "scale": 4, "kind": "srvgg",
        "license": "BSD-3-Clause", "size_mb": 2.5,
        "source": "community",
        "display_name": "Real-ESRGAN Anime v3",
        "description": "Real-ESRGAN Compact, anime-video tuned. Default recommendation.",
    },
    {
        "id": "4xLSDIRCompactv2",
        "filename": "4xLSDIRCompactv2.pth",
        "url": "https://github.com/Phhofm/models/releases/download/4xLSDIRCompact2/4xLSDIRCompactv2.pth",
        "scale": 4, "kind": "srvgg",
        "license": "CC-BY-4.0", "size_mb": 1.2,
        "source": "community",
        "display_name": "LSDIR Compact v2",
        "description": "Smallest preset; photo-pretrained but works on clean anime.",
    },
    {
        "id": "span_pix_pretrain_4x",
        "filename": "span_pix_pretrain_4x.pth",
        "url": "https://github.com/Phhofm/models/releases/download/4xNomosUni_span_multijpg/4xNomosUni_span_multijpg.pth",
        "scale": 4, "kind": "span",
        "license": "CC-BY-4.0", "size_mb": 4.3,
        "source": "community",
        "display_name": "SPAN Pix-Loss Pretrain",
        "description": "Phhofm canonical SPAN pix-loss pretrain. Higher quality than 4x animevideov3.",
    },
    {
        "id": "span_mssim_pretrain_4x",
        "filename": "span_mssim_pretrain_4x.pth",
        "url": "https://github.com/Phhofm/models/releases/download/4xNomosUni_span_multijpg_mssim/4xNomosUni_span_multijpg_mssim.pth",
        "scale": 4, "kind": "span",
        "license": "CC-BY-4.0", "size_mb": 4.3,
        "source": "community",
        "display_name": "SPAN MSSIM-Loss Pretrain",
        "description": "Phhofm SPAN mssim-loss pretrain (alternative warm-start).",
    },
    {
        "id": "eranet_N12_pretrain_325k",
        "filename": "eranet_N12_pretrain_325k.pth",
        "url": "https://github.com/NevermindNilas/eranet/releases/download/v1.1.0/eranet_N12_pretrain_325k.pth",
        "scale": 2, "kind": "era",
        "license": "MIT", "size_mb": 2.7,
        "source": "community",
        "display_name": "ERANet N12 (2x only)",
        "description": "ERANet reparameterized CNN, 2x only. Very fast on real-time anime.",
        "notes": "2x scale only; cannot upscale 4x.",
    },
    {
        "id": "AnimeSR_v2",
        "filename": "AnimeSR_v2.pth",
        "url": "https://drive.google.com/drive/folders/1gwNTbKLUjt5FlgT6PQQnBz5wFzmNUX8g",
        "scale": 4, "kind": "animesr",
        "license": "Apache-2.0", "size_mb": 67.0,
        "source": "community",
        "display_name": "AnimeSR v2 (3-frame recurrent)",
        "description": "TencentARC AnimeSR (CVPR 2022). Recurrent 3-frame; needs gdown.",
        "notes": "Download via gdown: pip install gdown && gdown --folder <url>",
    },
    {
        # In-house RFDN student distilled from the SPAN pretrain teacher (40 ep).
        # Checkpoint is vendored in-tree under pretrained/RFDN_distill_v1_4x_student.pth
        # No remote URL -- skip download for this preset.
        "id": "rfdn_distill_v1_4x",
        "filename": "RFDN_distill_v1_4x_student.pth",
        "url": "file://runs/distill_v1/student_best.pt",
        "scale": 4, "kind": "rfdn_student",
        "license": "Project-internal", "size_mb": 1.3,
        "source": "trained",
        "display_name": "RFDN Distill v1 * Trained",
        "description": "In-house trained 4x student (315K params, KD from SPAN V7, 40 epochs, 29.46 dB test PSNR with default TTA -- beats bicubic on the 99-frame held-out test split). Recommended default for anime frames.",
        "notes": "Trained in this repo. No remote download required; ships with the app.",
    },
    {
        # In-house SRVGG-body student distilled from the HFA2k PLKSR teacher
        # (Step-2 rebase, 50 ep). Vendored under pretrained/SRVGG_distill_v1_4x_student.pth.
        "id": "srvgg_distill_v1_4x",
        "filename": "SRVGG_distill_v1_4x_student.pth",
        "url": "file://runs/step2_srvgg_hfa_v1/student_best.pt",
        "scale": 4, "kind": "srvgg_student",
        "license": "Project-internal", "size_mb": 12.4,
        "source": "trained",
        "display_name": "SRVGG Distill v1 * Step-2 (HFA teacher)",
        "description": "Current best in-house student (Step-2 rebase, 317K params, KD from HFA2k PLKSR real teacher, 50 epochs: 33.55 dB / 0.9146 SSIM on the held-out test split, NIQE 7.38). Recommended default for anime frames.",
        "notes": "Trained in this repo. No remote download required; ships with the app.",
    },
    {
        "id": "RealESRGAN_x4plus_anime_6B",
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "scale": 4, "kind": "realesrgan-6B",
        "license": "BSD-3-Clause", "size_mb": 17.9,
        "source": "community",
        "display_name": "Real-ESRGAN 6B Anime (rrdb)",
        "description": "Stock Real-ESRGAN 6B anime model. Slow but high quality.",
        "notes": "Requires rrdb_arch support; not enabled in this MVP."
    },
]


# --------------------------------------------------------------------------- #
# Trained-model detection
# --------------------------------------------------------------------------- #
_TRAINED_FILENAME_PATTERNS: list = [
    # In-house training artifacts match by prefix. Extend at runtime with
    # add_trained_prefix() when you ship a freshly-trained variant.
    "RFDN_distill_v",   # e.g. RFDN_distill_v1_4x_student.pth
    "SRVGG_distill_v",  # e.g. SRVGG_distill_v1_4x_student.pth (step-2 HFA teacher)
    "anime_upscaler_",  # reserved for future internal variants
]


def add_trained_prefix(prefix: str) -> None:
    """Register an additional in-house training-artifact prefix at runtime.

    Plugins/training pipelines can call this instead of editing this module.
    Idempotent; empty or duplicated prefixes are ignored.
    """
    p = str(prefix).strip()
    if p and p.lower() not in {q.lower() for q in _TRAINED_FILENAME_PATTERNS}:
        _TRAINED_FILENAME_PATTERNS.append(p)


def is_trained_filename(filename: str) -> bool:
    """Return True if `filename` matches an in-house training artifact prefix."""
    f = str(filename).lower()
    return any(f.startswith(p.lower()) for p in _TRAINED_FILENAME_PATTERNS)


# Kind detection / support live in archs.py now -- one detector + loader per
# registered ArchSpec. These two names stay as the stable registry API used
# by app.py, widgets/models_panel.py and tests.
def _is_supported_kind(kind: str) -> bool:
    """Return True if the arch registry knows how to instantiate `kind`."""
    return archs.is_supported_kind(kind)


def _detect_kind_from_state(state: dict) -> Optional[str]:
    """Sniff the state_dict to identify the architecture (no unwrapping --
    `scan_installed` unwraps checkpoint wrappers before calling this)."""
    return archs.detect_kind_from_state(state)


@dataclass
class InstalledModel:
    filename: str
    path: Path
    kind: Optional[str] = None
    scale: int = 4
    tainted: bool = False  # warm-start taint signature (bias_mean in [0.35,0.45])
    supported: bool = True
    size_mb: float = 0.0
    license: str = ""
    description: str = ""
    # UI helpers. is_trained is True when the file matches an in-house
    # training pattern (see is_trained_filename). display_name is populated
    # from the matching preset if available; empty otherwise.
    is_trained: bool = False
    display_name: str = ""


@dataclass
class PresetEntry:
    id: str
    filename: str
    # url may be an empty string for local-only models (trained presets,
    # entries from a registry.json extension file without a remote source).
    url: str
    scale: int
    kind: str
    license: str
    size_mb: float
    description: str
    notes: str = ""
    # "trained" = in-house (no remote URL). "community" = downloadable from the URL.
    # Drives the GUI's Trained-vs-Community grouping.
    source: str = "community"
    # Human-readable label shown in the preset dropdown and installed list.
    # Falls back to `id` when empty.
    display_name: str = ""


def _validate_preset_entry(raw: dict, origin: str) -> Optional[PresetEntry]:
    """Validate one catalog entry; log and skip malformed ones.

    Required keys: id, filename, scale, kind, license, size_mb, description.
    Optional keys: url (default ""), notes, source, display_name.
    Unknown fields are ignored so the JSON schema can evolve.
    """
    if not isinstance(raw, dict):
        log.warning("registry preset: skipping non-object entry in %s", origin)
        return None
    required = ("id", "filename", "scale", "kind", "license", "size_mb", "description")
    missing = [k for k in required if k not in raw]
    if missing:
        log.warning("registry preset: skipping entry %r in %s (missing %s)",
                    raw.get("id", "?"), origin, ", ".join(missing))
        return None
    try:
        scale = int(raw["scale"])
        size_mb = float(raw["size_mb"])
    except (TypeError, ValueError):
        log.warning("registry preset: skipping entry %r in %s (bad scale/size_mb)",
                    raw.get("id", "?"), origin)
        return None
    if scale <= 0:
        log.warning("registry preset: skipping entry %r in %s (scale <= 0)",
                    raw.get("id", "?"), origin)
        return None
    return PresetEntry(
        id=str(raw["id"]),
        filename=str(raw["filename"]),
        url=str(raw.get("url", "") or ""),
        scale=scale,
        kind=str(raw["kind"]),
        license=str(raw["license"]),
        size_mb=size_mb,
        description=str(raw["description"]),
        notes=str(raw.get("notes", "")),
        source=str(raw.get("source", "community")),
        display_name=str(raw.get("display_name", "")),
    )


def _bundled_registry_json() -> Optional[Path]:
    """registry.json shipped next to the app (apps/anime_upscaler_gui/data/)."""
    candidate = Path(__file__).resolve().parent.parent / "data" / "registry.json"
    return candidate if candidate.is_file() else None


def _load_preset_catalog(explicit_path: Optional[Path] = None) -> List[PresetEntry]:
    """Baseline catalog + entries merged from an extension registry.json.

    The JSON format is the same as PRESET_CATALOG entries, under a top-level
    "models" array. Entries with an id already present in the baseline
    override it, so presets can be re-skinned or retired from data without
    touching code. Malformed entries are skipped with a warning instead of
    breaking model loading.
    """
    by_id: Dict[str, PresetEntry] = {}
    order: List[str] = []
    for p in PRESET_CATALOG:
        e = _validate_preset_entry(p, "<builtin PRESET_CATALOG>")
        if e is not None:
            if e.id in by_id:
                log.warning("registry preset: duplicate id %r in PRESET_CATALOG", e.id)
            by_id[e.id] = e
            order.append(e.id)
    # Extension candidates, later ones override by id:
    #   1. bundled   apps/anime_upscaler_gui/data/registry.json (ships with app)
    #   2. explicit  user extension (typically <app-data>/registry.json)
    candidates: List[Path] = []
    if _bundled_registry_json() is not None:
        candidates.append(_bundled_registry_json())
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    for path in candidates:
        if not path.is_file():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            models = data.get("models") if isinstance(data, dict) else None
            if isinstance(models, list):
                for raw in models:
                    e = _validate_preset_entry(raw, str(path))
                    if e is None:
                        continue
                    if e.id not in by_id:
                        order.append(e.id)
                    by_id[e.id] = e  # override replaces, base position kept
            else:
                log.warning("registry extension %s has no models array; ignored", path)
        except (OSError, json.JSONDecodeError) as ex:
            log.warning("registry extension %s failed to load: %s", path, ex)
    return [by_id[i] for i in order]


class _ModelRegistry:
    """Combines on-disk scan (pretrained/) with the preset catalog (baseline
    PRESET_CATALOG plus any registry.json extension file)."""

    def __init__(self, pretrained_dir: Path,
                 presets_path: Optional[Path] = None):
        self.pretrained_dir = Path(pretrained_dir)
        self.pretrained_dir.mkdir(parents=True, exist_ok=True)
        self._installed: List[InstalledModel] = []
        self._presets: List[PresetEntry] = _load_preset_catalog(presets_path)
        # filename -> display_name; populated so InstalledModel.display_name
        # resolves when a known preset is on disk.
        self._filename_to_display: Dict[str, str] = {
            p.filename: (p.display_name or p.id)
            for p in self._presets
        }

    # ---- scan ----
    def scan_installed(self) -> List[InstalledModel]:
        out: List[InstalledModel] = []
        for p in sorted(self.pretrained_dir.glob("*.pth")):
            try:
                size_mb = p.stat().st_size / 1e6
            except OSError:
                size_mb = 0.0
            kind = None
            scale = 4
            tainted = False
            supported = False
            try:
                ck = torch.load(p, map_location="cpu", weights_only=False)
                if isinstance(ck, dict):
                    # Common top-level wrappers across training pipelines:
                    #   params / params_ema / state_dict / model_state_dict: Real-ESRGAN, SPAN, etc.
                    #   student: anime_upscaler KD trainer (student + adapters + metadata).
                    # Use the first key whose value is itself a dict (i.e. a state_dict,
                    # not a scalar metadata field like "epoch"/"val_psnr").
                    state = None
                    for key in ("params", "params_ema", "state_dict",
                                "model_state_dict", "student", "adapters"):
                        if key in ck and isinstance(ck[key], dict) and len(ck[key]) > 5:
                            state = ck[key]
                            break
                    if state is None:
                        state = ck
                else:
                    state = ck
                kind = _detect_kind_from_state(state)
                if kind:
                    supported = _is_supported_kind(kind)
                    spec = archs.spec_of(kind)
                    if spec is not None:
                        scale = spec.default_scale
                    if kind == "srvgg_student":
                        # Sniff the upscale factor from the last body conv
                        # (out_ch = 3 * scale^2). Mirrors archs.build srvgg_student.
                        for k in sorted((k for k in state.keys()
                                         if k.endswith(".weight") and k.startswith("body.")
                                         and k.split(".")[1].isdigit()), reverse=True):
                            w = state[k]
                            if hasattr(w, "shape") and len(getattr(w, "shape", [])) == 4:
                                ratio = int(w.shape[0]) // 3
                                s = int(round(math.sqrt(max(ratio, 1))))
                                if s * s == ratio and s in (2, 3, 4, 8):
                                    scale = s
                                break
                # Taint signature (warm-start poisoning)
                bias = state.get("upsampler.0.bias") if isinstance(state, dict) else None
                if bias is not None:
                    bm = float(bias.mean())
                    bs = float(bias.std())
                    tainted = (0.35 <= bm <= 0.45) and (bs < 0.05)
            except Exception:
                pass
            out.append(InstalledModel(
                filename=p.name, path=p, kind=kind, scale=scale,
                tainted=tainted, supported=supported, size_mb=round(size_mb, 2),
                is_trained=is_trained_filename(p.name),
            ))
        self._installed = out
        # Resolve display_name from the preset catalog (best-effort, by filename).
        for m in self._installed:
            m.display_name = self._filename_to_display.get(m.filename, "")
        return self._installed

    # ---- presets ----
    def presets(self) -> List[PresetEntry]:
        # Trained models listed first (so the GUI dropdown can pin them at the top),
        # then community models in catalog order.
        trained = [p for p in self._presets if p.source == "trained"]
        community = [p for p in self._presets if p.source != "trained"]
        return trained + community

    def preset_by_id(self, pid: str) -> Optional[PresetEntry]:
        for p in self._presets:
            if p.id == pid:
                return p
        return None

    # ---- import ----
    def import_local(self, src: Path) -> InstalledModel:
        """Copy a user-picked .pth/.safetensors into pretrained/."""
        src = Path(src)
        if not src.exists():
            raise FileNotFoundError(src)
        dest = self.pretrained_dir / src.name
        shutil.copy2(src, dest)
        installed = self.scan_installed()
        for m in installed:
            if m.path == dest:
                return m
        return InstalledModel(filename=src.name, path=dest)

    # ---- download ----
    def download(self, url: str, dest_filename: str,
                 on_progress=None) -> Path:
        """Stream-download a URL to pretrained/dest_filename. Polls bytes-written."""
        dest = self.pretrained_dir / dest_filename
        if dest.exists():
            return dest  # already downloaded
        # Special-case Google Drive folder links (gdown recommended for these)
        if "drive.google.com" in url and "/folders/" in url:
            raise RuntimeError(
                "Google Drive folder downloads need gdown. Run:\n"
                f"  pip install gdown && gdown --folder {url}\n"
                "then click Refresh."
            )
        # Stream with urllib; fall back to requests if user has it.
        req = urllib.request.Request(url, headers={"User-Agent": "anime-upscaler-gui/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            written = 0
            chunk = 64 * 1024
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                written += len(buf)
                if on_progress and total:
                    on_progress(written, total)
        return dest
