"""Architecture registry for the anime-upscaler-gui.

Replaces the vendored archs.py. Model architectures live in the package-local
_models/ tree, generated from src/anime_sr/models (single source of truth) by
scripts/sync_vendored_models.py, so the GUI wheel is standalone (no anime_sr).

Supported architectures (kind strings used by registry.py and the GUI dropdown):
  - "srvgg"        SRVGGNetCompact (Real-ESRGAN compact: animevideov3, LSDIRCompactv2)
  - "span"         NeosrSPAN (Phhofm SPAN pretrains)
  - "era"          ERANet (NevermindNilas; 2x only)
  - "animesr"      MSRSWVSR (TencentARC AnimeSR; recurrent 3-frame)
  - "rfdn_student" RFDN teacher-distilled student (~315K params, 4x only)
"""
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import torch
import torch.nn as nn

from ._models.students import RFDN, TinySRVGGStudent
from ._models.span.neosr_span import NeosrSPAN
from ._models.eranet import ERANet
from ._models.msrsr import MSRSWVSR


# ============================================================================ #
# Loader helpers
# ============================================================================ #
# Opt-in escape hatch for legacy checkpoints that contain pickled (non-tensor)
# objects. Unpickling such files can execute arbitrary code, so this is off by
# default and the loader warns loudly whenever it is used.
UNSAFE_PICKLE_ENV_VAR = "ANIME_UPSCALER_ALLOW_UNSAFE_PICKLE"


def _unsafe_pickle_allowed() -> bool:
    """True only when the user deliberately set the exact opt-in value."""
    return os.environ.get(UNSAFE_PICKLE_ENV_VAR) == "1"


def _load_state_dict(path, *, allow_unsafe_pickle: Optional[bool] = None):
    """Load a checkpoint and extract its state dict (handles params/state_dict wrappers).

    The default is weights_only=True: pickled objects are refused because
    unpickling them can execute arbitrary code. A legacy checkpoint may be
    loaded only when the user has explicitly opted in with
    ANIME_UPSCALER_ALLOW_UNSAFE_PICKLE=1 (or a caller passes
    allow_unsafe_pickle=True); both paths emit a loud warning and must only be
    used for files whose origin is trusted.
    """
    if allow_unsafe_pickle is None:
        allow_unsafe_pickle = _unsafe_pickle_allowed()
    try:
        sd = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as safe_error:
        if not allow_unsafe_pickle:
            raise RuntimeError(
                f"refusing to load {path}: not a plain tensor checkpoint "
                f"(set {UNSAFE_PICKLE_ENV_VAR}=1 only for files you trust): "
                f"{safe_error}"
            ) from safe_error
        warnings.warn(
            f"UNSAFE pickle load: {UNSAFE_PICKLE_ENV_VAR}=1 permits arbitrary "
            f"code execution while loading {path}. Only load checkpoints you "
            "trust.",
            UserWarning,
            stacklevel=2,
        )
        sd = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict):
        for k in ('params', 'params_ema', 'state_dict', 'model_state_dict', 'student'):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    return sd


def _save_sr(out, path):
    """Save a (1, 3, H, W) tensor in [0,1] as a uint8 BGR PNG/JPG via cv2."""
    arr = (out.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), arr)


# ============================================================================ #
# Extensible architecture registry
# ============================================================================ #
@dataclass(frozen=True)
class Capability:
    """Runtime abilities the pipeline / TRT / VRAM-guard layers query."""
    tiled: bool = True
    tensorrt: bool = True
    batch_video: bool = True
    recurrent_frames: int = 0
    pad_multiple: int = 0
    out_multiple: int = 0
    tta: bool = True


@dataclass(frozen=True)
class ArchSpec:
    """How to build, detect and query one architecture kind."""
    kind: str
    detect: Callable[[dict], Optional[str]]
    loader: Callable[[str], nn.Module]
    default_scale: int = 4
    capability: Capability = Capability()


_REGISTRY: Dict[str, ArchSpec] = {}


def register_arch(spec: ArchSpec) -> ArchSpec:
    """Register (or replace) an architecture spec. Last registration wins."""
    _REGISTRY[str(spec.kind)] = spec
    return spec


def unregister_arch(kind: str) -> None:
    """Remove a registered kind (mainly for tests/experiments)."""
    _REGISTRY.pop(str(kind), None)


def registered_kinds() -> List[str]:
    """Kinds currently registered, in registration order."""
    return list(_REGISTRY)


def spec_of(kind: str) -> Optional[ArchSpec]:
    """Look up the spec for `kind`, or None."""
    return _REGISTRY.get(str(kind))


def capabilities(kind: str) -> Capability:
    """Capability flags for `kind`; unknown kinds get the safe defaults."""
    s = spec_of(kind)
    return s.capability if s else Capability()


def is_supported_kind(kind: str) -> bool:
    """True when the registry knows how to build this kind."""
    return str(kind) in _REGISTRY


def detect_kind_from_state(state: dict) -> Optional[str]:
    """Sniff the state_dict to identify the architecture."""
    if not isinstance(state, dict):
        return None
    for s in _REGISTRY.values():
        try:
            if s.detect(state):
                return s.kind
        except Exception:
            continue
    return None


def build(kind: str, ckpt_path) -> nn.Module:
    """Build + load + set eval() on the right architecture."""
    s = spec_of(kind)
    if s is None:
        raise ValueError("unsupported arch kind: %r. registered kinds: %s"
                         % (kind, ", ".join(sorted(_REGISTRY))))
    m = s.loader(str(ckpt_path))
    m.eval()
    return m


# ---- per-kind detection helpers ---------------------------------------- #

def _detect_span(state: dict) -> Optional[str]:
    keys = list(state.keys())
    keyset = set(keys)
    if "conv_1.sk.weight" in keyset and "upsampler.0.weight" in keyset:
        return "span"
    return None


def _detect_srvgg(state: dict) -> Optional[str]:
    keys = list(state.keys())
    keyset = set(keys)
    if any(k == "body.0.weight" for k in keys) and "body.34.weight" in keyset:
        return "srvgg"
    return None


def _load_srvgg(ckpt_path: str) -> nn.Module:
    from ._models.teachers.spandrel import SRVGGNetCompact
    m = SRVGGNetCompact()
    m.load_state_dict(_load_state_dict(ckpt_path), strict=False)
    return m


def _load_srvgg_student(ckpt_path: str) -> nn.Module:
    """SRVGG-body distilled student (Phase 4 I3)."""
    sd = _load_state_dict(ckpt_path)
    num_feat = 52
    first_w = sd.get('body.0.weight') if isinstance(sd, dict) else None
    if first_w is not None and hasattr(first_w, 'shape') and len(first_w.shape) == 4:
        num_feat = int(first_w.shape[0])
    body_conv_keys = sorted([k for k in sd.keys()
                             if k.startswith('body.') and k.endswith('.weight')
                             and (k.split('.')[1].isdigit()
                                  and int(k.split('.')[1]) % 2 == 0)],
                            key=lambda k: int(k.split('.')[1]))
    num_conv = max(len(body_conv_keys) - 2, 1)
    scale = 4
    if len(body_conv_keys) >= 2:
        last_w = sd[body_conv_keys[-1]]
        if hasattr(last_w, 'shape') and len(last_w.shape) == 4:
            out_ch = int(last_w.shape[0])
            ratio = out_ch // 3
            sq = int(round(math.sqrt(max(ratio, 1))))
            if sq * sq == ratio and sq in (2, 3, 4, 8):
                scale = sq
    m = TinySRVGGStudent(num_feat=num_feat, num_conv=num_conv, scale=scale)
    m.load_state_dict(sd, strict=False)
    return m


def _load_era(ckpt_path: str) -> nn.Module:
    m = ERANet(scale=2, C=32, N=12)
    m.load_state_dict(_load_state_dict(ckpt_path), strict=False)
    return m


def _load_animesr(ckpt_path: str) -> nn.Module:
    m = MSRSWVSR()
    m.load_state_dict(_load_state_dict(ckpt_path), strict=False)
    return m


def _load_rfdn_student(ckpt_path: str) -> nn.Module:
    """RFDN teacher-distilled student (Phase 4 I1)."""
    sd = _load_state_dict(ckpt_path)
    shortcut_mode = "bicubic"
    if isinstance(sd, dict) and "args" in sd:
        args = sd["args"]
        if isinstance(args, dict) and "shortcut_mode" in args:
            shortcut_mode = args["shortcut_mode"]
    m = RFDN(shortcut_mode=shortcut_mode)
    m.load_state_dict(sd, strict=False)
    return m


def _load_span(ckpt_path: str) -> nn.Module:
    m = NeosrSPAN(num_in_ch=3, num_out_ch=3, feature_channels=48, upscale=4)
    m.load_state_dict(_load_state_dict(ckpt_path), strict=False)
    return m


# ============================================================================ #
# Register builtin architectures
# ============================================================================ #
def _register_builtin_archs():
    register_arch(ArchSpec(
        kind="srvgg",
        detect=_detect_srvgg,
        loader=_load_srvgg,
        default_scale=4,
    ))
    register_arch(ArchSpec(
        kind="span",
        detect=_detect_span,
        loader=_load_span,
        default_scale=4,
    ))
    register_arch(ArchSpec(
        kind="era",
        detect=lambda s: "era" if "head.weight" in s and "body.0.conv.sk.weight" in s else None,
        loader=_load_era,
        default_scale=2,
    ))
    register_arch(ArchSpec(
        kind="animesr",
        detect=lambda s: "animesr" if "recurrent_cell.conv_s1_first.0.weight" in s else None,
        loader=_load_animesr,
        default_scale=4,
        capability=Capability(recurrent_frames=3, tta=False, tiled=False, tensorrt=False, batch_video=False, pad_multiple=4, out_multiple=4),
    ))
    register_arch(ArchSpec(
        kind="rfdn_student",
        detect=lambda s: "rfdn_student" if "head.weight" in s and "blocks.0.d1.weight" in s else None,
        loader=_load_rfdn_student,
        default_scale=4,
    ))
    register_arch(ArchSpec(
        kind="srvgg_student",
        detect=lambda s: "srvgg_student" if "body.0.weight" in s and "body.24.weight" in s else None,
        loader=_load_srvgg_student,
        default_scale=4,
    ))


_register_builtin_archs()
