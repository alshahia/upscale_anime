# anime_upscaler/teacher.py + spandrel_teacher.py
"""Frozen teachers for knowledge distillation.

SPANTeacher: the project's own anime fine-tuned SPAN (neosr arch).
  Reuses anime_sr.models.span.neosr_span.NeosrSPAN so no architecture is
  duplicated. Weights come from checkpoints/NEOSR_SPAN_V7_ANIME_001/finetune_best.pth
  (EMA state dict) by default; falls back to pretrained/span_pix_pretrain_4x.pth.

SpandrelTeacher: generic frozen KD teacher loaded through spandrel arch autodetection.
  Lets any community checkpoint serve as the distillation teacher without
  hand-writing its architecture.

Both teachers always run frozen, in eval mode, on [0, 1] RGB input.
forward_with_features() exposes intermediate features at LR resolution
for feature distillation.
"""
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from anime_sr.models.span.neosr_span import NeosrSPAN

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

DEFAULT_CKPT = ROOT / "checkpoints" / "NEOSR_SPAN_V7_ANIME_001" / "finetune_best.pth"
FALLBACK_CKPT = ROOT / "pretrained" / "span_pix_pretrain_4x.pth"
FEATURE_CHANNELS = 48
# Student tap i is distilled against teacher tap TAP_ORDER[i].
TAP_ORDER = ["conv_1", "block_2", "block_4", "conv_cat"]


class SPANTeacher(nn.Module):
    """Wraps NeosrSPAN with frozen weights and named feature taps."""

    def __init__(self, checkpoint_path=None, device="cuda"):
        super().__init__()
        ckpt_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CKPT
        if not ckpt_path.exists():
            ckpt_path = FALLBACK_CKPT
        if not ckpt_path.exists():
            raise FileNotFoundError("no teacher checkpoint found")

        self.net = NeosrSPAN(num_in_ch=3, num_out_ch=3,
                             feature_channels=FEATURE_CHANNELS, upscale=4)
        sd = self._load_state_dict(ckpt_path)
        missing, unexpected = self.net.load_state_dict(sd, strict=False)
        # 'mean' buffer may be absent in older checkpoints; default 0.5 is correct.
        real_missing = [k for k in missing if k != "mean"]
        if real_missing or unexpected:
            raise RuntimeError(
                f"teacher load mismatch from {ckpt_path.name}: "
                f"missing={real_missing[:5]} unexpected={list(unexpected)[:5]}")
        self.net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device
        self.ckpt_name = ckpt_path.name
        n_params = sum(p.numel() for p in self.net.parameters())
        print(f"[teacher] {ckpt_path.name}: {n_params:,} params (frozen)")

    @staticmethod
    def _load_state_dict(ckpt_path):
        # Fine-tuned project checkpoints pickle utils.config.Config -> need
        # src/ on sys.path and weights_only=False. Official pretrains are plain.
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if isinstance(ckpt, dict):
                for key in ("params_ema", "params", "state_dict"):
                    if key in ckpt and hasattr(ckpt[key], "keys"):
                        return ckpt[key]
            if hasattr(ckpt, "keys"):
                return ckpt
        except Exception:
            import warnings
            warnings.warn("Loading checkpoint with pickle fallback - only use with trusted sources!", UserWarning, stacklevel=2)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            for key in ("ema_state_dict", "ema_model_state_dict",
                        "model_state_dict", "params_ema", "params", "state_dict"):
                if key in ckpt and hasattr(ckpt[key], "keys"):
                    return ckpt[key]
        return ckpt

    @torch.no_grad()
    def forward(self, lr01):
        """[0,1] LR -> [0,1] SR."""
        return self.net(lr01)

    @torch.no_grad()
    def forward_with_features(self, lr01) -> List[torch.Tensor]:
        """[0,1] LR -> ([0,1] SR, feature taps aligned with TAP_ORDER).

        Mirrors NeosrSPAN.forward but keeps intermediates. All taps are
        48-channel tensors at LR resolution.

        NOTE: with norm=False (this architecture's default) NeosrSPAN feeds
        the input through WITHOUT mean subtraction - verified against
        net(); do not "fix" this into subtracting net.mean.
        """
        net = self.net
        x = lr01
        if net.is_norm:                       # keep exact parity with native fwd
            x = (x - net.mean.type_as(x)) * net.img_range
        f_conv1 = net.conv_1(x)
        b1, _, _ = net.block_1(f_conv1)
        b2, _, _ = net.block_2(b1)
        b3, _, _ = net.block_3(b2)
        b4, _, _ = net.block_4(b3)
        b5, _, _ = net.block_5(b4)
        b6, b5_2, _ = net.block_6(b5)
        b6 = net.conv_2(b6)
        cat_in = torch.cat([f_conv1, b6, b1, b5_2], dim=1)
        f_cat = net.conv_cat(cat_in)
        sr = net.upsampler(f_cat)
        feats = {"conv_1": f_conv1, "block_2": b2, "block_4": b4, "conv_cat": f_cat}
        return sr.clamp(0, 1), [feats[t] for t in TAP_ORDER]


if __name__ == "__main__":
    # Stage-2 smoke check: teacher loads and runs on a real frame crop.
    import numpy as np
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = SPANTeacher(device=dev)
    frame = sorted((ROOT / "data" / "anime_video_frames").glob("*.png"))[0]
    hr = Image.open(frame).convert("RGB").crop((400, 300, 788, 636))
    lr = hr.resize((hr.width // 4, hr.height // 4), Image.BICUBIC)
    lr01 = torch.from_numpy(np.asarray(lr)).permute(2, 0, 1)[None].float().div_(255).to(dev)
    sr, feats = t.forward_with_features(lr01)
    print("sr:", tuple(sr.shape), "| taps:", [tuple(f.shape) for f in feats],
          "| sr range [%.3f, %.3f]" % (sr.min(), sr.max()))

# ============================================================================ #
# Phase 3 (handoff A.3): RealESR-style frozen teachers (animevideov3, lsdir)
# ---------------------------------------------------------------------------- #
# SRVGGNetCompact is copied (verbatim) from apps/.../archs.py so the
# training pipeline stays hermetic (no GUI-package dependency). If you edit
# the stub there, mirror it here.
# ============================================================================ #


class SRVGGNetCompact(nn.Module):
    """Real-ESRGAN compact variant (animevideov3, LSDIRCompactv2).

    body is a ModuleList of (Conv, Act) pairs + final Conv; the forward adds
    nearest-upsampled input as a residual so the network learns high-frequency
    detail only.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16,
                 upscale=4, act_type="prelu"):
        super().__init__()
        if act_type == "prelu":
            Act = lambda: nn.PReLU(num_feat)
        elif act_type == "leakyrelu":
            Act = lambda: nn.LeakyReLU(0.1, inplace=True)
        else:
            Act = lambda: nn.ReLU(inplace=True)
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(Act())
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(Act())
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)
        self.upscale = upscale

    def forward(self, x):
        import torch.nn.functional as _F
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        out = out + _F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out


def _load_state_dict_realesr(ckpt_path):
    """Load state_dict from a RealESR-style .pth, handling common wrappers.

    Tries params_ema, params, state_dict, ema_state_dict in that order;
    returns the inner dict. Same logic as apps/.../archs.py::_load_state_dict.
    """
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        import warnings
        warnings.warn("Loading checkpoint with pickle fallback - only use with trusted sources!", UserWarning, stacklevel=2)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("params_ema", "params", "ema_state_dict",
                     "state_dict", "model_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if all(hasattr(v, "shape") for v in ckpt.values()):
            return ckpt
    return ckpt


class RealESRTeacher(nn.Module):
    """Frozen RealESR-style compact teacher (animevideov3 / LSDIR).

    Wraps SRVGGNetCompact with loaded weights, eval mode, no grad. Unlike
    SPANTeacher it does NOT expose intermediate taps (SRVGG is a single
    body conv stack with no skip taps) -- distillation against this teacher
    uses SR-response L1, not feature distillation.
    """

    def __init__(self, ckpt_path, scale=4, device="cuda"):
        super().__init__()
        self.net = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                                     num_conv=16, upscale=scale)
        sd = _load_state_dict_realesr(ckpt_path)
        missing, unexpected = self.net.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"RealESR teacher load mismatch from {Path(ckpt_path).name}: "
                f"missing={missing[:5]} unexpected={unexpected[:5]}")
        self.net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device
        self.scale = scale
        self.ckpt_name = Path(ckpt_path).name
        n_params = sum(p.numel() for p in self.net.parameters())
        print(f"[teacher] {self.ckpt_name}: {n_params:,} params (frozen) [real-esr]")

    @torch.no_grad()
    def forward(self, lr01):
        """[0, 1] LR -> [0, 1] SR (the teacher natively outputs RGB in this range)."""
        return self.net(lr01)


# Module-level registry used by distill.py to dispatch --teacher name to class.
# Each entry: (class_or_callable, dict_of_kwargs_for_constructor).
TEACHERS = {
    "span":         (SPANTeacher,    {}),
    "animevideov3": (RealESRTeacher, {"ckpt_path": str(ROOT / "pretrained" / "realesr-animevideov3.pth")}),
    "lsdir":        (RealESRTeacher, {"ckpt_path": str(ROOT / "pretrained" / "4xLSDIRCompactv2.pth")}),
}


# ============================================================================ #
# SpandrelTeacher: generic frozen KD teacher loaded through spandrel arch autodetection
# ============================================================================ #
# Lets any community checkpoint beat-tested by eval_teachers.py serve as the
# distillation teacher without hand-writing its architecture.
#
# forward_with_features() honors the same contract as SPANTeacher:
# (sr, taps) with exactly TAPS=4 intermediate feature maps at LR resolution.
# Taps are captured from the model's top-level stages (direct children);
# non-tensor / non-NCHW outputs are skipped and the trailing output stage is
# excluded so response-level loss does not double-count it.
#
# Input convention: [0,1] RGB - probe_domain() in eval_teachers.py picked
# the "01" convention for every benchmarked candidate.

TAPS = 4
_EXTRA_INSTALLED = False


class _StageRecorder:
    """Forward-hook collector for top-level stage modules."""

    def __init__(self, modules):
        self.captured = []
        self.handles = [m.register_forward_hook(self._hook) for m in modules]

    def _hook(self, _module, _inp, out):
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, torch.Tensor) and out.ndim == 4:
            self.captured.append(out)

    def remove(self):
        for h in self.handles:
            h.remove()


def _stage_modules(net):
    """Direct children of the outermost module (descend single wrappers)."""
    kids = list(net.children())
    while len(kids) == 1:
        kids = list(kids[0].children())
    return kids


class SpandrelTeacher(nn.Module):
    """Frozen spandrel-loaded teacher exposing TAPS LR-resolution features."""

    def __init__(self, ckpt_path, device="cuda", half=False):
        super().__init__()
        ckpt_path = Path(ckpt_path)
        global _EXTRA_INSTALLED
        from spandrel import ModelLoader
        if not _EXTRA_INSTALLED:
            import spandrel_extra_arches
            spandrel_extra_arches.install()
            _EXTRA_INSTALLED = True
        desc = ModelLoader().load_from_file(str(ckpt_path))
        net = desc.model.to(device).eval()
        # "half" means CUDA autocast, NOT .half() weights: attention-based
        # archs (GRL etc.) break under manual fp16 casts but run fine under
        # autocast with fp32 master weights.
        self._half = bool(half) and str(device).startswith("cuda")
        self.net = net
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device
        self.ckpt_name = ckpt_path.name
        n_params = sum(p.numel() for p in self.net.parameters())

        # Discover stage layout once with a dummy forward.
        rec = _StageRecorder(_stage_modules(self.net))
        try:
            with torch.no_grad():
                self.net(self._prep(torch.rand(1, 3, 48, 48, device=device)))
            n_stages = len(rec.captured)
        finally:
            rec.remove()
        usable = list(range(n_stages))
        if len(usable) > 1:
            usable = usable[:-1]        # drop trailing output-like stage
        k = min(TAPS, len(usable))
        self._tap_idx = sorted(
            {round(v) for v in torch.linspace(0, len(usable) - 1, k).tolist()})

        # Channel count per tap for adapter construction.
        _, taps = self.forward_with_features(
            torch.rand(1, 3, 48, 48, device=device))
        self.tap_channels = [t.shape[1] for t in taps]
        print(f"[teacher/spandrel] {ckpt_path.name}: "
              f"{n_params:,} params, stages={n_stages}, "
              f"taps={self._tap_idx}, chans={self.tap_channels}, "
              f"half={self._half}")

    def _prep(self, lr01):
        return lr01.to(self.device)

    def _ctx(self):
        return torch.autocast("cuda", dtype=torch.float16, enabled=self._half)

    def _finite(self, t):
        # fp16 attention can overflow on some inputs (verified data-dependent);
        # callers retry those batches in full fp32.
        return not self._half or bool(torch.isfinite(t).all())

    @torch.no_grad()
    def forward(self, lr01):
        """[0,1] float32 LR -> [0,1] float32 SR (fp32-retried if overflow)."""
        x = self._prep(lr01)
        with self._ctx():
            out = self.net(x)
        if not self._finite(out):
            out = self.net(x)
        return out.float().clamp(0, 1)

    @torch.no_grad()
    def forward_with_features(self, lr01):
        """[0,1] LR -> ([0,1] SR, list of TAPS float32 feature maps)."""
        rec = _StageRecorder(_stage_modules(self.net))
        try:
            with self._ctx():
                sr = self.net(self._prep(lr01))
            taps_ok = all(self._finite(f) for f in rec.captured)
            if not (self._finite(sr) and taps_ok):
                # SR finite is NOT enough: an intermediate tap can overflow
                # while later layers recover - MSE against an inf tap then
                # poisons the student. Retry the whole pass in exact fp32.
                rec.captured.clear()      # drop half-precision stage captures
                sr = self.net(self._prep(lr01))   # exact fp32 retry
        finally:
            rec.remove()
        feats_all = rec.captured
        taps = [feats_all[i].float() for i in self._tap_idx]
        return sr.float().clamp(0, 1), taps


if __name__ == "__main__":
    # Smoke: load every candidate under checkpoints/candidates.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cdir = ROOT / "checkpoints" / "candidates"
    for p in sorted(cdir.glob("*")):
        if p.suffix not in (".pth", ".safetensors"):
            continue
        try:
            t = SpandrelTeacher(p, device=dev)
            x = torch.rand(1, 3, 32, 32, device=dev)
            sr, taps = t.forward_with_features(x)
            print(p.name, "->", tuple(sr.shape),
                  [tuple(f.shape) for f in taps])
        except Exception as e:      # noqa: BLE001 smoke reports failures
            print(p.name, "FAIL:", type(e).__name__, str(e)[:120])

