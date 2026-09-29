# anime_upscaler/spandrel_teacher.py
"""Generic frozen KD teacher loaded through spandrel arch autodetection.

Lets any community checkpoint beat-tested by eval_teachers.py serve as the
distillation teacher without hand-writing its architecture.

forward_with_features() honors the same contract as SPANTeacher:
(sr, taps) with exactly TAPS=4 intermediate feature maps at LR resolution.
Taps are captured from the model's top-level stages (direct children);
non-tensor / non-NCHW outputs are skipped and the trailing output stage is
excluded so response-level loss does not double-count it.

Input convention: [0,1] RGB - probe_domain() in eval_teachers.py picked
the "01" convention for every benchmarked candidate.

Usage:
    python anime_upscaler/spandrel_teacher.py   # smoke over all candidates
"""
from pathlib import Path

import torch
import torch.nn as nn

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
    cdir = Path(__file__).resolve().parent.parent / "checkpoints" / "candidates"
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
