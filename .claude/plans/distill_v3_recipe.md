# Plan: v3 distillation recipe — implement Top5 techniques to lift student PSNR above bicubic

**Goal**: lift v1 student from 29.32 dB test PSNR (currently *below* bicubic 29.45 dB) to ~29.7–30.2 dB by adding the five lowest-effort, highest-impact techniques from `docs/distill_v3_research/REPORT.md` §3.

**Constraints**:
- 315K-param RFDN student stays unchanged (architecture is fine; plateau is on the recipe).
- SPAN-V7 teacher unchanged.
- Single-GPU RTX 4000 8GB, ~20s/epoch current; target ≤60s/epoch.
- All new dependencies already present in `.venv` (`lpips`, `piq`, `skimage`, `timm`, `imageio_ffmpeg`).
- No file outside `anime_upscaler/` is touched except a tiny CLI flag on `anime_upscaler/infer.py` for TTA and the GUI settings panel.

**Baseline numbers (must not regress)**:
- v1 student test PSNR = **29.32 dB**, bicubic = 29.45, teacher = 30.44, retention of teacher gain = -12.6% (negative!).
- v1 val PSNR plateaus at 29.88 dB; test 29.32 (val/test gap = 0.56 dB).
- ~315,844 student params, ~11MB teacher.

---

## Phase 1 — Loss rebalance + Charbonnier + MS-SSIM + LPIPS perceptual (rec.1+4)

**Files**: `anime_upscaler/distill.py` (one file; ~25 LOC change in the train loop).

**Imports at top of file**:
```python
from piq import charbonnier_loss as _charbonnier
from piq import ssim as _piq_ssim
import lpips as _lpips_pkg
```

**Lazy-init LPIPS module-level**:
```python
_lpips_net = None
def _get_lpips(device):
    global _lpips_net
    if _lpips_net is None:
        _lpips_net = _lpips_pkg.LPIPS(net="vgg").to(device).eval()
        for p in _lpips_net.parameters():
            p.requires_grad_(False)
    return _lpips_net
```

**Replace the inner loss block (currently lines ~230–240 in `distill.py`)**:
```python
# OLD:
loss_resp = l1(s_out, t_out)
loss_feat = sum(MSE_terms) / N
loss_gt   = l1(s_out, hr)
loss      = loss_resp + 0.5 * loss_feat + 0.2 * loss_gt

# NEW:
loss_resp = _charbonnier(s_out, t_out, eps=1e-3)                          # was L1
loss_feat = sum(cos(adapter(s_f), F.normalize(t_f, dim=1))) / N           # was plain MSE
ssim_term = 1.0 - _piq_ssim(s_out.clamp(0,1), hr, data_range=1.0, reduction="mean")
lp_term   = _get_lpips(device)(s_out.clamp(0,1)*2-1, hr*2-1).mean()       # LPIPS expects [-1,1]
loss_gt   = 0.5 * l1(s_out, hr) + 0.2 * ssim_term + 0.05 * lp_term
loss      = 0.3 * loss_resp + 0.5 * loss_feat + 1.0 * loss_gt            # rebalanced weights
```

**Validation gate**:
- `--smoke` run completes without NaN.
- `train_log.csv` shows `loss_total` finite and `loss_gt` larger than before (SSIM+LPIPS have different scales; gradients are correct).
- 5-epoch smoke (no val comparison) shows all four loss terms finite and declining.

**Risk**: `piq.charbonnier_loss` returns per-pixel mean (already /N). LPIPS on `[B,3,H,W]` returns `[B,1,1,1]`; `.mean()` is correct.

---

## Phase 2 — EMA of student weights (rec.3)

**Files**: `anime_upscaler/distill.py` (~30 LOC).

Add right after optimizer creation:
```python
class _EMA:
    """Minimal ModelEmaV2 in 20 LOC; no timm dep needed."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1-d)
            else:
                self.shadow[k].copy_(v.detach())
    def state_dict(self):
        return self.shadow

student_ema = _EMA(student, decay=0.999)
```

After `optimizer.step()` each iteration: `student_ema.update(student)`.

Replace the save block so `student_best.pt` records EMA state instead of raw state:
```python
state_raw = {..., "student": student.state_dict(), ...}
state_ema = {..., "student": student_ema.state_dict(), ...}
torch.save(state_raw, out_dir / "student_last.pt")
if metrics_ema["student"][0] > best_psnr:
    best_psnr = metrics_ema["student"][0]
    torch.save(state_ema, out_dir / "student_best.pt")
```

Where `metrics_ema` is computed by `student.load_state_dict(student_ema.state_dict()) → evaluate() → swap back`, **once per epoch end** (cheap; reuse existing `evaluate()` function).

**Validation gate**:
- `student_best.pt` exists and its `student` weight magnitudes show smaller per-step deltas vs `student_last.pt`.
- `--resume runs/distill_v1/student_last.pt` works (backward-compat preserved).

**Risk**: none (decay 0.999 is standard; no extra VRAM).

---

## Phase 3 — APISR two-stage degradation in `AnimePairDataset` (rec.4)

**Files**: `anime_upscaler/dataset.py` (~15 LOC), new helper `anime_upscaler/degradation.py` (~80 LOC).

**New `anime_upscaler/degradation.py`**: codec + ffmpeg wrappers. Full source in plan annex below.
- Stage-1: stochastic JPEG/WEBP codec re-encode (q60/75/90); h264/h265 if ffmpeg found.
- Stage-2: random resize 0.5× → 4× round-trip with codec re-encode (probability 0.5).
- Graceful fallback to jpeg/webp-only if ffmpeg absent.

**`AnimePairDataset` change**:
- `__init__` accepts `degradation_mode: str = "none"` (default = v1 behavior for backward compat).
- `__getitem__` applies `degradation.degrade(hr_crop)` to LR generation in `split="train"` only; val/test stay clean for apples-to-apples comparison with v1 baseline.

**`distill.py` CLI**:
- Add `--degradation` flag forwarded to `AnimePairDataset(degradation_mode=...)`. Default = `"apsisr_v1"`.

**Validation gate**:
- `--smoke` produces LR tensors with visibly different high-frequency stats than the clean path (sample image check).
- Val/test eval stays on clean LR (same as v1 baseline → apples-to-apples comparison).
- ffmpeg absent → degradation gracefully degrades to jpeg/webp-only without crashing.

**Risk**: ffmpeg path may fail in odd envs → swallowed by `try/finally`. No GPU work added.

---

## Phase 4 — Inference-only TTA 8× at the GUI + CLI (rec.5)

**Files**: `anime_upscaler/tta.py` (~45 LOC, new), `anime_upscaler/infer.py` (~10 LOC), `apps/anime_upscaler_gui/anime_upscaler_gui/app.py` (~30 LOC).

**New helper `anime_upscaler/tta.py`**:
```python
_D4 = []
for h in (False, True):
    for v in (False, True):
        for k in (0, 1):  # 0 = id, 1 = rot90
            def _aug(x, h=h, v=v, k=k):
                if h: x = torch.flip(x, dims=[-1])
                if v: x = torch.flip(x, dims=[-2])
                if k: x = torch.rot90(x, k=1, dims=[-2, -1])
                return x
            def _inv(x, h=h, v=v, k=k):
                if k: x = torch.rot90(x, k=-1, dims=[-2, -1])
                if v: x = torch.flip(x, dims=[-2])
                if h: x = torch.flip(x, dims=[-1])
                return x
            _D4.append((_aug, _inv))

def tta_forward(model, lr_01):
    out = None
    with torch.no_grad():
        for aug, inv in _D4:
            y = model(aug(lr_01))
            y = inv(y)
            out = y if out is None else out + y
    return (out / len(_D4)).clamp(0.0, 1.0)
```

**CLI in `infer.py`**: add `--tta` flag that calls `tta_forward` instead of raw `model(lr)`.

**GUI integration**: in `_run_image` and `_run_video` (per-frame), when a Settings panel checkbox **"TTA 8× (slower, sharper)"** is on (default OFF to preserve current speed UX), wrap the inference call in `tta.tta_forward(model, lr_01)`. Default OFF.

**Validation gate**:
- Smoke: same image, with and without `--tta`, produces output within 0.05 dB of each other.
- Output range [0,1] for any input.
- Per-frame TTA on 1080p video ≤10s on RTX 4000 (baseline 200ms/frame → ~1.6s/frame with TTA).

**Risk**: GUI must keep current default (no TTA) so existing users see no regression.

---

## Phase 5 — Re-run distillation + evaluate

**Script**: `scripts/run_distill_v3.ps1` (PowerShell).
```powershell
# 1. Smoke
.\.venv\Scripts\python.exe -m anime_upscaler.distill --smoke `
    --degradation apsisr_v1 `
    --out-dir runs/distill_v3_smoke 2>&1 | Tee-Object runs/distill_v3_smoke.log

# 2. Full 40-epoch run
.\.venv\Scripts\python.exe -m anime_upscaler.distill --epochs 40 `
    --batch-size 16 --num-workers 8 --lr 5e-5 `
    --degradation apsisr_v1 `
    --out-dir runs/distill_v3 2>&1 | Tee-Object runs/distill_v3.log

# 3. Final test eval (writes results_table.md)
.\.venv\Scripts\python.exe -m anime_upscaler.validate `
    --student runs/distill_v3/student_best.pt `
    --teacher spn_v7 `
    --data data/anime_video_frames `
    --tta 2>&1 | Tee-Object runs/distill_v3_eval.log
```

**Success bar**:
- PASS  — student test PSNR ≥ 29.50 dB (passes bicubic).
- STRETCH — ≥ 29.80 dB (50% retention of teacher gain).
- STRETCH+ — ≥ 30.20 dB (full target).

If first run is below 29.50, ablate one change at a time (disable APISR, then Charbonnier, etc.) to find the regression.

---

## Files changed (summary)

| File | LOC delta | Risk |
|---|---|---|
| `anime_upscaler/distill.py` | +80 / −10 | low (train-only; backward-compat via `--resume`) |
| `anime_upscaler/dataset.py` | +15 / −2 | low (new arg default `"none"` = v1 behavior) |
| `anime_upscaler/degradation.py` | +80 (new) | low (graceful fallback if ffmpeg absent) |
| `anime_upscaler/tta.py` | +45 (new) | low (CLI flag off by default) |
| `anime_upscaler/infer.py` | +10 / −2 | low (`--tta` off by default) |
| `apps/anime_upscaler_gui/anime_upscaler_gui/app.py` | +30 / −2 | low (checkbox off by default) |
| `scripts/run_distill_v3.ps1` | +20 (new) | none |

---

## Files NOT changed (deliberately)

- `anime_upscaler/student.py` — RFDN architecture stays.
- `anime_upscaler/teacher.py` — SPAN-V7 teacher stays.
- `apps/anime_upscaler_gui/anime_upscaler_gui/archs.py` — already wired for v1 student; v3 retrain will produce a compatible checkpoint (same arch).
- `apps/anime_upscaler_gui/anime_upscaler_gui/registry.py` — preset already exists; we will overwrite `RFDN_distill_v1_4x_student.pth` with the v3 ckpt, or add a new entry `rfdn_distill_v3_4x` (decision deferred to end of Phase 5).

---

## Validation gate (this plan is "done" only if all pass)

1. PASS — `py_compile` clean on every edited .py.
2. PASS — `distill.py --smoke` runs to completion with all five changes, no NaN.
3. PASS — `distill.py --smoke --degradation apsisr_v1` produces visible LR artifacts (sample image check).
4. PASS — `infer.py --tta` produces outputs within 0.05 dB of `--tta`-off run on the same input.
5. PASS — Full 40-epoch `distill_v3` run completes; `train_log.csv` shows all loss terms finite and `loss_resp < loss_gt` (proves rebalance worked).
6. PASS — `runs/distill_v3/results_table.md`: student PSNR ≥ 29.50 dB (passes bicubic); `student_best.pt` is the EMA state, not raw.
7. PASS — GUI model dropdown picks up the updated `RFDN_distill_v1_4x_student.pth` (or the new v3 entry) after re-scan; pipeline worker test still passes.
8. SKIPPED — Adversarial distillation, FDL, MambaIRv2 (deferred to next plan per `docs/distill_v3_research/REPORT.md` §8).

---

## Out-of-scope (explicitly)

- **Adversarial distillation**: known to hurt PSNR by 0.1–0.3 dB; skip until PSNR target is hit.
- **MambaIRv2 / NAFNet / DAT2 backbone swap**: Windows-blocked or 2–4 weeks of work; not a first move.
- **Patch-NCE / CUT**: needs unpaired LR data we don't have.
- **Model souping**: only one student checkpoint, no siblings to average.
- **VQD-SR codebook**: 2–3 weeks; survey reports ~0 gain on our val_hr.
- **Frequency-distillation (FDL)**: worth doing as a Phase 6 follow-up if the five here plateau below 30.0 dB — it is already in `src/losses/fdl_loss.py` and could be ported in another ~30 LOC.

---

## Annex A — Full `anime_upscaler/degradation.py` source

```python
"""APISR-style two-stage degradation for anime4x distillation training.
Stage-1: stochastic codec compression (jpeg/webp/h264/h265) at q0.7-0.85.
Stage-2: shuffled resize + degrade-before-crop (random scale 0.5x -> 4x round-trip).
Designed for HR images; produces LR with realistic encoder artifacts.
"""
import io, random, subprocess, tempfile
from pathlib import Path
from PIL import Image

CODECS = {
    "jpeg60": ("JPEG", 60), "jpeg75": ("JPEG", 75), "jpeg90": ("JPEG", 90),
    "webp60": ("WEBP", 60), "webp75": ("WEBP", 75),
}
H264_CRF = [28, 32, 36]
H265_CRF = [28, 32, 36]

def _find_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None

_FFMPEG = _find_ffmpeg()

def _encode_codec(img, codec, quality):
    if codec == "JPEG" and img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    if codec in ("JPEG", "WEBP"):
        img.save(buf, codec, quality=quality)
        buf.seek(0)
        return Image.open(buf).copy()
    if _FFMPEG and codec in ("h264", "h265"):
        crf = random.choice(H264_CRF if codec == "h264" else H265_CRF)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, "PNG")
            src = f.name
        dst = src + ".out.mp4"
        try:
            subprocess.run([_FFMPEG, "-y", "-loglevel", "error", "-i", src,
                "-c:v", codec, "-crf", str(crf), "-pix_fmt", "yuv420p",
                "-frames:v", "1", dst], check=True)
            return Image.open(dst).copy()
        finally:
            for p in (src, dst):
                try: Path(p).unlink()
                except FileNotFoundError: pass
    return img

def degrade(img, p_codec=1.0, p_resize=0.5):
    """APISR-style two-stage degradation. p_codec=1.0 always applies
    stage-1 codec round-trip (our data is webp-decoded; re-encode adds
    realistic artifacts). p_resize=0.5 applies stage-2 resize round-trip
    half the time. Both are cheap on a 192x192 PIL crop.
    """
    name, q = random.choice(list(CODECS.values()))
    img = _encode_codec(img, name, q)
    if random.random() < p_resize:
        W, H = img.size
        small = img.resize((W//2, H//2), Image.BICUBIC)
        name2, q2 = random.choice(list(CODECS.values()))
        small = _encode_codec(small, name2, q2)
        img = small.resize((W, H), Image.BICUBIC)
    return img
```

---

## Annex B — Phase checklist (machine-readable)

See `docs/distill_v3_research/TODO.md` for the runnable checklist.

## Annex C — Cross-session memory

See `.windsurf/memory/distill_v3_plan.md` for the resumption note.