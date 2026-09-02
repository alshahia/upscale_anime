# anime_upscaler/distill.py
"""Teacher-student knowledge distillation for the 4x anime upscaler.

Loss (v3 -- see .claude/plans/distill_v3_recipe.md Phase 1):
    L = 0.3 * Charbonnier(student_out, teacher_out)        # response distillation
      + 0.5 * (1 - cos_sim) per-tap feature distillation   # feature distillation
      + 1.0 * ( 0.5 * L1 + 0.2 * (1 - MS-SSIM)              # ground-truth anchor
                + 0.05 * LPIPS-VGG ) (student_out, hr)

Teacher is frozen (SPANTeacher by default; any spandrel-loadable community
checkpoint via --teacher-spandrel - see eval_teachers.py to pick one.
Only the RFDN student (+ its 1x1 feature
adapters, discarded at inference) receives gradients.

Usage:
    python anime_upscaler/distill.py --smoke
    python anime_upscaler/distill.py --epochs 40 --batch-size 16 \
        --data data/anime_video_frames --out-dir runs/distill_v3
"""
import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import AnimePairDataset, denorm01
from student import RFDN
from teacher import SPANTeacher, FEATURE_CHANNELS, TAP_ORDER, TEACHERS

# v3 loss terms (Phase 1 of .claude/plans/distill_v3_recipe.md):
#   Charbonnier response, MS-SSIM + LPIPS GT anchor, cosine-distance features.
# piq 0.8.0 dropped `charbonnier_loss` from its public API; inline equivalent.
from piq import multi_scale_ssim as _msssim
import lpips as _lpips_pkg

# Phase 3 handoff (A.1, A.2): adversarial + edge losses. Imported lazily inside
# build_adversarial() / build_edge() so the existing Phase 2 v3 recipe (no adv,
# no edge) doesn't pay the import + GPU init cost.
import json
import shutil


def _charbonnier(x, y, eps=1e-3):
    """Charbonnier distance = mean(sqrt((x - y)^2 + eps^2)).

    Matches historical `piq.charbonnier_loss` semantics: per-pixel mean.
    Robust to outliers vs L1; smoother gradient near zero vs MSE.
    """
    return torch.sqrt((x - y) ** 2 + eps ** 2).mean()


_lpips_net = None


def _get_lpips(device):
    """Lazy-init the LPIPS VGG network (~149 MB weights; downloads on first call).

    Cached globally so it survives across epochs. Eval mode, frozen grads.
    """
    global _lpips_net
    if _lpips_net is None:
        _lpips_net = _lpips_pkg.LPIPS(net="vgg").to(device).eval()
        for p in _lpips_net.parameters():
            p.requires_grad_(False)
    return _lpips_net

ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """Deterministic per-worker seeding (must be top-level for Windows spawn)."""
    ws = torch.initial_seed() % 2 ** 32
    np.random.seed(ws)
    random.seed(ws)


def psnr01(a, b):
    """Batch PSNR on [0,1] tensors, averaged over the batch."""
    mse = ((a.clamp(0, 1) - b.clamp(0, 1)) ** 2).mean(dim=(1, 2, 3))
    mse = mse.clamp(min=1e-10)
    return (-10 * torch.log10(mse)).mean().item()


def ssim01(a, b):
    """Mean SSIM over batch using skimage (data_range=1, channel axis last)."""
    from skimage.metrics import structural_similarity
    a = a.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    b = b.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    vals = [structural_similarity(a[i], b[i], channel_axis=2, data_range=1.0)
            for i in range(a.shape[0])]
    return float(np.mean(vals))


class StudentFeatureAdapters(nn.Module):
    """Learnable 1x1 projections student_ch -> teacher_ch per tap pair.

    Part of training only; dropped at inference/export time.
    """

    def __init__(self, student_channels, teacher_channels):
        """teacher_channels: int (same for all taps) or list per tap."""
        super().__init__()
        if isinstance(teacher_channels, int):
            n_taps = len(TAP_ORDER)
            chans = [teacher_channels] * n_taps
        else:
            chans = list(teacher_channels)
        self.adapters = nn.ModuleList(
            nn.Conv2d(student_channels, c, 1) for c in chans)

    def forward(self, feats):
        return [ad(f) for ad, f in zip(self.adapters, feats)]


class _EMA:
    """Minimal ModelEmaV2 in 20 LOC; no timm dep needed.

    Phase 2 of .claude/plans/distill_v3_recipe.md. Float tensors are blended
    with decay (matches timm/tf semantics); integer buffers (e.g.
    num_batches_tracked) are copied verbatim so non-fp state never decays.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        # detach().clone() so in-place updates on the live model never bleed
        # into the shadow.
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                # shadow = d*shadow + (1-d)*model
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                self.shadow[k].copy_(v.detach())

    def state_dict(self):
        return self.shadow


def _adv_lambda(epoch: int, peak: float) -> float:
    """Phase 3 piecewise ramp: 0 -> peak over 30 epochs.

    The discriminator is randomly initialized, so it MUST start at zero weight
    (ep 1-5) to let G stabilize. Then the peak is reached by ep 21.
    Schedule: ep 1-5 -> 0; ep 6-10 -> peak*0.1; ep 11-20 -> peak*0.5;
    ep 21+ -> peak. The "peak" arg is --lambda-adv (default 0.001).
    """
    if epoch <= 5:
        return 0.0
    if epoch <= 10:
        return peak * 0.1
    if epoch <= 20:
        return peak * 0.5
    return peak


def _shortcut_weight(epoch: int, mode: str) -> float:
    """Phase 3 bicubic residual shortcut anneal (handoff A.4).

    mode='1to0':      ep 1 -> 1.0, ep 5 -> 0.0 (linear); ep 6+ -> 0.0.
                      Designed for FROM-SCRATCH runs (handoff B.3 smoke OK).
    mode='1to0slow':  ep 1 -> 1.0, ep 15 -> 0.0 (linear); ep 16+ -> 0.0.
                      Designed for WARM-START runs: the v1 student was
                      trained with shortcut=1.0 throughout, so a 5-epoch
                      anneal causes mode collapse (Phase C.3 halt observed
                      in run on 2026-09-02: PSNR 29.57 -> 5.41 over 5 epochs).
                      A 15-epoch anneal gives the warm-started student
                      enough time to scale its delta-prediction up to a
                      full SR signal.
    mode='off':       always 1.0 (legacy behavior).
    """
    if mode == "off":
        return 1.0
    if mode == "1to0":
        return max(0.0, 1.0 - (epoch - 1) / 4.0)
    if mode == "1to0slow":
        return max(0.0, 1.0 - (epoch - 1) / 14.0)
    raise ValueError(f"unknown shortcut-anneal mode: {mode!r}")


@torch.no_grad()
def _lap_var_from_metrics(model, loader, device, max_batches=2):
    """Compute mean Laplacian variance of student SR outputs on val batches.

    Used for the per-epoch metrics.json so Phase D can pick the best epoch by
    lap_var without re-running inference. Mirrors scripts/compare_*.py logic
    but inline in torch (no cv2 dep here).
    """
    import cv2
    import numpy as np
    model.eval()
    s = getattr(model, "scale", getattr(model, "upscale", 4))
    vals = []
    nb = 0
    for lr_n, hr_n in loader:
        lr = denorm01(lr_n).to(device)
        sr = model(lr).clamp(0, 1)
        # [B, 3, H, W] -> single grayscale float per item
        gray = sr.mean(dim=1, keepdim=False)              # [B, H, W]
        for g in gray:
            arr = (g.cpu().numpy() * 255).astype("uint8")
            vals.append(float(cv2.Laplacian(arr, cv2.CV_32F, ksize=3).var()))
        nb += 1
        if nb >= max_batches:
            break
    return float(np.mean(vals)) if vals else 0.0


@torch.no_grad()
def evaluate(model, teacher, loader, device, max_batches=None):
    """Returns dict bicubic/student/teacher -> (psnr, ssim)."""
    model.eval()
    sums = {k: [0.0, 0.0] for k in ("bicubic", "student", "teacher")}
    nb = 0
    for lr_n, hr_n in loader:
        lr, hr = denorm01(lr_n).to(device), denorm01(hr_n).to(device)
        s = getattr(model, "scale", getattr(model, "upscale", 4))
        bic = torch.nn.functional.interpolate(lr, scale_factor=s, mode="bicubic",
                                              align_corners=False).clamp(0, 1)
        sr = model(lr).clamp(0, 1)
        # The teacher operates at its own scale (the SPAN pretrain is 4x);
        # when the student is 2x, skip the teacher PSNR/SSIM column -- the
        # teacher output doesn't match `hr` shape and would crash.
        skip_teacher = (s != 4)
        try:
            tt = teacher(lr).clamp(0, 1)
            teacher_ok = True
        except Exception:
            teacher_ok = False
        for name, pred in (("bicubic", bic), ("student", sr),
                           *((("teacher", tt),) if teacher_ok and not skip_teacher else ())):
            sums[name][0] += psnr01(pred, hr)
            sums[name][1] += ssim01(pred, hr)
        nb += 1
        if max_batches is not None and nb >= max_batches:
            break
    return {k: (v[0] / nb, v[1] / nb) for k, v in sums.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data" / "anime_video_frames"))
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "distill_v1"))
    ap.add_argument("--teacher-ckpt", default=None)
    ap.add_argument("--teacher-spandrel", default=None,
                    help="path to a spandrel-loadable teacher .pth/.safetensors (overrides SPAN)")
    ap.add_argument("--teacher-half", action="store_true",
                    help="run a spandrel teacher in fp16 (speed)")
    # Phase 3 handoff A.4: --teacher name dispatches via TEACHERS registry.
    # When teacher is animevideov3 or lsdir, the loss wiring switches to the
    # Phase 3 recipe (response-only distillation + adversarial + edge); when
    # teacher is span, the existing Phase 2 v3 recipe is preserved.
    ap.add_argument("--teacher", choices=["span", "animevideov3", "lsdir"],
                    default="animevideov3",
                    help="teacher name (Phase 3 default animevideov3; ")
    ap.add_argument("--lambda-adv", type=float, default=0.001,
                    help="peak adversarial weight (Phase 3 handoff A.4). ")
    ap.add_argument("--shortcut-anneal", choices=["off", "1to0", "1to0slow"], default="off",
                    help="bicubic residual shortcut anneal schedule (Phase 3 A.4). ")
    ap.add_argument("--no-ema", action="store_true",
                    help="disable EMA shadow (Phase 3 ablation only).")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-batches", type=int, default=8,
                    help="val minibatches per quick epoch-end eval")
    ap.add_argument("--resume", default=None,
                    help="path to student_last.pt/student_best.pt to continue training from")
    ap.add_argument("--fresh-epoch", action="store_true",
                    help="when --resume is set, ignore the ckpt's epoch counter "
                         "and start at epoch 1. Use for warm-start from v1: "
                         "load weights but reset the training schedule.")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny subset + 2 epochs: verify the whole loop runs")
    ap.add_argument("--degradation", default="apsisr_v1",
                    choices=["none", "apsisr_v1"],
                    help="Phase 3 LR degradation mode for split='train' only "
                         "(val/test always clean). Default 'apsisr_v1'; pass "
                         "'none' to disable (Phase 1/2 baseline). ")
    ap.add_argument("--no-lpips", action="store_true",
                    help="Set LPIPS weight to 0 in loss_gt (Phase 5 ablation only).")
    ap.add_argument("--no-msssim", action="store_true",
                    help="Set MS-SSIM weight to 0 in loss_gt (Phase 5 ablation only).")
    ap.add_argument("--scale", type=int, default=4, choices=(2, 3, 4),
                    help="Upscaling factor for the student (2 for the v2 cascade student, "
                         "4 for the v1 single-shot student). Default 4.")
    args = ap.parse_args()

    if args.smoke:
        args.epochs, args.batch_size, args.num_workers = 2, 4, 2
        args.out_dir = str(ROOT / "runs" / "distill_smoke")
        # Phase 3 handoff: smoke invocations via the legacy `python
        # anime_upscaler/distill.py --smoke` path stay on the Phase 2 v3
        # recipe (SPAN teacher, no adv, no anneal). scripts/train_v3_smoke.py
        # is the Phase 3 smoke entrypoint and passes --teacher explicitly.
        args.teacher = "span"
        args.lambda_adv = 0.0
        args.shortcut_anneal = "off"

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    kw = dict(max_files=12 if args.smoke else None,
              degradation_mode=args.degradation,
              scale=args.scale)
    train_ds = AnimePairDataset(args.data, "train", **kw)
    val_ds = AnimePairDataset(args.data, "val", **kw)
    test_ds = AnimePairDataset(args.data, "test", **kw)
    g = torch.Generator()
    g.manual_seed(args.seed)

    dl_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                     pin_memory=True, worker_init_fn=seed_worker,
                     persistent_workers=args.num_workers > 0)
    train_dl = DataLoader(train_ds, shuffle=True, generator=g,
                          drop_last=len(train_ds) >= args.batch_size, **dl_kwargs)
    val_dl = DataLoader(val_ds, shuffle=False, **dl_kwargs)
    test_dl = DataLoader(test_ds, shuffle=False, **dl_kwargs)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"batch={args.batch_size} epochs={args.epochs}")

    # ---- models ----
    # Phase 3 handoff A.4: --teacher dispatches via TEACHERS registry.
    # When the user supplies --teacher-spandrel or --teacher-ckpt, those
    # flags still take precedence (backward-compatible with the Phase 2 v3
    # recipe smoke runs that load a spandrel teacher for ablation).
    is_real_esr_teacher = False
    if args.teacher_spandrel:
        from spandrel_teacher import SpandrelTeacher
        teacher = SpandrelTeacher(args.teacher_spandrel, device=device,
                                  half=args.teacher_half)
        tap_chans = teacher.tap_channels
        print(f"[teacher] spandrel mode: {Path(args.teacher_spandrel).name}")
    elif args.teacher_ckpt:
        teacher = SPANTeacher(args.teacher_ckpt, device=device)
        tap_chans = FEATURE_CHANNELS
    elif args.teacher == "span":
        teacher = SPANTeacher(args.teacher_ckpt, device=device)
        tap_chans = FEATURE_CHANNELS
    else:
        # animevideov3 / lsdir (Phase 3 path).
        cls, kwargs = TEACHERS[args.teacher]
        teacher = cls(**kwargs, device=device).to(device) if "device" not in kwargs \
                  else cls(**kwargs)
        tap_chans = None  # SRVGG teachers do not expose feature taps
        is_real_esr_teacher = True
    student = RFDN(scale=args.scale).to(device)
    if tap_chans is not None:
        adapters = StudentFeatureAdapters(52, tap_chans).to(device)
    else:
        adapters = None  # RealESR teachers have no taps -> no adapters needed
    print(f"[student] {student.num_params():,} params (<600K)")
    assert student.num_params() < 600_000

    # Phase 3 handoff A.4: detect whether to use the new recipe. Triggered by
    # any of: SRVGG teacher (animevideov3, lsdir), lambda_adv > 0, shortcut
    # anneal != off. When false, the existing Phase 2 v3 wiring is unchanged.
    use_phase3 = is_real_esr_teacher or args.lambda_adv > 0 or args.shortcut_anneal != "off"
    print(f"[recipe] phase3={use_phase3}  teacher={args.teacher}  "
          f"lambda_adv={args.lambda_adv}  shortcut_anneal={args.shortcut_anneal}")

    # Build adversarial loss + discriminator (only when lambda_adv > 0).
    D = None
    gan_loss = None
    edge_loss = None
    if args.lambda_adv > 0:
        from adv_losses.adversarial import PatchGAN70, HingeGANLoss
        from adv_losses.edge_loss import EdgeLoss
        D = PatchGAN70(in_channels=6, num_features=64).to(device)
        gan_loss = HingeGANLoss()
        edge_loss = EdgeLoss().to(device)
        # TTUR (Heusel et al., 2017): G lr=2e-4, D lr=4e-4. G optimizer gets the
        # student + adapters; D optimizer gets only D.
        print(f"[adv] D={sum(p.numel() for p in D.parameters()):,} params  "
              f"edge_loss=ok")
    elif use_phase3:
        # SRVGG teacher without adversarial still gets the edge loss (Phase 3
        # spec: L_grad = 0.1 * EdgeLoss even when D is off).
        from adv_losses.edge_loss import EdgeLoss
        edge_loss = EdgeLoss().to(device)
        print(f"[adv] D=off  edge_loss=ok  (lambda_adv=0)")

    # ---- optimizers (Phase 3 handoff A.4: TTUR for G/D) ----
    g_params = list(student.parameters())
    if adapters is not None:
        g_params += list(adapters.parameters())
    optimizer = torch.optim.Adam(g_params, lr=args.lr)
    d_optimizer = None
    d_scheduler = None
    if D is not None:
        d_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr * 2.0,
                                       betas=(0.9, 0.99))
    # Optional resume: load weights and continue from next epoch with
    # a fresh Adam optimizer + cosine schedule over the REMAINING epochs.
    resume_from = 1
    best_psnr = -1.0  # sentinel; updated below if --resume, else first epoch's val_psnr wins.
    if args.resume:
        rs = torch.load(args.resume, map_location=device,
                        weights_only=False)
        student.load_state_dict(rs["student"])
        # Phase 3 handoff A.4: adapters is None for SRVGG teachers. Older
        # checkpoints saved with adapters -> skip the load when None.
        if adapters is not None and rs.get("adapters") is not None:
            adapters.load_state_dict(rs["adapters"])
        # Phase 3: also restore D state if present and D was rebuilt.
        if D is not None and rs.get("D") is not None:
            D.load_state_dict(rs["D"])
        if args.fresh_epoch:
            # Warm-start semantics: load weights but reset the training schedule.
            resume_from = 1
            best_psnr = -1.0
            print("[resume] warm-start from", args.resume,
                  "-> student weights loaded, epoch counter reset to 1")
        else:
            resume_from = rs["epoch"] + 1
            best_psnr = rs["val_psnr"]
            print("[resume] from epoch", rs["epoch"], "at", args.resume,
                  "val_psnr %.2f dB" % best_psnr)
    # Phase 2: construct EMA shadow AFTER --resume so it seeds from the loaded
    # weights (not from RFDN()'s random init). Phase 3 handoff A.4: --no-ema
    # disables the shadow entirely (ablation path).
    student_ema = None if args.no_ema else _EMA(student, decay=0.999)
    # Track the better of (raw student val PSNR, EMA val PSNR) so a slow-converging
    # EMA decay does not lock the final model behind the raw student. Saves the
    # winning variant to student_best.pt (canonical filename preserves downstream
    # consumers) plus always emits the raw-best state as student_best_raw.pt and
    # the EMA-best state as student_best_ema.pt for auditability.
    best_raw_psnr = best_psnr
    best_ema_psnr = -1.0
    remaining = max(args.epochs - resume_from + 1, 0)
    if remaining == 0:
        print("[resume] nothing left to train; exiting")
        return
    if args.resume:
        print("[resume] will run epochs", resume_from, "to", args.epochs,
              "(%d remaining)" % remaining)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=remaining, eta_min=1e-6)
    if d_optimizer is not None:
        d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            d_optimizer, T_max=remaining, eta_min=1e-6)
    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    log_path = out_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        # Phase 3 columns: loss_adv, loss_grad for adversarial + edge terms.
        csv.writer(f).writerow(
            ["epoch", "lr", "loss_response", "loss_feature", "loss_gt",
             "loss_adv", "loss_grad", "loss_total",
             "val_psnr_bicubic", "val_psnr_student",
             "val_psnr_student_ema", "val_psnr_teacher", "val_ssim_student",
             "seconds", "teacher_ms_per_iter"])

    for epoch in range(resume_from, args.epochs + 1):
        t0 = time.time()
        student.train()
        if D is not None:
            D.train()
        agg = {"resp": 0.0, "feat": 0.0, "gt": 0.0, "adv": 0.0, "grad": 0.0,
               "total": 0.0}
        n_iter = 0
        teacher_ms_total = 0.0
        for lr_n, hr_n in train_dl:
            lr = denorm01(lr_n).to(device, non_blocking=True)
            hr = denorm01(hr_n).to(device, non_blocking=True)

            # ---- Teacher forward ----
            tT = time.perf_counter()
            if is_real_esr_teacher:
                # SRVGG teachers: no feature taps, only SR output.
                with torch.no_grad():
                    t_out = teacher(lr)
                t_feats = []
            else:
                with torch.no_grad():
                    t_out, t_feats = teacher.forward_with_features(lr)
            if device == "cuda":
                torch.cuda.synchronize()
            teacher_ms_total += (time.perf_counter() - tT) * 1000.0

            # ---- Student forward ----
            if use_phase3 and is_real_esr_teacher:
                # Phase 3 + SRVGG: skip return_features (no taps to distill).
                s_out = student(lr)
                s_feats = []
            else:
                s_out, s_feats = student(lr, return_features=True)
                if adapters is not None:
                    s_feats = adapters(s_feats)

            # ---- Response distillation (always; works for any teacher) ----
            if s_out.shape[-2:] != t_out.shape[-2:]:
                s_out_for_resp = F.interpolate(
                    s_out, size=t_out.shape[-2:], mode="bicubic",
                    align_corners=False
                )
            else:
                s_out_for_resp = s_out
            if use_phase3:
                # Phase 3 spec: L_distill = 0.5 * L1(student_SR, teacher_SR)
                loss_distill = 0.5 * l1(s_out_for_resp, t_out)
            else:
                loss_distill = _charbonnier(s_out_for_resp, t_out, eps=1e-3)

            # ---- Feature distillation (Phase 2 v3 only; SRVGG teachers have no taps) ----
            loss_feat = torch.tensor(0.0, device=device)
            if (not use_phase3) or not is_real_esr_teacher:
                if s_feats and t_feats:
                    feat_terms = []
                    for sf, tf in zip(s_feats, t_feats):
                        if tf.shape[-2:] != sf.shape[-2:]:
                            tf = F.interpolate(tf, size=sf.shape[-2:],
                                               mode="bilinear",
                                               align_corners=False)
                        sf_n = F.normalize(sf, dim=1)
                        tf_n = F.normalize(tf, dim=1)
                        feat_terms.append(
                            (1.0 - (sf_n * tf_n).sum(dim=1, keepdim=True)).mean()
                        )
                    loss_feat = sum(feat_terms) / max(len(feat_terms), 1)

            # ---- GT anchor (Phase 3 has new weights) ----
            if s_out.shape[-2:] != hr.shape[-2:]:
                s_out_for_gt = F.interpolate(
                    s_out.clamp(0, 1), size=hr.shape[-2:], mode="bicubic",
                    align_corners=False
                )
            else:
                s_out_for_gt = s_out.clamp(0, 1)
            if use_phase3:
                # Phase 3: L_pix = 0.5 * L1, L_perc = 1.0 * LPIPS, no MS-SSIM.
                loss_gt = 0.5 * l1(s_out_for_gt, hr)
                if not args.no_lpips:
                    loss_gt = loss_gt + _get_lpips(device)(
                        s_out_for_gt * 2.0 - 1.0, hr * 2.0 - 1.0
                    ).mean()
            else:
                # Phase 2 v3 recipe: 0.5*L1 + 0.2*MS-SSIM + 0.05*LPIPS
                lp_weight = 0.0 if args.no_lpips else 0.05
                ss_weight = 0.0 if args.no_msssim else 0.2
                if lp_weight > 0 or ss_weight > 0:
                    if ss_weight > 0:
                        ssim_term = 1.0 - _msssim(
                            s_out_for_gt, hr, data_range=1.0, reduction="mean")
                    if lp_weight > 0:
                        lp_term = _get_lpips(device)(
                            s_out_for_gt * 2.0 - 1.0, hr * 2.0 - 1.0
                        ).mean()
                    loss_gt = 0.5 * l1(s_out_for_gt, hr)
                    if ss_weight > 0:
                        loss_gt = loss_gt + ss_weight * ssim_term
                    if lp_weight > 0:
                        loss_gt = loss_gt + lp_weight * lp_term
                else:
                    loss_gt = 0.5 * l1(s_out_for_gt, hr)

            # ---- Edge loss (Phase 3 only) ----
            loss_edge = torch.tensor(0.0, device=device)
            if use_phase3 and edge_loss is not None:
                loss_edge = 0.1 * edge_loss(s_out_for_gt, hr)

            # ---- Total G loss (Phase 3 spec or Phase 2 v3 spec) ----
            if use_phase3:
                # L = L_distill + L_gt + L_grad + (later) L_adv
                loss = loss_distill + loss_gt + loss_edge
            else:
                loss = 0.3 * loss_distill + 0.5 * loss_feat + 1.0 * loss_gt

            # ---- D step (Phase 3 only; standard ESRGAN pattern) ----
            loss_adv_g = torch.tensor(0.0, device=device)
            adv_w = 0.0
            if D is not None:
                adv_w = _adv_lambda(epoch, args.lambda_adv)
                if adv_w > 0:
                    # D forward on real (HR) and detached fake (SR).
                    lr_up = F.interpolate(
                        lr, size=hr.shape[-2:], mode="bicubic",
                        align_corners=False
                    )
                    d_real_in = torch.cat([lr_up, hr], dim=1)
                    d_fake_in = torch.cat([lr_up, s_out.detach()], dim=1)
                    d_real_out = D(d_real_in)
                    d_fake_out = D(d_fake_in)
                    d_loss, _ = gan_loss(d_real_out, d_fake_out)
                    d_optimizer.zero_grad(set_to_none=True)
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
                    d_optimizer.step()
                    # G adversarial term: another D forward with LIVE SR so G grads flow.
                    d_fake_in_g = torch.cat([lr_up, s_out], dim=1)
                    d_fake_out_g = D(d_fake_in_g)
                    _, loss_adv_g = gan_loss(d_real_out.detach(), d_fake_out_g)
                    loss = loss + adv_w * loss_adv_g

            # ---- G step ----
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            if adapters is not None:
                torch.nn.utils.clip_grad_norm_(adapters.parameters(), max_norm=1.0)
            optimizer.step()
            if student_ema is not None:
                student_ema.update(student)

            agg["resp"] += loss_distill.item()
            agg["feat"] += loss_feat.item()
            agg["gt"] += loss_gt.item()
            agg["adv"] += loss_adv_g.item()
            agg["grad"] += loss_edge.item()
            agg["total"] += loss.item()
            n_iter += 1

        scheduler.step()
        # Phase 3 handoff A.4: bicubic residual shortcut anneal.
        # Set BEFORE evaluation so val metrics reflect the active shortcut.
        sw = _shortcut_weight(epoch, args.shortcut_anneal)
        student.set_shortcut_weight(sw)
        # D scheduler step (TTUR; same epoch count as G).
        if d_scheduler is not None:
            d_scheduler.step()
        metrics = evaluate(student, teacher, val_dl, device,
                           max_batches=args.val_batches)
        # Phase 2: evaluate EMA shadow once per epoch (cheap; reuses
        # evaluate()). student_best.pt records the EMA state, not raw.
        # Phase 3 handoff A.4: --no-ema disables the shadow entirely.
        if student_ema is not None:
            backup = {k: v.detach().clone() for k, v in student.state_dict().items()}
            student.load_state_dict(student_ema.state_dict())
            metrics_ema = evaluate(student, teacher, val_dl, device,
                                   max_batches=args.val_batches)
            student.load_state_dict(backup)
        else:
            metrics_ema = metrics
        dt = time.time() - t0
        row = [epoch, scheduler.get_last_lr()[0],
               agg["resp"] / max(n_iter, 1), agg["feat"] / max(n_iter, 1),
               agg["gt"] / max(n_iter, 1),
               agg["adv"] / max(n_iter, 1), agg["grad"] / max(n_iter, 1),
               agg["total"] / max(n_iter, 1),
               metrics["bicubic"][0], metrics["student"][0],
               metrics_ema["student"][0], metrics["teacher"][0],
               metrics["student"][1], round(dt, 1),
               round(teacher_ms_total / max(n_iter, 1), 1)]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"[ep {epoch:03d}/{args.epochs}] loss={row[7]:.4f} "
              f"(resp {row[2]:.4f} feat {row[3]:.4f} gt {row[4]:.4f} "
              f"adv {row[5]:.4f} grad {row[6]:.4f}) "
              f"val PSNR b/s/t = {row[8]:.2f}/{row[9]:.2f}/{row[11]:.2f} dB "
              f"(ema s {row[10]:.2f}) "
              f"ssim_s={metrics['student'][1]:.4f} ({dt:.0f}s)")

        state_raw = {
            "epoch": epoch,
            "student": student.state_dict(),
            "adapters": adapters.state_dict() if adapters is not None else None,
            "D": D.state_dict() if D is not None else None,
            "val_psnr": metrics["student"][0],
            "args": vars(args),
        }
        # Phase 2 + Phase 5 fix: track best raw-state and best EMA-state
        # independently; write whichever was the global best this epoch into
        # student_best.pt. EMA decay=0.999 converges slowly on a 40-epoch run:
        # at ~16k iters the shadow only covers ~80% of the live model, so
        # evaluating student_best.pt strictly against EMA would lock in a
        # *worse* model than student_last.pt. Track both, save the winner.
        torch.save(state_raw, out_dir / "student_last.pt")
        if metrics["student"][0] > best_raw_psnr:
            best_raw_psnr = metrics["student"][0]
            torch.save({**state_raw, "val_psnr": best_raw_psnr},
                       out_dir / "student_best_raw.pt")
        if student_ema is not None and metrics_ema["student"][0] > best_ema_psnr:
            best_ema_psnr = metrics_ema["student"][0]
            state_ema = {**state_raw, "student": student_ema.state_dict(),
                         "val_psnr": best_ema_psnr}
            torch.save(state_ema, out_dir / "student_best_ema.pt")
        winner_is_raw = (metrics["student"][0] >= metrics_ema["student"][0]
                         or student_ema is None)
        cand_psnr = (metrics["student"][0] if winner_is_raw
                     else metrics_ema["student"][0])
        if cand_psnr > best_psnr:
            best_psnr = cand_psnr
            if winner_is_raw:
                torch.save({**state_raw, "val_psnr": best_psnr},
                           out_dir / "student_best.pt")
                print(f"  -> new best (raw {best_psnr:.2f} dB) saved")
            else:
                state_ema_save = {**state_raw,
                                  "student": student_ema.state_dict(),
                                  "val_psnr": best_psnr}
                torch.save(state_ema_save, out_dir / "student_best.pt")
                print(f"  -> new best (EMA {best_psnr:.2f} dB) saved")

        # ---- Phase 3 handoff A.4: per-epoch checkpoint rotation (REQUIRED) ----
        # Save every epoch's raw + ema + metrics JSON. Keep last 5 on disk;
        # move older ones into archive/. NEVER delete; user requirement Q6.
        ep_raw_path = out_dir / f"epoch_{epoch}.pt"
        ep_ema_path = out_dir / f"epoch_{epoch}_ema.pt"
        torch.save(state_raw, ep_raw_path)
        if student_ema is not None:
            torch.save({**state_raw,
                        "student": student_ema.state_dict()},
                       ep_ema_path)
            shutil.copy(ep_ema_path, out_dir / "latest_ema.pt")
        shutil.copy(ep_raw_path, out_dir / "latest.pt")
        with open(out_dir / f"epoch_{epoch}_metrics.json", "w") as f:
            json.dump({
                "epoch": epoch,
                "val_psnr": metrics["student"][0],
                "val_psnr_ema": metrics_ema["student"][0],
                "val_ssim": metrics["student"][1],
                "val_lap_var": _lap_var_from_metrics(student, val_dl, device),
                "shortcut_weight": sw,
                "lambda_adv": _adv_lambda(epoch, args.lambda_adv),
                "seconds": round(dt, 1),
            }, f, indent=2)
        # Rotate: keep last 5 epoch_N.pt on disk; archive older. Sort by
        # NUMERIC epoch (not alphabetical) so ep 9 wins over ep 30 with the
        # broken string sort (which kept only ep 8/9 in 2026-09-02 run).
        def _epoch_num(p):
            # Extract integer epoch from filename epoch_N[_ema].pt
            try:
                return int(p.stem.split("_")[1])
            except (IndexError, ValueError):
                return -1
        epoch_ckpts = sorted(out_dir.glob("epoch_*.pt"),
                             key=_epoch_num)
        # Unique epoch numbers in chronological order
        epoch_nums = sorted({_epoch_num(p) for p in epoch_ckpts if _epoch_num(p) >= 0})
        if len(epoch_nums) > 5:
            archive = out_dir / "archive"
            archive.mkdir(exist_ok=True)
            for old_ep in epoch_nums[:-5]:
                for suffix in (".pt", "_ema.pt", "_metrics.json"):
                    f = out_dir / f"epoch_{old_ep}{suffix}"
                    if f.exists():
                        shutil.move(str(f), str(archive / f.name))

    # ---- final held-out evaluation ----
    best = torch.load(out_dir / "student_best.pt", map_location=device,
                      weights_only=False)
    student.load_state_dict(best["student"])
    final = evaluate(student, teacher, test_dl, device)
    lines = ["| model | PSNR (dB) | SSIM |", "|---|---|---|"]
    for k in ("bicubic", "student", "teacher"):
        lines.append(f"| {k} | {final[k][0]:.2f} | {final[k][1]:.4f} |")
    gain_b = final["student"][0] - final["bicubic"][0]
    gain_t = final["teacher"][0] - final["bicubic"][0]
    retention = 100.0 * gain_b / gain_t if gain_t > 0 else float("nan")
    lines.append(f"| student retention of teacher gain | {retention:.1f}% | |")
    table = "\n".join(lines)
    print("\n=== FINAL TEST SET ===\n" + table)
    (out_dir / "results_table.md").write_text(table + "\n")
    with open(out_dir / "results_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "psnr_db", "ssim"])
        for k in ("bicubic", "student", "teacher"):
            w.writerow([k, "%.2f" % final[k][0], "%.4f" % final[k][1]])
        w.writerow(["retention_pct", "%.1f" % retention, ""])
    print(f"[done] logs: {log_path}")


if __name__ == "__main__":
    main()
