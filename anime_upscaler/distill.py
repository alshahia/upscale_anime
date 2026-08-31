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
from teacher import SPANTeacher, FEATURE_CHANNELS, TAP_ORDER

# v3 loss terms (Phase 1 of .claude/plans/distill_v3_recipe.md):
#   Charbonnier response, MS-SSIM + LPIPS GT anchor, cosine-distance features.
# piq 0.8.0 dropped `charbonnier_loss` from its public API; inline equivalent.
from piq import multi_scale_ssim as _msssim
import lpips as _lpips_pkg


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
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-batches", type=int, default=8,
                    help="val minibatches per quick epoch-end eval")
    ap.add_argument("--resume", default=None,
                    help="path to student_last.pt/student_best.pt to continue training from")
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
    if args.teacher_spandrel:
        from spandrel_teacher import SpandrelTeacher
        teacher = SpandrelTeacher(args.teacher_spandrel, device=device,
                                  half=args.teacher_half)
        tap_chans = teacher.tap_channels
        print(f"[teacher] spandrel mode: {Path(args.teacher_spandrel).name}")
    else:
        teacher = SPANTeacher(args.teacher_ckpt, device=device)
        tap_chans = FEATURE_CHANNELS
    student = RFDN(scale=args.scale).to(device)
    adapters = StudentFeatureAdapters(52, tap_chans).to(device)
    print(f"[student] {student.num_params():,} params (<600K)")
    assert student.num_params() < 600_000

    optimizer = torch.optim.Adam(
        list(student.parameters()) + list(adapters.parameters()), lr=args.lr)
    # Optional resume: load weights and continue from next epoch with
    # a fresh Adam optimizer + cosine schedule over the REMAINING epochs.
    resume_from = 1
    best_psnr = -1.0  # sentinel; updated below if --resume, else first epoch's val_psnr wins.
    if args.resume:
        rs = torch.load(args.resume, map_location=device,
                        weights_only=False)
        student.load_state_dict(rs["student"])
        adapters.load_state_dict(rs["adapters"])
        resume_from = rs["epoch"] + 1
        best_psnr = rs["val_psnr"]
        print("[resume] from epoch", rs["epoch"], "at", args.resume,
              "val_psnr %.2f dB" % best_psnr)
    # Phase 2: construct EMA shadow AFTER --resume so it seeds from the loaded
    # weights (not from RFDN()'s random init).
    student_ema = _EMA(student, decay=0.999)
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
    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    log_path = out_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "lr", "loss_response", "loss_feature", "loss_gt",
             "loss_total", "val_psnr_bicubic", "val_psnr_student",
             "val_psnr_student_ema", "val_psnr_teacher", "val_ssim_student",
             "seconds", "teacher_ms_per_iter"])

    for epoch in range(resume_from, args.epochs + 1):
        t0 = time.time()
        student.train()
        agg = {"resp": 0.0, "feat": 0.0, "gt": 0.0, "total": 0.0}
        n_iter = 0
        teacher_ms_total = 0.0
        for lr_n, hr_n in train_dl:
            lr = denorm01(lr_n).to(device, non_blocking=True)
            hr = denorm01(hr_n).to(device, non_blocking=True)

            tT = time.perf_counter()
            with torch.no_grad():
                t_out, t_feats = teacher.forward_with_features(lr)
            if device == "cuda":
                torch.cuda.synchronize()
            teacher_ms_total += (time.perf_counter() - tT) * 1000.0

            s_out, s_feats = student(lr, return_features=True)
            s_feats = adapters(s_feats)

            # Response distillation: works at scale=4 directly. At scale=2 the
            # student output (LR*2) and teacher output (LR*4) differ -- upsample
            # the student to LR*4 and Charbonnier-compare with the teacher. This
            # gives the 2x cascade student a teacher oracle at each iteration
            # (closes the 1.4 dB cascade regression measured in 2026-08-31 e2e
            # eval).
            if s_out.shape[-2:] != t_out.shape[-2:]:
                s_out_for_resp = F.interpolate(
                    s_out, size=t_out.shape[-2:], mode="bicubic",
                    align_corners=False
                )
            else:
                s_out_for_resp = s_out
            loss_resp = _charbonnier(s_out_for_resp, t_out, eps=1e-3)
            feat_terms = []
            for sf, tf in zip(s_feats, t_feats):
                if tf.shape[-2:] != sf.shape[-2:]:
                    # generic teachers may expose SR-res taps; align to student
                    tf = F.interpolate(tf, size=sf.shape[-2:], mode="bilinear",
                                       align_corners=False)
                # v3: cosine-distance feature distillation (Lee FDL / Jung CSD
                # style). Normalize both feature maps on the channel axis and
                # take 1 - cos sim. Robust to scale/magnitude collapse.
                sf_n = F.normalize(sf, dim=1)
                tf_n = F.normalize(tf, dim=1)
                feat_terms.append(
                    (1.0 - (sf_n * tf_n).sum(dim=1, keepdim=True)).mean()
                )
            loss_feat = sum(feat_terms) / max(len(feat_terms), 1)
            # v3 GT anchor: blend L1 + MS-SSIM + LPIPS-VGG perceptual.
            # MS-SSIM needs >=161x161; our SR is 192x192 (train) / 384x384 (val).
            # LPIPS expects [B,3,H,W] in [-1,1]; we map from [0,1] model space.
            # GT anchor: align s_out to hr scale. For the v1 4x student s_out
            # already matches hr (both are at LR*4). For the v2 2x student s_out
            # is at LR*2 and hr is at LR*4 -- bicubic-upsample s_out to hr scale
            # so MS-SSIM/LPIPS/L1 all compare apples to apples.
            if s_out.shape[-2:] != hr.shape[-2:]:
                s_out_for_gt = F.interpolate(
                    s_out.clamp(0, 1), size=hr.shape[-2:], mode="bicubic",
                    align_corners=False
                )
            else:
                s_out_for_gt = s_out.clamp(0, 1)
            lp_weight = 0.0 if args.no_lpips else 0.05
            ss_weight = 0.0 if args.no_msssim else 0.2
            if lp_weight > 0 or ss_weight > 0:
                if ss_weight > 0:
                    ssim_term = 1.0 - _msssim(s_out_for_gt, hr, data_range=1.0,
                                               reduction="mean")
                if lp_weight > 0:
                    lp_term = _get_lpips(device)(
                        s_out_for_gt * 2.0 - 1.0, hr * 2.0 - 1.0
                    ).mean()
                # Reassemble loss_gt from whichever terms are non-zero
                loss_gt = 0.5 * l1(s_out_for_gt, hr)
                if ss_weight > 0:
                    loss_gt = loss_gt + ss_weight * ssim_term
                if lp_weight > 0:
                    loss_gt = loss_gt + lp_weight * lp_term
            else:
                # Phase 5 ablation path: drop both perceptual terms; pure L1 GT anchor.
                loss_gt = 0.5 * l1(s_out_for_gt, hr)
            # v3 rebalanced weights: GT now dominates (strongest PSNR signal);
            # response down-weighted (Charbonnier is more aggressive than L1);
            # feature held (cosine is in [0,2], comparable to old MSE).
            loss = 0.3 * loss_resp + 0.5 * loss_feat + 1.0 * loss_gt

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            student_ema.update(student)  # Phase 2: EMA of student weights

            agg["resp"] += loss_resp.item()
            agg["feat"] += loss_feat.item()
            agg["gt"] += loss_gt.item()
            agg["total"] += loss.item()
            n_iter += 1

        scheduler.step()
        metrics = evaluate(student, teacher, val_dl, device,
                           max_batches=args.val_batches)
        # Phase 2: evaluate EMA shadow once per epoch (cheap; reuses
        # evaluate()). student_best.pt records the EMA state, not raw.
        backup = {k: v.detach().clone() for k, v in student.state_dict().items()}
        student.load_state_dict(student_ema.state_dict())
        metrics_ema = evaluate(student, teacher, val_dl, device,
                               max_batches=args.val_batches)
        student.load_state_dict(backup)
        dt = time.time() - t0
        row = [epoch, scheduler.get_last_lr()[0],
               agg["resp"] / max(n_iter, 1), agg["feat"] / max(n_iter, 1),
               agg["gt"] / max(n_iter, 1), agg["total"] / max(n_iter, 1),
               metrics["bicubic"][0], metrics["student"][0],
               metrics_ema["student"][0], metrics["teacher"][0],
               metrics["student"][1], round(dt, 1),
               round(teacher_ms_total / max(n_iter, 1), 1)]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"[ep {epoch:03d}/{args.epochs}] loss={row[5]:.4f} "
              f"(resp {row[2]:.4f} feat {row[3]:.4f} gt {row[4]:.4f}) "
              f"val PSNR b/s/t = {row[6]:.2f}/{row[7]:.2f}/{row[9]:.2f} dB "
              f"(ema s {row[8]:.2f}) "
              f"ssim_s={metrics['student'][1]:.4f} ({dt:.0f}s)")

        state_raw = {
            "epoch": epoch,
            "student": student.state_dict(),
            "adapters": adapters.state_dict(),
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
        if metrics_ema["student"][0] > best_ema_psnr:
            best_ema_psnr = metrics_ema["student"][0]
            state_ema = {**state_raw, "student": student_ema.state_dict(),
                         "val_psnr": best_ema_psnr}
            torch.save(state_ema, out_dir / "student_best_ema.pt")
        winner_is_raw = metrics["student"][0] >= metrics_ema["student"][0]
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
