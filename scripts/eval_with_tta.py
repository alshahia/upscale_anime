#!/usr/bin/env python3
"""Phase 5 helper: TTA evaluation for a specific checkpoint.

Usage:
    python scripts/eval_with_tta.py --ckpt pretrained/RFDN_distill_v1_4x_student.pth
    python scripts/eval_with_tta.py --ckpt runs/distill_v3_ablation_A_degradation_off/student_last.pt

Loads the named checkpoint, restores the student, and runs the FULL held-out
test split with D4 TTA enabled (8x for square LR).

Writes <ckpt-dir>/eval_with_tta_<split>_results_table.{md,csv}.
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "anime_upscaler"
for p in (ROOT, PKG):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from anime_upscaler.distill import (  # noqa: E402
    evaluate, set_seed, seed_worker, psnr01, ssim01,
)
from anime_upscaler.student import RFDN  # noqa: E402
from anime_upscaler.teacher import SPANTeacher  # noqa: E402
from tta import tta_forward  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
import anime_upscaler.dataset as _ds  # noqa: E402
AnimePairDataset = _ds.AnimePairDataset
denorm01 = _ds.denorm01


@torch.no_grad()
def tta_psnr_ssim(model, loader, device):
    """Run D4 TTA on each batch, accumulate bicubic / student(TTA) / teacher PSNR+SSIM."""
    sums = {k: [0.0, 0.0] for k in ("bicubic", "student", "teacher")}
    nb = 0
    n_aug = None
    names_seen = None
    for lr_n, hr_n in loader:
        lr, hr = denorm01(lr_n).to(device), denorm01(hr_n).to(device)
        bic = F.interpolate(lr, scale_factor=4, mode="bicubic",
                            align_corners=False).clamp(0, 1)
        sr, na, nm = tta_forward(model, lr)
        sr = sr.clamp(0, 1)
        tt = model.teacher(lr).clamp(0, 1) if hasattr(model, "teacher") else None
        # teacher fallback: caller passes teacher separately; here we just record student.
        for name, pred in (("bicubic", bic), ("student", sr)):
            sums[name][0] += psnr01(pred, hr)
            sums[name][1] += ssim01(pred, hr)
        if n_aug is None:
            n_aug, names_seen = na, nm
        nb += 1
    out = {k: (v[0] / nb, v[1] / nb) for k, v in sums.items()}
    out["_n_aug"] = n_aug
    out["_names"] = names_seen
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True,
                    help="path to a student .pt (last / best / best_raw)")
    ap.add_argument("--data", default=str(ROOT / "data" / "anime_video_frames"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.ckpt).resolve().parent
    tag = Path(args.ckpt).stem

    ds = AnimePairDataset(args.data, args.split)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True,
                    worker_init_fn=seed_worker,
                    persistent_workers=args.num_workers > 0)

    teacher = SPANTeacher(None, device=device)
    student = RFDN().to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    student.load_state_dict(ck["student"])
    print(f"[eval_tta] loaded {args.ckpt} (epoch={ck.get("epoch")} val_psnr={ck.get("val_psnr")})")
    print(f"[eval_tta] split={args.split} n={len(ds)} batch={args.batch_size}")

    # Quick baseline (no TTA) for comparison
    print("[eval_tta] running no-TTA baseline...")
    base = evaluate(student, teacher, dl, device)

    # TTA pass: re-run, this time via tta_forward
    print("[eval_tta] running TTA pass...")
    # Manual replicate of evaluate(), but the student fwd is tta_forward instead.
    sums = {k: [0.0, 0.0] for k in ("bicubic", "student", "teacher")}
    n_aug = None
    names_seen = None
    nb = 0
    student.eval()
    for lr_n, hr_n in dl:
        lr = denorm01(lr_n).to(device)
        hr = denorm01(hr_n).to(device)
        bic = F.interpolate(lr, scale_factor=4, mode="bicubic",
                            align_corners=False).clamp(0, 1)
        sr, na, nm = tta_forward(student, lr)
        sr = sr.clamp(0, 1)
        # teacher via frozen wrapper
        with torch.no_grad():
            tt = teacher(lr).clamp(0, 1)
        for name, pred in (("bicubic", bic), ("student", sr), ("teacher", tt)):
            sums[name][0] += psnr01(pred, hr)
            sums[name][1] += ssim01(pred, hr)
        if n_aug is None: n_aug, names_seen = na, nm
        nb += 1
    tta_res = {k: (v[0] / nb, v[1] / nb) for k, v in sums.items()}

    print(f"[eval_tta] D4 TTA: {n_aug}x ({names_seen})")
    print(f"[eval_tta] no-TTA  : student={base["student"][0]:.2f}/{base["student"][1]:.4f}")
    print(f"[eval_tta]   TTA    : student={tta_res["student"][0]:.2f}/{tta_res["student"][1]:.4f}")
    print(f"[eval_tta] teacher : {tta_res["teacher"][0]:.2f}/{tta_res["teacher"][1]:.4f}")

    delta_psnr = tta_res["student"][0] - base["student"][0]
    delta_ssim = tta_res["student"][1] - base["student"][1]
    gain = {"model": "retention"}
    print(f"[eval_tta] TTA delta vs no-TTA: {delta_psnr:+.2f} dB / {delta_ssim:+.4f} SSIM")

    # Write artifacts
    lines = ["| model | PSNR (dB) | SSIM |", "|---|---|---|"]
    for k in ("bicubic", "student", "teacher"):
        lines.append(f"| {k} | {tta_res[k][0]:.2f} | {tta_res[k][1]:.4f} |")
    lines.append(f"| TTA delta vs no-TTA ({n_aug}x) | {delta_psnr:+.2f} | {delta_ssim:+.4f} |")
    lines.append(f"| no-TTA student (control) | {base["student"][0]:.2f} | {base["student"][1]:.4f} |")
    table = "\n".join(lines)
    print("\n=== TTA-EVAL TEST SET ===\n" + table)

    md_path = out_dir / f"eval_tta_{args.split}_{tag}_results_table.md"
    csv_path = out_dir / f"eval_tta_{args.split}_{tag}_results_table.csv"
    md_path.write_text(table + "\n")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "psnr_db", "ssim"])
        for k in ("bicubic", "student", "teacher"):
            w.writerow([k, f"{tta_res[k][0]:.2f}", f"{tta_res[k][1]:.4f}"])
        w.writerow([f"tta_delta_{n_aug}x", f"{delta_psnr:+.2f}", f"{delta_ssim:+.4f}"])
        w.writerow(["no_tta_student", f"{base["student"][0]:.2f}", f"{base["student"][1]:.4f}"])
    print(f"[eval_tta] wrote {md_path} and {csv_path}")


if __name__ == "__main__":
    main()