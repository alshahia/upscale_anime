"""Phase 5 helper: test-set evaluation for a specific v3 checkpoint.

Usage:
    python scripts/eval_v3_ckpt.py --ckpt runs/distill_v3/student_last.pt
    python scripts/eval_v3_ckpt.py --ckpt runs/distill_v3/student_best.pt

Loads the named checkpoint, restores the student weight state, and runs
the FULL held-out test split (no max_batches cap). Writes:
    <ckpt-dir>/eval_<tag>_results_table.md
    <ckpt-dir>/eval_<tag>_results_table.csv
where <tag> = base ckpt filename with extension stripped (e.g. student_last).
"""
import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent

from anime_sr.training.distillation.distill import (  # noqa: E402
    evaluate, set_seed, seed_worker,
)
from anime_sr.models.students import RFDN  # noqa: E402
from anime_sr.models.teachers import SPANTeacher  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from anime_sr.data.datasets import image as _ds  # noqa: E402  (re-export AnimePairDataset, denorm01)
AnimePairDataset = _ds.AnimePairDataset
denorm01 = _ds.denorm01


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True,
                    help="path to a student_last.pt / student_best.pt")
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
    print(f"[eval] loaded {args.ckpt}")
    print(f"[eval] ckpt epoch={ck.get('epoch')} val_psnr={ck.get('val_psnr')}")
    print(f"[eval] split={args.split} n={len(ds)} batch={args.batch_size}")

    final = evaluate(student, teacher, dl, device)  # full split, no cap
    gain_b = final["student"][0] - final["bicubic"][0]
    gain_t = final["teacher"][0] - final["bicubic"][0]
    retention = 100.0 * gain_b / gain_t if gain_t > 0 else float("nan")
    lines = ["| model | PSNR (dB) | SSIM |", "|---|---|---|"]
    for k in ("bicubic", "student", "teacher"):
        lines.append(f"| {k} | {final[k][0]:.2f} | {final[k][1]:.4f} |")
    lines.append(f"| student retention of teacher gain | {retention:.1f}% | |")
    table = "\n".join(lines)
    print("\n=== EVAL TEST SET (ckpt=" + tag + ") ===\n" + table)

    md_path = out_dir / f"eval_{tag}_results_table.md"
    csv_path = out_dir / f"eval_{tag}_results_table.csv"
    md_path.write_text(table + "\n")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "psnr_db", "ssim"])
        for k in ("bicubic", "student", "teacher"):
            w.writerow([k, "%.2f" % final[k][0], "%.4f" % final[k][1]])
        w.writerow(["retention_pct", "%.1f" % retention, ""])
    print(f"[eval] wrote {md_path} and {csv_path}")


if __name__ == "__main__":
    main()
