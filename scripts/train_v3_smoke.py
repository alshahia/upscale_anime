#!/usr/bin/env python3
"""2-epoch smoke harness for the Phase 3 adversarial student (handoff B).

Loads anime_upscaler/distill.py as a library and runs a tiny subset of
the dataset for 2 epochs to verify the full loop (teacher forward, D step,
G step, EMA update, per-epoch ckpt, sample SR PNG) without paying the
~15 h training cost. Outputs to runs/smoke_v3_<tag>/ so multiple configs
(adv off / on / anneal on) can be compared side-by-side.

Phase 3 handoff: docs/plans/student_adversarial_handoff_2026_08.md (A.6, B.1-B.3).
"""
import argparse
import sys
from pathlib import Path

# Make anime_upscaler/ importable. Same PYTHONPATH rule as distill.py.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "anime_upscaler"))
sys.path.insert(0, str(REPO_ROOT / "apps" / "anime_upscaler_gui"))

import torch  # noqa: E402  (after sys.path mutation)
from PIL import Image  # noqa: E402
import numpy as np  # noqa: E402

# Import distill.py as a library (no main() invocation).
import distill as _distill  # noqa: E402


def _save_sample(sr_tensor, out_path):
    """Save a single SR tensor (1,3,H,W) in [0,1] as a PNG."""
    import cv2
    arr = (sr_tensor.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), arr)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=2,
                    help="number of epochs (default 2; handoff B spec)")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="mini-batch size (default 4 for smoke)")
    ap.add_argument("--teacher", default="animevideov3",
                    choices=["span", "animevideov3", "lsdir"],
                    help="teacher name (handoff A.4 spec)")
    ap.add_argument("--lambda-adv", type=float, default=0.001,
                    help="peak adversarial weight (set 0 for B.1 adv-off smoke)")
    ap.add_argument("--shortcut-anneal", default="off",
                    choices=["off", "1to0"],
                    help="bicubic shortcut anneal (handoff A.4)")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default: runs/smoke_v3_<tag>)")
    ap.add_argument("--resume", default=None,
                    help="warm-start ckpt path (handoff C.1)")
    args = ap.parse_args()

    # Build tag from config (default locations match handoff B.1-B.3).
    if args.out_dir is None:
        parts = ["smoke_v3"]
        parts.append(args.teacher)
        parts.append("adv" if args.lambda_adv > 0 else "noadv")
        parts.append("anneal" if args.shortcut_anneal != "off" else "noanneal")
        args.out_dir = str(REPO_ROOT / "runs" / "_".join(parts))

    # Run distill.main() via the embedded smoke path: rebuild sys.argv
    # to the minimal set distill.main() needs. Easiest: pass arguments
    # via a fresh argv that distill.py parses normally.
    print(f"[smoke] config: teacher={args.teacher} lambda_adv={args.lambda_adv} "
          f"shortcut_anneal={args.shortcut_anneal} out_dir={args.out_dir}")
    sys.argv = [
        "distill.py",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--teacher", args.teacher,
        "--lambda-adv", str(args.lambda_adv),
        "--shortcut-anneal", args.shortcut_anneal,
        "--out-dir", args.out_dir,
        "--val-batches", "2",
        "--num-workers", "2",
    ]
    if args.resume:
        sys.argv += ["--resume", args.resume]
    print("[smoke] delegating to distill.main() with argv:")
    print("  ", " ".join(sys.argv[1:]))
    _distill.main()

    # After the run, save a sample SR PNG from the best epoch so the user
    # can visually inspect the result without re-running inference.
    out = Path(args.out_dir)
    samples = sorted(out.glob("epoch_*_metrics.json"))
    if not samples:
        print("[smoke] no per-epoch metrics found; sample PNG skipped")
        return 0
    # Pick the last (most-trained) epoch
    last = samples[-1]
    last_ckpt = out / (last.stem.replace("_metrics", "") + ".pt")
    if not last_ckpt.exists():
        print(f"[smoke] expected {last_ckpt} but missing; sample skipped")
        return 0
    # Lazy-build a tiny loader to render one sample.
    from dataset import AnimePairDataset, denorm01  # noqa: E402
    from student import RFDN  # noqa: E402
    ds = AnimePairDataset(str(REPO_ROOT / "data" / "anime_video_frames"),
                          split="val", max_files=1)
    lr_n, hr_n = ds[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
    student = RFDN(scale=4).to(device)
    student.load_state_dict(ckpt["student"])
    student.eval()
    lr = denorm01(lr_n).unsqueeze(0).to(device)
    with torch.no_grad():
        sr = student(lr).clamp(0, 1)
    sample_path = out / "sample.png"
    _save_sample(sr, sample_path)
    print(f"[smoke] sample SR -> {sample_path}  shape={tuple(sr.shape)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
