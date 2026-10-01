#!/usr/bin/env python3
"""Audit all finetune checkpoints for warm-start poisoning.

Reads every .pth under checkpoints/NEOSR_SPAN_V* and checks:
  1. upsampler.0.bias statistics (mean, std) -- taint signature is mean ~0.43, std ~0.02
  2. Forward-pass output statistics on a reference image

Writes results/audit/taint_report.csv and per-checkpoint SR PNGs.

Usage:
  python scripts/audit_pretrain_ckpts.py [--ref-img PATH] [--out-dir DIR]
"""
import argparse
import csv
import sys
import warnings
from pathlib import Path

import torch
import cv2
import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from anime_sr.models.span import create_neosr_span  # noqa: E402


TAINT_BIAS_MIN = 0.35
TAINT_BIAS_MAX = 0.45
TAINT_BIAS_STD_MAX = 0.05


def load_state(path: Path, ckpt_root: Path):
    """Load a finetune checkpoint and return (model_state_dict, epoch, run_name)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict):
        if "model_state_dict" in ck:
            sd = ck["model_state_dict"]
        elif "state_dict" in ck:
            sd = ck["state_dict"]
        else:
            sd = ck
        epoch = ck.get("epoch", "?") if isinstance(ck, dict) else "?"
    else:
        sd = ck
        epoch = "?"
    run_name = path.parent.name if path.parent != ckpt_root else "root"
    return sd, epoch, run_name


def forward_sr(model, img_path: Path, out_path: Path, scale: int = 4) -> dict:
    """Run forward pass on a reference image; save SR PNG. Returns stats dict."""
    img = cv2.imread(str(img_path))
    if img is None:
        return {"output_mean": "?", "output_std": "?", "err": "img missing"}
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        out = model(t)
    stats = {
        "output_shape": "x".join(str(s) for s in out.shape),
        "output_min": round(float(out.min()), 4),
        "output_max": round(float(out.max()), 4),
        "output_mean": round(float(out.mean()), 4),
        "output_std": round(float(out.std()), 4),
    }
    out_np = (out.squeeze(0).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
    out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out_bgr)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Audit finetune checkpoints for warm-start taint")
    parser.add_argument("--ref-img", default="data/test_mini_sr/2.png", help="Reference LR image for forward pass")
    parser.add_argument("--out-dir", default="results/audit", help="Output directory for CSV + PNGs")
    parser.add_argument("--ckpts-root", default="checkpoints", help="Root directory to scan")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = Path(args.ref_img)
    if not img_path.exists():
        print(f"[FATAL] reference image not found: {img_path}")
        sys.exit(1)

    ckpt_root = Path(args.ckpts_root)
    ckpt_paths = sorted(set(ckpt_root.glob("*/*.pth")) | set(ckpt_root.glob("*.pth")))
    if not ckpt_paths:
        print(f"[FATAL] no .pth files found under {ckpt_root}/")
        sys.exit(1)

    print(f"[Audit] scanning {len(ckpt_paths)} checkpoints under {ckpt_root}/")
    print(f"[Audit] reference image: {img_path}")
    print(f"[Audit] output dir: {out_dir}\n")

    rows = []
    model_cache = {}  # run_name -> model (reuse across checkpoints in same run)

    for p in ckpt_paths:
        sd, epoch, run = load_state(p, ckpt_root)
        b = sd.get("upsampler.0.bias")
        if b is None:
            print(f"[WARN] {p}: no upsampler.0.bias -- skipping")
            continue
        bias_min = float(b.min())
        bias_max = float(b.max())
        bias_mean = float(b.mean())
        bias_std = float(b.std())

        tainted = (TAINT_BIAS_MIN <= bias_mean <= TAINT_BIAS_MAX) and (bias_std < TAINT_BIAS_STD_MAX)
        verdict = "TAINTED" if tainted else "clean"

        # Build / reuse model for forward pass
        if run not in model_cache:
            m = create_neosr_span({"type": "neosr_span", "scale": 4})
            model_cache[run] = m
        model = model_cache[run]
        try:
            model.load_state_dict(sd, strict=False)
            model.eval()
            sr_name = f"{run}_{Path(p).stem}.png"
            sr_path = out_dir / sr_name
            fwd_stats = forward_sr(model, img_path, sr_path)
        except Exception as e:
            fwd_stats = {"err": str(e)}

        rows.append({
            "run": run,
            "checkpoint": Path(p).name,
            "epoch": epoch,
            "bias_min": round(bias_min, 4),
            "bias_max": round(bias_max, 4),
            "bias_mean": round(bias_mean, 4),
            "bias_std": round(bias_std, 4),
            "output_min": fwd_stats.get("output_min", "?"),
            "output_max": fwd_stats.get("output_max", "?"),
            "output_mean": fwd_stats.get("output_mean", "?"),
            "output_std": fwd_stats.get("output_std", "?"),
            "verdict": verdict,
        })
        print(f"[{verdict:>7}] {run}/{Path(p).name:>26}  bias_mean={bias_mean:.4f} bias_std={bias_std:.4f}  "
              f"output_mean={fwd_stats.get('output_mean', '?')} output_std={fwd_stats.get('output_std', '?')}")

    csv_path = out_dir / "taint_report.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    tainted_count = sum(1 for r in rows if r["verdict"] == "TAINTED")
    clean_count = sum(1 for r in rows if r["verdict"] == "clean")
    print(f"\n[Audit] Done. {len(rows)} checkpoints: {tainted_count} TAINTED, {clean_count} clean")
    print(f"[Audit] CSV: {csv_path}")
    if tainted_count:
        print("[Audit] TAINTED runs (recommend deletion):")
        seen = set()
        for r in rows:
            if r["verdict"] == "TAINTED" and r["run"] not in seen:
                seen.add(r["run"])
                print(f"  - checkpoints/{r['run']}/")


if __name__ == "__main__":
    main()