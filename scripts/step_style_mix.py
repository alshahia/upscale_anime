#!/usr/bin/env python3
"""Option 2 (style dial): interpolate two same-arch SRVGG models.

realesr-animevideov3 and 4xLSDIRCompactv2 are both SRVGGNetCompact with
identical key-shapes (621,424 params), so their weights can be blended:

    w_alpha = alpha * animevideov3 + (1 - alpha) * LSDIR

alpha=0.0 -> pure LSDIR look, 1.0 -> pure animevideov3 look.
Each alpha is benched with scripts/step_bench_one_model.py on the same
5 s clip (lr_frames), results merged into summary.csv.

    python scripts/step_style_mix.py [--alphas 0.0,0.25,0.5,0.75,1.0]
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "apps" / "anime_upscaler_gui"))
PY = sys.executable

SRC_A = ROOT / "pretrained" / "realesr-animevideov3.pth"   # anime-video look
SRC_B = ROOT / "pretrained" / "4xLSDIRCompactv2.pth"       # LSDIR "clean" look


def load_sd(path: Path) -> dict:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "params" in sd and isinstance(sd["params"], dict):
        sd = sd["params"]
    return sd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--out", default=str(ROOT / "results" / "style_mix"))
    ap.add_argument("--bench-src", default=str(ROOT / "results" / "bench_all"),
                    help="existing bench dir holding lr_frames + meta.json")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    weights = out / "weights"
    weights.mkdir(exist_ok=True)
    bench = out / "bench"
    bench.mkdir(exist_ok=True)

    # reuse the LR frames + metadata from the main bench
    if not (bench / "lr_frames").exists():
        shutil.copytree(Path(args.bench_src) / "lr_frames", bench / "lr_frames")
        meta = json.loads((Path(args.bench_src) / "meta.json").read_text())
        meta["source"] = args.bench_src  # note the provenance of the frames
        meta["note"] = "frames copied from results/bench_all; alpha-blended SRVGG weights"
        (bench / "meta.json").write_text(json.dumps(meta, indent=2))

    sa = load_sd(SRC_A)
    sb = load_sd(SRC_B)
    keys_a, keys_b = set(sa), set(sb)
    assert keys_a == keys_b, f"arch mismatch: {len(keys_a ^ keys_b)} differing keys"
    bad = [k for k in sa if sa[k].shape != sb[k].shape]
    assert not bad, f"shape mismatch: {bad[:5]}"

    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    rows = []
    for alpha in alphas:
        mid = f"stylemix_a{alpha:.2f}"
        blended = {}
        for k in sa:
            blended[k] = (alpha * sa[k].float() + (1.0 - alpha) * sb[k].float()).to(sa[k].dtype)
        ckpt = weights / f"{mid}.pth"
        torch.save({"params": blended, "scale": 4}, ckpt)
        print(f"[mix] alpha={alpha:.2f} -> {ckpt}", flush=True)
        r = subprocess.run([PY, str(ROOT / "scripts" / "step_bench_one_model.py"),
                            "--ckpt", str(ckpt), "--kind", "srvgg",
                            "--model-id", mid, "--scale", "4",
                            "--bench-dir", str(bench)],
                           check=False, cwd=str(ROOT))
        rp = bench / "models" / mid / "row.csv"
        if rp.exists():
            import csv as _csv
            rr = list(_csv.reader(open(rp, newline="")))
            if len(rr) >= 2 and len(rr[0]) == len(rr[1]):
                rows.append(dict(zip(rr[0], rr[1])))
        else:
            rows.append({"model_id": mid, "alpha": alpha, "status": "crashed"})

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        w.writeheader()
        w.writerows(rows)
    print(f"[done] {out / 'summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
