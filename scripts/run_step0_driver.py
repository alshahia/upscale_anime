# scripts/run_step0_shootout.ps1-style driver is replaced by this Python driver.
# Runs each candidate in its OWN process (GPU memory fully reset per model),
# then merges the per-model CSVs into results/step0_model_shootout.csv.
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SCRIPT = ROOT / "anime_upscaler" / "eval_step0.py"
PARTS = ROOT / "results" / "step0_parts"

MODELS = [
    "bicubic",
    "realesr-animevideov3",
    "RFDN_distill_v1_student",
    "NEOSR_SPAN_V7_ANIME_best",
    "4x-AnimeSharp",
    "4x_APISR_GRL_GAN",
    "4x_APISR_DAT_GAN",
    "4x_APISR_RRDB_GAN",
]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--degrade", type=int, default=None)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--iqa", default="clipiqa,niqe")
    ap.add_argument("--skip-speed", action="store_true")
    ap.add_argument("--quality-crop", type=int, default=384)
    args = ap.parse_args()

    PARTS.mkdir(parents=True, exist_ok=True)
    for m in args.models:
        out = PARTS / ("step0_" + m + ".csv")
        cmd = [str(PY), str(SCRIPT), "--models", m,
               "--limit", str(args.limit), "--out-csv", str(out),
               "--iqa", args.iqa, "--quality-crop", str(args.quality_crop)]
        if args.degrade is not None:
            cmd += ["--degrade", str(args.degrade)]
        if args.skip_speed:
            cmd += ["--skip-speed"]
        print("\n=== ", m, " ===")
        sys.stdout.flush()
        env = dict(**__import__("os").environ,
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        subprocess.run(cmd, cwd=str(ROOT), env=env)

    # merge
    import csv as _csv
    from eval_step0_merge import merge_csvs  # placed next to this driver
    rows = merge_csvs(sorted(PARTS.glob("*.csv")))
    merged = ROOT / "results" / "step0_model_shootout.csv"
    if rows:
        cols = list(rows[0].keys())
        with merged.open("w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print("wrote", merged)

if __name__ == "__main__":
    main()
