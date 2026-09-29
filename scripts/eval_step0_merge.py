# scripts/eval_step0_merge.py
import csv
from pathlib import Path


def merge_csvs(paths):
    out = []
    for p in paths:
        with Path(p).open(encoding="utf-8") as fh:
            r = csv.DictReader(fh)
            out.extend(list(r))
    return out


if __name__ == "__main__":
    import sys
    rows = merge_csvs([Path(x) for x in sys.argv[1:]])
    for r in rows:
        print(r)
