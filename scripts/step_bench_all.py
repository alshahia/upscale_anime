#!/usr/bin/env python3
"""Bench ALL available models on a middle 5-second clip of one video.

Structure written (AC power / full clocks required for honest fps numbers):
  <out>/source_clip_5s.mp4          the 5 s source segment (as-is, 480p)
  <out>/lr_frames/                  extracted LR input frames (PNG)
  <out>/meta.json                   source metadata + seek position
  <out>/models/<model_id>/upscaled.mp4   per-model 4x upscale result
  <out>/models/<model_id>/frames.csv     per-frame GPU timing (ms)
  <out>/models/<model_id>/row.csv        per-model summary row
  <out>/summary.csv                 all model rows merged (+ skipped models)
  <out>/summary.xlsx                spreadsheet version (if an xlsx engine exists)

Usage:
    python scripts/step_bench_all.py [source.mp4] [--seconds 5] [--out results/bench_all]
"""
import argparse
import csv
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "apps" / "anime_upscaler_gui"))

DEFAULT_SRC = Path("C:/Users/Ahmad Mahmoud/Downloads/Video/"
                   "[Anime3rb.com] Yi Ren Zhi Xia - 1 [480p].mp4")
PY = sys.executable
KID = ROOT / "scripts" / "step_bench_one_model.py"


def extract_clip(src: Path, bench: Path, seconds: int) -> dict:
    import av
    container = av.open(str(src))
    stream = container.streams.video[0]
    fps = float(stream.average_rate or 25)
    dur = float(stream.duration * stream.time_base) if stream.duration else None
    if stream.duration:
        dur = stream.duration * stream.time_base
    else:
        dur = None
    W = stream.codec_context.width
    H = stream.codec_context.height
    container.close()
    t_start = max(0.0, (dur if dur else seconds) / 2.0 - seconds / 2.0)

    container = av.open(str(src))
    stream = container.streams.video[0]
    container.seek(int(t_start / stream.time_base), stream=stream)
    out_path = bench / "source_clip_5s.mp4"
    n_target = int(round(fps * seconds))
    got = 0
    with av.open(str(out_path), "w") as oc:
        ov = oc.add_stream("libx264", rate=round(fps))
        ov.width, ov.height = W, H
        ov.pix_fmt = "yuv420p"
        for frame in container.decode(stream):
            frame.pts = got
            for pkt in ov.encode(frame):
                oc.mux(pkt)
            got += 1
            if got >= n_target:
                break
        for pkt in ov.encode():
            oc.mux(pkt)
    container.close()
    return {"source": str(src), "t_start": round(t_start, 3), "seconds": seconds,
            "fps": fps, "width": W, "height": H, "frames": got,
            "src_clip": str(out_path)}


def extract_lr_frames(bench: Path, meta: dict) -> None:
    import av
    import imageio
    frames_dir = bench / "lr_frames"
    frames_dir.mkdir(exist_ok=True)
    container = av.open(meta["src_clip"])
    stream = container.streams.video[0]
    i = 0
    for frame in container.decode(stream):
        a = frame.to_image()                      # PIL RGB at native 480p res
        imageio.imwrite(str(frames_dir / f"{i:05d}.png"), a)
        i += 1
    container.close()
    meta["lr_frames"] = i


def model_list():
    from anime_upscaler_gui.registry import ModelRegistry
    reg = ModelRegistry(ROOT / "pretrained")
    entries = []
    for m in reg.scan_installed():
        entries.append({
            "model_id": m.path.stem,
            "ckpt": str(m.path),
            "kind": m.kind or "",
            "scale": m.scale,
            "supported": bool(m.kind and (m.kind in ("srvgg", "span", "era",
                                                     "animesr", "rfdn_student",
                                                     "srvgg_student"))),
            "is_trained": m.is_trained,
            "size_mb": m.size_mb,
        })
    # pretrained/ also holds .pt arch weights that the scan (glob *.pth) misses
    for p in sorted((ROOT / "pretrained").glob("*.pt")):
        entries.append({"model_id": p.stem, "ckpt": str(p), "kind": "?",
                        "scale": "?", "supported": False, "is_trained": False,
                        "size_mb": round(p.stat().st_size / 1e6, 2)})
    return entries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default=str(DEFAULT_SRC))
    ap.add_argument("--seconds", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "results" / "bench_all"))
    args = ap.parse_args()
    src = Path(args.source)
    assert src.exists(), f"missing source video: {src}"
    bench = Path(args.out)
    bench.mkdir(parents=True, exist_ok=True)
    (bench / "models").mkdir(exist_ok=True)

    # 1) clip + frames
    meta = extract_clip(src, bench, args.seconds)
    extract_lr_frames(bench, meta)
    (bench / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[clip] {meta['frames']} frames @ {meta['fps']:.2f} fps, "
          f"{meta['width']}x{meta['height']} from t={meta['t_start']:.2f}s", flush=True)

    entries = model_list()
    with open(bench / "models_manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(entries[0].keys()))
        wr.writeheader()
        wr.writerows(entries)

    # 2) run each supported model in a clean subprocess
    rows = []
    for e in entries:
        mid = e["model_id"]
        mdir = bench / "models" / mid
        if not e["supported"]:
            rows.append({"model_id": mid, "kind": e["kind"], "status":
                         "skipped-unsupported-arch", "ckpt": e["ckpt"],
                         "size_mb": e["size_mb"]})
            print(f"[skip] {mid}: unsupported arch (kind={e['kind']})", flush=True)
            continue
        t0 = time.time()
        subprocess.run([PY, str(KID), "--ckpt", e["ckpt"], "--kind", e["kind"],
                        "--model-id", mid, "--scale", str(e["scale"]),
                        "--bench-dir", str(bench)],
                       check=False, cwd=str(ROOT))
        row_path = mdir / "row.csv"
        if row_path.exists():
            with open(row_path, newline="") as f:
                rr = list(csv.reader(f))
            if rr and rr[0][:2] == ["status", "error"]:
                msg = rr[1][2] if (len(rr) > 1 and len(rr[1]) > 2) else (rr[0][2] if len(rr[0]) > 2 else rr[0][1])
                rows.append({"model_id": mid, "kind": e["kind"], "status": f"failed: {msg}"})
            elif len(rr) >= 2 and len(rr[0]) == len(rr[1]):
                rows.append(dict(zip(rr[0], rr[1])))
            else:
                rows.append({"model_id": mid, "kind": e["kind"], "status": "crashed"})
        else:
            rows.append({"model_id": mid, "kind": e["kind"],
                         "status": "crashed", "ckpt": e["ckpt"]})
        print(f"[bench] {mid} done in {time.time()-t0:.1f}s", flush=True)

    # 3) bicubic reference
    try:
        import imageio
        import numpy as np
        from PIL import Image
        frames = sorted((bench / "lr_frames").glob("*.png"))
        wc = meta["width"] * 4
        hc = meta["height"] * 4
        outs = []
        ms = []
        for i, p in enumerate(frames):
            im = Image.open(p).convert("RGB")
            t1 = time.perf_counter()
            outs.append(np.asarray(im.resize((wc, hc), Image.BICUBIC)))
            ms.append((time.perf_counter() - t1) * 1000)
        mp4 = bench / "models" / "bicubic_ref" / "upscaled.mp4"
        mp4.parent.mkdir(exist_ok=True)
        with imageio.get_writer(str(mp4), fps=meta["fps"], macro_block_size=8) as vw:
            for a in outs:
                vw.append_data(a)
        iqas = {}
        try:
            import pyiqa
            import torch
            DEV = torch.device("cuda")
            for name in ("clipiqa", "niqe"):
                m = pyiqa.create_metric(name, device=str(DEV))
                vals = []
                with torch.inference_mode():
                    for idx in np.linspace(0, len(outs) - 1, 8).astype(int):
                        vals.append(float(m(torch.from_numpy(outs[idx]).permute(2, 0, 1)[None].to(DEV).float() / 255.0).item()))
                iqas[name] = vals
        except Exception as ex:
            iqas["error"] = repr(ex)
            print(f"[warn] bicubic iqa failed: {ex!r}", flush=True)
        rows.append({"model_id": "bicubic_ref", "kind": "cpu-resize", "scale": 4,
                     "status": "ok", "backend": "PIL-bicubic (CPU)",
                     "device": "cpu", "sr_out": f"{wc}x{hc}", "n_frames": len(outs),
                     "mean_ms": round(float(np.mean(ms)), 3),
                     "fps_1_over_mean": round(1000.0 / float(np.mean(ms)), 1),
                     "fps_throughput": "",
                     "clipiqa_mean": (round(np.mean(iqas.get("clipiqa", [math.nan])), 2)
                                      if "clipiqa" in iqas else ""),
                     "niqe_mean": (round(np.mean(iqas.get("niqe", [math.nan])), 2)
                                   if "niqe" in iqas else "")})
    except Exception as ex:
        rows.append({"model_id": "bicubic_ref", "status": f"failed: {ex!r}"})

    # 4) merge + xlsx
    all_keys = sorted(set().union(*(r.keys() for r in rows)))
    with open(bench / "summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=all_keys)
        wr.writeheader()
        wr.writerows(rows)
    try:
        import pandas as pd
        pd.DataFrame(rows).to_excel(bench / "summary.xlsx", index=False)
        print("[done] wrote summary.xlsx", flush=True)
    except Exception as ex:
        print(f"[note] summary.xlsx skipped (engine missing): {ex!r}", flush=True)
    print(f"[done] wrote {bench / 'summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
