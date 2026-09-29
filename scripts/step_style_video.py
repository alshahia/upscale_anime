#!/usr/bin/env python3
"""Render a chosen time range of a video with a (style-blended) SRVGG model,
optionally applying a reference-image "style" via color-statistics matching.

Usage:
  python scripts/step_style_video.py --start 20:40 --end 20:50 \
      --ckpt results/style_mix/weights/stylemix_a0.50.pth \
      [--style-ref path/to/style.png]  [--out results/style_video]

Style control follows the layer order:
  1) weights  (alpha blend of two same-arch nets, step_style_mix.py)
  2) tone     (color mean/std transfer toward a reference image, post-SR)
  3) geometry (sharpen/denoise strength constants)
A style reference changes look 2 only - it steering palette/contrast while
structure/lines come from the SR model.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import imageio
import av
from fractions import Fraction

ROOT = Path(__file__).resolve().parent.parent


def parse_hms(s: str) -> float:
    """Parse 'MM:SS', 'HH:MM:SS', or plain seconds. This video is 23:44 long,
    so an 'HH:MM:SS' input whose hours would exceed the source is re-read as
    MINUTES:SECONDS (e.g. 20:45:00 -> 1245 s, matching 20 min 45 s)."""
    parts = [float(p) for p in s.split(":")]
    t = 0.0
    for p in parts:
        t = t * 60 + p
    if len(parts) == 3 and t > 1424.1:  # source duration sanity check
        return parts[0] * 60 + parts[1] + parts[2] / 60.0
    return t


def color_transfer(img: np.ndarray, ref: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Per-channel mean/std matching of img toward ref (both uint8 RGB)."""
    out = img.astype(np.float32) / 255.0
    rf = ref.astype(np.float32) / 255.0
    m, sd = out.mean((0, 1)), out.std((0, 1)) + 1e-8
    mr, sr = rf.mean((0, 1)), rf.std((0, 1)) + 1e-8
    tgt_m = m + strength * (mr - m)
    tgt_s = sd * (1.0 + strength * (sr / sd - 1.0))
    out = (out - m) / sd * tgt_s + tgt_m
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=r"C:/Users/Ahmad Mahmoud/Downloads/Video/[Anime3rb.com] Yi Ren Zhi Xia - 1 [480p].mp4")
    ap.add_argument("--start", default="20:40", help="HH:MM:SS or MM:SS")
    ap.add_argument("--end", default="20:50")
    ap.add_argument("--ckpt", default=str(ROOT / "results/style_mix/weights/stylemix_a0.50.pth"))
    ap.add_argument("--style-ref", default="", help="optional reference image for color style matching")
    ap.add_argument("--style-strength", type=float, default=1.0)
    ap.add_argument("--alpha-name", default="", help="label for the output filename")
    ap.add_argument("--out", default=str(ROOT / "results" / "style_video"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    DEV = torch.device("cuda")

    sys.path.insert(0, str(ROOT / "apps" / "anime_upscaler_gui"))
    from anime_upscaler_gui.archs import build
    model = build("srvgg", args.ckpt).to(DEV).eval().half()
    label = args.alpha_name or Path(args.ckpt).stem
    mp4_path = out / f"upscaled_{label}.mp4"
    clip_path = out / f"src_clip_{args.start.replace(':', '')}_{args.end.replace(':', '')}.mp4"

    # 1) extract the requested segment (re-encoded mp4 via imageio/ffmpeg writer,
    #    which stamps a correct fps/duration timebase)
    t0, t1 = parse_hms(args.start), parse_hms(args.end)
    c = av.open(args.source)
    s = c.streams.video[0]
    fps = float(s.average_rate or 25)
    c.seek(int(t0 / s.time_base), stream=s)
    n = int(round((t1 - t0) * fps))
    got = 0
    with imageio.get_writer(str(clip_path), fps=round(fps), macro_block_size=8) as vw:
        for frame in c.decode(s):
            # exact timeline: skip frames decoded before the requested start
            if frame.time is not None and frame.time < t0 - 1e-4:
                continue
            vw.append_data(frame.to_ndarray(format="rgb24"))
            got += 1
            if got >= n:
                break
    c.close()
    print(f"[extract] got={got} frames -> {clip_path} (exists={clip_path.exists()})", flush=True)
    if got == 0 or not clip_path.exists():
        raise SystemExit("extraction produced no clip")

    # 2) decode the clip, upscale with the blended model, optional style ref
    ref = None
    if args.style_ref:
        ref = np.asarray(imageio.imread(args.style_ref), np.uint8)[..., :3]
        ref = np.asarray(ref, np.float32)
        print("[style-ref] using color statistics of", args.style_ref, flush=True)

    cc = av.open(str(clip_path))
    sv = cc.streams.video[0]
    frames = [f.to_ndarray(format="rgb24") for f in cc.decode(sv)]
    cc.close()
    print(f"[clip] {len(frames)} frames @ {fps:.2f} fps, t={t0:.0f}-{t1:.0f}s", flush=True)

    def clear_stat():
        import torch
        return torch.cuda.max_memory_allocated()/2**10

    iidx = 0
    outs = []
    ms = []
    with torch.inference_mode():
        for a in frames:
            t1s = time.perf_counter()
            t = torch.from_numpy(a).permute(2, 0, 1)[None].to(DEV).half().div(255.0).contiguous()
            y = model(t).clamp(0, 1)
            torch.cuda.synchronize()
            ms.append((time.perf_counter() - t1s) * 1000.0)
            o = (y[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)
            if ref is not None:
                o = color_transfer(o, ref, args.style_strength)
            outs.append(o)
            iidx += 1
    with imageio.get_writer(str(mp4_path), fps=fps, macro_block_size=8) as vw:
        for o in outs:
            vw.append_data(o)
    import statistics
    ms = np.array(ms)
    stats = {"ckpt": args.ckpt, "style_ref": args.style_ref,
             "style_strength": args.style_strength,
             "frames": len(outs), "mean_ms": round(float(ms.mean()), 2),
             "fps_throughput": round(len(ms) / (ms.sum() / 1000), 2),
             "peak_GB": round(torch.cuda.max_memory_allocated()/2**30, 2)}
    (out / f"stats_{label}.json").write_text(json.dumps(stats, indent=2))
    print("[done]", json.dumps(stats), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
