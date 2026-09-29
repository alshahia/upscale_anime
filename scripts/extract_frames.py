# scripts/extract_frames.py
"""Sample diverse distinct 1080p frames from data/anime_vid episodes.

Uniform sampling (~1 frame per SAMPLE_SECS), aHash-16x16 hamming-dedup to drop
static-scene duplicates, JPEG q=95 output (constrained disk space).
Output: data/anime_fullframes/<videotag>_f<num>.jpg

Usage: python scripts/extract_frames.py [--videos data/anime_vid] [--out data/anime_fullframes]
"""
import argparse
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_SECS = 0.7            # ~1 frame per 0.7s across 25 fps sources
TARGET_TOTAL = 13000
DUPLICATE_HAMMING = 7        # aHash 16x16 -> 256 bits; 7/256 ~= static-scene twin
AVG_EVERY = 40               # check an average-hash bucket every N samples


def ahash16(img_gray, size=16):
    """RGB/gray image -> aHash bit-packed array (16x16 = 256 bits)."""
    g = img_gray.convert("L")
    a = np.asarray(g.resize((64, 64), Image.NEAREST), np.float32)
    a = a.reshape(size, 4, size, 4).mean(axis=(1, 3))
    bits = (a > a.mean()).reshape(-1)
    return np.packbits(bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=str(ROOT / "data" / "anime_vid"))
    ap.add_argument("--out", default=str(ROOT / "data" / "anime_fullframes"))
    ap.add_argument("--target", type=int, default=TARGET_TOTAL)
    args = ap.parse_args()
    import av
    video_files = sorted(Path(args.videos).glob("*.mp4"))
    if not video_files:
        raise SystemExit("no mp4 under " + str(Path(args.videos)))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen = []
    t0 = time.time()
    saved_total = 0
    per_video_budget = args.target // len(video_files) + 1
    for vf in video_files:
        if saved_total >= args.target:
            break
        tag = re.sub(r"[^A-Za-z0-9]+", "_", vf.stem)[:24]
        saved_here = 0
        try:
            container = av.open(str(vf))
        except Exception as e:  # noqa: BLE001
            print("SKIP", vf.name, e)
            continue
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 25)
        step = max(1, int(round(fps * SAMPLE_SECS)))
        w, h = stream.width, stream.height
        small_src = (w or 1920) * (h or 1080) < 600_000     # LR video -> upsampled frame
        with container:
            frame = None
            i = 0
            base = container.decode(stream)
            for j, frame in enumerate(base):
                if j % step:
                    continue
                i += 1
                if saved_total >= args.target:
                    break
                img = frame.to_image().convert("RGB")
                if img.width < 400 or img.height < 300:
                    continue
                hsh = ahash16(img)
                dup = False
                for s in seen[-min(len(seen), 4000):]:
                    if np.count_nonzero(hsh != s) <= DUPLICATE_HAMMING:
                        dup = True
                        break
                if dup:
                    continue
                seen.append(hsh)
                out = out_dir / (tag + "_f%05d.jpg" % i)
                img.save(out, format="JPEG", quality=95, subsampling=0)
                saved_total += 1
                saved_here += 1
                if saved_here % 250 == 0:
                    print(vf.name[:40], "saved", saved_here,
                          "total", saved_total, "elapsed", round(time.time() - t0), "s")
                    if saved_here > per_video_budget * 1.5:
                        break
        container.close()
        print("DONE", vf.name[:50], "frames_saved", saved_here)
    print("TOTAL saved:", saved_total, "elapsed:", round(time.time() - t0), "s")
    (out_dir / "_manifest.txt").write_text(
        f"total={saved_total}\nduration_s={int(time.time() - t0)}", encoding="utf-8")


if __name__ == "__main__":
    main()
