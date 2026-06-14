#!/usr/bin/env python3
"""
Build deduplicated anime training corpus from anime_hr + anime_video_frames.

Heavily deduplicates to reduce redundancy from the 5-video source.
Subsamples to target size.

Usage:
    python scripts/build_anime_corpus.py --out data/anime_hr_v6 --target 3000
"""
import argparse
import hashlib
import shutil
from pathlib import Path

from PIL import Image


def compute_image_hash(img_path: Path, thumb_size: tuple = (64, 64)) -> str:
    """Compute perceptual hash via MD5 of downsampled thumbnail."""
    try:
        with Image.open(img_path) as img:
            thumb = img.copy()
            thumb.thumbnail(thumb_size, Image.Resampling.LANCZOS)
            thumb = thumb.convert("RGB")
            return hashlib.md5(thumb.tobytes()).hexdigest()
    except Exception:
        return ""


def deduplicate_images(image_paths: list[Path]) -> list[Path]:
    """Remove near-duplicate images based on thumbnail hash."""
    seen = set()
    kept = []
    for img_path in image_paths:
        h = compute_image_hash(img_path)
        if h and h not in seen:
            seen.add(h)
            kept.append(img_path)
    return kept


def main():
    parser = argparse.ArgumentParser(description="Build deduplicated anime corpus")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--target", type=int, default=3000, help="Target image count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    sources = [
        Path("data/anime_hr"),
        Path("data/anime_video_frames"),
    ]

    all_images = []
    for src in sources:
        if not src.exists():
            print(f"Warning: {src} not found, skipping")
            continue
        images = list(src.glob("*.jpg")) + list(src.glob("*.png")) + list(src.glob("*.jpeg"))
        print(f"Found {len(images)} images in {src}")
        all_images.extend(images)

    print(f"\nTotal images before dedup: {len(all_images)}")

    kept = deduplicate_images(all_images)
    print(f"After dedup (thumbnail MD5): {len(kept)}")

    if len(kept) > args.target:
        kept = random.sample(kept, args.target)
        print(f"Subsampled to target: {len(kept)}")

    args.out.mkdir(parents=True, exist_ok=True)
    for i, img_path in enumerate(kept):
        dst = args.out / f"{i:06d}{img_path.suffix}"
        shutil.copy2(img_path, dst)

    print(f"\nWrote {len(kept)} images to {args.out}")

    manifest_path = args.out / "manifest.txt"
    with open(manifest_path, "w") as f:
        for img_path in kept:
            f.write(f"{img_path}\n")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
