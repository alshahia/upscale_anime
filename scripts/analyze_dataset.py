"""
Dataset analysis script for anime super-resolution.
Analyzes dataset quality, size, and provides recommendations.

Usage:
    python scripts/analyze_dataset.py --dir data/anime_hr
"""
import argparse
import sys
from pathlib import Path
from PIL import Image
import numpy as np


def analyze_dataset(hr_dir: str):
    hr_path = Path(hr_dir)
    if not hr_path.exists():
        print(f"[ERROR] Directory not found: {hr_dir}")
        return

    images = list(hr_path.glob("*.png")) + list(hr_path.glob("*.jpg")) + list(hr_path.glob("*.jpeg"))
    if not images:
        print(f"[ERROR] No images found in {hr_dir}")
        return

    print(f"Dataset: {hr_dir}")
    print(f"Total images: {len(images)}")

    widths = []
    heights = []
    for img_path in images[:100]:
        with Image.open(img_path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)

    if widths:
        print(f"\nResolution (sampled {len(widths)} images):")
        print(f"  Width:  min={min(widths)}, max={max(widths)}, avg={np.mean(widths):.0f}")
        print(f"  Height: min={min(heights)}, max={max(heights)}, avg={np.mean(heights):.0f}")

    print(f"\nRecommendations:")
    if len(images) < 500:
        print(f"  [WARNING] Dataset is small ({len(images)} images). APISR used 1000+ images.")
        print(f"  Consider adding more anime frames for better generalization.")
    else:
        print(f"  [OK] Dataset size is adequate ({len(images)} images).")

    if widths and np.mean(widths) < 720:
        print(f"  [WARNING] Average resolution is low ({np.mean(widths):.0f}px). Target 720p+ for best results.")
    elif widths:
        print(f"  [OK] Average resolution is good ({np.mean(widths):.0f}px).")

    print(f"\nAPISR benchmark comparison:")
    print(f"  APISR: NIQE=6.72, MANIQA=0.514, CLIPIQA=0.711")
    print(f"  (with 1000+ images, 720p+, APISR-style degradation)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze anime SR dataset")
    parser.add_argument("--dir", required=True, help="Path to HR image directory")
    args = parser.parse_args()

    analyze_dataset(args.dir)
