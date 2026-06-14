#!/usr/bin/env python3
"""
Sample one image from each data subfolder, apply training degradation, save LR/HR pairs.

This shows what the training pipeline sees:
- HR (64x64): the ground truth crop
- LR (16x16): the degraded input that feeds the model
- LR_upscaled (64x64): bicubic upsample of LR (what the model starts with)
"""
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, "src")
from data.degradation_pipeline import DegradationPipeline


def main():
    data_dir = Path("data")
    sample_dir = Path("sample")
    sample_dir.mkdir(exist_ok=True)
    (sample_dir / "hr").mkdir(exist_ok=True)
    (sample_dir / "lr").mkdir(exist_ok=True)
    (sample_dir / "lr_upscaled").mkdir(exist_ok=True)

    # Collect one image per subfolder
    samples = []
    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("."):
            continue
        images = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png")) + list(subdir.glob("*.jpeg"))
        if images:
            img = random.choice(images)
            samples.append((subdir.name, img))
            print(f"[{subdir.name}] {img.name}")

    print(f"\nCollected {len(samples)} samples")

    # Initialize degradation pipeline (APISR-style from v6 config)
    pipeline = DegradationPipeline(
        mode="anime_heavy",
        scale=4,
        seed=42,
        two_stage=True,
        compression_stage1=["jpeg", "webp"],
        compression_stage2=["avif", "h264", "jpeg"],
    )

    # Process each sample
    for folder_name, img_path in samples:
        # Load image
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Crop to 64x64 (or smaller if image is tiny)
        crop_size = min(64, w, h)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        img_crop = img.crop((left, top, left + crop_size, top + crop_size))

        # Save HR (ground truth)
        hr_path = sample_dir / "hr" / f"{folder_name}.png"
        img_crop.save(hr_path)

        # Apply degradation to get LR
        img_np = np.array(img_crop)
        lr_np = pipeline(img_np)
        lr_img = Image.fromarray(lr_np)

        # Save LR (degraded input, 16x16)
        lr_path = sample_dir / "lr" / f"{folder_name}.png"
        lr_img.save(lr_path)

        # Upscale LR back to 64x64 using bicubic (what the model starts with)
        lr_upscaled = lr_img.resize((crop_size, crop_size), Image.Resampling.BICUBIC)
        lr_up_path = sample_dir / "lr_upscaled" / f"{folder_name}.png"
        lr_upscaled.save(lr_up_path)

        print(f"  {folder_name}: HR={crop_size}x{crop_size} -> LR={lr_np.shape[1]}x{lr_np.shape[0]}")

    print(f"\nDone. Check {sample_dir}/")
    print(f"  hr/          - Ground truth crops (64x64)")
    print(f"  lr/          - Degraded inputs (16x16)")
    print(f"  lr_upscaled/ - Bicubic upscaled (64x64)")


if __name__ == "__main__":
    random.seed(42)
    main()
