"""
Pre-compute LR/HR degradation pairs for faster training.
Applies the same degradation pipeline as BaseDataset but saves results to disk.
This eliminates on-the-fly degradation overhead during training.

Usage:
    python scripts/precompute_degradation.py --config configs/finetune_neosr_span_v4.yaml --output data/precomputed
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch

from anime_sr.data.base import BaseDataset, DatasetFactory


def load_config(config_path: str) -> dict:
    """Load and merge config files."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'base' in config:
        base_path = os.path.join(os.path.dirname(config_path), config['base'])
        with open(base_path, 'r') as f:
            base_config = yaml.safe_load(f)
        merged = {**base_config, **config}
        return merged
    return config


def precompute_pairs(
    hr_dir: str,
    output_dir: str,
    scale: int = 4,
    crop_size: int = 128,
    degradation_cfg: dict = None,
    max_samples: int = None,
):
    """
    Pre-compute LR/HR pairs using the same degradation pipeline as training.
    
    Args:
        hr_dir: Directory with HR images
        output_dir: Directory to save precomputed pairs
        scale: Upscale factor
        crop_size: Crop size for patches
        degradation_cfg: Degradation configuration
        max_samples: Maximum number of samples to precompute (None = all)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    lr_dir = output_path / 'lr'
    hr_dir_out = output_path / 'hr'
    lr_dir.mkdir(exist_ok=True)
    hr_dir_out.mkdir(exist_ok=True)

    dataset_cfg = {
        'name': 'precompute',
        'hr_dir': hr_dir,
        'enabled': True,
        'scale': scale,
        'crop_size': crop_size,
        'augment': False,
        'degradation': degradation_cfg,
    }

    dataset = DatasetFactory.create(dataset_cfg)

    num_samples = min(max_samples or len(dataset), len(dataset))
    print(f"Pre-computing {num_samples} LR/HR pairs")
    print(f"  HR dir: {hr_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Scale: {scale}x, Crop: {crop_size}")

    for idx in tqdm(range(num_samples)):
        try:
            sample = dataset[idx]
            lr_tensor = sample['lr']
            hr_tensor = sample['hr']

            lr_np = (lr_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            hr_np = (hr_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            Image.fromarray(lr_np).save(lr_dir / f'{idx:06d}.png')
            Image.fromarray(hr_np).save(hr_dir_out / f'{idx:06d}.png')
        except Exception as e:
            print(f"\n  [WARNING] Failed to process sample {idx}: {e}")
            continue

    meta = {
        'num_samples': num_samples,
        'scale': scale,
        'crop_size': crop_size,
        'degradation': degradation_cfg,
    }
    with open(output_path / 'meta.yaml', 'w') as f:
        yaml.dump(meta, f)

    print(f"\n[Done] Saved {num_samples} pairs to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Pre-compute LR/HR degradation pairs')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output', type=str, default='data/precomputed', help='Output directory')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum samples to precompute')
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get('data', {})
    degradation_cfg = data_cfg.get('degradation', {})
    scale = data_cfg.get('scale', 4)
    crop_size = data_cfg.get('crop_size', 128)

    datasets = data_cfg.get('datasets', [])
    if not datasets:
        print("[ERROR] No datasets found in config")
        return

    hr_dir = datasets[0].get('hr_dir', 'data/anime_hr')
    if not os.path.exists(hr_dir):
        print(f"[ERROR] HR directory not found: {hr_dir}")
        return

    precompute_pairs(
        hr_dir=hr_dir,
        output_dir=args.output,
        scale=scale,
        crop_size=crop_size,
        degradation_cfg=degradation_cfg,
        max_samples=args.max_samples,
    )


if __name__ == '__main__':
    main()
