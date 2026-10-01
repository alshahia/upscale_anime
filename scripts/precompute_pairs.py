"""
Precompute LR/HR pairs for fast training.

This script pre-generates low-resolution images from high-resolution sources
with configurable degradation settings. Supports size estimation before
precomputing and resume capability.

Usage:
    # Estimate storage only
    python scripts/precompute_pairs.py --estimate-only --crop_size 128

    # Precompute with confirmation
    python scripts/precompute_pairs.py --hr_dir data/anime_hr_1 --crop_size 128 --scale 4

    # Precompute multiple datasets
    python scripts/precompute_pairs.py --all_datasets --crop_size 128 --workers 8

    # Resume interrupted precomputation
    python scripts/precompute_pairs.py --hr_dir data/anime_hr_1 --resume
"""
import os
import sys
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np
import cv2
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from anime_sr.data.storage_estimator import (
    estimate_dataset_storage, estimate_multiple_datasets,
    check_against_available_space, print_storage_estimate,
    create_default_dataset_configs
)
from anime_sr.data.compression_modules import CompressionPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute LR/HR pairs for training",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input/Output
    parser.add_argument('--hr_dir', type=str, default=None,
                       help='HR images directory (for single dataset)')
    parser.add_argument('--output_dir', type=str, default='data/precomputed_4x',
                       help='Output directory for precomputed pairs')
    
    # Preprocessing options
    parser.add_argument('--crop_size', type=int, default=128,
                       help='Training crop size (used for storage estimation)')
    parser.add_argument('--scale', type=int, default=4,
                       help='Upscale factor')
    parser.add_argument('--format', type=str, default='png',
                       choices=['png', 'jpeg'],
                       help='Output format for precomputed pairs')
    parser.add_argument('--jpeg_quality', type=int, default=90,
                       help='JPEG quality if using JPEG format')
    
    # Dataset options
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                       help='Ratio of images to precompute (0.0-1.0)')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to precompute')
    parser.add_argument('--all_datasets', action='store_true',
                       help='Precompute all configured datasets')
    parser.add_argument('--include_validation', action='store_true',
                       help='Include validation datasets')
    
    # Degradation options
    parser.add_argument('--degradation_mode', type=str, default='anime_heavy',
                       choices=['light', 'medium', 'heavy', 'anime', 'anime_heavy'],
                       help='Degradation preset to use')
    parser.add_argument('--degrade_before_crop', action='store_true', default=True,
                       help='Apply degradation on full image before cropping')
    parser.add_argument('--shuffled_resize', action='store_true', default=True,
                       help='Randomize degradation operation order')
    parser.add_argument('--two_stage_compression', action='store_true', default=True,
                       help='Use two-stage compression')
    
    # Storage estimation
    parser.add_argument('--estimate-only', '--estimate_only', dest='estimate_only',
                        action='store_true',
                        help='Only estimate storage, do not precompute')
    parser.add_argument('--available-space', '--available_space', dest='available_space',
                        type=float, default=None,
                        help='Available disk space in GB for estimation')
    
    # Execution
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--resume', action='store_true',
                       help='Resume interrupted precomputation')
    parser.add_argument('--force', action='store_true',
                       help='Overwrite existing files')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    return parser.parse_args()


def get_degradation_config(mode: str) -> Dict:
    """Get degradation configuration for specified mode."""
    presets = {
        'light': {
            'blur_prob': 0.3, 'blur_sigma': [0.1, 1.5],
            'noise_prob': 0.2, 'noise_sigma': [0, 15],
            'jpeg_prob': 0.2, 'jpeg_quality': [80, 90, 95],
            'compression_prob': 0.3, 'second_order_prob': 0.1
        },
        'medium': {
            'blur_prob': 0.7, 'blur_sigma': [0.1, 2.5],
            'noise_prob': 0.5, 'noise_sigma': [0, 25],
            'jpeg_prob': 0.5, 'jpeg_quality': [70, 80, 90],
            'compression_prob': 0.5, 'second_order_prob': 0.3
        },
        'heavy': {
            'blur_prob': 0.9, 'blur_sigma': [0.5, 4.0],
            'noise_prob': 0.8, 'noise_sigma': [5, 50],
            'jpeg_prob': 0.7, 'jpeg_quality': [60, 70, 80],
            'compression_prob': 0.7, 'second_order_prob': 0.4
        },
        'anime': {
            'blur_prob': 0.7, 'blur_sigma': [0.1, 3.0],
            'noise_prob': 0.5, 'noise_sigma': [0, 30],
            'jpeg_prob': 0.6, 'jpeg_quality': [60, 70, 80, 90],
            'compression_prob': 0.7, 'second_order_prob': 0.3,
            'anime_degradation': True, 'color_quantization': True
        },
        'anime_heavy': {
            'blur_prob': 0.8, 'blur_sigma': [0.1, 3.0],
            'noise_prob': 0.6, 'noise_sigma': [0, 30],
            'jpeg_prob': 0.6, 'jpeg_quality': [50, 60, 70, 80, 90],
            'compression_prob': 0.8, 'second_order_prob': 0.4,
            'anime_degradation': True, 'color_quantization': True, 'banding_simulation': True
        }
    }
    return presets.get(mode, presets['anime_heavy'])


def apply_degradation(img: np.ndarray, cfg: Dict, scale: int) -> np.ndarray:
    """Apply degradation to create LR image."""
    h, w = img.shape[:2]
    
    # First-order degradation
    if random.random() < cfg.get('blur_prob', 0.7):
        kernel_sizes = cfg.get('blur_kernel_size', [7, 9, 11])
        kernel_size = random.choice(kernel_sizes)
        sigma_range = cfg.get('blur_sigma', [0.1, 3.0])
        sigma = random.uniform(sigma_range[0], sigma_range[1])
        img = apply_gaussian_blur(img, kernel_size, sigma)
    
    if random.random() < cfg.get('noise_prob', 0.5):
        sigma_range = cfg.get('noise_sigma', [0, 25])
        sigma = random.uniform(sigma_range[0], sigma_range[1])
        if sigma > 0:
            noise = np.random.randn(*img.shape).astype(np.float32) * sigma
            img = img.astype(np.float32) + noise
            img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Random resize
    if random.random() < 0.5:
        resize_range = cfg.get('resize_range', [0.15, 1.5])
        factor = random.uniform(resize_range[0], resize_range[1])
        new_h, new_w = int(h * factor), int(w * factor)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # JPEG compression
    if random.random() < cfg.get('jpeg_prob', 0.5):
        quality = random.choice(cfg.get('jpeg_quality', [70, 80, 90]))
        img = apply_jpeg_compression(img, quality)
    
    # Resize to LR size
    lr_h, lr_w = h // scale, w // scale
    img = cv2.resize(img, (lr_w, lr_h), interpolation=cv2.INTER_LINEAR)
    
    # Second-order degradation
    if random.random() < cfg.get('second_order_prob', 0.3):
        kernel_size = random.choice([3, 5, 7])
        sigma = random.uniform(0.1, 1.0)
        img = apply_gaussian_blur(img, kernel_size, sigma)
    
    if random.random() < cfg.get('second_order_prob', 0.3):
        sigma = random.uniform(0, 10)
        if sigma > 0:
            noise = np.random.randn(*img.shape).astype(np.float32) * sigma
            img = img.astype(np.float32) + noise
            img = np.clip(img, 0, 255).astype(np.uint8)
    
    if random.random() < cfg.get('second_order_prob', 0.3):
        quality = random.choice([70, 80, 90])
        img = apply_jpeg_compression(img, quality)
    
    return img


def apply_gaussian_blur(img: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """Apply Gaussian blur."""
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = kernel * kernel.T
    return cv2.filter2D(img, -1, kernel)


def apply_jpeg_compression(img: np.ndarray, quality: int) -> np.ndarray:
    """Apply JPEG compression."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encimg, 1)


def precompute_single_dataset(
    hr_dir: str,
    output_dir: str,
    crop_size: int,
    scale: int,
    format: str = 'png',
    jpeg_quality: int = 90,
    sample_ratio: float = 1.0,
    max_images: Optional[int] = None,
    degradation_mode: str = 'anime_heavy',
    resume: bool = False,
    force: bool = False,
    workers: int = 4,
    verbose: bool = False
) -> Dict:
    """
    Precompute LR/HR pairs for a single dataset.
    
    Returns:
        Dict with 'success', 'total', 'skipped', 'failed' counts
    """
    hr_path = Path(hr_dir)
    out_path = Path(output_dir)
    
    if not hr_path.exists():
        return {'success': False, 'error': f"HR directory not found: {hr_dir}"}
    
    # Create output directories
    lr_dir = out_path / 'lr'
    hr_dir_out = out_path / 'hr'
    lr_dir.mkdir(parents=True, exist_ok=True)
    hr_dir_out.mkdir(parents=True, exist_ok=True)
    
    # Get HR images
    hr_images = sorted(hr_path.glob("*.png")) + \
                sorted(hr_path.glob("*.jpg")) + \
                sorted(hr_path.glob("*.jpeg"))
    
    if len(hr_images) == 0:
        return {'success': False, 'error': f"No images found in {hr_dir}"}
    
    # Apply sampling
    num_to_use = len(hr_images)
    if sample_ratio < 1.0:
        num_to_use = int(len(hr_images) * sample_ratio)
    if max_images is not None:
        num_to_use = min(num_to_use, max_images)
    
    hr_images = hr_images[:num_to_use]
    
    # Get degradation config
    deg_cfg = get_degradation_config(degradation_mode)
    
    # Create compression pipeline for post-processing
    compression_pipe = None
    if format == 'jpeg':
        compression_pipe = CompressionPipeline(
            compression_types=['jpeg'],
            quality_ranges=[(jpeg_quality, jpeg_quality)]
        )
    
    # Process images
    success_count = 0
    skipped_count = 0
    failed_count = 0
    
    print(f"\nPrecomputing {len(hr_images)} images from {hr_path.name}")
    print(f"Output: {out_path}")
    print(f"Degradation: {degradation_mode}")
    
    for idx, hr_img_path in enumerate(tqdm(hr_images, desc="Precomputing")):
        lr_name = hr_img_path.stem + ('.jpg' if format == 'jpeg' else '.png')
        lr_out_path = lr_dir / lr_name
        hr_out_path = hr_dir_out / lr_name
        
        # Check if already exists (resume mode)
        if lr_out_path.exists() and hr_out_path.exists() and not force:
            if resume:
                skipped_count += 1
                continue
            else:
                # Check if file is valid
                if lr_out_path.stat().st_size > 0 and hr_out_path.stat().st_size > 0:
                    skipped_count += 1
                    continue
        
        try:
            # Load HR image
            hr_img = cv2.imread(str(hr_img_path), cv2.IMREAD_COLOR)
            if hr_img is None:
                failed_count += 1
                continue
            
            # Apply degradation to create LR
            lr_img = apply_degradation(hr_img, deg_cfg, scale)
            
            # Save LR
            if format == 'jpeg':
                cv2.imwrite(str(lr_out_path), lr_img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            else:
                cv2.imwrite(str(lr_out_path), lr_img)
            
            # Save HR (as PNG for quality preservation)
            cv2.imwrite(str(hr_out_path), hr_img)
            
            success_count += 1
            
        except Exception as e:
            failed_count += 1
            if verbose:
                print(f"Error processing {hr_img_path.name}: {e}")
    
    return {
        'success': True,
        'total': len(hr_images),
        'completed': success_count,
        'skipped': skipped_count,
        'failed': failed_count
    }


def precompute_all_datasets(
    crop_size: int,
    scale: int,
    format: str = 'png',
    jpeg_quality: int = 90,
    degradation_mode: str = 'anime_heavy',
    include_validation: bool = False,
    resume: bool = False,
    force: bool = False,
    workers: int = 4,
    verbose: bool = False
) -> List[Dict]:
    """Precompute all configured datasets."""
    datasets = create_default_dataset_configs()
    
    # Filter enabled datasets
    datasets_to_process = [
        ds for ds in datasets
        if ds.get('enabled', True) and not ds.get('is_validation', False)
    ]
    
    if include_validation:
        datasets_to_process.extend([
            ds for ds in datasets
            if ds.get('is_validation', False) and ds.get('enabled', True)
        ])
    
    results = []
    for ds in datasets_to_process:
        hr_dir = ds['hr_dir']
        sample_ratio = ds.get('sample_ratio', 1.0)
        
        # Create output dir for this dataset
        output_dir = f"data/precomputed_4x_{Path(hr_dir).name}"
        
        result = precompute_single_dataset(
            hr_dir=hr_dir,
            output_dir=output_dir,
            crop_size=crop_size,
            scale=scale,
            format=format,
            jpeg_quality=jpeg_quality,
            sample_ratio=sample_ratio,
            degradation_mode=degradation_mode,
            resume=resume,
            force=force,
            workers=workers,
            verbose=verbose
        )
        
        result['dataset'] = Path(hr_dir).name
        results.append(result)
    
    return results


def save_metadata(output_dir: str, config: Dict, degradation_mode: str):
    """Save metadata about the precomputed dataset."""
    import yaml
    
    meta = {
        'scale': config.get('scale', 4),
        'crop_size': config.get('crop_size', 128),
        'format': config.get('format', 'png'),
        'degradation_mode': degradation_mode,
        'degradation_config': get_degradation_config(degradation_mode),
    }
    
    meta_path = Path(output_dir) / 'meta.yaml'
    with open(meta_path, 'w') as f:
        yaml.dump(meta, f)


def main():
    args = parse_args()
    
    print("=" * 60)
    print("PRECOMPUTE LR/HR PAIRS")
    print("=" * 60)
    
    # Estimate storage if requested
    if args.estimate_only:
        print("\n[ESTIMATE MODE] Storage estimation only\n")
        
        if args.all_datasets:
            datasets = create_default_dataset_configs()
            estimate = estimate_multiple_datasets(
                datasets,
                crop_size=args.crop_size,
                scale=args.scale,
                format=args.format,
                include_validation=args.include_validation
            )
            
            if args.available_space:
                estimate = check_against_available_space(estimate, args.available_space)
            
            print_storage_estimate(estimate)
            
            # Print recommendation
            if estimate.get('sufficient_space', False):
                print("\n[SUGGESTED COMMAND]")
                print(f"  python scripts/precompute_pairs.py --all_datasets --crop_size {args.crop_size} --scale {args.scale}")
            else:
                print("\n[SUGGESTED ACTIONS]")
                print(f"  1. Reduce sample_ratio in dataset configs")
                print(f"  2. Use smaller crop_size (e.g., --crop_size 96)")
                print(f"  3. Use JPEG format instead of PNG (--format jpeg)")
        else:
            if not args.hr_dir:
                print("Error: --hr_dir required for single dataset estimation")
                return
            
            estimate = estimate_dataset_storage(
                hr_dir=args.hr_dir,
                crop_size=args.crop_size,
                scale=args.scale,
                format=args.format,
                sample_ratio=args.sample_ratio
            )
            
            if args.available_space:
                estimate = check_against_available_space(estimate, args.available_space)
            
            print_storage_estimate(estimate)
        
        print("\n" + "=" * 60)
        return
    
    # Run precomputation
    if args.all_datasets:
        print("\n[PRECOMPUTE MODE] All datasets\n")
        
        results = precompute_all_datasets(
            crop_size=args.crop_size,
            scale=args.scale,
            format=args.format,
            jpeg_quality=args.jpeg_quality,
            degradation_mode=args.degradation_mode,
            include_validation=args.include_validation,
            resume=args.resume,
            force=args.force,
            workers=args.workers,
            verbose=args.verbose
        )
        
        print("\n" + "=" * 60)
        print("PRECOMPUTE RESULTS")
        print("=" * 60)
        for result in results:
            ds_name = result.get('dataset', 'unknown')
            print(f"\n{ds_name}:")
            print(f"  Total: {result['total']}")
            print(f"  Completed: {result['completed']}")
            print(f"  Skipped: {result['skipped']}")
            print(f"  Failed: {result['failed']}")
            
            # Save metadata for each dataset
            if result['success']:
                output_dir = f"data/precomputed_4x_{ds_name}"
                save_metadata(output_dir, vars(args), args.degradation_mode)
    
    else:
        if not args.hr_dir:
            print("Error: --hr_dir required for single dataset precomputation")
            print("Or use --all_datasets to precompute all configured datasets")
            return
        
        result = precompute_single_dataset(
            hr_dir=args.hr_dir,
            output_dir=args.output_dir,
            crop_size=args.crop_size,
            scale=args.scale,
            format=args.format,
            jpeg_quality=args.jpeg_quality,
            sample_ratio=args.sample_ratio,
            max_images=args.max_images,
            degradation_mode=args.degradation_mode,
            resume=args.resume,
            force=args.force,
            workers=args.workers,
            verbose=args.verbose
        )
        
        print("\n" + "=" * 60)
        print("PRECOMPUTE RESULTS")
        print("=" * 60)
        print(f"\nTotal: {result['total']}")
        print(f"Completed: {result['completed']}")
        print(f"Skipped: {result['skipped']}")
        print(f"Failed: {result['failed']}")
        
        if result['success']:
            save_metadata(args.output_dir, vars(args), args.degradation_mode)
    
    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()