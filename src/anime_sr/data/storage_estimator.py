"""
Storage estimation tool for precomputed datasets.
Calculates worst-case and best-case storage requirements before precomputing.
"""
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np


def get_image_dimensions(img_path: Path) -> Tuple[int, int]:
    """Get image dimensions without loading full image."""
    try:
        from PIL import Image
        with Image.open(img_path) as img:
            return img.size  # (width, height)
    except Exception:
        return 0, 0


def estimate_single_image_size(width: int, height: int, channels: int = 3, 
                               format: str = "png") -> Dict[str, float]:
    """
    Estimate storage size for a single HR+LR image pair.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels  
        channels: Number of channels (default 3 for RGB)
        format: Storage format ("png" or "jpeg")
    
    Returns:
        Dict with size estimates in MB for HR, LR, and total
    """
    # Raw bytes for float32 (4 bytes per channel)
    raw_bytes_per_pixel = channels * 4
    
    # HR size (raw)
    hr_raw_bytes = width * height * raw_bytes_per_pixel
    
    # LR size (at 4x downscale)
    lr_width = width // 4
    lr_height = height // 4
    lr_raw_bytes = lr_width * lr_height * raw_bytes_per_pixel
    
    # Compression ratios (approximate)
    # PNG: ~3-5x compression for typical images
    # JPEG at 90%: ~10-15x compression
    if format.lower() == "png":
        hr_compressed = hr_raw_bytes / 3.5  # PNG typically 3-5x smaller
        lr_compressed = lr_raw_bytes / 3.5
    else:  # jpeg
        hr_compressed = hr_raw_bytes / 12.0  # JPEG at 90% quality
        lr_compressed = lr_raw_bytes / 12.0
    
    return {
        'hr_mb': hr_compressed / (1024 * 1024),
        'lr_mb': lr_compressed / (1024 * 1024),
        'total_mb': (hr_compressed + lr_compressed) / (1024 * 1024)
    }


def estimate_dataset_storage(
    hr_dir: str,
    crop_size: int = 128,
    scale: int = 4,
    format: str = "png",
    sample_ratio: float = 1.0,
    sample_images: int = 30,
    seed: Optional[int] = None
) -> Dict:
    """
    Estimate storage requirements for a dataset.

    Args:
        hr_dir: Path to HR images directory
        crop_size: Training crop size (will use this for storage calc)
        scale: Upscale factor (4)
        format: Storage format ("png" or "jpeg")
        sample_ratio: Ratio of images to use (0.0-1.0)
        sample_images: Number of images to sample for dimension analysis
        seed: Optional integer seed for deterministic image sampling. If None,
            a fresh random sample is drawn. Pass an int for reproducible
            estimates across runs (Issue #13: np.random reproducibility).

    Returns:
        Dict with storage estimates and breakdown
    """
    hr_path = Path(hr_dir)

    if not hr_path.exists():
        return {'error': f"Directory not found: {hr_dir}"}

    # Get all images
    hr_images = sorted(hr_path.glob("*.png")) + \
                sorted(hr_path.glob("*.jpg")) + \
                sorted(hr_path.glob("*.jpeg"))

    if len(hr_images) == 0:
        return {'error': f"No images found in {hr_dir}"}

    total_images = len(hr_images)
    images_to_use = int(total_images * sample_ratio) if sample_ratio < 1.0 else total_images

    # Sample images to get dimension distribution. Use a Generator so that
    # `torch.manual_seed` and explicit `seed` give reproducible samples
    # (AGENTS.md: np.random in nn.Module forward breaks reproducibility;
    # same principle applies here for any caller that wants determinism).
    sample_size = min(sample_images, total_images)
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(total_images, sample_size, replace=False)
    
    dimensions = []
    for idx in sample_indices:
        w, h = get_image_dimensions(hr_images[idx])
        if w > 0 and h > 0:
            dimensions.append((w, h))
    
    if not dimensions:
        return {'error': "Could not read image dimensions"}
    
    # Calculate per-dimension storage
    # For crop-based storage, we use crop_size as the reference
    # But actual storage might be at original resolution or cropped
    avg_width = np.mean([d[0] for d in dimensions])
    avg_height = np.mean([d[1] for d in dimensions])
    
    # Storage calculation using crop_size as reference
    # For precomputed pairs, we store full images (not crops) at original resolution
    # but crop during training
    sizes = []
    for w, h in dimensions:
        # Use actual image dimensions for storage
        size_info = estimate_single_image_size(w, h, format=format)
        sizes.append(size_info['total_mb'])
    
    # Calculate statistics
    avg_size_per_image = np.mean(sizes)
    worst_case_size = np.max(sizes)  # Largest image
    best_case_size = np.min(sizes)   # Smallest image
    
    # Total estimate
    total_storage_mb = avg_size_per_image * images_to_use
    worst_case_total_mb = worst_case_size * images_to_use
    best_case_total_mb = best_case_size * images_to_use
    
    return {
        'hr_dir': str(hr_path),
        'total_images': total_images,
        'images_to_use': images_to_use,
        'sample_ratio': sample_ratio,
        'avg_image_dims': f"{avg_width:.0f}x{avg_height:.0f}",
        'crop_size': crop_size,
        'format': format,
        'avg_size_per_image_mb': avg_size_per_image,
        'worst_case_per_image_mb': worst_case_size,
        'best_case_per_image_mb': best_case_size,
        'total_storage_mb': total_storage_mb,
        'total_storage_gb': total_storage_mb / 1024,
        'worst_case_total_gb': worst_case_total_mb / 1024,
        'best_case_total_gb': best_case_total_mb / 1024,
        'per_dataset_breakdown': {
            str(hr_path.name): {
                'images': images_to_use,
                'estimated_gb': total_storage_mb / 1024
            }
        }
    }


def estimate_multiple_datasets(
    datasets: List[Dict],
    crop_size: int,
    scale: int = 4,
    format: str = "png",
    include_validation: bool = False,
    seed: Optional[int] = None
) -> Dict:
    """
    Estimate storage for multiple datasets.

    Args:
        datasets: List of dicts with 'hr_dir', 'sample_ratio', 'is_validation'
        crop_size: Training crop size
        scale: Upscale factor
        format: Storage format
        include_validation: Whether to include validation datasets
        seed: Optional integer seed forwarded to estimate_dataset_storage for
            deterministic sampling across all datasets in the run.

    Returns:
        Combined storage estimate with breakdown per dataset
    """
    total_mb = 0
    worst_case_mb = 0
    breakdown = {}

    for ds in datasets:
        # Skip validation unless requested
        if ds.get('is_validation', False) and not include_validation:
            continue

        estimate = estimate_dataset_storage(
            hr_dir=ds['hr_dir'],
            crop_size=crop_size,
            scale=scale,
            format=format,
            sample_ratio=ds.get('sample_ratio', 1.0),
            sample_images=30,
            seed=seed,
        )
        
        if 'error' not in estimate:
            dataset_name = Path(ds['hr_dir']).name
            breakdown[dataset_name] = {
                'images': estimate['images_to_use'],
                'estimated_gb': estimate['total_storage_gb'],
                'worst_case_gb': estimate['worst_case_total_gb']
            }
            total_mb += estimate['total_storage_mb']
            worst_case_mb += estimate['worst_case_total_gb'] * 1024
    
    return {
        'total_storage_gb': total_mb / 1024,
        'worst_case_total_gb': worst_case_mb / 1024,
        'include_validation': include_validation,
        'per_dataset_breakdown': breakdown,
        'sufficient_space': True  # Will be updated by check_against_available
    }


def check_against_available_space(
    estimate: Dict,
    available_gb: float
) -> Dict:
    """
    Check if estimated storage fits within available space.
    
    Args:
        estimate: Storage estimate from estimate_dataset_storage or estimate_multiple_datasets
        available_gb: Available disk space in GB
    
    Returns:
        Updated estimate with space check results
    """
    worst_case_gb = estimate.get('worst_case_total_gb', estimate.get('total_storage_gb', 0))
    
    result = estimate.copy()
    result['available_space_gb'] = available_gb
    result['sufficient_space'] = worst_case_gb <= available_gb
    
    if worst_case_gb > available_gb:
        result['space_shortage_gb'] = worst_case_gb - available_gb
        result['recommendation'] = (
            f"Need {worst_case_gb:.1f}GB but only {available_gb:.1f}GB available. "
            f"Consider: (1) Reducing sample_ratio, (2) Using smaller crop_size, "
            f"(3) Using JPEG format instead of PNG"
        )
    else:
        result['space_remaining_gb'] = available_gb - worst_case_gb
        result['recommendation'] = f"Sufficient space. {available_gb - worst_case_gb:.1f}GB will remain."
    
    return result


def print_storage_estimate(estimate: Dict, verbose: bool = True) -> str:
    """Format and print storage estimate nicely."""
    if 'error' in estimate:
        return f"Error: {estimate['error']}"
    
    lines = []
    lines.append("=" * 60)
    lines.append("STORAGE ESTIMATION REPORT")
    lines.append("=" * 60)
    
    if 'per_dataset_breakdown' in estimate:
        # Multiple datasets
        lines.append(f"\nDatasets:")
        for name, data in estimate['per_dataset_breakdown'].items():
            lines.append(f"  - {name}: {data['images']} images, ~{data['estimated_gb']:.2f}GB")
        
        lines.append(f"\nTotal Storage Required:")
        lines.append(f"  Estimated: {estimate['total_storage_gb']:.2f}GB")
        lines.append(f"  Worst Case: {estimate['worst_case_total_gb']:.2f}GB")
        best_case = estimate.get('best_case_total_gb')
        if best_case is not None and not isinstance(best_case, str):
            lines.append(f"  Best Case: {best_case:.2f}GB")
        else:
            lines.append(f"  Best Case: N/A")
        lines.append(f"  Available: {estimate.get('available_space_gb', 'N/A')}GB")
        
        if 'sufficient_space' in estimate:
            if estimate['sufficient_space']:
                lines.append(f"  [OK] Sufficient space ({estimate.get('space_remaining_gb', 0):.2f}GB remaining)")
            else:
                lines.append(f"  [WARNING] Insufficient space ({estimate.get('space_shortage_gb', 0):.2f}GB needed)")
    else:
        # Single dataset
        lines.append(f"\nDataset: {estimate.get('hr_dir', 'Unknown')}")
        lines.append(f"Total Images: {estimate.get('total_images', 0)}")
        lines.append(f"Images to Use: {estimate.get('images_to_use', 0)} (ratio: {estimate.get('sample_ratio', 1.0):.2f})")
        lines.append(f"Avg Image Size: {estimate.get('avg_image_dims', 'N/A')}")
        lines.append(f"Crop Size: {estimate.get('crop_size', 0)}")
        lines.append(f"Format: {estimate.get('format', 'png')}")
        lines.append(f"\nStorage per Image:")
        lines.append(f"  Average: {estimate.get('avg_size_per_image_mb', 0):.3f}MB")
        lines.append(f"  Worst Case: {estimate.get('worst_case_per_image_mb', 0):.3f}MB")
        lines.append(f"  Best Case: {estimate.get('best_case_per_image_mb', 0):.3f}MB")
        lines.append(f"\nTotal Storage:")
        lines.append(f"  Estimated: {estimate.get('total_storage_gb', 0):.2f}GB")
        lines.append(f"  Worst Case: {estimate.get('worst_case_total_gb', 0):.2f}GB")
        lines.append(f"  Best Case: {estimate.get('best_case_total_gb', 0):.2f}GB")
        
        if 'available_space_gb' in estimate:
            lines.append(f"\nAvailable Space: {estimate['available_space_gb']:.2f}GB")
            if estimate['sufficient_space']:
                lines.append(f"[OK] {estimate.get('space_remaining_gb', 0):.2f}GB remaining")
            else:
                lines.append(f"[WARNING] Need {estimate.get('space_shortage_gb', 0):.2f}GB more")
    
    if 'recommendation' in estimate:
        lines.append(f"\nRecommendation: {estimate['recommendation']}")
    
    lines.append("=" * 60)
    
    output = "\n".join(lines)
    if verbose:
        print(output)
    return output


def create_default_dataset_configs() -> List[Dict]:
    """Create default dataset configurations for anime project."""
    return [
        {
            'name': 'anime_video_frames',
            'hr_dir': 'data/anime_video_frames',
            'sample_ratio': 0.3,
            'is_validation': False,
            'enabled': True,
            'weight': 1.0
        },
        {
            'name': 'anime_hr',
            'hr_dir': 'data/anime_hr',
            'sample_ratio': 0.5,
            'is_validation': False,
            'enabled': True,
            'weight': 1.0
        },
        {
            'name': 'val_hr',
            'hr_dir': 'data/val_hr',
            'sample_ratio': 1.0,
            'is_validation': True,
            'enabled': False,  # Default: don't precompute validation
            'weight': 1.0
        },
        {
            'name': 'val_hr_1',
            'hr_dir': 'data/val_hr 1',
            'sample_ratio': 1.0,
            'is_validation': True,
            'enabled': False,  # Default: don't precompute validation
            'weight': 1.0
        },
    ]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Estimate storage for precomputed datasets")
    parser.add_argument('--hr_dir', type=str, help='HR images directory')
    parser.add_argument('--crop_size', type=int, default=128, help='Training crop size')
    parser.add_argument('--scale', type=int, default=4, help='Upscale factor')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'jpeg'], help='Storage format')
    parser.add_argument('--sample_ratio', type=float, default=1.0, help='Sample ratio (0.0-1.0)')
    parser.add_argument('--available_space', type=float, default=None, help='Available disk space in GB')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible image sampling (Issue #13)')
    parser.add_argument('--all_datasets', action='store_true', help='Estimate all known datasets')
    
    args = parser.parse_args()
    
    if args.all_datasets:
        datasets = create_default_dataset_configs()
        estimate = estimate_multiple_datasets(
            datasets, args.crop_size, args.scale, args.format,
            include_validation=False, seed=args.seed,
        )
    elif args.hr_dir:
        estimate = estimate_dataset_storage(
            args.hr_dir, args.crop_size, args.scale, args.format,
            args.sample_ratio, seed=args.seed,
        )
    else:
        print("Specify --hr_dir or --all_datasets")
        exit(1)
    
    if args.available_space:
        estimate = check_against_available_space(estimate, args.available_space)
    
    print_storage_estimate(estimate)