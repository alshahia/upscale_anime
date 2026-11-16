"""
Data utilities for validation, preparation, and debugging
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import shutil


def validate_dataset(
    hr_dir: str,
    expected_count: Optional[int] = None,
    check_corruption: bool = False,
    min_resolution: Tuple[int, int] = (64, 64),
) -> Dict:
    """
    Validate dataset and return statistics.
    
    Args:
        hr_dir: Directory with HR images
        expected_count: Expected number of images (optional)
        check_corruption: Whether to check for corrupted images
        min_resolution: Minimum [width, height] allowed
    
    Returns:
        Dictionary with validation results
    """
    hr_dir = Path(hr_dir)
    
    # Check directory exists
    if not hr_dir.exists():
        return {
            'valid': False,
            'error': f'Directory does not exist: {hr_dir}',
            'count': 0,
            'corrupted': [],
            'too_small': [],
        }
    
    # Find images
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    images = []
    for ext in image_extensions:
        images.extend(hr_dir.glob(f'*{ext}'))
        images.extend(hr_dir.glob(f'*{ext.upper()}'))
    
    images = sorted(list(set(images)))  # Remove duplicates
    
    if len(images) == 0:
        return {
            'valid': False,
            'error': f'No images found in {hr_dir}',
            'count': 0,
            'corrupted': [],
            'too_small': [],
        }
    
    # Check each image
    corrupted = []
    too_small = []
    resolutions = []
    
    for img_path in images:
        if check_corruption:
            img = cv2.imread(str(img_path))
            if img is None:
                corrupted.append(str(img_path))
                continue
            
            h, w = img.shape[:2]
            resolutions.append((w, h))
            
            if w < min_resolution[0] or h < min_resolution[1]:
                too_small.append((str(img_path), (w, h)))
    
    # Check expected count
    count_ok = True
    if expected_count and len(images) != expected_count:
        count_ok = False
    
    valid = len(corrupted) == 0 and len(too_small) == 0 and count_ok
    
    result = {
        'valid': valid,
        'count': len(images),
        'corrupted': corrupted,
        'too_small': too_small,
        'resolutions': resolutions if check_corruption else None,
    }
    
    if not valid:
        errors = []
        if corrupted:
            errors.append(f'{len(corrupted)} corrupted images')
        if too_small:
            errors.append(f'{len(too_small)} images below minimum resolution')
        if not count_ok:
            errors.append(f'Expected {expected_count} images, found {len(images)}')
        result['error'] = '; '.join(errors)
    
    return result


def create_dummy_dataset(
    output_dir: str,
    num_images: int = 10,
    image_size: Tuple[int, int] = (256, 256),
    pattern: str = 'gradient',
) -> List[Path]:
    """
    Create dummy dataset for testing.
    
    Args:
        output_dir: Where to save images
        num_images: Number of images to create
        image_size: [width, height] of images
        pattern: 'gradient', 'noise', 'checkerboard'
    
    Returns:
        List of created image paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    created = []
    
    for i in range(num_images):
        w, h = image_size
        
        if pattern == 'gradient':
            # Create a gradient image
            img = np.zeros((h, w, 3), dtype=np.uint8)
            for y in range(h):
                for x in range(w):
                    img[y, x] = [
                        int(255 * x / w),
                        int(255 * y / h),
                        int(255 * (x + y) / (w + h))
                    ]
        
        elif pattern == 'noise':
            # Random noise
            img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        
        elif pattern == 'checkerboard':
            # Checkerboard pattern
            img = np.zeros((h, w, 3), dtype=np.uint8)
            square_size = 32
            for y in range(0, h, square_size):
                for x in range(0, w, square_size):
                    color = 255 if ((x // square_size + y // square_size) % 2 == 0) else 0
                    img[y:y+square_size, x:x+square_size] = color
        
        else:
            # Solid color
            color = [(i * 50) % 255, (i * 80) % 255, (i * 110) % 255]
            img = np.full((h, w, 3), color, dtype=np.uint8)
        
        # Save
        output_path = output_dir / f'dummy_{i:04d}.png'
        cv2.imwrite(str(output_path), img)
        created.append(output_path)
    
    return created


def check_dataset_balance(
    dataset_dirs: List[str],
) -> Dict:
    """
    Check balance across multiple datasets.
    
    Args:
        dataset_dirs: List of dataset directories
    
    Returns:
        Dictionary with balance statistics
    """
    stats = {}
    total = 0
    
    for d in dataset_dirs:
        dir_path = Path(d)
        if not dir_path.exists():
            stats[d] = {'exists': False, 'count': 0}
            continue
        
        # Count images
        count = len(list(dir_path.glob('*.png'))) + len(list(dir_path.glob('*.jpg')))
        stats[d] = {'exists': True, 'count': count}
        total += count
    
    # Calculate weights
    if total > 0:
        for d in stats:
            if stats[d]['exists']:
                stats[d]['weight'] = stats[d]['count'] / total
    
    stats['total'] = total
    return stats


def auto_crop_images(
    input_dir: str,
    output_dir: str,
    crop_size: int = 512,
    min_file_size: int = 100 * 1024,  # 100KB
) -> int:
    """
    Auto-crop large images into patches for training.
    
    Args:
        input_dir: Directory with source images
        output_dir: Where to save cropped patches
        crop_size: Size of square crops
        min_file_size: Minimum file size in bytes (skip small files)
    
    Returns:
        Number of crops created
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images = list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg'))
    
    total_crops = 0
    
    for img_path in images:
        # Skip small files
        if img_path.stat().st_size < min_file_size:
            continue
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # Create crops
        crop_idx = 0
        for y in range(0, h - crop_size + 1, crop_size):
            for x in range(0, w - crop_size + 1, crop_size):
                crop = img[y:y+crop_size, x:x+crop_size]
                
                # Save
                crop_path = output_dir / f"{img_path.stem}_crop{crop_idx:04d}.png"
                cv2.imwrite(str(crop_path), crop)
                
                crop_idx += 1
                total_crops += 1
    
    return total_crops


def get_dataset_info(hr_dir: str, scale: int = 4) -> Dict:
    """
    Get comprehensive dataset information.
    
    Args:
        hr_dir: HR images directory
        scale: Downsampling factor
    
    Returns:
        Dictionary with dataset info
    """
    hr_dir = Path(hr_dir)
    
    if not hr_dir.exists():
        return {
            'exists': False,
            'error': f'Directory not found: {hr_dir}',
        }
    
    # Get images
    images = list(hr_dir.glob('*.png')) + list(hr_dir.glob('*.jpg'))
    
    if not images:
        return {
            'exists': True,
            'error': 'No images found',
            'count': 0,
        }
    
    # Sample a few images for resolution stats
    resolutions = []
    for img_path in images[:10]:  # Check first 10
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            resolutions.append((w, h))
    
    # Calculate average
    if resolutions:
        avg_w = sum(r[0] for r in resolutions) / len(resolutions)
        avg_h = sum(r[1] for r in resolutions) / len(resolutions)
    else:
        avg_w = avg_h = 0
    
    return {
        'exists': True,
        'path': str(hr_dir.absolute()),
        'count': len(images),
        'avg_resolution': (int(avg_w), int(avg_h)),
        'total_size_gb': sum(f.stat().st_size for f in images) / (1024**3),
        'estimated_lr_resolution': (int(avg_w // scale), int(avg_h // scale)),
        'estimated_patches_128': len(images) * (avg_w // 128) * (avg_h // 128) if avg_w > 128 and avg_h > 128 else 0,
    }
