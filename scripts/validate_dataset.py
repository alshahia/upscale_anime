#!/usr/bin/env python
"""
Dataset validation script for anime super-resolution training.
Checks image quality, removes duplicates, and validates dataset structure.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def compute_image_hash(image_path: Path, hash_size: int = 16) -> str:
    """Compute perceptual hash (average hash) for duplicate detection."""
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale and resize
            img = img.convert('L').resize((hash_size, hash_size), Image.Resampling.LANCZOS)
            pixels = np.array(img)
            # Compute average
            avg = pixels.mean()
            # Create hash: 1 if pixel > avg, else 0
            diff = pixels > avg
            # Convert to hex string
            hash_str = ''.join(str(int(b)) for b in diff.flatten())
            return hex(int(hash_str, 2))[2:].zfill(hash_size * hash_size // 4)
    except Exception as e:
        return None


def compute_file_hash(image_path: Path) -> str:
    """Compute MD5 hash of file for exact duplicate detection."""
    try:
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None


def analyze_image(image_path: Path) -> Dict:
    """Analyze image quality and properties."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            mode = img.mode
            
            # Check resolution
            min_resolution = 512
            resolution_ok = width >= min_resolution and height >= min_resolution
            
            # Load for quality analysis
            img_array = np.array(img.convert('RGB'))
            
            # Check for blur (Laplacian variance)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = laplacian_var
            
            # Check for compression artifacts (estimate from DCT coefficients)
            compression_score = estimate_compression_quality(image_path)
            
            # Check color variance (reject flat/blank images)
            color_variance = np.var(img_array, axis=(0, 1)).mean()
            
            # Convert numpy types to Python native types for JSON serialization
            is_valid = bool(resolution_ok and blur_score > 50 and color_variance > 100)
            
            return {
                'path': str(image_path),
                'width': int(width),
                'height': int(height),
                'mode': mode,
                'resolution_ok': bool(resolution_ok),
                'blur_score': float(blur_score),
                'compression_score': float(compression_score),
                'color_variance': float(color_variance),
                'valid': is_valid,
                'error': None
            }
    except Exception as e:
        return {
            'path': str(image_path),
            'valid': False,
            'error': str(e)
        }


def estimate_compression_quality(image_path: Path) -> float:
    """Estimate JPEG compression quality (simplified)."""
    try:
        if image_path.suffix.lower() in ['.jpg', '.jpeg']:
            file_size = image_path.stat().st_size
            with Image.open(image_path) as img:
                pixels = img.size[0] * img.size[1]
                bpp = (file_size * 8) / pixels
                return min(100, bpp * 10)
        return 100.0
    except Exception:
        return 0.0


def find_duplicates(image_paths: List[Path], threshold: int = 5, use_perceptual: bool = True) -> List[Tuple[Path, Path]]:
    """Find duplicate images using perceptual or exact hashing."""
    duplicates = []
    hashes = {}
    
    print(f"Computing hashes for {len(image_paths)} images...")
    
    for path in tqdm(image_paths, desc="Hashing"):
        if use_perceptual:
            h = compute_image_hash(path)
        else:
            h = compute_file_hash(path)
        
        if h is None:
            continue
            
        if h in hashes:
            duplicates.append((path, hashes[h]))
        else:
            hashes[h] = path
    
    return duplicates


def validate_dataset(data_dir: str, remove_duplicates: bool = False, 
                    remove_invalid: bool = False, num_workers: int = 4) -> Dict:
    """Validate entire dataset."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    
    # Find all images
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.webp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(data_path.glob(f'*{ext}'))
        image_paths.extend(data_path.glob(f'*{ext.upper()}'))
    
    image_paths = sorted(set(image_paths))
    
    print(f"\n{'='*60}")
    print(f"Dataset Validation: {data_dir}")
    print(f"{'='*60}")
    print(f"Total images found: {len(image_paths)}")
    
    # Analyze all images
    print("\nAnalyzing images...")
    results = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(analyze_image, path): path for path in image_paths}
        
        for future in tqdm(as_completed(futures), total=len(image_paths), desc="Analyzing"):
            result = future.result()
            results.append(result)
    
    # Categorize results
    valid_images = [r for r in results if r.get('valid', False) and r.get('error') is None]
    invalid_images = [r for r in results if not r.get('valid', False) or r.get('error')]
    
    # Check for duplicates
    print("\nChecking for duplicates...")
    valid_paths = [Path(r['path']) for r in valid_images]
    duplicate_pairs = find_duplicates(valid_paths, use_perceptual=True)
    
    # Remove duplicates from valid list
    duplicate_paths = set()
    for dup, orig in duplicate_pairs:
        duplicate_paths.add(dup)
    
    final_valid = [r for r in valid_images if Path(r['path']) not in duplicate_paths]
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Validation Results")
    print(f"{'='*60}")
    print(f"Total images:        {len(image_paths)}")
    print(f"Valid images:        {len(valid_images)} ({len(valid_images)/len(image_paths)*100:.1f}%)")
    print(f"Invalid images:      {len(invalid_images)} ({len(invalid_images)/len(image_paths)*100:.1f}%)")
    print(f"Duplicates found:    {len(duplicate_pairs)}")
    print(f"Final valid count:   {len(final_valid)}")
    
    # Save report
    report_path = data_path / 'validation_report.json'
    with open(report_path, 'w') as f:
        json.dump({
            'summary': {
                'total': len(image_paths),
                'valid': len(valid_images),
                'invalid': len(invalid_images),
                'duplicates': len(duplicate_pairs),
                'final_valid': len(final_valid)
            },
            'valid_images': valid_images,
            'invalid_images': invalid_images,
            'duplicates': [[str(d), str(o)] for d, o in duplicate_pairs]
        }, f, indent=2)
    
    print(f"\nValidation report saved to: {report_path}")
    
    return {
        'total': len(image_paths),
        'valid': len(final_valid),
        'invalid': len(invalid_images),
        'duplicates': len(duplicate_pairs),
        'report_path': str(report_path)
    }


def main():
    parser = argparse.ArgumentParser(description='Validate anime SR dataset')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--remove-duplicates', action='store_true', help='Move duplicates to separate folder')
    parser.add_argument('--remove-invalid', action='store_true', help='Move invalid images to separate folder')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    validate_dataset(
        data_dir=args.data_dir,
        remove_duplicates=args.remove_duplicates,
        remove_invalid=args.remove_invalid,
        num_workers=args.num_workers
    )


if __name__ == '__main__':
    main()
