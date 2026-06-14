"""
Test Data Manager - Central utility for test data handling.

Provides fallback mechanisms to ensure tests always have data to work with,
whether using real images or auto-generated dummy data.
"""
import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
from PIL import Image
import torch

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
REAL_VAL_HR = PROJECT_ROOT / 'data' / 'val_hr'
REAL_TEST_HR = PROJECT_ROOT / 'data' / 'test_hr'
TEMP_DATA_DIR = PROJECT_ROOT / 'tests' / 'temp_test_data'


def ensure_test_data(
    min_images: int = 4,
    image_size: Tuple[int, int] = (256, 256),
    pattern: str = 'gradient',
    verbose: bool = True
) -> Path:
    """
    Ensure test data exists. Uses real data if available, otherwise creates dummy data.
    
    Args:
        min_images: Minimum number of images required
        image_size: Size for dummy images if created
        pattern: Pattern for dummy images ('gradient', 'noise', 'checkerboard', 'anime_style')
        verbose: Print status messages
        
    Returns:
        Path to directory containing test images
    """
    # Try real validation data first
    if REAL_VAL_HR.exists() and len(list(REAL_VAL_HR.glob('*.png'))) >= min_images:
        if verbose:
            print(f"[TestDataManager] Using real validation data: {REAL_VAL_HR}")
            print(f"  Images: {len(list(REAL_VAL_HR.glob('*.png')))}")
        return REAL_VAL_HR
    
    # Try real test data second
    if REAL_TEST_HR.exists() and len(list(REAL_TEST_HR.glob('*.png'))) >= min_images:
        if verbose:
            print(f"[TestDataManager] Using real test data: {REAL_TEST_HR}")
            print(f"  Images: {len(list(REAL_TEST_HR.glob('*.png')))}")
        return REAL_TEST_HR
    
    # Create temporary dummy data as fallback
    temp_dir = create_fallback_data(min_images, image_size, pattern, verbose)
    return temp_dir


def create_fallback_data(
    num_images: int = 5,
    image_size: Tuple[int, int] = (256, 256),
    pattern: str = 'gradient',
    verbose: bool = True,
    output_dir: Optional[Path] = None
) -> Path:
    """
    Create dummy test images when real data is not available.
    
    Args:
        num_images: Number of images to create
        image_size: (height, width) for images
        pattern: Type of pattern ('gradient', 'noise', 'checkerboard', 'anime_style', 'natural')
        verbose: Print status messages
        output_dir: Optional custom output directory (default: tests/temp_test_data/)
        
    Returns:
        Path to directory containing created images
    """
    output_dir = output_dir or TEMP_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"[TestDataManager] Creating {num_images} dummy images in: {output_dir}")
        print(f"  Size: {image_size}, Pattern: {pattern}")
    
    created_paths = []
    
    for i in range(num_images):
        # Create image based on pattern
        if pattern == 'gradient':
            img_array = create_gradient_image(image_size, seed=i)
        elif pattern == 'noise':
            img_array = create_noise_image(image_size, seed=i)
        elif pattern == 'checkerboard':
            img_array = create_checkerboard_image(image_size, squares=8)
        elif pattern == 'anime_style':
            img_array = create_anime_style_image(image_size, seed=i)
        elif pattern == 'natural':
            img_array = create_natural_image(image_size, seed=i)
        else:
            img_array = create_gradient_image(image_size, seed=i)
        
        # Save as PNG
        img_path = output_dir / f'test_img_{i:03d}_{pattern}.png'
        Image.fromarray(img_array).save(img_path)
        created_paths.append(img_path)
    
    if verbose:
        print(f"[TestDataManager] Created {len(created_paths)} images")
        for p in created_paths[:3]:
            print(f"  - {p.name}")
        if len(created_paths) > 3:
            print(f"  ... and {len(created_paths) - 3} more")
    
    return output_dir


def create_gradient_image(size: Tuple[int, int], seed: int = 0) -> np.ndarray:
    """Create smooth gradient image."""
    np.random.seed(seed)
    h, w = size
    
    # Create RGB channels with different gradients
    r = np.linspace(0, 255, w).reshape(1, w).repeat(h, axis=0)
    g = np.linspace(0, 255, h).reshape(h, 1).repeat(w, axis=1)
    b = np.ones((h, w)) * 128
    
    # Add some variation
    noise = np.random.randn(h, w) * 10
    r = np.clip(r + noise, 0, 255).astype(np.uint8)
    g = np.clip(g + noise, 0, 255).astype(np.uint8)
    b = np.clip(b + noise, 0, 255).astype(np.uint8)
    
    return np.stack([r, g, b], axis=2).astype(np.uint8)


def create_noise_image(size: Tuple[int, int], seed: int = 0) -> np.ndarray:
    """Create random noise texture image."""
    np.random.seed(seed)
    h, w = size
    
    # Perlin-like noise using multiple octaves
    img = np.zeros((h, w, 3), dtype=np.float32)
    
    for octave in range(4):
        scale = 2 ** octave
        octave_noise = np.random.rand(h // scale + 1, w // scale + 1, 3)
        
        # Simple upsampling
        from scipy.ndimage import zoom
        upsampled = zoom(octave_noise, (scale, scale, 1), order=1)
        upsampled = upsampled[:h, :w, :]
        
        img += upsampled * (0.5 ** octave)
    
    img = (img / img.max() * 255).astype(np.uint8)
    return img


def create_checkerboard_image(size: Tuple[int, int], squares: int = 8) -> np.ndarray:
    """Create checkerboard pattern for edge testing."""
    h, w = size
    
    # Create checkerboard
    checker = np.indices((h, w)).sum(axis=0) // (min(h, w) // squares) % 2
    
    # Convert to RGB
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[checker == 0] = [255, 255, 255]  # White
    img[checker == 1] = [0, 0, 0]  # Black
    
    return img


def create_anime_style_image(size: Tuple[int, int], seed: int = 0) -> np.ndarray:
    """Create simple anime-style image with flat colors and lines."""
    np.random.seed(seed)
    h, w = size
    
    # Start with flat color background
    bg_color = np.random.randint(200, 255, size=3)
    img = np.ones((h, w, 3), dtype=np.uint8) * bg_color
    
    # Add some "hair" (curved lines)
    num_hair_strands = 20
    for _ in range(num_hair_strands):
        x = np.random.randint(0, w)
        color = np.random.randint(50, 150, size=3)
        thickness = np.random.randint(2, 5)
        
        # Draw curved line
        for dy in range(h):
            dx = int(20 * np.sin(dy / h * 4 * np.pi + np.random.rand()))
            px = np.clip(x + dx, 0, w - 1)
            cv2 = try_import_cv2()
            if cv2:
                cv2.circle(img, (px, dy), thickness, color.tolist(), -1)
    
    # Add simple eyes (circles)
    eye_y = h // 2
    left_eye_x = w // 3
    right_eye_x = 2 * w // 3
    eye_radius = min(h, w) // 8
    
    cv2 = try_import_cv2()
    if cv2:
        # White of eye
        cv2.circle(img, (left_eye_x, eye_y), eye_radius, [240, 240, 240], -1)
        cv2.circle(img, (right_eye_x, eye_y), eye_radius, [240, 240, 240], -1)
        # Pupil
        cv2.circle(img, (left_eye_x, eye_y), eye_radius // 2, [50, 50, 50], -1)
        cv2.circle(img, (right_eye_x, eye_y), eye_radius // 2, [50, 50, 50], -1)
    
    return img


def create_natural_image(size: Tuple[int, int], seed: int = 0) -> np.ndarray:
    """Create natural-looking texture using Perlin noise."""
    np.random.seed(seed)
    h, w = size
    
    # Generate base noise
    noise = np.random.rand(h, w, 3)
    
    # Apply gaussian filter for smoothness
    from scipy.ndimage import gaussian_filter
    for i in range(3):
        noise[:, :, i] = gaussian_filter(noise[:, :, i], sigma=5)
    
    # Scale to 0-255
    img = (noise * 255).astype(np.uint8)
    return img


def try_import_cv2():
    """Try to import cv2, return None if not available."""
    try:
        import cv2
        return cv2
    except ImportError:
        return None


def get_val_hr_path() -> Path:
    """Get path to validation HR images (real or fallback)."""
    return ensure_test_data(min_images=4, pattern='gradient')


def get_test_hr_path() -> Path:
    """Get path to test HR images (real or fallback)."""
    if REAL_TEST_HR.exists() and len(list(REAL_TEST_HR.glob('*.png'))) > 0:
        print(f"[TestDataManager] Using real test data: {REAL_TEST_HR}")
        return REAL_TEST_HR
    return ensure_test_data(min_images=4, pattern='gradient')


def create_temp_dataset(
    num_images: int = 5,
    image_size: Tuple[int, int] = (256, 256),
    pattern: str = 'gradient'
) -> Path:
    """Create a temporary dataset for a specific test."""
    temp_dir = TEMP_DATA_DIR / f'temp_{pattern}_{num_images}'
    return create_fallback_data(num_images, image_size, pattern, output_dir=temp_dir)


def cleanup_temp_data(verbose: bool = False):
    """Clean up all temporary test data."""
    if TEMP_DATA_DIR.exists():
        import shutil
        shutil.rmtree(TEMP_DATA_DIR)
        if verbose:
            print(f"[TestDataManager] Cleaned up: {TEMP_DATA_DIR}")


def get_data_summary() -> dict:
    """Get summary of available test data."""
    summary = {
        'real_val_hr': {
            'exists': REAL_VAL_HR.exists(),
            'path': str(REAL_VAL_HR),
            'count': len(list(REAL_VAL_HR.glob('*.png'))) if REAL_VAL_HR.exists() else 0
        },
        'real_test_hr': {
            'exists': REAL_TEST_HR.exists(),
            'path': str(REAL_TEST_HR),
            'count': len(list(REAL_TEST_HR.glob('*.png'))) if REAL_TEST_HR.exists() else 0
        },
        'temp_dir': str(TEMP_DATA_DIR),
        'total_real_images': 0
    }
    summary['total_real_images'] = (
        summary['real_val_hr']['count'] + summary['real_test_hr']['count']
    )
    return summary


def print_data_summary():
    """Print summary of available test data."""
    summary = get_data_summary()
    print("\n" + "="*60)
    print("TEST DATA SUMMARY")
    print("="*60)
    print(f"Real validation data: {summary['real_val_hr']['count']} images")
    print(f"  Path: {summary['real_val_hr']['path']}")
    print(f"  Exists: {summary['real_val_hr']['exists']}")
    print(f"\nReal test data: {summary['real_test_hr']['count']} images")
    print(f"  Path: {summary['real_test_hr']['path']}")
    print(f"  Exists: {summary['real_test_hr']['exists']}")
    print(f"\nTotal real images: {summary['total_real_images']}")
    print(f"Temp directory: {summary['temp_dir']}")
    print("="*60 + "\n")


# Auto-cleanup on module load (optional)
def setup_cleanup():
    """Register cleanup to run at exit."""
    import atexit
    atexit.register(cleanup_temp_data)


if __name__ == '__main__':
    # Test the module
    print_data_summary()
    
    # Create some test data
    print("\nCreating sample fallback data...")
    test_dir = create_fallback_data(5, (256, 256), 'gradient', verbose=True)
    print(f"\nCreated at: {test_dir}")
    
    # Show what we got
    images = list(test_dir.glob('*.png'))
    print(f"Images: {[i.name for i in images]}")
