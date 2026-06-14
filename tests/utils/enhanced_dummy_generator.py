"""
Enhanced Dummy Data Generator for Testing

Creates realistic test images when real data is not available.
More advanced patterns than basic test_data_manager.py

Patterns:
- gradient: Smooth gradients
- noise: Random noise for texture
- checkerboard: Sharp edges for testing
- anime_style: Simple line art + flat colors
- natural: Perlin noise for realistic texture
- striped: Horizontal/vertical stripes
- radial: Radial gradients
- fractal: Simple fractal patterns
"""
import numpy as np
from PIL import Image, ImageDraw
import cv2
from pathlib import Path
from typing import Tuple, List, Optional


def create_gradient_image(size: Tuple[int, int], seed: int = 0, 
                          direction: str = 'horizontal') -> np.ndarray:
    """Create smooth gradient image."""
    np.random.seed(seed)
    h, w = size
    
    if direction == 'horizontal':
        r = np.linspace(0, 255, w).reshape(1, w).repeat(h, axis=0)
        g = np.linspace(128, 255, h).reshape(h, 1).repeat(w, axis=1)
        b = np.ones((h, w)) * 200
    elif direction == 'vertical':
        r = np.linspace(128, 255, h).reshape(h, 1).repeat(w, axis=1)
        g = np.linspace(0, 255, w).reshape(1, w).repeat(h, axis=0)
        b = np.ones((h, w)) * 180
    else:  # diagonal
        r = np.indices((h, w)).sum(axis=0) / (h + w) * 255
        g = np.indices((h, w)).sum(axis=0) / (h + w) * 200
        b = np.indices((h, w)).sum(axis=0) / (h + w) * 150
    
    noise = np.random.randn(h, w) * 10
    r = np.clip(r + noise, 0, 255).astype(np.uint8)
    g = np.clip(g + noise, 0, 255).astype(np.uint8)
    b = np.clip(b + noise, 0, 255).astype(np.uint8)
    
    return np.stack([r, g, b], axis=2)


def create_noise_image(size: Tuple[int, int], seed: int = 0,
                       octaves: int = 4) -> np.ndarray:
    """Create random noise texture using multiple octaves."""
    np.random.seed(seed)
    h, w = size
    
    img = np.zeros((h, w), dtype=np.float32)
    
    for octave in range(octaves):
        scale = 2 ** octave
        octave_noise = np.random.rand(max(1, h // scale), max(1, w // scale))
        
        # Upsample
        from scipy.ndimage import zoom
        upsampled = zoom(octave_noise, (h / octave_noise.shape[0], 
                                        w / octave_noise.shape[1]), order=1)
        upsampled = upsampled[:h, :w]
        
        img += upsampled * (0.5 ** octave)
    
    # Normalize
    img = (img - img.min()) / (img.max() - img.min())
    
    # Create RGB with slight variations
    r = (img * 255).astype(np.uint8)
    g = np.roll(r, 10, axis=1)
    b = np.roll(r, 20, axis=0)
    
    return np.stack([r, g, b], axis=2)


def create_checkerboard_image(size: Tuple[int, int], squares: int = 8) -> np.ndarray:
    """Create checkerboard pattern for edge testing."""
    h, w = size
    
    checker = np.indices((h, w)).sum(axis=0) // (min(h, w) // squares) % 2
    
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[checker == 0] = [255, 255, 255]
    img[checker == 1] = [0, 0, 0]
    
    return img


def create_anime_style_image(size: Tuple[int, int], seed: int = 0) -> np.ndarray:
    """Create simple anime-style image with flat colors and lines."""
    np.random.seed(seed)
    h, w = size
    
    # Flat background
    bg_color = np.random.randint(180, 255, size=3)
    img = np.ones((h, w, 3), dtype=np.uint8) * bg_color
    
    # Add simple shapes
    if cv2 is not None:
        # Draw circles (eyes)
        eye_y = h // 2
        left_eye_x = w // 3
        right_eye_x = 2 * w // 3
        eye_radius = min(h, w) // 8
        
        # White of eyes
        cv2.circle(img, (left_eye_x, eye_y), eye_radius, [240, 240, 240], -1)
        cv2.circle(img, (right_eye_x, eye_y), eye_radius, [240, 240, 240], -1)
        
        # Pupils
        cv2.circle(img, (left_eye_x, eye_y), eye_radius // 2, [40, 40, 40], -1)
        cv2.circle(img, (right_eye_x, eye_y), eye_radius // 2, [40, 40, 40], -1)
        
        # Hair (random colored areas)
        hair_color = np.random.randint(50, 150, size=3).tolist()
        for i in range(5):
            x = np.random.randint(0, w)
            y = 0
            pts = np.array([[x, y], [x-20, h//3], [x+20, h//3]], np.int32)
            cv2.fillPoly(img, [pts], hair_color)
    
    return img


def create_natural_image(size: Tuple[int, int], seed: int = 0) -> np.ndarray:
    """Create natural-looking texture using Perlin noise."""
    np.random.seed(seed)
    h, w = size
    
    # Base noise
    noise = np.random.rand(h, w, 3)
    
    # Smooth
    from scipy.ndimage import gaussian_filter
    for i in range(3):
        noise[:, :, i] = gaussian_filter(noise[:, :, i], sigma=5)
    
    # Scale to RGB
    img = (noise * 255).astype(np.uint8)
    
    return img


def create_striped_image(size: Tuple[int, int], seed: int = 0,
                         orientation: str = 'horizontal') -> np.ndarray:
    """Create striped pattern."""
    np.random.seed(seed)
    h, w = size
    
    if orientation == 'horizontal':
        stripe_width = max(1, h // 10)
        stripes = np.arange(h) // stripe_width % 2
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(3):
            color1 = np.random.randint(200, 255)
            color2 = np.random.randint(50, 150)
            channel = np.where(stripes.reshape(h, 1) == 0, color1, color2)
            img[:, :, i] = channel
    else:
        stripe_width = max(1, w // 10)
        stripes = np.arange(w) // stripe_width % 2
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(3):
            color1 = np.random.randint(200, 255)
            color2 = np.random.randint(50, 150)
            channel = np.where(stripes.reshape(1, w) == 0, color1, color2)
            img[:, :, i] = channel
    
    return img


def create_radial_image(size: Tuple[int, int], seed: int = 0) -> np.ndarray:
    """Create radial gradient image."""
    np.random.seed(seed)
    h, w = size
    
    # Center
    cy, cx = h // 2, w // 2
    
    # Distance from center
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - cy)**2 + (x - cx)**2)
    
    # Normalize
    max_dist = np.sqrt((h//2)**2 + (w//2)**2)
    normalized = dist / max_dist
    
    # Create RGB
    r = (normalized * 255).astype(np.uint8)
    g = ((1 - normalized) * 255).astype(np.uint8)
    b = (np.sin(normalized * np.pi) * 255).astype(np.uint8)
    
    return np.stack([r, g, b], axis=2)


def create_fractal_image(size: Tuple[int, int], seed: int = 0,
                        iterations: int = 50) -> np.ndarray:
    """Create simple Mandelbrot-like fractal."""
    np.random.seed(seed)
    h, w = size
    
    # Complex plane
    x = np.linspace(-2, 1, w)
    y = np.linspace(-1.5, 1.5, h)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    
    Z = np.zeros_like(C)
    M = np.zeros(C.shape)
    
    for i in range(iterations):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask]
        M[~mask & (M == 0)] = i
    
    # Color mapping
    M = M / iterations
    r = (M * 255).astype(np.uint8)
    g = (np.sin(M * np.pi) * 255).astype(np.uint8)
    b = ((1 - M) * 255).astype(np.uint8)
    
    return np.stack([r, g, b], axis=2)


# Pattern registry
PATTERNS = {
    'gradient': create_gradient_image,
    'noise': create_noise_image,
    'checkerboard': create_checkerboard_image,
    'anime_style': create_anime_style_image,
    'natural': create_natural_image,
    'striped': create_striped_image,
    'radial': create_radial_image,
    'fractal': create_fractal_image,
}


def create_dummy_dataset(
    output_dir: Path,
    num_images: int = 10,
    image_size: Tuple[int, int] = (256, 256),
    pattern: str = 'gradient',
    **kwargs
) -> List[Path]:
    """
    Create dummy dataset with specified pattern.
    
    Args:
        output_dir: Directory to save images
        num_images: Number of images to create
        image_size: (height, width) for images
        pattern: Image pattern type
        **kwargs: Additional pattern-specific arguments
    
    Returns:
        List of created image paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if pattern not in PATTERNS:
        raise ValueError(f"Unknown pattern: {pattern}. Available: {list(PATTERNS.keys())}")
    
    creator = PATTERNS[pattern]
    created_paths = []
    
    for i in range(num_images):
        # Create image
        img_array = creator(image_size, seed=i, **kwargs)
        
        # Save
        img_path = output_dir / f'dummy_{pattern}_{i:03d}.png'
        Image.fromarray(img_array).save(img_path)
        created_paths.append(img_path)
    
    return created_paths


def get_available_patterns() -> List[str]:
    """Get list of available patterns."""
    return list(PATTERNS.keys())


# Backward compatibility - re-export from test_data_manager
try:
    from .test_data_manager import (
        ensure_test_data,
        get_val_hr_path,
        get_test_hr_path,
        cleanup_temp_data,
        print_data_summary
    )
except ImportError:
    pass


if __name__ == '__main__':
    # Test all patterns
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        print("Testing all patterns:")
        for pattern in get_available_patterns():
            try:
                paths = create_dummy_dataset(
                    tmpdir / pattern,
                    num_images=2,
                    image_size=(128, 128),
                    pattern=pattern
                )
                print(f"  ✓ {pattern}: Created {len(paths)} images")
            except Exception as e:
                print(f"  ✗ {pattern}: {e}")
