"""
Video frame extraction module with quality filtering and deduplication
Supports extracting high-quality frames from anime videos for training
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Callable
from tqdm import tqdm
import tempfile
import shutil


def calculate_frame_quality(frame: np.ndarray) -> float:
    """
    Calculate image quality score for a frame.
    Returns score from 0-1 where 1 is highest quality.
    
    Metrics:
    - Sharpness (Laplacian variance)
    - Contrast (standard deviation)
    - Resolution check
    """
    if frame is None or frame.size == 0:
        return 0.0
    
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Metric 1: Sharpness (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize: good sharpness typically > 100, max around 1000+
    sharpness_score = min(laplacian_var / 500, 1.0)
    
    # Metric 2: Contrast (standard deviation)
    contrast = np.std(gray) / 128.0  # Normalize to 0-1
    contrast_score = min(contrast, 1.0)
    
    # Metric 3: Resolution
    h, w = frame.shape[:2]
    min_dim = min(h, w)
    resolution_score = min(min_dim / 720, 1.0)  # At least 720px for full score
    
    # Combined weighted score
    score = (sharpness_score * 0.5 + contrast_score * 0.3 + resolution_score * 0.2)
    
    return float(score)


def calculate_frame_hash(frame: np.ndarray, hash_size: int = 8) -> str:
    """
    Calculate perceptual hash for duplicate detection.
    Uses average hash algorithm.
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Resize to hash_size x hash_size
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    
    # Calculate mean
    mean = resized.mean()
    
    # Create hash: 1 if pixel > mean, else 0
    hash_bits = (resized > mean).flatten().astype(int)
    
    # Convert to hex string
    hash_str = ''.join(str(b) for b in hash_bits)
    
    return hash_str


def hash_similarity(hash1: str, hash2: str) -> float:
    """Calculate similarity between two hashes (0-1)"""
    if len(hash1) != len(hash2):
        return 0.0
    
    # Hamming distance
    distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    max_distance = len(hash1)
    
    # Convert to similarity
    similarity = 1.0 - (distance / max_distance)
    
    return similarity


def is_duplicate_frame(
    frame: np.ndarray,
    prev_frames_hashes: List[str],
    threshold: float = 0.95,
    hash_size: int = 8
) -> bool:
    """
    Check if frame is duplicate of any previous frame.
    Uses perceptual hashing for efficiency.
    
    Args:
        frame: Current frame
        prev_frames_hashes: List of hashes from previous frames
        threshold: Similarity threshold (0-1), higher = stricter
        hash_size: Size of hash (8 = 64-bit hash)
    
    Returns:
        True if frame is duplicate
    """
    if not prev_frames_hashes:
        return False
    
    current_hash = calculate_frame_hash(frame, hash_size)
    
    for prev_hash in prev_frames_hashes[-10:]:  # Check last 10 frames only
        similarity = hash_similarity(current_hash, prev_hash)
        if similarity >= threshold:
            return True
    
    return False


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    extract_every_n_frames: int = 30,
    quality_threshold: float = 0.7,
    remove_duplicates: bool = True,
    min_resolution: Tuple[int, int] = (720, 720),
    duplicate_threshold: float = 0.95,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict:
    """
    Extract high-quality frames from a video file.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        extract_every_n_frames: Extract 1 frame every N frames
        quality_threshold: Minimum quality score (0-1)
        remove_duplicates: Whether to remove duplicate/similar frames
        min_resolution: Minimum (width, height) to keep frame
        duplicate_threshold: Similarity threshold for duplicate detection
        progress_callback: Optional callback(current, total)
    
    Returns:
        Dictionary with extraction statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return {
            'success': False,
            'error': f'Failed to open video: {video_path}',
            'extracted': 0,
            'filtered_quality': 0,
            'filtered_duplicates': 0,
            'filtered_resolution': 0
        }
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    stats = {
        'success': True,
        'video_path': str(video_path),
        'total_frames': total_frames,
        'fps': fps,
        'duration': duration,
        'extracted': 0,
        'filtered_quality': 0,
        'filtered_duplicates': 0,
        'filtered_resolution': 0
    }
    
    frame_count = 0
    saved_count = 0
    prev_hashes = []
    
    try:
        with tqdm(total=total_frames, desc=f"Extracting {video_path.name}", disable=progress_callback is not None) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check if we should extract this frame
                if frame_count % extract_every_n_frames == 0:
                    h, w = frame.shape[:2]
                    
                    # Check minimum resolution
                    if w < min_resolution[0] or h < min_resolution[1]:
                        stats['filtered_resolution'] += 1
                        frame_count += 1
                        pbar.update(1)
                        continue
                    
                    # Check quality
                    quality = calculate_frame_quality(frame)
                    if quality < quality_threshold:
                        stats['filtered_quality'] += 1
                        frame_count += 1
                        pbar.update(1)
                        continue
                    
                    # Check duplicates
                    if remove_duplicates:
                        if is_duplicate_frame(frame, prev_hashes, duplicate_threshold):
                            stats['filtered_duplicates'] += 1
                            frame_count += 1
                            pbar.update(1)
                            continue
                        
                        # Add hash to history
                        prev_hashes.append(calculate_frame_hash(frame))
                        # Keep only last 100 hashes to save memory
                        if len(prev_hashes) > 100:
                            prev_hashes.pop(0)
                    
                    # Save frame
                    output_name = f"{video_path.stem}_frame{frame_count:06d}_q{quality:.2f}.png"
                    output_path = output_dir / output_name
                    cv2.imwrite(str(output_path), frame)
                    saved_count += 1
                    
                    if progress_callback:
                        progress_callback(saved_count, total_frames // extract_every_n_frames)
                
                frame_count += 1
                pbar.update(1)
    
    finally:
        cap.release()
    
    stats['extracted'] = saved_count
    return stats


def extract_from_folder(
    video_folder: Path,
    output_dir: Path,
    video_extensions: Tuple[str, ...] = ('.mp4', '.avi', '.mkv', '.mov', '.webm'),
    **kwargs
) -> Dict:
    """
    Extract frames from all videos in a folder.
    
    Args:
        video_folder: Directory containing videos
        output_dir: Directory to save extracted frames
        video_extensions: Tuple of supported video extensions
        **kwargs: Additional arguments passed to extract_frames_from_video
    
    Returns:
        Dictionary with combined statistics
    """
    video_folder = Path(video_folder)
    output_dir = Path(output_dir)
    
    if not video_folder.exists():
        return {
            'success': False,
            'error': f'Video folder does not exist: {video_folder}',
            'total_videos': 0,
            'total_extracted': 0
        }
    
    # Find all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_folder.glob(f'*{ext}'))
        video_files.extend(video_folder.glob(f'*{ext.upper()}'))
    
    video_files = sorted(list(set(video_files)))
    
    if not video_files:
        return {
            'success': False,
            'error': f'No video files found in {video_folder}',
            'supported_formats': list(video_extensions),
            'total_videos': 0,
            'total_extracted': 0
        }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_stats = []
    total_extracted = 0
    
    print(f"Found {len(video_files)} video(s) in {video_folder}")
    
    for video_path in video_files:
        print(f"\nProcessing: {video_path.name}")
        stats = extract_frames_from_video(video_path, output_dir, **kwargs)
        all_stats.append(stats)
        
        if stats['success']:
            total_extracted += stats['extracted']
            print(f"  Extracted: {stats['extracted']} frames")
            print(f"  Filtered (quality): {stats['filtered_quality']}")
            print(f"  Filtered (duplicates): {stats['filtered_duplicates']}")
            print(f"  Filtered (resolution): {stats['filtered_resolution']}")
        else:
            print(f"  Error: {stats['error']}")
    
    return {
        'success': True,
        'total_videos': len(video_files),
        'total_extracted': total_extracted,
        'output_dir': str(output_dir),
        'video_stats': all_stats
    }


def get_video_info(video_path: Path) -> Dict:
    """
    Get information about a video file without extracting frames.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return {'success': False, 'error': f'Failed to open {video_path}'}
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        return {
            'success': True,
            'path': str(video_path),
            'name': video_path.name,
            'total_frames': total_frames,
            'fps': fps,
            'width': width,
            'height': height,
            'duration': duration,
            'estimated_frames_30fps': total_frames // 30  # Rough estimate
        }
    finally:
        cap.release()


def preview_extraction(
    video_path: Path,
    num_samples: int = 5,
    extract_every_n_frames: int = 30
) -> List[Tuple[int, float]]:
    """
    Preview extraction quality by sampling frames.
    
    Args:
        video_path: Path to video file
        num_samples: Number of frames to sample
        extract_every_n_frames: Frame interval
    
    Returns:
        List of (frame_number, quality_score) tuples
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    samples = []
    
    try:
        # Sample evenly distributed frames
        for i in range(num_samples):
            frame_num = (i * total_frames // num_samples) // extract_every_n_frames * extract_every_n_frames
            frame_num = min(frame_num, total_frames - 1)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                quality = calculate_frame_quality(frame)
                samples.append((frame_num, quality))
    finally:
        cap.release()
    
    return samples
