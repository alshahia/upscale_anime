#!/usr/bin/env python3
"""
Data preparation script
- Extracts I-frames from videos
- Filters low-quality images
- Organizes datasets
"""
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def calculate_image_quality(image_path: Path) -> float:
    """
    Calculate image quality score using multiple metrics.
    Returns score from 0-1 where 1 is highest quality.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return 0.0
    
    # Metric 1: Sharpness (Laplacian variance)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(laplacian_var / 500, 1.0)  # Normalize
    
    # Metric 2: Contrast (standard deviation)
    contrast = np.std(gray) / 128.0  # Normalize to 0-1
    
    # Metric 3: Resolution
    h, w = img.shape[:2]
    min_dim = min(h, w)
    resolution_score = min(min_dim / 720, 1.0)  # At least 720px
    
    # Combined score
    score = (sharpness_score * 0.5 + contrast * 0.3 + resolution_score * 0.2)
    
    return score


def extract_iframes(video_path: Path, output_dir: Path, quality_threshold: float = 0.7):
    """Extract I-frames from video file"""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return 0
    
    frame_count = 0
    saved_count = 0
    
    with tqdm(desc=f"Extracting {video_path.name}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if this is an I-frame (keyframe)
            # OpenCV doesn't directly expose this, but we can save every N frames
            # or use more sophisticated scene detection
            if frame_count % 30 == 0:  # Save every 30 frames (roughly 1/sec for 30fps)
                output_path = output_dir / f"{video_path.stem}_frame{frame_count:06d}.png"
                cv2.imwrite(str(output_path), frame)
                saved_count += 1
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    return saved_count


def filter_quality(input_dir: Path, output_dir: Path, threshold: float = 0.7):
    """Filter images by quality"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg'))
    
    good_count = 0
    bad_count = 0
    
    for img_path in tqdm(image_files, desc="Filtering by quality"):
        score = calculate_image_quality(img_path)
        
        if score >= threshold:
            # Copy to output
            output_path = output_dir / img_path.name
            shutil.copy(img_path, output_path)
            good_count += 1
        else:
            bad_count += 1
    
    print(f"Kept: {good_count}, Filtered: {bad_count}")
    return good_count


def organize_dataset(input_dir: Path, output_dir: Path, train_ratio: float = 0.9):
    """Organize dataset into train/val splits"""
    # Get all images
    image_files = list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg'))
    
    # Shuffle
    np.random.shuffle(image_files)
    
    # Split
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Create directories
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    for f in tqdm(train_files, desc="Copying train files"):
        shutil.copy(f, train_dir / f.name)
    
    for f in tqdm(val_files, desc="Copying val files"):
        shutil.copy(f, val_dir / f.name)
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for training')
    parser.add_argument('--input', '-i', type=str, required=True,
                      help='Input directory or video file')
    parser.add_argument('--output', '-o', type=str, required=True,
                      help='Output directory')
    parser.add_argument('--mode', '-m', type=str, required=True,
                      choices=['extract', 'filter', 'organize', 'full'],
                      help='Preparation mode')
    parser.add_argument('--quality-threshold', '-q', type=float, default=0.7,
                      help='Quality threshold (0-1)')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                      help='Train/validation split ratio')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.mode == 'extract':
        if input_path.is_file():
            # Single video
            count = extract_iframes(input_path, output_path, args.quality_threshold)
            print(f"Extracted {count} frames from {input_path.name}")
        elif input_path.is_dir():
            # Directory of videos
            video_extensions = {'.mp4', '.avi', '.mkv', '.mov'}
            videos = [f for f in input_path.iterdir() if f.suffix.lower() in video_extensions]
            
            total = 0
            for video in videos:
                count = extract_iframes(video, output_path, args.quality_threshold)
                total += count
            
            print(f"Extracted {total} frames from {len(videos)} videos")
    
    elif args.mode == 'filter':
        filtered_dir = output_path / 'filtered'
        count = filter_quality(input_path, filtered_dir, args.quality_threshold)
        print(f"Kept {count} high-quality images")
    
    elif args.mode == 'organize':
        organize_dataset(input_path, output_path, args.train_ratio)
    
    elif args.mode == 'full':
        # Full pipeline: extract -> filter -> organize
        print("=" * 50)
        print("Step 1: Extracting frames")
        print("=" * 50)
        
        temp_dir = output_path / 'temp_extracted'
        temp_dir.mkdir(exist_ok=True)
        
        if input_path.is_file():
            extract_iframes(input_path, temp_dir, args.quality_threshold)
        else:
            video_extensions = {'.mp4', '.avi', '.mkv', '.mov'}
            videos = [f for f in input_path.iterdir() if f.suffix.lower() in video_extensions]
            for video in videos:
                extract_iframes(video, temp_dir, args.quality_threshold)
        
        print("\n" + "=" * 50)
        print("Step 2: Filtering by quality")
        print("=" * 50)
        
        filtered_dir = output_path / 'filtered'
        filter_quality(temp_dir, filtered_dir, args.quality_threshold)
        
        print("\n" + "=" * 50)
        print("Step 3: Organizing dataset")
        print("=" * 50)
        
        final_dir = output_path / 'dataset'
        organize_dataset(filtered_dir, final_dir, args.train_ratio)
        
        # Cleanup
        shutil.rmtree(temp_dir)
        shutil.rmtree(filtered_dir)
        
        print("\n" + "=" * 50)
        print("Dataset preparation complete!")
        print(f"Output: {final_dir}")
        print("=" * 50)


if __name__ == '__main__':
    main()
