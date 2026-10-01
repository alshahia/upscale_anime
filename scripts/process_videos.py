#!/usr/bin/env python3
"""
Video processing script for anime super-resolution training
Extracts high-quality frames from videos with filtering and deduplication
"""
import sys
import argparse
from pathlib import Path
import json

# Package installed via pip install -e . - no sys.path needed

try:
    from anime_sr.data.video_extraction import (
        extract_from_folder,
        get_video_info,
        preview_extraction
    )
except ImportError:
    from anime_sr.data.video_extraction import (
        extract_from_folder,
        get_video_info,
        preview_extraction
    )


def cmd_extract(args):
    """Extract frames from videos"""
    print("=" * 60)
    print("Video Frame Extraction")
    print("=" * 60)
    
    # Determine input path
    if args.input:
        input_path = Path(args.input)
    else:
        # Auto-detect anime_vid folder
        input_path = Path("data/anime_vid")
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("data/anime_video_frames")
    
    print(f"Input:  {input_path.absolute()}")
    print(f"Output: {output_path.absolute()}")
    print(f"Settings:")
    print(f"  - Extract every: {args.extract_every} frames")
    print(f"  - Quality threshold: {args.quality_threshold}")
    print(f"  - Remove duplicates: {args.remove_duplicates}")
    print(f"  - Min resolution: {args.min_resolution}x{args.min_resolution}")
    print()
    
    # Run extraction
    result = extract_from_folder(
        video_folder=input_path,
        output_dir=output_path,
        extract_every_n_frames=args.extract_every,
        quality_threshold=args.quality_threshold,
        remove_duplicates=args.remove_duplicates,
        min_resolution=(args.min_resolution, args.min_resolution),
        duplicate_threshold=args.duplicate_threshold
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Extraction Summary")
    print("=" * 60)
    
    if result['success']:
        print(f"Videos processed: {result['total_videos']}")
        print(f"Total frames extracted: {result['total_extracted']}")
        print(f"Output directory: {result['output_dir']}")
        
        # Save stats to JSON
        stats_file = output_path / "extraction_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Stats saved to: {stats_file}")
    else:
        print(f"Error: {result['error']}")
        return 1
    
    return 0


def cmd_info(args):
    """Show information about video folder"""
    print("=" * 60)
    print("Video Folder Information")
    print("=" * 60)
    
    # Determine input path
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = Path("data/anime_vid")
    
    if not input_path.exists():
        print(f"Error: Path does not exist: {input_path}")
        return 1
    
    # Find videos
    video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.webm')
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_path.glob(f"*{ext}"))
        video_files.extend(input_path.glob(f"*{ext.upper()}"))
    video_files = sorted(list(set(video_files)))
    
    if not video_files:
        print(f"No video files found in {input_path}")
        print(f"Supported formats: {', '.join(video_extensions)}")
        return 1
    
    print(f"Found {len(video_files)} video(s) in {input_path}")
    print()
    
    total_frames = 0
    total_duration = 0
    
    for i, video_path in enumerate(video_files):
        info = get_video_info(video_path)
        
        if info['success']:
            print(f"[{i+1}] {info['name']}")
            print(f"    Resolution: {info['width']}x{info['height']}")
            print(f"    FPS: {info['fps']:.2f}")
            print(f"    Total frames: {info['total_frames']}")
            print(f"    Duration: {info['duration']:.1f}s ({info['duration']/60:.1f}m)")
            print(f"    Est. extractable (every 30 frames): {info['estimated_frames_30fps']}")
            print()
            
            total_frames += info['total_frames']
            total_duration += info['duration']
    
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total videos: {len(video_files)}")
    print(f"Total frames: {total_frames:,}")
    print(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f}m)")
    print(f"Est. extractable (every 30 frames): {total_frames // 30:,}")
    
    return 0


def cmd_preview(args):
    """Preview extraction quality from sample frames"""
    print("=" * 60)
    print("Extraction Quality Preview")
    print("=" * 60)
    
    if not args.input:
        print("Error: --input required for preview mode")
        return 1
    
    video_path = Path(args.input)
    
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return 1
    
    print(f"Video: {video_path.name}")
    print(f"Sampling {args.num_samples} frames...")
    print()
    
    samples = preview_extraction(
        video_path=video_path,
        num_samples=args.num_samples,
        extract_every_n_frames=args.extract_every
    )
    
    if not samples:
        print("Error: Could not extract samples")
        return 1
    
    print("Sample quality scores:")
    for frame_num, quality in samples:
        bar = "█" * int(quality * 20) + "░" * (20 - int(quality * 20))
        status = "✓" if quality >= args.quality_threshold else "✗"
        print(f"  Frame {frame_num:6d}: [{bar}] {quality:.2f} {status}")
    
    avg_quality = sum(q for _, q in samples) / len(samples)
    print()
    print(f"Average quality: {avg_quality:.2f}")
    
    if avg_quality >= args.quality_threshold:
        print("✓ This video should produce good training frames")
    else:
        print("⚠ Consider lowering --quality-threshold for this video")
    
    return 0


def cmd_config(args):
    """Generate YAML config snippet for video dataset"""
    print("# Add this to your config file (e.g., configs/base.yaml)")
    print()
    print("data:")
    print("  # Existing image datasets...")
    print("  datasets:")
    print("    - name: \"div2k\"")
    print("      weight: 0.5")
    print("      enabled: true")
    print("      hr_dir: \"data/DIV2K_train_HR\"")
    print("    - name: \"anime_video\"")
    print("      weight: 0.5")
    print("      enabled: true")
    print("      hr_dir: \"data/anime_video_frames\"  # Pre-extracted")
    print("      # OR use on-demand extraction:")
    print("      # video_dir: \"data/anime_vid\"")
    print("      # extract_mode: \"on_demand\"")
    print()
    print("  # Video extraction settings")
    print("  video:")
    print("    enabled: true")
    print(f"    video_dir: \"{args.video_dir or 'data/anime_vid'}\"")
    print(f"    extract_mode: \"{args.mode}\"")
    print(f"    extract_every_n_frames: {args.extract_every}")
    print(f"    quality_threshold: {args.quality_threshold}")
    print(f"    remove_duplicates: {str(args.remove_duplicates).lower()}")
    print(f"    min_resolution: {args.min_resolution}")
    print("    output_dir: \"data/anime_video_frames\"")
    print("    cache_size_gb: 10  # For on-demand mode")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Process anime videos for super-resolution training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from default anime_vid folder
  python scripts/process_videos.py extract

  # Extract with custom settings
  python scripts/process_videos.py extract \\
      --input data/my_videos \\
      --output data/extracted_frames \\
      --extract-every 60 \\
      --quality-threshold 0.8

  # Show video folder info
  python scripts/process_videos.py info

  # Preview extraction quality
  python scripts/process_videos.py preview --input video.mp4

  # Generate config snippet
  python scripts/process_videos.py config --mode on_demand
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract frames from videos')
    extract_parser.add_argument('--input', '-i', type=str, help='Input video folder (default: data/anime_vid)')
    extract_parser.add_argument('--output', '-o', type=str, help='Output folder (default: data/anime_video_frames)')
    extract_parser.add_argument('--extract-every', type=int, default=30, help='Extract 1 frame every N frames (default: 30)')
    extract_parser.add_argument('--quality-threshold', type=float, default=0.7, help='Quality threshold 0-1 (default: 0.7)')
    extract_parser.add_argument('--remove-duplicates', action='store_true', default=True, help='Remove duplicate frames')
    extract_parser.add_argument('--no-remove-duplicates', action='store_false', dest='remove_duplicates', help='Keep duplicate frames')
    extract_parser.add_argument('--min-resolution', type=int, default=720, help='Minimum frame resolution (default: 720)')
    extract_parser.add_argument('--duplicate-threshold', type=float, default=0.95, help='Duplicate similarity threshold (default: 0.95)')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show video folder information')
    info_parser.add_argument('--input', '-i', type=str, help='Video folder to analyze (default: data/anime_vid)')
    
    # Preview command
    preview_parser = subparsers.add_parser('preview', help='Preview extraction quality')
    preview_parser.add_argument('--input', '-i', type=str, required=True, help='Video file to preview')
    preview_parser.add_argument('--num-samples', type=int, default=5, help='Number of sample frames (default: 5)')
    preview_parser.add_argument('--extract-every', type=int, default=30, help='Frame interval (default: 30)')
    preview_parser.add_argument('--quality-threshold', type=float, default=0.7, help='Quality threshold (default: 0.7)')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Generate YAML config snippet')
    config_parser.add_argument('--mode', type=str, default='pre', choices=['pre', 'on_demand'], help='Extraction mode')
    config_parser.add_argument('--video-dir', type=str, help='Video directory path')
    config_parser.add_argument('--extract-every', type=int, default=30, help='Frame interval')
    config_parser.add_argument('--quality-threshold', type=float, default=0.7, help='Quality threshold')
    config_parser.add_argument('--remove-duplicates', action='store_true', default=True, help='Remove duplicates')
    config_parser.add_argument('--min-resolution', type=int, default=720, help='Min resolution')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Dispatch command
    commands = {
        'extract': cmd_extract,
        'info': cmd_info,
        'preview': cmd_preview,
        'config': cmd_config,
    }
    
    return commands[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
