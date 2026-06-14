#!/usr/bin/env python3
"""
Create dummy test data for quick testing
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data import create_dummy_dataset, get_dataset_info


def main():
    parser = argparse.ArgumentParser(description='Create test dataset')
    parser.add_argument('--output', '-o', type=str, default='data/test_hr',
                      help='Output directory')
    parser.add_argument('--num-images', '-n', type=int, default=10,
                      help='Number of images to create')
    parser.add_argument('--size', '-s', type=int, default=256,
                      help='Image size (square)')
    parser.add_argument('--pattern', '-p', type=str, default='gradient',
                      choices=['gradient', 'noise', 'checkerboard', 'solid'],
                      help='Image pattern')
    
    args = parser.parse_args()
    
    print(f"Creating {args.num_images} dummy images in {args.output}")
    print(f"Size: {args.size}x{args.size}, Pattern: {args.pattern}")
    
    # Create dataset
    created = create_dummy_dataset(
        output_dir=args.output,
        num_images=args.num_images,
        image_size=(args.size, args.size),
        pattern=args.pattern,
    )
    
    print(f"\nCreated {len(created)} images:")
    for path in created[:5]:
        print(f"  - {path.name}")
    if len(created) > 5:
        print(f"  ... and {len(created) - 5} more")
    
    # Show info
    info = get_dataset_info(args.output, scale=4)
    print(f"\nDataset info:")
    print(f"  Path: {info['path']}")
    print(f"  Images: {info['count']}")
    print(f"  Avg resolution: {info['avg_resolution']}")
    print(f"  Total size: {info['total_size_gb']:.2f} GB")
    print(f"\nReady for training!")
    print(f"  python scripts/train.py --config configs/example_quick_test.yaml")


if __name__ == '__main__':
    main()
