#!/usr/bin/env python3
"""
Download pretrained teacher models for Stage 1 training.

Usage:
    python scripts/download_models.py --model all --scale 4
    python scripts/download_models.py --model edsr --scale 4
    python scripts/download_models.py --list
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from anime_sr.utils.pretrained_models import (
    download_pretrained_model,
    download_all_teacher_models,
    list_available_models,
)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download pretrained teacher models (EDSR, RCAN, SwinIR)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all teacher models for 4x upscaling
  python scripts/download_models.py --model all --scale 4
  
  # Download only EDSR model
  python scripts/download_models.py --model edsr --scale 4
  
  # List available models
  python scripts/download_models.py --list
  
  # Force re-download (overwrite existing)
  python scripts/download_models.py --model all --scale 4 --force
        """
    )
    
    parser.add_argument(
        '--model', '-m', 
        type=str, 
        choices=['edsr', 'rcan', 'swinir', 'all'],
        default='all', 
        help='Model to download (default: all)'
    )
    parser.add_argument(
        '--scale', '-s', 
        type=int, 
        choices=[2, 3, 4], 
        default=4, 
        help='Upscaling factor (default: 4)'
    )
    parser.add_argument(
        '--force', '-f', 
        action='store_true',
        help='Re-download even if file exists'
    )
    parser.add_argument(
        '--list', '-l', 
        action='store_true',
        help='List available models without downloading'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
        return
    
    print(f"\n{'='*60}")
    print(f"Pretrained Model Downloader")
    print(f"{'='*60}\n")
    
    if args.model == 'all':
        results = download_all_teacher_models(args.scale, force=args.force)
        
        # Exit with error code if any download failed
        if any(p is None for p in results.values()):
            print("\n[WARN]  Some downloads failed. Check your internet connection.")
            sys.exit(1)
        else:
            print("\n[OK] All models downloaded successfully!")
            print(f"Models saved to: pretrained/")
            
    else:
        path = download_pretrained_model(args.model, args.scale, force=args.force)
        
        if path:
            print(f"\n[OK] Model ready: {path}")
            sys.exit(0)
        else:
            print(f"\n[ERROR] Download failed")
            sys.exit(1)


if __name__ == '__main__':
    main()
