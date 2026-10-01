"""
Download SPAN-F (XiaomiMM) pretrained weights for NTIRE 2025.

SPAN-F is the 2nd place winner (XiaomiMM team) of NTIRE 2025 Efficient Super-Resolution Challenge.
It's a lighter variant of SPAN with 32 channels and optimized for speed.

Official sources:
- GitHub: https://github.com/hongyuanyu/SPAN
- Google Drive: https://drive.google.com/file/d/1iYUA2TzKuxI0vzmA-UXr_nB43XgPOXUg/view?usp=sharing
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from anime_sr.utils.pretrained_models import get_pretrained_dir


def download_spanf_from_googledrive(output_dir: Path = None) -> Path:
    """
    Provide instructions for downloading SPAN-F weights from Google Drive.
    
    Note: Google Drive direct download requires special handling.
    This function provides the download link and instructions.
    """
    if output_dir is None:
        output_dir = get_pretrained_dir()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "SPANF_x4.pth"
    
    print("=" * 70)
    print("SPAN-F Pretrained Weights Download")
    print("=" * 70)
    print()
    print("SPAN-F (XiaomiMM) - NTIRE 2025 2nd Place Winner")
    print("Architecture: 32 channels, optimized for speed")
    print()
    print("Download Instructions:")
    print("-" * 70)
    print()
    print("1. Visit the Google Drive link:")
    print("   https://drive.google.com/file/d/1iYUA2TzKuxI0vzmA-UXr_nB43XgPOXUg/view?usp=sharing")
    print()
    print("2. Download the checkpoint file (should be named something like:")
    print("   'SPANF_x4.pth' or similar)")
    print()
    print(f"3. Place the downloaded file in:")
    print(f"   {output_path}")
    print()
    print("Alternative: Use gdown")
    print("-" * 70)
    print()
    print("If you have gdown installed, you can try:")
    print("   gdown 1iYUA2TzKuxI0vzmA-UXr_nB43XgPOXUg -O " + str(output_path))
    print()
    print("To install gdown:")
    print("   pip install gdown")
    print()
    print("=" * 70)
    print()
    
    # Check if file already exists
    if output_path.exists():
        print(f"✓ SPAN-F weights already exist at:")
        print(f"  {output_path}")
        print()
        return output_path
    else:
        print(f"✗ SPAN-F weights not found at expected location:")
        print(f"  {output_path}")
        print()
        return None


def verify_spanf_weights(checkpoint_path: Path) -> bool:
    """
    Verify that SPAN-F weights can be loaded and match expected architecture.
    """
    import torch
    
    try:
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except Exception:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'params' in checkpoint:
            state_dict = checkpoint['params']
        else:
            state_dict = checkpoint
        
        # Count parameters
        total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
        
        print(f"✓ Checkpoint loaded successfully")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Expected for SPAN-F: ~250K-350K parameters")
        print()
        
        # Try to load into model
        try:
            from anime_sr.models.span import SPANF
            model = SPANF(scale=4)
            model.load_state_dict(state_dict, strict=True)
            print("✓ Weights compatible with SPANF architecture")
            return True
        except RuntimeError as e:
            print(f"⚠ Architecture mismatch: {e}")
            print("  The checkpoint may need key remapping.")
            return False
            
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download SPAN-F pretrained weights (NTIRE 2025 2nd Place)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Directory to save weights (default: ./pretrained)'
    )
    parser.add_argument(
        '--verify', '-v',
        type=str,
        default=None,
        help='Verify existing checkpoint file'
    )
    parser.add_argument(
        '--instructions',
        action='store_true',
        help='Show download instructions only'
    )
    
    args = parser.parse_args()
    
    if args.verify:
        checkpoint_path = Path(args.verify)
        if checkpoint_path.exists():
            print(f"Verifying: {checkpoint_path}")
            verify_spanf_weights(checkpoint_path)
        else:
            print(f"✗ File not found: {checkpoint_path}")
        return
    
    if args.instructions:
        download_spanf_from_googledrive(args.output_dir)
        return
    
    # Default: show instructions and check for existing file
    result = download_spanf_from_googledrive(args.output_dir)
    
    if result:
        print("Verifying weights...")
        verify_spanf_weights(result)
    else:
        print("Please download the weights manually following the instructions above.")
        print()
        print("After downloading, verify with:")
        print(f"   python scripts/download_spanf.py --verify <path_to_downloaded_file>")


if __name__ == '__main__':
    main()
