#!/usr/bin/env python3
"""
Anime-Specific Training Tuning Script
Helps tune parameters for anime video frame super-resolution.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from utils.config import Config
from data.anime_degradation import AnimeDegradationPipeline
from losses.anime_losses import AnimeCombinedLoss


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Tune Anime Training Parameters'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/stage1_anime.yaml',
        help='Base config file'
    )
    
    parser.add_argument(
        '--line-art-weight',
        type=float,
        help='Line art preservation weight'
    )
    
    parser.add_argument(
        '--color-weight',
        type=float,
        help='Color consistency weight'
    )
    
    parser.add_argument(
        '--flat-weight',
        type=float,
        help='Flat region preservation weight'
    )
    
    parser.add_argument(
        '--anime-degradation',
        action='store_true',
        help='Enable anime-specific degradation'
    )
    
    parser.add_argument(
        '--temporal',
        action='store_true',
        help='Enable temporal consistency for video'
    )
    
    parser.add_argument(
        '--preview-degradation',
        action='store_true',
        help='Preview anime degradation effects'
    )
    
    parser.add_argument(
        '--test-losses',
        action='store_true',
        help='Test anime losses on sample images'
    )
    
    return parser.parse_args()


def preview_degradation():
    """Preview anime degradation effects."""
    import numpy as np
    from PIL import Image
    
    print("\n" + "="*60)
    print("ANIME DEGRADATION PREVIEW")
    print("="*60)
    
    # Create synthetic anime-like image
    img = np.ones((256, 256, 3), dtype=np.float32) * 0.5
    
    # Add flat color regions
    img[:128, :128] = [0.2, 0.4, 0.8]  # Blue region
    img[:128, 128:] = [0.9, 0.3, 0.2]  # Red region
    img[128:, :128] = [0.2, 0.8, 0.3]  # Green region
    img[128:, 128:] = [0.9, 0.9, 0.2]  # Yellow region
    
    # Add sharp edges
    img = np.clip(img, 0, 1)
    
    # Convert to tensor
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    
    # Apply degradation
    pipeline = AnimeDegradationPipeline(enable_all=True)
    degraded = pipeline(x)
    
    # Save preview
    degraded_img = degraded[0].permute(1, 2, 0).numpy()
    degraded_img = (degraded_img * 255).clip(0, 255).astype(np.uint8)
    
    output_path = Path('anime_degradation_preview.png')
    Image.fromarray(degraded_img).save(output_path)
    
    print(f"\n✓ Degradation preview saved to: {output_path.absolute()}")
    print("\nEffects applied:")
    print("  • Anime blur (preserves sharp lines)")
    print("  • Directional blur (motion/interlacing)")
    print("  • Color quantization (8-bit banding)")
    print("  • Banding artifacts (gradient compression)")
    print("  • Ringing artifacts (edge compression)")
    
    # Also save original for comparison
    orig_img = (img * 255).clip(0, 255).astype(np.uint8)
    orig_path = Path('anime_original.png')
    Image.fromarray(orig_img).save(orig_path)
    print(f"\n✓ Original image saved to: {orig_path.absolute()}")


def test_anime_losses():
    """Test anime losses on sample images."""
    print("\n" + "="*60)
    print("TESTING ANIME LOSSES")
    print("="*60)
    
    # Create test images
    # 1. Perfect prediction
    pred_perfect = torch.rand(2, 3, 128, 128)
    target = pred_perfect.clone()
    
    # 2. Blurry prediction (line art problem)
    pred_blurry = torch.nn.functional.avg_pool2d(
        target, 3, stride=1, padding=1, count_include_pad=False
    )
    
    # 3. Color shifted prediction (color problem)
    pred_color_shift = target.clone()
    pred_color_shift[:, 0] += 0.1  # Red shift
    pred_color_shift = torch.clamp(pred_color_shift, 0, 1)
    
    # 4. Noisy flat regions (flat region problem)
    pred_noisy = target.clone()
    noise = torch.randn_like(pred_noisy) * 0.05
    pred_noisy = torch.clamp(pred_noisy + noise, 0, 1)
    
    # Test line art loss
    line_loss = AnimeCombinedLoss(
        line_art_weight=1.0,
        color_weight=0.0,
        flat_weight=0.0,
    )
    
    print("\n1. Line Art Preservation Loss:")
    loss_perfect = line_loss(pred_perfect, target)
    loss_blurry = line_loss(pred_blurry, target)
    
    print(f"   Perfect: {loss_perfect['line_art'].item():.4f}")
    print(f"   Blurry:  {loss_blurry['line_art'].item():.4f}")
    print(f"   Ratio:   {loss_blurry['line_art'].item() / (loss_perfect['line_art'].item() + 1e-8):.2f}x")
    
    # Test color loss
    color_loss = AnimeCombinedLoss(
        line_art_weight=0.0,
        color_weight=1.0,
        flat_weight=0.0,
    )
    
    print("\n2. Color Consistency Loss:")
    loss_perfect = color_loss(pred_perfect, target)
    loss_color_shift = color_loss(pred_color_shift, target)
    
    print(f"   Perfect:     {loss_perfect['color_consistency'].item():.4f}")
    print(f"   Color shift: {loss_color_shift['color_consistency'].item():.4f}")
    print(f"   Ratio:       {loss_color_shift['color_consistency'].item() / (loss_perfect['color_consistency'].item() + 1e-8):.2f}x")
    
    # Test flat region loss
    flat_loss = AnimeCombinedLoss(
        line_art_weight=0.0,
        color_weight=0.0,
        flat_weight=1.0,
    )
    
    # Create flat target
    flat_target = torch.ones(2, 3, 128, 128) * 0.5
    pred_perfect_flat = flat_target.clone()
    pred_noisy_flat = torch.clamp(flat_target + torch.randn_like(flat_target) * 0.05, 0, 1)
    
    print("\n3. Flat Region Preservation Loss:")
    loss_perfect = flat_loss(pred_perfect_flat, flat_target)
    loss_noisy = flat_loss(pred_noisy_flat, flat_target)
    
    print(f"   Perfect: {loss_perfect['flat_preservation'].item():.4f}")
    print(f"   Noisy:   {loss_noisy['flat_preservation'].item():.4f}")
    print(f"   Ratio:   {loss_noisy['flat_preservation'].item() / (loss_perfect['flat_preservation'].item() + 1e-8):.2f}x")
    
    print("\n" + "="*60)


def print_recommendations(args):
    """Print recommendations based on args."""
    print("\n" + "="*60)
    print("ANIME TRAINING RECOMMENDATIONS")
    print("="*60)
    
    print("\nRecommended Settings:")
    
    if args.line_art_weight is None:
        print("  • Line art weight: 1.0-1.2 (preserve sharp lines)")
    else:
        print(f"  • Line art weight: {args.line_art_weight} ✓")
    
    if args.color_weight is None:
        print("  • Color weight: 0.5-0.8 (prevent bleeding)")
    else:
        print(f"  • Color weight: {args.color_weight} ✓")
    
    if args.flat_weight is None:
        print("  • Flat region weight: 0.5-0.6 (smooth flat areas)")
    else:
        print(f"  • Flat region weight: {args.flat_weight} ✓")
    
    print("\nData Preparation:")
    print("  • Use high-quality source (BD rips preferred)")
    print("  • Extract at native resolution (avoid upscaling)")
    print("  • Remove duplicates and low-quality frames")
    
    if args.temporal:
        print("\nTemporal Training:")
        print("  • Use frame_window=3 for local consistency")
        print("  • Enable scene change detection")
        print("  • Set temporal_weight=0.2-0.3")
    
    if args.anime_degradation:
        print("\nAnime Degradation:")
        print("  • Smaller blur kernels (preserve lines)")
        print("  • JPEG quality 60-95 (simulate compression)")
        print("  • Enable color quantization (8-bit banding)")
    
    print("\nMonitoring:")
    print("  • Watch edge_coverage metric (should increase)")
    print("  • Monitor flat_coverage (should maintain)")
    print("  • Check for color bleeding in validation")
    
    print("\n" + "="*60)


def main():
    """Main function."""
    args = parse_args()
    
    print("\n" + "="*60)
    print("ANIME TRAINING TUNING")
    print("="*60)
    
    # Preview degradation if requested
    if args.preview_degradation:
        preview_degradation()
    
    # Test losses if requested
    if args.test_losses:
        test_anime_losses()
    
    # Load and modify config
    print(f"\nLoading config: {args.config}")
    config = Config.from_file(args.config)
    
    # Apply overrides
    if args.line_art_weight is not None:
        config['training']['stage1']['anime']['line_art_weight'] = args.line_art_weight
    
    if args.color_weight is not None:
        config['training']['stage1']['anime']['color_weight'] = args.color_weight
    
    if args.flat_weight is not None:
        config['training']['stage1']['anime']['flat_weight'] = args.flat_weight
    
    if args.anime_degradation:
        config['data']['degradation']['anime_degradation'] = True
        config['data']['degradation']['color_quantization'] = True
        config['data']['degradation']['banding_simulation'] = True
    
    if args.temporal:
        config['training']['stage1']['temporal_consistency'] = True
        config['data']['video']['temporal_training'] = True
    
    # Print current config
    print("\nCurrent Anime Settings:")
    anime_cfg = config['training']['stage1']['anime']
    print(f"  • Line art preservation: {anime_cfg['line_art_preservation']} (weight: {anime_cfg['line_art_weight']})")
    print(f"  • Color consistency: {anime_cfg['color_consistency']} (weight: {anime_cfg['color_weight']})")
    print(f"  • Flat region preservation: {anime_cfg['flat_region_preservation']} (weight: {anime_cfg['flat_weight']})")
    
    # Print recommendations
    print_recommendations(args)
    
    print("\n✓ Configuration ready!")
    print(f"  Start training with: python scripts/train.py --config {args.config}")


if __name__ == '__main__':
    main()
