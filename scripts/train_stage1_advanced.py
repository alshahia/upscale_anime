#!/usr/bin/env python3
"""
Advanced Stage 1 Training Script
Demonstrates all new features: adaptive aggregation, OHEM, combined losses, monitoring.
"""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from utils.config import Config
from data.dataloader import build_dataloader
from training import create_orchestrator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Advanced Stage 1 Training with All Features'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/stage1_full.yaml',
        help='Config file path'
    )
    
    parser.add_argument(
        '--aggregation',
        type=str,
        choices=['simple', 'adaptive', 'multiscale'],
        default='adaptive',
        help='Aggregation type'
    )
    
    parser.add_argument(
        '--ohem',
        action='store_true',
        help='Enable Online Hard Example Mining'
    )
    
    parser.add_argument(
        '--ohem-ratio',
        type=float,
        default=0.7,
        help='OHEM hard example ratio (0.0-1.0)'
    )
    
    parser.add_argument(
        '--warmup',
        type=int,
        default=10,
        help='Warmup epochs'
    )
    
    parser.add_argument(
        '--combined-loss',
        action='store_true',
        help='Use combined loss (L1 + Wavelet + Gradient + Diversity)'
    )
    
    parser.add_argument(
        '--monitoring',
        action='store_true',
        help='Enable advanced monitoring (teacher disagreement, frequency analysis)'
    )
    
    parser.add_argument(
        '--anime-losses',
        action='store_true',
        help='Enable anime-specific losses'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate config and exit'
    )
    
    return parser.parse_args()


def print_feature_banner(args):
    """Print feature banner."""
    print("\n" + "="*60)
    print("ADVANCED STAGE 1 TRAINING")
    print("="*60)
    print(f"\nEnabled Features:")
    print(f"  • Aggregation: {args.aggregation}")
    print(f"  • OHEM: {'Yes' if args.ohem else 'No'} (ratio: {args.ohem_ratio})")
    print(f"  • Warmup: {args.warmup} epochs")
    print(f"  • Combined Loss: {'Yes' if args.combined_loss else 'No'}")
    print(f"  • Advanced Monitoring: {'Yes' if args.monitoring else 'No'}")
    print(f"  • Anime Losses: {'Yes' if args.anime_losses else 'No'}")
    print("\n" + "="*60 + "\n")


def main():
    """Main function."""
    args = parse_args()
    
    # Print feature banner
    print_feature_banner(args)
    
    # Load config
    print(f"Loading config: {args.config}")
    config = Config.from_file(args.config)
    
    # Apply overrides
    config['training']['stage1']['aggregation_type'] = args.aggregation
    config['training']['stage1']['warmup_epochs'] = args.warmup
    
    if args.ohem:
        config['training']['stage1']['ohem_enabled'] = True
        config['training']['stage1']['ohem_ratio'] = args.ohem_ratio
    
    if args.combined_loss:
        config['training']['stage1']['use_combined_loss'] = True
    
    if args.monitoring:
        config['training']['stage1']['log_teacher_disagreement'] = True
        config['training']['stage1']['log_frequency_loss'] = True
    
    if args.anime_losses:
        config['training']['stage1']['anime']['line_art_preservation'] = True
        config['training']['stage1']['anime']['color_consistency'] = True
        config['training']['stage1']['anime']['flat_region_preservation'] = True
    
    # Dry run
    if args.dry_run:
        print("\n✓ Config validated successfully!")
        print("\nConfiguration:")
        print(f"  - Stage 1: {config['training']['stage1']['epochs']} epochs")
        print(f"  - Aggregation: {config['training']['stage1']['aggregation_type']}")
        print(f"  - Batch size: {config['training']['stage1']['batch_size']}")
        print(f"  - Learning rate: {config['training']['stage1']['lr']}")
        print(f"  - Warmup: {config['training']['stage1']['warmup_epochs']} epochs")
        print(f"  - OHEM: {config['training']['stage1']['ohem_enabled']}")
        print(f"  - Combined Loss: {config['training']['stage1']['use_combined_loss']}")
        print("\nExiting (dry run).")
        return
    
    # Setup
    print("Setting up training...")
    orchestrator = create_orchestrator(config)
    orchestrator.setup()
    
    # Build dataloaders
    print("Building dataloaders...")
    train_loader = build_dataloader(config, mode='train')
    val_loader = build_dataloader(config, mode='val')
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Train
    print("\nStarting training...\n")
    try:
        orchestrator.train(train_loader, val_loader)
        print("\n✓ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Checkpoint saved. Resume with --resume flag.")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        raise


if __name__ == '__main__':
    main()
