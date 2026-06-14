#!/usr/bin/env python3
"""
Training Visualization & Documentation Script
============================================

Generate comprehensive training reports, visualizations, and architecture documentation
for the Anime Super-Resolution training pipeline.

Usage:
    # Generate complete training documentation
    python scripts/visualize_training.py --mode full
    
    # Analyze a specific checkpoint
    python scripts/visualize_training.py --checkpoint checkpoints/best.pth
    
    # Generate training report from history
    python scripts/visualize_training.py --history logs/training_history.json --model-name "Model A"
    
    # Compare multiple models
    python scripts/visualize_training.py --compare \
        --model-a logs/model_a_history.json \
        --model-b logs/model_b_history.json

Features:
    - Training pipeline diagrams
    - Architecture documentation (HTML)
    - Training curve visualizations
    - Checkpoint analysis
    - Model comparison reports

Author: Anime SR Training System
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.training_visualizer import (
    TrainingVisualizer,
    print_welcome_message,
    create_visualizer_for_training,
)
from utils.config import Config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Training Visualization & Documentation Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate complete architecture documentation
    python scripts/visualize_training.py --mode docs
    
    # Analyze checkpoints in a directory
    python scripts/visualize_training.py --mode analyze --checkpoints-dir checkpoints/
    
    # Create training report from config and history
    python scripts/visualize_training.py --mode report \
        --config configs/model_a_ntire.yaml \
        --history logs/history.json \
        --model-name "Model A (SPAN)"
        
    # Generate pipeline diagram only
    python scripts/visualize_training.py --mode diagram
        """
    )
    
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'docs', 'analyze', 'report', 'diagram', 'compare'],
                       help='Visualization mode')
    parser.add_argument('--config', '-c', type=str,
                       help='Path to config YAML file')
    parser.add_argument('--checkpoints-dir', type=str, default='checkpoints',
                       help='Directory containing checkpoints to analyze')
    parser.add_argument('--checkpoint', type=str,
                       help='Single checkpoint to analyze')
    parser.add_argument('--history', type=str,
                       help='Path to training history JSON file')
    parser.add_argument('--model-name', type=str, default='Model',
                       help='Name of the model for reports')
    parser.add_argument('--output-dir', '-o', type=str, default='training_reports',
                       help='Output directory for reports')
    parser.add_argument('--model-a', type=str,
                       help='History file for Model A (for comparison)')
    parser.add_argument('--model-b', type=str,
                       help='History file for Model B (for comparison)')
    
    return parser.parse_args()


def load_training_history(history_path: str) -> Optional[Dict]:
    """Load training history from JSON file."""
    history_file = Path(history_path)
    if not history_file.exists():
        print(f"Warning: History file not found: {history_path}")
        return None
    
    try:
        with open(history_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading history file: {e}")
        return None


def generate_full_documentation(visualizer: TrainingVisualizer, args):
    """Generate complete documentation package."""
    print("\nGenerating full documentation package...")
    
    # 1. Create training pipeline diagram
    print("  - Creating training pipeline diagram...")
    diagram_path = visualizer.create_training_pipeline_diagram()
    print(f"    Saved to: {diagram_path}")
    
    # 2. Generate architecture documentation
    print("  - Generating architecture documentation...")
    docs_path = visualizer.create_architecture_documentation()
    print(f"    Saved to: {docs_path}")
    
    # 3. Print config summary if available
    if args.config:
        print("  - Configuration summary:")
        try:
            config = Config.load(args.config)
            visualizer.print_config_summary(config.to_dict())
        except Exception as e:
            print(f"    Warning: Could not load config: {e}")
    
    print("\nDocumentation generation complete!")
    print(f"All outputs saved to: {Path(args.output_dir).absolute()}")


def analyze_checkpoints(visualizer: TrainingVisualizer, args):
    """Analyze checkpoints in directory or single checkpoint."""
    if args.checkpoint:
        # Analyze single checkpoint
        print(f"\nAnalyzing checkpoint: {args.checkpoint}")
        visualizer.print_checkpoint_summary(args.checkpoint)
    else:
        # Analyze directory
        checkpoints_dir = Path(args.checkpoints_dir)
        if not checkpoints_dir.exists():
            print(f"Error: Checkpoints directory not found: {checkpoints_dir}")
            return
        
        print(f"\nAnalyzing checkpoints in: {checkpoints_dir}")
        
        checkpoint_files = sorted(checkpoints_dir.glob('*.pth'))
        if not checkpoint_files:
            print("  No checkpoint files found (.pth)")
            return
        
        print(f"  Found {len(checkpoint_files)} checkpoint(s)\n")
        
        for ckpt_file in checkpoint_files[:5]:  # Analyze first 5
            visualizer.print_checkpoint_summary(str(ckpt_file))
            print()


def generate_training_report(visualizer: TrainingVisualizer, args):
    """Generate training report from history and config."""
    # Load config
    config = {}
    if args.config:
        try:
            config = Config.load(args.config).to_dict()
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
    
    # Load history
    history = {}
    if args.history:
        history = load_training_history(args.history) or {}
    
    if not history:
        print("Error: No training history provided or found.")
        print("Use --history to specify a history JSON file.")
        return
    
    print(f"\nGenerating training report for: {args.model_name}")
    
    # Generate report
    report_path = visualizer.generate_training_report(
        model_name=args.model_name,
        training_history=history,
        checkpoints_dir=args.checkpoints_dir,
        config=config,
    )
    
    print(f"\nReport saved to: {report_path}")
    
    # Also plot training curves separately
    curves_path = visualizer.plot_training_curves(
        history=history,
        title=f'{args.model_name} Training History',
        model_name=args.model_name,
    )
    print(f"Training curves saved to: {curves_path}")


def compare_models(visualizer: TrainingVisualizer, args):
    """Compare multiple models' training histories."""
    if not args.model_a or not args.model_b:
        print("Error: --model-a and --model-b are required for comparison mode")
        return
    
    # Load histories
    history_a = load_training_history(args.model_a)
    history_b = load_training_history(args.model_b)
    
    if not history_a or not history_b:
        print("Error: Could not load one or both history files")
        return
    
    print("\nGenerating model comparison...")
    
    models_data = [
        {'name': 'Model A', 'history': history_a},
        {'name': 'Model B', 'history': history_b},
    ]
    
    # Plot comparison for loss
    loss_comparison = visualizer.plot_model_comparison(
        models_data=models_data,
        metric='loss',
        output_path=Path(args.output_dir) / 'model_comparison_loss.png',
    )
    print(f"  Loss comparison: {loss_comparison}")
    
    # Plot comparison for PSNR if available
    if any('val_psnr' in h for h in [history_a, history_b]):
        psnr_comparison = visualizer.plot_model_comparison(
            models_data=models_data,
            metric='psnr',
            output_path=Path(args.output_dir) / 'model_comparison_psnr.png',
        )
        print(f"  PSNR comparison: {psnr_comparison}")
    
    print("\nModel comparison complete!")


def main():
    """Main entry point."""
    # Print welcome message
    print_welcome_message()
    
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config if provided
    config = {}
    if args.config:
        try:
            config = Config.load(args.config).to_dict()
            print(f"Loaded config: {args.config}")
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
    
    # Create visualizer
    visualizer = create_visualizer_for_training(config, output_dir=str(output_dir))
    
    # Execute based on mode
    if args.mode == 'full':
        generate_full_documentation(visualizer, args)
    elif args.mode == 'docs':
        print("\nGenerating architecture documentation...")
        docs_path = visualizer.create_architecture_documentation()
        print(f"Documentation saved to: {docs_path}")
    elif args.mode == 'diagram':
        print("\nGenerating training pipeline diagram...")
        diagram_path = visualizer.create_training_pipeline_diagram()
        print(f"Diagram saved to: {diagram_path}")
    elif args.mode == 'analyze':
        analyze_checkpoints(visualizer, args)
    elif args.mode == 'report':
        generate_training_report(visualizer, args)
    elif args.mode == 'compare':
        compare_models(visualizer, args)
    
    print("\nVisualization complete!")
    print(f"Output directory: {output_dir.absolute()}")


if __name__ == '__main__':
    main()
