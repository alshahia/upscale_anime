"""
Enhanced Inference Script with EMA, TTA, and Model Soup support.

Features:
- EMA model inference (uses EMA weights from checkpoints)
- Test-Time Augmentation (TTA) for quality boost
- Model Soup (weight averaging across checkpoints)
- Benchmarking with different configurations

Usage:
    # Basic inference with EMA
    python scripts/enhanced_inference.py --checkpoint checkpoints/best.pth --input input.png --output output.png

    # Inference with TTA (slower but better quality)
    python scripts/enhanced_inference.py --checkpoint checkpoints/best.pth --input input.png --output output.png --tta

    # Create model soup from multiple checkpoints
    python scripts/enhanced_inference.py --create-soup --checkpoint-dir checkpoints --output soup.pth --last-n 5

    # Compare regular vs EMA vs TTA
    python scripts/enhanced_inference.py --checkpoint checkpoints/best.pth --input input.png --compare
"""

import argparse
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from inference import InferenceEngine, ModelSoup, quick_inference


def create_soup_from_directory(
    checkpoint_dir: str,
    output_path: str,
    last_n: int = 5,
    pattern: str = 'epoch_*.pth',
    prefer_ema: bool = True,
):
    """Create model soup from recent checkpoints."""
    checkpoint_dir = Path(checkpoint_dir)
    
    # Find checkpoints
    checkpoints = ModelSoup.find_checkpoints(checkpoint_dir, pattern, last_n)
    
    if len(checkpoints) < 2:
        print(f"Error: Found only {len(checkpoints)} checkpoint(s), need at least 2 for soup")
        return
    
    print(f"Creating soup from {len(checkpoints)} checkpoints:")
    for ckpt in checkpoints:
        print(f"  - {ckpt.name}")
    
    # Create soup
    soup = ModelSoup.from_checkpoints(checkpoints, prefer_ema=prefer_ema)
    
    # Save
    soup.save_soup(output_path)
    print(f"\nModel soup created: {output_path}")


def run_inference(
    checkpoint_path: str,
    input_path: str,
    output_path: str,
    model_type: str = 'span',
    use_ema: bool = True,
    use_tta: bool = False,
    device: str = 'cuda',
):
    """Run inference with specified options."""
    # Create engine
    print(f"Loading model from {checkpoint_path}...")
    engine = InferenceEngine.from_checkpoint(
        checkpoint_path, 
        model_type=model_type, 
        device=device,
        use_ema=use_ema
    )
    
    # Load image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Failed to load image: {input_path}")
    print(f"Input image: {image.shape}")
    
    # Run inference
    start_time = time.time()
    
    if use_tta:
        print("Running inference with TTA (Test-Time Augmentation)...")
        sr_image = engine.run_tta(image)
    else:
        print("Running inference...")
        sr_image = engine.run(image)
    
    elapsed = time.time() - start_time
    print(f"Inference time: {elapsed:.3f}s")
    print(f"Output image: {sr_image.shape}")
    
    # Save
    cv2.imwrite(output_path, sr_image)
    print(f"Saved to: {output_path}")


def compare_methods(
    checkpoint_path: str,
    input_path: str,
    output_dir: str,
    model_type: str = 'span',
    device: str = 'cuda',
):
    """Compare regular, EMA, and TTA inference."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Failed to load image: {input_path}")
    
    print("=" * 60)
    print("Comparing Inference Methods")
    print("=" * 60)
    
    # Method 1: Regular
    print("\n1. Regular inference (no EMA)...")
    engine_regular = InferenceEngine.from_checkpoint(
        checkpoint_path, model_type, device, use_ema=False
    )
    start = time.time()
    result_regular = engine_regular.run(image)
    time_regular = time.time() - start
    cv2.imwrite(str(output_dir / 'regular.png'), result_regular)
    
    # Method 2: EMA
    print("\n2. EMA model inference...")
    engine_ema = InferenceEngine.from_checkpoint(
        checkpoint_path, model_type, device, use_ema=True
    )
    start = time.time()
    result_ema = engine_ema.run(image)
    time_ema = time.time() - start
    cv2.imwrite(str(output_dir / 'ema.png'), result_ema)
    
    # Method 3: EMA + TTA
    print("\n3. EMA + TTA inference...")
    start = time.time()
    result_tta = engine_ema.run_tta(image)
    time_tta = time.time() - start
    cv2.imwrite(str(output_dir / 'ema_tta.png'), result_tta)
    
    # Summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Regular:    {time_regular:.3f}s -> {output_dir / 'regular.png'}")
    print(f"EMA:        {time_ema:.3f}s -> {output_dir / 'ema.png'}")
    print(f"EMA + TTA:  {time_tta:.3f}s -> {output_dir / 'ema_tta.png'}")
    print(f"\nTTA overhead: {time_tta / time_ema:.1f}x slower")
    print(f"Expected quality gain: +0.2-0.4 dB with TTA")


def benchmark_inference(
    checkpoint_path: str,
    model_type: str = 'span',
    device: str = 'cuda',
    num_runs: int = 100,
):
    """Benchmark inference speed."""
    engine = InferenceEngine.from_checkpoint(
        checkpoint_path, model_type, device, use_ema=True
    )
    
    results = engine.benchmark(num_runs=num_runs)
    
    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)
    print(f"Mean time:   {results['mean_time_ms']:.2f} ms")
    print(f"Std dev:     {results['std_time_ms']:.2f} ms")
    print(f"FPS:         {results['fps']:.2f}")
    print(f"Resolution:  {results['output_resolution']}")


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Inference with EMA, TTA, and Model Soup'
    )
    
    # Mode selection
    parser.add_argument('--create-soup', action='store_true',
                        help='Create model soup from checkpoints')
    parser.add_argument('--compare', action='store_true',
                        help='Compare regular vs EMA vs TTA')
    parser.add_argument('--benchmark', action='store_true',
                        help='Benchmark inference speed')
    
    # Paths
    parser.add_argument('--checkpoint', type=str,
                        help='Path to model checkpoint')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory containing checkpoints (for soup creation)')
    parser.add_argument('--input', '-i', type=str,
                        help='Input image path')
    parser.add_argument('--output', '-o', type=str,
                        help='Output image path')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for compare mode')
    
    # Model options
    parser.add_argument('--model-type', type=str, default='span',
                        choices=['span', 'mamba_pan', 'tiny_student'],
                        help='Model architecture type')
    parser.add_argument('--no-ema', action='store_true',
                        help='Do not use EMA weights (use regular weights)')
    parser.add_argument('--tta', action='store_true',
                        help='Use Test-Time Augmentation')
    
    # Soup options
    parser.add_argument('--last-n', type=int, default=5,
                        help='Use last N checkpoints for soup')
    parser.add_argument('--pattern', type=str, default='epoch_*.pth',
                        help='Glob pattern for checkpoint files')
    
    # Other options
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for inference')
    parser.add_argument('--num-runs', type=int, default=100,
                        help='Number of benchmark runs')
    
    args = parser.parse_args()
    
    # Execute based on mode
    if args.create_soup:
        if not args.checkpoint_dir:
            parser.error('--checkpoint-dir required for soup creation')
        output_path = args.output or 'checkpoints/soup.pth'
        create_soup_from_directory(
            args.checkpoint_dir,
            output_path,
            args.last_n,
            args.pattern,
            prefer_ema=not args.no_ema
        )
    
    elif args.compare:
        if not args.checkpoint or not args.input:
            parser.error('--checkpoint and --input required for comparison')
        compare_methods(
            args.checkpoint,
            args.input,
            args.output_dir,
            args.model_type,
            args.device
        )
    
    elif args.benchmark:
        if not args.checkpoint:
            parser.error('--checkpoint required for benchmarking')
        benchmark_inference(
            args.checkpoint,
            args.model_type,
            args.device,
            args.num_runs
        )
    
    else:
        # Standard inference
        if not args.checkpoint or not args.input or not args.output:
            parser.error('--checkpoint, --input, and --output required for inference')
        run_inference(
            args.checkpoint,
            args.input,
            args.output,
            args.model_type,
            use_ema=not args.no_ema,
            use_tta=args.tta,
            device=args.device
        )


if __name__ == '__main__':
    main()
