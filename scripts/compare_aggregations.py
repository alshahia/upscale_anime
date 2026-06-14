#!/usr/bin/env python3
"""
Benchmark and Compare Aggregation Architectures
Tests Simple, Adaptive, and Multi-scale aggregations.
"""
import sys
import argparse
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn as nn
from tqdm import tqdm

from distillation.mtkd import create_aggregation_network


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare Aggregation Architectures'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for testing'
    )
    
    parser.add_argument(
        '--image-size',
        type=int,
        default=256,
        help='HR image size (LR will be 1/4)'
    )
    
    parser.add_argument(
        '--num-teachers',
        type=int,
        default=3,
        help='Number of teachers'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Number of iterations for timing'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    
    return parser.parse_args()


def benchmark_aggregation(agg_type, config, device, batch_size, image_size, num_iterations):
    """
    Benchmark a single aggregation type.
    
    Returns:
        Dict with results: time, memory, params
    """
    print(f"\nBenchmarking {agg_type}...")
    
    # Create model
    model = create_aggregation_network(config)
    model.to(device)
    model.eval()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    
    # Create dummy inputs
    hr_size = image_size
    teacher_outputs = [
        torch.randn(batch_size, 3, hr_size, hr_size, device=device)
        for _ in range(config['training']['stage1']['teachers'].__len__())
    ]
    
    target = torch.randn(batch_size, 3, hr_size, hr_size, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(teacher_outputs)
    
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Benchmark forward pass
    times = []
    with torch.no_grad():
        for _ in tqdm(range(num_iterations), desc=f"{agg_type} forward"):
            start = time.time()
            output = model(teacher_outputs)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - start)
    
    # Benchmark backward pass
    model.train()
    times_backward = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for _ in tqdm(range(num_iterations), desc=f"{agg_type} backward"):
        teacher_outputs = [
            torch.randn(batch_size, 3, hr_size, hr_size, device=device, requires_grad=False)
            for _ in range(config['training']['stage1']['teachers'].__len__())
        ]
        
        start = time.time()
        optimizer.zero_grad()
        output = model(teacher_outputs)
        loss = nn.functional.l1_loss(output, target)
        loss.backward()
        optimizer.step()
        if device == 'cuda':
            torch.cuda.synchronize()
        times_backward.append(time.time() - start)
    
    # Get memory usage
    memory_mb = 0
    if device == 'cuda':
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    return {
        'type': agg_type,
        'params': num_params,
        'forward_time': sum(times) / len(times) * 1000,  # ms
        'backward_time': sum(times_backward) / len(times_backward) * 1000,  # ms
        'memory_mb': memory_mb,
        'output_shape': output.shape,
    }


def print_results(results):
    """Print benchmark results."""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    # Header
    print(f"\n{'Type':<15} {'Params':<12} {'Forward (ms)':<15} {'Backward (ms)':<15} {'Memory (MB)':<12}")
    print("-"*80)
    
    # Results
    for r in results:
        print(f"{r['type']:<15} {r['params']:<12,} {r['forward_time']:<15.2f} {r['backward_time']:<15.2f} {r['memory_mb']:<12.1f}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    # Find fastest
    fastest = min(results, key=lambda x: x['forward_time'])
    print(f"\nFastest Forward:  {fastest['type']}")
    
    # Find most memory efficient
    if any(r['memory_mb'] > 0 for r in results):
        most_efficient = min(results, key=lambda x: x['memory_mb'] if x['memory_mb'] > 0 else float('inf'))
        print(f"Most Memory Efficient: {most_efficient['type']}")
    
    # Parameter count
    smallest = min(results, key=lambda x: x['params'])
    print(f"Fewest Parameters: {smallest['type']}")
    
    print("\n" + "="*80)


def main():
    """Main function."""
    args = parse_args()
    
    print("="*80)
    print("AGGREGATION ARCHITECTURE BENCHMARK")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Image size: {args.image_size}x{args.image_size} (HR)")
    print(f"  Num teachers: {args.num_teachers}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Device: {args.device}")
    
    # Base config
    base_config = {
        'training': {
            'stage1': {
                'num_blocks': 6,
                'embed_dim': 96,
                'teachers': [{'name': f't{i}'} for i in range(args.num_teachers)],
            }
        },
        'model': {'scale': 4}
    }
    
    # Benchmark each aggregation type
    results = []
    
    for agg_type in ['simple', 'adaptive', 'multiscale']:
        config = base_config.copy()
        config['training'] = base_config['training'].copy()
        config['training']['stage1'] = base_config['training']['stage1'].copy()
        config['training']['stage1']['aggregation_type'] = agg_type
        
        if agg_type == 'multiscale':
            config['training']['stage1']['scale_factors'] = [1, 2]
        
        try:
            result = benchmark_aggregation(
                agg_type,
                config,
                args.device,
                args.batch_size,
                args.image_size,
                args.iterations
            )
            results.append(result)
        except Exception as e:
            print(f"✗ Failed to benchmark {agg_type}: {e}")
    
    # Print results
    if results:
        print_results(results)
        
        # Save results to file
        output_file = Path('aggregation_benchmark.txt')
        with open(output_file, 'w') as f:
            f.write("Aggregation Architecture Benchmark Results\n")
            f.write("="*80 + "\n\n")
            f.write(f"Configuration: batch_size={args.batch_size}, "
                   f"image_size={args.image_size}, "
                   f"num_teachers={args.num_teachers}, "
                   f"device={args.device}\n\n")
            
            for r in results:
                f.write(f"{r['type']}:\n")
                f.write(f"  Parameters: {r['params']:,}\n")
                f.write(f"  Forward time: {r['forward_time']:.2f} ms\n")
                f.write(f"  Backward time: {r['backward_time']:.2f} ms\n")
                f.write(f"  Memory: {r['memory_mb']:.1f} MB\n\n")
        
        print(f"\nResults saved to: {output_file.absolute()}")
    else:
        print("\n✗ No successful benchmarks")


if __name__ == '__main__':
    main()
