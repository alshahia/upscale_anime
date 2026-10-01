#!/usr/bin/env python
"""
Benchmark data loading speed and GPU utilization.

Usage:
    python scripts/benchmark_data_loading.py --config configs/benchmark_data_loading.yaml
    python scripts/benchmark_data_loading.py --config configs/stage1_fast.yaml --duration 60
"""
import argparse
import time
import torch
import sys
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import io

# Add src directory to path for proper imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Suppress tqdm progress bars during benchmark
import os
os.environ['TQDM_DISABLE'] = '1'

from anime_sr.data.dataloader import DataLoaderFactory
from anime_sr.utils.config import Config


def benchmark_data_loading(config_path, duration_seconds=60, batch_limit=None):
    """
    Benchmark data loading for specified duration.
    
    Args:
        config_path: Path to config file
        duration_seconds: How long to run benchmark
        batch_limit: Max batches to process (optional)
    """
    print(f"\n{'='*70}")
    print(f"Data Loading Benchmark")
    print(f"{'='*70}")
    print(f"Config: {config_path}")
    print(f"Duration: {duration_seconds}s")
    print(f"{'='*70}\n")
    
    # Load config
    config = Config.load(config_path)
    
    # Create dataloader
    train_loader = DataLoaderFactory.create(config, is_train=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"DataLoader: {len(train_loader)} batches")
    print(f"Batch size: {config.get('training', {}).get('batch_size', 8)}")
    print()
    
    # Warmup (first few batches often slower due to initialization)
    print("Warming up...")
    for i, batch in enumerate(train_loader):
        _ = batch['lr'].to(device, non_blocking=True)
        _ = batch['hr'].to(device, non_blocking=True)
        if i >= 5:
            break
    
    # Benchmark
    print(f"Benchmarking for {duration_seconds} seconds...\n")
    
    num_batches = 0
    total_load_time = 0
    total_transfer_time = 0
    start_time = time.time()
    
    for batch in train_loader:
        batch_start = time.time()
        
        # Time data loading (already happened when we got batch)
        load_time = time.time() - batch_start
        
        # Time GPU transfer
        transfer_start = time.time()
        lr = batch['lr'].to(device, non_blocking=True)
        hr = batch['hr'].to(device, non_blocking=True)
        
        # Simulate minimal GPU work to force synchronization
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        transfer_time = time.time() - transfer_start
        
        total_load_time += load_time
        total_transfer_time += transfer_time
        num_batches += 1
        
        # Check time limit
        elapsed = time.time() - start_time
        if elapsed > duration_seconds:
            break
            
        if batch_limit and num_batches >= batch_limit:
            break
    
    total_time = time.time() - start_time
    
    # Report
    print(f"\n{'='*70}")
    print(f"Results")
    print(f"{'='*70}")
    print(f"Total time:          {total_time:.2f}s")
    print(f"Batches processed:   {num_batches}")
    print(f"Throughput:          {num_batches/total_time:.2f} batches/sec")
    print(f"Avg batch time:      {total_time/num_batches*1000:.2f} ms")
    print(f"Avg transfer time:   {total_transfer_time/num_batches*1000:.2f} ms")
    print(f"{'='*70}\n")
    
    # GPU utilization check
    if device.type == 'cuda':
        print("GPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"  Reserved:  {torch.cuda.memory_reserved()/1024**3:.2f} GB")
        print()
    
    return {
        'total_time': total_time,
        'batches': num_batches,
        'throughput': num_batches/total_time,
        'avg_batch_ms': total_time/num_batches*1000,
    }


def compare_configs(config_paths, duration=30):
    """Compare multiple configurations."""
    print(f"\n{'='*70}")
    print(f"Comparing {len(config_paths)} configurations")
    print(f"{'='*70}\n")
    
    results = []
    for config_path in config_paths:
        name = Path(config_path).stem
        print(f"\nTesting: {name}")
        print("-" * 70)
        result = benchmark_data_loading(config_path, duration)
        result['name'] = name
        results.append(result)
    
    # Summary table
    print(f"\n{'='*70}")
    print(f"Comparison Summary")
    print(f"{'='*70}")
    print(f"{'Config':<30} {'Throughput':>12} {'Avg Time':>12}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['name']:<30} {r['throughput']:>12.2f} {r['avg_batch_ms']:>12.2f}")
    
    print(f"{'='*70}\n")
    
    # Find best
    best = max(results, key=lambda x: x['throughput'])
    print(f"Best configuration: {best['name']}")
    print(f"  Throughput: {best['throughput']:.2f} batches/sec")
    print()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark data loading performance')
    parser.add_argument('--config', '-c', type=str, help='Path to config file')
    parser.add_argument('--duration', '-d', type=int, default=60, help='Benchmark duration in seconds')
    parser.add_argument('--compare', nargs='+', help='Compare multiple configs')
    parser.add_argument('--batches', '-b', type=int, help='Max batches to process')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_configs(args.compare, args.duration)
    elif args.config:
        benchmark_data_loading(args.config, args.duration, args.batches)
    else:
        # Default benchmark with base config
        benchmark_data_loading('configs/benchmark_data_loading.yaml', args.duration, args.batches)


if __name__ == '__main__':
    main()
