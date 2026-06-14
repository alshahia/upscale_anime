#!/usr/bin/env python3
"""
Performance benchmarking script for the anime super-resolution project.
Tests model performance, memory usage, and optimization effectiveness.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.span.checkpoint_compatible_exact import CheckpointCompatibleSPANExact
from utils.performance_optimizer import PerformanceOptimizer, optimize_model_for_inference, benchmark_model_performance


def benchmark_model_performance():
    """Benchmark model performance across different scenarios."""
    print("[LAUNCH] Starting Performance Benchmark")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = PerformanceOptimizer()
    
    # Create model
    print("[PKG] Creating model...")
    model = CheckpointCompatibleSPANExact(
        scale=4, channels=48, hidden_channels=96, num_blocks=6
    )
    
    # Get baseline performance
    print("[CHART] Measuring baseline performance...")
    baseline_results = {}
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    baseline_results["model_info"] = {
        "total_parameters": total_params,
        "parameter_size_mb": total_params * 4 / 1024 / 1024,
        "model_type": "CheckpointCompatibleSPANExact"
    }
    
    # Memory usage
    baseline_memory = optimizer.get_memory_info()
    baseline_results["baseline_memory"] = baseline_memory
    
    # Benchmark different input sizes
    input_sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]
    baseline_benchmark = optimizer.benchmark_model(model, input_sizes, num_runs=5)
    baseline_results["baseline_benchmark"] = baseline_benchmark
    
    print("[OK] Baseline performance measured")
    
    # Test optimizations
    print("\n[CONFIG] Testing optimizations...")
    
    # 1. Memory optimization
    print("  [UP] Testing memory optimization...")
    optimized_model = optimizer.optimize_model_memory(model)
    memory_optimized_results = optimizer.benchmark_model(optimized_model, input_sizes, num_runs=5)
    
    # 2. Speed optimization
    print("  [FAST] Testing speed optimization...")
    speed_optimized_model = optimizer.optimize_inference_speed(model)
    speed_optimized_results = optimizer.benchmark_model(speed_optimized_model, input_sizes, num_runs=5)
    
    # 3. Full optimization
    print("  [TARGET] Testing full optimization...")
    fully_optimized = optimize_model_for_inference(model)
    fully_optimized_results = optimizer.benchmark_model(fully_optimized, input_sizes, num_runs=5)
    
    print("[OK] Optimization tests completed")
    
    # Batch size optimization
    print("\n[PKG] Testing batch size optimization...")
    input_size = (256, 256)
    optimal_batch_size = optimizer.auto_tune_batch_size(model, input_size, memory_limit_mb=8000)
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8, optimal_batch_size]
    batch_results = {}
    
    for batch_size in batch_sizes:
        if batch_size > optimal_batch_size:
            continue
            
        try:
            result = optimizer.optimize_batch_processing(model, batch_size, input_size)
            batch_results[f"batch_{batch_size}"] = result
            print(f"  Batch size {batch_size}: {result.get('single_batch', {}).get('time', 'N/A'):.3f}s")
        except Exception as e:
            print(f"  Batch size {batch_size}: Failed - {e}")
    
    print("[OK] Batch size optimization completed")
    
    # Performance monitoring
    print("\n[UP] Testing performance monitoring...")
    monitoring_results = optimizer.monitor_performance(model, duration_seconds=10)
    
    print("[OK] Performance monitoring completed")
    
    # Compile results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline": baseline_results,
        "optimizations": {
            "memory_optimized": memory_optimized_results,
            "speed_optimized": speed_optimized_results,
            "fully_optimized": fully_optimized_results
        },
        "batch_optimization": {
            "optimal_batch_size": optimal_batch_size,
            "batch_results": batch_results
        },
        "monitoring": monitoring_results
    }
    
    # Calculate improvements
    print("\n[CHART] Calculating performance improvements...")
    improvements = calculate_improvements(baseline_benchmark, fully_optimized_results)
    results["improvements"] = improvements
    
    # Save results
    output_file = "performance_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("[CHART] PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 50)
    
    print(f"Model Parameters: {total_params:,}")
    print(f"Optimal Batch Size: {optimal_batch_size}")
    print(f"Baseline Memory: {baseline_memory.get('system_memory_mb', 0):.1f}MB")
    
    if improvements:
        print(f"\n[LAUNCH] Performance Improvements:")
        for size, improvement in improvements.items():
            if improvement.get('speed_improvement'):
                speed_imp = improvement['speed_improvement']
                print(f"  {size}: {speed_imp:.1f}% faster")
            if improvement.get('memory_improvement'):
                mem_imp = improvement['memory_improvement']
                print(f"  {size}: {mem_imp:.1f}% less memory")
    
    print(f"\n[UP] Monitoring Results:")
    print(f"  Samples/sec: {monitoring_results.get('avg_samples_per_second', 0):.1f}")
    print(f"  Avg Memory: {monitoring_results.get('avg_memory_mb', 0):.1f}MB")
    
    return results


def calculate_improvements(baseline: Dict[str, Any], optimized: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate performance improvements between baseline and optimized results."""
    improvements = {}
    
    for i, size in enumerate(baseline.get("input_sizes", [])):
        if i < len(optimized.get("input_sizes", [])):
            baseline_time = baseline.get("forward_times", [])[i] if i < len(baseline.get("forward_times", [])) else 0
            optimized_time = optimized.get("forward_times", [])[i] if i < len(optimized.get("forward_times", [])) else 0
            
            baseline_memory = baseline.get("memory_usage", [])[i] if i < len(baseline.get("memory_usage", [])) else 0
            optimized_memory = optimized.get("memory_usage", [])[i] if i < len(optimized.get("memory_usage", [])) else 0
            
            improvement = {}
            
            if baseline_time > 0:
                speed_improvement = ((baseline_time - optimized_time) / baseline_time) * 100
                improvement["speed_improvement"] = speed_improvement
            
            if baseline_memory > 0:
                memory_improvement = ((baseline_memory - optimized_memory) / baseline_memory) * 100
                improvement["memory_improvement"] = memory_improvement
            
            improvements[size] = improvement
    
    return improvements


def test_memory_optimization():
    """Test memory optimization specifically."""
    print("🧠 Testing Memory Optimization")
    print("=" * 30)
    
    import torch
    optimizer = PerformanceOptimizer()
    
    # Create model
    model = CheckpointCompatibleSPANExact(
        scale=4, channels=48, hidden_channels=96, num_blocks=6
    )
    
    # Test memory before optimization
    print("[CHART] Measuring memory usage before optimization...")
    memory_before = optimizer.get_memory_info()
    print(f"  System Memory: {memory_before.get('system_memory_mb', 0):.1f}MB")
    if torch.cuda.is_available():
        print(f"  GPU Memory: {memory_before.get('gpu_memory_allocated_mb', 0):.1f}MB")
    
    # Apply memory optimization
    print("[CONFIG] Applying memory optimization...")
    optimized_model = optimizer.optimize_model_memory(model)
    
    # Test memory after optimization
    print("[CHART] Measuring memory usage after optimization...")
    memory_after = optimizer.get_memory_info()
    print(f"  System Memory: {memory_after.get('system_memory_mb', 0):.1f}MB")
    if torch.cuda.is_available():
        print(f"  GPU Memory: {memory_after.get('gpu_memory_allocated_mb', 0):.1f}MB")
    
    # Test with forward pass
    print("[LAUNCH] Testing forward pass memory usage...")
    input_tensor = torch.randn(1, 3, 256, 256)
    
    memory_before_forward = optimizer.get_memory_info()
    with torch.no_grad():
        output = optimized_model(input_tensor)
    memory_after_forward = optimizer.get_memory_info()
    
    if torch.cuda.is_available():
        forward_memory = memory_after_forward.get('gpu_memory_allocated_mb', 0) - memory_before_forward.get('gpu_memory_allocated_mb', 0)
    else:
        forward_memory = memory_after_forward.get('system_memory_mb', 0) - memory_before_forward.get('system_memory_mb', 0)
    
    print(f"  Forward Pass Memory: {forward_memory:.1f}MB")
    print(f"  Output Shape: {output.shape}")
    
    # Clean up
    optimizer.optimize_memory_usage()
    
    print("[OK] Memory optimization test completed")


def test_batch_processing():
    """Test batch processing optimization."""
    print("[PKG] Testing Batch Processing")
    print("=" * 30)
    
    import torch
    optimizer = PerformanceOptimizer()
    
    # Create model
    model = CheckpointCompatibleSPANExact(
        scale=4, channels=48, hidden_channels=96, num_blocks=6
    )
    
    # Test different batch sizes
    input_size = (256, 256)
    batch_sizes = [1, 2, 4, 8]
    
    print(f"[CHART] Testing batch processing for input size {input_size}:")
    
    for batch_size in batch_sizes:
        try:
            result = optimizer.optimize_batch_processing(model, batch_size, input_size)
            
            if "single_batch" in result:
                single_batch = result["single_batch"]
                print(f"  Batch {batch_size}: {single_batch.get('time', 'N/A'):.3f}s, {single_batch.get('memory', 'N/A'):.1f}MB")
            
            if "gradient_accumulation" in result:
                grad_acc = result["gradient_accumulation"]
                print(f"    Gradient Accum: {grad_acc.get('time', 'N/A'):.3f}s, {grad_acc.get('memory', 'N/A'):.1f}MB")
                
        except Exception as e:
            print(f"  Batch {batch_size}: Failed - {e}")
    
    # Find optimal batch size
    print("\n[TARGET] Finding optimal batch size...")
    optimal_size = optimizer.auto_tune_batch_size(model, input_size, memory_limit_mb=8000)
    print(f"  Optimal Batch Size: {optimal_size}")
    
    print("[OK] Batch processing test completed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Performance benchmarking script")
    parser.add_argument("--test", choices=["full", "memory", "batch", "monitor"], 
                       default="full", help="Specific test to run")
    parser.add_argument("--output", default="performance_benchmark_results.json", 
                       help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Set up logging
    import logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Import torch here to avoid issues
    import torch
    
    print(f"[LAUNCH] Performance Benchmarking Script")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"PyTorch Version: {torch.__version__}")
    print("=" * 50)
    
    try:
        if args.test == "full":
            results = benchmark_model_performance()
        elif args.test == "memory":
            test_memory_optimization()
        elif args.test == "batch":
            test_batch_processing()
        elif args.test == "monitor":
            optimizer = PerformanceOptimizer()
            model = CheckpointCompatibleSPANExact(scale=4, channels=48, hidden_channels=96, num_blocks=6)
            results = optimizer.monitor_performance(model, duration_seconds=30)
            print(f"[UP] Monitoring Results: {results}")
        
        print("\n[OK] All tests completed successfully!")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
