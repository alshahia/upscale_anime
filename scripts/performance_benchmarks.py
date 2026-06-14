#!/usr/bin/env python3
"""
Comprehensive performance benchmarks for the SPAN training system.

Benchmarks various aspects of the implementation:
- Model performance (forward pass speed, memory usage)
- Training performance (steps per second, gradient computation)
- Checkpoint loading performance
- Loss computation performance
- Monitoring overhead
- Comparison between different configurations
"""

import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

import torch
import torch.nn as nn
import time
import psutil
import gc
from typing import Dict, List, Any, Tuple
import yaml
from pathlib import Path
import json


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking system.
    """
    
    def __init__(self, device: str = 'auto'):
        self.device = torch.device('cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu')
        self.results = {}
        
        # System info
        self.system_info = {
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'python_version': sys.version,
            'torch_version': torch.__version__
        }
        
        if torch.cuda.is_available():
            self.system_info.update({
                'gpu_name': torch.cuda.get_device_name(),
                'gpu_memory': torch.cuda.get_device_properties(0).total_memory,
                'cuda_version': torch.version.cuda
            })
    
    def benchmark_model_performance(self, model: nn.Module, input_shape: Tuple[int, ...], num_runs: int = 100) -> Dict[str, Any]:
        """
        Benchmark model forward pass performance.
        
        Args:
            model: Model to benchmark
            input_shape: Input tensor shape
            num_runs: Number of benchmark runs
            
        Returns:
            Performance metrics
        """
        print(f"Benchmarking model: {type(model).__name__}")
        
        model = model.to(self.device)
        model.eval()
        
        # Create test input
        test_input = torch.randn(*input_shape).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_input)
        
        # Benchmark forward pass
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                output = model(test_input)
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        end_time = time.time()
        
        forward_time = (end_time - start_time) / num_runs
        
        # Memory usage
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(test_input)
            peak_memory = torch.cuda.max_memory_allocated()
        else:
            peak_memory = 0
        
        # Calculate throughput
        batch_size = input_shape[0]
        throughput = batch_size / forward_time
        
        results = {
            'model_name': type(model).__name__,
            'input_shape': input_shape,
            'forward_time_ms': forward_time * 1000,
            'throughput_samples_per_sec': throughput,
            'peak_memory_mb': peak_memory / (1024**2),
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'flops_estimate': self._estimate_flops(model, input_shape)
        }
        
        print(f"  Forward time: {results['forward_time_ms']:.2f}ms")
        print(f"  Throughput: {results['throughput_samples_per_sec']:.1f} samples/sec")
        print(f"  Peak memory: {results['peak_memory_mb']:.1f}MB")
        
        return results
    
    def benchmark_training_step(self, model: nn.Module, input_shape: Tuple[int, ...], num_runs: int = 50) -> Dict[str, Any]:
        """
        Benchmark complete training step performance.
        
        Args:
            model: Model to benchmark
            input_shape: Input tensor shape
            num_runs: Number of benchmark runs
            
        Returns:
            Training performance metrics
        """
        print(f"Benchmarking training step for: {type(model).__name__}")
        
        model = model.to(self.device)
        model.train()
        
        # Setup optimizer and loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_fn = nn.L1Loss()
        
        # Create test data
        lr_input = torch.randn(*input_shape).to(self.device)
        hr_target = torch.randn(input_shape[0], input_shape[1], input_shape[2]*4, input_shape[3]*4).to(self.device)
        
        # Warmup
        for _ in range(5):
            optimizer.zero_grad()
            output = model(lr_input)
            loss = loss_fn(output, hr_target)
            loss.backward()
            optimizer.step()
        
        # Benchmark training step
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = time.time()
        
        for _ in range(num_runs):
            optimizer.zero_grad()
            output = model(lr_input)
            loss = loss_fn(output, hr_target)
            loss.backward()
            
            # Apply gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        end_time = time.time()
        
        step_time = (end_time - start_time) / num_runs
        
        results = {
            'model_name': type(model).__name__,
            'input_shape': input_shape,
            'step_time_ms': step_time * 1000,
            'throughput_steps_per_sec': 1.0 / step_time,
            'forward_ratio': 0.6,  # Approximate forward pass time ratio
            'backward_ratio': 0.4,  # Approximate backward pass time ratio
        }
        
        print(f"  Step time: {results['step_time_ms']:.2f}ms")
        print(f"  Throughput: {results['throughput_steps_per_sec']:.1f} steps/sec")
        
        return results
    
    def benchmark_checkpoint_loading(self, model: nn.Module, checkpoint_path: str, num_runs: int = 10) -> Dict[str, Any]:
        """
        Benchmark checkpoint loading performance.
        
        Args:
            model: Model to load checkpoint into
            checkpoint_path: Path to checkpoint file
            num_runs: Number of benchmark runs
            
        Returns:
            Checkpoint loading metrics
        """
        print(f"Benchmarking checkpoint loading: {checkpoint_path}")
        
        if not Path(checkpoint_path).exists():
            return {'error': f'Checkpoint not found: {checkpoint_path}'}
        
        # Load checkpoint once to get size
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except Exception:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        checkpoint_size = Path(checkpoint_path).stat().st_size
        
        results = {
            'checkpoint_path': checkpoint_path,
            'checkpoint_size_mb': checkpoint_size / (1024**2),
            'parameter_count': len(checkpoint.get('params', checkpoint)),
            'loading_times': []
        }
        
        for run in range(num_runs):
            # Create fresh model
            fresh_model = type(model)()
            
            # Benchmark loading
            start_time = time.time()
            
            try:
                from utils.checkpoint_loader import load_checkpoint_compatible_span
                success, loading_info = load_checkpoint_compatible_span(
                    fresh_model, checkpoint_path, verbose=False
                )
                
                end_time = time.time()
                loading_time = end_time - start_time
                
                results['loading_times'].append(loading_time)
                
                if run == 0:
                    results['loading_success'] = success
                    results['load_percentage'] = loading_info.get('load_percentage', 0)
                    
            except Exception as e:
                results['loading_times'].append(float('inf'))
                if run == 0:
                    results['loading_error'] = str(e)
        
        # Calculate statistics
        valid_times = [t for t in results['loading_times'] if t != float('inf')]
        if valid_times:
            results['avg_loading_time_s'] = sum(valid_times) / len(valid_times)
            results['min_loading_time_s'] = min(valid_times)
            results['max_loading_time_s'] = max(valid_times)
        
        print(f"  Checkpoint size: {results['checkpoint_size_mb']:.1f}MB")
        print(f"  Loading time: {results.get('avg_loading_time_s', 'N/A'):.3f}s")
        print(f"  Load success: {results.get('loading_success', False)}")
        
        return results
    
    def benchmark_loss_computation(self, num_runs: int = 1000) -> Dict[str, Any]:
        """
        Benchmark loss computation performance.
        
        Args:
            num_runs: Number of benchmark runs
            
        Returns:
            Loss computation metrics
        """
        print("Benchmarking loss computation")
        
        # Create test data
        pred = torch.randn(2, 3, 256, 256).to(self.device)
        target = torch.randn(2, 3, 256, 256).to(self.device)
        
        # Test different loss functions
        losses = {
            'L1Loss': nn.L1Loss(),
            'MSELoss': nn.MSELoss(),
            'SmoothL1Loss': nn.SmoothL1Loss()
        }
        
        results = {}
        
        for loss_name, loss_fn in losses.items():
            loss_fn = loss_fn.to(self.device)
            
            # Warmup
            for _ in range(10):
                _ = loss_fn(pred, target)
            
            # Benchmark
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            start_time = time.time()
            
            for _ in range(num_runs):
                loss = loss_fn(pred, target)
            
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_runs
            
            results[loss_name] = {
                'avg_time_ms': avg_time * 1000,
                'throughput_ops_per_sec': 1.0 / avg_time
            }
            
            print(f"  {loss_name}: {avg_time * 1000:.3f}ms")
        
        return results
    
    def benchmark_monitoring_overhead(self, num_runs: int = 100) -> Dict[str, Any]:
        """
        Benchmark monitoring system overhead.
        
        Args:
            num_runs: Number of benchmark runs
            
        Returns:
            Monitoring overhead metrics
        """
        print("Benchmarking monitoring overhead")
        
        from utils.training_monitor import TrainingMonitor
        from utils.nan_handler import NaNHandler
        
        # Create monitoring components
        monitor = TrainingMonitor(verbose=False)
        nan_handler = NaNHandler(enabled=False)  # Disable to reduce noise
        
        # Benchmark monitoring overhead
        overhead_times = []
        
        for run in range(num_runs):
            start_time = time.time()
            
            # Simulate monitoring operations
            monitor.log_step(
                step=run,
                loss=0.5 + run * 0.001,
                gradient_norm=0.1 + run * 0.0001,
                learning_rate=1e-4
            )
            
            nan_handler.detect_nan_in_tensor(torch.randn(100), "test")
            
            end_time = time.time()
            overhead_times.append(end_time - start_time)
        
        avg_overhead = sum(overhead_times) / len(overhead_times)
        
        results = {
            'avg_overhead_us': avg_overhead * 1000000,
            'overhead_percentage': (avg_overhead / 0.01) * 100,  # Assuming 10ms step time
            'total_runs': num_runs
        }
        
        print(f"  Average overhead: {avg_overhead * 1000000:.1f}μs")
        print(f"  Overhead percentage: {results['overhead_percentage']:.2f}%")
        
        return results
    
    def _estimate_flops(self, model: nn.Module, input_shape: Tuple[int, ...]) -> int:
        """
        Rough FLOPS estimation for the model.
        
        Args:
            model: Model to estimate FLOPS for
            input_shape: Input tensor shape
            
        Returns:
            Estimated FLOPS
        """
        # This is a very rough estimation
        # For accurate FLOPS, you would use specialized tools like fvcore
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Rough estimation: 2 FLOPs per parameter for forward pass
        # This is highly inaccurate but gives a ballpark figure
        estimated_flops = total_params * 2
        
        return estimated_flops
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """
        Run comprehensive benchmark suite.
        
        Returns:
            Complete benchmark results
        """
        print("COMPREHENSIVE PERFORMANCE BENCHMARK")
        print("=" * 60)
        
        results = {
            'system_info': self.system_info,
            'timestamp': time.time(),
            'benchmarks': {}
        }
        
        try:
            # Test different models
            from models.span import create_span_model
            
            # Checkpoint-compatible model
            config = {
                'type': 'checkpoint_compatible',
                'scale': 4,
                'channels': 48,
                'hidden_channels': 96,
                'num_blocks': 6
            }
            
            model = create_span_model(config)
            
            # Model performance benchmarks
            results['benchmarks']['model_performance'] = self.benchmark_model_performance(
                model, (2, 3, 64, 64), num_runs=50
            )
            
            # Training step benchmarks
            results['benchmarks']['training_performance'] = self.benchmark_training_step(
                model, (2, 3, 64, 64), num_runs=25
            )
            
            # Checkpoint loading benchmarks
            checkpoint_path = 'checkpoints/span/spanx4_ch48.pth'
            if Path(checkpoint_path).exists():
                results['benchmarks']['checkpoint_loading'] = self.benchmark_checkpoint_loading(
                    model, checkpoint_path, num_runs=5
                )
            
            # Loss computation benchmarks
            results['benchmarks']['loss_computation'] = self.benchmark_loss_computation(num_runs=500)
            
            # Monitoring overhead benchmarks
            results['benchmarks']['monitoring_overhead'] = self.benchmark_monitoring_overhead(num_runs=50)
            
        except Exception as e:
            print(f"Benchmark failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str = None):
        """
        Save benchmark results to file.
        
        Args:
            results: Benchmark results
            output_path: Output file path
        """
        if output_path is None:
            output_path = f"performance_benchmark_{int(time.time())}.json"
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to: {output_path}")


def main():
    """Run comprehensive performance benchmarks."""
    print("PERFORMANCE BENCHMARK SUITE")
    print("=" * 60)
    
    # Create benchmark
    benchmark = PerformanceBenchmark()
    
    # Run benchmarks
    results = benchmark.run_comprehensive_benchmark()
    
    # Save results
    benchmark.save_results(results, "performance_benchmark_results.json")
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    if 'error' in results:
        print(f"[ERROR] Benchmark failed: {results['error']}")
        return False
    
    benchmarks = results['benchmarks']
    
    # Model performance
    if 'model_performance' in benchmarks:
        model_perf = benchmarks['model_performance']
        print(f"Model Performance:")
        print(f"  Forward time: {model_perf['forward_time_ms']:.2f}ms")
        print(f"  Throughput: {model_perf['throughput_samples_per_sec']:.1f} samples/sec")
        print(f"  Memory: {model_perf['peak_memory_mb']:.1f}MB")
    
    # Training performance
    if 'training_performance' in benchmarks:
        train_perf = benchmarks['training_performance']
        print(f"\nTraining Performance:")
        print(f"  Step time: {train_perf['step_time_ms']:.2f}ms")
        print(f"  Throughput: {train_perf['throughput_steps_per_sec']:.1f} steps/sec")
    
    # Checkpoint loading
    if 'checkpoint_loading' in benchmarks:
        ckpt_perf = benchmarks['checkpoint_loading']
        if 'avg_loading_time_s' in ckpt_perf:
            print(f"\nCheckpoint Loading:")
            print(f"  Loading time: {ckpt_perf['avg_loading_time_s']:.3f}s")
            print(f"  Load success: {ckpt_perf.get('loading_success', False)}")
    
    # Loss computation
    if 'loss_computation' in benchmarks:
        loss_perf = benchmarks['loss_computation']
        print(f"\nLoss Computation:")
        for loss_name, metrics in loss_perf.items():
            print(f"  {loss_name}: {metrics['avg_time_ms']:.3f}ms")
    
    # Monitoring overhead
    if 'monitoring_overhead' in benchmarks:
        mon_perf = benchmarks['monitoring_overhead']
        print(f"\nMonitoring Overhead:")
        print(f"  Overhead: {mon_perf['avg_overhead_us']:.1f}μs")
        print(f"  Percentage: {mon_perf['overhead_percentage']:.2f}%")
    
    print(f"\n[OK] Benchmark completed successfully")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
