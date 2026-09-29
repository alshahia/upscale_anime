"""
Performance optimization utilities for the anime super-resolution project.
Includes memory optimization, speed improvements, and resource management.
"""

import torch
import torch.nn as nn
import gc
import psutil
import os
import time
from typing import Dict, Any, Optional, List
import logging
from contextlib import contextmanager


logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """Comprehensive performance optimization utilities."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimization_history = []
        
    def optimize_model_memory(self, model: nn.Module) -> nn.Module:
        """Apply memory optimizations to model."""
        logger.info("Applying memory optimizations to model...")
        
        # Enable gradient checkpointing if available
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
        
        # Use fused operations where available
        self._enable_fused_operations(model)
        
        # Optimize buffer usage
        self._optimize_buffers(model)
        
        # Set model to use memory efficient attention
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            self._enable_memory_efficient_attention(model)
        
        return model
    
    def _enable_fused_operations(self, model: nn.Module):
        """Enable fused operations for better performance."""
        try:
            # Enable CUDA fused operations
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                logger.info("Enabled CUDA fused operations")
        except Exception as e:
            logger.warning(f"Could not enable fused operations: {e}")
    
    def _optimize_buffers(self, model: nn.Module):
        """Optimize model buffers for memory efficiency."""
        try:
            # Convert buffers to more memory-efficient formats
            for name, buffer in model.named_buffers():
                if buffer.dtype == torch.float32:
                    # Convert to half precision where safe
                    if 'weight' in name.lower() or 'bias' in name.lower():
                        continue  # Keep parameters in full precision
                    # Skip BatchNorm running statistics: converting these to
                    # half corrupts them and breaks normalization.
                    if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                        continue
                    buffer.data = buffer.data.half()
            
            logger.info("Optimized buffer data types")
        except Exception as e:
            logger.warning(f"Could not optimize buffers: {e}")
    
    def _enable_memory_efficient_attention(self, model: nn.Module):
        """Enable memory-efficient attention mechanisms."""
        try:
            # This would require model-specific implementation
            # For now, just log that we're looking for attention modules
            for name, module in model.named_modules():
                if 'attention' in name.lower() or 'attn' in name.lower():
                    logger.debug(f"Found attention module: {name}")
        except Exception as e:
            logger.warning(f"Could not enable memory-efficient attention: {e}")
    
    def optimize_inference_speed(self, model: nn.Module) -> nn.Module:
        """Apply speed optimizations for inference."""
        logger.info("Applying speed optimizations...")
        
        # Suppress compilation errors so torch.compile falls back to
        # eager mode when no C compiler is available (e.g. minimal Windows
        # installs without MSVC). Without this, the lazy compilation at
        # first forward pass raises "cl is not found".
        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            logger.info(f"suppress_errors set to: {torch._dynamo.config.suppress_errors}")
        except Exception as e:
            logger.warning(f"Failed to set suppress_errors: {e}")

        # Compile model if using PyTorch 2.0+
        original_model = model
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode="max-autotune")
                logger.info("Enabled model compilation (torch.compile)")
                
                # Test forward pass to verify compilation works.
                # torch.compile is lazy — compilation happens on first forward.
                # If no C compiler is available (e.g. minimal Windows installs
                # without MSVC), this raises "cl is not found". suppress_errors
                # should handle this, but as a safety net we test here and
                # fall back to the original model if compilation fails.
                try:
                    with torch.no_grad():
                        device = next(original_model.parameters()).device
                        dummy = torch.zeros(1, 3, 4, 4, device=device)
                        model(dummy)
                    logger.info("Model compilation verified with test forward pass")
                except Exception as e:
                    logger.warning(f"torch.compile test forward failed, falling back to eager: {e}")
                    model = original_model
            except Exception as e:
                logger.warning(f"Could not compile model: {e}")
                model = original_model
        
        # Set to evaluation mode
        model.eval()
        
        # Disable gradients
        for param in model.parameters():
            param.requires_grad = False
        
        # Enable inference mode optimizations
        self._enable_inference_mode(model)
        
        return model
    
    def _enable_inference_mode(self, model: nn.Module):
        """Enable inference mode optimizations."""
        try:
            # Use torch.inference_mode for better performance
            # This is a context manager, so we'll apply it during forward pass
            logger.info("Inference mode optimizations ready")
        except Exception as e:
            logger.warning(f"Could not enable inference mode: {e}")
    
    @contextmanager
    def inference_context(self, model: nn.Module):
        """Context manager for optimized inference."""
        # Enable optimizations
        original_training = model.training
        model.eval()
        
        # Use inference mode if available
        inference_mode = None
        if hasattr(torch, 'inference_mode'):
            inference_mode = torch.inference_mode()
            inference_mode.__enter__()
        
        try:
            yield model
        finally:
            # Restore original state
            model.train(original_training)
            if inference_mode is not None:
                inference_mode.__exit__(None, None, None)
    
    def optimize_memory_usage(self):
        """Optimize system memory usage."""
        logger.info("Optimizing system memory usage...")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")
        
        # Force garbage collection
        gc.collect()
        logger.info("Forced garbage collection")
        
        # Monitor memory usage
        memory_info = self.get_memory_info()
        logger.info(f"Memory usage after optimization: {memory_info}")
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information."""
        info = {}
        
        # System memory
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        info["system_memory_mb"] = memory_info.rss / 1024 / 1024
        info["system_memory_percent"] = process.memory_percent()
        
        # GPU memory
        if torch.cuda.is_available():
            info["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            info["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
            # The deprecated CUDA cache query was removed in modern PyTorch;
            # memory_reserved() reports the caching allocator's held memory.
            info["gpu_memory_cached_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        
        return info
    
    def benchmark_model(self, model: nn.Module, input_sizes: List[tuple], 
                       num_runs: int = 10) -> Dict[str, Any]:
        """Benchmark model performance across different input sizes."""
        logger.info(f"Benchmarking model with {len(input_sizes)} input sizes...")
        
        results = {
            "input_sizes": [],
            "forward_times": [],
            "memory_usage": [],
            "throughput": []
        }
        
        model.eval()
        
        for size in input_sizes:
            logger.info(f"Benchmarking input size: {size}")
            
            # Create input tensor
            input_tensor = torch.randn(1, 3, size[0], size[1], device=self.device)
            
            # Warm up
            with torch.no_grad():
                for _ in range(3):
                    _ = model(input_tensor)
            
            # Benchmark forward pass
            forward_times = []
            memory_usage = []
            
            for _ in range(num_runs):
                # Clear cache before each run
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Measure memory before
                memory_before = self.get_memory_info()
                
                # Time forward pass
                start_time = time.time()
                with torch.no_grad():
                    output = model(input_tensor)
                end_time = time.time()
                
                # Measure memory after
                memory_after = self.get_memory_info()
                
                forward_time = end_time - start_time
                forward_times.append(forward_time)
                
                # Calculate memory usage for this run
                if torch.cuda.is_available():
                    memory_used = memory_after.get("gpu_memory_allocated_mb", 0) - memory_before.get("gpu_memory_allocated_mb", 0)
                else:
                    memory_used = memory_after.get("system_memory_mb", 0) - memory_before.get("system_memory_mb", 0)
                
                memory_usage.append(memory_used)
            
            # Calculate statistics
            avg_forward_time = sum(forward_times) / len(forward_times)
            avg_memory_usage = sum(memory_usage) / len(memory_usage)
            throughput = 1.0 / avg_forward_time
            
            results["input_sizes"].append(f"{size[0]}x{size[1]}")
            results["forward_times"].append(avg_forward_time)
            results["memory_usage"].append(avg_memory_usage)
            results["throughput"].append(throughput)
            
            logger.info(f"  Size {size[0]}x{size[1]}: {avg_forward_time:.3f}s, {throughput:.1f} FPS, {avg_memory_usage:.1f}MB")
        
        return results
    
    def optimize_batch_processing(self, model: nn.Module, batch_size: int, 
                                 input_size: tuple) -> Dict[str, Any]:
        """Optimize batch processing for given batch size and input size."""
        logger.info(f"Optimizing batch processing: batch_size={batch_size}, input_size={input_size}")
        
        # Test different batch processing strategies
        strategies = {
            "single_batch": self._process_single_batch,
            "gradient_accumulation": self._process_with_gradient_accumulation,
        }
        
        results = {}
        
        for strategy_name, strategy_func in strategies.items():
            try:
                result = strategy_func(model, batch_size, input_size)
                results[strategy_name] = result
                logger.info(f"  {strategy_name}: {result['time']:.3f}s, {result['memory']:.1f}MB")
            except Exception as e:
                logger.warning(f"  {strategy_name}: Failed - {e}")
                results[strategy_name] = {"error": str(e)}
        
        return results
    
    def _process_single_batch(self, model: nn.Module, batch_size: int, input_size: tuple) -> Dict[str, Any]:
        """Process entire batch at once."""
        model.eval()
        
        # Create batch
        input_tensor = torch.randn(batch_size, 3, input_size[0], input_size[1], device=self.device)
        
        # Measure memory before
        memory_before = self.get_memory_info()
        
        # Process batch
        start_time = time.time()
        with torch.no_grad():
            output = model(input_tensor)
        end_time = time.time()
        
        # Measure memory after
        memory_after = self.get_memory_info()
        
        if torch.cuda.is_available():
            memory_used = memory_after.get("gpu_memory_allocated_mb", 0) - memory_before.get("gpu_memory_allocated_mb", 0)
        else:
            memory_used = memory_after.get("system_memory_mb", 0) - memory_before.get("system_memory_mb", 0)
        
        return {
            "time": end_time - start_time,
            "memory": memory_used,
            "output_shape": output.shape
        }
    
    def _process_with_gradient_accumulation(self, model: nn.Module, batch_size: int, input_size: tuple) -> Dict[str, Any]:
        """Process batch with gradient accumulation."""
        # This is mainly for training, but we can simulate it
        chunk_size = min(4, batch_size)  # Process in smaller chunks
        num_chunks = (batch_size + chunk_size - 1) // chunk_size
        
        model.eval()
        
        memory_before = self.get_memory_info()
        start_time = time.time()
        
        outputs = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, batch_size)
            
            chunk_input = torch.randn(end_idx - start_idx, 3, input_size[0], input_size[1], device=self.device)
            
            with torch.no_grad():
                chunk_output = model(chunk_input)
            outputs.append(chunk_output)
        
        # Concatenate outputs
        output = torch.cat(outputs, dim=0)
        end_time = time.time()
        
        memory_after = self.get_memory_info()
        
        if torch.cuda.is_available():
            memory_used = memory_after.get("gpu_memory_allocated_mb", 0) - memory_before.get("gpu_memory_allocated_mb", 0)
        else:
            memory_used = memory_after.get("system_memory_mb", 0) - memory_before.get("system_memory_mb", 0)
        
        return {
            "time": end_time - start_time,
            "memory": memory_used,
            "output_shape": output.shape,
            "chunks": num_chunks
        }
    

    def auto_tune_batch_size(self, model: nn.Module, input_size: tuple, 
                            max_batch_size: int = 32, memory_limit_mb: float = 8000) -> int:
        """Automatically find optimal batch size based on memory constraints."""
        logger.info(f"Auto-tuning batch size for input {input_size} with memory limit {memory_limit_mb}MB")
        
        optimal_batch_size = 1
        tested_sizes = []
        
        for batch_size in [1, 2, 4, 8, 16, 32]:
            if batch_size > max_batch_size:
                break
            
            try:
                # Test this batch size
                result = self._process_single_batch(model, batch_size, input_size)
                memory_used = result["memory"]
                
                tested_sizes.append((batch_size, memory_used))
                
                if memory_used <= memory_limit_mb:
                    optimal_batch_size = batch_size
                else:
                    break  # Memory limit exceeded
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    break  # OOM error, stop testing
                else:
                    raise e
        
        logger.info(f"Optimal batch size: {optimal_batch_size}")
        return optimal_batch_size
    
    def enable_mixed_precision(self, model: nn.Module) -> nn.Module:
        """Enable mixed precision for the model using torch.autocast.

        The model is kept in FP32; mixed precision is applied at forward-pass
        time via :meth:`autocast_context`. Converting modules to half instead
        would corrupt BatchNorm running statistics and lose precision on
        sensitive layers, so autocast keeps master weights in FP32 while
        running compute-heavy ops in FP16.
        """
        logger.info("Enabling mixed precision (torch.autocast)...")
        model._mixed_precision_enabled = True
        logger.info(
            "Mixed precision enabled. Wrap forward passes in "
            "optimizer.autocast_context(model) to apply FP16 autocast."
        )
        return model

    @contextmanager
    def autocast_context(self, model: nn.Module, enabled: bool = True):
        """Run forward passes under torch.autocast for mixed precision.

        Args:
            model: The model being run.
            enabled: Whether to apply autocast. When False (or when the model
                was not marked via :meth:`enable_mixed_precision`), this is a
                no-op passthrough.
        """
        if not enabled or not getattr(model, "_mixed_precision_enabled", False):
            yield model
            return
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                yield model
        else:
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                yield model
    
    def create_optimized_dataloader_config(self, dataset_size: int, 
                                        memory_limit_mb: float = 8000) -> Dict[str, Any]:
        """Create optimized dataloader configuration."""
        logger.info(f"Creating optimized dataloader config for dataset size {dataset_size}")
        
        config = {
            "batch_size": 4,  # Will be auto-tuned
            "num_workers": min(8, os.cpu_count() or 4),
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": True,
            "prefetch_factor": 2,
            "drop_last": True
        }
        
        # Auto-tune batch size based on memory
        # This would require actual model testing, for now use reasonable defaults
        
        if memory_limit_mb < 4000:
            config["batch_size"] = 2
        elif memory_limit_mb < 8000:
            config["batch_size"] = 4
        else:
            config["batch_size"] = 8
        
        logger.info(f"Optimized dataloader config: {config}")
        return config
    
    def monitor_performance(self, model: nn.Module, duration_seconds: int = 60) -> Dict[str, Any]:
        """Monitor model performance over time."""
        logger.info(f"Monitoring performance for {duration_seconds} seconds...")
        
        start_time = time.time()
        samples_processed = 0
        memory_samples = []
        
        # Create sample input
        input_tensor = torch.randn(1, 3, 256, 256, device=self.device)
        
        while time.time() - start_time < duration_seconds:
            try:
                # Process sample
                with torch.no_grad():
                    output = model(input_tensor)
                
                samples_processed += 1
                
                # Sample memory usage
                if samples_processed % 10 == 0:
                    memory_info = self.get_memory_info()
                    memory_samples.append(memory_info)
                
                # Small delay to prevent overwhelming
                time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Error during monitoring: {e}")
                break
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate statistics
        avg_samples_per_second = samples_processed / total_time
        
        # Memory statistics
        if memory_samples:
            avg_memory = sum(m.get("system_memory_mb", 0) for m in memory_samples) / len(memory_samples)
            max_memory = max(m.get("system_memory_mb", 0) for m in memory_samples)
        else:
            avg_memory = max_memory = 0
        
        results = {
            "duration_seconds": total_time,
            "samples_processed": samples_processed,
            "avg_samples_per_second": avg_samples_per_second,
            "avg_memory_mb": avg_memory,
            "max_memory_mb": max_memory
        }
        
        logger.info(f"Performance monitoring: {avg_samples_per_second:.1f} samples/sec, avg memory: {avg_memory:.1f}MB")
        
        return results


# Global optimizer instance
optimizer = PerformanceOptimizer()


def optimize_model_for_inference(model: nn.Module) -> nn.Module:
    """Convenience function to optimize model for inference."""
    return optimizer.optimize_inference_speed(optimizer.optimize_model_memory(model))


def get_optimal_batch_size(model: nn.Module, input_size: tuple, memory_limit_mb: float = 8000) -> int:
    """Convenience function to get optimal batch size."""
    return optimizer.auto_tune_batch_size(model, input_size, memory_limit_mb=memory_limit_mb)


def benchmark_model_performance(model: nn.Module, input_sizes: List[tuple] = None) -> Dict[str, Any]:
    """Convenience function to benchmark model performance."""
    if input_sizes is None:
        input_sizes = [(256, 256), (512, 512), (1024, 1024)]
    
    return optimizer.benchmark_model(model, input_sizes)
