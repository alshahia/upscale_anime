"""
System monitoring utilities
GPU/VRAM monitoring and resource management
"""
import torch
import psutil
import time
from typing import Dict, Optional
from contextlib import contextmanager


class GPUMonitor:
    """GPU resource monitoring"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.has_cuda = torch.cuda.is_available()
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get GPU memory information"""
        if not self.has_cuda:
            return {'available': False}
        
        torch.cuda.synchronize(self.device_id)
        
        allocated = torch.cuda.memory_allocated(self.device_id) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(self.device_id) / 1024**3   # GB
        total = torch.cuda.get_device_properties(self.device_id).total_memory / 1024**3
        
        return {
            'available': True,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': total - reserved,
            'utilization_percent': (reserved / total) * 100,
        }
    
    def get_device_name(self) -> str:
        """Get GPU device name"""
        if not self.has_cuda:
            return "CPU"
        
        return torch.cuda.get_device_name(self.device_id)
    
    def print_memory_summary(self):
        """Print GPU memory summary"""
        info = self.get_memory_info()
        
        if not info['available']:
            print("CUDA not available")
            return
        
        print(f"\nGPU Memory (Device {self.device_id}: {self.get_device_name()})")
        print(f"  Total:     {info['total_gb']:.2f} GB")
        print(f"  Reserved:  {info['reserved_gb']:.2f} GB ({info['utilization_percent']:.1f}%)")
        print(f"  Allocated: {info['allocated_gb']:.2f} GB")
        print(f"  Free:      {info['free_gb']:.2f} GB")
    
    def clear_cache(self):
        """Clear GPU cache"""
        if self.has_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device_id)


class SystemMonitor:
    """System resource monitoring"""
    
    def __init__(self):
        self.process = psutil.Process()
    
    def get_cpu_info(self) -> Dict[str, float]:
        """Get CPU information"""
        return {
            'percent': psutil.cpu_percent(interval=1),
            'count': psutil.cpu_count(),
            'freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
        }
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get system memory information"""
        mem = psutil.virtual_memory()
        
        return {
            'total_gb': mem.total / 1024**3,
            'available_gb': mem.available / 1024**3,
            'used_gb': mem.used / 1024**3,
            'percent': mem.percent,
        }
    
    def get_process_info(self) -> Dict[str, float]:
        """Get current process information"""
        mem_info = self.process.memory_info()
        
        return {
            'rss_mb': mem_info.rss / 1024**2,
            'vms_mb': mem_info.vms / 1024**2,
            'cpu_percent': self.process.cpu_percent(),
            'num_threads': self.process.num_threads(),
        }
    
    def print_system_summary(self):
        """Print system resource summary"""
        cpu = self.get_cpu_info()
        mem = self.get_memory_info()
        proc = self.get_process_info()
        
        print("\nSystem Resources")
        print(f"  CPU: {cpu['percent']:.1f}% ({cpu['count']} cores)")
        print(f"  RAM: {mem['used_gb']:.2f} / {mem['total_gb']:.2f} GB ({mem['percent']:.1f}%)")
        print(f"  Process: {proc['rss_mb']:.1f} MB RSS, {proc['num_threads']} threads")


class ResourceMonitor:
    """Combined resource monitor with timing"""
    
    def __init__(self, device_id: int = 0):
        self.gpu_monitor = GPUMonitor(device_id)
        self.system_monitor = SystemMonitor()
    
    def get_full_report(self) -> Dict:
        """Get full resource report"""
        return {
            'gpu': self.gpu_monitor.get_memory_info(),
            'cpu': self.system_monitor.get_cpu_info(),
            'memory': self.system_monitor.get_memory_info(),
            'process': self.system_monitor.get_process_info(),
        }
    
    def print_full_report(self):
        """Print full resource report"""
        self.gpu_monitor.print_memory_summary()
        self.system_monitor.print_system_summary()
    
    def check_vram_available(self, required_gb: float = 1.0) -> bool:
        """
        Check if required VRAM is available.
        
        Args:
            required_gb: Required VRAM in GB
        
        Returns:
            True if sufficient VRAM available
        """
        info = self.gpu_monitor.get_memory_info()
        
        if not info['available']:
            return False
        
        return info['free_gb'] >= required_gb
    
    def estimate_batch_size(self, model_memory_mb: float = 100) -> int:
        """
        Estimate safe batch size based on available VRAM.
        
        Args:
            model_memory_mb: Estimated model memory in MB
        
        Returns:
            Recommended batch size
        """
        info = self.gpu_monitor.get_memory_info()
        
        if not info['available']:
            return 1
        
        # Rough estimation: 1 batch ≈ model_memory_mb MB
        free_mb = info['free_gb'] * 1024
        safe_batch = int((free_mb * 0.8) / model_memory_mb)
        
        # Clamp to reasonable range
        return max(1, min(safe_batch, 32))


@contextmanager
def monitor_execution(name: str = "Operation"):
    """
    Context manager to monitor execution time and resources.
    
    Usage:
        with monitor_execution("Training"):
            train_model()
    """
    monitor = ResourceMonitor()
    start_time = time.time()
    
    print(f"\n{'='*50}")
    print(f"Starting: {name}")
    print(f"{'='*50}")
    
    # Initial report
    monitor.print_full_report()
    
    try:
        yield monitor
    finally:
        elapsed = time.time() - start_time
        
        print(f"\n{'='*50}")
        print(f"Completed: {name}")
        print(f"Duration: {elapsed:.2f}s ({elapsed/60:.2f}m)")
        print(f"{'='*50}")
        
        # Final report
        monitor.print_full_report()


def get_optimal_worker_count() -> int:
    """
    Get optimal number of DataLoader workers.
    Based on CPU count and system load.
    """
    cpu_count = psutil.cpu_count(logical=False) or 4
    
    # Use half the CPU cores, but at least 2 and at most 8
    optimal = max(2, min(cpu_count // 2, 8))
    
    return optimal


def print_environment_info():
    """Print complete environment information"""
    print("\n" + "="*50)
    print("Environment Information")
    print("="*50)
    
    # PyTorch
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Device Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\n  Device {i}: {props.name}")
            print(f"    Compute Capability: {props.major}.{props.minor}")
            print(f"    Total Memory: {props.total_memory / 1024**3:.2f} GB")
    
    # System
    print(f"\nCPU Count: {psutil.cpu_count()}")
    print(f"CPU Freq: {psutil.cpu_freq().current:.0f} MHz" if psutil.cpu_freq() else "N/A")
    
    mem = psutil.virtual_memory()
    print(f"RAM: {mem.total / 1024**3:.2f} GB")
    
    print("="*50 + "\n")


# Convenience function
def check_system_ready(required_vram_gb: float = 4.0) -> bool:
    """
    Check if system is ready for training.
    
    Args:
        required_vram_gb: Required VRAM in GB
    
    Returns:
        True if system is ready
    """
    monitor = ResourceMonitor()
    
    print("Checking system readiness...")
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available")
        return False
    
    # Check VRAM
    if not monitor.check_vram_available(required_vram_gb):
        info = monitor.gpu_monitor.get_memory_info()
        print(f"[ERROR] Insufficient VRAM: {info['free_gb']:.2f} GB available, {required_vram_gb:.2f} GB required")
        return False
    
    # Check system RAM
    mem = monitor.system_monitor.get_memory_info()
    if mem['available_gb'] < 4.0:
        print(f"[WARN]  Low system RAM: {mem['available_gb']:.2f} GB available")
    
    print("[OK] System ready for training")
    return True
