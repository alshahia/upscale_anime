"""
Asynchronous data prefetcher using CUDA streams.
Overlaps data loading with GPU computation for maximum throughput.
"""
import torch
from typing import Iterator, Dict, Optional


class AsyncDataPrefetcher:
    """
    Prefetches next batch on GPU while current batch is being processed.
    
    This eliminates the CPU->GPU transfer bottleneck by overlapping
    data transfer with GPU computation using CUDA streams.
    
    Usage:
        prefetcher = AsyncDataPrefetcher(dataloader, device, num_prefetch=2)
        for batch in prefetcher:
            # batch is already on GPU
            loss = model(batch)
    
    Args:
        dataloader: PyTorch DataLoader instance
        device: Target device (typically CUDA)
        num_prefetch: Number of batches to prefetch ahead (default: 2)
    """
    
    def __init__(
        self,
        dataloader,
        device: torch.device,
        num_prefetch: int = 2,
    ):
        self.dataloader = dataloader
        self.device = device
        self.num_prefetch = num_prefetch
        
        # Create dedicated CUDA stream for data transfer
        self.stream = torch.cuda.Stream(device=device) if device.type == 'cuda' else None
        
        self.loader_iter = iter(dataloader)
        self.prefetch_queue = []
        
        # Pre-fill queue
        self._fill_queue()
    
    def _fill_queue(self):
        """Fill prefetch queue with next batches."""
        while len(self.prefetch_queue) < self.num_prefetch:
            try:
                batch = next(self.loader_iter)
                
                if self.stream is not None:
                    # Asynchronous transfer on dedicated stream
                    with torch.cuda.stream(self.stream):
                        batch = self._transfer_batch(batch)
                else:
                    # CPU mode: just transfer synchronously
                    batch = self._transfer_batch(batch)
                
                self.prefetch_queue.append(batch)
            except StopIteration:
                break
    
    def _transfer_batch(self, batch: Dict) -> Dict:
        """Transfer batch tensors to target device."""
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                # Use non-blocking transfer for overlap
                result[key] = value.to(
                    self.device,
                    non_blocking=self.stream is not None,
                )
            else:
                result[key] = value
        return result
    
    def __iter__(self) -> Iterator[Dict]:
        """Return iterator over prefetched batches."""
        return self
    
    def __next__(self) -> Dict:
        """Get next batch (already on GPU)."""
        if not self.prefetch_queue:
            raise StopIteration
        
        # Get next batch from queue
        batch = self.prefetch_queue.pop(0)
        
        # Wait for this batch to be ready on default stream
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        
        # Prefetch next batch to fill queue
        self._fill_queue()
        
        return batch
    
    def __len__(self) -> int:
        """Return length of underlying dataloader."""
        return len(self.dataloader)
    
    def reset(self):
        """Reset prefetcher for new epoch."""
        self.loader_iter = iter(self.dataloader)
        self.prefetch_queue = []
        self._fill_queue()


def create_prefetcher(
    dataloader,
    config: Dict,
    device: torch.device,
):
    """
    Factory function to create AsyncDataPrefetcher based on config.
    
    Args:
        dataloader: PyTorch DataLoader
        config: Training configuration dict
        device: Target device
        
    Returns:
        AsyncDataPrefetcher if enabled and CUDA available, else original dataloader
    """
    use_prefetcher = config.get('training', {}).get('use_async_prefetcher', False)
    
    if not use_prefetcher:
        return dataloader
    
    if device.type != 'cuda':
        print("[Prefetcher] Warning: Async prefetcher requires CUDA, using standard dataloader")
        return dataloader
    
    num_prefetch = config.get('training', {}).get('prefetcher_num_prefetch', 2)
    
    print(f"[Prefetcher] Enabling async data prefetching (num_prefetch={num_prefetch})")
    
    return AsyncDataPrefetcher(
        dataloader=dataloader,
        device=device,
        num_prefetch=num_prefetch,
    )
