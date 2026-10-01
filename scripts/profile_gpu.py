"""
GPU Utilization Profiler for training pipeline.
Measures time spent in: data loading, forward pass, loss computation, backward pass.
Run with: python scripts/profile_gpu.py --config configs/finetune_neosr_span_v4.yaml --batches 20
"""
import sys
import os
import time
import torch
import argparse
import logging
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from anime_sr.training.neosr_finetuner import NeosrSPANFinetuner
from anime_sr.data.dataloader import DataLoaderFactory

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def profile_training(config_path: str, num_batches: int = 20, device: str = 'cuda'):
    """Profile one epoch to find GPU bottlenecks."""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Resolve base config
    if 'base' in config:
        base_path = os.path.join(os.path.dirname(config_path), config['base'])
        with open(base_path, 'r') as f:
            base_config = yaml.safe_load(f)
        # Merge base with override
        def merge(base, override):
            for k, v in override.items():
                if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                    merge(base[k], v)
                else:
                    base[k] = v
        merge(base_config, config)
        config = base_config
    
    print(f"\n{'='*60}")
    print(f"GPU UTILIZATION PROFILER")
    print(f"{'='*60}")
    print(f"Config: {config_path}")
    print(f"Device: {device}")
    print(f"Batches to profile: {num_batches}")
    
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available")
        return
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create dataloader
    print("\n[1/4] Creating dataloader...")
    train_loader = DataLoaderFactory.create(config, is_train=True)
    print(f"  Dataset size: {len(train_loader.dataset)}")
    print(f"  Batch size: {train_loader.batch_size}")
    print(f"  Num workers: {train_loader.num_workers}")
    print(f"  Prefetch factor: {train_loader.prefetch_factor}")
    
    # Create model
    print("\n[2/4] Creating model...")
    finetuner = NeosrSPANFinetuner(config, device=device)
    model = finetuner.model
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params/1e6:.2f}M")
    print(f"  Trainable params: {trainable_params/1e6:.2f}M")
    
    # Profiling accumulators
    timing = defaultdict(list)
    gpu_util_samples = []
    
    # Warmup
    print(f"\n[3/4] Warmup (3 batches)...")
    model.train()
    warmup_batches = 3
    for i, batch in enumerate(train_loader):
        if i >= warmup_batches:
            break
        lr = batch['lr'].to(device, non_blocking=True)
        hr = batch['hr'].to(device, non_blocking=True)
        with torch.amp.autocast('cuda'):
            sr = model(lr)
        loss_dict = finetuner._compute_total_loss(sr, hr, batch_idx=i)
        loss = loss_dict['total']
        loss.backward()
        finetuner.optimizer.step()
        finetuner.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
    
    # Profile
    print(f"[4/4] Profiling {num_batches} batches...")
    print(f"{'Batch':<6} {'DataLoad':<10} {'Forward':<10} {'Loss':<10} {'Backward':<10} {'OptStep':<10} {'GPU%':<6}")
    print(f"{'-'*62}")
    
    torch.cuda.reset_peak_memory_stats()
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
        
        # Measure data loading time
        t0 = time.perf_counter()
        lr = batch['lr'].to(device, non_blocking=True)
        hr = batch['hr'].to(device, non_blocking=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        data_load_time = t1 - t0
        
        # Measure forward pass
        t2 = time.perf_counter()
        with torch.amp.autocast('cuda'):
            sr = model(lr)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        forward_time = t3 - t2
        
        # Measure loss computation
        t4 = time.perf_counter()
        loss_dict = finetuner._compute_total_loss(sr, hr, batch_idx=batch_idx)
        torch.cuda.synchronize()
        t5 = time.perf_counter()
        loss_time = t5 - t4
        
        # Measure backward pass
        t6 = time.perf_counter()
        loss_dict['total'].backward()
        torch.cuda.synchronize()
        t7 = time.perf_counter()
        backward_time = t7 - t6
        
        # Measure optimizer step
        t8 = time.perf_counter()
        finetuner.optimizer.step()
        finetuner.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t9 = time.perf_counter()
        opt_step_time = t9 - t8
        
        # Sample GPU utilization (approximate via active time ratio)
        total_compute_time = forward_time + loss_time + backward_time + opt_step_time
        total_batch_time = data_load_time + total_compute_time
        gpu_active_ratio = (total_compute_time / total_batch_time * 100) if total_batch_time > 0 else 0
        
        timing['data_load'].append(data_load_time)
        timing['forward'].append(forward_time)
        timing['loss'].append(loss_time)
        timing['backward'].append(backward_time)
        timing['opt_step'].append(opt_step_time)
        timing['gpu_ratio'].append(gpu_active_ratio)
        
        print(f"{batch_idx:<6} {data_load_time*1000:<10.1f} {forward_time*1000:<10.1f} {loss_time*1000:<10.1f} {backward_time*1000:<10.1f} {opt_step_time*1000:<10.1f} {gpu_active_ratio:<6.0f}%")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"PROFILING SUMMARY ({num_batches} batches)")
    print(f"{'='*60}")
    
    avg_data_load = sum(timing['data_load']) / len(timing['data_load'])
    avg_forward = sum(timing['forward']) / len(timing['forward'])
    avg_loss = sum(timing['loss']) / len(timing['loss'])
    avg_backward = sum(timing['backward']) / len(timing['backward'])
    avg_opt_step = sum(timing['opt_step']) / len(timing['opt_step'])
    avg_gpu_ratio = sum(timing['gpu_ratio']) / len(timing['gpu_ratio'])
    
    total_time = avg_data_load + avg_forward + avg_loss + avg_backward + avg_opt_step
    
    print(f"\n{'Phase':<20} {'Avg (ms)':<12} {'% of Total':<12}")
    print(f"{'-'*44}")
    print(f"{'Data Loading':<20} {avg_data_load*1000:<12.1f} {avg_data_load/total_time*100:<12.1f}%")
    print(f"{'Forward Pass':<20} {avg_forward*1000:<12.1f} {avg_forward/total_time*100:<12.1f}%")
    print(f"{'Loss Computation':<20} {avg_loss*1000:<12.1f} {avg_loss/total_time*100:<12.1f}%")
    print(f"{'Backward Pass':<20} {avg_backward*1000:<12.1f} {avg_backward/total_time*100:<12.1f}%")
    print(f"{'Optimizer Step':<20} {avg_opt_step*1000:<12.1f} {avg_opt_step/total_time*100:<12.1f}%")
    print(f"{'-'*44}")
    print(f"{'Total per batch':<20} {total_time*1000:<12.1f} {'100.0%':<12}")
    
    print(f"\nEstimated GPU Utilization: {avg_gpu_ratio:.0f}%")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    # Identify bottleneck
    print(f"\n{'='*60}")
    print(f"BOTTLENECK ANALYSIS")
    print(f"{'='*60}")
    
    phases = [
        ('Data Loading', avg_data_load),
        ('Forward Pass', avg_forward),
        ('Loss Computation', avg_loss),
        ('Backward Pass', avg_backward),
        ('Optimizer Step', avg_opt_step),
    ]
    phases.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop bottlenecks (by time):")
    for i, (name, t) in enumerate(phases[:3]):
        print(f"  {i+1}. {name}: {t*1000:.1f}ms ({t/total_time*100:.1f}%)")
    
    # Recommendations
    print(f"\n{'='*60}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*60}")
    
    if avg_data_load > avg_forward:
        print(f"\n[DATA LOADING BOTTLENECK]")
        print(f"  Data loading ({avg_data_load*1000:.1f}ms) is slower than forward pass ({avg_forward*1000:.1f}ms)")
        print(f"  GPU waits {((avg_data_load - avg_forward) / avg_data_load * 100):.0f}% of the time")
        print(f"\n  Fixes:")
        print(f"  1. Increase num_workers (current: {train_loader.num_workers})")
        print(f"  2. Increase prefetch_factor (current: {train_loader.prefetch_factor})")
        print(f"  3. Use precomputed dataset: run scripts/precompute_degradation.py")
        print(f"  4. Reduce degradation complexity (disable two_stage_compression)")
        print(f"  5. Set degrade_before_crop: false (crop first, degrade smaller patch)")
        print(f"  6. Enable preload: true in config (if RAM allows)")
    
    if avg_loss > avg_forward * 0.5:
        print(f"\n[LOSS COMPUTATION BOTTLENECK]")
        print(f"  Loss computation ({avg_loss*1000:.1f}ms) is taking significant time")
        print(f"  This is likely due to multiple heavy perceptual losses (DISTS, VGG, DINOv2)")
        print(f"\n  Fixes:")
        print(f"  1. Increase fdl.compute_every (current: 2, try 4 or 8)")
        print(f"  2. Disable wavelet_guided loss (heavy Haar decomposition)")
        print(f"  3. Reduce perceptual loss types (use single VGG instead of twin)")
        print(f"  4. Disable frequency loss if not critical")
    
    if avg_gpu_ratio < 70:
        print(f"\n[LOW GPU UTILIZATION]")
        print(f"  GPU is active only {avg_gpu_ratio:.0f}% of the time")
        print(f"  This means {100 - avg_gpu_ratio:.0f}% of time GPU is idle waiting for CPU")
        print(f"\n  Fixes:")
        print(f"  1. Increase batch_size (current: {train_loader.batch_size})")
        print(f"  2. Use torch.compile (set torch_compile: true if on Linux)")
        print(f"  3. Enable gradient accumulation to simulate larger batches")
        print(f"  4. Reduce CPU-heavy operations (degradation, augmentation)")
    
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Profile GPU utilization during training')
    parser.add_argument('--config', type=str, required=True, help='Path to training config')
    parser.add_argument('--batches', type=int, default=20, help='Number of batches to profile')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    profile_training(args.config, args.batches, args.device)
