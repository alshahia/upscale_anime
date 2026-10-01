"""
Detailed Loss Profiler - measures time per individual loss component.
Run with: .venv\Scripts\python.exe scripts/profile_loss_breakdown.py --config configs/finetune_neosr_span_v4.yaml --batches 10
"""
import sys
import os
import time
import torch
import argparse
import logging
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from anime_sr.training.neosr_finetuner import NeosrSPANFinetuner
from anime_sr.data.dataloader import DataLoaderFactory

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def profile_loss_breakdown(config_path: str, num_batches: int = 10, device: str = 'cuda'):
    """Profile individual loss components to find the slowest ones."""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'base' in config:
        base_path = os.path.join(os.path.dirname(config_path), config['base'])
        with open(base_path, 'r') as f:
            base_config = yaml.safe_load(f)
        def merge(base, override):
            for k, v in override.items():
                if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                    merge(base[k], v)
                else:
                    base[k] = v
        merge(base_config, config)
        config = base_config
    
    print(f"\n{'='*70}")
    print(f"LOSS COMPONENT BREAKDOWN PROFILER")
    print(f"{'='*70}")
    
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available")
        return
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create dataloader
    print("\nCreating dataloader and model...")
    train_loader = DataLoaderFactory.create(config, is_train=True)
    finetuner = NeosrSPANFinetuner(config, device=device)
    model = finetuner.model
    
    # Print enabled losses
    print(f"\nEnabled losses:")
    print(f"  Pixel: Charbonnier (weight={finetuner.pixel_weight})")
    print(f"  Perceptual: Twin VGG19+ResNet50 (weight={finetuner.perceptual_weight})" if finetuner.use_perceptual else "  Perceptual: disabled")
    print(f"  Line Art: enabled (weight={finetuner.line_art_weight})" if finetuner.use_line_art else "  Line Art: disabled")
    print(f"  Flat Region: enabled (weight={finetuner.flat_weight})" if finetuner.use_flat else "  Flat Region: disabled")
    print(f"  Frequency: enabled (weight={finetuner.frequency_weight})" if finetuner.use_frequency else "  Frequency: disabled")
    print(f"  FDL (DINOv2): enabled (weight={finetuner.fdl_weight}, compute_every={finetuner.fdl_compute_every})" if finetuner.use_fdl else "  FDL: disabled")
    print(f"  Wavelet-Guided: enabled (weight={finetuner.wavelet_guided_weight})" if finetuner.use_wavelet_guided else "  Wavelet-Guided: disabled")
    print(f"  Adversarial: enabled (weight={finetuner.adversarial_weight})" if finetuner.use_adversarial else "  Adversarial: disabled")
    
    # Warmup
    print(f"\nWarmup (2 batches)...")
    model.train()
    for i, batch in enumerate(train_loader):
        if i >= 2:
            break
        lr = batch['lr'].to(device, non_blocking=True)
        hr = batch['hr'].to(device, non_blocking=True)
        with torch.amp.autocast('cuda'):
            sr = model(lr)
        loss_dict = finetuner._compute_total_loss(sr, hr, batch_idx=i)
        loss_dict['total'].backward()
        finetuner.optimizer.step()
        finetuner.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
    
    # Profile individual losses
    print(f"\nProfiling {num_batches} batches...")
    loss_times = defaultdict(list)
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
        
        lr = batch['lr'].to(device, non_blocking=True)
        hr = batch['hr'].to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda'):
            sr = model(lr)
        torch.cuda.synchronize()
        
        # Profile each loss individually
        # Pixel loss
        t0 = time.perf_counter()
        pixel_loss = finetuner.pixel_loss(sr, hr)
        torch.cuda.synchronize()
        loss_times['pixel'].append(time.perf_counter() - t0)
        
        # Perceptual loss
        if finetuner.use_perceptual and finetuner.perceptual_loss is not None:
            t0 = time.perf_counter()
            with torch.amp.autocast('cuda', enabled=False):
                perc_loss = finetuner.perceptual_loss(sr.float(), hr.float())
            torch.cuda.synchronize()
            loss_times['perceptual'].append(time.perf_counter() - t0)
        
        # Line art loss
        if finetuner.use_line_art and finetuner.line_art_loss is not None:
            t0 = time.perf_counter()
            line_result = finetuner.line_art_loss(sr, hr)
            torch.cuda.synchronize()
            loss_times['line_art'].append(time.perf_counter() - t0)
        
        # Flat region loss
        if finetuner.use_flat and finetuner.flat_loss is not None:
            t0 = time.perf_counter()
            flat_result = finetuner.flat_loss(sr, hr)
            torch.cuda.synchronize()
            loss_times['flat'].append(time.perf_counter() - t0)
        
        # Frequency loss
        if finetuner.use_frequency and finetuner.frequency_loss is not None:
            t0 = time.perf_counter()
            with torch.amp.autocast('cuda', enabled=False):
                freq_loss = finetuner.frequency_loss(sr.float(), hr.float())
            torch.cuda.synchronize()
            loss_times['frequency'].append(time.perf_counter() - t0)
        
        # FDL loss
        if finetuner.use_fdl and finetuner.fdl_loss is not None:
            t0 = time.perf_counter()
            with torch.amp.autocast('cuda', enabled=False):
                fdl_loss = finetuner.fdl_loss(sr.float(), hr.float())
            torch.cuda.synchronize()
            loss_times['fdl'].append(time.perf_counter() - t0)
        
        # Wavelet-guided loss
        if finetuner.use_wavelet_guided and finetuner.wavelet_guided_loss is not None:
            t0 = time.perf_counter()
            with torch.amp.autocast('cuda', enabled=False):
                wg_loss = finetuner.wavelet_guided_loss(sr.float(), hr.float(), epoch=finetuner.current_epoch)
            torch.cuda.synchronize()
            loss_times['wavelet'].append(time.perf_counter() - t0)
        
        # Discriminator (if adversarial)
        if finetuner.use_adversarial and finetuner.discriminator is not None:
            t0 = time.perf_counter()
            with torch.amp.autocast('cuda', enabled=False):
                disc_real = finetuner.discriminator(hr.float())
                disc_fake = finetuner.discriminator(sr.float())
            torch.cuda.synchronize()
            loss_times['discriminator'].append(time.perf_counter() - t0)
    
    # Print results
    print(f"\n{'='*70}")
    print(f"LOSS COMPONENT TIMING (avg ms per batch)")
    print(f"{'='*70}")
    print(f"\n{'Loss Component':<25} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'% of Loss Time':<15}")
    print(f"{'-'*76}")
    
    total_loss_time = 0
    results = []
    for name, times in loss_times.items():
        avg_ms = sum(times) / len(times) * 1000
        min_ms = min(times) * 1000
        max_ms = max(times) * 1000
        total_loss_time += avg_ms
        results.append((name, avg_ms, min_ms, max_ms))
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    for name, avg_ms, min_ms, max_ms in results:
        pct = avg_ms / total_loss_time * 100 if total_loss_time > 0 else 0
        print(f"{name:<25} {avg_ms:<12.1f} {min_ms:<12.1f} {max_ms:<12.1f} {pct:<15.1f}%")
    
    print(f"{'-'*76}")
    print(f"{'Total Loss Time':<25} {total_loss_time:<12.1f}")
    
    # Recommendations
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION RECOMMENDATIONS")
    print(f"{'='*70}")
    
    # Find top 3 slowest losses
    top_slow = results[:3]
    print(f"\nTop 3 slowest loss components:")
    for i, (name, avg_ms, _, _) in enumerate(top_slow):
        print(f"  {i+1}. {name}: {avg_ms:.1f}ms")
    
    # Specific recommendations
    print(f"\nSpecific fixes to reduce loss computation time:")
    
    for name, avg_ms, _, _ in results:
        if name == 'fdl' and avg_ms > 20:
            print(f"\n  [FDL Loss] {avg_ms:.1f}ms - DINOv2 forward pass is expensive")
            print(f"    -> Increase compute_every from 2 to 4 or 8 (saves 50-75% of FDL time)")
            print(f"    -> Config: training.finetune.loss.fdl.compute_every: 4")
        
        if name == 'perceptual' and avg_ms > 50:
            print(f"\n  [Perceptual Loss] {avg_ms:.1f}ms - Twin VGG19+ResNet50 runs TWO networks")
            print(f"    -> Switch to single VGG or DISTS (type: vgg or type: dists)")
            print(f"    -> Config: training.finetune.loss.perceptual.type: dists")
        
        if name == 'wavelet' and avg_ms > 10:
            print(f"\n  [Wavelet-Guided Loss] {avg_ms:.1f}ms - Haar wavelet decomposition")
            print(f"    -> Disable if not critical for your use case")
            print(f"    -> Config: training.finetune.loss.wavelet_guided.enabled: false")
        
        if name == 'frequency' and avg_ms > 10:
            print(f"\n  [Frequency Loss] {avg_ms:.1f}ms - DCT-based frequency analysis")
            print(f"    -> Disable if not critical")
            print(f"    -> Config: training.finetune.loss.frequency.enabled: false")
        
        if name == 'line_art' and avg_ms > 20:
            print(f"\n  [Line Art Loss] {avg_ms:.1f}ms - Multi-scale edge detection")
            print(f"    -> Switch to simpler edge method (type: sobel instead of multi_scale)")
            print(f"    -> Config: training.finetune.loss.anime.line_art_preservation.edge_method: sobel")
        
        if name == 'discriminator' and avg_ms > 20:
            print(f"\n  [Discriminator] {avg_ms:.1f}ms - GAN discriminator forward pass")
            print(f"    -> Already runs in FP32 (required for stability)")
            print(f"    -> Consider reducing discriminator layers if not critical")
    
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Profile individual loss components')
    parser.add_argument('--config', type=str, required=True, help='Path to training config')
    parser.add_argument('--batches', type=int, default=10, help='Number of batches to profile')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    profile_loss_breakdown(args.config, args.batches, args.device)
