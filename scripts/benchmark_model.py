#!/usr/bin/env python
"""
Comprehensive benchmarking script for super-resolution models.
Evaluates PSNR, SSIM, LPIPS, DISTS metrics.
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.span import create_span_model
from utils.image_utils import tensor_to_image, image_to_tensor


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """Calculate PSNR between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate SSIM between two images."""
    try:
        from skimage.metrics import structural_similarity as ssim
        # Convert to grayscale if RGB
        if len(img1.shape) == 3:
            img1_gray = np.mean(img1, axis=2)
            img2_gray = np.mean(img2, axis=2)
        else:
            img1_gray = img1
            img2_gray = img2
        
        return ssim(img1_gray, img2_gray, data_range=img1.max() - img1.min())
    except ImportError:
        print("Warning: scikit-image not available, skipping SSIM")
        return 0.0


def calculate_lpips(img1: torch.Tensor, img2: torch.Tensor, device: str = 'cuda') -> float:
    """Calculate LPIPS perceptual distance."""
    try:
        import lpips
        loss_fn = lpips.LPIPS(net='alex').to(device)
        
        # Normalize to [-1, 1]
        img1_norm = img1 * 2 - 1
        img2_norm = img2 * 2 - 1
        
        with torch.no_grad():
            dist = loss_fn(img1_norm, img2_norm)
        
        return dist.item()
    except ImportError:
        print("Warning: lpips not available, skipping LPIPS")
        return 0.0


def calculate_dists(img1: torch.Tensor, img2: torch.Tensor, device: str = 'cuda') -> float:
    """Calculate DISTS perceptual metric (anime-optimized)."""
    try:
        from models.loss import DISTSLoss
        dists_loss = DISTSLoss().to(device)
        
        with torch.no_grad():
            dist = dists_loss(img1, img2)
        
        return dist.item()
    except (ImportError, RuntimeError) as e:
        # Fallback to simple perceptual metric
        print(f"DISTS not available, using fallback: {e}")
        return torch.mean(torch.abs(img1 - img2)).item()


def benchmark_single_image(model: nn.Module,
                          lr_path: Path,
                          hr_path: Path,
                          device: str = 'cuda') -> Dict:
    """
    Benchmark model on a single image pair.
    
    Returns:
        Dictionary with metrics
    """
    # Load images
    lr_img = Image.open(lr_path).convert('RGB')
    hr_img = Image.open(hr_path).convert('RGB')
    
    # Convert to tensors
    lr_tensor = image_to_tensor(lr_img).unsqueeze(0).to(device)
    hr_tensor = image_to_tensor(hr_img).unsqueeze(0).to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        sr_tensor = model(lr_tensor)
        inference_time = time.time() - start_time
    
    # Convert to numpy for PSNR/SSIM
    sr_np = (tensor_to_image(sr_tensor.squeeze(0)) * 255).astype(np.uint8)
    hr_np = (tensor_to_image(hr_tensor.squeeze(0)) * 255).astype(np.uint8)
    
    # Calculate metrics
    metrics = {
        'psnr': calculate_psnr(sr_np, hr_np),
        'ssim': calculate_ssim(sr_np, hr_np),
        'lpips': calculate_lpips(sr_tensor, hr_tensor, device),
        'dists': calculate_dists(sr_tensor, hr_tensor, device),
        'inference_time': inference_time
    }
    
    return metrics


def benchmark_directory(model: nn.Module,
                       lr_dir: Path,
                       hr_dir: Path,
                       device: str = 'cuda') -> Dict:
    """
    Benchmark model on entire directory.
    
    Returns:
        Dictionary with average metrics
    """
    # Find all image pairs
    lr_paths = sorted(lr_dir.glob('*.png')) + sorted(lr_dir.glob('*.jpg'))
    
    all_metrics = []
    
    for lr_path in tqdm(lr_paths, desc="Benchmarking"):
        # Find corresponding HR image
        hr_path = hr_dir / lr_path.name
        if not hr_path.exists():
            # Try without extension matching
            for ext in ['.png', '.jpg', '.jpeg']:
                hr_path = hr_dir / (lr_path.stem + ext)
                if hr_path.exists():
                    break
        
        if not hr_path.exists():
            print(f"Warning: No HR image found for {lr_path}")
            continue
        
        try:
            metrics = benchmark_single_image(model, lr_path, hr_path, device)
            metrics['filename'] = lr_path.name
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error processing {lr_path}: {e}")
    
    # Calculate averages
    if not all_metrics:
        return {}
    
    avg_metrics = {
        'psnr': np.mean([m['psnr'] for m in all_metrics]),
        'ssim': np.mean([m['ssim'] for m in all_metrics]),
        'lpips': np.mean([m['lpips'] for m in all_metrics]),
        'dists': np.mean([m['dists'] for m in all_metrics]),
        'inference_time': np.mean([m['inference_time'] for m in all_metrics]),
        'num_images': len(all_metrics)
    }
    
    # Calculate std deviations
    avg_metrics['psnr_std'] = np.std([m['psnr'] for m in all_metrics])
    avg_metrics['ssim_std'] = np.std([m['ssim'] for m in all_metrics])
    
    return {
        'average': avg_metrics,
        'per_image': all_metrics
    }


def load_model(checkpoint_path: str, device: str = 'cuda') -> nn.Module:
    """Load model from checkpoint."""
    print(f"Loading model: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model
    from omegaconf import OmegaConf
    config = OmegaConf.create({
        'model': {
            'type': 'span',
            'channels': 32,
            'scale': 4
        }
    })
    
    model = create_span_model(config.model)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    return model


def compare_models(checkpoint_paths: List[str],
                  lr_dir: Path,
                  hr_dir: Path,
                  device: str = 'cuda') -> Dict:
    """
    Compare multiple models on the same test set.
    
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    for checkpoint_path in checkpoint_paths:
        model_name = Path(checkpoint_path).stem
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_name}")
        print(f"{'='*60}")
        
        model = load_model(checkpoint_path, device)
        metrics = benchmark_directory(model, lr_dir, hr_dir, device)
        
        results[model_name] = metrics.get('average', {})
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark SR models')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Model checkpoint path (or comma-separated list for comparison)')
    parser.add_argument('--lr-dir', type=str, required=True, help='Directory with LR test images')
    parser.add_argument('--hr-dir', type=str, required=True, help='Directory with HR test images')
    parser.add_argument('--output', type=str, default='benchmark_results.json', help='Output JSON path')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--compare', action='store_true', help='Compare multiple models')
    
    args = parser.parse_args()
    
    lr_dir = Path(args.lr_dir)
    hr_dir = Path(args.hr_dir)
    
    if args.compare:
        # Compare multiple models
        checkpoint_paths = args.checkpoint.split(',')
        results = compare_models(checkpoint_paths, lr_dir, hr_dir, args.device)
        
        # Print comparison table
        print(f"\n{'='*80}")
        print("Model Comparison")
        print(f"{'='*80}")
        print(f"{'Model':<30} {'PSNR':<10} {'SSIM':<10} {'LPIPS':<10} {'Time (s)':<10}")
        print(f"{'-'*80}")
        
        for model_name, metrics in results.items():
            print(f"{model_name:<30} {metrics.get('psnr', 0):.2f}     "
                  f"{metrics.get('ssim', 0):.4f}   "
                  f"{metrics.get('lpips', 0):.4f}   "
                  f"{metrics.get('inference_time', 0):.4f}")
    else:
        # Benchmark single model
        model = load_model(args.checkpoint, args.device)
        results = benchmark_directory(model, lr_dir, hr_dir, args.device)
        
        # Print summary
        if 'average' in results:
            avg = results['average']
            print(f"\n{'='*60}")
            print("Benchmark Summary")
            print(f"{'='*60}")
            print(f"Images tested:    {avg['num_images']}")
            print(f"PSNR:            {avg['psnr']:.2f} ± {avg.get('psnr_std', 0):.2f} dB")
            print(f"SSIM:            {avg['ssim']:.4f} ± {avg.get('ssim_std', 0):.4f}")
            print(f"LPIPS:           {avg['lpips']:.4f}")
            print(f"DISTS:           {avg['dists']:.4f}")
            print(f"Avg inference:   {avg['inference_time']*1000:.2f} ms")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
