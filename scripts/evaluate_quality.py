#!/usr/bin/env python3
"""
Quality evaluation script using no-reference metrics.
Computes NIQE, MANIQA, CLIPIQA, TOPIQ, and MUSIQ scores for super-resolved images.

Usage:
    python scripts/evaluate_quality.py --input results/ --output quality_report.csv
    python scripts/evaluate_quality.py --input sr_images/ --gt gt_images/ --metrics psnr ssim lpips
    python scripts/evaluate_quality.py --input results/ --smoke 4 --metrics clipiqa topiq_nr niqe
"""
import sys
import argparse
import logging
import torch
import cv2
import numpy as np
import csv
import statistics
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from anime_sr.utils.metrics import (
    calculate_niqe,
    calculate_maniqa,
    calculate_clipiqa,
    calculate_topiq,
    calculate_musiq,
    calculate_psnr,
    calculate_ssim,
    calculate_lpips,
    PYIQA_DIRECTION,
    PYIQA_RANGE,
)

logger = logging.getLogger(__name__)

# Maps CLI metric name -> calculate_<name> function
METRIC_FUNCTIONS = {
    'niqe': calculate_niqe,
    'maniqa': calculate_maniqa,
    'clipiqa': calculate_clipiqa,
    'topiq_nr': calculate_topiq,
    'musiq': calculate_musiq,
}


def load_image(image_path: str) -> Optional[torch.Tensor]:
    """Load and preprocess image to tensor."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    return img_tensor.unsqueeze(0)


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """Calculate PSNR."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    psnr = 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """Calculate SSIM."""
    import torch.nn.functional as F
    
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2
    
    sigma = 1.5
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / (2 * sigma**2)) for x in range(window_size)])
    gauss = gauss / gauss.sum()
    window = gauss.unsqueeze(1) * gauss.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0).to(img1.device)
    window = window.repeat(img1.size(1), 1, 1, 1)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 ** 2, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def evaluate_image(
    sr_path: str,
    gt_path: Optional[str] = None,
    metrics_to_compute: List[str] = None,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Evaluate a single image with specified metrics.
    
    Args:
        sr_path: Path to super-resolved image
        gt_path: Optional path to ground truth image
        metrics_to_compute: List of metrics to compute
        device: Device for computation
    
    Returns:
        Dictionary of metric results. Failed metrics are stored as NaN.
    """
    if metrics_to_compute is None:
        metrics_to_compute = ['niqe', 'maniqa', 'clipiqa', 'topiq_nr']
    
    sr_tensor = load_image(sr_path)
    if sr_tensor is None:
        logger.warning(f"Could not load image: {sr_path}")
        return {}
    
    sr_tensor = sr_tensor.to(device)
    
    results = {}
    
    # No-reference metrics (dispatched via METRIC_FUNCTIONS map).
    for metric_name, fn in METRIC_FUNCTIONS.items():
        if metric_name in metrics_to_compute:
            try:
                results[metric_name] = fn(sr_tensor)
            except Exception as e:
                logger.warning(f"{metric_name.upper()} failed on {sr_path}: {e}")
                results[metric_name] = float('nan')
    
    # Full-reference metrics (if ground truth provided)
    if gt_path:
        gt_tensor = load_image(gt_path)
        if gt_tensor is not None:
            gt_tensor = gt_tensor.to(device)
            
            # Resize SR to match GT if needed
            if sr_tensor.shape != gt_tensor.shape:
                sr_tensor = torch.nn.functional.interpolate(
                    sr_tensor, size=gt_tensor.shape[2:], mode='bicubic', align_corners=False
                )
            
            if 'psnr' in metrics_to_compute:
                results['psnr'] = calculate_psnr(sr_tensor, gt_tensor)
            
            if 'ssim' in metrics_to_compute:
                results['ssim'] = calculate_ssim(sr_tensor, gt_tensor)
            
            if 'lpips' in metrics_to_compute:
                try:
                    results['lpips'] = calculate_lpips(sr_tensor, gt_tensor)
                except Exception as e:
                    logger.warning(f"LPIPS failed on {sr_path}: {e}")
                    results['lpips'] = float('nan')
    
    return results


def _summarize(values: List[float]) -> Dict[str, float]:
    """Return mean/median/std/min/max for a list of floats, ignoring NaNs."""
    finite = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not finite:
        nan = float('nan')
        return {'mean': nan, 'median': nan, 'std': nan, 'min': nan, 'max': nan, 'count': 0}
    return {
        'mean': float(np.mean(finite)),
        'median': float(np.median(finite)),
        'std': float(np.std(finite)) if len(finite) > 1 else 0.0,
        'min': float(np.min(finite)),
        'max': float(np.max(finite)),
        'count': len(finite),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate super-resolution quality')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input directory with SR images')
    parser.add_argument('--gt', '-g', type=str, default=None,
                       help='Ground truth directory (optional)')
    parser.add_argument('--output', '-o', type=str, default='quality_report.csv',
                       help='Output CSV file')
    parser.add_argument('--metrics', '-m', type=str, nargs='+',
                       default=['niqe', 'maniqa', 'clipiqa', 'topiq_nr'],
                       help='Metrics to compute: niqe, maniqa, clipiqa, topiq_nr, musiq, psnr, ssim, lpips')
    parser.add_argument('--device', '-d', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for computation')
    parser.add_argument('--format', '-f', type=str, default='csv',
                       choices=['csv', 'table'],
                       help='Output format')
    parser.add_argument('--smoke', '-s', type=int, default=None,
                       help='Process only the first N images (fast CI mode)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path not found: {input_path}")
        return
    
    gt_path = Path(args.gt) if args.gt else None
    if gt_path and not gt_path.exists():
        print(f"Warning: GT path not found, skipping full-reference metrics")
        gt_path = None
    
    # Get image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = sorted([f for f in input_path.iterdir()
                        if f.suffix.lower() in image_extensions])
    
    if not image_files:
        print(f"No images found in {input_path}")
        return
    
    if args.smoke is not None and args.smoke > 0:
        image_files = image_files[:args.smoke]
        print(f"[SMOKE MODE] Limiting to first {len(image_files)} image(s)")
    
    print(f"Found {len(image_files)} images to evaluate")
    print(f"Metrics: {', '.join(args.metrics)}")
    print()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    all_results = []
    
    for img_path in tqdm(image_files, desc="Evaluating"):
        gt_img_path = None
        if gt_path:
            gt_img_path = gt_path / img_path.name
        
        results = evaluate_image(
            str(img_path),
            str(gt_img_path) if gt_img_path and gt_img_path.exists() else None,
            args.metrics,
            str(device),
        )
        
        results['filename'] = img_path.name
        all_results.append(results)
    
    # Compute averages and per-metric summary stats
    metric_keys = [m for m in args.metrics if m in all_results[0]] if all_results else []
    averages = {}
    summaries = {}
    for key in metric_keys:
        values = [r.get(key, float('nan')) for r in all_results]
        finite = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
        averages[key] = float(np.mean(finite)) if finite else float('nan')
        summaries[key] = _summarize(values)
    
    # Output results
    print("\n" + "=" * 60)
    print("QUALITY EVALUATION RESULTS")
    print("=" * 60)
    
    if args.format == 'table':
        print(f"\n{'Filename':<40}", end='')
        for key in metric_keys:
            print(f"{key.upper():>12}", end='')
        print()
        print("-" * (40 + 12 * len(metric_keys)))
        
        for result in all_results:
            print(f"{result['filename']:<40}", end='')
            for key in metric_keys:
                val = result.get(key, float('nan'))
                print(f"{val:>12.4f}", end='')
            print()
        
        print("-" * (40 + 12 * len(metric_keys)))
        print(f"{'AVERAGE':<40}", end='')
        for key in metric_keys:
            print(f"{averages[key]:>12.4f}", end='')
        print()
    
    else:
        with open(args.output, 'w', newline='') as f:
            fieldnames = ['filename'] + metric_keys
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
            
            # Add average row
            avg_row = {'filename': 'AVERAGE'}
            avg_row.update(averages)
            writer.writerow(avg_row)
        
        print(f"Results saved to: {args.output}")
    
    # Per-metric summary (mean / median / std / min / max)
    print("\nSUMMARY (per metric, NaNs ignored):")
    header = f"  {'METRIC':<10} {'N':>4} {'MEAN':>10} {'MEDIAN':>10} {'STD':>10} {'MIN':>10} {'MAX':>10}  DIRECTION"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in metric_keys:
        s = summaries[key]
        # Resolve direction: pyiqa metrics use PYIQA_DIRECTION; FR metrics hardcoded.
        if key in PYIQA_DIRECTION:
            direction = PYIQA_DIRECTION[key]
        elif key == 'lpips':
            direction = 'lower'
        elif key in ('psnr', 'ssim'):
            direction = 'higher'
        else:
            direction = 'n/a'
        print(
            f"  {key.upper():<10} {s['count']:>4d} "
            f"{s['mean']:>10.4f} {s['median']:>10.4f} {s['std']:>10.4f} "
            f"{s['min']:>10.4f} {s['max']:>10.4f}  ({direction} is better)"
        )
    
    print("\nMetric Interpretation Guide:")
    print("  PSNR: >25 dB is good, >30 dB is excellent (higher is better)")
    print("  SSIM: >0.85 is good, >0.95 is excellent (higher is better)")
    print("  LPIPS: <0.25 is good, <0.1 is excellent (lower is better)")
    print("  NIQE: <5.0 is good, <3.0 is excellent (lower is better)")
    print("  MANIQA: >0.7 is good, >0.85 is excellent (higher is better)")
    print("  CLIPIQA: >0.7 is good, >0.85 is excellent (higher is better)")
    print("  TOPIQ: >0.5 is good, >0.7 is excellent (higher is better)")
    print("  MUSIQ: higher is better; range ~0-100")


if __name__ == '__main__':
    main()