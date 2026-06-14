#!/usr/bin/env python3
"""
Evaluation script for benchmarking on standard datasets
Computes PSNR, SSIM metrics
"""
import sys
import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.span import create_span_model
from utils.config import Config
from utils.metrics import calculate_psnr, calculate_ssim, calculate_batch_metrics


def load_model(checkpoint_path: str, config_path: str = None, device: str = 'cuda'):
    """Load trained model from checkpoint"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load config if provided
    if config_path:
        config = Config.load(config_path)
        model = create_span_model(config.get('model', {}))
    else:
        model = create_span_model({'type': 'span', 'scale': 4})
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, device


def load_image_pair(lr_path: Path, hr_path: Path, scale: int = 4):
    """Load LR and HR image pair"""
    # Load HR
    hr = cv2.imread(str(hr_path), cv2.IMREAD_COLOR)
    if hr is None:
        raise ValueError(f"Failed to load HR image: {hr_path}")
    hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
    
    # Load or generate LR
    if lr_path.exists():
        lr = cv2.imread(str(lr_path), cv2.IMREAD_COLOR)
        if lr is None:
            raise ValueError(f"Failed to load LR image: {lr_path}")
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
    else:
        # Bicubic downsample
        h, w = hr.shape[:2]
        lr_h, lr_w = h // scale, w // scale
        lr = cv2.resize(hr, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
    
    # Convert to tensors
    lr = torch.from_numpy(lr.transpose(2, 0, 1)).float() / 255.0
    hr = torch.from_numpy(hr.transpose(2, 0, 1)).float() / 255.0
    
    # Add batch dimension
    lr = lr.unsqueeze(0)
    hr = hr.unsqueeze(0)
    
    return lr, hr


def evaluate_dataset(model, device, lr_dir: Path, hr_dir: Path, scale: int = 4) -> dict:
    """Evaluate on a dataset"""
    # Find all HR images
    hr_images = sorted(hr_dir.glob('*.png')) + sorted(hr_dir.glob('*.jpg'))
    
    psnr_list = []
    ssim_list = []
    
    pbar = tqdm(hr_images, desc="Evaluating")
    
    for hr_path in pbar:
        # Construct LR path
        lr_path = lr_dir / hr_path.name
        
        try:
            # Load pair
            lr, hr = load_image_pair(lr_path, hr_path, scale)
            lr = lr.to(device)
            hr = hr.to(device)
            
            # Inference
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    sr = model(lr)
            
            # Crop to match HR size (if needed)
            h, w = hr.shape[2:]
            sr = sr[:, :, :h, :w]
            
            # Calculate metrics
            psnr = calculate_psnr(sr, hr, max_val=1.0)
            ssim = calculate_ssim(sr, hr, max_val=1.0)
            
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            
            # Update progress bar
            pbar.set_postfix({
                'PSNR': f"{psnr:.2f}",
                'SSIM': f"{ssim:.4f}"
            })
            
        except Exception as e:
            print(f"Error processing {hr_path}: {e}")
            continue
    
    # Calculate averages
    results = {
        'num_images': len(psnr_list),
        'psnr_mean': np.mean(psnr_list) if psnr_list else 0,
        'psnr_std': np.std(psnr_list) if psnr_list else 0,
        'ssim_mean': np.mean(ssim_list) if ssim_list else 0,
        'ssim_std': np.std(ssim_list) if ssim_list else 0,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Super-Resolution Model')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--dataset', '-d', type=str, required=True,
                      help='Dataset name or path')
    parser.add_argument('--hr-dir', type=str,
                      help='HR images directory (if not using standard dataset)')
    parser.add_argument('--lr-dir', type=str,
                      help='LR images directory (optional)')
    parser.add_argument('--config', type=str,
                      help='Path to model config YAML')
    parser.add_argument('--scale', type=int, default=4,
                      help='Upsampling factor')
    parser.add_argument('--device', type=str, default='cuda',
                      choices=['cuda', 'cpu'],
                      help='Device for evaluation')
    parser.add_argument('--output', '-o', type=str,
                      help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, device = load_model(args.checkpoint, args.config, args.device)
    print(f"Device: {device}")
    
    # Determine dataset path
    data_root = Path('data')
    
    if args.hr_dir:
        # Custom dataset
        hr_dir = Path(args.hr_dir)
        lr_dir = Path(args.lr_dir) if args.lr_dir else hr_dir.parent / f'{hr_dir.name}_LR'
    else:
        # Standard dataset
        dataset_name = args.dataset.lower()
        
        dataset_paths = {
            'set5': (data_root / 'Set5/LR_bicubic' / f'X{args.scale}', data_root / 'Set5/HR'),
            'set14': (data_root / 'Set14/LR_bicubic' / f'X{args.scale}', data_root / 'Set14/HR'),
            'urban100': (data_root / 'Urban100/LR_bicubic' / f'X{args.scale}', data_root / 'Urban100/HR'),
            'bsd100': (data_root / 'BSD100/LR_bicubic' / f'X{args.scale}', data_root / 'BSD100/HR'),
            'div2k': (data_root / 'DIV2K/DIV2K_valid_LR_bicubic' / f'X{args.scale}', data_root / 'DIV2K/DIV2K_valid_HR'),
        }
        
        if dataset_name not in dataset_paths:
            print(f"Unknown dataset: {dataset_name}")
            print(f"Available: {list(dataset_paths.keys())}")
            return
        
        lr_dir, hr_dir = dataset_paths[dataset_name]
    
    if not hr_dir.exists():
        print(f"HR directory not found: {hr_dir}")
        print("Please download the dataset first")
        return
    
    print(f"\nEvaluating on {args.dataset}")
    print(f"HR dir: {hr_dir}")
    print(f"LR dir: {lr_dir}")
    
    # Run evaluation
    results = evaluate_dataset(model, device, lr_dir, hr_dir, args.scale)
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Results on {args.dataset}:")
    print(f"{'='*50}")
    print(f"Number of images: {results['num_images']}")
    print(f"PSNR:  {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
    print(f"SSIM:  {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
    print(f"{'='*50}")
    
    # Save results
    if args.output:
        results['dataset'] = args.dataset
        results['checkpoint'] = args.checkpoint
        results['scale'] = args.scale
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
