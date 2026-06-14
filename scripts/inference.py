#!/usr/bin/env python3
"""
Inference script for upscaling single images or batches
"""
import sys
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.span import create_span_model, create_neosr_span
from utils.config import Config


def load_model(checkpoint_path: str, config_path: str = None, device: str = 'cuda', model_type: str = 'span'):
    """Load trained model from checkpoint"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load config if provided
    if config_path:
        config = Config.load(config_path)
        model_cfg = config.get('model', {})
        # Detect model type from config
        if model_cfg.get('type') == 'neosr_span':
            model = create_neosr_span(model_cfg)
        else:
            model = create_span_model(model_cfg)
    else:
        # Default model config
        if model_type == 'neosr_span':
            model = create_neosr_span({'type': 'neosr_span', 'scale': 4})
        else:
            model = create_span_model({'type': 'checkpoint_compatible', 'scale': 4})
    
    # Load checkpoint (handle different checkpoint formats)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        # Check if this is a checkpoint-compatible model
        model_type = model.__class__.__name__
        if 'CheckpointCompatible' in model_type or 'Exact' in model_type:
            # Use specialized loader for checkpoint-compatible models
            from models.span.checkpoint_compatible_exact import load_checkpoint_compatible_weights_exact
            success, loading_info = load_checkpoint_compatible_weights_exact(
                model, checkpoint_path, device=device, verbose=True
            )
            if not success:
                print(f"Warning: Only {loading_info['loaded_params']}/{loading_info['total_checkpoint_params']} parameters loaded")
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'params' in checkpoint:
        # Use specialized loader for checkpoint-compatible models
        from models.span.checkpoint_compatible_exact import load_checkpoint_compatible_weights_exact
        success, loading_info = load_checkpoint_compatible_weights_exact(
            model, checkpoint_path, device=device, verbose=True
        )
        if not success:
            print(f"Warning: Only {loading_info['loaded_params']}/{loading_info['total_checkpoint_params']} parameters loaded")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    return model, device


def preprocess_image(image_path: str, scale: int = 4) -> torch.Tensor:
    """Load and preprocess image for inference"""
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor [C, H, W], float32, [0, 1]
    img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    
    # Add batch dimension
    img = img.unsqueeze(0)
    
    return img


def postprocess_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert model output to numpy image"""
    # Remove batch dimension and clamp
    tensor = tensor.squeeze(0).clamp(0, 1)
    
    # Convert to numpy [H, W, C], uint8
    img = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img


def inference(model, lr_image: torch.Tensor, device: torch.device, use_amp: bool = True) -> torch.Tensor:
    """Run inference on a single image"""
    lr_image = lr_image.to(device)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=use_amp and device.type == 'cuda'):
            output = model(lr_image)
    
    return output

#python scripts/inference.py --checkpoint checkpoints\NEOSR_SPAN_V5_FULL_004\finetune_best.pth --input data/test_hr/1.png --output data/test_hr/test_sr_1.png
def main():
    parser = argparse.ArgumentParser(description='Super-Resolution Inference')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input image path or directory')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output image path or directory')
    parser.add_argument('--config', type=str,
                       help='Path to model config YAML')
    parser.add_argument('--scale', type=int, default=4,
                       help='Upsampling factor')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for inference')
    parser.add_argument('--batch', action='store_true',
                       help='Process directory of images')
    parser.add_argument('--fp16', action='store_true',
                       help='Use FP16 for faster inference')
    parser.add_argument('--model-type', type=str, default='span',
                       choices=['span', 'neosr_span', 'mamba_pan'],
                       help='Model architecture type')
    parser.add_argument('--gt', type=str, default=None,
                       help='Ground truth image for metrics computation')
    parser.add_argument('--metrics', action='store_true',
                       help='Compute quality metrics (requires --gt for full-ref metrics)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path not found: {args.input}")
        return
    
    # Load model
    print("Loading model...")
    model, device = load_model(args.checkpoint, args.config, args.device, args.model_type)
    
    output_path = Path(args.output)
    
    if args.batch or input_path.is_dir():
        # Batch processing
        if not input_path.is_dir():
            print(f"Error: --batch specified but input is not a directory: {input_path}")
            return
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"Found {len(image_files)} images to process")
        
        for img_path in tqdm(image_files, desc="Processing"):
            try:
                # Load
                lr = preprocess_image(str(img_path), args.scale)
                
                # Inference
                sr = inference(model, lr, device, args.fp16)
                
                # Save
                output_img = postprocess_image(sr)
                output_file = output_path / f"{img_path.stem}_sr{img_path.suffix}"
                cv2.imwrite(str(output_file), output_img)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        print(f"\nResults saved to: {output_path}")
        
    else:
        # Single image processing
        print(f"Processing: {input_path}")
        
        # Load
        lr = preprocess_image(str(input_path), args.scale)
        
        # Inference
        import time
        start = time.time()
        sr = inference(model, lr, device, args.fp16)
        elapsed = time.time() - start
        
        # Save
        output_img = postprocess_image(sr)
        cv2.imwrite(str(output_path), output_img)
        
        # Report
        input_h, input_w = lr.shape[2:]
        output_h, output_w = sr.shape[2:]
        print(f"\nInput: {input_w}x{input_h}")
        print(f"Output: {output_w}x{output_h}")
        print(f"Inference time: {elapsed:.3f}s")
        print(f"Saved to: {output_path}")
        
        # Compute quality metrics if ground truth is provided
        if args.gt and Path(args.gt).exists():
            try:
                gt = preprocess_image(str(args.gt), args.scale)
                gt = gt.to(device)
                sr_for_metrics = sr.to(device)
                
                # Resize SR to match GT if needed
                if sr_for_metrics.shape != gt.shape:
                    sr_for_metrics = torch.nn.functional.interpolate(
                        sr_for_metrics, size=gt.shape[2:], mode='bicubic', align_corners=False
                    )
                
                from utils.metrics import calculate_psnr, calculate_ssim, calculate_lpips
                
                psnr = calculate_psnr(sr_for_metrics, gt)
                ssim = calculate_ssim(sr_for_metrics, gt)
                
                print(f"\nQuality Metrics (vs GT):")
                print(f"  PSNR: {psnr:.2f} dB")
                print(f"  SSIM: {ssim:.4f}")
                
                if args.metrics:
                    try:
                        lpips_val = calculate_lpips(sr_for_metrics, gt)
                        print(f"  LPIPS: {lpips_val:.4f} (lower is better)")
                    except Exception:
                        pass
                    
                    try:
                        from utils.metrics import calculate_niqe, calculate_maniqa, calculate_clipiqa
                        niqe = calculate_niqe(sr)
                        maniqa = calculate_maniqa(sr)
                        clipiqa = calculate_clipiqa(sr)
                        print(f"\nNo-Reference Metrics:")
                        print(f"  NIQE: {niqe:.4f} (lower is better)")
                        print(f"  MANIQA: {maniqa:.4f} (higher is better)")
                        print(f"  CLIPIQA: {clipiqa:.4f} (higher is better)")
                    except Exception:
                        pass
            except Exception as e:
                print(f"Warning: Could not compute metrics: {e}")


if __name__ == '__main__':
    main()
