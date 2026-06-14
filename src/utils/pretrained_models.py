"""
Pretrained model downloader for teacher models.
Downloads EDSR, RCAN, and SwinIR models from reliable sources.
"""
import os
import torch
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Callable
from tqdm import tqdm


# Model URLs from reliable sources
# EDSR/RCAN: HuggingFace (eugenesiow) - PyTorch format
# SwinIR: Official GitHub releases
MODEL_URLS: Dict[str, Dict[str, str]] = {
    "edsr": {
        "x2": "https://huggingface.co/eugenesiow/edsr-base/resolve/main/pytorch_model_2x.pt",
        "x3": "https://huggingface.co/eugenesiow/edsr-base/resolve/main/pytorch_model_3x.pt",
        "x4": "https://huggingface.co/eugenesiow/edsr-base/resolve/main/pytorch_model_4x.pt",
    },
    "rcan": {
        # RCAN only has x4 available on HuggingFace
        "x2": "",  # Not available - would need manual download from Google Drive
        "x3": "",  # Not available - would need manual download from Google Drive
        "x4": "https://huggingface.co/eugenesiow/rcan-bam/resolve/main/pytorch_model_4x.pt",
    },
    "swinir": {
        "x2": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth",
        "x3": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x3.pth",
        "x4": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth",
    },
}

# Alternative filenames used when saving
MODEL_FILENAMES: Dict[str, Dict[str, str]] = {
    "edsr": {
        "x2": "EDSR_x2.pt",
        "x3": "EDSR_x3.pt",
        "x4": "EDSR_x4.pt",
    },
    "rcan": {
        "x2": "RCAN_x2.pt",
        "x3": "RCAN_x3.pt",
        "x4": "RCAN_x4.pt",
    },
    "swinir": {
        "x2": "SwinIR_x2.pt",
        "x3": "SwinIR_x3.pt",
        "x4": "SwinIR_x4.pt",
    },
    "spanf": {
        "x4": "SPANF_x4.pth",  # NTIRE 2025 2nd place (XiaomiMM)
    },
}

# Model information for display
MODEL_INFO: Dict[str, Dict[str, str]] = {
    "edsr": {
        "name": "EDSR",
        "description": "Enhanced Deep Residual Networks for Single Image Super-Resolution",
        "paper": "CVPR 2017",
    },
    "rcan": {
        "name": "RCAN",
        "description": "Residual Channel Attention Networks",
        "paper": "ECCV 2018",
    },
    "swinir": {
        "name": "SwinIR",
        "description": "Image Restoration Using Swin Transformer",
        "paper": "ICCV 2021",
    },
    "spanf": {
        "name": "SPAN-F",
        "description": "SPAN-Fast: NTIRE 2025 2nd Place Efficient SR (XiaomiMM)",
        "paper": "CVPRW 2025",
        "download_url": "https://drive.google.com/file/d/1iYUA2TzKuxI0vzmA-UXr_nB43XgPOXUg/view?usp=sharing",
        "github": "https://github.com/hongyuanyu/SPAN",
        "params": "~300K",
        "architecture": "32 channels, 10 blocks",
    },
}


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: str, desc: Optional[str] = None) -> bool:
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        output_path: Local path to save file
        desc: Description for progress bar
        
    Returns:
        True if successful, False otherwise
    """
    # Check if URL is empty (model not available for this scale)
    if not url:
        print(f"  Model not available for this scale")
        return False
    
    try:
        desc = desc or f"Downloading {os.path.basename(output_path)}"
        
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Set up request with headers to avoid 403 errors
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Download with progress bar
        request = urllib.request.Request(url, headers=headers)
        
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
            # Open URL with custom headers
            with urllib.request.urlopen(request, timeout=300) as response:
                # Get total size if available
                if 'Content-Length' in response.headers:
                    t.total = int(response.headers['Content-Length'])
                
                # Read and write in chunks
                chunk_size = 8192
                with open(output_path, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        t.update(len(chunk))
        
        return True
        
    except urllib.error.HTTPError as e:
        print(f"  HTTP Error {e.code}: {e.reason}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False
    except Exception as e:
        print(f"  Error downloading: {e}")
        # Clean up partial download
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def get_pretrained_dir(config: Optional[Dict] = None) -> Path:
    """
    Get the pretrained models directory.
    
    Args:
        config: Optional config dict with paths.pretrained_dir
        
    Returns:
        Path to pretrained directory
    """
    if config and 'paths' in config and 'pretrained_dir' in config['paths']:
        pretrained_dir = Path(config['paths']['pretrained_dir'])
    else:
        # Default to project_root/pretrained
        # Try to find project root by looking for common markers
        current = Path.cwd()
        while current != current.parent:
            if (current / 'src').exists() or (current / 'configs').exists():
                return current / 'pretrained'
            current = current.parent
        
        # Fallback to current working directory
        pretrained_dir = Path('pretrained')
    
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    return pretrained_dir


def check_model_exists(model_name: str, scale: int = 4, config: Optional[Dict] = None) -> bool:
    """
    Check if a pretrained model exists locally.
    
    Args:
        model_name: Model name (edsr, rcan, swinir)
        scale: Upscaling factor (2, 3, 4)
        config: Optional config dict
        
    Returns:
        True if model exists locally
    """
    pretrained_dir = get_pretrained_dir(config)
    scale_key = f"x{scale}"
    
    if model_name.lower() not in MODEL_FILENAMES:
        return False
    
    if scale_key not in MODEL_FILENAMES[model_name.lower()]:
        return False
    
    filename = MODEL_FILENAMES[model_name.lower()][scale_key]
    model_path = pretrained_dir / filename
    
    return model_path.exists()


def download_pretrained_model(
    model_name: str, 
    scale: int = 4, 
    config: Optional[Dict] = None,
    force: bool = False
) -> Optional[Path]:
    """
    Download a pretrained teacher model if it doesn't exist.
    
    Args:
        model_name: Model name (edsr, rcan, swinir)
        scale: Upscaling factor (2, 3, 4)
        config: Optional config dict
        force: Re-download even if file exists
        
    Returns:
        Path to downloaded model or None if failed
    """
    model_name = model_name.lower()
    scale_key = f"x{scale}"
    
    # Validate model name
    if model_name not in MODEL_URLS:
        print(f"Unknown model: {model_name}")
        print(f"Available models: {list(MODEL_URLS.keys())}")
        return None
    
    # Validate scale
    if scale_key not in MODEL_URLS[model_name]:
        print(f"Scale {scale} not available for {model_name}")
        print(f"Available scales: {list(MODEL_URLS[model_name].keys())}")
        return None
    
    # Check if URL is available for this scale
    url = MODEL_URLS[model_name][scale_key]
    if not url:
        print(f"{model_name.upper()} x{scale}: Not available for auto-download")
        if model_name == "rcan" and scale in [2, 3]:
            print("  Please download manually from:")
            print("  https://drive.google.com/file/d/1U0RmWkkacyw0HNC7CBts2LcX1ewAulCn/view?usp=sharing")
            print("  Or use the Dropbox link in the RCAN repository")
        return None
    
    # Get paths
    pretrained_dir = get_pretrained_dir(config)
    filename = MODEL_FILENAMES[model_name][scale_key]
    output_path = pretrained_dir / filename
    
    # Check if already exists
    if output_path.exists() and not force:
        print(f"Model already exists: {output_path}")
        return output_path
    
    # Download
    print(f"Downloading {model_name.upper()} x{scale}")
    print(f"  From: {url[:60]}...")
    
    if download_file(url, str(output_path), desc=f"  {model_name.upper()} x{scale}"):
        # Verify file size is reasonable
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"  ✓ Downloaded: {file_size:.1f} MB")
        print(f"  ✓ Saved to: {output_path}")
        return output_path
    else:
        print(f"  ✗ Failed to download {model_name.upper()} x{scale}")
        return None


def download_all_teacher_models(
    scale: int = 4, 
    config: Optional[Dict] = None,
    force: bool = False
) -> Dict[str, Optional[Path]]:
    """
    Download all teacher models (EDSR, RCAN, SwinIR).
    
    Args:
        scale: Upscaling factor (2, 3, 4)
        config: Optional config dict
        force: Re-download even if files exist
        
    Returns:
        Dict mapping model names to paths (None if download failed)
    """
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Downloading Teacher Models (scale={scale})")
    print(f"{'='*60}\n")
    
    for model_name in ['edsr', 'rcan', 'swinir']:
        path = download_pretrained_model(model_name, scale, config, force)
        results[model_name] = path
        print()
    
    # Summary
    print(f"{'='*60}")
    print("Download Summary")
    print(f"{'='*60}")
    
    success_count = sum(1 for p in results.values() if p is not None)
    total_count = len(results)
    
    for model_name, path in results.items():
        status = "✓" if path else "✗"
        location = str(path) if path else "Failed"
        print(f"{status} {model_name.upper()}: {location}")
    
    print(f"\nTotal: {success_count}/{total_count} models ready")
    print(f"{'='*60}\n")
    
    return results


def get_teacher_model_path(
    model_name: str, 
    scale: int = 4, 
    config: Optional[Dict] = None,
    auto_download: bool = True
) -> Optional[Path]:
    """
    Get path to a teacher model, downloading if necessary.
    
    Args:
        model_name: Model name (edsr, rcan, swinir)
        scale: Upscaling factor (2, 3, 4)
        config: Optional config dict
        auto_download: Download model if not found
        
    Returns:
        Path to model or None if not available
    """
    model_name = model_name.lower()
    scale_key = f"x{scale}"
    
    # Check if valid model
    if model_name not in MODEL_FILENAMES:
        print(f"Unknown model: {model_name}")
        return None
    
    if scale_key not in MODEL_FILENAMES[model_name]:
        print(f"Scale {scale} not available for {model_name}")
        return None
    
    # Get local path
    pretrained_dir = get_pretrained_dir(config)
    filename = MODEL_FILENAMES[model_name][scale_key]
    local_path = pretrained_dir / filename
    
    # Return if exists
    if local_path.exists():
        return local_path
    
    # Auto-download if enabled
    if auto_download:
        print(f"Model not found locally: {local_path}")
        return download_pretrained_model(model_name, scale, config)
    
    return None


def verify_model_weights(model_path: str) -> bool:
    """
    Verify that a model file can be loaded.
    
    Args:
        model_path: Path to model file
        
    Returns:
        True if model can be loaded
    """
    try:
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        except Exception:
            # Fallback for checkpoints with custom objects
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Check for state dict
        if 'state_dict' in checkpoint or 'params' in checkpoint or 'model' in checkpoint:
            return True
        
        # Check if it's a raw state dict
        if isinstance(checkpoint, dict) and len(checkpoint) > 0:
            return True
        
        return False
        
    except Exception as e:
        print(f"Error verifying model: {e}")
        return False


def list_available_models():
    """Print list of available pretrained models."""
    print(f"\n{'='*60}")
    print("Available Pretrained Models")
    print(f"{'='*60}\n")
    
    for model_name, scales in MODEL_URLS.items():
        print(f"{model_name.upper()}:")
        for scale in scales.keys():
            filename = MODEL_FILENAMES[model_name][scale]
            print(f"  - {scale}: {filename}")
        print()
    
    print(f"{'='*60}")
    print("Models will be saved to: ./pretrained/")
    print(f"{'='*60}\n")


# For command-line usage
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download pretrained teacher models')
    parser.add_argument('--model', '-m', type=str, choices=['edsr', 'rcan', 'swinir', 'all'],
                       default='all', help='Model to download')
    parser.add_argument('--scale', '-s', type=int, choices=[2, 3, 4], 
                       default=4, help='Upscaling factor')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Re-download even if file exists')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available models')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
    elif args.model == 'all':
        download_all_teacher_models(args.scale, force=args.force)
    else:
        path = download_pretrained_model(args.model, args.scale, force=args.force)
        if path:
            print(f"\nModel ready: {path}")
            # Verify the model
            if verify_model_weights(str(path)):
                print("✓ Model verification passed")
            else:
                print("✗ Model verification failed")
        else:
            print("\n✗ Download failed")
