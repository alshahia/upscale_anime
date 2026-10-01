"""
Visualization utilities for training and results
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert torch tensor to numpy image.
    
    Args:
        tensor: [C, H, W] or [B, C, H, W] tensor in range [0, 1]
    
    Returns:
        numpy image [H, W, C] in range [0, 255]
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first batch
    
    # Move to CPU and convert
    img = tensor.detach().cpu().numpy()
    
    # Convert [C, H, W] to [H, W, C]
    img = np.transpose(img, (1, 2, 0))
    
    # Scale to [0, 255]
    img = (img * 255).clip(0, 255).astype(np.uint8)
    
    return img


def save_image_comparison(
    lr: torch.Tensor,
    sr: torch.Tensor,
    hr: Optional[torch.Tensor],
    output_path: str,
    titles: List[str] = None,
):
    """
    Save side-by-side comparison of LR, SR, and HR images.
    
    Args:
        lr: Low-resolution image
        sr: Super-resolution output
        hr: High-resolution ground truth (optional)
        output_path: Where to save
        titles: List of titles for each subplot
    """
    if titles is None:
        titles = ["LR (Input)", "SR (Output)", "HR (Ground Truth)"]
    
    # Convert to images
    lr_img = tensor_to_image(lr)
    sr_img = tensor_to_image(sr)
    
    images = [lr_img, sr_img]
    if hr is not None:
        hr_img = tensor_to_image(hr)
        images.append(hr_img)
    
    # Create figure
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=(5 * n_images, 5))
    
    if n_images == 1:
        axes = [axes]
    
    for i, (img, ax) in enumerate(zip(images, axes)):
        ax.imshow(img)
        ax.set_title(titles[i] if i < len(titles) else "")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(
    history: dict,
    output_path: str,
    metrics: List[str] = None,
):
    """
    Plot training history.
    
    Args:
        history: Dict with 'train' and 'val' lists
        output_path: Where to save plot
        metrics: List of metrics to plot
    """
    if metrics is None:
        metrics = ['loss']
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    for metric, ax in zip(metrics, axes):
        train_key = f"train_{metric}"
        val_key = f"val_{metric}"
        
        if train_key in history:
            ax.plot(history[train_key], label='Train')
        if val_key in history:
            ax.plot(history[val_key], label='Val')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} over time')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_grid_visualization(
    images: List[torch.Tensor],
    output_path: str,
    nrow: int = 4,
    titles: Optional[List[str]] = None,
):
    """
    Create a grid visualization of multiple images.
    
    Args:
        images: List of image tensors
        output_path: Where to save
        nrow: Number of images per row
        titles: Optional titles for each image
    """
    n_images = len(images)
    ncol = (n_images + nrow - 1) // nrow
    
    fig, axes = plt.subplots(ncol, nrow, figsize=(4 * nrow, 4 * ncol))
    
    if ncol == 1:
        axes = axes.reshape(1, -1)
    
    for i, img_tensor in enumerate(images):
        row = i // nrow
        col = i % nrow
        
        img = tensor_to_image(img_tensor)
        axes[row, col].imshow(img)
        
        if titles and i < len(titles):
            axes[row, col].set_title(titles[i])
        
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(n_images, ncol * nrow):
        row = i // nrow
        col = i % nrow
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_degradation(
    hr: torch.Tensor,
    lr: torch.Tensor,
    output_path: str,
):
    """
    Visualize the degradation from HR to LR.
    
    Args:
        hr: High-resolution image
        lr: Low-resolution image
        output_path: Where to save
    """
    # Upsample LR back to HR size for comparison
    lr_up = torch.nn.functional.interpolate(
        lr.unsqueeze(0), 
        size=hr.shape[1:], 
        mode='bicubic', 
        align_corners=False
    ).squeeze(0)
    
    # Create difference map
    diff = torch.abs(hr - lr_up)
    
    # Convert to images
    hr_img = tensor_to_image(hr)
    lr_img = tensor_to_image(lr_up)
    diff_img = tensor_to_image(diff)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(hr_img)
    axes[0].set_title('HR (Original)')
    axes[0].axis('off')
    
    axes[1].imshow(lr_img)
    axes[1].set_title('LR (Degraded + Upsampled)')
    axes[1].axis('off')
    
    axes[2].imshow(diff_img)
    axes[2].set_title('Difference Map')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_batch_comparison(
    lr_batch: torch.Tensor,
    sr_batch: torch.Tensor,
    hr_batch: Optional[torch.Tensor],
    output_dir: str,
    batch_idx: int,
    max_samples: int = 4,
):
    """
    Save comparison images for a batch.
    
    Args:
        lr_batch: Batch of LR images [B, C, H, W]
        sr_batch: Batch of SR images [B, C, H, W]
        hr_batch: Batch of HR images [B, C, H, W] (optional)
        output_dir: Directory to save images
        batch_idx: Batch index for naming
        max_samples: Maximum number of samples to save
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_samples = min(lr_batch.size(0), max_samples)
    
    for i in range(n_samples):
        lr = lr_batch[i]
        sr = sr_batch[i]
        hr = hr_batch[i] if hr_batch is not None else None
        
        output_path = output_dir / f"batch{batch_idx:04d}_sample{i}.png"
        save_image_comparison(lr, sr, hr, str(output_path))


def log_image_to_tensorboard(
    writer,
    tag: str,
    image: torch.Tensor,
    global_step: int,
):
    """
    Log image to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        tag: Tag for the image
        image: Image tensor [C, H, W] or [B, C, H, W]
        global_step: Global step
    """
    if writer is None:
        return
    
    # Convert to [0, 1] range if needed
    if image.max() > 1.5:
        image = image / 255.0
    
    # Add batch dimension if needed
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    # Create grid
    from torchvision.utils import make_grid
    grid = make_grid(image, nrow=4, normalize=True, value_range=(0, 1))
    
    writer.add_image(tag, grid, global_step)
