#!/usr/bin/env python
"""
Test-time adaptation for super-resolution.
Adapts model to specific test image using self-supervised learning.
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from anime_sr.models.span import create_span_model
from anime_sr.utils.image_utils import tensor_to_image, image_to_tensor


class TestTimeAdapter:
    """
    Test-time adaptation using self-supervised learning.
    Fine-tunes model on patches from the test image itself.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 adaptation_steps: int = 50,
                 lr: float = 1e-5,
                 patch_size: int = 64,
                 num_patches: int = 16,
                 mask_ratio: float = 0.3):
        """
        Args:
            model: Pre-trained SR model
            adaptation_steps: Number of fine-tuning steps
            lr: Learning rate for adaptation
            patch_size: Size of patches to extract
            num_patches: Number of patches to use
            mask_ratio: Ratio of pixels to mask for self-supervision
        """
        self.model = model
        self.adaptation_steps = adaptation_steps
        self.lr = lr
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio
        
        # Create optimizer for adaptation
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999)
        )
    
    def extract_patches(self, lr_image: torch.Tensor) -> torch.Tensor:
        """
        Extract random patches from LR image.
        
        Args:
            lr_image: (1, C, H, W) LR image
        
        Returns:
            (N, C, patch_size, patch_size) patches
        """
        _, c, h, w = lr_image.shape
        
        patches = []
        for _ in range(self.num_patches):
            # Random position
            if h > self.patch_size and w > self.patch_size:
                x = np.random.randint(0, h - self.patch_size + 1)
                y = np.random.randint(0, w - self.patch_size + 1)
                patch = lr_image[:, :, x:x+self.patch_size, y:y+self.patch_size]
            else:
                # Resize if image is smaller
                patch = F.interpolate(
                    lr_image,
                    size=(self.patch_size, self.patch_size),
                    mode='bicubic',
                    align_corners=False
                )
            
            patches.append(patch)
        
        return torch.cat(patches, dim=0)
    
    def create_mask(self, shape: tuple) -> torch.Tensor:
        """Create random mask for masked prediction."""
        mask = torch.rand(shape) > self.mask_ratio
        return mask.float()
    
    def masked_reconstruction_loss(self, 
                                   prediction: torch.Tensor,
                                   target: torch.Tensor,
                                   mask: torch.Tensor) -> torch.Tensor:
        """
        Compute loss only on masked (visible) regions.
        
        Args:
            prediction: (N, C, H, W) predicted HR
            target: (N, C, H, W) target HR (ground truth from bicubic upscale)
            mask: (N, C, H, W) mask (1 = visible, 0 = masked)
        
        Returns:
            Masked reconstruction loss
        """
        # Apply mask
        masked_pred = prediction * mask
        masked_target = target * mask
        
        # Compute L1 loss on visible regions
        loss = F.l1_loss(masked_pred, masked_target)
        
        # Normalize by number of visible pixels
        num_visible = mask.sum()
        if num_visible > 0:
            loss = loss * (mask.numel() / num_visible)
        
        return loss
    
    def adapt(self, lr_image: torch.Tensor, device: str = 'cuda') -> nn.Module:
        """
        Adapt model to test image.
        
        Args:
            lr_image: (1, C, H, W) LR image
            device: Device to use
        
        Returns:
            Adapted model
        """
        self.model.train()
        self.model.to(device)
        lr_image = lr_image.to(device)
        
        print(f"[TTA] Starting adaptation with {self.adaptation_steps} steps...")
        
        for step in tqdm(range(self.adaptation_steps), desc="Adapting"):
            # Extract patches
            patches = self.extract_patches(lr_image)
            
            # Create HR target via bicubic interpolation
            scale = 4  # Assuming 4x SR
            target_hr = F.interpolate(
                patches,
                scale_factor=scale,
                mode='bicubic',
                align_corners=False
            )
            
            # Forward pass
            pred_hr = self.model(patches)
            
            # Create random mask
            mask = self.create_mask(pred_hr.shape).to(device)
            
            # Compute loss
            loss = self.masked_reconstruction_loss(pred_hr, target_hr, mask)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if step % 10 == 0:
                print(f"  Step {step}/{self.adaptation_steps}: Loss = {loss.item():.6f}")
        
        print("[TTA] Adaptation complete")
        self.model.eval()
        
        return self.model
    
    def forward(self, lr_image: torch.Tensor, device: str = 'cuda') -> torch.Tensor:
        """
        Adapt model and process image.
        
        Args:
            lr_image: (1, C, H, W) LR image
            device: Device to use
        
        Returns:
            (1, C, H*4, W*4) SR output
        """
        # Adapt model
        adapted_model = self.adapt(lr_image, device)
        
        # Process full image
        with torch.no_grad():
            sr_output = adapted_model(lr_image.to(device))
        
        return sr_output


def load_model(checkpoint_path: str, device: str = 'cuda') -> nn.Module:
    """Load SR model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    
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


def process_image(input_path: str, 
                 output_path: str,
                 checkpoint_path: str,
                 adaptation_steps: int = 50,
                 device: str = 'cuda'):
    """
    Process single image with test-time adaptation.
    
    Args:
        input_path: Path to input LR image
        output_path: Path to save SR output
        checkpoint_path: Path to model checkpoint
        adaptation_steps: Number of adaptation steps
        device: Device to use
    """
    # Load image
    print(f"Loading image: {input_path}")
    img = Image.open(input_path).convert('RGB')
    lr_tensor = image_to_tensor(img).unsqueeze(0)
    
    # Load model
    model = load_model(checkpoint_path, device)
    
    # Create adapter and process
    adapter = TestTimeAdapter(
        model=model,
        adaptation_steps=adaptation_steps
    )
    
    sr_tensor = adapter.forward(lr_tensor, device)
    
    # Save output
    sr_img = tensor_to_image(sr_tensor.squeeze(0))
    sr_img.save(output_path)
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Test-time adaptation for SR')
    parser.add_argument('--input', type=str, required=True, help='Input LR image path')
    parser.add_argument('--output', type=str, required=True, help='Output SR image path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--adaptation-steps', type=int, default=50, help='Number of adaptation steps')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--lr', type=float, default=1e-5, help='Adaptation learning rate')
    
    args = parser.parse_args()
    
    process_image(
        input_path=args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        adaptation_steps=args.adaptation_steps,
        device=args.device
    )


if __name__ == '__main__':
    main()
