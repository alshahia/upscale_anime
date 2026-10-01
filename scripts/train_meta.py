#!/usr/bin/env python
"""
Meta-learning (MAML-style) training for super-resolution.
Trains model to quickly adapt to new images.
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm
import copy

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from anime_sr.models.span import create_span_model
from anime_sr.training.base_trainer import BaseTrainer
from anime_sr.data.base import BaseDataset
from torch.utils.data import DataLoader


class MetaTask:
    """Single task for meta-learning (image-specific adaptation)."""
    
    def __init__(self, 
                 support_lr: torch.Tensor,
                 support_hr: torch.Tensor,
                 query_lr: torch.Tensor,
                 query_hr: torch.Tensor):
        """
        Args:
            support_lr: Support set LR patches
            support_hr: Support set HR patches
            query_lr: Query set LR patches
            query_hr: Query set HR patches
        """
        self.support_lr = support_lr
        self.support_hr = support_hr
        self.query_lr = query_lr
        self.query_hr = query_hr


class MAMLTrainer:
    """
    Model-Agnostic Meta-Learning (MAML) trainer for SR.
    """
    
    def __init__(self,
                 model: nn.Module,
                 inner_lr: float = 1e-2,
                 inner_steps: int = 5,
                 meta_lr: float = 1e-4,
                 first_order: bool = True):
        """
        Args:
            model: Base model to meta-train
            inner_lr: Learning rate for inner loop (task adaptation)
            inner_steps: Number of gradient steps in inner loop
            meta_lr: Learning rate for meta-update (outer loop)
            first_order: Use first-order approximation (FOMAML)
        """
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_lr = meta_lr
        self.first_order = first_order
        
        # Meta-optimizer
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.meta_lr
        )
        
        self.criterion = nn.L1Loss()
    
    def inner_loop(self, task: MetaTask) -> nn.Module:
        """
        Inner loop: Adapt model to a specific task.
        
        Args:
            task: Meta-task with support and query sets
        
        Returns:
            Adapted model (fast weights)
        """
        # Clone model for this task
        adapted_model = copy.deepcopy(self.model)
        adapted_optimizer = torch.optim.SGD(
            adapted_model.parameters(),
            lr=self.inner_lr
        )
        
        # Inner loop updates
        for _ in range(self.inner_steps):
            # Forward on support set
            pred_hr = adapted_model(task.support_lr)
            loss = self.criterion(pred_hr, task.support_hr)
            
            # Backward and update
            adapted_optimizer.zero_grad()
            loss.backward()
            adapted_optimizer.step()
        
        return adapted_model
    
    def outer_loop(self, tasks: List[MetaTask]) -> float:
        """
        Outer loop: Meta-update across multiple tasks.
        
        Args:
            tasks: List of meta-tasks
        
        Returns:
            Average meta-loss
        """
        meta_loss = 0.0
        
        for task in tasks:
            # Inner loop adaptation
            adapted_model = self.inner_loop(task)
            
            # Evaluate on query set
            query_pred = adapted_model(task.query_lr)
            task_loss = self.criterion(query_pred, task.query_hr)
            
            meta_loss += task_loss
        
        meta_loss = meta_loss / len(tasks)
        
        # Meta-update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def train_epoch(self, task_loader: DataLoader, device: str = 'cuda') -> float:
        """
        Train for one meta-learning epoch.
        
        Args:
            task_loader: DataLoader that yields batches of tasks
            device: Device to use
        
        Returns:
            Average epoch loss
        """
        self.model.to(device)
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_tasks in tqdm(task_loader, desc="Meta-training"):
            # Move tasks to device
            tasks = []
            for task in batch_tasks:
                tasks.append(MetaTask(
                    support_lr=task['support_lr'].to(device),
                    support_hr=task['support_hr'].to(device),
                    query_lr=task['query_lr'].to(device),
                    query_hr=task['query_hr'].to(device)
                ))
            
            # Meta-update
            loss = self.outer_loop(tasks)
            total_loss += loss
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0


class TaskDataset:
    """
    Dataset that creates meta-learning tasks from images.
    """
    
    def __init__(self,
                 image_paths: List[Path],
                 num_patches_per_image: int = 4,
                 support_ratio: float = 0.5,
                 patch_size: int = 64,
                 scale: int = 4):
        """
        Args:
            image_paths: List of HR image paths
            num_patches_per_image: Number of patches to extract per image
            support_ratio: Ratio of patches for support set
            patch_size: Size of HR patches
            scale: SR scale factor
        """
        self.image_paths = image_paths
        self.num_patches_per_image = num_patches_per_image
        self.support_ratio = support_ratio
        self.patch_size = patch_size
        self.scale = scale
        self.lr_size = patch_size // scale
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx) -> Dict:
        """Get a meta-task from a single image."""
        from PIL import Image
        from anime_sr.data.base import BaseDataset
        
        # Load image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        
        _, h, w = img_tensor.shape
        
        # Extract random patches
        hr_patches = []
        for _ in range(self.num_patches_per_image):
            if h > self.patch_size and w > self.patch_size:
                x = np.random.randint(0, h - self.patch_size + 1)
                y = np.random.randint(0, w - self.patch_size + 1)
                patch = img_tensor[:, x:x+self.patch_size, y:y+self.patch_size]
            else:
                # Resize if image is smaller
                patch = F.interpolate(
                    img_tensor.unsqueeze(0),
                    size=(self.patch_size, self.patch_size),
                    mode='bicubic'
                ).squeeze(0)
            
            hr_patches.append(patch)
        
        hr_patches = torch.stack(hr_patches)
        
        # Create LR patches
        lr_patches = F.interpolate(
            hr_patches,
            size=(self.lr_size, self.lr_size),
            mode='bicubic',
            align_corners=False
        )
        
        # Split into support and query sets
        num_support = int(self.num_patches_per_image * self.support_ratio)
        indices = torch.randperm(self.num_patches_per_image)
        
        support_indices = indices[:num_support]
        query_indices = indices[num_support:]
        
        return {
            'support_lr': lr_patches[support_indices],
            'support_hr': hr_patches[support_indices],
            'query_lr': lr_patches[query_indices],
            'query_hr': hr_patches[query_indices]
        }


def main():
    parser = argparse.ArgumentParser(description='Meta-learning (MAML) training for SR')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with training images')
    parser.add_argument('--epochs', type=int, default=100, help='Number of meta-training epochs')
    parser.add_argument('--tasks-per-batch', type=int, default=4, help='Number of tasks per meta-batch')
    parser.add_argument('--inner-lr', type=float, default=1e-2, help='Inner loop learning rate')
    parser.add_argument('--inner-steps', type=int, default=5, help='Inner loop gradient steps')
    parser.add_argument('--meta-lr', type=float, default=1e-4, help='Meta learning rate')
    parser.add_argument('--save-dir', type=str, default='checkpoints/meta', help='Checkpoint save directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Find all images
    data_dir = Path(args.data_dir)
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_paths.extend(data_dir.glob(ext))
    
    print(f"Found {len(image_paths)} images for meta-training")
    
    # Create task dataset
    task_dataset = TaskDataset(
        image_paths=image_paths,
        num_patches_per_image=4,
        support_ratio=0.5,
        patch_size=64,
        scale=4
    )
    
    # Create dataloader
    task_loader = DataLoader(
        task_dataset,
        batch_size=args.tasks_per_batch,
        shuffle=True,
        num_workers=4
    )
    
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
    
    # Create MAML trainer
    maml = MAMLTrainer(
        model=model,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps,
        meta_lr=args.meta_lr,
        first_order=True  # FOMAML for efficiency
    )
    
    # Training loop
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        loss = maml.train_epoch(task_loader, device=args.device)
        
        print(f"Meta-loss: {loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = save_dir / f'meta_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': maml.model.state_dict(),
                'meta_optimizer_state_dict': maml.meta_optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = save_dir / 'meta_final.pth'
    torch.save({
        'model_state_dict': maml.model.state_dict(),
        'meta_lr': args.meta_lr,
        'inner_lr': args.inner_lr,
        'inner_steps': args.inner_steps,
    }, final_path)
    print(f"\nFinal model saved: {final_path}")


if __name__ == '__main__':
    main()
