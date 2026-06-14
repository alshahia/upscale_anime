"""
Precomputed dataset for fast loading of pre-generated LR/HR pairs.
Eliminates on-the-fly degradation overhead during training.
"""
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from typing import Optional, Dict


class PrecomputedDataset(Dataset):
    """
    Dataset that loads pre-computed LR/HR pairs from disk.
    Much faster than BaseDataset since no degradation is applied on-the-fly.
    """

    def __init__(
        self,
        precomputed_dir: str,
        augment: bool = True,
        scale: int = 4,
    ):
        self.precomputed_dir = Path(precomputed_dir)
        self.augment = augment
        self.scale = scale

        self.lr_dir = self.precomputed_dir / 'lr'
        self.hr_dir = self.precomputed_dir / 'hr'

        if not self.lr_dir.exists() or not self.hr_dir.exists():
            raise ValueError(f"Precomputed directory not found: {precomputed_dir}")

        self.lr_files = sorted([f for f in self.lr_dir.glob('*.png')])
        self.hr_files = sorted([f for f in self.hr_dir.glob('*.png')])

        if len(self.lr_files) != len(self.hr_files):
            raise ValueError(f"Mismatch: {len(self.lr_files)} LR files, {len(self.hr_files)} HR files")

        if len(self.lr_files) == 0:
            raise ValueError(f"No precomputed pairs found in {precomputed_dir}")

        self._load_meta()

    def _load_meta(self):
        """Load metadata if available."""
        meta_path = self.precomputed_dir / 'meta.yaml'
        if meta_path.exists():
            try:
                import yaml
                with open(meta_path, 'r') as f:
                    self.meta = yaml.safe_load(f)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Failed to load meta.yaml at {meta_path}: {e}"
                )
                self.meta = {}
        else:
            self.meta = {}

    def __len__(self) -> int:
        return len(self.lr_files)

    def _augment(self, lr: np.ndarray, hr: np.ndarray) -> tuple:
        """Apply random augmentations."""
        if not self.augment:
            return lr, hr

        flip_h = random.random() > 0.5
        flip_v = random.random() > 0.5
        rotate = random.randint(0, 3)

        if flip_h:
            lr = np.flip(lr, axis=1).copy()
            hr = np.flip(hr, axis=1).copy()
        if flip_v:
            lr = np.flip(lr, axis=0).copy()
            hr = np.flip(hr, axis=0).copy()

        lr = np.rot90(lr, rotate, axes=(0, 1)).copy()
        hr = np.rot90(hr, rotate, axes=(0, 1)).copy()

        return lr, hr

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        lr_path = self.lr_files[idx]
        hr_path = self.hr_files[idx]

        lr = np.array(Image.open(lr_path)).astype(np.float32) / 255.0
        hr = np.array(Image.open(hr_path)).astype(np.float32) / 255.0

        lr, hr = self._augment(lr, hr)

        lr_tensor = torch.from_numpy(lr).permute(2, 0, 1).contiguous()
        hr_tensor = torch.from_numpy(hr).permute(2, 0, 1).contiguous()

        return {
            'lr': lr_tensor,
            'hr': hr_tensor,
            'lr_path': str(lr_path),
            'hr_path': str(hr_path),
        }
