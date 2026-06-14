"""
Dataset sampling utilities for controlled data subset selection.
Provides global and per-dataset sampling control for training.
"""
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np


class DatasetSampler:
    """
    Handles dataset sampling with global ratio and per-dataset overrides.
    
    Supports:
    - Global sample_ratio applied across all datasets
    - Per-dataset sample_ratio overrides
    - Hard caps on total images (max_images_total)
    - Weighted sampling across datasets
    """
    
    def __init__(
        self,
        dataset_configs: List[Dict],
        global_sample_ratio: Optional[float] = None,
        max_images_total: Optional[int] = None,
        seed: int = 42
    ):
        """
        Args:
            dataset_configs: List of dicts with 'hr_dir', 'sample_ratio', 'weight', 'is_validation', 'enabled'
            global_sample_ratio: Global ratio (0.0-1.0) applied to all datasets unless overridden
            max_images_total: Hard cap on total images across all datasets
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = random.Random(seed)
        np.random.seed(seed)
        
        # Parse dataset configs
        self.datasets = []
        for cfg in dataset_configs:
            if not cfg.get('enabled', True):
                continue
            
            # Skip validation datasets by default
            if cfg.get('is_validation', False) and not cfg.get('include_in_sampling', False):
                continue
            
            hr_dir = Path(cfg['hr_dir'])
            if not hr_dir.exists():
                continue
            
            # Get images
            hr_images = sorted(hr_dir.glob("*.png")) + \
                       sorted(hr_dir.glob("*.jpg")) + \
                       sorted(hr_dir.glob("*.jpeg"))
            
            if len(hr_images) == 0:
                continue
            
            # Determine sample ratio
            if global_sample_ratio is not None:
                sample_ratio = global_sample_ratio
            else:
                sample_ratio = cfg.get('sample_ratio', 1.0)
            
            num_to_use = int(len(hr_images) * sample_ratio)
            
            self.datasets.append({
                'name': cfg.get('name', hr_dir.name),
                'hr_dir': hr_dir,
                'images': hr_images,
                'total_images': len(hr_images),
                'sample_ratio': sample_ratio,
                'num_to_use': num_to_use,
                'weight': cfg.get('weight', 1.0),
                'is_validation': cfg.get('is_validation', False)
            })
        
        # Calculate total
        self.total_images = sum(ds['num_to_use'] for ds in self.datasets)
        
        # Apply max_images_total cap if specified
        if max_images_total is not None:
            self._apply_global_cap(max_images_total)
    
    def _apply_global_cap(self, max_total: int):
        """Apply hard cap on total images across all datasets."""
        if self.total_images <= max_total:
            return
        
        # Proportional scaling to respect weights
        total_weight = sum(ds['weight'] for ds in self.datasets)
        
        for ds in self.datasets:
            proportion = ds['weight'] / total_weight
            ds['num_to_use'] = max(1, int(max_total * proportion))
        
        # Recalculate total
        self.total_images = sum(ds['num_to_use'] for ds in self.datasets)
    
    def get_sampled_indices(self, dataset_idx: int) -> List[int]:
        """Get sampled indices for a specific dataset."""
        if dataset_idx >= len(self.datasets):
            return []
        
        ds = self.datasets[dataset_idx]
        indices = list(range(ds['total_images']))
        self.rng.shuffle(indices)
        return indices[:ds['num_to_use']]
    
    def get_image_paths(self, dataset_idx: int) -> List[Path]:
        """Get sampled image paths for a specific dataset."""
        ds = self.datasets[dataset_idx]
        sampled_indices = self.get_sampled_indices(dataset_idx)
        return [ds['images'][i] for i in sampled_indices]
    
    def get_all_image_paths(self) -> List[Tuple[str, Path]]:
        """Get all sampled image paths with dataset name."""
        result = []
        for ds in self.datasets:
            for img_path in self.get_image_paths(self.datasets.index(ds)):
                result.append((ds['name'], img_path))
        return result
    
    def get_dataset_info(self) -> List[Dict]:
        """Get info about all datasets after sampling."""
        return [
            {
                'name': ds['name'],
                'hr_dir': str(ds['hr_dir']),
                'total_images': ds['total_images'],
                'sampled_images': ds['num_to_use'],
                'sample_ratio': ds['sample_ratio'],
                'weight': ds['weight']
            }
            for ds in self.datasets
        ]
    
    def __len__(self) -> int:
        return self.total_images
    
    def __repr__(self) -> str:
        lines = [f"DatasetSampler(total_images={self.total_images})"]
        for ds in self.datasets:
            lines.append(
                f"  - {ds['name']}: {ds['num_to_use']}/{ds['total_images']} "
                f"(ratio={ds['sample_ratio']:.2f}, weight={ds['weight']})"
            )
        return "\n".join(lines)


def create_sampler_from_config(config: Dict) -> Optional[DatasetSampler]:
    """
    Create a DatasetSampler from configuration dict.
    
    Expected config structure:
    {
        'data': {
            'datasets': [...],
            'sample_ratio': 0.5,  # global ratio
            'max_images_total': 5000  # optional hard cap
        }
    }
    """
    data_cfg = config.get('data', {})
    dataset_configs = data_cfg.get('datasets', [])
    
    if not dataset_configs:
        return None
    
    global_sample_ratio = data_cfg.get('sample_ratio')
    max_images_total = data_cfg.get('max_images_total')
    
    return DatasetSampler(
        dataset_configs=dataset_configs,
        global_sample_ratio=global_sample_ratio,
        max_images_total=max_images_total,
        seed=config.get('system', {}).get('seed', 42)
    )


if __name__ == "__main__":
    # Test with default configs
    configs = [
        {
            'name': 'anime_video_frames',
            'hr_dir': 'data/anime_video_frames',
            'sample_ratio': 0.3,
            'weight': 1.0,
            'is_validation': False,
            'enabled': True
        },
        {
            'name': 'anime_hr_1',
            'hr_dir': 'data/anime_hr_1',
            'sample_ratio': 0.5,
            'weight': 1.0,
            'is_validation': False,
            'enabled': True
        }
    ]
    
    sampler = DatasetSampler(configs, max_images_total=3000)
    print(sampler)
    print(f"\nTotal sampled: {len(sampler)}")
    print("\nDataset info:")
    for info in sampler.get_dataset_info():
        print(f"  {info['name']}: {info['sampled_images']} images")