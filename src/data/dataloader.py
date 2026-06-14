"""
Universal DataLoader factory
Supports single and multi-dataset loading with configurable workers, batch size, etc.
Also supports video datasets with on-demand or pre-extraction modes.
"""
import copy
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import Dict, Optional, List
import psutil
from pathlib import Path

try:
    from data.base import BaseDataset, DatasetFactory, MultiDataset
    from data.video_dataset import VideoDataset, TemporalDataset
    from data.video_extraction import extract_from_folder
    from data.precomputed_dataset import PrecomputedDataset
except ImportError:
    from base import BaseDataset, DatasetFactory, MultiDataset
    from video_dataset import VideoDataset, TemporalDataset
    from video_extraction import extract_from_folder
    from precomputed_dataset import PrecomputedDataset


def _resolve_batch_size(train_cfg: dict, data_cfg: dict, default: int = 8) -> int:
    """
    Resolve batch_size from config with proper precedence and fallback notification.
    
    Priority:
    1. training.batch_size (main training config)
    2. data.batch_size (data-level config)
    3. Default value (with warning)
    
    Args:
        train_cfg: Training configuration dict
        data_cfg: Data configuration dict
        default: Fallback default value
        
    Returns:
        Resolved batch_size
    """
    # Check training.batch_size first
    training_batch = train_cfg.get('batch_size')
    if training_batch is not None:
        return training_batch
    
    # Check data.batch_size
    data_batch = data_cfg.get('batch_size')
    if data_batch is not None:
        print(f"[Config] Using data.batch_size={data_batch} (training.batch_size not set)")
        return data_batch
    
    # Use default with warning
    print(f"[WARNING] batch_size not found in config, using default={default}")
    print(f"  To fix: Add 'training.batch_size' or 'data.batch_size' to your config file")
    return default


def get_optimal_num_workers(data_cfg: dict, batch_size: int) -> int:
    """
    Auto-detect optimal num_workers based on system resources.
    
    Rules:
    - Use 4 workers per GPU as baseline
    - Cap at CPU cores - 2 (leave headroom)
    - Reduce if system RAM < 16GB
    - Account for gradient accumulation (need more workers)
    
    Args:
        data_cfg: Data configuration dict
        batch_size: Training batch size
        
    Returns:
        Optimal number of workers
    """
    import os
    import psutil
    
    # User override takes precedence (unless explicitly set to null/None)
    user_workers = data_cfg.get('num_workers')
    auto_tune = data_cfg.get('auto_tune_workers', True)
    
    if user_workers is not None and not auto_tune:
        return max(0, min(user_workers, os.cpu_count() or 4))
    
    if not auto_tune and user_workers is not None:
        return max(0, min(user_workers, os.cpu_count() or 4))
    
    # Detect resources
    cpu_cores = os.cpu_count() or 4
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    # Get gradient accumulation from training config (affects throughput needs)
    # Note: This is accessed via parent config, default to 1 if not available
    accumulation_steps = data_cfg.get('gradient_accumulation_steps', 1)
    
    # Calculate optimal workers
    workers_per_gpu = 6
    base_workers = workers_per_gpu * num_gpus
    
    # Scale up for gradient accumulation (need to keep GPU fed)
    if accumulation_steps > 1:
        base_workers = max(base_workers, int(base_workers * min(accumulation_steps, 4) / 2))
    
    # Cap by CPU cores (leave 2 cores for main process and system)
    max_workers = max(1, cpu_cores - 2)
    num_workers = min(base_workers, max_workers)
    
    # Reduce for limited RAM (each worker holds a copy of data)
    if ram_gb < 8:
        num_workers = min(num_workers, 2)
    elif ram_gb < 16:
        num_workers = min(num_workers, 4)
    
    # Hard cap at 16 workers (diminishing returns beyond this)
    num_workers = min(num_workers, 16)
    
    print(f"[DataLoader] Auto-tuned num_workers: {num_workers} "
          f"(GPUs: {num_gpus}, CPUs: {cpu_cores}, RAM: {ram_gb:.1f}GB, "
          f"Accum: {accumulation_steps}x)")
    
    return num_workers


class DataLoaderFactory:
    """
    Factory for creating DataLoaders from config.
    Handles dataset creation, multi-dataset weighting, and DataLoader setup.
    """
    
    @staticmethod
    def _handle_video_extraction(data_cfg: Dict, is_train: bool) -> Optional[str]:
        """
        Handle video extraction based on config.
        
        Returns:
            Path to extracted frames directory, or None if no video processing
        """
        video_cfg = data_cfg.get('video', {})
        
        if not video_cfg.get('enabled', False):
            return None
        
        # Determine video directory
        video_dir = video_cfg.get('video_dir', 'data/anime_vid')
        video_path = Path(video_dir)
        
        # Auto-detect if default path doesn't exist but alternate does
        if not video_path.exists() and video_dir == 'data/anime_vid':
            # Try to find videos in data folder
            data_root = Path(data_cfg.get('data_root', 'data'))
            for potential in ['anime_vid', 'videos', 'video']:
                alt_path = data_root / potential
                if alt_path.exists():
                    video_path = alt_path
                    break
        
        if not video_path.exists():
            print(f"Warning: Video directory not found: {video_path}")
            return None
        
        # Check for video files
        video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.webm')
        has_videos = any(
            list(video_path.glob(f'*{ext}')) or list(video_path.glob(f'*{ext.upper()}'))
            for ext in video_extensions
        )
        
        if not has_videos:
            print(f"Warning: No video files found in {video_path}")
            return None
        
        # Determine extraction mode
        extract_mode = video_cfg.get('extract_mode', 'pre')
        
        if extract_mode == 'pre':
            # Pre-extraction: extract all frames before training
            output_dir = Path(video_cfg.get('output_dir', 'data/anime_video_frames'))
            
            # Check if already extracted
            if output_dir.exists() and any(output_dir.glob('*.png')):
                print(f"Using pre-extracted frames from: {output_dir}")
                return str(output_dir)
            
            print(f"Pre-extracting frames from videos in: {video_path}")
            print(f"Output directory: {output_dir}")
            
            result = extract_from_folder(
                video_folder=video_path,
                output_dir=output_dir,
                extract_every_n_frames=video_cfg.get('extract_every_n_frames', 30),
                quality_threshold=video_cfg.get('quality_threshold', 0.7),
                remove_duplicates=video_cfg.get('remove_duplicates', True),
                min_resolution=(video_cfg.get('min_resolution', 720),) * 2
            )
            
            if result['success'] and result['total_extracted'] > 0:
                print(f"Extracted {result['total_extracted']} frames successfully")
                return str(output_dir)
            else:
                print(f"Warning: Frame extraction failed or produced no frames")
                return None
        
        elif extract_mode == 'on_demand':
            # Return video_dir to indicate on-demand mode
            return str(video_path)
        
        return None
    
    @staticmethod
    def create(config: Dict, is_train: bool = True) -> DataLoader:
        """
        Create DataLoader from config.
        
        Args:
            config: Configuration dict with 'data' and 'training' sections
            is_train: Whether this is for training (affects shuffle, drop_last)
        
        Returns:
            Configured DataLoader
        """
        data_cfg = config.get('data', {})
        train_cfg = config.get('training', {})
        
        # Get sampling mode for weight calculation
        sampling_mode = data_cfg.get('sampling_mode', 'fixed')  # 'fixed' or 'size_proportional'
        if sampling_mode == 'size_proportional':
            print(f"[DataLoader] Using size-proportional sampling mode")
        
        # Handle video extraction if enabled
        video_result = DataLoaderFactory._handle_video_extraction(data_cfg, is_train)
        video_cfg = data_cfg.get('video', {})
        
        # Get dataset configurations
        dataset_configs = data_cfg.get('datasets', [])
        
        # Filter enabled datasets
        enabled_configs = [cfg for cfg in dataset_configs if cfg.get('enabled', True)]
        
        # For validation mode, filter to only validation datasets if any are marked
        if not is_train:
            has_val_datasets = any(cfg.get('is_validation', False) for cfg in enabled_configs)
            if has_val_datasets:
                enabled_configs = [cfg for cfg in enabled_configs if cfg.get('is_validation', False)]
                print(f"[DataLoader] Validation mode: using {len(enabled_configs)} validation dataset(s)")
        
        # Add video dataset if video processing returned a path
        if video_result:
            extract_mode = video_cfg.get('extract_mode', 'pre')
            
            if extract_mode == 'pre':
                # Add as regular image dataset
                video_dataset_cfg = {
                    'name': 'anime_video',
                    'weight': video_cfg.get('weight', 0.5),
                    'enabled': True,
                    'hr_dir': video_result,
                    'scale': data_cfg.get('scale', 4),
                    'crop_size': data_cfg.get('crop_size', 128),
                    'augment': data_cfg.get('augment', True) and is_train,
                    'degradation': data_cfg.get('degradation') if is_train else None,
                }
                enabled_configs.append(video_dataset_cfg)
            
            elif extract_mode == 'on_demand':
                # Will create VideoDataset separately
                pass
        
        if len(enabled_configs) == 0 and not (video_result and video_cfg.get('extract_mode') == 'on_demand'):
            raise ValueError("No enabled datasets found in config")

        # Check for precomputed dataset (fastest option - no on-the-fly degradation)
        precomputed_dir = data_cfg.get('precomputed_dir')
        if precomputed_dir and Path(precomputed_dir).exists():
            print(f"[DataLoader] Using precomputed dataset from: {precomputed_dir}")
            dataset = PrecomputedDataset(
                precomputed_dir=precomputed_dir,
                augment=data_cfg.get('augment', True) and is_train,
                scale=data_cfg.get('scale', 4),
            )
        else:
            if precomputed_dir:
                print(f"[DataLoader] Precomputed dir not found: {precomputed_dir}, falling back to standard dataset")

            # Create dataset(s)
            datasets = []

            # Create standard datasets
            # Get preprocessing config for PreprocessingManager (NEW)
            preprocessing_cfg = data_cfg.get('preprocessing')
            
            if len(enabled_configs) == 1:
                datasets.append(DatasetFactory.create(enabled_configs[0], preprocessing_config=preprocessing_cfg))
            elif len(enabled_configs) > 1:
                weights = [cfg.get('weight', 1.0) for cfg in enabled_configs]

                # Merge dataset configs with global data settings
                full_configs = []
                for cfg in enabled_configs:
                    full_cfg = {
                        'scale': data_cfg.get('scale', 4),
                        'crop_size': data_cfg.get('crop_size', 128),
                        'augment': data_cfg.get('augment', True) and is_train,
                        'degradation': data_cfg.get('degradation') if is_train else None,
                        **cfg,
                    }
                    full_configs.append(full_cfg)

                datasets.append(DatasetFactory.create_multi_dataset(
                    full_configs, weights, weight_mode=sampling_mode,
                    preprocessing_config=preprocessing_cfg
                ))

            # Create VideoDataset for on-demand mode
            if video_result and video_cfg.get('extract_mode') == 'on_demand':
                degradation = data_cfg.get('degradation') or {}
                use_quality_analyzer = (
                    degradation.get('mode') in ['auto', 'smart'] or
                    data_cfg.get('use_quality_analyzer', False)
                )

                video_dataset = VideoDataset(
                    video_dir=video_result,
                    scale=data_cfg.get('scale', 4),
                    crop_size=data_cfg.get('crop_size', 128),
                    augment=data_cfg.get('augment', True) and is_train,
                    degradation=degradation if is_train else None,
                    extract_every_n_frames=video_cfg.get('extract_every_n_frames', 30),
                    quality_threshold=video_cfg.get('quality_threshold', 0.7),
                    remove_duplicates=video_cfg.get('remove_duplicates', True),
                    min_resolution=(video_cfg.get('min_resolution', 720),) * 2,
                    cache_size_gb=video_cfg.get('cache_size_gb', 10.0),
                    crop_size_mode=data_cfg.get('crop_size_mode', 'manual'),
                    auto_crop_max=data_cfg.get('auto_crop_max', 960),
                    auto_crop_min=data_cfg.get('auto_crop_min', 64),
                    use_quality_analyzer=use_quality_analyzer,
                )
                datasets.append(video_dataset)

            # Combine datasets if multiple
            if len(datasets) == 1:
                dataset = datasets[0]
            elif len(datasets) > 1:
                weights = [1.0] * len(datasets)
                dataset = MultiDataset(datasets, weights, weight_mode=sampling_mode)
            else:
                raise ValueError("No datasets could be created")
        
        # Wrap with TemporalDataset if temporal training is enabled
        if video_cfg.get('temporal_training', False):
            frame_window = video_cfg.get('frame_window', 3)
            scene_threshold = video_cfg.get('scene_change_threshold', 0.3)
            print(f"[DataLoader] Enabling temporal training with frame_window={frame_window}")
            dataset = TemporalDataset(
                base_dataset=dataset,
                frame_window=frame_window,
                scene_change_threshold=scene_threshold,
            )
        
        # DataLoader parameters - resolve batch_size with fallback notification
        batch_size = _resolve_batch_size(train_cfg, data_cfg, default=8)
        
        # Use auto-tuning for num_workers (or user override)
        num_workers = get_optimal_num_workers(data_cfg, batch_size)
        
        pin_memory = data_cfg.get('pin_memory', True)
        prefetch_factor = data_cfg.get('prefetch_factor', 4)
        persistent_workers = data_cfg.get('persistent_workers', True) and num_workers > 0
        
        # For small datasets, don't drop last batch to ensure we have data
        dataset_len = len(dataset) if hasattr(dataset, '__len__') else 0
        should_drop_last = is_train and dataset_len > batch_size * 2
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=should_drop_last,  # Only drop if we have enough data
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
        )
        
        return dataloader
    
    @staticmethod
    def create_train_val_loaders(config: Dict) -> tuple:
        """
        Create both training and validation DataLoaders.
        
        Returns:
            (train_loader, val_loader)
        """
        train_loader = DataLoaderFactory.create(config, is_train=True)
        
        # For validation, disable augmentation and use smaller batch
        val_config = copy.deepcopy(config)
        val_config['data']['augment'] = False
        val_config['training']['batch_size'] = config.get('training', {}).get('batch_size', 8)
        
        val_loader = DataLoaderFactory.create(val_config, is_train=False)
        
        return train_loader, val_loader
    
    @staticmethod
    def create_separate(config: Dict) -> tuple:
        """
        Create separate training and validation DataLoaders.
        
        Training datasets: is_validation=False (or not set)
        Validation datasets: is_validation=True
        
        Returns:
            (train_loader, val_loader)
        """
        data_cfg = config.get('data', {})
        train_cfg = config.get('training', {})
        
        # Get all dataset configs
        dataset_configs = data_cfg.get('datasets', [])
        
        # Separate by is_validation flag
        train_configs = [cfg for cfg in dataset_configs 
                        if cfg.get('enabled', True) and not cfg.get('is_validation', False)]
        val_configs = [cfg for cfg in dataset_configs 
                      if cfg.get('enabled', True) and cfg.get('is_validation', False)]
        
        print(f"[DataLoader] Separate loaders: {len(train_configs)} train, {len(val_configs)} val datasets")
        
        # Build train loader
        if not train_configs:
            raise ValueError("No training datasets found")
        
        train_dataset = DataLoaderFactory._build_dataset(train_configs, data_cfg, 
                                                         is_train=True, use_weights=True)
        train_loader = DataLoaderFactory._build_dataloader(train_dataset, train_cfg, data_cfg, is_train=True)
        
        # Build val loader
        if val_configs:
            val_dataset = DataLoaderFactory._build_dataset(val_configs, data_cfg,
                                                          is_train=False, use_weights=False)
            val_loader = DataLoaderFactory._build_dataloader(val_dataset, train_cfg, data_cfg, is_train=False)
        else:
            print("[DataLoader] Warning: No validation datasets found")
            val_loader = None
        
        return train_loader, val_loader
    
    @staticmethod
    def _build_dataset(configs: list, data_cfg: dict, is_train: bool, use_weights: bool = True) -> Dataset:
        """Build a dataset from configs."""
        if len(configs) == 1:
            return DatasetFactory.create(configs[0])
        
        # Multiple datasets - create MultiDataset
        if use_weights:
            # Use size-proportional or user weights
            sampling_mode = data_cfg.get('sampling_mode', 'fixed')
            weights = [cfg.get('weight', 1.0) for cfg in configs]
        else:
            # Validation: equal weights, sequential access
            sampling_mode = 'fixed'
            weights = [1.0] * len(configs)
        
        # Merge with global settings
        full_configs = []
        for cfg in configs:
            full_cfg = {
                'scale': data_cfg.get('scale', 4),
                'crop_size': data_cfg.get('crop_size', 128),
                'augment': data_cfg.get('augment', True) and is_train,
                'degradation': data_cfg.get('degradation') if is_train else None,
                **cfg,
            }
            full_configs.append(full_cfg)
        
        return DatasetFactory.create_multi_dataset(full_configs, weights, weight_mode=sampling_mode)
    
    @staticmethod
    def _build_dataloader(dataset: Dataset, train_cfg: dict, data_cfg: dict, is_train: bool) -> DataLoader:
        """Build a DataLoader from dataset."""
        batch_size = _resolve_batch_size(train_cfg, data_cfg, default=8)
        num_workers = get_optimal_num_workers({'num_workers': 4}, batch_size)
        pin_memory = True
        prefetch_factor = 4
        persistent_workers = num_workers > 0
        
        dataset_len = len(dataset) if hasattr(dataset, '__len__') else 0
        should_drop_last = is_train and dataset_len > batch_size * 2
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,  # Shuffle only for training
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=should_drop_last,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers,
        )


def build_dataloader(config: Dict, mode: str = 'train') -> DataLoader:
    """
    Convenience function to build dataloader.
    
    Args:
        config: Full config dict
        mode: 'train', 'val', or 'test'
    
    Returns:
        DataLoader
    """
    is_train = (mode == 'train')
    return DataLoaderFactory.create(config, is_train=is_train)


def build_separate_loaders(config: Dict) -> tuple:
    """
    Build separate training and validation DataLoaders.
    
    This is the recommended approach when you have training and validation
    datasets with different sizes. It ensures:
    - Training data is sampled with replacement/weighting
    - Validation data is processed completely each epoch (no oversampling)
    - Proper epoch boundaries for both sets
    
    Args:
        config: Full config dict with 'data' and 'training' sections
        
    Returns:
        (train_loader, val_loader)
    """
    return DataLoaderFactory.create_separate(config)
