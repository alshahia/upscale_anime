"""Centralized utility imports with proper dependency management."""
import warnings
import logging
import os
from typing import Optional, List, Dict, Any

# Only warn once in main process (suppress in dataloader workers)
_IMPORT_WARNED = False

def _warn_once(msg: str):
    global _IMPORT_WARNED
    if not _IMPORT_WARNED:
        _IMPORT_WARNED = True
        warnings.warn(msg, stacklevel=2)

# Try to import from src package first
try:
    from src.utils.config import Config
    from src.utils.metrics import calculate_psnr, calculate_ssim, calculate_batch_metrics, MetricsTracker
    from src.utils.visualizer import (
        tensor_to_image,
        save_image_comparison,
        plot_training_history,
        create_grid_visualization,
        visualize_degradation,
        save_batch_comparison,
    )
    from src.utils.training_visualizer import (
        TrainingVisualizer,
        print_welcome_message,
        create_visualizer_for_training,
    )
    from src.utils.training_history import (
        TrainingHistory,
        HistoryCallback,
        create_history_tracker,
        load_history,
    )
    from src.utils.system import (
        GPUMonitor,
        SystemMonitor,
        ResourceMonitor,
        monitor_execution,
        get_optimal_worker_count,
        print_environment_info,
        check_system_ready,
    )
    from src.utils.pretrained_models import (
        download_pretrained_model,
        download_all_teacher_models,
        get_teacher_model_path,
        check_model_exists,
        verify_model_weights,
        list_available_models,
        get_pretrained_dir,
    )
    from src.utils.checkpoint_loader import (
        load_pretrained_weights,
        get_checkpoint_info,
        convert_spanf_checkpoint,
    )
    IMPORTS_AVAILABLE = True
    
except ImportError as e:
    # Fallback to relative imports with detailed error reporting
    _warn_once(f"Failed to import from src package: {e}. Falling back to relative imports.")
    IMPORTS_AVAILABLE = False
    
    try:
        from utils.config import Config
        from utils.metrics import calculate_psnr, calculate_ssim, calculate_batch_metrics, MetricsTracker
        from utils.visualizer import (
            tensor_to_image,
            save_image_comparison,
            plot_training_history,
            create_grid_visualization,
            visualize_degradation,
            save_batch_comparison,
        )
        from utils.training_visualizer import (
            TrainingVisualizer,
            print_welcome_message,
            create_visualizer_for_training,
        )
        from utils.training_history import (
            TrainingHistory,
            HistoryCallback,
            create_history_tracker,
            load_history,
        )
        from utils.system import (
            GPUMonitor,
            SystemMonitor,
            ResourceMonitor,
            monitor_execution,
            get_optimal_worker_count,
            print_environment_info,
            check_system_ready,
        )
        from utils.pretrained_models import (
            download_pretrained_model,
            download_all_teacher_models,
            get_teacher_model_path,
            check_model_exists,
            verify_model_weights,
            list_available_models,
            get_pretrained_dir,
        )
        from utils.checkpoint_loader import (
            load_pretrained_weights,
            get_checkpoint_info,
            convert_spanf_checkpoint,
        )
    except ImportError as fallback_error:
        # Both absolute and relative imports failed. Raise a clear error
        # instead of silently degrading to stub functions that return None.
        missing_module = str(e).split("'")[1].split()[-1] if "'" in str(e) else str(e).split()[-1]
        raise ImportError(
            f"Failed to import required utilities from src.utils. "
            f"Missing dependency: {missing_module!r}. "
            f"Original import error: {e}. "
            f"Fallback import error: {fallback_error}. "
            f"Ensure the package is installed and the working directory is the project root."
        ) from fallback_error

__all__ = [
    'Config',
    'calculate_psnr',
    'calculate_ssim',
    'calculate_batch_metrics',
    'MetricsTracker',
    'tensor_to_image',
    'save_image_comparison',
    'plot_training_history',
    'create_grid_visualization',
    'visualize_degradation',
    'save_batch_comparison',
    'TrainingVisualizer',
    'print_welcome_message',
    'create_visualizer_for_training',
    'TrainingHistory',
    'HistoryCallback',
    'create_history_tracker',
    'load_history',
    'GPUMonitor',
    'SystemMonitor',
    'ResourceMonitor',
    'monitor_execution',
    'get_optimal_worker_count',
    'print_environment_info',
    'check_system_ready',
    'download_pretrained_model',
    'download_all_teacher_models',
    'get_teacher_model_path',
    'check_model_exists',
    'verify_model_weights',
    'list_available_models',
    'get_pretrained_dir',
    # Checkpoint loader
    'load_pretrained_weights',
    'get_checkpoint_info',
    'convert_spanf_checkpoint',
]
