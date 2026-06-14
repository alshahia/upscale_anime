"""Centralized utility imports with proper dependency management."""
import warnings
import logging
import os
from typing import Optional, List, Dict, Any

# Only warn once in main process (suppress in dataloader workers)
_IMPORT_WARNED = False

def _warn_once(msg: str):
    global _IMPORT_WARNED
    if not _IMPORT_WARNED and os.getpid() == os.getppid() or not _IMPORT_WARNED:
        # Only warn if this is the main process or first import
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
        # More specific error handling for fallback imports
        missing_module = str(e).split("'")[1].split()[-1] if "'" in str(e) else str(e).split()[-1]
        warnings.warn(f"Missing optional dependency: {missing_module}. Some features may be limited.")
        # Provide minimal fallbacks
        Config = None
        calculate_psnr = None
        calculate_ssim = None
        calculate_batch_metrics = None
        MetricsTracker = None
        
        # Create minimal stubs for missing dependencies
        def stub_function(*args, **kwargs):
            warnings.warn(f"Function {missing_module} not available. Install with: pip install {missing_module}")
            return None
        
        tensor_to_image = stub_function
        save_image_comparison = stub_function
        plot_training_history = stub_function
        create_grid_visualization = stub_function
        visualize_degradation = stub_function
        save_batch_comparison = stub_function
        TrainingVisualizer = stub_function
        print_welcome_message = stub_function
        create_visualizer_for_training = stub_function
        TrainingHistory = stub_function
        HistoryCallback = stub_function
        create_history_tracker = stub_function
        load_history = stub_function
        GPUMonitor = stub_function
        SystemMonitor = stub_function
        ResourceMonitor = stub_function
        monitor_execution = stub_function
        get_optimal_worker_count = stub_function
        print_environment_info = stub_function
        check_system_ready = stub_function
        download_pretrained_model = stub_function
        download_all_teacher_models = stub_function
        get_teacher_model_path = stub_function
        check_model_exists = stub_function
        verify_model_weights = stub_function
        list_available_models = stub_function
        get_pretrained_dir = stub_function
        load_pretrained_weights = stub_function
        get_checkpoint_info = stub_function
        convert_spanf_checkpoint = stub_function

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
