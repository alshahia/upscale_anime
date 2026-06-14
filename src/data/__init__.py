try:
    from src.data.base import BaseDataset, DatasetFactory, MultiDataset
    from src.data.dataloader import DataLoaderFactory, build_dataloader
    from src.data.data_utils import (
        validate_dataset,
        create_dummy_dataset,
        check_dataset_balance,
        auto_crop_images,
        get_dataset_info,
    )
    from src.data.video_dataset import VideoDataset, TemporalDataset
    from src.data.video_extraction import (
        extract_frames_from_video,
        extract_from_folder,
        get_video_info,
        preview_extraction,
        calculate_frame_quality,
    )
    from src.data.quality_analyzer import (
        QualityAnalyzer,
        calculate_image_quality,
        analyze_dataset_quality,
        compute_optimal_crop_size,
        get_degradation_preset_config,
    )
    from src.data.augmentation import (
        MixupAugmentation,
        CutMixAugmentation,
        RandomResizedCrop,
        AugmentationPipeline,
        apply_geometric_augmentation,
        apply_color_jitter,
    )
    from src.data.preprocessing_manager import (
        PreprocessingManager,
        OnTheFlyProcessor,
    )
    from src.data.precomputed_dataset import PrecomputedDataset
    from src.data.dataset_sampler import DatasetSampler, create_sampler_from_config
    from src.data.line_enhancement import LineEnhancer
    from src.data.storage_estimator import (
        estimate_dataset_storage,
        estimate_multiple_datasets,
        check_against_available_space,
        print_storage_estimate,
        create_default_dataset_configs,
    )
except ImportError:
    try:
        from data.base import BaseDataset, DatasetFactory, MultiDataset
        from data.dataloader import DataLoaderFactory, build_dataloader
        from data.data_utils import (
            validate_dataset,
            create_dummy_dataset,
            check_dataset_balance,
            auto_crop_images,
            get_dataset_info,
        )
        from data.video_dataset import VideoDataset, TemporalDataset
        from data.video_extraction import (
            extract_frames_from_video,
            extract_from_folder,
            get_video_info,
            preview_extraction,
            calculate_frame_quality,
        )
        from data.quality_analyzer import (
            QualityAnalyzer,
            calculate_image_quality,
            analyze_dataset_quality,
            compute_optimal_crop_size,
            get_degradation_preset_config,
        )
        from data.augmentation import (
            MixupAugmentation,
            CutMixAugmentation,
            RandomResizedCrop,
            AugmentationPipeline,
            apply_geometric_augmentation,
            apply_color_jitter,
        )
        from data.preprocessing_manager import (
            PreprocessingManager,
            OnTheFlyProcessor,
        )
        from data.precomputed_dataset import PrecomputedDataset
        from data.dataset_sampler import DatasetSampler, create_sampler_from_config
        from data.line_enhancement import LineEnhancer
        from data.storage_estimator import (
            estimate_dataset_storage,
            estimate_multiple_datasets,
            check_against_available_space,
            print_storage_estimate,
            create_default_dataset_configs,
        )
    except ImportError:
        from base import BaseDataset, DatasetFactory, MultiDataset
        from dataloader import DataLoaderFactory, build_dataloader
        from data_utils import (
            validate_dataset,
            create_dummy_dataset,
            check_dataset_balance,
            auto_crop_images,
            get_dataset_info,
        )
        from video_dataset import VideoDataset, TemporalDataset
        from video_extraction import (
            extract_frames_from_video,
            extract_from_folder,
            get_video_info,
            preview_extraction,
            calculate_frame_quality,
        )
        from quality_analyzer import (
            QualityAnalyzer,
            calculate_image_quality,
            analyze_dataset_quality,
            compute_optimal_crop_size,
            get_degradation_preset_config,
        )
        from augmentation import (
            MixupAugmentation,
            CutMixAugmentation,
            RandomResizedCrop,
            AugmentationPipeline,
            apply_geometric_augmentation,
            apply_color_jitter,
        )
        from preprocessing_manager import (
            PreprocessingManager,
            OnTheFlyProcessor,
        )
        from precomputed_dataset import PrecomputedDataset
        from dataset_sampler import DatasetSampler, create_sampler_from_config
        from line_enhancement import LineEnhancer
        from storage_estimator import (
            estimate_dataset_storage,
            estimate_multiple_datasets,
            check_against_available_space,
            print_storage_estimate,
            create_default_dataset_configs,
        )

__all__ = [
    'BaseDataset',
    'DatasetFactory',
    'MultiDataset',
    'DataLoaderFactory',
    'build_dataloader',
    'validate_dataset',
    'create_dummy_dataset',
    'check_dataset_balance',
    'auto_crop_images',
    'get_dataset_info',
    'VideoDataset',
    'TemporalDataset',
    'extract_frames_from_video',
    'extract_from_folder',
    'get_video_info',
    'preview_extraction',
    'calculate_frame_quality',
    'QualityAnalyzer',
    'calculate_image_quality',
    'analyze_dataset_quality',
    'compute_optimal_crop_size',
    'get_degradation_preset_config',
    # Augmentation
    'MixupAugmentation',
    'CutMixAugmentation',
    'RandomResizedCrop',
    'AugmentationPipeline',
    'apply_geometric_augmentation',
    'apply_color_jitter',
    # Preprocessing pipeline (Phase 1 / Issue #5)
    'PreprocessingManager',
    'OnTheFlyProcessor',
    'PrecomputedDataset',
    'DatasetSampler',
    'create_sampler_from_config',
    'LineEnhancer',
    'estimate_dataset_storage',
    'estimate_multiple_datasets',
    'check_against_available_space',
    'print_storage_estimate',
    'create_default_dataset_configs',
]
