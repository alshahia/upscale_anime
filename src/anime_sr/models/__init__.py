try:
    from anime_sr.models.base import BaseSRModel

    # Teacher models for knowledge distillation
    from anime_sr.models.teachers import (
        EDSR, RCAN, SwinIR,
        load_teacher_model,
        create_teacher_model,
        auto_detect_architecture,
    )
    __all__ = [
        'BaseSRModel',
        'EDSR', 'RCAN', 'SwinIR',
        'load_teacher_model',
        'create_teacher_model',
        'auto_detect_architecture',
    ]
except ImportError:
    from anime_sr.models.base import BaseSRModel

    # Teacher models for knowledge distillation
    try:  # noqa: F401  (re-exports; pyflakes misses the nested __all__ below)
        from anime_sr.models.teachers import (
            EDSR,
            RCAN,
            SwinIR,
            load_teacher_model,
            create_teacher_model,
            auto_detect_architecture,
        )
        __all__ = [
            'BaseSRModel',
            'EDSR', 'RCAN', 'SwinIR',
            'load_teacher_model',
            'create_teacher_model',
            'auto_detect_architecture',
        ]
    except ImportError:
        # Teacher models not available
        __all__ = ['BaseSRModel']
