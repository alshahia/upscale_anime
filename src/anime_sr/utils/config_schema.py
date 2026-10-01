"""
Default configuration schema (WP-5).

Defines the canonical shape of a config: required fields, types, defaults,
and valid ranges. Consumed by anime_sr.utils.config.validate_config().

Schema format — dotted key -> field spec:
    {
        "model.scale": {"type": int, "required": True, "range": (2, 4)},
        "training.optimizer": {"type": str, "choices": ["adam", "adamw", "sgd"]},
        "logging.level": {"type": str, "default": "INFO"},
    }

Field spec keys:
    - type:      expected Python type (or tuple of types). bool is never
                 accepted where int is expected.
    - required:  True  -> missing/None value is a validation error.
    - default:   documented default (informational; not applied on validate).
    - range:     (min, max) inclusive numeric bounds.
    - choices:   iterable of allowed values.

Keep this schema in sync with configs/default.yaml — it is the machine
readable form of that file's documented defaults.
"""
from typing import Any, Dict

# Types that may appear where "str or null" is acceptable
_STR_OR_NONE = (str, type(None))
_NUM_OR_NONE = (int, float, type(None))

DEFAULT_SCHEMA: Dict[str, Dict[str, Any]] = {
    # ---- model ----
    "model.name": {"type": str, "required": True},
    "model.type": {
        "type": str,
        "required": True,
        "choices": ["span", "mamba_pan", "tiny_student", "edsr", "rcan", "swinir"],
    },
    "model.scale": {"type": int, "required": True, "range": (2, 4)},
    "model.num_features": {"type": int, "default": 32, "range": (1, 512)},
    "model.num_blocks": {"type": int, "default": 12, "range": (1, 64)},
    "model.in_channels": {"type": int, "default": 3, "choices": [1, 3]},
    "model.out_channels": {"type": int, "default": 3, "choices": [1, 3]},
    "model.path": {"type": str, "default": "checkpoints/model_a_best.pth"},
    "model.pretrained_path": {"type": _STR_OR_NONE, "default": None},
    "model.strict_load": {"type": bool, "default": True},
    "model.use_lora": {"type": bool, "default": False},
    "model.lora_rank": {"type": int, "default": 4, "range": (1, 64)},
    "model.lora_alpha": {"type": float, "default": 1.0, "range": (0.0, 100.0)},

    # ---- training ----
    "training.mode": {
        "type": str,
        "choices": [
            "model_a", "model_b", "both_parallel", "both_sequential",
            "ensemble", "full", "finetune", "auto_stage",
        ],
    },
    "training.device": {"type": str, "choices": ["cuda", "cpu", "auto"]},
    "training.seed": {"type": int, "default": 42},
    "training.optimizer": {"type": str, "choices": ["adam", "adamw", "sgd"]},
    "training.learning_rate": {"type": float, "default": 1e-4, "range": (0.0, 1.0)},
    "training.lr": {"type": float, "default": 1e-4, "range": (0.0, 1.0)},
    "training.weight_decay": {"type": float, "default": 0.0, "range": (0.0, 1.0)},
    "training.betas": {"type": list, "default": [0.9, 0.99]},
    "training.scheduler": {
        "type": str,
        "choices": ["step", "multistep", "cosine", "constant", "plateau"],
    },
    "training.epochs": {"type": int, "required": True, "range": (1, 100000)},
    "training.batch_size": {"type": int, "required": True, "range": (1, 4096)},
    "training.accumulation_steps": {"type": int, "default": 1, "range": (1, 256)},
    "training.mixed_precision": {"type": bool, "default": True},
    "training.gradient_checkpointing": {"type": bool, "default": False},
    "training.clip_grad_norm": {"type": _NUM_OR_NONE, "default": None, "range": (0.0, 1000.0)},
    "training.val_interval": {"type": int, "default": 5, "range": (1, 10000)},
    "training.save_interval": {"type": int, "default": 5, "range": (1, 10000)},
    "training.log_interval": {"type": int, "default": 100, "range": (1, 100000)},
    "training.use_ema": {"type": bool, "default": True},
    "training.ema_decay": {"type": float, "default": 0.999, "range": (0.0, 1.0)},
    "training.warmup_epochs": {"type": int, "default": 0, "range": (0, 10000)},
    "training.min_lr": {"type": float, "default": 1e-8, "range": (0.0, 1.0)},

    # ---- data ----
    "data.crop_size": {"type": int, "required": True, "range": (8, 4096)},
    "data.scale": {"type": int, "required": True, "range": (2, 4)},
    "data.augment": {"type": bool, "default": True},
    "data.num_workers": {"type": int, "default": 8, "range": (0, 64)},
    "data.pin_memory": {"type": bool, "default": True},
    "data.prefetch_factor": {"type": int, "default": 4, "range": (1, 32)},
    "data.persistent_workers": {"type": bool, "default": True},
    "data.val_split": {"type": float, "default": 0.1, "range": (0.0, 1.0)},
    "data.val_crop_size": {"type": int, "default": 256, "range": (8, 8192)},
    "data.sample_ratio": {"type": _NUM_OR_NONE, "default": None, "range": (0.0, 1.0)},

    # ---- inference ----
    "inference.device": {"type": str, "choices": ["cuda", "cpu", "auto"]},
    "inference.tile_size": {"type": int, "default": 0, "range": (0, 8192)},
    "inference.tile_overlap": {"type": int, "default": 16, "range": (0, 512)},
    "inference.tta": {"type": bool, "default": False},
    "inference.use_amp": {"type": bool, "default": True},
    "inference.batch_size": {"type": int, "default": 1, "range": (1, 64)},
    "inference.num_workers": {"type": int, "default": 4, "range": (0, 64)},
    "inference.output_format": {"type": str, "choices": ["png", "jpg", "webp"]},
    "inference.output_quality": {"type": int, "default": 95, "range": (1, 100)},
    "inference.color_mode": {"type": str, "choices": ["rgb", "ycbcr"]},

    # ---- logging ----
    "logging.level": {
        "type": str,
        "choices": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    },
    "logging.format": {
        "type": str,
        "default": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    },
    "logging.output": {"type": _STR_OR_NONE, "default": "logs"},
    "logging.use_tensorboard": {"type": bool, "default": True},

    # ---- loss ----
    "loss.pixel_loss.type": {"type": str, "choices": ["l1", "l2", "charbonnier"]},
    "loss.pixel_loss.weight": {"type": float, "default": 1.0, "range": (0.0, 100.0)},
    "loss.perceptual_loss.enabled": {"type": bool, "default": False},
    "loss.perceptual_loss.weight": {"type": float, "default": 0.1, "range": (0.0, 100.0)},
    "loss.adversarial_loss.enabled": {"type": bool, "default": False},
    "loss.adversarial_loss.weight": {"type": float, "default": 0.1, "range": (0.0, 100.0)},
    "loss.distillation.enabled": {"type": bool, "default": False},
    "loss.distillation.mtkd_weight": {"type": float, "default": 1.0, "range": (0.0, 100.0)},
    "loss.distillation.fakd_weight": {"type": float, "default": 0.5, "range": (0.0, 100.0)},

    # ---- checkpoint ----
    "checkpoint.save_dir": {"type": str, "default": "checkpoints"},
    "checkpoint.keep_last_n": {"type": int, "default": 5, "range": (0, 100)},
    "checkpoint.keep_best": {"type": bool, "default": True},

    # ---- output / paths ----
    "output.path": {"type": str, "default": "output"},
    "output.run_name": {"type": _STR_OR_NONE, "default": None},
    "output.duplicate_handling": {"type": str, "choices": ["auto_increment", "ask"]},

    # ---- security ----
    "security.allow_pickle_checkpoint": {"type": bool, "default": False},
    "security.verify_checksums": {"type": bool, "default": False},

    # ---- system ----
    "system.cudnn_benchmark": {"type": bool, "default": True},
    "system.cudnn_deterministic": {"type": bool, "default": False},
    "system.cuda_visible_devices": {"type": _STR_OR_NONE, "default": None},

    # ---- api ----
    "api.host": {"type": str, "default": "0.0.0.0"},
    "api.port": {"type": int, "default": 8000, "range": (1, 65535)},
    "api.max_file_size_mb": {"type": int, "default": 50, "range": (1, 10240)},
    "api.rate_limit_per_min": {"type": int, "default": 60, "range": (1, 100000)},
}


def get_schema() -> Dict[str, Dict[str, Any]]:
    """Return a deep copy of the default schema (safe to extend/modify)."""
    import copy
    return copy.deepcopy(DEFAULT_SCHEMA)
