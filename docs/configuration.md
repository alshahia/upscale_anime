# Configuration Management (WP-5)

The project uses a unified, hierarchical YAML configuration system implemented in
'src/anime_sr/utils/config.py`. All configs ultimately derive from
'configs/default.yaml`, either directly or through 'base:' inheritance.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Config Layout](#config-layout)
3. [Inheritance ('base' key)](#inheritance-base-key)
4. [Environment Variable Overrides](#environment-variable-overrides)
5. [Validation](#validation)
6. [Config Reference](#config-reference)
   - [Model](#model)
   - [Training](#training)
   - [Data](#data)
   - [Inference](#inference)
   - [Logging](#logging)
   - [Loss](#loss)
   - [Ensemble](#ensemble)
   - [Checkpoint](#checkpoint)
   - [Paths / Output](#paths--output)
   - [Security](#security)
   - [System](#system)
   - [API](#api)
7. [Python API](#python-api)
8. [Examples](#examples)

---

## Quick Start

~~~bash
# Load a config (inheritance + env overrides applied automatically)
python -c "import sys; sys.path.insert(0, 'src'); \
    from anime_sr.utils.config import load_config; \
    c = load_config('configs/training/model_a.yaml'); \
    print(c['model']['scale'])"

# Validate a config against the schema
python -c "import sys; sys.path.insert(0, 'src'); \
    from anime_sr.utils.config import load_config, validate_config; \
    validate_config(load_config('configs/default.yaml')); print('valid')"

# Train with a config
python scripts/train.py --config configs/training/model_a.yaml --dry-run
~~~

## Config Layout

~~~
configs/
├── default.yaml            # Canonical default — all sections, all defaults
├── training/
│   ├── model_a.yaml        # SPAN training (base: ../default.yaml)
│   ├── model_b.yaml        # Mamba-PAN training (base: ../default.yaml)
│   ├── ensemble.yaml       # Ensemble → tiny student (base: ../default.yaml)
│   └── distill.yaml        # Generic knowledge distillation (base: ../default.yaml)
├── inference/
│   └── default.yaml        # Inference defaults (base: ../default.yaml)
└── api/
    └── default.yaml        # API server defaults (base: ../default.yaml)
~~~

Precedence (lowest → highest):

1. 'configs/default.yaml' values
2. 'base:' inheritance (parent → child deep merge)
3. Environment variables ('DSH_' prefix)
4. CLI arguments

## Inheritance ('base' key)

A config can extend another config via the 'base' key. The path is resolved
**relative to the config file that declares it**:

~~~yaml
# configs/training/model_a.yaml
base: ../default.yaml        # inherits everything from configs/default.yaml

model:
  num_features: 26           # overrides just this one value
~~~

- Nested dicts are merged recursively; scalars and lists are replaced.
- The 'base' key is consumed during loading (never appears in the result).
- Inheritance chains are supported (a base may itself have a 'base').
- Loading with inheritance: 'load_config(path)' or 'Config.load(path)'.
- Programmatic merge: 'merge_configs(base_dict, override_dict)' (returns a new
  dict; inputs are never mutated).

## Environment Variable Overrides

Environment variables override config values at load time. Two mechanisms,
both using the 'DSH_' prefix:

### Explicit mappings

| Env var | Config key | Example |
|---|---|---|
| DSH_MODEL_PATH | model.path | DSH_MODEL_PATH=/models/best.pth |
| DSH_OUTPUT_PATH | output.path | DSH_OUTPUT_PATH=/mnt/results |
| DSH_LOG_LEVEL | logging.level | DSH_LOG_LEVEL=DEBUG |
| DSH_CUDA_VISIBLE_DEVICES | system.cuda_visible_devices | DSH_CUDA_VISIBLE_DEVICES=0,1 |
| DSH_ALLOW_PICKLE_CHECKPOINT | security.allow_pickle_checkpoint | DSH_ALLOW_PICKLE_CHECKPOINT=true |
| DSH_MAX_FILE_SIZE_MB | api.max_file_size_mb | DSH_MAX_FILE_SIZE_MB=100 |
| DSH_RATE_LIMIT_PER_MIN | api.rate_limit_per_min | DSH_RATE_LIMIT_PER_MIN=120 |
| DSH_API_KEY | api.api_key | DSH_API_KEY=secret |

### Generic dotted-path mapping

Any 'DSH_<SECTION>__<KEY>' env var maps to the dotted config path
'<section>.<key>' (double underscore separates path segments):

| Env var | Config key |
|---|---|
| DSH_TRAINING__BATCH_SIZE | training.batch_size |
| DSH_MODEL__SCALE | model.scale |
| DSH_TRAINING__MIXED_PRECISION | training.mixed_precision |
| DSH_INFERENCE__TILE_SIZE | inference.tile_size |

Generic overrides only apply to keys that **already exist** in the merged
config, so a typo cannot inject unexpected keys.

Values are coerced to the type of the existing config value:
'DSH_TRAINING__BATCH_SIZE=16' becomes the integer '16',
'DSH_TRAINING__MIXED_PRECISION=false' becomes boolean 'false',
'DSH_TRAINING__BETAS=0.8,0.95' becomes the list ['0.8', '0.95'].

Disable env overrides when loading:

~~~python
config = load_config('configs/default.yaml', apply_env=False)
~~~

## Validation

'validate_config(config, schema)' checks a config dict against a schema and
raises 'ConfigValidationError' listing **all** problems found:

~~~python
from anime_sr.utils.config import load_config, validate_config, ConfigValidationError

config = load_config('configs/default.yaml')
validate_config(config)  # uses the built-in DEFAULT_SCHEMA

try:
    validate_config({'model': {'scale': 7}})
except ConfigValidationError as e:
    print(e)
    # Config validation failed with N error(s):
    #   - model.name: missing required value
    #   - model.scale: 7 outside valid range [2, 4]
    #   ...
~~~

Schema format (dotted key → field spec), as defined in
'src/anime_sr/utils/config_schema.py':

~~~python
{
    'model.scale': {'type': int, 'required': True, 'range': (2, 4)},
    'training.optimizer': {'type': str, 'choices': ['adam', 'adamw', 'sgd']},
    'logging.level': {'type': str, 'default': 'INFO'},
}
~~~

Field spec keys: 'type' (bool is never accepted for int), 'required',
'default' (documented only), 'range' ((min, max) inclusive), 'choices'.

---

## Config Reference

All options below live in 'configs/default.yaml' with the shown defaults.

### Model

| Key | Default | Type | Valid range / choices |
|---|---|---|---|
| model.name | 'span' | str | required |
| model.type | 'span' | str | span, mamba_pan, tiny_student, edsr, rcan, swinir |
| model.scale | 4 | int | 2, 3, 4 |
| model.num_features | 32 | int | 1–512 |
| model.num_blocks | 12 | int | 1–64 |
| model.in_channels | 3 | int | 1, 3 |
| model.out_channels | 3 | int | 1, 3 |
| model.path | 'checkpoints/model_a_best.pth' | str | env: DSH_MODEL_PATH |
| model.pretrained_path | null | str/null | warm-start weights |
| model.strict_load | true | bool | |
| model.use_lora | false | bool | |
| model.lora_rank | 4 | int | 1–64 |
| model.lora_alpha | 1.0 | float | 0–100 |

### Training

| Key | Default | Type | Valid range / choices |
|---|---|---|---|
| training.mode | 'model_a' | str | model_a, model_b, both_parallel, both_sequential, ensemble, full, finetune, auto_stage |
| training.device | 'cuda' | str | cuda, cpu, auto |
| training.seed | 42 | int | |
| training.optimizer | 'adam' | str | adam, adamw, sgd |
| training.learning_rate / training.lr | 0.0001 | float | (0, 1) |
| training.weight_decay | 0.0 | float | [0, 1] |
| training.betas | [0.9, 0.99] | list | |
| training.scheduler | 'cosine' | str | step, multistep, cosine, constant, plateau |
| training.step_size | 200000 | int | step scheduler decay interval |
| training.gamma | 0.5 | float | step scheduler decay factor |
| training.milestones | [100, 200, 300] | list | multistep scheduler |
| training.T_max | 1000 | int | cosine scheduler total epochs |
| training.warmup_epochs | 0 | int | 0–10000 |
| training.min_lr | 1e-8 | float | (0, 1) |
| training.epochs | 1000 | int | required, 1–100000 |
| training.batch_size | 8 | int | required, 1–4096 |
| training.accumulation_steps | 1 | int | 1–256 |
| training.mixed_precision | true | bool | |
| training.gradient_checkpointing | false | bool | |
| training.clip_grad_norm | null | float/null | 0–1000 |
| training.val_interval | 5 | int | 1–10000 |
| training.save_interval | 5 | int | 1–10000 |
| training.log_interval | 100 | int | 1–100000 |
| training.use_ema | true | bool | |
| training.ema_decay | 0.999 | float | [0, 1] |
| training.early_stopping.* | see default.yaml | | enabled, monitor, mode, patience, min_delta, restore_best_weights, min_epochs |
| training.stage1.* | see default.yaml | | knowledge aggregation (teachers, aggregation_type, ...) |
| training.stage2.* | see default.yaml | | student distillation stage |

### Data

| Key | Default | Type | Valid range / choices |
|---|---|---|---|
| data.datasets | see default.yaml | list | entries: name, weight, enabled, hr_dir, lr_dir, preload, max_images |
| data.crop_size | 128 | int | required, 8–4096 |
| data.scale | 4 | int | required, 2–4 (must match model.scale) |
| data.augment | true | bool | random hflip/vflip/rot90 |
| data.crop_size_mode | 'manual' | str | manual, auto |
| data.degradation.* | see default.yaml | | mode: auto, light, medium, heavy, anime, disabled |
| data.num_workers | 8 | int | 0–64 |
| data.pin_memory | true | bool | |
| data.prefetch_factor | 4 | int | 1–32 |
| data.persistent_workers | true | bool | |
| data.val_split | 0.1 | float | [0, 1] |
| data.val_crop_size | 256 | int | 8–8192 |
| data.sample_ratio | null | float/null | [0, 1] |
| data.max_images_total | null | int/null | |

### Inference

| Key | Default | Type | Valid range / choices |
|---|---|---|---|
| inference.device | 'cuda' | str | cuda, cpu, auto |
| inference.tile_size | 0 | int | 0–8192 (0 = no tiling) |
| inference.tile_overlap | 16 | int | 0–512 (LR-space pixels) |
| inference.tta | false | bool | 8x D4 flip/rot ensemble |
| inference.tta_mode | 'd4' | str | d4 |
| inference.use_amp | true | bool | |
| inference.batch_size | 1 | int | 1–64 |
| inference.num_workers | 4 | int | 0–64 |
| inference.output_format | 'png' | str | png, jpg, webp |
| inference.output_quality | 95 | int | 1–100 (jpg/webp) |
| inference.color_mode | 'rgb' | str | rgb, ycbcr |
| inference.save_intermediate | false | bool | |

### Logging

| Key | Default | Type | Valid range / choices |
|---|---|---|---|
| logging.level | 'INFO' | str | DEBUG, INFO, WARNING, ERROR, CRITICAL; env: DSH_LOG_LEVEL |
| logging.format | '%(asctime)s - %(name)s - %(levelname)s - %(message)s' | str | |
| logging.output | 'logs' | str/null | null = console only |
| logging.file | 'anime_sr.log' | str | |
| logging.max_bytes | 10485760 | int | 10 MB per rotating file |
| logging.backup_count | 5 | int | |
| logging.use_tensorboard | true | bool | |
| logging.tensorboard_dir | 'runs' | str | |
| logging.log_gpu_stats | false | bool | |
| logging.log_system_stats | false | bool | |

### Loss

| Key | Default | Type | Valid range / choices |
|---|---|---|---|
| loss.pixel_loss.type | 'l1' | str | l1, l2, charbonnier |
| loss.pixel_loss.weight | 1.0 | float | 0–100 |
| loss.pixel_loss.eps | 1e-6 | float | charbonnier epsilon |
| loss.perceptual_loss.enabled | false | bool | |
| loss.perceptual_loss.type | 'vgg' | str | vgg, resnet, dists, dual |
| loss.perceptual_loss.weight | 0.1 | float | 0–100 |
| loss.adversarial_loss.enabled | false | bool | |
| loss.adversarial_loss.type | 'vanilla' | str | vanilla, relativistic, hinge |
| loss.adversarial_loss.weight | 0.1 | float | 0–100 |
| loss.distillation.enabled | false | bool | |
| loss.distillation.mtkd_weight | 1.0 | float | 0–100 |
| loss.distillation.fakd_weight | 0.5 | float | 0–100 |
| loss.distillation.fakd_layers | [2, 4, 6, 8] | list | |
| loss.distillation.fakd_layer_weights | [0.1, 0.2, 0.3, 0.4] | list | |

### Ensemble

| Key | Default | Type | Valid range / choices |
|---|---|---|---|
| ensemble.weights | [0.6, 0.4] | list | must sum to 1.0 |
| ensemble.learnable_weights | false | bool | |
| ensemble.model_a_checkpoint | 'checkpoints/model_a_best.pth' | str | |
| ensemble.model_b_checkpoint | 'checkpoints/model_b_best.pth' | str | |

### Checkpoint

| Key | Default | Type | Valid range / choices |
|---|---|---|---|
| checkpoint.save_dir | 'checkpoints' | str | |
| checkpoint.keep_last_n | 5 | int | 0–100 |
| checkpoint.keep_best | true | bool | |
| checkpoint.save_optimizer | true | bool | |
| checkpoint.save_scaler | true | bool | |

### Paths / Output

| Key | Default | Type | Valid range / choices |
|---|---|---|---|
| paths.data_root | 'data' | str | |
| paths.log_dir | 'logs' | str | |
| paths.checkpoint_dir | 'checkpoints' | str | |
| paths.result_dir | 'results' | str | |
| paths.pretrained_dir | 'pretrained' | str | |
| paths.output | 'output' | str | |
| output.path | 'output' | str | env: DSH_OUTPUT_PATH |
| output.run_name | null | str/null | null = auto (timestamp_model_suffix) |
| output.duplicate_handling | 'auto_increment' | str | auto_increment, ask |

### Security

| Key | Default | Type | Valid range / choices |
|---|---|---|---|
| security.allow_pickle_checkpoint | false | bool | env: DSH_ALLOW_PICKLE_CHECKPOINT; only enable for trusted checkpoints |
| security.verify_checksums | false | bool | |

### System

| Key | Default | Type | Valid range / choices |
|---|---|---|---|
| system.cudnn_benchmark | true | bool | |
| system.cudnn_deterministic | false | bool | |
| system.cuda_visible_devices | null | str/null | env: DSH_CUDA_VISIBLE_DEVICES |
| system.deterministic | false | bool | full reproducibility (slow) |
| system.warn_fallback | true | bool | |

### API

| Key | Default | Type | Valid range / choices |
|---|---|---|---|
| api.host | '0.0.0.0' | str | |
| api.port | 8000 | int | 1–65535 |
| api.api_key | null | str/null | env: DSH_API_KEY; null = dev mode (auth disabled) |
| api.allowed_origins | ['http://localhost:3000', 'http://localhost:8000'] | list | |
| api.max_file_size_mb | 50 | int | 1–10240; env: DSH_MAX_FILE_SIZE_MB |
| api.rate_limit_per_min | 60 | int | 1–100000; env: DSH_RATE_LIMIT_PER_MIN |
| api.workers | 1 | int | uvicorn workers |
| api.timeout_keep_alive | 5 | int | |

---

## Python API

All functions live in 'anime_sr.utils.config' (schema in
'anime_sr.utils.config_schema'):

~~~python
from anime_sr.utils.config import (
    load_config,          # load YAML with base: inheritance + env overrides
    merge_configs,        # deep-merge two dicts (override wins)
    validate_config,      # schema validation, raises ConfigValidationError
    get_config_value,     # dot-notation access: get_config_value(c, 'training.lr')
    apply_env_overrides,  # apply DSH_* env vars to a dict, returns applied map
    Config,               # hierarchical config object (get/set/merge/save)
    ConfigValidationError,
)
from anime_sr.utils.config_schema import DEFAULT_SCHEMA
~~~

~~~python
# Load with inheritance and env overrides
config = load_config('configs/training/model_a.yaml')

# Load without env overrides
config = load_config('configs/default.yaml', apply_env=False)

# Dot-notation access with default
lr = get_config_value(config, 'training.learning_rate', 1e-4)

# Merge programmatically
merged = merge_configs(base_dict, override_dict)

# Validate (uses DEFAULT_SCHEMA when schema=None)
validate_config(config)

# Config object (attribute-style CLI merging, save to YAML)
cfg = Config.load('configs/default.yaml')
cfg.set('training.batch_size', 16)
cfg.save('configs/my_run.yaml')
~~~

## Examples

### Override via environment

~~~bash
# Use a different checkpoint and quiet the logs
DSH_MODEL_PATH=/models/span_tiny.pth DSH_LOG_LEVEL=WARNING \
    python scripts/inference.py --config configs/inference/default.yaml --input lr.png --output sr.png

# Generic dotted-path override
DSH_TRAINING__BATCH_SIZE=16 DSH_TRAINING__EPOCHS=500 \
    python scripts/train.py --config configs/training/model_a.yaml
~~~

### Custom config via inheritance

~~~yaml
# configs/training/my_experiment.yaml
base: ../default.yaml

model:
  num_features: 48
  num_blocks: 16

training:
  epochs: 500
  batch_size: 4
  lr: 0.0002

data:
  crop_size: 96
~~~

### Validate before a long run

~~~python
import sys
sys.path.insert(0, 'src')
from anime_sr.utils.config import load_config, validate_config, ConfigValidationError

try:
    validate_config(load_config('configs/training/my_experiment.yaml'))
    print('config OK')
except ConfigValidationError as e:
    print(e)  # lists every problem; fix all before launching training
    sys.exit(1)
~~~

---

## Notes

- 'configs/base.yaml' is the legacy root config kept for existing scripts; new
  configs should inherit from 'configs/default.yaml'.
- The 'base' key is consumed during loading and never appears in the merged
  result.
- Env overrides are applied **after** inheritance, so they always win over
  both the child config and its base.
- 'bool' values from env vars accept '1/true/yes/on' (case-insensitive) and
  '0/false/no/off'.