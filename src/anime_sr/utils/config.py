"""
Configuration management system
Hierarchical YAML configs with CLI override support

WP-5 extensions:
- merge_configs(base, override): deep-merge two config dicts
- validate_config(config, schema): schema-based validation with clear errors
- get_config_value(config, key, default): dot-notation nested access
- Environment variable overrides (DSH_* prefix) applied on load
- Config inheritance via the 'base' key (pre-existing, now shared logic)
"""
import copy
import os
import yaml
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple
from dataclasses import dataclass, field


# ------------------------------------------------------------
# Errors
# ------------------------------------------------------------
class ConfigValidationError(ValueError):
    """Raised when a config fails schema validation.

    The message lists every problem found (not just the first),
    so users can fix all issues in one pass.
    """


# ------------------------------------------------------------
# Standalone helpers (WP-5)
# ------------------------------------------------------------
def get_config_value(config: Dict, key: str, default: Any = None) -> Any:
    """Get a nested config value using dot notation.

    Args:
        config: Config dict (e.g. loaded via load_config).
        key: Dot-notation path, e.g. 'training.batch_size'.
             Numeric segments index into lists: 'data.datasets.0.hr_dir'.
        default: Value returned when the path is missing or empty.

    Returns:
        The value at 'key', or 'default' if not present.
    """
    if not isinstance(config, dict) or not key:
        return default

    keys = key.split('.')
    value: Any = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        elif isinstance(value, list) and k.isdigit() and int(k) < len(value):
            value = value[int(k)]
        else:
            return default
    return value


def _config_has_key(config: Dict, key: str) -> bool:
    """True when the dotted path exists in config (value may be None)."""
    keys = key.split('.')
    value: Any = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        elif isinstance(value, list) and k.isdigit() and int(k) < len(value):
            value = value[int(k)]
        else:
            return False
    return True


def merge_configs(base: Optional[Dict], override: Optional[Dict]) -> Dict:
    """Deep-merge two config dicts; override wins on conflicts.

    - Nested dicts are merged recursively.
    - Lists and scalars in 'override' replace the base value.
    - Neither input is mutated; a new dict is returned.

    Args:
        base: Base config dict (e.g. from a 'base:' inheritance target).
        override: Override config dict (e.g. the child config).

    Returns:
        A new dict containing the merged result.
    """
    if base is None:
        base = {}
    if override is None:
        override = {}
    if not isinstance(base, dict) or not isinstance(override, dict):
        raise TypeError(
            f"merge_configs expects two dicts, got {type(base).__name__} and {type(override).__name__}"
        )

    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _matches_type(value: Any, expected: Union[type, Tuple[type, ...]]) -> bool:
    """Type check that keeps bool distinct from int (bool is an int subclass)."""
    types = expected if isinstance(expected, tuple) else (expected,)
    if isinstance(value, bool) and bool not in types:
        return False
    return isinstance(value, types)


def validate_config(config: Dict, schema: Optional[Dict] = None) -> bool:
    """Validate a config dict against a schema.

    Schema format — dotted key -> field spec:
        {
            "model.scale": {"type": int, "required": True, "range": (2, 4)},
            "training.optimizer": {"type": str, "choices": ["adam", "adamw", "sgd"]},
            "logging.level": {"type": str, "default": "INFO"},
        }

    Field spec keys:
        - type: expected Python type (or tuple of types). bool is never
          accepted where int is expected.
        - required: when True, a missing/None value is an error.
        - default: documented default (not applied during validation).
        - range: (min, max) inclusive numeric bounds.
        - choices: iterable of allowed values.

    Args:
        config: Config dict to validate.
        schema: Schema dict. When None, the built-in DEFAULT_SCHEMA
            (from anime_sr.utils.config_schema) is used.

    Returns:
        True when the config is valid.

    Raises:
        ConfigValidationError: listing every problem found.
    """
    if schema is None:
        from anime_sr.utils.config_schema import DEFAULT_SCHEMA
        schema = DEFAULT_SCHEMA

    errors = []
    for key, spec in schema.items():
        if not isinstance(spec, dict):
            continue
        value = get_config_value(config, key)

        if value is None:
            if spec.get("required"):
                errors.append(f"{key}: missing required value")
            continue

        expected_type = spec.get("type")
        if expected_type is not None and not _matches_type(value, expected_type):
            type_names = (
                expected_type if isinstance(expected_type, tuple) else (expected_type,)
            )
            expected_str = " or ".join(t.__name__ for t in type_names)
            errors.append(
                f"{key}: expected type {expected_str}, got {type(value).__name__} ({value!r})"
            )
            continue  # range/choices checks are meaningless on a type mismatch

        choices = spec.get("choices")
        if choices is not None and value not in choices:
            errors.append(f"{key}: {value!r} not in allowed choices {list(choices)}")

        range_spec = spec.get("range")
        if range_spec is not None:
            low, high = range_spec
            try:
                if value < low or value > high:
                    errors.append(f"{key}: {value} outside valid range [{low}, {high}]")
            except TypeError:
                errors.append(f"{key}: {value!r} is not comparable against range [{low}, {high}]")

    if errors:
        raise ConfigValidationError(
            "Config validation failed with " + str(len(errors)) + " error(s):\n  - "
            + "\n  - ".join(errors)
        )
    return True


# ------------------------------------------------------------
# Environment variable overrides
# ------------------------------------------------------------
# Explicit, documented mappings (see docs/configuration.md).
ENV_VAR_OVERRIDES: Dict[str, str] = {
    "DSH_MODEL_PATH": "model.path",
    "DSH_OUTPUT_PATH": "output.path",
    "DSH_LOG_LEVEL": "logging.level",
    "DSH_CUDA_VISIBLE_DEVICES": "system.cuda_visible_devices",
    "DSH_ALLOW_PICKLE_CHECKPOINT": "security.allow_pickle_checkpoint",
    "DSH_MAX_FILE_SIZE_MB": "api.max_file_size_mb",
    "DSH_RATE_LIMIT_PER_MIN": "api.rate_limit_per_min",
    "DSH_API_KEY": "api.api_key",
}

ENV_PREFIX = "DSH_"


def _coerce_env_value(raw: str, current: Any) -> Any:
    """Coerce an env var string to the type of the existing config value."""
    if isinstance(current, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        return [item.strip() for item in raw.split(",") if item.strip()]
    if current is None:
        # No existing value to match: infer the type.
        low = raw.strip().lower()
        if low in ("true", "false"):
            return low == "true"
        try:
            return int(raw)
        except ValueError:
            pass
        try:
            return float(raw)
        except ValueError:
            pass
        return raw
    return raw


def config_set_nested(config: Dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using dot notation (creates intermediates)."""
    keys = dotted_key.split('.')
    node = config
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def apply_env_overrides(
    config: Dict,
    prefix: str = ENV_PREFIX,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Apply environment variable overrides to a config dict (in place).

    Two mechanisms, both using the DSH_ prefix:

    1. Explicit mappings (ENV_VAR_OVERRIDES), e.g.
       DSH_MODEL_PATH -> model.path, DSH_LOG_LEVEL -> logging.level.
    2. Generic dotted-path mapping: a double underscore separates path
       segments, e.g. DSH_TRAINING__BATCH_SIZE -> training.batch_size,
       DSH_MODEL__SCALE -> model.scale. Generic overrides only apply to
       keys that already exist in the config (typos cannot inject keys).

    Values are coerced to the type of the existing config value
    (bool/int/float/list), so DSH_TRAINING__BATCH_SIZE=16 becomes the
    integer 16, not the string "16".

    Args:
        config: Config dict to update (mutated in place).
        prefix: Only env vars starting with this prefix are considered.
        env: Environment mapping (defaults to os.environ; for testing).

    Returns:
        Dict of applied overrides: {dotted_key: new_value}.
    """
    if env is None:
        env = dict(os.environ)
    if not isinstance(config, dict):
        raise TypeError(f"apply_env_overrides expects a dict, got {type(config).__name__}")

    applied: Dict[str, Any] = {}

    # 1. Explicit mappings (always trusted, even when the key is null/missing).
    for env_var, dotted_key in ENV_VAR_OVERRIDES.items():
        raw = env.get(env_var)
        if raw is None:
            continue
        if not _config_has_key(config, dotted_key):
            continue
        current = get_config_value(config, dotted_key)
        config_set_nested(config, dotted_key, _coerce_env_value(raw, current))
        applied[dotted_key] = get_config_value(config, dotted_key)

    # 2. Generic DSH_<SECTION>__<KEY> -> dotted.path overrides.
    for env_var, raw in env.items():
        if not env_var.startswith(prefix) or env_var in ENV_VAR_OVERRIDES:
            continue
        path_parts = env_var[len(prefix):].split("__")
        if len(path_parts) < 2 or any(not part for part in path_parts):
            continue
        dotted_key = ".".join(part.strip().lower() for part in path_parts)
        if not _config_has_key(config, dotted_key):
            continue
        current = get_config_value(config, dotted_key)
        config_set_nested(config, dotted_key, _coerce_env_value(raw, current))
        applied[dotted_key] = get_config_value(config, dotted_key)

    return applied


# ------------------------------------------------------------
# YAML loading with inheritance + env overrides
# ------------------------------------------------------------
def load_config(path: Union[str, Path], apply_env: bool = True) -> dict:
    """Load a YAML config file (with base: inheritance) as a plain dict.

    Mirrors the per-script `load_config` helpers in scripts/precompute_degradation.py,
    scripts/debug_transfer_learning.py, scripts/manage_stages.py. Returns the merged
    dict (NOT a Config instance) for compatibility with those call sites.

    Inheritance: a config may specify 'base: path/to/base.yaml' (resolved
    relative to the config file). The base is loaded first, then the child
    is deep-merged over it via merge_configs().

    Environment overrides: when apply_env is True (default), DSH_* env vars
    are applied after inheritance (see apply_env_overrides and
    docs/configuration.md).

    Args:
        path: Path to the YAML config file.
        apply_env: Apply DSH_* environment variable overrides.

    Returns:
        The fully merged config dict (the 'base' key is consumed).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f) or {}

    if 'base' in config_dict:
        base_path = path.parent / config_dict.pop('base')
        base = load_config(base_path, apply_env=False)
        config_dict = merge_configs(base, config_dict)

    if apply_env:
        apply_env_overrides(config_dict)

    return config_dict


class Config:
    """
    Hierarchical configuration manager.
    Supports: YAML loading, CLI argument merging, nested access
    """

    def __init__(self, config_dict: Optional[Dict] = None):
        self._config = config_dict or {}

    @classmethod
    def load(cls, path: Union[str, Path], apply_env: bool = True) -> 'Config':
        """Load config from YAML file"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        # Handle inheritance (base config)
        if 'base' in config_dict:
            base_path = path.parent / config_dict.pop('base')
            base_config = cls.load(base_path, apply_env=False)
            base_config.merge(config_dict)
            config_dict = base_config._config

        if apply_env:
            apply_env_overrides(config_dict)

        return cls(config_dict)

    def merge(self, other: Union[Dict, 'Config']):
        """Merge another config into this one (recursive)"""
        if isinstance(other, Config):
            other = other._config
        self._config = merge_configs(self._config, other)

    def merge_args(self, args: argparse.Namespace):
        """Merge CLI arguments (dot notation: --model.channels 32)"""
        args_dict = vars(args)
        for key, value in args_dict.items():
            if value is not None:
                self._set_nested(key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with dot notation (e.g., 'model.channels')"""
        return get_config_value(self._config, key, default)

    def set(self, key: str, value: Any):
        """Set value with dot notation"""
        self._set_nested(key, value)

    def _set_nested(self, key: str, value: Any):
        """Set nested value using dot notation, supports list indices"""
        # Validate key format
        if not key or not isinstance(key, str):
            raise ValueError(f"Config key must be a non-empty string, got: {type(key).__name__}")

        keys = key.split('.')

        # Validate each key segment
        for i, k in enumerate(keys):
            # Check for path traversal sequences
            if k == '..' or k.startswith('..') or '/../' in k or '\\..\\' in k:
                raise ValueError(f"Invalid config key (path traversal detected): {key}")
            # Allow numeric indices for lists and valid identifiers for dicts
            if not (k.isdigit() or k.isidentifier() or k.replace('-', '').isalnum()):
                raise ValueError(f"Invalid config key segment '{k}' in key: {key}")

        config = self._config

        for i, k in enumerate(keys[:-1]):
            # Check if next key is numeric (list index)
            next_key = keys[i + 1]
            is_next_numeric = next_key.isdigit()

            # If k is numeric string, use as list index
            if k.isdigit():
                idx = int(k)
                # Ensure list is long enough
                while len(config) <= idx:
                    config.append({} if not is_next_numeric else [])
                config = config[idx]
            else:
                # k is a string key
                if k not in config:
                    # Create list or dict based on next key
                    config[k] = [] if is_next_numeric else {}
                config = config[k]

        # Set final value
        final_key = keys[-1]
        if final_key.isdigit():
            idx = int(final_key)
            while len(config) <= idx:
                config.append(None)
            config[idx] = value
        else:
            config[final_key] = value

    def validate(self, schema: Optional[Dict] = None) -> bool:
        """Validate config against schema (basic checks)"""
        required_keys = [
            'model.name',
            'model.scale',
            'training.batch_size',
            'training.epochs',
            'data.crop_size',
        ]

        for key in required_keys:
            if self.get(key) is None:
                raise ValueError(f"Missing required config key: {key}")

        # Type and range validation for critical values
        batch_size = self.get('training.batch_size')
        if batch_size is not None and batch_size <= 0:
            raise ValueError(f"training.batch_size must be positive, got {batch_size}")

        epochs = self.get('training.epochs')
        if epochs is not None and epochs <= 0:
            raise ValueError(f"training.epochs must be positive, got {epochs}")

        lr = self.get('training.learning_rate') or self.get('training.lr')
        if lr is not None and lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")

        scale = self.get('model.scale')
        if scale is not None and scale not in [2, 3, 4]:
            raise ValueError(f"model.scale must be 2, 3, or 4, got {scale}")

        crop_size = self.get('data.crop_size')
        if crop_size is not None and crop_size <= 0:
            raise ValueError(f"data.crop_size must be positive, got {crop_size}")

        return True

    def resolve_with_fallback(self, keys: list, default: Any = None, warning_msg: str = None) -> Any:
        """
        Try multiple config keys in order, return first non-None value.
        Prints warning if fallback to default is used.

        Args:
            keys: List of dot-notation keys to try in order
            default: Default value if all keys are None
            warning_msg: Custom warning message (optional)

        Returns:
            First non-None value from keys, or default
        """
        for key in keys:
            value = self.get(key)
            if value is not None:
                return value

        # All keys were None, using default
        if warning_msg:
            print(f"[WARNING] {warning_msg}")
        else:
            print(f"[WARNING] None of {keys} found in config, using default={default}")
            print(f"  To fix: Add one of these keys to your config file")

        return default

    def to_dict(self) -> Dict:
        """Export to dictionary (deep copy to prevent mutation)"""
        return copy.deepcopy(self._config)

    def save(self, path: Union[str, Path]):
        """Save config to YAML file"""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False)

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access: config['model.channels']"""
        return self.get(key)

    def __setitem__(self, key: str, value: Any):
        """Allow dict-like setting: config['model.channels'] = 32"""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """Allow 'in' checks: 'model.channels' in config"""
        return self.get(key) is not None

    def __repr__(self) -> str:
        return f"Config({self._config})"

    @staticmethod
    def add_argument_parser(parser: argparse.ArgumentParser):
        """Add standard arguments to argparse"""
        parser.add_argument('--config', '-c', type=str, required=True,
                          help='Path to config YAML file')
        parser.add_argument('--mode', '-m', type=str,
                          choices=['model_a', 'model_b', 'both_parallel', 'both_sequential', 'ensemble', 'full'],
                          help='Training mode')
        parser.add_argument('--resume', '-r', type=str,
                          help='Resume from checkpoint path')
        parser.add_argument('--device', type=str, default='cuda',
                          help='Device to use (cuda/cpu)')
        parser.add_argument('--num-workers', type=int,
                          help='Number of dataloader workers')
        parser.add_argument('--batch-size', type=int,
                          help='Override batch size')
        parser.add_argument('--epochs', type=int,
                          help='Override number of epochs')
        parser.add_argument('--lr', type=float,
                          help='Override learning rate')
        parser.add_argument('--dry-run', action='store_true',
                          help='Validate config and exit')
        parser.add_argument('--tensorboard', action='store_true',
                          help='Enable TensorBoard logging')
