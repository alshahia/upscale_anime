"""
Configuration management system
Hierarchical YAML configs with CLI override support
"""
import yaml
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field


def load_config(path: Union[str, Path]) -> dict:
    """Load a YAML config file (with base: inheritance) as a plain dict.

    Mirrors the per-script `load_config` helpers in scripts/precompute_degradation.py,
    scripts/debug_transfer_learning.py, scripts/manage_stages.py. Returns the merged
    dict (NOT a Config instance) for compatibility with those call sites.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f) or {}
    if 'base' in config_dict:
        base_path = path.parent / config_dict.pop('base')
        base = load_config(base_path)

        def _merge(b, o):
            for k, v in o.items():
                if k in b and isinstance(b[k], dict) and isinstance(v, dict):
                    b[k] = _merge(b[k], v)
                else:
                    b[k] = v
            return b

        return _merge(base, config_dict)
    return config_dict


class Config:
    """
    Hierarchical configuration manager.
    Supports: YAML loading, CLI argument merging, nested access
    """
    
    def __init__(self, config_dict: Optional[Dict] = None):
        self._config = config_dict or {}
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Config':
        """Load config from YAML file"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # Handle inheritance (base config)
        if 'base' in config_dict:
            base_path = path.parent / config_dict.pop('base')
            base_config = cls.load(base_path)
            base_config.merge(config_dict)
            return base_config
        
        return cls(config_dict)
    
    def merge(self, other: Union[Dict, 'Config']):
        """Merge another config into this one (recursive)"""
        if isinstance(other, Config):
            other = other._config
        
        def recursive_merge(base: Dict, override: Dict) -> Dict:
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    base[key] = recursive_merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        self._config = recursive_merge(self._config, other)
    
    def merge_args(self, args: argparse.Namespace):
        """Merge CLI arguments (dot notation: --model.channels 32)"""
        args_dict = vars(args)
        for key, value in args_dict.items():
            if value is not None:
                self._set_nested(key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value with dot notation (e.g., 'model.channels')"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
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
        import copy
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
