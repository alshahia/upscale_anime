"""Test the full adaptive data pipeline"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Direct imports
import importlib.util
spec = importlib.util.spec_from_file_location('base', Path(__file__).parent.parent / "src" / "data" / "base.py")
base_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_module)
DatasetFactory = base_module.DatasetFactory

import yaml

# Load config
with open(Path(__file__).parent.parent / "configs" / "auto_stage_pipeline.yaml", 'r') as f:
    config = yaml.safe_load(f)

print("Testing DataLoader creation with auto crop_size and smart degradation...")

# Test dataset creation
dataset_cfg = config['data']['datasets'][0]
dataset_cfg['scale'] = config['model']['scale']
dataset_cfg['augment'] = config['data']['augment']
dataset_cfg['degradation'] = config['data']['degradation']
dataset_cfg['crop_size'] = config['data']['crop_size']
dataset_cfg['crop_size_mode'] = config['data']['crop_size_mode']
dataset_cfg['auto_crop_max'] = config['data']['auto_crop_max']
dataset_cfg['auto_crop_min'] = config['data']['auto_crop_min']

ds = DatasetFactory.create(dataset_cfg)
print(f"✓ Dataset created: {len(ds)} images")
print(f"✓ Auto-detected crop_size: {ds.crop_size}")
print(f"✓ Smart degradation enabled: {ds.use_quality_analyzer}")

# Test loading a sample
sample = ds[0]
lr_shape = sample['lr'].shape
hr_shape = sample['hr'].shape
print(f"✓ Sample loaded: LR shape {lr_shape}, HR shape {hr_shape}")

# Verify crop size matches
expected_hr_size = ds.crop_size
assert hr_shape[1] == expected_hr_size, f"HR height mismatch: {hr_shape[1]} != {expected_hr_size}"
assert hr_shape[2] == expected_hr_size, f"HR width mismatch: {hr_shape[2]} != {expected_hr_size}"

print("SUCCESS: Full pipeline working correctly!")
