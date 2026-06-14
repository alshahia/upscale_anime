"""Test MultiDataset with batch loading"""
import sys
from pathlib import Path

def main():
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    import yaml
    import torch

    # Import with direct loading
    import importlib.util
    spec = importlib.util.spec_from_file_location('dataloader', Path(__file__).parent.parent / "src" / "data" / "dataloader.py")
    dl_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl_module)
    DataLoaderFactory = dl_module.DataLoaderFactory

    # Load config
    with open(Path(__file__).parent.parent / "configs" / "auto_stage_pipeline.yaml", 'r') as f:
        config = yaml.safe_load(f)

    print("Testing MultiDataset with batch loading...")

    # Test training loader
    train_loader = DataLoaderFactory.create(config, is_train=True)
    print(f"✓ Training loader: {len(train_loader)} batches")

    # Load a batch
    batch = next(iter(train_loader))
    print(f"✓ Batch loaded: LR {batch['lr'].shape}, HR {batch['hr'].shape}")

    # Verify all samples have same shape
    assert batch['lr'].shape[0] == batch['hr'].shape[0], "Batch size mismatch"
    print(f"✓ Batch size: {batch['lr'].shape[0]}")

    print("\nSUCCESS: MultiDataset batch collation works!")

if __name__ == '__main__':
    main()
