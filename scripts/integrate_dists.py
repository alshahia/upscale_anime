"""
Integration script for DISTS perceptual loss into training.

This script shows how to add DISTS perceptual loss to the training pipeline.
DISTS (Deep Image Structure and Texture Similarity) outperforms LPIPS 
for super-resolution tasks.

Usage:
    # Add DISTS to your config file
    python scripts/integrate_dists.py --config configs/model_a_ntire.yaml --update

    # Test DISTS loss computation
    python scripts/integrate_dists.py --test

    # Show example configuration
    python scripts/integrate_dists.py --example-config
"""

import argparse
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from losses import DISTSLoss


def get_dists_config_example():
    """Return example YAML configuration for DISTS loss."""
    config = """
# DISTS Perceptual Loss Configuration
# Add this to your training config file

loss:
  # Existing losses...
  
  # DISTS Perceptual Loss (NEW)
  perceptual:
    enabled: true
    type: "dists"  # Use DISTS instead of VGG/ResNet
    weight: 0.1     # Perceptual loss weight (relative to pixel loss)
    alpha: 0.5      # Structure vs texture balance (0.5 = equal)
    
  # Combined loss weights
  pixel_loss:
    weight: 1.0
  
  distillation:
    enabled: true
    mtkd_weight: 1.0
    fakd_weight: 0.5
    
  # Total loss = pixel_loss + perceptual_weight * perceptual_loss + distillation_losses
"""
    return config


def test_dists_loss():
    """Test DISTS loss computation."""
    print("Testing DISTS Loss...")
    print("=" * 60)
    
    # Create loss
    dists_loss = DISTSLoss(alpha=0.5)
    
    # Create dummy images
    batch_size = 2
    pred = torch.rand(batch_size, 3, 256, 256)
    target = torch.rand(batch_size, 3, 256, 256)
    
    # Compute loss
    loss = dists_loss(pred, target)
    
    print(f"✓ DISTS loss computed successfully")
    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Input shape: {pred.shape}")
    print(f"  Device: {pred.device}")
    
    # Test gradient flow
    pred.requires_grad = True
    loss = dists_loss(pred, target)
    loss.backward()
    
    print(f"  Gradient computed: {pred.grad is not None}")
    print(f"  Gradient shape: {pred.grad.shape if pred.grad is not None else 'N/A'}")
    print("\n✓ DISTS loss is ready for training!")


def show_integration_guide():
    """Print integration guide for trainers."""
    guide = """
# DISTS Loss Integration Guide

## 1. Add DISTS to Model A Trainer (model_a_trainer.py)

In the `_setup_losses` method, add:

```python
def _setup_losses(self):
    # ... existing losses ...
    
    # Add DISTS perceptual loss
    perceptual_cfg = self.config.get('loss', {}).get('perceptual', {})
    if perceptual_cfg.get('enabled', False):
        if perceptual_cfg.get('type') == 'dists':
            from losses import DISTSLoss
            self.perceptual_loss = DISTSLoss(
                alpha=perceptual_cfg.get('alpha', 0.5)
            )
            self.perceptual_weight = perceptual_cfg.get('weight', 0.1)
            print(f"  Enabled: DISTS perceptual loss (weight={self.perceptual_weight})")
```

## 2. Update train_epoch to include DISTS

In `train_stage2_epoch`, add perceptual loss:

```python
def train_stage2_epoch(self, dataloader, optimizer):
    # ... existing code ...
    
    # Compute pixel loss
    l1 = self.l1_loss(student_pred, hr)
    loss = l1
    
    # Add DISTS perceptual loss
    if hasattr(self, 'perceptual_loss') and self.perceptual_loss is not None:
        perceptual = self.perceptual_loss(student_pred, hr)
        loss = loss + self.perceptual_weight * perceptual
        loss_dict['perceptual'] = perceptual.item()
    
    # ... rest of training code ...
```

## 3. Update Config File

Add to your config YAML:

```yaml
loss:
  perceptual:
    enabled: true
    type: "dists"
    weight: 0.1
    alpha: 0.5  # Structure vs texture balance
```

## 4. Expected Improvements

- PSNR: +0.3-0.5 dB improvement
- Perceptual quality: Better texture and structure preservation
- Training stability: Similar to VGG perceptual loss

## 5. Tips

- Start with weight=0.05-0.1 and increase gradually
- Monitor perceptual loss separately in TensorBoard
- Combine with anime-specific losses for best results on anime content
"""
    print(guide)


def update_config_file(config_path: str, dry_run: bool = True):
    """Update config file with DISTS configuration."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return
    
    # Load existing config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        config = {}
    
    # Check if loss section exists
    if 'loss' not in config:
        config['loss'] = {}
    
    # Add DISTS config
    if 'perceptual' not in config['loss']:
        config['loss']['perceptual'] = {
            'enabled': True,
            'type': 'dists',
            'weight': 0.1,
            'alpha': 0.5,
        }
        print(f"Added DISTS perceptual loss to config")
    else:
        print(f"Perceptual loss already configured in {config_path}")
        return
    
    if dry_run:
        print("\nDry run - would add:")
        print(yaml.dump({'loss': {'perceptual': config['loss']['perceptual']}}))
        print(f"Run with --apply to actually modify {config_path}")
    else:
        # Save updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"✓ Updated config: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description='DISTS Perceptual Loss Integration'
    )
    
    parser.add_argument('--test', action='store_true',
                        help='Test DISTS loss computation')
    parser.add_argument('--example-config', action='store_true',
                        help='Show example configuration')
    parser.add_argument('--integration-guide', action='store_true',
                        help='Show integration guide for trainers')
    parser.add_argument('--config', type=str,
                        help='Path to config file to update')
    parser.add_argument('--apply', action='store_true',
                        help='Actually apply config changes (not dry run)')
    
    args = parser.parse_args()
    
    if args.test:
        test_dists_loss()
    elif args.example_config:
        print(get_dists_config_example())
    elif args.integration_guide:
        show_integration_guide()
    elif args.config:
        update_config_file(args.config, dry_run=not args.apply)
    else:
        # Show all information
        print("=" * 60)
        print("DISTS Perceptual Loss Integration")
        print("=" * 60)
        print("\n1. Testing DISTS loss:")
        test_dists_loss()
        print("\n2. Example configuration:")
        print(get_dists_config_example())
        print("\n3. Integration guide:")
        show_integration_guide()
        print("\nUse --help for more options")


if __name__ == '__main__':
    main()
