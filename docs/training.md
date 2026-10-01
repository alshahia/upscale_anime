# Training Guide

## Overview

This guide covers training super-resolution models using the anime_sr framework, including knowledge distillation from teacher models.

---

## Training Pipeline

The training pipeline consists of the following stages:

1. **Data Preparation**: Prepare HR/LR image pairs
2. **Teacher Training** (optional): Train or load a teacher model
3. **Student Training**: Train a student model with knowledge distillation
4. **Validation**: Evaluate model quality
5. **Export**: Export to ONNX/TorchScript for deployment

---

## Data Preparation

### Directory Structure

```
data/
├── hr_images/          # High-resolution images (1920x1080 recommended)
│   ├── frame_001.png
│   ├── frame_002.png
│   └── ...
├── lr_images/          # Low-resolution images (optional, auto-generated if missing)
│   ├── frame_001.png
│   ├── frame_002.png
│   └── ...
└── test/               # Test set for validation
    ├── hr/
    └── lr/
```

### Data Configuration

```yaml
# configs/data.yaml
data:
  hr_dir: data/hr_images
  lr_dir: data/lr_images  # Optional
  scale: 4
  crop_size: 192
  augment: true
  degradation: anime_heavy  # bicubic, light, medium, heavy, anime, anime_heavy, apisr
  num_workers: 4
  batch_size: 16
```

### Degradation Presets

| Preset | Description | Use Case |
|--------|-------------|----------|
| `bicubic` | Pure bicubic downsample | Baseline |
| `light` | Mild blur + noise + JPEG | Light degradation |
| `medium` | Moderate blur + noise + JPEG | Medium degradation |
| `heavy` | Strong blur + noise + JPEG | Heavy degradation |
| `anime` | Anime-tuned (banding, color quantization) | Anime content |
| `anime_heavy` | APISR-style anime heavy | Default for anime |
| `apisr` | v7 APISR-aligned preset | Production |

---

## Training Modes

### Model A (Standard SR)

Train a standard super-resolution model without distillation:

```bash
python -m anime_sr train \
    --config configs/train_model_a.yaml \
    --model span
```

### Model B (Distillation)

Train a student model with knowledge distillation:

```bash
python -m anime_sr train \
    --config configs/train_model_b.yaml \
    --model rfdn
```

### Both Sequential

Train Model A first, then use it as teacher for Model B:

```bash
python -m anime_sr train \
    --config configs/train_both_sequential.yaml
```

### Full Pipeline

Train the complete pipeline: Model A → Model B → Ensemble:

```bash
python -m anime_sr train \
    --config configs/train_full.yaml
```

### Auto Stage

Automatically determine the training stage from checkpoint:

```bash
python -m anime_sr train \
    --config configs/train_auto.yaml \
    --resume runs/exp1/checkpoint.pt
```

---

## Knowledge Distillation

### MTKD (Multi-Teacher Knowledge Distillation)

Wavelet-based distillation loss:

```python
from anime_sr.distillation.mtkd import WaveletDistillationLoss, FullMTKDLoss

# Wavelet distillation loss
criterion = WaveletDistillationLoss(
    pixel_weight=0.3,
    wavelet_weight=0.5,
    wavelet_levels=3
)

# Full MTKD loss
criterion = FullMTKDLoss(
    l1_weight=0.5,
    wavelet_weight=0.2,
    wavelet_levels=3
)
```

### FAKD (Feature Affinity Knowledge Distillation)

Second-order statistics-based feature matching:

```python
from anime_sr.distillation.fakd import FeatureAffinityLoss

criterion = FeatureAffinityLoss(
    layers=[0, 1, 2, 3],
    layer_weights=[1.0, 0.5, 0.25, 0.125],
    affinity_type="covariance",  # or "gram", "correlation"
    temperature=1.0
)
```

### Distillation Training Script

```bash
python -m anime_sr.training.distillation.distill \
    --data data/anime_video_frames \
    --out-dir runs/distill_v1 \
    --teacher span \
    --arch rfdn \
    --epochs 40 \
    --batch-size 16 \
    --lr 5e-5
```

---

## Loss Functions

### Pixel Losses

```python
from anime_sr.losses import L1Loss, L2Loss, CharbonnierLoss, LossFactory

# Direct instantiation
criterion = CharbonnierLoss(eps=1e-3)

# Factory method
criterion = LossFactory.create("charbonnier", eps=1e-3)
criterion = LossFactory.create("l1")
criterion = LossFactory.create("l2")
```

### Perceptual Losses

```python
from anime_sr.losses import VGGPerceptualLoss, PerceptualLossFactory

# VGG perceptual loss
criterion = VGGPerceptualLoss(
    layers=[3, 8, 15, 22],
    weights=[1.0, 0.5, 0.25, 0.125]
)

# Factory method
criterion = PerceptualLossFactory.create("vgg", layers=[3, 8, 15, 22])
criterion = PerceptualLossFactory.create("resnet", layers=[1, 2, 3, 4])
criterion = PerceptualLossFactory.create("dists")
```

### Adversarial Losses

```python
from anime_sr.losses import create_adversarial_loss, SRDiscriminator

# Create adversarial loss
criterion = create_adversarial_loss("hinge")  # or "vanilla", "relativistic"

# Create discriminator
discriminator = SRDiscriminator(in_channels=3, num_channels=64)
```

### Combined Losses

```python
from anime_sr.losses import Stage1CombinedLoss, AnimeCombinedLoss

# Stage 1 combined loss
criterion = Stage1CombinedLoss(
    l1_weight=1.0,
    wavelet_weight=0.2,
    gradient_weight=0.1,
    diversity_weight=0.05
)

# Anime-specific combined loss
criterion = AnimeCombinedLoss(
    pixel_weight=1.0,
    perceptual_weight=0.5,
    adversarial_weight=0.01,
    line_weight=0.1,
    color_weight=0.05
)
```

---

## Training Configuration

### Basic Config

```yaml
# configs/train_basic.yaml
model:
  type: rfdn
  scale: 4

training:
  epochs: 100
  batch_size: 16
  learning_rate: 1.0e-4
  weight_decay: 0.0
  optimizer: adam
  scheduler: cosine
  mode: model_b

data:
  hr_dir: data/hr_images
  scale: 4
  crop_size: 192
  augment: true
  degradation: anime_heavy
  num_workers: 4

loss:
  type: combined
  pixel_weight: 1.0
  perceptual_weight: 0.5
  adversarial_weight: 0.01
  wavelet_weight: 0.2

distillation:
  teacher: span
  teacher_checkpoint: checkpoints/span_teacher.pth
  feature_weight: 0.5
  response_weight: 0.3
```

### Advanced Config

```yaml
# configs/train_advanced.yaml
base: train_basic.yaml

training:
  epochs: 200
  batch_size: 8
  gradient_accumulation_steps: 4
  mixed_precision: true
  ema_decay: 0.999
  swa_start: 150

loss:
  type: adaptive
  initial_weights:
    l1: 1.0
    perceptual: 0.5
    adversarial: 0.01
  adapt_every: 10

distillation:
  teachers:
    - name: span
      checkpoint: checkpoints/span_teacher.pth
      weight: 0.5
    - name: edsr
      checkpoint: checkpoints/edsr_teacher.pth
      weight: 0.3
    - name: rcan
      checkpoint: checkpoints/rcan_teacher.pth
      weight: 0.2
```

---

## Training Loop

### Basic Training Loop

```python
import torch
from anime_sr.models.students import RFDN
from anime_sr.training import BaseTrainer

# Create model
model = RFDN(scale=4)

# Create trainer
trainer = BaseTrainer(config=config, model=model)

# Training loop
for epoch in range(num_epochs):
    # Train
    train_metrics = trainer.train_epoch(train_loader, epoch)

    # Validate
    val_metrics = trainer.validate(val_loader)

    # Save checkpoint
    trainer.save_checkpoint(epoch, val_metrics, f"checkpoints/epoch_{epoch}.pt")

    print(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, val_psnr={val_metrics['psnr']:.2f}")
```

### Training with EMA

```python
# Enable EMA in config
config.training.ema_decay = 0.999

# Trainer automatically maintains EMA weights
trainer = BaseTrainer(config=config, model=model)

# Save EMA weights
trainer.save_checkpoint(epoch, val_metrics, ema=True)
```

### Training with SWA

```python
# Enable SWA in config
config.training.swa_start = 150
config.training.swa_lr = 1e-4

# Trainer automatically maintains SWA weights
trainer = BaseTrainer(config=config, model=model)

# Save SWA weights
trainer.save_checkpoint(epoch, val_metrics, swa=True)
```

---

## Checkpointing

### Save Checkpoint

```python
trainer.save_checkpoint(
    epoch=epoch,
    metrics=val_metrics,
    path=f"checkpoints/epoch_{epoch}.pt",
    ema=False,
    swa=False
)
```

### Load Checkpoint

```python
# Load weights only
trainer.load_checkpoint("checkpoints/best.pt")

# Load with optimizer state
trainer.load_checkpoint(
    "checkpoints/best.pt",
    optimizer=optimizer,
    scheduler=scheduler
)
```

### Resume Training

```bash
python -m anime_sr train \
    --config configs/train.yaml \
    --resume runs/exp1/checkpoint_epoch_50.pt
```

---

## Validation

### Metrics

```python
from anime_sr.utils import calculate_psnr, calculate_ssim, MetricsTracker

# Calculate PSNR
psnr = calculate_psnr(sr_image, hr_image, data_range=1.0)

# Calculate SSIM
ssim = calculate_ssim(sr_image, hr_image, data_range=1.0)

# Track metrics
tracker = MetricsTracker()
tracker.update("psnr", psnr)
tracker.update("ssim", ssim)
summary = tracker.summary()
```

### Validation Loop

```python
# Validate model
val_metrics = trainer.validate(val_loader)

print(f"PSNR: {val_metrics['psnr']:.2f} dB")
print(f"SSIM: {val_metrics['ssim']:.4f}")
print(f"LPIPS: {val_metrics['lpips']:.4f}")
```

---

## Callbacks

### Early Stopping

```python
from anime_sr.training.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    patience=10,
    min_delta=0.01,
    monitor="val_psnr"
)

for epoch in range(num_epochs):
    train_metrics = trainer.train_epoch(train_loader, epoch)
    val_metrics = trainer.validate(val_loader)

    if early_stopping(val_metrics):
        print(f"Early stopping at epoch {epoch}")
        break
```

### Learning Rate Scheduler

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# Step LR
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
```

### Model Checkpoint

```python
from anime_sr.training.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    dirpath="checkpoints",
    filename="model_{epoch}_{val_psnr:.2f}",
    monitor="val_psnr",
    mode="max",
    save_top_k=3
)
```

---

## Hyperparameter Tuning

### Grid Search

```python
import itertools

learning_rates = [1e-4, 5e-5, 1e-5]
batch_sizes = [8, 16, 32]

for lr, bs in itertools.product(learning_rates, batch_sizes):
    config.training.learning_rate = lr
    config.training.batch_size = bs

    trainer = BaseTrainer(config=config, model=model)
    # Train and evaluate
```

### Optuna Integration

```python
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    config.training.learning_rate = lr
    config.training.batch_size = batch_size

    trainer = BaseTrainer(config=config, model=model)
    # Train and evaluate
    return val_metrics["psnr"]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
```

---

## Distributed Training

### DDP (Distributed Data Parallel)

```bash
# Single node, multiple GPUs
torchrun --nproc_per_node=4 -m anime_sr train --config configs/train.yaml

# Multiple nodes
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --node_rank=0 \
    --master_addr="192.168.1.100" \
    --master_port=1234 \
    -m anime_sr train --config configs/train.yaml
```

### DDP Configuration

```python
import torch.distributed as dist

# Initialize process group
dist.init_process_group(backend="nccl")

# Create DDP model
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank]
)
```

---

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
export DSH_TRAINING__BATCH_SIZE=8

# Use gradient accumulation
export DSH_TRAINING__GRADIENT_ACCUMULATION_STEPS=4

# Use mixed precision
export DSH_TRAINING__MIXED_PRECISION=true
```

### NaN Losses

```python
# Enable gradient clipping
config.training.gradient_clip = 1.0

# Use NaN handler
from anime_sr.utils import NanHandler
nan_handler = NanHandler()
nan_handler.check(loss)
```

### Slow Training

```bash
# Increase num_workers
export DSH_DATA__NUM_WORKERS=8

# Use precomputed LR images
export DSH_DATA__PRELOAD=true

# Profile training
python scripts/profile_gpu.py --config configs/train.yaml
```

---

## See Also

- [Python API Reference](api.md) - Python API documentation
- [CLI Reference](cli.md) - Command-line interface
- [Model Architectures](models.md) - Model architecture details
- [Development Setup](development.md) - Development environment
- [Deployment Guide](deployment.md) - Production deployment
