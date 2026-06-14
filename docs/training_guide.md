# Training Guide

Complete guide for training anime super-resolution models.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Data Preparation](#data-preparation)
3. [Training Modes](#training-modes)
4. [Configuration](#configuration)
5. [Monitoring Training](#monitoring-training)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Prepare Test Data

```bash
# Create dummy test data (10 images)
python scripts/create_test_data.py --output data/test_hr --num-images 10
```

### 2. Quick Test Training

```bash
# Run quick test (3 epochs)
python scripts/train.py --config configs/example_quick_test.yaml
```

### 3. Train on Your Data

```bash
# Copy your images
mkdir -p data/anime_hr
cp /path/to/your/images/*.png data/anime_hr/

# Train Model A
python scripts/train.py --config configs/example_anime_only.yaml
```

---

## Data Preparation

### Supported Formats

- PNG (recommended - lossless)
- JPEG/JPG (acceptable - check quality)
- WEBP (acceptable)
- BMP (supported)

### Image Requirements

- **Minimum resolution**: 256x256 pixels
- **Color**: RGB (3 channels)
- **Content**: High-quality anime frames

### Data Sources

1. **Blu-ray Screenshots** - Best quality
2. **Video I-frames** - Use `scripts/prepare_data.py`
3. **Web Images** - Filter for quality

### Preparing Video Sources

```bash
# Extract I-frames from videos
python scripts/prepare_data.py \
    --input path/to/videos \
    --output data/processed \
    --mode full \
    --quality-threshold 0.7
```

This will:
1. Extract I-frames from all videos
2. Filter low-quality images
3. Auto-crop large images
4. Split into train/val sets

---

## Training Modes

### Model A (NTIRE + MTKD + FAKD)

Best for: General anime, fast inference

```bash
python scripts/train.py --config configs/model_a_ntire.yaml --mode model_a
```

**Features:**
- SPAN architecture (26 channels)
- Parameter-free attention
- ConvLoRA for efficient training
- ~250K parameters

### Model B (Mamba-PAN)

Best for: Complex patterns, texture details

```bash
python scripts/train.py --config configs/model_b_mamba.yaml --mode model_b
```

**Features:**
- Mamba State Space Model
- 4-direction scanning (H, V, RH, RV)
- Linear complexity O(N)
- ~300K-400K parameters

**Note:** Install `mamba-ssm` for best performance:
```bash
pip install mamba-ssm causal-conv1d>=1.1.0  # CUDA 11.8+ required
```

### Both Parallel

Train A and B simultaneously (alternating batches).

```bash
python scripts/train.py --config configs/both_parallel.yaml --mode both_parallel
```

**Requirements:**
- 8GB+ VRAM
- Lower batch size (4 per model)

### Both Sequential

Train A first, then B.

```bash
python scripts/train.py --config configs/both_parallel.yaml --mode both_sequential
```

### Ensemble

Train tiny student from pre-trained A and B.

```bash
# First train both models, then:
python scripts/train.py --config configs/ensemble.yaml --mode ensemble
```

### Full Pipeline

Complete A → B → Ensemble pipeline.

```bash
python scripts/train.py --config configs/full_pipeline.yaml --mode full
```

---

## Stage 1: Advanced Training Features

The Stage 1 training (Knowledge Aggregation) now supports multiple advanced features for improved quality.

### Configuration Presets

Four preset configurations are provided for different use cases:

| Config | Features | Expected PSNR | Training Time |
|--------|----------|---------------|---------------|
| `stage1_fast.yaml` | Tier 1 only | +0.5 dB | 1.0x |
| `stage1_balanced.yaml` | Tiers 1+2 (recommended) | +0.7 dB | 1.1x |
| `stage1_anime.yaml` | Anime-optimized | +0.9 dB | 1.3x |
| `stage1_full.yaml` | All features | +1.0-1.2 dB | 1.5x |

### Aggregation Architectures

Four aggregation architectures are available:

**1. SimpleKnowledgeAggregation (Default)**
- Residual conv blocks, NaN-safe
- Fastest and most stable
- Use for: Quick training, baseline

**2. AdaptiveTeacherAggregation**
- Input-dependent teacher gating
- Learns per-sample teacher importance
- Use for: Better quality, when teachers vary

**3. MultiScaleKnowledgeAggregation**
- Processes at 1x, 0.5x, 0.25x scales
- Better for detail-rich content
- Use for: Complex scenes, high-frequency content

**4. FeatureKnowledgeAggregation**
- Aggregates in feature space (before upsampling)
- Cross-teacher attention
- Use for: Efficiency, large-scale training

### Usage Examples

```bash
# Use specific aggregation
python scripts/train.py --config configs/stage1_balanced.yaml \
    training.stage1.aggregation_type=adaptive

# Enable OHEM (Online Hard Example Mining)
python scripts/train.py --config configs/stage1_full.yaml \
    training.stage1.ohem_enabled=true \
    training.stage1.ohem_ratio=0.7

# Use combined loss with custom weights
python scripts/train.py --config configs/stage1_full.yaml \
    training.stage1.use_combined_loss=true \
    training.stage1.loss_weights.l1=0.4 \
    training.stage1.loss_weights.wavelet=0.3 \
    training.stage1.loss_weights.gradient=0.2 \
    training.stage1.loss_weights.diversity=0.1

# Enable anime-specific losses
python scripts/train.py --config configs/stage1_anime.yaml \
    training.stage1.anime.line_art_preservation=true \
    training.stage1.anime.line_art_weight=1.2
```

### Advanced Monitoring

Enable advanced metrics for insights:

```bash
python scripts/train.py --config configs/stage1_full.yaml \
    training.stage1.log_teacher_disagreement=true \
    training.stage1.log_frequency_loss=true
```

This logs:
- Teacher variance per batch
- Per-frequency-band PSNR (LL, LH, HL, HH)
- Teacher correlation matrix

### Example Scripts

Use the provided scripts for advanced workflows:

```bash
# Advanced training with all features
python scripts/train_stage1_advanced.py \
    --config configs/stage1_full.yaml \
    --aggregation adaptive \
    --ohem \
    --combined-loss \
    --monitoring

# Benchmark different aggregations
python scripts/compare_aggregations.py \
    --batch-size 4 \
    --image-size 256 \
    --iterations 100

# Tune anime-specific parameters
python scripts/tune_anime.py \
    --line-art-weight 1.2 \
    --color-weight 0.8 \
    --test-losses
```

---

## Teacher Models for Stage 1 (MTKD)

Stage 1 training requires pretrained teacher models for Multi-Teacher Knowledge Distillation (MTKD). The system supports EDSR, RCAN, and SwinIR architectures.

### Downloading Teacher Models

```bash
# Download all standard teachers (EDSR, RCAN, SwinIR)
python -m src.utils.pretrained_models --model all --scale 4

# Download specific model
python -m src.utils.pretrained_models --model edsr --scale 4
```

### Expected Directory Structure

```
pretrained/
├── EDSR_x4.pt          # EDSR (16 blocks, 64 channels)
├── RCAN_x4.pt          # RCAN (10 groups, 20 RCABs)
└── SwinIR_x4.pt        # SwinIR (6 RSTB blocks)
```

### Disabling Stage 1

If you don't have teacher models, disable Stage 1 and train only Stage 2:

```yaml
training:
  stage1:
    enabled: false
  stage2:
    enabled: true
    epochs: 200
```

Stage 2 (student distillation) works without Stage 1 but may have reduced quality.

### Supported Architectures

| Model | Parameters | Speed | Quality |
|-------|------------|-------|---------|
| EDSR | 1.5M | Fast | Good |
| RCAN | 15M | Medium | Better |
| SwinIR | 12M | Slow | Best |

---

## Configuration

### Base Config Structure

```yaml
model:
  name: "model_name"
  type: "span"  # or "mamba_pan"
  scale: 4
  channels: 26
  num_blocks: 12

training:
  mode: "model_a"
  epochs: 100
  batch_size: 8
  lr: 0.0001
  
data:
  datasets:
    - name: "anime_train"
      weight: 1.0
      enabled: true
      hr_dir: "data/anime_hr"
  
  crop_size: 128
  degradation:
    enabled: true

loss:
  pixel_loss:
    weight: 1.0
  distillation:
    enabled: false
```

### CLI Overrides

Override any config value via CLI:

```bash
# Change batch size
python scripts/train.py --config configs/base.yaml --batch-size 4

# Change learning rate
python scripts/train.py --config configs/base.yaml --lr 0.0002

# Change data path
python scripts/train.py --config configs/base.yaml --data.datasets.0.hr_dir "path/to/data"
```

### Degradation Settings

For anime content, adjust degradation:

```yaml
degradation:
  enabled: true
  blur_kernel_size: [5, 7, 9]  # Smaller for sharp anime lines
  blur_sigma: [0.5, 1.5]       # Less blur
  noise_sigma: [5, 15]          # Less noise
  jpeg_quality: [60, 90]        # More compression (anime artifacts)
```

---

## Monitoring Training

### TensorBoard

```bash
# Enable TensorBoard logging
python scripts/train.py --config configs/base.yaml --tensorboard

# View logs
tensorboard --logdir logs/
```

### Metrics to Watch

| Metric | Target | Notes |
|--------|--------|-------|
| PSNR | > 30 dB | Higher is better |
| L1 Loss | < 0.01 | Pixel accuracy |
| FAKD Loss | < 1.0 | Feature similarity |

### Checkpointing

Checkpoints saved automatically:
- `best.pth` - Best model by validation loss
- `latest.pth` - Most recent checkpoint
- `epoch_N.pth` - Every N epochs

### Resume Training

```bash
python scripts/train.py --config configs/base.yaml --resume checkpoints/latest.pth
```

---

## Troubleshooting

### Out of Memory

**Symptoms:** CUDA out of memory error

**Solutions:**
1. Reduce batch size: `--batch-size 4`
2. Enable gradient checkpointing: Add `gradient_checkpointing: true` to config
3. Use smaller model: Change `channels: 26` to `channels: 20`
4. Reduce crop size: `crop_size: 96`

### Training Too Slow

**Solutions:**
1. Increase num_workers: `--num-workers 4`
2. Enable mixed precision: `mixed_precision: true` (default)
3. Use smaller validation interval: `val_interval: 20`

### Poor Quality Results

**Solutions:**
1. Train longer (increase epochs)
2. Use Charbonnier loss: `type: "charbonnier"`
3. Enable perceptual loss: Add VGG loss
4. Check data quality - use higher resolution images

### NaN Loss

**Causes:** Learning rate too high, numerical instability

**Solutions:**
1. Lower learning rate: `--lr 0.00005`
2. Check for corrupted images
3. Enable gradient clipping (add to trainer)

---

## Advanced Topics

### Multi-GPU Training

```python
# Add to training script
model = nn.DataParallel(model)
```

### Custom Loss Functions

See `src/losses/` for examples. Key losses:
- `pixel_loss.py` - L1, L2, Charbonnier
- `wavelet_loss.py` - MTKD wavelet distillation
- `perceptual_loss.py` - VGG/ResNet features
- `adversarial_loss.py` - GAN-based

### Feature Extraction

For FAKD, extract features at specific layers:

```python
from models.span import SPANModel

model = SPANModel()
features = model.get_feature_maps(input_tensor, layer_indices=[2, 4, 6, 8])
```

---

## Best Practices

1. **Always validate first**: Use `--dry-run` to check config
2. **Start small**: Test with 10 images before full training
3. **Monitor VRAM**: Use `nvidia-smi` or system monitoring
4. **Save frequently**: Lower `save_interval` for important runs
5. **Use EMA**: Enable `use_ema: true` for better convergence
6. **Log everything**: Enable TensorBoard for analysis
