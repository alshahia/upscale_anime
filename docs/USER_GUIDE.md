# Anime Super-Resolution - User Guide

## What This Does

Upscales anime images and videos using AI super-resolution models. Two model options:
- **Model A**: Fast, lightweight (SPAN-Tiny) - best for most users
- **Model B**: Higher quality (Mamba-PAN) - requires more VRAM

## Quick Start

### 1. Install

```bash
# Python 3.10+ required
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### 2. Prepare Data

Place your high-resolution anime images in `data/`:
```
data/
  anime_image_1.png
  anime_image_2.png
  ...
```

For videos, place in `data/anime_vid/` (frames extracted automatically).

### 3. Train a Model

```bash
# Model A (recommended)
python scripts/train.py --config configs/model_a_ntire.yaml

# Model B (requires mamba-ssm)
python scripts/train.py --config configs/model_b.yaml

# Dry run (validate config without training)
python scripts/train.py --config configs/model_a_ntire.yaml --dry-run
```

### 4. Upscale Images

```bash
python scripts/inference.py \
  --checkpoint checkpoints/model_a_best.pth \
  --input data/ \
  --output results/ \
  --scale 4
```

## Configuration

### Key Config Options

All configs are in `configs/`. Key settings:

| Setting | Description | Default |
|---------|-------------|---------|
| `model.scale` | Upscale factor (2, 3, 4) | 4 |
| `training.batch_size` | Images per batch | 16 |
| `training.epochs` | Training epochs | 100 |
| `data.crop_size` | Training patch size | 128 |
| `data.degradation.mode` | LR generation mode | `real_esrgan` |

### Override Settings

```bash
# Change batch size and epochs
python scripts/train.py --config configs/model_a_ntire.yaml \
  --batch_size 8 --epochs 50

# Override any config value
python scripts/train.py --config configs/model_a_ntire.yaml \
  --config_overrides data.batch_size=8 training.epochs=50
```

### Degradation Presets

| Preset | Use Case |
|--------|----------|
| `light` | Clean source, minimal degradation |
| `medium` | Standard anime screenshots |
| `heavy` | Low-quality, compressed sources |
| `anime` | Optimized for anime-style content |

## Training Modes

| Mode | Description |
|------|-------------|
| `model_a` | Train Model A only |
| `model_b` | Train Model B only |
| `both_sequential` | Train A then B |
| `ensemble` | Combine both models |
| `full` | Full pipeline with all stages |

## Hardware Requirements

| Model | Minimum VRAM | Recommended |
|-------|--------------|-------------|
| Model A | 4GB | 8GB (RTX 4000 Mobile) |
| Model B | 8GB | 12GB+ |

## Troubleshooting

### Common Issues

**"Config file not found"**
- Check path: `python scripts/train.py --config configs/model_a_ntire.yaml`
- Config must exist in `configs/` directory

**"No images found in data/"**
- Create `data/` directory and add images
- Supported formats: `.png`, `.jpg`, `.jpeg`
- Create test data: `python -c "from data.data_utils import create_dummy_dataset; create_dummy_dataset('data', num_images=10)"`

**NaN losses during training**
- Normal in early epochs with multiple teachers
- Use `use_simple_aggregation: true` in config
- Reduce batch size if persistent

**Out of memory**
- Reduce `batch_size` in config
- Reduce `crop_size` (e.g., 128 -> 96)
- Enable gradient accumulation: `training.gradient_accumulation_steps: 2`

**Windows encoding errors**
- Fixed in latest version - all output is ASCII-compatible
- If using older version, set `PYTHONIOENCODING=utf-8`

### Resume Training

Training auto-saves checkpoints. To resume:
```bash
python scripts/train.py --config configs/model_a_ntire.yaml --resume
```

### Download Pre-trained Teachers

```bash
python scripts/download_models.py --model all --scale 4
```

## Fine-tuning neosr SPAN

### Pretrained Weights

Three SPAN pretrained models are available in `pretrained/`:

| File | Description | Size |
|------|-------------|------|
| `span_nomosuni_4x.pth` | Universal (anime+realistic), JPEG degradation | 4.3 MB |
| `span_pix_pretrain_4x.pth` | L1 pixel loss pretrain | 8.6 MB |
| `span_mssim_pretrain_4x.pth` | MS-SSIM loss pretrain | 8.6 MB |

### Quick Fine-tuning

```bash
# Edit configs/finetune_neosr_span.yaml to set your data path and epochs
python scripts/train.py --config configs/finetune_neosr_span.yaml
```

### Custom Fine-tuning Config

```yaml
training:
  mode: finetune
  finetune:
    enabled: true
    epochs: 50
    batch_size: 4
    lr: 0.0001
    min_lr: 0.00001
    warmup_epochs: 5
    gradient_accumulation_steps: 2
    pretrained_path: pretrained/span_nomosuni_4x.pth
    freeze_layers: []  # Add layer names to freeze
    loss: l1
    val_interval: 5
    save_interval: 10

model:
  type: neosr_span
  feature_channels: 48
  num_blocks: 6
  upscale: 4
```

### Freeze Layers

To freeze specific layers (e.g., early blocks):

```yaml
training:
  finetune:
    freeze_layers:
      - block_1
      - block_2
      - conv_1
```

## File Structure

```
upscale_anime/
  data/           # Your images/videos (git-ignored)
  checkpoints/    # Saved models (git-ignored)
  results/        # Output images (git-ignored)
  configs/        # Training configurations
  scripts/        # CLI entry points
  src/            # Library code
  tests/          # Test suite
```

## Advanced Usage

### Video Processing

```bash
# Extract frames and process
python scripts/process_videos.py --input data/anime_vid/ --output data/

# Train with video dataset
python scripts/train.py --config configs/auto_stage_pipeline.yaml
```

### Custom Degradation

Create a custom degradation preset in your config:
```yaml
data:
  degradation:
    mode: custom
    blur_prob: 0.7
    noise_prob: 0.5
    jpeg_prob: 0.5
    blur_kernel_size: [7, 9, 11]
    noise_sigma: [0, 25]
    jpeg_quality: [60, 70, 80, 90, 100]
```

### Export Model

```bash
python scripts/export_model.py \
  --checkpoint checkpoints/model_a_best.pth \
  --output exported/model_a.onnx \
  --format onnx
```

## Support

- Issues: https://github.com/anomalyco/opencode/issues
- See `FIXES_APPLIED.md` for all applied fixes and known resolutions
