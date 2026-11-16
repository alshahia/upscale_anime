# anime_sr CLI Reference

## Installation

### From Source

```bash
git clone https://github.com/your-org/anime_sr.git
cd anime_sr
pip install -e .
```

### Verify Installation

```bash
anime-sr --help
# or
python -m anime_sr --help
```

---

## Global Options

These options are available for all subcommands:

```bash
anime-sr [--help] [--version] <subcommand> [options]
```

| Option | Description |
|--------|-------------|
| `--help` | Show help message and exit |
| `--version` | Show version and exit |

---

## Commands

### `train`

Train a super-resolution model.

```bash
anime-sr train --config configs/default.yaml [options]
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config` | str | `configs/default.yaml` | Path to configuration file |
| `--model` | str | from config | Model architecture (`span`, `mamba_pan`, `rfdn`, `srvgg`) |
| `--resume` | str | None | Path to checkpoint to resume from |
| `--output` | str | from config | Output directory for checkpoints |

#### Examples

```bash
# Train with default config
anime-sr train

# Train specific model
anime-sr train --model rfdn --config configs/train_rfdn.yaml

# Resume training
anime-sr train --resume runs/exp1/checkpoint_epoch_10.pt

# Custom output directory
anime-sr train --output runs/my_experiment
```

#### Environment Variables

```bash
export DSH_TRAINING__EPOCHS=100
export DSH_TRAINING__BATCH_SIZE=16
export DSH_TRAINING__LEARNING_RATE=1e-4
export DSH_CUDA_VISIBLE_DEVICES=0,1
```

---

### `infer`

Upscale images using a trained model.

```bash
anime-sr infer --input <path> --checkpoint <path> [options]
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input` | str | required | Input image or directory |
| `--output` | str | `output/upscaled` | Output directory |
| `--checkpoint` | str | required | Path to model checkpoint |
| `--model` | str | auto | Model type (`rfdn`, `srvgg`, `mambair`, `auto`) |
| `--scale` | int | 4 | Upscaling factor (2, 3, 4) |
| `--tile` | int | 0 | Tile size for tiled inference (0 = disabled) |
| `--tta` | flag | False | Enable test-time augmentation |
| `--device` | str | auto | Device (`cuda`, `cpu`, `auto`) |

#### Examples

```bash
# Single image
anime-sr infer --input image.png --checkpoint runs/distill_v1/student_best.pt

# Batch processing
anime-sr infer --input images/ --checkpoint runs/distill_v1/student_best.pt --output output/

# With TTA (8x slower, better quality)
anime-sr infer --input image.png --checkpoint runs/distill_v1/student_best.pt --tta

# Tiled inference for large images
anime-sr infer --input large.png --checkpoint runs/distill_v1/student_best.pt --tile 256

# Specific model type
anime-sr infer --input image.png --checkpoint runs/distill_v1/student_best.pt --model rfdn
```

---

### `export`

Export a trained model to ONNX or TorchScript.

```bash
anime-sr export --checkpoint <path> --format onnx [options]
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--checkpoint` | str | required | Path to model checkpoint |
| `--format` | str | `onnx` | Export format (`onnx`, `torchscript`) |
| `--output` | str | `exports/student.onnx` | Output file path |
| `--model` | str | `span` | Model type (`span`, `mamba_pan`, `ensemble`) |

#### Examples

```bash
# Export to ONNX
anime-sr export --checkpoint runs/distill_v1/student_best.pt --format onnx

# Explicit output path and architecture
anime-sr export --checkpoint runs/distill_v1/student_best.pt --format onnx --output exports/student.onnx --model span

# Export to TorchScript
anime-sr export --checkpoint runs/distill_v1/student_best.pt --format torchscript

# Quiet export
anime-sr export --checkpoint runs/distill_v1/student_best.pt --format onnx --output exports/student.onnx --quiet
```

---

### `serve`

Start the inference API server.

```bash
anime-sr serve --config configs/api.yaml [options]
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config` | str | `configs/api.yaml` | Path to API configuration |
| `--host` | str | `127.0.0.1` | Server host (pass `0.0.0.0` to expose the API) |
| `--port` | int | 8000 | Server port |
| `--reload` | flag | False | Enable auto-reload (development) |
| `--workers` | int | 1 | Number of worker processes |

#### Examples

```bash
# Start server
anime-sr serve

# Custom host and port
anime-sr serve --host 127.0.0.1 --port 8080

# Development mode with auto-reload
anime-sr serve --reload

# Production with multiple workers
anime-sr serve --workers 4
```

#### Environment Variables

```bash
export DSH_API_KEY=your-secret-key
export DSH_RATE_LIMIT_PER_MIN=60
export DSH_MAX_FILE_SIZE_MB=10
```

---

### Graphical interface

The desktop GUI is a standalone Tkinter application, not a subcommand of
`anime-sr`:

```bash
# From the repository root
python -m apps.anime_upscaler_gui

# Windows: double-click the launcher
apps\anime_upscaler_gui\LAUNCHER.bat
```

See `apps/anime_upscaler_gui/README.md` for the full walkthrough.

---

### `validate`

Check that a checkpoint loads cleanly (architecture match and parameter count
are reported). Dataset-based metric evaluation (PSNR/SSIM) is not implemented
yet.

```bash
anime-sr validate --checkpoint <path> [options]
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--checkpoint` | str | required | Path to model checkpoint |
| `--model` | str | auto | Model type (`span`, `neosr_span`, `mamba_pan`, `tiny_student`) |

#### Examples

```bash
# Validate that a checkpoint loads
anime-sr validate --checkpoint runs/distill_v1/student_best.pt

# Validate as a specific architecture
anime-sr validate --checkpoint ./checkpoints/animesr.pth --model neosr_span
```

---

### `benchmark`

Benchmark model inference performance.

```bash
anime-sr benchmark --checkpoint <path> [options]
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--checkpoint` | str | required | Path to model checkpoint |
| `--model` | str | auto | Model type (`rfdn`, `srvgg`, `mambair`, `auto`) |
| `--input` | str | None | Input image for benchmarking |
| `--iterations` | int | 100 | Number of iterations |

#### Examples

```bash
# Basic benchmark
anime-sr benchmark --checkpoint runs/distill_v1/student_best.pt

# Benchmark with specific input
anime-sr benchmark --checkpoint runs/distill_v1/student_best.pt --input test.png

# Custom iterations
anime-sr benchmark --checkpoint runs/distill_v1/student_best.pt --iterations 200
```

---

## Configuration Reference

### Configuration File Structure

```yaml
# configs/default.yaml

# Model configuration
model:
  type: rfdn              # span, mamba_pan, rfdn, srvgg
  scale: 4
  in_channels: 3
  out_channels: 3

# Training configuration
training:
  epochs: 100
  batch_size: 16
  learning_rate: 1.0e-4
  weight_decay: 0.0
  optimizer: adam          # adam, adamw, sgd
  scheduler: cosine        # cosine, step, plateau
  mode: both_sequential    # model_a, model_b, both_parallel, both_sequential, ensemble, full, auto_stage

# Data configuration
data:
  hr_dir: data/hr_images
  lr_dir: data/lr_images    # Optional
  scale: 4
  crop_size: 192
  augment: true
  degradation: anime_heavy  # bicubic, light, medium, heavy, anime, anime_heavy, apisr
  num_workers: 4

# Loss configuration
loss:
  type: combined
  pixel_weight: 1.0
  perceptual_weight: 0.5
  adversarial_weight: 0.01
  wavelet_weight: 0.2

# Distillation configuration
distillation:
  teacher: span            # span, edsr, rcan, swinir, realesr
  teacher_checkpoint: checkpoints/span_teacher.pth
  feature_weight: 0.5
  response_weight: 0.3

# Inference configuration
inference:
  tile_size: 0
  tta: false
  device: auto

# Export configuration
export:
  format: onnx
  opset: 17
  half: false

# API configuration
api:
  host: 0.0.0.0
  port: 8000
  api_key: null
  rate_limit: 60
  max_file_size_mb: 10
```

### Configuration Inheritance

Configs can inherit from base configs using the `base:` key:

```yaml
# configs/train_rfdn.yaml
base: default.yaml

model:
  type: rfdn

training:
  epochs: 50
```

### Environment Variable Overrides

Any config value can be overridden via environment variables using the pattern `DSH_<SECTION>__<KEY>`:

```bash
# Override training epochs
export DSH_TRAINING__EPOCHS=200

# Override model type
export DSH_MODEL__TYPE=span

# Override data directory
export DSH_DATA__HR_DIR=/path/to/images

# Override API key
export DSH_API_KEY=secret-key
```

---

## Common Workflows

### Training a Student Model

```bash
# 1. Prepare data
# Place HR images in data/hr_images/

# 2. Train teacher (if not using pretrained)
anime-sr train --model span --config configs/train_teacher.yaml

# 3. Train student with distillation
anime-sr train --model rfdn --config configs/train_student.yaml

# 4. Validate
anime-sr validate --checkpoint runs/student/student_best.pt --data data/test/

# 5. Export
anime-sr export --checkpoint runs/student/student_best.pt --format onnx
```

### Inference Pipeline

```bash
# 1. Upscale single image
anime-sr infer --input image.png --checkpoint runs/student/student_best.pt

# 2. Batch upscale
anime-sr infer --input images/ --checkpoint runs/student/student_best.pt --output output/

# 3. Upscale with TTA
anime-sr infer --input image.png --checkpoint runs/student/student_best.pt --tta

# 4. Upscale large image with tiling
anime-sr infer --input large.png --checkpoint runs/student/student_best.pt --tile 256
```

### Deployment

```bash
# 1. Export model
anime-sr export --checkpoint runs/student/student_best.pt --format onnx --output exports/student.onnx

# 2. Start API server
anime-sr serve --config configs/api.yaml --port 8000

# 3. Test API
curl -X POST http://localhost:8000/inference \
  -H "Authorization: Bearer $DSH_API_KEY" \
  -F "file=@test.png"
```

---

## Troubleshooting

### Common Issues

**CUDA out of memory**
```bash
# Reduce batch size
export DSH_TRAINING__BATCH_SIZE=8

# Use tiled inference
anime-sr infer --input large.png --checkpoint model.pt --tile 128
```

**Model not found**
```bash
# Check checkpoint path
ls -la runs/distill_v1/student_best.pt

# Use auto-detect
anime-sr infer --input image.png --checkpoint model.pt --model auto
```

**Slow inference**
```bash
# Export to ONNX
anime-sr export --checkpoint model.pt --format onnx --output exports/student.onnx

# NOTE: the CLI has no --onnx inference flag. Run the exported model with
# onnxruntime directly, or use the desktop GUI (apps/anime_upscaler_gui),
# which has an ONNX backend.

# Disable TTA
anime-sr infer --input image.png --checkpoint model.pt  # without --tta
```

---

## See Also

- [Python API Reference](api.md) - Python API documentation
- [Model Architectures](models.md) - Model architecture details
- [Training Guide](training.md) - Training procedures
- [Development Setup](development.md) - Development environment
- [Deployment Guide](deployment.md) - Production deployment
