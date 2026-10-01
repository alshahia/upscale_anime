# Anime Super-Resolution

A PyTorch implementation supporting NTIRE-winning SPAN architecture with MTKD+FAKD distillation, and Mamba-PAN with directional scanning.

## Features

- **Model A (NTIRE)**: SPAN-Tiny with parameter-free attention + ConvLoRA + MTKD + FAKD
- **Model B (Mamba-PAN)**: State Space Model with 4-direction scanning + MTKD + FAKD
- **Progressive Ensemble**: Train both models, ensemble as meta-teacher, distill to ultra-lightweight student
- **Universal Data Module**: Configurable dataset handling with RealESRGAN-style degradation
- **VRAM Optimized**: Runs on 8GB GPUs (RTX 4000 Mobile)
- **Training Control System**: Early stopping, adaptive LR, auto-stage transitions
- **Desktop GUI**: Standalone Tkinter app with job queue, VRAM guard, and themes
- **Small Dataset Techniques**: 10-phase approach for training with limited data (5000+ images)
  - Self-supervised pre-training, transfer learning, meta-learning
  - Advanced augmentation (Mixup, CutMix), progressive crop sizing
  - Feature distillation, test-time adaptation
  - Benchmarking and production export

## Installation

### Prerequisites

- Python 3.12+
- PyTorch 2.4+ with CUDA support
- 8GB+ VRAM recommended

### Install from source

```bash
# Clone the repository
git clone <repository-url>
cd upscale_anime

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode
pip install -e .

# Install with optional dependencies
pip install -e ".[dev]"      # Development tools
pip install -e ".[api]"      # API server dependencies
pip install -e ".[gui]"      # GUI dependencies
pip install -e ".[mamba]"    # Mamba model support
```

## Quick Start

### CLI

```bash
# Train a model
anime-sr train --config configs/train_config.yaml --tensorboard

# Run inference
anime-sr infer --input ./input_images --output ./output --model ./checkpoints/model_best.pt --scale 4

# Export to ONNX
anime-sr export --model ./checkpoints/model_best.pt --output ./exported/model.onnx --format onnx

# Start API server
anime-sr serve --model ./checkpoints/model_best.pt --host 0.0.0.0 --port 8000

# Launch the desktop GUI (standalone app — see "Desktop GUI" below)
python -m apps.anime_upscaler_gui

# Validate model
anime-sr validate --model ./checkpoints/model_best.pt --data ./validation_data

# Benchmark performance
anime-sr benchmark --model ./checkpoints/model_best.pt --data ./test_data
```

### Python API

```python
from anime_sr.inference.engine import InferenceEngine

# Initialize engine
engine = InferenceEngine(
    model_path="./checkpoints/model_best.pt",
    scale=4,
    tile_size=512,
)

# Upscale a single image
engine.infer_image("input.png", "output.png")

# Upscale a directory
engine.infer_directory("./input_images", "./output")
```

### API Server

```bash
# Start the server
anime-sr serve --model ./checkpoints/model_best.pt --port 8000

# Or using Python
from anime_sr.api.inference_api import create_app
import uvicorn

app = create_app(model_path="./checkpoints/model_best.pt")
uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Desktop GUI

A standalone Tkinter app (`apps/anime_upscaler_gui`) for upscaling images and videos without writing code:

```bash
# From the repository root
python -m apps.anime_upscaler_gui

# Or double-click apps/anime_upscaler_gui/LAUNCHER.bat (Windows)
```

- **Model catalog** — auto-detects checkpoints in `pretrained/`, with presets for SRVGG, SPAN, ERANet, AnimeSR, and the in-house distilled students
- **Queue controls** — batch multiple files with pause/resume/cancel, live preview, log panel, and GPU monitor
- **VRAM guard** — warn, auto-downscale, or tile processing to fit 8GB GPUs
- **Themes & accessibility** — light / dark / high-contrast themes, keyboard shortcuts, English/Arabic UI
- **Settings** — persisted per-user outside the source tree

See the [GUI README](apps/anime_upscaler_gui/README.md) for the full walkthrough, supported-models table, and keyboard shortcuts.

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `DSH_API_KEY` | - | API authentication key |
| `DSH_MODEL_PATH` | `./checkpoints` | Path to model checkpoints |
| `DSH_OUTPUT_PATH` | `./output` | Path for output files |
| `DSH_LOG_LEVEL` | `INFO` | Logging level |
| `DSH_CUDA_VISIBLE_DEVICES` | `0` | GPU device IDs |
| `DSH_ALLOW_PICKLE_CHECKPOINT` | `false` | Allow pickle checkpoints (security) |
| `DSH_MAX_FILE_SIZE_MB` | `50` | Max upload size in MB |
| `DSH_RATE_LIMIT_PER_MIN` | `60` | API rate limit per minute |

### Training Config

Training is configured via YAML files. See `configs/` for examples.

## Development

### Setup

```bash
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=anime_sr --cov-report=html

# Run specific test categories
pytest -m "not slow"   # Skip slow tests
pytest -m integration  # Run only integration tests
```

### Code Quality

```bash
# Format code
black src/

# Lint
flake8 src/

# Type check
mypy src/
```

## Project Structure

```
upscale_anime/
├── src/
│   └── anime_sr/          # Main package
│       ├── api/           # REST API server
│       ├── data/          # Data loading and augmentation
│       ├── distillation/  # MTKD and FAKD distillation
│       ├── export/        # Model export (ONNX, etc.)
│       ├── inference/     # Inference engine
│       ├── losses/        # Loss functions
│       ├── models/        # Model architectures (SPAN, Mamba-PAN)
│       ├── training/      # Training loops and trainers
│       └── utils/         # Utilities and helpers
├── configs/               # Training configuration files
├── scripts/               # Utility scripts and test scripts
├── tests/                 # Test suite
├── apps/
│   └── anime_upscaler_gui/  # Standalone desktop GUI (Tkinter)
├── docs/                  # Documentation
├── assets/                # Example images and assets
│   └── examples/          # Example input/output images
├── backups/               # Backup archives
├── results/               # Benchmark and evaluation results
├── checkpoints/           # Model checkpoints (gitignored)
├── output/                # Output files (gitignored)
└── data/                  # Training data (gitignored)
```

## Documentation

### Getting Started

- [Documentation Index](docs/index.md) - Overview and navigation
- [Python API Reference](docs/api.md) - Complete Python API documentation
- [CLI Reference](docs/cli.md) - Command-line interface reference
- [Development Setup](docs/development.md) - Setting up development environment
- [GUI User Guide](apps/anime_upscaler_gui/README.md) - Desktop app walkthrough, models, and shortcuts

### Core Concepts

- [Model Architectures](docs/models.md) - Detailed model descriptions
- [Training Guide](docs/training.md) - Training procedures and best practices
- [Deployment Guide](docs/deployment.md) - Production deployment

### Additional Resources

- [Architecture](docs/architecture.md) - System architecture
- [Configuration](docs/configuration.md) - Configuration system
- [User Guide](docs/USER_GUIDE.md) - End-user guide
- [Public API](docs/PUBLIC_API.md) - REST API documentation
- [Model Architectures](docs/model_architectures.md) - Model details
- [Quick Reference](docs/QUICK_REFERENCE.md) - Quick reference
- [Troubleshooting](docs/troubleshooting.md) - Troubleshooting guide

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- SPAN: [NTIRE 2024 Challenge](https://github.com/SHI-Labs/SPAN)
- Mamba-PAN: State Space Model for super-resolution
- RealESRGAN: Degradation pipeline
