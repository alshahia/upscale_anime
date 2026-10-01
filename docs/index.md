# anime_sr Documentation

## Overview

**anime_sr** is a Python package for anime super-resolution using knowledge distillation. It provides a complete pipeline for training lightweight student models from powerful teacher models, with support for multiple architectures, loss functions, and deployment options.

### Key Features

- **Knowledge Distillation**: MTKD (Multi-Teacher Knowledge Distillation) and FAKD (Feature Affinity Knowledge Distillation)
- **Multiple Architectures**: SPAN, Mamba-PAN, RFDN, TinySRVGG, and more
- **Comprehensive Losses**: Pixel, perceptual, adversarial, wavelet, gradient, and anime-specific losses
- **Flexible Data Pipeline**: RealESRGAN-style degradation, multiple preprocessing modes
- **Easy Deployment**: ONNX export, TorchScript, FastAPI server, Docker support

---

## Quick Start

### Installation

```bash
pip install -e .
```

### Train a Model

```bash
python -m anime_sr train --config configs/default.yaml
```

### Run Inference

```bash
python -m anime_sr infer --input image.png --checkpoint runs/distill_v1/student_best.pt
```

### Start API Server

```bash
python -m anime_sr serve --port 8000
```

---

## Documentation

### Getting Started

- [Python API Reference](api.md) - Complete Python API documentation
- [CLI Reference](cli.md) - Command-line interface reference
- [Development Setup](development.md) - Setting up development environment

### Core Concepts

- [Model Architectures](models.md) - Detailed model descriptions
- [Training Guide](training.md) - Training procedures and best practices
- [Deployment Guide](deployment.md) - Production deployment

### Advanced Topics

- [Configuration](../docs/configuration.md) - Configuration system
- [Architecture](../docs/architecture.md) - System architecture
- [User Guide](../docs/USER_GUIDE.md) - End-user guide
- [Public API](../docs/PUBLIC_API.md) - REST API documentation

---

## Examples

### Basic Inference

```python
from anime_sr.inference.engine import InferenceEngine

engine = InferenceEngine.from_checkpoint(
    checkpoint_path="runs/distill_v1/student_best.pt",
    model_type="rfdn",
    device="cuda"
)

result = engine.run("input.png", "output.png")
```

### Training a Student Model

```python
from anime_sr.models.students import RFDN
from anime_sr.training import BaseTrainer

model = RFDN(scale=4)
trainer = BaseTrainer(config=config, model=model)

for epoch in range(num_epochs):
    train_metrics = trainer.train_epoch(train_loader, epoch)
    val_metrics = trainer.validate(val_loader)
```

### Exporting to ONNX

```bash
python -m anime_sr export \
    --checkpoint runs/distill_v1/student_best.pt \
    --format onnx \
    --output exports/student.onnx
```

---

## Project Structure

```
anime_sr/
├── src/anime_sr/           # Main package
│   ├── models/            # Model architectures
│   ├── training/          # Training system
│   ├── distillation/      # Distillation losses
│   ├── losses/            # Loss functions
│   ├── data/              # Data pipeline
│   ├── inference/         # Inference engine
│   ├── export/            # Model export
│   ├── api/               # API server
│   └── utils/             # Utilities
├── configs/               # Configuration files
├── tests/                 # Test suite
├── docs/                  # Documentation
└── scripts/               # Utility scripts
```

---

## Contributing

See [Development Setup](development.md) for information on setting up the development environment and contributing to the project.

---

## License

This project is licensed under the MIT License.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/your-org/anime_sr/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/anime_sr/discussions)
- **Documentation**: This documentation

---

## See Also

- [Python API Reference](api.md) - Python API documentation
- [CLI Reference](cli.md) - Command-line interface
- [Model Architectures](models.md) - Model architecture details
- [Training Guide](training.md) - Training procedures
- [Development Setup](development.md) - Development environment
- [Deployment Guide](deployment.md) - Production deployment
