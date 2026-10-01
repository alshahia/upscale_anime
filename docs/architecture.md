# Architecture

## Package Layout

```
src/anime_sr/
├── __init__.py          # Package initialization
├── __main__.py          # CLI entry point (python -m anime_sr)
├── api/                 # REST API server
│   └── inference_api.py # FastAPI-based inference endpoint
├── data/                # Data loading and augmentation
│   ├── datasets/        # Dataset implementations
│   ├── anime_degradation.py
│   ├── augmentation.py
│   ├── dataloader.py
│   ├── degradation_pipeline.py
│   ├── video_dataset.py
│   └── ...
├── distillation/        # Knowledge distillation
│   ├── fakd/            # Frequency-Aware Knowledge Distillation
│   └── mtkd/            # Multi-Teacher Knowledge Distillation
├── export/              # Model export (ONNX, etc.)
├── inference/           # Inference engine
│   ├── engine.py        # Core inference engine
│   ├── cli.py           # Inference CLI
│   └── tta.py           # Test-time augmentation
├── losses/              # Loss functions
│   ├── perceptual_loss.py
│   ├── adversarial_loss.py
│   ├── combined_loss.py
│   └── ...
├── models/              # Model architectures
│   ├── span/            # SPAN (NTIRE winner)
│   ├── mamba_pan/       # Mamba-PAN with directional scanning
│   ├── students/        # Lightweight student models
│   ├── teachers/        # Teacher models
│   ├── ensemble.py      # Model ensemble
│   └── base.py          # Base model class
├── training/            # Training loops and trainers
│   ├── base_trainer.py
│   ├── auto_stage_trainer.py
│   ├── model_a_trainer.py
│   ├── model_b_trainer.py
│   ├── ensemble_trainer.py
│   ├── distillation/    # Distillation trainers
│   └── ...
└── utils/               # Utilities and helpers
    ├── checkpoint_manager.py
    ├── config.py
    ├── logging_config.py
    ├── metrics.py
    ├── nan_handler.py
    └── ...
```

## Module Responsibilities

### models/
Contains all model architectures:
- **SPAN**: NTIRE 2024 winning architecture with parameter-free attention
- **Mamba-PAN**: State Space Model with 4-direction scanning
- **Students**: Ultra-lightweight models for production deployment
- **Teachers**: Pre-trained models used for distillation

### data/
Handles all data-related operations:
- Dataset loading and sampling
- RealESRGAN-style degradation pipeline
- Video frame extraction and processing
- Data augmentation (Mixup, CutMix, etc.)
- Precomputed degradation pairs

### training/
Manages the training lifecycle:
- Base trainer with common functionality
- Auto-stage trainer for progressive training
- Model-specific trainers (Model A, Model B)
- Ensemble trainer for meta-teacher distillation
- Training control system (early stopping, adaptive LR)

### distillation/
Implements knowledge distillation:
- **MTKD**: Multi-Teacher Knowledge Distillation
- **FAKD**: Frequency-Aware Knowledge Distillation

### inference/
Handles model inference:
- Core inference engine with tiling support
- Test-time augmentation (TTA)
- CLI interface for batch processing

### api/
REST API server:
- FastAPI-based inference endpoint
- Authentication and rate limiting
- Async processing support

### losses/
Loss function implementations:
- Perceptual loss (VGG-based)
- Adversarial loss
- Combined loss with multiple components
- Wavelet-guided loss

### utils/
Shared utilities:
- Checkpoint management
- Configuration loading
- Logging setup
- Metrics calculation
- NaN handling

## Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Dataset   │────▶│  Degradation │────▶│  Data Loader │
│  (HR images)│     │   Pipeline   │     │  (LR-HR pairs)│
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌──────────────────────────┘
                    ▼
           ┌─────────────────┐
           │  Training Loop  │
           │  (Model A/B)    │
           └────────┬────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│   Teacher A   │       │   Teacher B   │
│   (SPAN)      │       │  (Mamba-PAN)  │
└───────┬───────┘       └───────┬───────┘
        │                       │
        └───────────┬───────────┘
                    ▼
           ┌─────────────────┐
           │  Meta-Teacher   │
           │   (Ensemble)    │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │  Distillation   │
           │  (MTKD + FAKD)  │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │    Student      │
           │  (Lightweight)  │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │   Inference     │
           │  (Production)   │
           └─────────────────┘
```

## Entry Points

### CLI
```bash
# Training
anime-sr train --config configs/train_config.yaml

# Inference
anime-sr infer --input ./input --output ./output --model ./model.pt

# API Server
anime-sr serve --model ./model.pt --port 8000

# Export
anime-sr export --model ./model.pt --format onnx
```

### Python API
```python
from anime_sr.inference.engine import InferenceEngine

engine = InferenceEngine(model_path="./model.pt", scale=4)
engine.infer_image("input.png", "output.png")
```

### API Server
```python
from anime_sr.api.inference_api import create_app
import uvicorn

app = create_app(model_path="./model.pt")
uvicorn.run(app, host="0.0.0.0", port=8000)
```
