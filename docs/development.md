# Development Setup

## Prerequisites

### Required

- Python 3.10+
- pip or uv package manager
- Git

### Recommended

- CUDA-capable GPU (8GB+ VRAM recommended)
- CUDA Toolkit 12.1+ (if using GPU)

---

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/anime_sr.git
cd anime_sr
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# Using uv (recommended)
uv venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
# Install PyTorch with CUDA support (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### 4. Verify Installation

```bash
python -c "import anime_sr; print(anime_sr.__version__)"
anime-sr --help
```

---

## Project Structure

```
anime_sr/
├── src/
│   └── anime_sr/
│       ├── __init__.py          # Package init, version
│       ├── __main__.py          # CLI entry point
│       ├── models/              # Model architectures
│       │   ├── base.py          # BaseSRModel
│       │   ├── span/            # SPAN models
│       │   ├── mamba_pan/       # Mamba-PAN models
│       │   ├── students/        # Student models (RFDN, TinySRVGG)
│       │   ├── teachers/        # Teacher models (EDSR, RCAN, SwinIR)
│       │   └── ensemble.py      # Ensemble teacher
│       ├── training/            # Training system
│       │   ├── base_trainer.py  # BaseTrainer
│       │   ├── orchestrator.py  # TrainingOrchestrator
│       │   └── distillation/    # Distillation training
│       ├── distillation/        # Distillation losses
│       │   ├── mtkd/            # MTKD losses
│       │   └── fakd/            # FAKD losses
│       ├── losses/              # Loss functions
│       │   ├── pixel_loss.py    # L1, L2, Charbonnier
│       │   ├── perceptual_loss.py  # VGG, ResNet, DISTS
│       │   ├── adversarial_loss.py # GAN losses
│       │   └── combined_loss.py    # Combined losses
│       ├── data/                # Data pipeline
│       │   ├── base.py          # BaseDataset
│       │   ├── degradation_pipeline.py  # Degradation
│       │   └── datasets/        # Dataset implementations
│       ├── inference/           # Inference engine
│       │   ├── engine.py        # InferenceEngine
│       │   └── tta.py           # Test-time augmentation
│       ├── export/              # Model export
│       ├── api/                 # API server
│       └── utils/               # Utilities
│           ├── config.py        # Configuration
│           ├── metrics.py       # Metrics
│           └── ...
├── configs/                     # Configuration files
├── tests/                       # Test suite
├── docs/                        # Documentation
└── scripts/                     # Utility scripts
```

---

## Running Tests

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test File

```bash
pytest tests/test_models/test_span.py
```

### Run with Coverage

```bash
pytest --cov=anime_sr tests/
```

### Run Tests in Parallel

```bash
pytest -n auto tests/
```

---

## Code Style

### Formatting

```bash
# Format code
black src/ tests/

# Check formatting
black --check src/ tests/
```

### Linting

```bash
# Lint code
flake8 src/ tests/

# Lint with specific rules
flake8 --max-line-length=100 src/
```

### Type Checking

```bash
# Type check (if mypy is installed)
mypy src/anime_sr/
```

---

## Adding a New Model

### 1. Create Model File

Create a new file in `src/anime_sr/models/`:

```python
# src/anime_sr/models/my_model.py
import torch
import torch.nn as nn
from anime_sr.models.base import BaseSRModel

class MyModel(BaseSRModel):
    def __init__(self, scale: int = 4, **kwargs):
        super().__init__(scale=scale)
        # Define layers
        self.conv = nn.Conv2d(3, 64, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    def forward_with_features(self, x: torch.Tensor):
        features = []
        # Forward pass with feature extraction
        out = self.conv(x)
        features.append(out)
        return out, features
```

### 2. Register in Factory

Add to `src/anime_sr/models/__init__.py`:

```python
from .my_model import MyModel

def create_my_model(config):
    return MyModel(
        scale=config.get("scale", 4),
        **config.get("kwargs", {})
    )
```

### 3. Add Configuration

Create `configs/my_model.yaml`:

```yaml
model:
  type: my_model
  scale: 4
  kwargs:
    num_channels: 64
```

### 4. Add Tests

Create `tests/test_models/test_my_model.py`:

```python
import torch
from anime_sr.models.my_model import MyModel

def test_my_model_forward():
    model = MyModel(scale=4)
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    assert y.shape == (1, 3, 256, 256)
```

---

## Adding a New Loss Function

### 1. Create Loss File

```python
# src/anime_sr/losses/my_loss.py
import torch
import torch.nn as nn

class MyLoss(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        loss = torch.mean(torch.abs(sr - hr))
        return loss * self.weight
```

### 2. Register in Factory

Add to `src/anime_sr/losses/__init__.py`:

```python
from .my_loss import MyLoss

def create_my_loss(config):
    return MyLoss(weight=config.get("weight", 1.0))
```

### 3. Add to Config

```yaml
loss:
  type: my_loss
  weight: 1.0
```

---

## Adding a New Dataset

### 1. Create Dataset File

```python
# src/anime_sr/data/datasets/my_dataset.py
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_dir: str, scale: int = 4):
        self.data_dir = data_dir
        self.scale = scale
        # Load data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return (lr, hr) pair
        return lr, hr
```

### 2. Register in Factory

Add to `src/anime_sr/data/__init__.py`:

```python
from .datasets.my_dataset import MyDataset

def create_my_dataset(config):
    return MyDataset(
        data_dir=config["data_dir"],
        scale=config.get("scale", 4)
    )
```

---

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Use Debugger

```python
import pdb; pdb.set_trace()
```

### Profile Code

```python
import cProfile
cProfile.run('my_function()')
```

### Check GPU Memory

```python
import torch
print(torch.cuda.memory_summary())
```

---

## Common Development Tasks

### Run Training with Custom Config

```bash
python -m anime_sr train --config configs/my_config.yaml
```

### Run Inference

```bash
python -m anime_sr infer --input image.png --checkpoint model.pt
```

### Export Model

```bash
python -m anime_sr export --checkpoint model.pt --format onnx
```

### Start API Server

```bash
python -m anime_sr serve --port 8000
```

---

## Pre-Commit Checklist

Before committing code:

- [ ] All tests pass: `pytest tests/`
- [ ] Code formatted: `black src/ tests/`
- [ ] Linting passes: `flake8 src/ tests/`
- [ ] No debug print statements
- [ ] No hardcoded paths
- [ ] Documentation updated (if needed)
- [ ] Type hints added (if applicable)

---

## Troubleshooting

### Import Errors

```bash
# Reinstall package
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

### CUDA Errors

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA version
python -c "import torch; print(torch.version.cuda)"
```

### Test Failures

```bash
# Run with verbose output
pytest -v tests/

# Run specific test
pytest tests/test_models/test_span.py::test_forward -v
```

---

## See Also

- [Python API Reference](api.md) - Python API documentation
- [CLI Reference](cli.md) - Command-line interface
- [Model Architectures](models.md) - Model architecture details
- [Training Guide](training.md) - Training procedures
- [Deployment Guide](deployment.md) - Production deployment
