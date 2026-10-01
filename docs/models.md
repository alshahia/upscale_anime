# Model Architectures

## Overview

The anime super-resolution project supports multiple model architectures for both teacher and student models. This document describes each architecture, its use case, and how to use it.

---

## Base Architecture

All models inherit from `BaseSRModel`:

```python
class BaseSRModel(nn.Module):
    def __init__(self, scale: int = 4):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return (output, feature_maps) for distillation."""
        raise NotImplementedError
```

---

## Teacher Models

### SPAN (Super-Resolution with Parameter-free Attention)

**Paper**: NTIRE 2024 Winner
**Use Case**: Primary teacher model for distillation

SPAN uses parameter-free attention mechanisms and ConvLoRA for efficient super-resolution.

```python
from anime_sr.models.span import SPANModel, SPANTiny, SPANF, create_span_model

# SPANTiny: 26 channels, 12 blocks (recommended for distillation)
model = SPANTiny(scale=4)

# SPANF: 32 channels, 10 blocks (higher quality, slower)
model = SPANF(scale=4)

# From config
config = {
    "type": "span_tiny",
    "scale": 4,
    "num_channels": 26,
    "num_blocks": 12,
}
model = create_span_model(config)
```

#### Architecture Details

- **Parameter-free attention**: No learnable parameters in attention mechanism
- **ConvLoRA**: Low-rank adaptation for convolutional layers
- **Feature extraction**: `forward_with_features()` returns intermediate features for distillation

#### Pretrained Weights

```python
from anime_sr.models.teachers import load_teacher_model

teacher = load_teacher_model(
    checkpoint_path="checkpoints/span_teacher.pth",
    scale=4,
    arch="span"
)
```

---

### EDSR (Enhanced Deep Residual Networks)

**Paper**: CVPR 2017
**Use Case**: Lightweight teacher model

```python
from anime_sr.models.teachers import EDSR

model = EDSR(scale=4, num_blocks=16, num_features=64)
```

#### Architecture Details

- **Residual blocks**: 16 blocks with 64 features
- **No batch normalization**: Simpler architecture
- **Global residual learning**: Skip connection from input to output

---

### RCAN (Residual Channel Attention Network)

**Paper**: ECCV 2018
**Use Case**: High-quality teacher model

```python
from anime_sr.models.teachers import RCAN

model = RCAN(scale=4, num_blocks=10, num_features=64)
```

#### Architecture Details

- **Channel attention**: CALayer for channel-wise feature recalibration
- **Residual groups**: 10 groups with 10 blocks each
- **Global skip connection**: Improves gradient flow

---

### SwinIR (Swin Transformer for Image Restoration)

**Paper**: ICCV 2021
**Use Case**: State-of-the-art teacher model

```python
from anime_sr.models.teachers import SwinIR

model = SwinIR(scale=4, img_size=64, patch_size=1)
```

#### Architecture Details

- **Swin Transformer blocks**: Window-based multi-head self-attention
- **Shifted windows**: Cross-window connections for global modeling
- **Patch merging/unmerging**: For downsampling/upsampling

---

### RealESRGAN

**Paper**: ICCV 2021
**Use Case**: Real-world super-resolution teacher

```python
from anime_sr.models.teachers import RealESRTeacher

model = RealESRTeacher(scale=4)
```

#### Architecture Details

- **SRVGG body**: Stacked 3x3 convolutions + PReLU
- **Residual connections**: Nearest-neighbor upsampling + residual
- **Real-world trained**: On real-world degraded images

---

### Spandrel Teacher

Load any spandrel-compatible model:

```python
from anime_sr.models.teachers import SpandrelTeacher

model = SpandrelTeacher(
    model_path="checkpoints/realesrgan_x4plus.pth",
    scale=4
)
```

---

### Auto-Detect Architecture

```python
from anime_sr.models.teachers import auto_detect_architecture, create_teacher_model

# Auto-detect from checkpoint
arch = auto_detect_architecture("checkpoints/unknown.pth")
print(f"Detected architecture: {arch}")

# Create model
model = create_teacher_model(
    arch=arch,
    scale=4,
    checkpoint_path="checkpoints/unknown.pth"
)
```

---

## Student Models

### RFDN (Residual Feature Distillation Network)

**Use Case**: Primary student model for distillation
**Parameters**: <600K

```python
from anime_sr.models.students import RFDN, build_student

# Create RFDN
model = RFDN(scale=4, nf=52, num_blocks=6)

# Build from checkpoint
model = build_student(
    checkpoint_path="runs/distill_v1/student_best.pt",
    arch="rfdn"
)
```

#### Architecture Details

- **Feature distillation**: Intermediate features for knowledge transfer
- **Residual blocks**: 6 blocks with 52 features
- **1x1 adapters**: Learnable projections for feature matching (training only)

#### Forward with Features

```python
# During training
output, features = model(lr_image, return_features=True)

# During inference (faster, no features)
output = model(lr_image)
```

---

### TinySRVGG

**Use Case**: Lightweight student with SRVGG body
**Parameters**: ~100K

```python
from anime_sr.models.students import TinySRVGGStudent

model = TinySRVGGStudent(scale=4)
```

#### Architecture Details

- **SRVGG body**: Stacked 3x3 convolutions + PReLU
- **No TAPs**: Training-free artifact prevention
- **Residual connections**: Nearest-neighbor upsampling + residual

---

### MambaIRv2

**Use Case**: State-of-the-art student with state-space modeling
**Note**: Requires `mamba-ssm` package (Linux only)

```python
from anime_sr.models.students.mambair import MambaIRv2Student

model = MambaIRv2Student(
    scale=4,
    embed_dim=48,
    num_blocks=8,
    d_state=16
)
```

#### Architecture Details

- **State space model**: Efficient long-range modeling
- **VSS blocks**: Vision state space blocks
- **Vectorized scan**: Efficient 4-direction scanning

---

### TinyStudent

**Use Case**: Ultra-lightweight student for edge devices
**Parameters**: ~100K

```python
from anime_sr.models.ensemble import TinyStudent, create_tiny_student

model = create_tiny_student(scale=4)
```

#### Architecture Details

- **16 channels**: Very lightweight
- **3 blocks**: Minimal depth
- **Fast inference**: Optimized for real-time applications

---

## Ensemble Models

### EnsembleTeacher

Combines two teacher models with weighted averaging:

```python
from anime_sr.models.ensemble import EnsembleTeacher, create_ensemble_teacher

# Create ensemble
teacher = create_ensemble_teacher(
    a_ckpt="checkpoints/teacher_a.pth",
    b_ckpt="checkpoints/teacher_b.pth",
    weight_a=0.5,
    weight_b=0.5
)

# Forward pass (frozen)
with torch.no_grad():
    output = teacher(lr_image)
```

#### Weighted Ensemble

```python
# Custom weights
teacher = EnsembleTeacher(
    model_a=model_a,
    model_b=model_b,
    weight_a=0.7,
    weight_b=0.3
)
```

---

## Model Comparison

### Teacher Models

| Model | Parameters | Quality | Speed | Use Case |
|-------|------------|---------|-------|----------|
| SPAN-Tiny | ~2.5M | High | Medium | Primary teacher |
| SPANF | ~3.5M | Very High | Slow | High-quality teacher |
| EDSR | ~1.5M | Medium | Fast | Lightweight teacher |
| RCAN | ~15M | High | Slow | High-quality teacher |
| SwinIR | ~11M | Very High | Slow | SOTA teacher |
| RealESRGAN | ~16M | High | Medium | Real-world teacher |

### Student Models

| Model | Parameters | Quality | Speed | Use Case |
|-------|------------|---------|-------|----------|
| RFDN | ~600K | Good | Fast | Primary student |
| TinySRVGG | ~100K | Good | Very Fast | Lightweight student |
| MambaIRv2 | ~800K | Very Good | Medium | SOTA student |
| TinyStudent | ~100K | Medium | Very Fast | Edge devices |

---

## Feature Extraction

### Teacher Features

```python
# Get intermediate features from teacher
output, features = teacher.forward_with_features(lr_image)

# features is a list of tensors from different layers
for i, feat in enumerate(features):
    print(f"Feature {i}: {feat.shape}")
```

### Student Features

```python
# Get intermediate features from student
output, features = student(lr_image, return_features=True)

# Features are used for feature distillation loss
loss = feature_loss(student_features, teacher_features)
```

---

## Model Conversion

### Export to ONNX

```bash
python -m anime_sr export \
    --checkpoint runs/distill_v1/student_best.pt \
    --format onnx \
    --output exports/student.onnx
```

### Export to TorchScript

```bash
python -m anime_sr export \
    --checkpoint runs/distill_v1/student_best.pt \
    --format torchscript \
    --output exports/student.pt
```

---

## Loading Models

### From Checkpoint

```python
from anime_sr.models.students import build_student

# Auto-detect architecture
model = build_student("runs/distill_v1/student_best.pt")

# Specify architecture
model = build_student("runs/distill_v1/student_best.pt", arch="rfdn")
```

### From Pretrained

```python
from anime_sr.utils import download_pretrained_model, load_pretrained_weights

# Download pretrained model
path = download_pretrained_model("span_teacher")

# Load weights into model
load_pretrained_weights(model, path)
```

---

## Model Summary

### Print Model Summary

```python
from torchsummary import summary

model = RFDN(scale=4)
summary(model, input_size=(3, 64, 64))
```

### Count Parameters

```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Trainable parameters: {count_parameters(model):,}")
```

---

## See Also

- [Python API Reference](api.md) - Python API documentation
- [CLI Reference](cli.md) - Command-line interface
- [Training Guide](training.md) - Training procedures
- [Development Setup](development.md) - Development environment
- [Deployment Guide](deployment.md) - Production deployment
