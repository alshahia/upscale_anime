# Model Architectures

Reference documentation for teacher model architectures used in knowledge distillation.

---

## EDSR (Enhanced Deep Residual Network)

**Paper:** "Enhanced Deep Residual Networks for Single Image Super-Resolution" (CVPR 2017)

### Architecture

```
Input (3×H×W)
    ↓
Head Conv (3→64 channels)
    ↓
Residual Blocks × N
├── Conv2d (64→64)
├── ReLU
├── Conv2d (64→64)
└── Residual × 0.1
    ↓
Tail Conv (64→64)
    ↓
Upsampler (PixelShuffle)
    ↓
Reconstruction Conv (64→3)
    ↓
Output (3×scaleH×scaleW)
```

### Configurations

| Variant | ResBlocks | Channels | Parameters |
|---------|-----------|----------|------------|
| EDSR | 16 | 64 | ~1.5M |
| EDSR-Large | 32 | 256 | ~40M |

### Key Features

- **Residual Scaling:** 0.1 to stabilize deep networks
- **No Batch Normalization:** Removes artifacts
- **PixelShuffle Upsampling:** Efficient upsampling

### Usage

```python
from src.models.teachers import create_edsr

# Base model
model = create_edsr(scale=4, large=False)

# Large model
model = create_edsr(scale=4, large=True)
```

---

## RCAN (Residual Channel Attention Network)

**Paper:** "Residual Channel Attention Networks for Image Super-Resolution" (ECCV 2018)

### Architecture

```
Input (3×H×W)
    ↓
Head Conv (3→64)
    ↓
Residual Groups × 10
├── RCAB × 20
│   ├── Conv2d
│   ├── ReLU
│   ├── Conv2d
│   └── Channel Attention
│       ├── GlobalAvgPool
│       ├── FC (64→4)
│       ├── ReLU
│       ├── FC (4→64)
│       └── Sigmoid
├── Conv2d
└── Residual × 0.1
    ↓
Tail Conv
    ↓
Upsampler
    ↓
Reconstruction Conv
    ↓
Output
```

### Channel Attention

Squeeze-and-Excitation style attention:

```python
# Global average pooling
y = global_avg_pool(x)  # [B, C, 1, 1]

# FC layers
y = fc1(y)  # [B, C//16, 1, 1]
y = relu(y)
y = fc2(y)  # [B, C, 1, 1]
y = sigmoid(y)

# Scale features
output = x * y  # [B, C, H, W]
```

### Configurations

| Variant | Groups | RCABs/Group | Parameters |
|---------|--------|-------------|------------|
| RCAN | 10 | 20 | ~15M |

### Usage

```python
from src.models.teachers import create_rcan

model = create_rcan(scale=4)
```

---

## SwinIR (Swin Transformer for Image Restoration)

**Paper:** "SwinIR: Image Restoration Using Swin Transformer" (ICCVW 2021)

### Architecture

```
Input (3×H×W)
    ↓
Shallow Feature Extraction (Conv 3→embed_dim)
    ↓
RSTB × 6
├── Swin Transformer Block × depth
│   ├── LayerNorm
│   ├── Window Attention (W-MSA)
│   │   └── Multi-head Self-Attention in 8×8 windows
│   ├── LayerNorm
│   └── MLP
└── Conv Layer + Residual
    ↓
Deep Feature Aggregation (Conv)
    ↓
Upsampler (PixelShuffle)
    ↓
Reconstruction Conv
    ↓
Output
```

### Swin Transformer Block

**Window Attention:**
- Divide feature map into 8×8 non-overlapping windows
- Apply self-attention within each window
- Complexity: O(N) where N = H×W

**Shifted Window Attention:**
- Shift windows by (4, 4) pixels
- Apply cyclic padding
- Cross-window connections

**Relative Position Bias:**
```python
# Learnable bias table: (2W-1)² × num_heads
bias_table = Parameter(torch.zeros((225, 6)))  # W=8, heads=6

# Index into table using relative positions
bias = bias_table[relative_position_index]
attn = q @ k.T + bias  # Add bias to attention scores
```

### Configurations

| Variant | embed_dim | depths | num_heads | Parameters |
|---------|-----------|--------|-----------|------------|
| SwinIR-S | 60 | [6,6,6,6] | [6,6,6,6] | ~1M |
| SwinIR-M | 180 | [6,6,6,6,6,6] | [6,6,6,6,6,6] | ~12M |

### Usage

```python
from src.models.teachers import create_swinir

# Small model (faster)
model = create_swinir(scale=4, small=True)

# Medium model (better quality)
model = create_swinir(scale=4, small=False)
```

---

## Comparison

| Feature | EDSR | RCAN | SwinIR |
|---------|------|------|--------|
| Type | CNN | CNN+Attention | Transformer |
| Complexity | O(N) | O(N) | O(N) |
| Receptive Field | Local | Local | Global |
| Speed | Fast | Medium | Slow |
| Memory | Low | Medium | High |
| Quality | Good | Better | Best |

---

## Loading Teacher Models

### Auto-Detection

```python
from src.models.teachers import load_teacher_model

# Automatically detect architecture from checkpoint
teacher = load_teacher_model(
    checkpoint_path='pretrained/EDSR_x4.pt',
    scale=4,
    device='cuda'
)
```

### Manual Selection

```python
# Specify architecture explicitly
teacher = load_teacher_model(
    checkpoint_path='checkpoint.pth',
    scale=4,
    arch='rcan',  # 'edsr', 'rcan', 'swinir'
    device='cuda'
)
```

### Multiple Teachers

```python
from src.models.teachers import load_multiple_teachers

checkpoints = [
    'pretrained/EDSR_x4.pt',
    'pretrained/RCAN_x4.pt',
    'pretrained/SwinIR_x4.pt'
]

teachers = load_multiple_teachers(checkpoints, scale=4, device='cuda')
```

---

## Extending with Custom Teachers

To add a custom teacher model:

1. **Create model file** in `src/models/teachers/`:
```python
# src/models/teachers/custom.py
class CustomTeacher(nn.Module):
    def __init__(self, scale=4):
        super().__init__()
        # Your architecture

    def forward(self, x):
        # Forward pass
        return x
```

2. **Add to teacher_loader.py**:
```python
from .custom import CustomTeacher

def auto_detect_architecture(checkpoint):
    # Add detection heuristics
    if 'custom_key' in key_str:
        return 'custom'

def create_teacher_model(arch, scale, ...):
    if arch == 'custom':
        return CustomTeacher(scale)
```

3. **Export in __init__.py**:
```python
from .custom import CustomTeacher

__all__ = [..., 'CustomTeacher']
```

---

## Feature Extraction for FAKD

All teacher models support intermediate feature extraction:

```python
# Get intermediate features
features = model.get_features(x)

# Returns dict with keys like:
# - 'head': Head output
# - 'block_X': X-th block output
# - 'body_out': Final features before upsampling
```

Use these features for Feature Affinity Knowledge Distillation (FAKD).
