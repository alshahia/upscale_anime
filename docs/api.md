# anime_sr Python API Reference

## Package Overview

`anime_sr` is a Python package for anime super-resolution using knowledge distillation. It provides a complete pipeline for training lightweight student models from powerful teacher models, with support for multiple architectures, loss functions, and deployment options.

### Installation

```bash
pip install -e .
```

### Quick Start

```python
import torch
from anime_sr.models.students import build_student
from anime_sr.inference.engine import InferenceEngine

# Load a trained student model
engine = InferenceEngine.from_checkpoint(
    checkpoint_path="runs/distill_v1/student_best.pt",
    model_type="rfdn",
    device="cuda"
)

# Upscale an image
result = engine.run("input.png", "output.png")
print(f"Output shape: {result.shape}")
```

---

## Model Architectures

### Base Classes

All models inherit from `BaseSRModel` which provides common functionality:

```python
from anime_sr.models.base import BaseSRModel

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

### SPAN (Super-Resolution with Parameter-free Attention)

SPAN is the NTIRE 2024 winner architecture using parameter-free attention and ConvLoRA.

```python
from anime_sr.models.span import SPANModel, SPANTiny, SPANF, create_span_model

# Create from config
config = {
    "type": "span_tiny",  # or "spanf"
    "in_channels": 3,
    "out_channels": 3,
    "scale": 4,
    "num_channels": 26,   # SPANTiny: 26, SPANF: 32
    "num_blocks": 12,    # SPANTiny: 12, SPANF: 10
}
model = create_span_model(config)

# Or instantiate directly
model = SPANTiny(scale=4)   # 26 channels, 12 blocks
model = SPANF(scale=4)      # 32 channels, 10 blocks

# Forward pass
output = model(lr_image)  # (B, 3, H, W) -> (B, 3, 4H, 4W)

# Get intermediate features for distillation
output, features = model.forward_with_features(lr_image)

# Merge LoRA weights for inference
model.merge_lora_weights()
```

### Mamba-PAN

State-space model with 4-direction scanning for efficient long-range modeling.

```python
from anime_sr.models.mamba_pan import MambaPANModel, create_mamba_pan_model

config = {
    "type": "mamba_pan",
    "in_channels": 3,
    "out_channels": 3,
    "scale": 4,
    "embed_dim": 48,
    "num_blocks": 8,
    "d_state": 16,
}
model = create_mamba_pan_model(config)

# Forward with direction-specific features
output = model(lr_image)
output, features = model.forward_with_features(lr_image)
output, direction_outputs = model.forward_with_direction_features(lr_image)
```

### RFDN (Residual Feature Distillation Network)

Lightweight student model with <600K parameters.

```python
from anime_sr.models.students import RFDN, build_student

# Create RFDN student
model = RFDN(scale=4, nf=52, num_blocks=6)

# Forward with features for distillation
output, features = model(lr_image, return_features=True)

# Build from checkpoint
model = build_student(checkpoint_path="runs/distill_v1/student_best.pt", arch="rfdn")
```

### TinySRVGG

SRVGG-body student without TAPs (Training-free Artifact Prevention).

```python
from anime_sr.models.students import TinySRVGGStudent

model = TinySRVGGStudent(scale=4)
output = model(lr_image)
```

### Ensemble Teacher

Combines two teacher models with weighted averaging.

```python
from anime_sr.models.ensemble import EnsembleTeacher, create_ensemble_teacher

# Create ensemble from two checkpoints
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

### Tiny Student

Ultra-lightweight student (~100K parameters).

```python
from anime_sr.models.ensemble import TinyStudent, create_tiny_student

model = create_tiny_student(scale=4)
output = model(lr_image)
```

### Teacher Models

```python
from anime_sr.models.teachers import (
    EDSR, RCAN, SwinIR, SPANTeacher, RealESRTeacher,
    load_teacher_model, create_teacher_model, auto_detect_architecture
)

# Load from checkpoint
teacher = load_teacher_model(
    checkpoint_path="checkpoints/span_teacher.pth",
    scale=4,
    arch="span",  # or "edsr", "rcan", "swinir", "realesr"
    device="cuda"
)

# Auto-detect architecture from checkpoint
arch = auto_detect_architecture("checkpoints/unknown.pth")
teacher = create_teacher_model(arch=arch, scale=4, checkpoint_path="checkpoints/unknown.pth")

# Forward with features
output, features = teacher.forward_with_features(lr_image)
```

---

## Training API

### BaseTrainer

```python
from anime_sr.training import BaseTrainer

trainer = BaseTrainer(config=config, model=model)

# Training loop
for epoch in range(num_epochs):
    train_metrics = trainer.train_epoch(train_loader, epoch)
    val_metrics = trainer.validate(val_loader)

    # Save checkpoint
    trainer.save_checkpoint(epoch, val_metrics, path=f"checkpoints/epoch_{epoch}.pt")

    # Load checkpoint
    trainer.load_checkpoint("checkpoints/best.pt")
```

### ModelATrainer / ModelBTrainer

```python
from anime_sr.training import ModelATrainer, ModelBTrainer

# Model A: Standard SR training
trainer_a = ModelATrainer(config=config, model=model_a)

# Model B: Distillation training
trainer_b = ModelBTrainer(config=config, model=model_b, teacher=teacher)
```

### TrainingOrchestrator

Manages multi-stage training with different modes.

```python
from anime_sr.training import TrainingOrchestrator, TrainingMode

config = {
    "training": {
        "mode": "both_sequential",  # or "model_a", "model_b", "ensemble", "full", "auto_stage"
        "epochs": 100,
        "batch_size": 16,
        "learning_rate": 1e-4,
    }
}

orchestrator = TrainingOrchestrator(config)
orchestrator.setup()

# Train
orchestrator.train(train_loader, val_loader)
```

### Training Modes

| Mode | Description |
|------|-------------|
| `model_a` | Train only Model A (standard SR) |
| `model_b` | Train only Model B (distillation) |
| `both_parallel` | Train A and B simultaneously |
| `both_sequential` | Train A first, then B |
| `ensemble` | Train ensemble teacher |
| `full` | Full pipeline: A → B → ensemble |
| `auto_stage` | Automatically determine stage from checkpoint |

---

## Distillation API

### MTKD (Multi-Teacher Knowledge Distillation)

Wavelet-based distillation loss.

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

# Compute loss
loss = criterion(student_output, teacher_output, hr_target)
```

### FAKD (Feature Affinity Knowledge Distillation)

Second-order statistics-based feature matching.

```python
from anime_sr.distillation.fakd import (
    FeatureAffinityLoss,
    DirectionalFeatureAffinityLoss,
    CrossDirectionConsistencyLoss
)

# Feature affinity loss
criterion = FeatureAffinityLoss(
    layers=[0, 1, 2, 3],
    layer_weights=[1.0, 0.5, 0.25, 0.125],
    affinity_type="covariance",  # or "gram", "correlation"
    temperature=1.0
)

# Directional feature affinity (for Mamba-PAN)
criterion = DirectionalFeatureAffinityLoss(
    directions=["horizontal", "vertical", "diagonal"]
)

# Cross-direction consistency
criterion = CrossDirectionConsistencyLoss(
    consistency_type="mse",  # or "cosine"
    weight=0.1
)
```

### Distillation Training Script

```python
from anime_sr.training.distillation.distill import main

# Run via CLI
# python -m anime_sr.training.distillation.distill #     --data data/anime_video_frames #     --out-dir runs/distill_v1 #     --teacher span #     --arch rfdn #     --epochs 40
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
from anime_sr.losses import (
    VGGPerceptualLoss, ResNetPerceptualLoss, DISTSLoss,
    PerceptualLossFactory
)

# VGG perceptual loss
criterion = VGGPerceptualLoss(
    layers=[3, 8, 15, 22],
    weights=[1.0, 0.5, 0.25, 0.125],
    loss_type="l1"
)

# Factory method
criterion = PerceptualLossFactory.create("vgg", layers=[3, 8, 15, 22])
criterion = PerceptualLossFactory.create("resnet", layers=[1, 2, 3, 4])
criterion = PerceptualLossFactory.create("dists")
```

### Adversarial Losses

```python
from anime_sr.losses import (
    AdversarialLoss, VanillaGANLoss, RelativisticGANLoss,
    HingeGANLoss, SRDiscriminator, create_adversarial_loss
)

# Create adversarial loss
criterion = create_adversarial_loss("hinge")  # or "vanilla", "relativistic"

# Create discriminator
discriminator = SRDiscriminator(in_channels=3, num_channels=64)
```

### Combined Losses

```python
from anime_sr.losses import (
    Stage1CombinedLoss, AdaptiveCombinedLoss, AnimeCombinedLoss
)

# Stage 1 combined loss
criterion = Stage1CombinedLoss(
    l1_weight=1.0,
    wavelet_weight=0.2,
    gradient_weight=0.1,
    diversity_weight=0.05
)

# Adaptive combined loss
criterion = AdaptiveCombinedLoss(
    initial_weights={"l1": 1.0, "perceptual": 0.5, "adversarial": 0.01},
    adapt_every=10
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

### Wavelet Losses

```python
from anime_sr.losses import WaveletLoss, DirectionalWaveletLoss

criterion = WaveletLoss(levels=3, wavelet="haar")
criterion = DirectionalWaveletLoss(levels=3, directions=["h", "v", "d"])
```

### Gradient Losses

```python
from anime_sr.losses import GradientLoss, LaplacianLoss, CombinedGradientLoss

criterion = GradientLoss(kernel_size=3)
criterion = LaplacianLoss(levels=3)
criterion = CombinedGradientLoss(gradient_weight=0.5, laplacian_weight=0.5)
```

### Anime-Specific Losses

```python
from anime_sr.losses import (
    LineArtPreservationLoss, ColorConsistencyLoss,
    FlatRegionPreservationLoss
)

criterion = LineArtPreservationLoss(weight=0.1)
criterion = ColorConsistencyLoss(weight=0.05)
criterion = FlatRegionPreservationLoss(weight=0.1)
```

### Temporal Losses

```python
from anime_sr.losses import TemporalConsistencyLoss, FlowGuidedTemporalLoss

criterion = TemporalConsistencyLoss(weight=0.1)
criterion = FlowGuidedTemporalLoss(flow_weight=0.5, consistency_weight=0.5)
```

### Frequency-Aware Losses

```python
from anime_sr.losses import FrequencyAwareLoss, CombinedFrequencyLoss

criterion = FrequencyAwareLoss(
    low_freq_weight=1.0,
    high_freq_weight=0.5,
    cutoff=0.25
)
criterion = CombinedFrequencyLoss(
    pixel_weight=1.0,
    frequency_weight=0.5
)
```

### FDL (Feature Distillation Loss)

```python
from anime_sr.losses import FDLLoss

criterion = FDLLoss(
    teacher_layers=[0, 1, 2],
    student_layers=[0, 1, 2],
    temperature=1.0
)
```

---

## Data API

### BaseDataset

```python
from anime_sr.data import BaseDataset

dataset = BaseDataset(
    hr_dir="data/hr_images",
    lr_dir="data/lr_images",  # Optional: pre-computed LR
    scale=4,
    crop_size=192,
    augment=True,
    degradation="anime_heavy",  # or "bicubic", "light", "medium", "heavy", "apisr"
    preload=False,
    max_images=None
)

# Get item
lr, hr = dataset[0]  # (3, 48, 48), (3, 192, 192)
```

### DatasetFactory

```python
from anime_sr.data import DatasetFactory

config = {
    "type": "base",
    "hr_dir": "data/hr_images",
    "scale": 4,
    "crop_size": 192,
    "augment": True,
    "degradation": "anime_heavy"
}

dataset = DatasetFactory.create(config)
```

### MultiDataset

```python
from anime_sr.data import MultiDataset

datasets = [dataset1, dataset2, dataset3]
multi_dataset = MultiDataset(
    datasets=datasets,
    weights=[0.5, 0.3, 0.2],
    weight_mode="weighted"  # or "uniform", "sequential"
)
```

### DataLoaderFactory

```python
from anime_sr.data import DataLoaderFactory

train_loader = DataLoaderFactory.create(
    dataset=train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

### PreprocessingManager

```python
from anime_sr.data import PreprocessingManager

manager = PreprocessingManager(
    mode="on_the_fly",  # or "precomputed", "gpu_degradation", "hybrid", "quality_adaptive"
    scale=4,
    degradation="anime_heavy"
)

# Process image
lr_image = manager.process(hr_image)
```

### Video Datasets

```python
from anime_sr.data import VideoDataset, TemporalDataset, extract_frames_from_video

# Extract frames from video
frames = extract_frames_from_video(
    video_path="video.mp4",
    output_dir="frames/",
    fps=24
)

# Video dataset
dataset = VideoDataset(
    video_dir="videos/",
    scale=4,
    crop_size=192
)

# Temporal dataset (for video SR)
dataset = TemporalDataset(
    frame_dir="frames/",
    sequence_length=5,
    scale=4
)
```

### Augmentation

```python
from anime_sr.data import MixupAugmentation, CutMixAugmentation, AugmentationPipeline

# Mixup
mixup = MixupAugmentation(alpha=0.2)
lr_mix, hr_mix = mixup(lr1, hr1, lr2, hr2)

# CutMix
cutmix = CutMixAugmentation(alpha=1.0)
lr_mix, hr_mix = cutmix(lr1, hr1, lr2, hr2)

# Pipeline
pipeline = AugmentationPipeline([
    MixupAugmentation(alpha=0.2),
    CutMixAugmentation(alpha=1.0)
])
```

### Line Enhancement

```python
from anime_sr.data import LineEnhancer

enhancer = LineEnhancer(
    method="xdog",  # or "canny", "sobel"
    threshold=0.5
)

enhanced = enhancer.enhance(image)
```

---

## Inference API

### InferenceEngine

```python
from anime_sr.inference.engine import InferenceEngine

# From checkpoint
engine = InferenceEngine.from_checkpoint(
    checkpoint_path="runs/distill_v1/student_best.pt",
    model_type="rfdn",  # or "srvgg", "mambair"
    device="cuda"
)

# From ensemble
engine = InferenceEngine.from_ensemble(
    checkpoints=["runs/model_a.pt", "runs/model_b.pt"],
    weights=[0.5, 0.5],
    device="cuda"
)

# Single image
result = engine.run("input.png", "output.png")

# Batch processing
results = engine.run_batch(
    inputs=["img1.png", "img2.png"],
    outputs=["out1.png", "out2.png"]
)

# With TTA (Test-Time Augmentation)
result = engine.run_tta("input.png", "output.png")

# Tiled inference (for large images)
result = engine.run_tiled(
    input_path="large.png",
    output_path="output.png",
    tile_size=256,
    overlap=16
)

# Process image array
import numpy as np
from PIL import Image
img = np.array(Image.open("input.png"))
result = engine.process_image(img)

# Benchmark
stats = engine.benchmark(
    input_size=(1, 3, 256, 256),
    num_iterations=100
)
print(f"FPS: {stats['fps']}, Latency: {stats['latency_ms']:.2f}ms")
```

### Quick Inference

```python
from anime_sr.inference.engine import quick_inference

# One-liner
quick_inference(
    checkpoint="runs/distill_v1/student_best.pt",
    input_path="input.png",
    output_path="output.png"
)
```

### ModelSoup

Weight averaging from multiple checkpoints.

```python
from anime_sr.inference.engine import ModelSoup

# Create soup
soup = ModelSoup.from_checkpoints([
    "runs/exp1/student_best.pt",
    "runs/exp2/student_best.pt",
    "runs/exp3/student_best.pt"
])

# Create averaged model
model = soup.create_soup(method="uniform")  # or "weighted", "greedy"

# Save soup
soup.save_soup("runs/soup/student_soup.pt")
```

### TTA (Test-Time Augmentation)

```python
from anime_sr.inference.tta import tta_forward, D4_AUGMENTATIONS

# Apply D4 TTA (8x dihedral group)
output, n_aug, names = tta_forward(
    model=lambda x: model(x),
    lr=lr_tensor,
    use_clip=True
)
print(f"Applied {n_aug} augmentations: {names}")
```

---

## Export API

### ONNX Export

```python
from anime_sr.export import export_onnx, load_student

# Load model
model = load_student(
    run_dir="runs/distill_v1",
    device="cuda",
    arch="rfdn"
)

# Export to ONNX
export_onnx(
    model=model,
    out_path="exports/student.onnx",
    half=False  # Set True for FP16
)
```

### ONNX Runtime

```python
from anime_sr.export import ort_session

# Create ONNX Runtime session
session = ort_session("exports/student.onnx")

# Run inference
import numpy as np
lr = np.random.randn(1, 3, 256, 256).astype(np.float32)
output = session.run(["sr"], {"lr": lr})[0]
```

### Calibration

```python
from anime_sr.export import make_calibration_reader

# Create calibration data reader for INT8 quantization
reader = make_calibration_reader(
    n_samples=100,
    crop_size=256
)
```

### Verification

```python
from anime_sr.export import verify_parity

# Verify ONNX matches PyTorch
verify_parity(
    torch_model=model,
    onnx_path="exports/student.onnx",
    providers=["CUDAExecutionProvider"],
    tag="fp32"
)
```

### Benchmarking

```python
from anime_sr.export import bench

# Benchmark inference
stats = bench(
    fn=lambda: model(lr_tensor),
    warmup=10,
    iters=100
)
print(f"Mean: {stats['mean_ms']:.2f}ms, Std: {stats['std_ms']:.2f}ms")
```

---

## Configuration API

### Config Class

```python
from anime_sr.utils.config import Config, load_config, merge_configs, validate_config

# Load config
config = load_config("configs/default.yaml")

# Or use Config class
config = Config.load("configs/default.yaml")

# Merge configs
merged = merge_configs(base_config, override_config)

# Validate
validate_config(config, schema)

# Access values
lr = config.get("training.learning_rate", default=1e-4)
config.set("training.batch_size", 32)

# Save
config.save("configs/modified.yaml")
```

### Environment Variables

```bash
# Override config values via environment variables
export DSH_TRAINING__LEARNING_RATE=1e-3
export DSH_TRAINING__BATCH_SIZE=32
export DSH_MODEL_PATH=/path/to/model.pth
export DSH_OUTPUT_PATH=/path/to/output
export DSH_LOG_LEVEL=DEBUG
export DSH_CUDA_VISIBLE_DEVICES=0,1
export DSH_ALLOW_PICKLE_CHECKPOINT=1
```

---

## Utility API

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

### Visualizer

```python
from anime_sr.utils import (
    tensor_to_image, save_image_comparison,
    plot_training_history, create_grid_visualization
)

# Convert tensor to image
img = tensor_to_image(tensor)

# Save comparison
save_image_comparison(
    lr=lr_image,
    sr=sr_image,
    hr=hr_image,
    save_path="comparison.png"
)

# Plot training history
plot_training_history(
    history={"train_loss": [...], "val_psnr": [...]},
    save_path="history.png"
)
```

### System Monitor

```python
from anime_sr.utils import GPUMonitor, SystemMonitor, get_optimal_worker_count

# GPU monitoring
gpu_monitor = GPUMonitor()
gpu_info = gpu_monitor.get_info()

# System monitoring
system_monitor = SystemMonitor()
sys_info = system_monitor.get_info()

# Get optimal workers
workers = get_optimal_worker_count()
```

### Pretrained Models

```python
from anime_sr.utils import (
    download_pretrained_model, download_all_teacher_models,
    get_teacher_model_path, check_model_exists
)

# Download a model
path = download_pretrained_model("span_teacher")

# Download all teachers
download_all_teacher_models()

# Check if exists
exists = check_model_exists("span_teacher")
```

### Checkpoint Loader

```python
from anime_sr.utils import load_pretrained_weights, get_checkpoint_info

# Load weights
load_pretrained_weights(
    model=model,
    checkpoint_path="checkpoints/pretrained.pth",
    strict=False
)

# Get checkpoint info
info = get_checkpoint_info("checkpoints/model.pth")
print(f"Epoch: {info['epoch']}, PSNR: {info['psnr']}")
```

---

## API Server

### FastAPI Server

```python
from anime_sr.api import create_app

app = create_app(config={
    "model_path": "runs/distill_v1/student_best.pt",
    "model_type": "rfdn",
    "device": "cuda",
    "api_key": "your-secret-key",
    "rate_limit": 60,  # requests per minute
    "max_file_size_mb": 10
})

# Run with uvicorn
# uvicorn anime_sr.api:create_app --factory --host 0.0.0.0 --port 8000
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/models` | GET | List available models |
| `/load_model/{model_name}` | POST | Load a model |
| `/inference` | POST | Run inference |
| `/models/{model_name}` | DELETE | Unload a model |
| `/optimize_memory` | POST | Optimize memory |
| `/stats` | GET | Server statistics |

---

## Common Patterns

### Training a Student Model

```python
import torch
from anime_sr.models.students import RFDN
from anime_sr.models.teachers import load_teacher_model
from anime_sr.training.distillation.distill import main

# Load teacher
teacher = load_teacher_model(
    checkpoint_path="checkpoints/span_teacher.pth",
    scale=4,
    arch="span"
)

# Create student
student = RFDN(scale=4)

# Train (via CLI recommended)
# python -m anime_sr.training.distillation.distill #     --data data/anime_video_frames #     --teacher span #     --arch rfdn #     --epochs 40
```

### Inference Pipeline

```python
from anime_sr.inference.engine import InferenceEngine

# Load model
engine = InferenceEngine.from_checkpoint(
    checkpoint_path="runs/distill_v1/student_best.pt",
    model_type="rfdn",
    device="cuda"
)

# Process single image
engine.run("input.png", "output.png")

# Process with TTA
engine.run_tta("input.png", "output.png")

# Process large image with tiling
engine.run_tiled("large.png", "output.png", tile_size=256)
```

### Custom Loss Function

```python
import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.l1 = nn.L1Loss()

    def forward(self, sr, hr, teacher_out=None):
        loss = self.l1(sr, hr)
        if teacher_out is not None:
            loss += 0.1 * self.l1(sr, teacher_out)
        return loss * self.weight

# Use in training
criterion = CustomLoss(weight=1.0)
loss = criterion(student_output, hr, teacher_output)
```

### Custom Dataset

```python
from torch.utils.data import Dataset
from anime_sr.data.degradation_pipeline import DegradationPipeline

class CustomDataset(Dataset):
    def __init__(self, image_dir, scale=4):
        self.images = [...]  # Load image paths
        self.scale = scale
        self.degradation = DegradationPipeline(mode="anime_heavy", scale=scale)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        hr = load_image(self.images[idx])
        lr = self.degradation(hr)
        return to_tensor(lr), to_tensor(hr)
```

---

## See Also

- [CLI Reference](cli.md) - Command-line interface
- [Model Architectures](models.md) - Detailed model descriptions
- [Training Guide](training.md) - Training procedures
- [Development Setup](development.md) - Development environment
- [Deployment Guide](deployment.md) - Production deployment
- [API Server](../docs/PUBLIC_API.md) - REST API documentation
