# Configuration Reference

Complete reference for all configuration options.

## Table of Contents

1. [Model Configuration](#model-configuration)
2. [Training Configuration](#training-configuration)
3. [Data Configuration](#data-configuration)
4. [Loss Configuration](#loss-configuration)
5. [Degradation Configuration](#degradation-configuration)
6. [System Configuration](#system-configuration)

---

## Model Configuration

### SPAN Model

```yaml
model:
  name: "model_a"           # Model identifier
  type: "span"              # Model type: span, mamba_pan, tiny_student
  scale: 4                  # Upsampling factor: 2, 3, 4
  
  # Architecture
  channels: 26              # Base channels (26 for Tiny, 32 for larger)
  num_blocks: 12            # Number of SPAB blocks
  
  # ConvLoRA
  use_lora: true            # Enable LoRA training
  lora_rank: 4              # LoRA rank (typically 4-8)
  lora_alpha: 1.0           # LoRA scaling factor
  
  # I/O
  in_channels: 3            # Input channels (3 for RGB)
  out_channels: 3           # Output channels
```

### Mamba-PAN Model

```yaml
model:
  name: "model_b"
  type: "mamba_pan"
  scale: 4
  
  # Architecture
  channels: 32              # Base channels
  num_blocks: 8             # Fewer blocks than SPAN
  
  # Mamba settings
  mamba_d_state: 16         # SSM state dimension
  mamba_d_conv: 4           # Convolution kernel size
  mamba_expand: 2           # Expansion factor
  
  # Directional scanning
  directions: ["h", "v", "rh", "rv"]
  use_direction_fusion: true
```

### Tiny Student

```yaml
model:
  name: "tiny_student"
  type: "tiny_student"
  scale: 4
  channels: 16              # Minimal channels
  num_blocks: 3             # Minimal blocks
  # ~100K parameters, fastest inference
```

---

## Training Configuration

```yaml
training:
  # Mode
  mode: "model_a"           # model_a, model_b, both_parallel, both_sequential, ensemble, full
  
  # Basic settings
  epochs: 100               # Total training epochs
  batch_size: 8             # Batch size per GPU
  lr: 0.0001                # Learning rate
  
  # Optimization
  optimizer: "adam"         # adam, adamw, sgd
  weight_decay: 0.0           # L2 regularization
  betas: [0.9, 0.99]        # Adam betas
  
  # Scheduler
  scheduler: "step"           # step, multistep, cosine
  step_size: 200000         # Steps between LR decay
  gamma: 0.5                  # LR decay factor
  milestones: [100, 200, 300]  # For multistep scheduler
  T_max: 100                  # For cosine scheduler
  
  # Training stages
  stage1:
    enabled: true
    epochs: 100
    freeze_teachers: true
  
  stage2:
    enabled: true
    epochs: 200
    use_stage1_checkpoint: true
  
  # Advanced settings
  mixed_precision: true       # Use AMP (recommended)
  gradient_checkpointing: false  # For memory saving
  accumulation_steps: 1       # Gradient accumulation
  
  # EMA
  use_ema: true             # Exponential moving average
  ema_decay: 0.999            # EMA decay rate
  
  # Validation
  val_interval: 10            # Validate every N epochs
  save_interval: 50           # Save checkpoint every N epochs
  log_interval: 100           # Log every N iterations
  
  # Hardware
  device: "cuda"              # cuda or cpu
  num_workers: 4              # DataLoader workers
  cudnn_benchmark: true       # CUDNN benchmark (for fixed sizes)
  cudnn_deterministic: false  # Reproducibility vs speed
```

---

## Data Configuration

```yaml
data:
  # Scale factor
  scale: 4                  # Must match model scale
  
  # Datasets (list for multiple)
  datasets:
    - name: "anime_train"   # Dataset identifier
      weight: 1.0           # Sampling weight
      enabled: true         # Enable/disable
      hr_dir: "data/anime_hr"  # HR images directory
      lr_dir: null          # Optional LR directory (null = generate)
      max_images: null      # Limit dataset size (for testing)
    
    - name: "div2k"         # Multiple datasets supported
      weight: 0.5
      enabled: true
      hr_dir: "data/DIV2K_train_HR"
  
  # Preprocessing
  crop_size: 128            # Training patch size
  augment: true             # Random flip/rotate
  
  # Dataloader
  num_workers: 4            # Parallel loading workers
  pin_memory: true          # Pin memory for GPU transfer
  prefetch_factor: 2        # Prefetch batches
  persistent_workers: true    # Keep workers alive
  
  # Validation
  val_split: 0.1            # Validation split (if no val_dir)
  val_crop_size: 256        # Validation crop size
```

### Quality Filtering

```yaml
data:
  quality_filter:
    enabled: true
    min_resolution: [256, 256]
    min_sharpness: 100      # Laplacian variance
    max_blur: 0.5
```

---

## Loss Configuration

```yaml
loss:
  # Pixel loss
  pixel_loss:
    type: "l1"              # l1, l2, charbonnier
    weight: 1.0
    eps: 1e-6               # For Charbonnier loss
  
  # Perceptual loss
  perceptual_loss:
    enabled: false
    type: "vgg"             # vgg or resnet
    layers: ["relu3_3"]     # VGG layers
    weights: [1.0]          # Layer weights
    weight: 0.1             # Overall weight
    loss_type: "l1"         # l1, l2, cosine
  
  # Adversarial loss
  adversarial_loss:
    enabled: false
    type: "vanilla"         # vanilla, relativistic, hinge
    weight: 0.1
    
    # Discriminator settings
    discriminator:
      num_features: 64
      num_layers: 3
      lr: 0.0001
  
  # Distillation losses
  distillation:
    enabled: false
    
    # MTKD wavelet loss
    mtkd_weight: 1.0
    wavelet_levels: 3
    wavelet_type: "haar"
    
    # FAKD feature affinity
    fakd_weight: 0.5
    fakd_layers: [2, 4, 6, 8]
    fakd_layer_weights: [0.1, 0.2, 0.3, 0.4]
    affinity_type: "spatial"  # spatial or channel
    
    # Mamba-specific
    consistency:
      enabled: true
      weight: 0.1
      type: "variance"      # variance or cosine
```

---

## Degradation Configuration

```yaml
degradation:
  enabled: true
  scale: 4                  # Downsampling factor
  
  # Blur
  blur_kernel_size: [7, 9, 11, 13]
  blur_sigma: [0.1, 3.0]
  blur_prob: 0.7
  
  # Noise
  noise_sigma: [0, 10, 25, 50]
  noise_prob: 0.5
  
  # Resize
  resize_prob: [0.2, 0.7, 0.1]  # up, down, keep
  resize_range: [0.15, 1.5]
  
  # JPEG compression
  jpeg_quality: [60, 70, 80, 90, 100]
  jpeg_prob: 0.5
  
  # High-order degradation (RealESRGAN style)
  first_order:
    blur_prob: 0.7
    noise_prob: 0.5
    resize_prob: 0.3
    jpeg_prob: 0.5
  
  second_order:
    blur_prob: 0.3
    noise_prob: 0.3
    resize_prob: 0.2
    jpeg_prob: 0.3
```

---

## System Configuration

```yaml
system:
  seed: 42                  # Random seed (for reproducibility)
  
  # CUDNN
  cudnn_benchmark: true     # Speed for fixed input sizes
  cudnn_deterministic: false  # True for reproducibility
  
  # Paths
  data_root: "data"
  log_dir: "logs"
  checkpoint_dir: "checkpoints"
  result_dir: "results"
  
  # Logging
  use_tensorboard: true
  log_level: "INFO"         # DEBUG, INFO, WARNING, ERROR
```

---

## CLI Argument Reference

Override any config via command line:

```bash
# Model settings
--model.channels 32
--model.num_blocks 16

# Training settings
--batch-size 16
--epochs 200
--lr 0.0002
--num-workers 8

# Data settings
--data.datasets.0.hr_dir "/path/to/data"
--data.crop_size 256
--data.augment false

# Loss settings
--loss.pixel_loss.type "charbonnier"
--loss.distillation.enabled true
--loss.distillation.mtkd_weight 1.5

# System settings
--device cpu
--seed 123
```

---

## Example Configurations

### Fast Training (Testing)

```yaml
training:
  epochs: 3
  batch_size: 4
  val_interval: 1
  use_ema: false

data:
  crop_size: 64
  max_images: 10
```

### High Quality Training

```yaml
model:
  channels: 32
  num_blocks: 16

training:
  epochs: 500
  batch_size: 8
  lr: 0.0001
  
loss:
  pixel_loss:
    type: "charbonnier"
  perceptual_loss:
    enabled: true
    weight: 0.1
```

### Memory-Constrained Training

```yaml
training:
  batch_size: 4
  gradient_checkpointing: true
  
model:
  channels: 20
  num_blocks: 8

data:
  crop_size: 96
```
