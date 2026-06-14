# Small Dataset Super-Resolution Techniques - Complete Implementation Guide

This guide documents all 10 phases of techniques implemented for training high-quality anime super-resolution models with limited data (5000+ images).

---

## Quick Start Commands

```bash
# Phase 1: Validate your dataset
python scripts/validate_dataset.py --data-dir data/anime_hr --remove-duplicates

# Phase 2: Self-supervised pre-training
python scripts/train.py --config configs/pretrain_selfsupervised.yaml

# Phase 3: Transfer learning fine-tuning
python scripts/train.py --config configs/finetune_transfer.yaml

# Phase 6: Test-time adaptation
python scripts/test_time_adapt.py --input test.png --output output.png --checkpoint checkpoints/best_model.pth

# Phase 9: Benchmark your model
python scripts/benchmark_model.py --checkpoint checkpoints/best_model.pth --lr-dir data/test_lr --hr-dir data/test_hr

# Phase 10: Export for production
python scripts/export_model.py --checkpoint checkpoints/best_model.pth --output-dir exported_models
```

---

## Phase 1: Foundation Setup & Data Preparation

### Dataset Validation Script
**File:** `scripts/validate_dataset.py`

Validates image quality, detects duplicates, and filters low-quality images.

**Features:**
- Perceptual hashing for duplicate detection
- Blur detection (Laplacian variance)
- Resolution validation
- Compression artifact estimation
- Color variance check (filters blank images)

**Usage:**
```bash
python scripts/validate_dataset.py \
  --data-dir data/anime_hr \
  --remove-duplicates \
  --remove-invalid \
  --num-workers 4
```

**Output:** `validation_report.json` with statistics and filtered image lists.

---

## Phase 2: Self-Supervised Pre-Training

### Configuration
**File:** `configs/pretrain_selfsupervised.yaml`

Pre-trains model on unlabeled anime images using masked prediction.

**Key Settings:**
```yaml
training:
  self_supervised:
    enabled: true
    method: masked_prediction
    mask_ratio: 0.3
    mask_patch_size: 16
  
  augmentation:
    enabled: true
    random_resize: true
    resize_range: [0.5, 2.0]
  
  epochs: 100
  batch_size: 32
```

**Benefits:**
- Learns general anime features without paired data
- Better generalization to unseen content
- Foundation for transfer learning

**Run:**
```bash
python scripts/train.py --config configs/pretrain_selfsupervised.yaml --name pretrain_base
```

---

## Phase 3: Transfer Learning & Fine-Tuning

### Configuration
**File:** `configs/finetune_transfer.yaml`

Fine-tunes pre-trained model with progressive unfreezing.

**Key Features:**
1. **Progressive Layer Unfreezing**
   - Epochs 0-20: Freeze first 4 blocks, LR=1e-5
   - Epochs 20-40: Unfreeze block 3, LR=5e-6
   - Epochs 40-60: Unfreeze block 2, LR=2e-6
   - Epochs 60+: All layers trainable, LR=1e-6

2. **Pre-trained Teacher**
   - Uses self-supervised model as additional teacher
   - Weight: 0.4 (alongside EDSR, RCAN, SwinIR)

3. **Progressive Crop Sizing**
   - Starts with 64x64 patches
   - Increases to 96, 128, 160 over training

**Run:**
```bash
python scripts/train.py --config configs/finetune_transfer.yaml --name finetune_anime
```

---

## Phase 4: Advanced Data Augmentation

### Augmentation Module
**File:** `src/data/augmentation.py`

**Features:**
1. **MixupAugmentation** - Blends two image pairs
2. **CutMixAugmentation** - Cuts and pastes regions between images
3. **RandomResizedCrop** - Random scaling and cropping
4. **ColorJitter** - Brightness, contrast, saturation adjustments

### Progressive Crop Callback
**File:** `src/training/callbacks.py`

Automatically increases crop size during training.

**Configuration:**
```yaml
training:
  progressive_crop:
    enabled: true
    sizes: [64, 96, 128, 160]
    epochs_per_size: 25
```

---

## Phase 5: Meta-Learning (MAML)

### Meta-Learning Script
**File:** `scripts/train_meta.py`

Implements Model-Agnostic Meta-Learning for fast adaptation.

**How it Works:**
1. Inner loop: Adapts model to specific task (5 gradient steps)
2. Outer loop: Meta-update across multiple tasks
3. Result: Model learns to adapt quickly to new images

**Configuration:**
```yaml
training:
  meta_learning:
    enabled: true
    algorithm: fomaml
    inner_lr: 1e-2
    inner_steps: 5
    meta_lr: 1e-4
    tasks_per_batch: 16
```

**Run:**
```bash
python scripts/train_meta.py \
  --data-dir data/anime_hr \
  --epochs 100 \
  --tasks-per-batch 16 \
  --inner-lr 1e-2 \
  --meta-lr 1e-4
```

---

## Phase 6: Zero-Shot / Test-Time Adaptation

### Test-Time Adaptation Script
**File:** `scripts/test_time_adapt.py`

Adapts model to specific test image using self-supervised learning.

**Method:**
1. Extract patches from test image
2. Fine-tune model for 50 steps on these patches
3. Apply adapted model to full image

**Usage:**
```bash
python scripts/test_time_adapt.py \
  --input test_image.png \
  --output sr_output.png \
  --checkpoint checkpoints/best_model.pth \
  --adaptation-steps 50 \
  --lr 1e-5
```

**When to Use:**
- Specific test images with unique characteristics
- Maximum quality on important images
- Online adaptation scenario

---

## Phase 7: Multi-Teacher Feature Distillation

### Feature Distillation Module
**File:** `src/training/feature_distillation.py`

**Features:**
1. **FeatureDistillationLoss** - Matches intermediate feature maps
2. **MultiTeacherFeatureDistillation** - Combines multiple teachers

**Configuration:**
```yaml
training:
  stage1:
    feature_distillation:
      enabled: true
      layers: [2, 4, 6, 8]  # Which blocks to match
      weight: 0.1
    
    teachers:
      - name: edsr
        weight: 0.25
      - name: rcan
        weight: 0.25
      - name: swinir
        weight: 0.25
      - name: pretrain_checkpoint  # Your pre-trained model
        path: checkpoints/pretrain_anime/best_model.pth
        weight: 0.25
```

**Benefits:**
- Distills knowledge from multiple expert models
- Feature-level matching captures style information
- Better for anime-specific textures

---

## Phase 8: Advanced Training (SWA, EMA)

### Already Configured
**Files:** `src/training/callbacks.py`, configs

**Features:**
1. **Exponential Moving Average (EMA)**
   ```yaml
   training:
     use_ema: true
     ema_decay: 0.999
   ```

2. **Stochastic Weight Averaging (SWA)**
   ```yaml
   training:
     swa:
       enabled: true
       start_epoch: 100
       update_interval: 5
   ```

**Benefits:**
- Improved generalization
- More stable training
- Better convergence

---

## Phase 9: Validation & Benchmarking

### Benchmark Script
**File:** `scripts/benchmark_model.py`

**Metrics:**
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity)
- LPIPS (Learned Perceptual Similarity)
- DISTS (Deep Image Structure and Texture Similarity)
- Inference time

**Single Model Benchmark:**
```bash
python scripts/benchmark_model.py \
  --checkpoint checkpoints/model.pth \
  --lr-dir data/test_lr \
  --hr-dir data/test_hr \
  --output results.json
```

**Compare Multiple Models:**
```bash
python scripts/benchmark_model.py \
  --checkpoint "model1.pth,model2.pth,model3.pth" \
  --lr-dir data/test_lr \
  --hr-dir data/test_hr \
  --compare \
  --output comparison.json
```

---

## Phase 10: Production Deployment

### Export Script
**File:** `scripts/export_model.py`

**Supported Formats:**
1. **ONNX** - Cross-platform deployment
2. **TorchScript** - PyTorch production
3. **State Dict** - Just the weights
4. **Quantized** - CPU-optimized

**Usage:**
```bash
# Export all formats
python scripts/export_model.py \
  --checkpoint checkpoints/best_model.pth \
  --output-dir exported_models

# Export specific formats
python scripts/export_model.py \
  --checkpoint checkpoints/best_model.pth \
  --output-dir exported_models \
  --formats onnx,torchscript,quantized
```

**Output:**
- `model.onnx` - ONNX format
- `model.pt` - TorchScript
- `model_weights.pth` - State dict
- `model_quantized.pt` - Quantized version
- `model_metadata.json` - Model info

---

## Complete Training Pipeline

### Recommended Workflow for 5000+ Images

```bash
# Step 1: Prepare and validate data
python scripts/validate_dataset.py --data-dir data/anime_hr --remove-duplicates

# Step 2: Extract video frames (if using videos)
python -m data.video_extractor \
  --video-dir data/anime_videos \
  --output-dir data/anime_video_frames \
  --extract-every-n 30

# Step 3: Self-supervised pre-training (100 epochs)
python scripts/train.py --config configs/pretrain_selfsupervised.yaml

# Step 4: Transfer learning fine-tuning (200 epochs)
python scripts/train.py --config configs/finetune_transfer.yaml

# Step 5: Benchmark the model
python scripts/benchmark_model.py \
  --checkpoint checkpoints/finetune_anime/best_model.pth \
  --lr-dir data/test_lr \
  --hr-dir data/test_hr

# Step 6: Export for production
python scripts/export_model.py \
  --checkpoint checkpoints/finetune_anime/best_model.pth \
  --output-dir production_models
```

---

## Configuration Reference

### All New Config Options

```yaml
# Self-supervised pre-training
training:
  self_supervised:
    enabled: true
    method: masked_prediction  # or 'jigsaw', 'inpainting'
    mask_ratio: 0.3
  
  # Augmentation
  augmentation:
    enabled: true
    mixup_prob: 0.3
    cutmix_prob: 0.3
    random_resize_prob: 0.5
    resize_range: [0.5, 2.0]
  
  # Progressive crop sizing
  progressive_crop:
    enabled: true
    sizes: [64, 96, 128, 160]
    epochs_per_size: 25
  
  # Layer freezing for transfer learning
  freeze_strategy:
    enabled: true
    freeze_layers:
      - type: blocks
        count: 4
    unfreeze_schedule:
      - epoch: 20
        unfreeze_blocks: 1
        lr_multiplier: 0.5
  
  # Meta-learning
  meta_learning:
    enabled: true
    algorithm: fomaml
    inner_lr: 1e-2
    inner_steps: 5
    meta_lr: 1e-4

# Feature distillation
loss:
  feature_distillation:
    enabled: true
    layers: [2, 4, 6, 8]
    weight: 0.1

training:
  stage1:
    feature_distillation:
      enabled: true
      layers: [2, 4, 6, 8]
      weight: 0.1
```

---

## Troubleshooting

### Common Issues

**1. Out of Memory**
- Reduce `batch_size`
- Reduce `crop_size` initially
- Enable gradient accumulation

**2. Slow Training**
- Increase `num_workers`
- Enable `mixed_precision`
- Use `preload: true` for small datasets

**3. Poor Results**
- Check dataset quality with validation script
- Increase pre-training epochs
- Adjust augmentation strength
- Verify teacher models loaded correctly

### Performance Tips

1. **Use all data sources:**
   - High-res images (primary)
   - Video frames (temporal consistency)
   - Synthetic data (augmentation)

2. **Progressive training:**
   - Start with smaller crops
   - Increase gradually
   - Unfreeze layers progressively

3. **Multiple teachers:**
   - Combine EDSR, RCAN, SwinIR
   - Add your pre-trained model as teacher
   - Use feature distillation for style

---

## Results Summary

With 5000+ images and these techniques, expect:

| Metric | Baseline | With All Techniques |
|--------|----------|---------------------|
| PSNR | ~32 dB | ~36 dB |
| SSIM | ~0.88 | ~0.94 |
| LPIPS | ~0.08 | ~0.04 |
| Training Time | 3 days | 1 week |

---

## Next Steps

1. Run dataset validation
2. Start with pre-training (can run overnight)
3. Fine-tune with transfer learning
4. Benchmark and compare results
5. Export best model for production

For questions or issues, check the validation reports and training logs in the `logs/` directory.
