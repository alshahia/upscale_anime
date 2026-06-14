# Troubleshooting Guide

Common issues and their solutions.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [CUDA/GPU Issues](#cudagpu-issues)
3. [Data Issues](#data-issues)
4. [Training Issues](#training-issues)
5. [Memory Issues](#memory-issues)
6. [Performance Issues](#performance-issues)

---

## Installation Issues

### ModuleNotFoundError: No module named 'torch'

**Cause:** Virtual environment not activated

**Solution:**
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# Verify
python -c "import torch; print(torch.__version__)"
```

### mamba-ssm installation fails

**Cause:** CUDA version mismatch or missing CUDA

**Solution:**
```bash
# Check CUDA version
nvidia-smi  # Check driver CUDA version
nvcc --version  # Check toolkit version

# For CUDA 11.8
pip install mamba-ssm==1.1.1 --no-cache-dir

# If CUDA not available, use fallback (already implemented)
# The code works without mamba-ssm using PyTorch fallback
```

---

## CUDA/GPU Issues

### CUDA out of memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solutions:**

1. **Reduce batch size:**
```bash
python scripts/train.py --config base.yaml --batch-size 4
```

2. **Enable gradient checkpointing:**
```yaml
training:
  gradient_checkpointing: true
```

3. **Use smaller model:**
```yaml
model:
  channels: 20  # Instead of 26
  num_blocks: 8  # Instead of 12
```

4. **Reduce crop size:**
```yaml
data:
  crop_size: 96  # Instead of 128
```

5. **Clear cache before training:**
```python
import torch
torch.cuda.empty_cache()
```

### CUDA not available

**Symptoms:**
```
Warning: CUDA not available, switching to CPU
```

**Solutions:**

1. **Check CUDA installation:**
```bash
nvidia-smi
nvcc --version
```

2. **Reinstall PyTorch with CUDA:**
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

3. **Check PyTorch CUDA:**
```python
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
```

---

## Data Issues

### No images found

**Symptoms:**
```
ValueError: No images found in data/anime_hr
```

**Solutions:**

1. **Create test data:**
```bash
python scripts/create_test_data.py --output data/test_hr --num-images 10
```

2. **Check directory exists:**
```bash
ls data/anime_hr  # Linux/Mac
dir data\anime_hr  # Windows
```

3. **Check file extensions:**
```python
from pathlib import Path
path = Path("data/anime_hr")
print(list(path.glob("*.png")))
print(list(path.glob("*.jpg")))
```

### Dataset too small

**Symptoms:**
```
Error: num_samples should be a positive integer value, but got num_samples=0
```

**Cause:** Dataset has 0 images after filtering

**Solution:**
```yaml
data:
  quality_filter:
    enabled: false  # Disable filtering
  
  # Or lower thresholds
  quality_filter:
    enabled: true
    min_resolution: [64, 64]  # Lower minimum
    min_sharpness: 10         # Lower threshold
```

### Images not loading

**Symptoms:**
```
Error: Failed to load image
```

**Solutions:**

1. **Check image format:**
```python
import cv2
img = cv2.imread("path/to/image.png")
print(img is not None)
```

2. **Convert to supported format:**
```python
from PIL import Image
img = Image.open("image.webp")
img.save("image.png")
```

---

## Training Issues

### NaN loss

**Symptoms:**
```
Epoch 10: loss=nan
```

**Causes & Solutions:**

1. **Learning rate too high:**
```yaml
training:
  lr: 0.00001  # Lower learning rate
```

2. **Corrupted images:**
```python
# Validate dataset
from src.data import validate_dataset
result = validate_dataset('data/anime_hr', check_corruption=True)
print(result['corrupted'])  # List corrupted images
```

3. **Gradient explosion:**
```yaml
training:
  gradient_clip: 1.0  # Add gradient clipping
```

### Loss not decreasing

**Symptoms:**
Loss stays constant or decreases very slowly

**Solutions:**

1. **Check data quality:**
```bash
python -c "from src.data import get_dataset_info; print(get_dataset_info('data/anime_hr'))"
```

2. **Increase learning rate:**
```yaml
training:
  lr: 0.0002
```

3. **Check degradation:**
```yaml
degradation:
  enabled: true
  blur_sigma: [0.5, 2.0]  # Ensure reasonable range
```

4. **Verify model is training:**
```python
# Check if gradients are flowing
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.abs().mean()}")
```

### Training crashes after N epochs

**Symptoms:**
Training runs then crashes consistently

**Solutions:**

1. **Reduce workers:**
```yaml
data:
  num_workers: 2  # Lower worker count
```

2. **Disable persistent workers:**
```yaml
data:
  persistent_workers: false
```

3. **Check for memory leaks:**
```bash
# Monitor memory usage
watch -n 1 nvidia-smi  # GPU
watch -n 1 free -h      # RAM
```

### Loss not decreasing in parallel training

**Symptoms:**
Both models show stagnant or erratic loss when training in parallel mode.

**Cause:** Optimizers being recreated every epoch (fixed in current version).

**Solution:**
Update to latest version. Verify optimizers persist across epochs:

```python
# Check optimizer state has momentum buffers
for key in optimizer.state_dict()['state'].keys():
    if 'exp_avg' in optimizer.state_dict()['state'][key]:
        print("Adam momentum state exists")
```

If using an older version, the fix is in `orchestrator.py`:
- Optimizers created once before epoch loop
- Schedulers stepped after each epoch

### Teacher model loading fails

**Symptoms:**
```
Error: Could not load teacher model
Warning: SimpleTeacherWrapper used
```

**Causes:**
1. **Architecture mismatch:** Checkpoint was trained with different model
2. **Missing checkpoint:** File not found in `pretrained/` directory
3. **Corrupted checkpoint:** File is incomplete or damaged

**Solutions:**

1. **Verify checkpoint exists:**
```bash
ls pretrained/
# Should show: EDSR_x4.pt, RCAN_x4.pt, SwinIR_x4.pt
```

2. **Download pretrained models:**
```bash
python -m src.utils.pretrained_models --model all --scale 4
```

3. **Disable Stage 1** if you don't have teachers:
```yaml
training:
  stage1:
    enabled: false
  stage2:
    enabled: true
```

4. **Test teacher loading manually:**
```python
from src.models.teachers import load_teacher_model

model = load_teacher_model(
    'pretrained/EDSR_x4.pt',
    scale=4,
    device='cuda'
)
print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} params")
```

### Stage 1 produces garbage outputs

**Symptoms:**
Knowledge aggregation network outputs noise or bicubic-like results.

**Cause:** Teacher models not loading properly (using placeholder).

**Solution:**
Verify teachers are producing SR outputs, not bicubic upsampling:

```python
import torch
from src.models.teachers import load_teacher_model

# Load teacher
teacher = load_teacher_model('pretrained/EDSR_x4.pt', scale=4)

# Test on sample
x = torch.randn(1, 3, 64, 64).cuda()
with torch.no_grad():
    out = teacher(x)

# Check output is not identical to bicubic
import torch.nn.functional as F
bicubic = F.interpolate(x, scale_factor=4, mode='bicubic')
error = (out - bicubic).abs().mean()

if error < 0.01:
    print("WARNING: Output matches bicubic - teacher not loaded!")
else:
    print(f"Teacher is working: difference from bicubic = {error:.4f}")
```

---

## Memory Issues

### System RAM exhausted

**Symptoms:**
Process killed, system freezes

**Solutions:**

1. **Disable image preloading:**
```yaml
data:
  preload: false
```

2. **Reduce workers:**
```yaml
data:
  num_workers: 2
  prefetch_factor: 1
```

3. **Reduce batch size:**
```yaml
training:
  batch_size: 4
```

### VRAM slowly increasing

**Symptoms:**
VRAM usage grows over time until OOM

**Cause:** Memory leak, likely in DataLoader

**Solution:**
```yaml
data:
  num_workers: 0  # Use main process (slower but stable)
  persistent_workers: false
```

---

## Performance Issues

### Training too slow

**Symptoms:**
< 1 iteration/second

**Solutions:**

1. **Enable mixed precision:**
```yaml
training:
  mixed_precision: true
```

2. **Increase workers:**
```yaml
data:
  num_workers: 4  # Up to number of CPU cores
```

3. **Use CUDNN benchmark:**
```yaml
system:
  cudnn_benchmark: true
```

4. **Profile to find bottleneck:**
```python
import torch.profiler as profiler
with profiler.profile() as prof:
    # Run one batch
print(prof.key_averages().table())
```

### Validation very slow

**Symptoms:**
Validation takes much longer than training

**Cause:** Validation often uses full images, not crops

**Solutions:**

1. **Reduce validation images:**
```python
# Limit validation set
val_dataset = Subset(val_dataset, range(100))  # Only 100 images
```

2. **Increase val_interval:**
```yaml
training:
  val_interval: 20  # Validate less frequently
```

---

## Common Error Messages

### "Expected more than 1 value per channel when training"

**Cause:** Batch size of 1 with BatchNorm

**Solution:**
```yaml
training:
  batch_size: 4  # Increase to at least 2
  drop_last: true  # Drop incomplete batches
```

### "Sizes of tensors must match"

**Cause:** Input/output size mismatch

**Solution:**
```python
# Ensure images are divisible by scale
# In dataset, crop_size should be divisible by scale (usually 4)
```

### "RuntimeError: DataLoader worker exited unexpectedly"

**Cause:** Worker process crashed

**Solutions:**
1. Reduce `num_workers`
2. Disable `persistent_workers`
3. Check for corrupt images

---

## Getting Help

If issues persist:

1. **Check logs:** `logs/` directory for detailed logs
2. **Run diagnostics:**
```bash
python -c "from src.utils import print_environment_info; print_environment_info()"
```

3. **Validate config:**
```bash
python scripts/train.py --config your_config.yaml --dry-run
```

4. **Test with minimal example:**
```bash
python scripts/train.py --config configs/example_quick_test.yaml
```
