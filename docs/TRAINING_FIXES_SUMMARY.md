# Training System Fixes - Implementation Summary

**Date:** April 28, 2026  
**Status:** ✅ All Critical, High, and Medium Priority Issues Fixed  
**Files Modified:** 12  
**Files Created:** 5

---

## Issues Fixed

### ✅ Critical Issues

#### Issue #1: Parallel Training Optimizer Bug (FIXED)
**File:** `src/training/orchestrator.py`

**Problem:** New optimizers created every epoch in `_train_both_parallel()`, resetting Adam momentum buffers.

**Fix:** 
- Create optimizers/schedulers once before epoch loop
- Store and reuse them across epochs
- Step schedulers after each epoch (was never called)
- Added best checkpoint saving based on validation loss
- Added learning rate logging every 10 epochs

---

### ✅ High Priority Issues

#### Issue #2: Teacher Model Loading (FIXED)
**Files Created:**
- `src/models/teachers/__init__.py`
- `src/models/teachers/edsr.py` - Full EDSR implementation
- `src/models/teachers/rcan.py` - Full RCAN with Channel Attention
- `src/models/teachers/swinir.py` - Full SwinIR with shifted window attention
- `src/models/teachers/teacher_loader.py` - Auto-detection and flexible loading

**Files Modified:**
- `src/models/__init__.py` - Added teacher model exports
- `src/training/model_a_trainer.py` - Replaced placeholder with actual loader

**Problem:** `SimpleTeacherWrapper` used bicubic upsampling instead of actual architectures.

**Fix:**
- Implemented full EDSR, RCAN, SwinIR architectures
- Added automatic architecture detection from checkpoint keys
- Flexible state dict loading with size mismatch handling
- Validation with test forward pass

#### Issue #3: Scheduler State Restoration (FIXED)
**Files:**
- `src/training/base_trainer.py`
- `src/training/model_a_trainer.py`
- `src/training/model_b_trainer.py`

**Problem:** Scheduler state saved but not properly restored (scheduler created after model init).

**Fix:**
- Added `_pending_checkpoint` storage for deferred loading
- Added `_load_scheduler_state_if_available()` method
- Added `set_scheduler()` method that triggers state restoration
- Updated Model A and B trainers to use `self.scheduler` consistently

---

### ✅ Medium Priority Issues

#### Issue #4: PSNR Numerical Instability (FIXED)
**Files:**
- `src/training/model_a_trainer.py`
- `src/training/model_b_trainer.py`
- `src/training/ensemble_trainer.py`
- `src/utils/metrics.py`

**Problem:** `torch.log10(1.0 / mse)` produces `inf` when MSE=0.

**Fix:** Changed to `1.0 / (mse + 1e-10)` in all locations.

#### Issue #5: Loss Computation Inconsistency (FIXED)
**File:** `src/training/model_b_trainer.py`

**Problem:** Model B overwrote loss with MTKD total: `loss = mtkd_dict['total']`

**Fix:** Changed to additive pattern matching Model A:
```python
mtkd_loss, mtkd_dict = self.mtkd_loss(...)
loss = loss + mtkd_loss  # Additive, not overwrite
```

---

### ✅ Low Priority Issues

#### Issue #6: Double Gradient Checkpointing (FIXED)
**File:** `src/training/model_b_trainer.py`

**Problem:** Gradient checkpointing enabled in both `BaseTrainer` and `ModelBTrainer`.

**Fix:** Removed redundant call from `ModelBTrainer.__init__()`.

---

## Architecture Details

### EDSR (Enhanced Deep Residual Network)
- Residual blocks with configurable depth (16/32)
- Feature channels: 64 (base) or 256 (large)
- Residual scaling: 0.1
- PixelShuffle upsampling

### RCAN (Residual Channel Attention Network)
- Channel Attention mechanism (Squeeze-and-Excitation style)
- Residual in Residual (RIR) structure
- 10 groups × 20 RCAB blocks
- Reduction ratio r=16

### SwinIR (Swin Transformer for Image Restoration)
- Swin Transformer blocks with shifted window attention
- RSTB (Residual Swin Transformer Blocks)
- Relative position bias
- Multi-head self-attention
- Window size: 8×8

---

## API Usage

### Loading Teacher Models

```python
from src.models.teachers import load_teacher_model

# Auto-detect architecture
teacher = load_teacher_model(
    checkpoint_path='pretrained/EDSR_x4.pt',
    scale=4,
    device='cuda'
)

# Specify architecture explicitly
teacher = load_teacher_model(
    checkpoint_path='checkpoint.pth',
    scale=4,
    arch='rcan',  # 'edsr', 'rcan', or 'swinir'
    device='cuda'
)
```

### Creating Teacher Models

```python
from src.models.teachers import create_edsr, create_rcan, create_swinir

edsr = create_edsr(scale=4, large=False)  # Base: 16 blocks, 64 features
rcan = create_rcan(scale=4)
swinir = create_swinir(scale=4, small=True)  # Small: embed_dim=60
```

---

## Testing

Run regression tests:
```bash
python tests/test_training_issues.py
```

Tests cover:
1. Parallel training optimizer persistence
2. PSNR numerical stability (no inf)
3. Teacher model architecture creation
4. Scheduler state restoration
5. Loss computation consistency
6. Single gradient checkpointing
7. Teacher loader auto-detection

---

## Verification Checklist

- [x] Parallel training uses persistent optimizers
- [x] Schedulers step after each epoch
- [x] Teacher models load with actual architectures (not placeholder)
- [x] Scheduler state restores on resume
- [x] PSNR calculation handles MSE=0 gracefully
- [x] Model A and B use consistent loss patterns
- [x] No duplicate gradient checkpointing calls
- [x] All checkpoint loading uses `weights_only=True`

---

## Breaking Changes

None. All changes are backward compatible:
- Existing checkpoints load normally
- Configs work without modification
- Trainer APIs unchanged

## Performance Impact

- Teacher models: ~1.5M-15M parameters (similar to original papers)
- Parallel training: ~2x memory usage for holding both models
- Training speed: Unchanged for single model, ~2x for parallel (expected)

---

## Next Steps

1. Train with actual teacher models to validate full pipeline
2. Download pretrained EDSR/RCAN/SwinIR checkpoints
3. Test full MTKD + FAKD training with real teachers
4. Profile memory usage for 8GB VRAM target
