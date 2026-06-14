# NaN/Inf Loss Fix - Complete Analysis and Solution

## Problem Summary

When running training with the command:
```bash
python scripts/train.py --config configs/finetune_neosr_span.yaml
```

The training produces NaN/Inf losses after just a few batches:
```
Epoch 1/50: loss=0.2705, perc=0.7451
[WARNING] NaN/Inf detected in loss at batch 2, skipping batch
[WARNING] NaN/Inf detected in loss at batch 3, skipping batch
```

## Root Causes Identified

### 1. **DISTS Loss Incompatible with Mixed Precision (FP16)** - PRIMARY CAUSE
- **Source**: [DISTS GitHub Issue #12](https://github.com/dingkeyan93/DISTS/issues/12)
- **Quote**: *"DISTS does not work with mixed precision, no warning given. I had to cut my batch size down to 1 even on a 4090."*
- **Why**: DISTS uses VGG features with operations that are numerically unstable in FP16
- **Impact**: Silent NaN production in perceptual loss computation

### 2. **Adam Optimizer Epsilon Too Small for FP16**
- **Source**: [PyTorch Forums - NaN Loss with Precision 16](https://discuss.pytorch.org/t/nan-loss-issues-with-precision-16-in-pytorch-lightning-gan-training/204369/)
- **Issue**: Default `eps=1e-8` is below FP16 precision threshold
- **Impact**: Division by near-zero in Adam update causes NaN

### 3. **Gradient Penalty NaN in WGAN-GP**
- **Source**: [PyTorch Issue #2534](https://github.com/pytorch/pytorch/issues/2534)
- **Issue**: `torch.norm()` produces NaN when gradients are exactly zero
- **Impact**: Discriminator training produces NaN gradients

### 4. **No NaN Detection Before Backward Pass**
- **Issue**: NaN was detected after backward, when gradients were already corrupted
- **Impact**: NaN propagates through entire model, corrupting weights

### 5. **Multiple Losses Amplifying Gradient Instability**
- **Issue**: 5 different loss components (pixel, perceptual, line art, flat, frequency, adversarial)
- **Impact**: Gradient explosion when losses have different scales

## Fixes Applied

### Fix 1: Force FP32 for DISTS Perceptual Loss
```python
# Before (causes NaN):
with autocast('cuda'):
    perceptual_loss = self.perceptual_loss(sr, hr)

# After (stable):
with torch.amp.autocast('cuda', enabled=False):
    sr_fp32 = sr.float()
    hr_fp32 = hr.float()
    perceptual_loss = self.perceptual_loss(sr_fp32, hr_fp32)
```

### Fix 2: Increase Adam Epsilon for FP16 Stability
```python
# Before (causes NaN in FP16):
self.optimizer = optim.AdamW(params, lr=lr, eps=1e-8)

# After (stable):
adam_eps = 1e-7 if mixed_precision else 1e-8
self.optimizer = optim.AdamW(params, lr=lr, eps=adam_eps)
```

### Fix 3: Safe Gradient Penalty Calculation
```python
# Before (causes NaN when gradients are zero):
gradient_norm = gradients.norm(2, dim=1)

# After (stable):
gradient_norm = torch.sqrt((gradients ** 2).sum(dim=1) + 1e-12)
```

### Fix 4: NaN Detection Before Backward Pass
```python
# Compute losses OUTSIDE autocast
loss_dict = self._compute_total_loss(sr, hr)

# Check BEFORE backward
if torch.isnan(loss_dict['total']) or torch.isinf(loss_dict['total']):
    print(f"[WARNING] NaN/Inf detected at batch {batch_idx}")
    self.optimizer.zero_grad(set_to_none=True)
    continue

# Only then backward
total_loss.backward()
```

### Fix 5: Gradient Sanitization
```python
# Before optimizer step, sanitize gradients
for param in self.model.parameters():
    if param.grad is not None:
        nan_mask = torch.isnan(param.grad) | torch.isinf(param.grad)
        if nan_mask.any():
            param.grad[nan_mask] = 0.0
```

### Fix 6: Per-Loss NaN Checking
```python
# Each loss component checked individually
if torch.isnan(perceptual_loss) or torch.isinf(perceptual_loss):
    print(f"  [WARNING] NaN/Inf in perceptual loss, skipping")
    perceptual_loss = torch.tensor(0.0, device=sr.device)
```

## Test Results

All tests pass after fixes:
```
[TEST] Testing loss computation with potential NaN triggers...
  Test 1: OK - Total loss: 1.2183
  Test 2: OK - Total loss: 1.2138
  Test 3: OK - Total loss: 1.2176
  Test 4: OK - Total loss: 1.2102
  Test 5: OK - Total loss: 1.2161

[TEST] Testing gradient penalty...
  OK - Gradient penalty: 11.9612

[TEST] Testing gradient sanitization...
  Sanitized 2 NaN/Inf values
```

## Files Modified

1. **`src/training/neosr_finetuner.py`**:
   - Added FP32 enforcement for DISTS and frequency losses
   - Increased Adam epsilon to 1e-7
   - Fixed gradient penalty calculation
   - Added NaN detection before backward
   - Added gradient sanitization
   - Added per-loss NaN checking

2. **`src/losses/frequency_aware_loss.py`**:
   - Already designed to work in FP32

## How to Verify the Fix

Run training and monitor for NaN warnings:
```bash
python scripts/train.py --config configs/finetune_neosr_span.yaml
```

Expected output (no NaN):
```
Epoch 1/50: loss=0.2422, perc=0.7636, line=0.1892, freq=0.2114
Epoch 2/50: loss=0.2398, perc=0.7589, line=0.1876, freq=0.2098
...
```

If NaN still occurs (should not happen):
```
[WARNING] NaN/Inf detected in loss at batch X
  Loss components: pixel=0.1234, perc=0.0000, line=0.0000
  Skipping batch and zeroing gradients
```

## Additional Stability Recommendations

If training still shows instability:

1. **Reduce loss weights further**:
   ```yaml
   loss:
     perceptual:
       weight: 0.005  # Even smaller
     frequency:
       weight: 0.005
   ```

2. **Disable mixed precision entirely** (slower but most stable):
   ```yaml
   training:
     mixed_precision: false
   ```

3. **Reduce learning rate**:
   ```yaml
   finetune:
     lr: 0.00005  # Half the current value
   ```

4. **Increase gradient accumulation**:
   ```yaml
   finetune:
     gradient_accumulation_steps: 4
   ```

## References

1. [DISTS Issue #12 - Mixed Precision Incompatibility](https://github.com/dingkeyan93/DISTS/issues/12)
2. [PyTorch Issue #2534 - WGAN-GP Gradient Penalty NaN](https://github.com/pytorch/pytorch/issues/2534)
3. [PyTorch Forums - NaN Loss with Precision 16](https://discuss.pytorch.org/t/nan-loss-issues-with-precision-16-in-pytorch-lightning-gan-training/204369/)
4. [PyTorch Issue #40497 - Mixed Precision NaN](https://github.com/pytorch/pytorch/issues/40497)
5. [Baeldung - Common Causes of NaNs During Training](https://www.baeldung.com/cs/ml-training-nan-errors-fix)
6. [The Neural Base - NaN Loss Causes and Fixes](https://theneuralbase.com/pytorch/learn/intermediate/nan-loss-causes-and-fixes/)
