# PLAN: Training Speed Optimization — All Phases Implemented

## Status: COMPLETE - All fixes applied and verified

### Changes Applied

| Phase | Feature | Status | Files Changed |
|-------|---------|--------|---------------|
| 1.1 | PyAV-based H.264 compression (3-5x faster) | Done | `src/data/compression_modules.py` |
| 1.2 | Buffer reuse for WebP/AVIF | Done | `src/data/compression_modules.py` |
| 2.1 | torch.compile() integration | Done (disabled on Windows, needs Triton) | `src/training/neosr_finetuner.py` |
| 2.2 | Discriminator compile support | Done | `src/training/neosr_finetuner.py` |
| 3.1 | Gradient checkpointing in SPAN model | Done | `src/models/span/neosr_span.py` |
| 3.2 | Wired checkpointing into finetuner | Done | `src/training/neosr_finetuner.py` |
| 4.1 | val_interval: 5 (was 2) | Done | `configs/finetune_neosr_span_v4.yaml` |
| 5.1 | workers_per_gpu: 6 (was 4) | Done | `src/data/dataloader.py` |
| 5.2 | prefetch_factor: 4 (was 2) | Done | `src/data/dataloader.py` |
| 6.1 | Precompute degradation script | Done | `scripts/precompute_degradation.py` |
| 6.2 | PrecomputedDataset class | Done | `src/data/precomputed_dataset.py` |
| 6.3 | Wired into dataloader factory | Done | `src/data/dataloader.py` |
| 7.1 | cudnn.benchmark + allow_tf32 | Done | `src/training/neosr_finetuner.py` |
| 8.1 | FDL compute_every: 2 | Done | `src/training/neosr_finetuner.py` + config |
| 10.1 | Epochs: 100 (was 150) | Done | `configs/finetune_neosr_span_v4.yaml` |
| 10.2 | Early stopping patience: 15 (was 25) | Done | `configs/finetune_neosr_span_v4.yaml` |
| 10.3 | Phase1 epochs: 40 (was 50) | Done | `configs/finetune_neosr_span_v4.yaml` |

### Follow-up Fixes (Post-Implementation)

| Issue | Fix | Status |
|-------|-----|--------|
| SPAN model lacked `gradient_checkpointing_enable()` | Added method + checkpoint wrapper for SPAB blocks | Done |
| PrecomputedDataset not wired into dataloader | Added `precomputed_dir` config support in `DataLoaderFactory.create()` | Done |
| `use_compile` scope error in discriminator | Changed to `self.finetune_cfg.get('torch_compile', False)` | Done |
| Triton not available on Windows | Set `torch_compile: false` by default | Done |

### Config Summary (`finetune_neosr_span_v4.yaml`)

```yaml
data:
  batch_size: 24
  num_workers: 8
  prefetch_factor: 4
  # precomputed_dir: data/precomputed  # Uncomment after running precompute script

training.finetune:
  epochs: 100
  batch_size: 24
  val_interval: 5
  save_interval: 5
  torch_compile: false          # Set true on Linux with Triton
  gradient_checkpointing: true  # Saves ~30% VRAM

  two_phase_training:
    phase1_epochs: 40

  loss.fdl:
    compute_every: 2            # Run FDL every 2nd batch (50% savings)

  early_stopping:
    patience: 15
```

### How to Use Precomputed Dataset (Optional, Maximum Speed)

```bash
# Step 1: Pre-compute LR/HR pairs (one-time, ~10-30 min for 3812 images)
python scripts/precompute_degradation.py --config configs/finetune_neosr_span_v4.yaml --output data/precomputed

# Step 2: Enable in config
# Add to finetune_neosr_span_v4.yaml:
#   data:
#     precomputed_dir: data/precomputed

# Step 3: Train (will skip on-the-fly degradation entirely)
python scripts/train.py --config configs/finetune_neosr_span_v4.yaml
```

### Expected Speed Improvements

| Optimization | Expected Speedup |
|--------------|------------------|
| PyAV compression | 3-5x faster data loading (H.264 path) |
| Buffer reuse (WebP/AVIF) | 10-20% faster compression |
| cudnn.benchmark | 10-20% faster convolutions |
| FDL compute_every=2 | 50% less DINOv2 overhead |
| val_interval 2->5 | 60% less validation time |
| Epochs 150->100 | 33% fewer epochs |
| workers 6/gpu, prefetch=4 | Better GPU utilization |
| Gradient checkpointing | ~30% less VRAM (allows larger batch) |
| Precomputed dataset (optional) | 5-10x faster data loading |
| **Total estimated** | **2-3x faster overall** |

### Files Modified

- `src/data/compression_modules.py` — PyAV + buffer reuse
- `src/data/dataloader.py` — workers, prefetch, precomputed support
- `src/data/precomputed_dataset.py` — NEW
- `src/models/span/neosr_span.py` — gradient checkpointing
- `src/training/neosr_finetuner.py` — torch.compile, cudnn, FDL interval
- `scripts/precompute_degradation.py` — NEW
- `configs/finetune_neosr_span_v4.yaml` — all config changes
