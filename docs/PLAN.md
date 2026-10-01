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
\n\n---\n\n# PLAN: GUI Queue Controls + GPU Decode/Encode (branch perf/queue-controls-gpu-codec)\n\n**Branch**: perf/queue-controls-gpu-codec (cut from efactor/gui-extensibility at 6f7c1f6)\n**Target hardware**: NVIDIA Turing/Ampere/Ada (verified on RTX 4000 8 GB, CUDA 12.6, PyTorch 2.12.0+cu126)\n**Status**: Q1 DONE, Q2-Q5 PLANNED\n\nFull roadmap: docs/ROADMAP_QUEUE_CONTROLS_GPU_CODEC.md. Summary:\n\n| Phase | Title | Status | Files |\n|-------|-------|--------|-------|\n| Q1    | Documentation (this section + ROADMAP + handoff) | **DONE** | PLAN.md, PROJECT_STATUS.md, docs/ROADMAP_QUEUE_CONTROLS_GPU_CODEC.md |\n| Q2    | Pause / Cancel / Resume buttons in GUI | PLANNED | input panel, queue controller, worker, jobs.py |\n| Q3    | GPU decode (NVDEC) + GPU encode (NVENC) wired through worker | PLANNED | decoders.py, ffmpeg.py, worker.py, settings panel |\n| Q4    | TF32 + async default + GPU util telemetry | PLANNED | tensors.py, settings.py, gpu_monitor widget |\n| Q5    | Tests + smoke + commit | PLANNED | test_queue_controls.py, test_gpu_codec.py, test_tf32.py, test_gpu_monitor.py |\n\n## Expected speedups on RTX 4000 (1080p -> 4K)\n\n| Stage                  | Before    | After Q2+Q3+Q4 | Notes |\n|------------------------|-----------|----------------|-------|\n| Decode                 | CPU PyAV  | NVDEC          | ~3x on 1080p H.264 |\n| H2D                    | sync 1x   | async + pinned | overlaps with next decode |\n| Model                  | PyTorch eager fp16, batch=1 | TF32 + auto-TRT fp16 | 2-3x |\n| Encode                 | libx264 CPU pipe | NVENC zero-copy pipe | ~5x, frees CPU |\n\nEnd-to-end on 	mp/mid2s_job.py smoke target: 5-10 fps vs current 0.09 fps.\nRealistic floor (no auto-TRT, just Q2+Q4): 1-2 fps.\n\n## Risks / rollback\n\n- NVDEC codec coverage falls back to PyAV automatically.\n- NVENC preset p1 is fastest but lower quality; users can pick p4.\n- Cancel waits for the next frame boundary; partial output is cleaned up.\n- TF32 is fp32-only; fp16 paths unchanged. Free win on Turing+.\n\n## Out of scope\n\n- Batched video (separate perf/batched-video).\n- CUDA Graphs (separate perf/cuda-graphs).\n- Quality-vs-speed model selection (separate eat/quality-modes).\n
