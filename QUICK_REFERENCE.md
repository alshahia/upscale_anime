# Quick Reference: Small Dataset SR Techniques

## 🚀 One-Line Commands

### Dataset Preparation
```bash
# Validate and clean dataset
python scripts/validate_dataset.py --data-dir data/anime_hr --remove-duplicates --remove-invalid
```

### Full Automated Pipeline
```bash
# Run complete pipeline (validation → pre-train → fine-tune → benchmark → export)
python scripts/run_full_pipeline.py --config pipeline_config.json

# Skip pre-training (use existing checkpoint)
python scripts/run_full_pipeline.py --no-pretrain

# Only validate and benchmark
python scripts/run_full_pipeline.py --no-pretrain --no-finetune --no-export
```

### Experiment Comparison
```bash
# Compare multiple training runs
python scripts/compare_experiments.py \
  --experiments "pretrain_baseline,pretrain_augmented,pretrain_distillation" \
  --visualize \
  --output comparison_report.md
```

### Training Pipeline
```bash
# Phase 1: Self-supervised pre-training (100 epochs)
python scripts/train.py --config configs/pretrain_selfsupervised.yaml --name pretrain

# Phase 2: Transfer learning fine-tuning (200 epochs)
python scripts/train.py --config configs/finetune_transfer.yaml --name finetune
```

### Meta-Learning (Alternative Approach)
```bash
# MAML-style meta-learning
python scripts/train_meta.py --data-dir data/anime_hr --epochs 100 --tasks-per-batch 16
```

### Test-Time Adaptation
```bash
# Adapt model to specific test image
python scripts/test_time_adapt.py \
  --input test_image.png \
  --output sr_output.png \
  --checkpoint checkpoints/best_model.pth \
  --adaptation-steps 50
```

### Benchmarking
```bash
# Single model benchmark
python scripts/benchmark_model.py \
  --checkpoint checkpoints/model.pth \
  --lr-dir data/test_lr \
  --hr-dir data/test_hr

# Compare multiple models
python scripts/benchmark_model.py \
  --checkpoint "model1.pth,model2.pth,model3.pth" \
  --lr-dir data/test_lr \
  --hr-dir data/test_hr \
  --compare
```

### Production Export
```bash
# Export all formats
python scripts/export_model.py \
  --checkpoint checkpoints/best_model.pth \
  --output-dir production_models
```

---

## 📊 Expected Results with 5000+ Images

| Technique | PSNR Gain | Training Time |
|-----------|-----------|---------------|
| Baseline | ~32 dB | 3 days |
| + Pre-training | +1.5 dB | +1 day |
| + Transfer learning | +1.0 dB | +2 days |
| + Feature distillation | +0.5 dB | - |
| + Advanced augmentation | +0.5 dB | - |
| **All Combined** | **~36 dB** | **~1 week** |

---

## 🔧 Configuration Quick Reference

### Enable Progressive Crop Sizing
```yaml
training:
  progressive_crop:
    enabled: true
    sizes: [64, 96, 128, 160]
    epochs_per_size: 25
```

### Enable Layer Freezing
```yaml
training:
  freeze_strategy:
    enabled: true
    freeze_layers:
      - type: blocks
        count: 4
    unfreeze_schedule:
      - epoch: 20
        unfreeze_blocks: 1
        lr_multiplier: 0.5
```

### Enable Feature Distillation
```yaml
training:
  stage1:
    feature_distillation:
      enabled: true
      layers: [2, 4, 6, 8]
      weight: 0.1
```

### Enable Mixup/CutMix
```yaml
training:
  augmentation:
    enabled: true
    mixup_prob: 0.3
    cutmix_prob: 0.3
```

---

## 📁 File Locations

| Component | Path |
|-----------|------|
| Pre-training config | `configs/pretrain_selfsupervised.yaml` |
| Fine-tuning config | `configs/finetune_transfer.yaml` |
| Pipeline config example | `pipeline_config.example.json` |
| Dataset validation | `scripts/validate_dataset.py` |
| Full pipeline runner | `scripts/run_full_pipeline.py` |
| Meta-learning | `scripts/train_meta.py` |
| Test-time adaptation | `scripts/test_time_adapt.py` |
| Benchmarking | `scripts/benchmark_model.py` |
| Experiment comparison | `scripts/compare_experiments.py` |
| Export | `scripts/export_model.py` |
| Augmentation | `src/data/augmentation.py` |
| Feature distillation | `src/training/feature_distillation.py` |
| New callbacks | `src/training/callbacks.py` |

---

## 📚 Documentation

- **Full Guide:** `SMALL_DATASET_TECHNIQUES_GUIDE.md`
- **Implementation Summary:** `IMPLEMENTATION_SUMMARY.md`
- **Original README:** `README.md`

---

## 🐛 Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Use `gradient_accumulation_steps: 2`
- Start with smaller `crop_size`

### Slow Training
- Increase `num_workers` (default: 4)
- Enable `mixed_precision: true`
- Use `preload: true` for small datasets

### Poor Results
1. Check dataset quality: `python scripts/validate_dataset.py --data-dir data/anime_hr`
2. Verify teacher models downloaded
3. Check `logs/` for training issues
4. Try test-time adaptation for specific images

---

## ✅ Verification Checklist

- [ ] Dataset validated and cleaned
- [ ] Teacher models downloaded (`pretrained/` directory)
- [ ] Pre-training completed (100 epochs)
- [ ] Fine-tuning completed (200 epochs)
- [ ] Benchmarked on test set
- [ ] Exported for production

**Ready to train! 🎉**
