# Anime Super-Resolution Model Analysis
## Progressive Ensemble Distillation Strategy

**Date**: April 27, 2026  
**Hardware Target**: RTX 4000 Mobile (8GB VRAM)  
**Objective**: Fast training + inference with state-of-the-art anime upscaling quality

---

## Executive Summary

Approved approach: **Progressive Ensemble Distillation**  
Train two complementary models independently, then ensemble as meta-teacher for final lightweight student.

```
Stage 1: Train Model A (NTIRE+MTKD+FAKD) ──┐
                                           ├──► Ensemble ──► Final Student
Stage 2: Train Model B (Mamba-PAN+MTKD+FAKD)┘
```

---

## Model A: NTIRE Architecture + MTKD + FAKD

### Architecture Foundation
- **Base**: SPAN-Tiny (26-32 channels) or SPANF variant
- **Core Innovation**: Parameter-free attention (channel mean + std) + ConvLoRA
- **Parameters**: ~250K-400K
- **Inference**: 10-20× faster than RealESRGAN

### Knowledge Distillation Stack

| Stage | Method | Purpose |
|-------|--------|---------|
| **Stage 1** | MTKD Knowledge Aggregation | Fuse 3+ teachers (EDSR, RCAN, SwinIR) via DCTSwin blocks |
| **Stage 2** | MTKD Wavelet Distillation | Multi-scale learning (DWT: LL, LH, HL, HH bands) |
| **Throughout** | FAKD | Second-order statistics via spatial affinity matrices |

### Loss Function
```
L_total = λ₁·L1(I_stu, I_GT)           # Pixel ground truth
        + λ₂·L_wavelet(I_stu, I_MT)    # MTKD aggregated target  
        + λ₃·L_affinity(F_stu, F_MT)   # FAKD feature correlation
```

### Pros
| Aspect | Benefit |
|--------|---------|
| **Speed** | Fastest inference (0.001-0.005s per 720p frame) |
| **Proven** | EMSR won NTIRE 2025 with this approach |
| **VRAM** | Fits 8GB easily (batch=8-16 with AMP) |
| **Training** | ConvLoRA enables efficient fine-tuning |
| **Stability** | Mature codebase, well-tested |

### Cons
| Aspect | Limitation |
|--------|------------|
| **Context** | Limited long-range modeling vs Mamba |
| **Two-stage** | Must train Knowledge Aggregation first |
| **Teachers** | Requires 3+ pretrained teacher models |

### Training Configuration (8GB VRAM)
```python
config_ntire = {
    'model': 'SPAN-Tiny',
    'channels': 26,              # EMSR winning config
    'teachers': ['EDSR', 'RCAN', 'SwinIR'],
    'batch_size': 8,
    'patch_size': 128,
    'mixed_precision': True,
    'gradient_checkpointing': False,  # Not needed for SPAN
    'stage1_iterations': 30000,  # Knowledge Aggregation
    'stage2_iterations': 50000,  # Student distillation
    'expected_time': '8-12 hours',
    'peak_vram': '~6.5GB',
}
```

### Expected Performance
- **PSNR**: ~32.5 dB on Set5 (×4)
- **SSIM**: ~0.89
- **Speed**: Real-time capable (>200 FPS on 720p)

---

## Model B: Mamba-PAN + MTKD + FAKD

### Architecture Foundation
- **Base**: Hi-Mamba (Hierarchical Mamba Blocks)
- **Core Innovation**: Linear-complexity SSM with 4-direction scanning (H/V/RH/RV)
- **Attention**: Parameter-free Pixel Attention (PAN-style)
- **Parameters**: ~300K-500K
- **Complexity**: O(N) vs O(N²) for transformers

### Modified Distillation Stack

| Stage | Method | Modification for Mamba |
|-------|--------|------------------------|
| **Stage 1** | MTKD Knowledge Aggregation | Same DCTSwin aggregation |
| **Stage 2** | MTKD Wavelet Distillation | Per-direction wavelet loss (4 scans) |
| **Throughout** | FAKD | Direction-aware affinity matrices |
| **New** | Cross-direction Consistency | Ensures coherent 4-directional outputs |

### Loss Function
```
L_total = λ₁·L1(I_stu, I_GT)           
        + λ₂·Σ_d L_wavelet(I_stu^d, I_MT)    # Per-direction (d ∈ {H,V,RH,RV})
        + λ₃·Σ_l Σ_d ||A_stu^d - A_MT||      # Per-direction FAKD
        + λ₄·L_consistency(directions)       # Cross-direction coherence
```

### Pros
| Aspect | Benefit |
|--------|---------|
| **Long-range** | Excellent global context (SSM core) |
| **Edges** | Perfect for anime's sharp directional lines |
| **Scalability** | Linear complexity handles larger images |
| **Novelty** | Paper-worthy: first Mamba+multi-teacher distillation |
| **VRAM** | Lower activation memory than attention |

### Cons
| Aspect | Limitation |
|--------|------------|
| **Speed** | Slower than SPAN (0.003-0.008s per frame) |
| **Stability** | Mamba training can be unstable |
| **Code** | `mamba-ssm` dependency issues; experimental |
| **Setup** | No pretrained anime Mamba; train from scratch |
| **Complexity** | Direction-aware FAKD adds implementation burden |

### Training Configuration (8GB VRAM)
```python
config_mamba = {
    'model': 'Hi-Mamba',
    'channels': 32,
    'directions': ['H', 'V', 'RH', 'RV'],
    'teachers': ['EDSR', 'RCAN', 'SwinIR'],
    'batch_size': 4,              # Lower due to 4-direction overhead
    'patch_size': 128,
    'mixed_precision': True,
    'gradient_checkpointing': True,  # Essential for Mamba
    'stage1_iterations': 30000,
    'stage2_iterations': 60000,   # More iterations for stability
    'expected_time': '12-18 hours',
    'peak_vram': '~7.2GB',
}
```

### Expected Performance
- **PSNR**: ~32.0-32.8 dB (comparable; may exceed on anime artifacts)
- **SSIM**: ~0.89-0.90
- **Speed**: ~120-150 FPS on 720p (slower but acceptable)

---

## Progressive Ensemble Distillation

### Stage 3: Meta-Teacher Fusion

After training Model A and Model B independently:

```python
# Ensemble as meta-teacher
ensemble_output = α * model_span(I) + β * model_mamba(I)
# Typical: α=0.6, β=0.4 (weight toward faster SPAN)

# Final ultra-lightweight student
final_student = TinySR(channels=16, blocks=3)  # ~100K params

# Distill ensemble knowledge
loss = FAKD(final_student, ensemble_output)
```

### Benefits of Ensemble

| Benefit | Mechanism |
|---------|-----------|
| **Speed-Quality Balance** | SPAN speed + Mamba edge quality |
| **Robustness** | Two architectures compensate for each other's failures |
| **Deployment Flexibility** | Can ship both; choose at runtime |
| **Knowledge Transfer** | Final student learns distilled "wisdom" of both |

### Final Student Configuration
```python
config_final = {
    'model': 'TinyStudent',
    'channels': 16,
    'blocks': 3,
    'parameters': '~100K',
    'inference': '0.0005s per 720p frame',
    'training_time': '2-4 hours (distillation only)',
}
```

---

## Training Strategy Options

### Option 1: Train Model A Only (Recommended First)
```
Week 1: Implement SPAN baseline
Week 2: Add MTKD Stage 1 (Knowledge Aggregation)  
Week 3: Add MTKD Stage 2 + FAKD
Week 4: Fine-tune on anime dataset
```

### Option 2: Train Model B Only
```
Week 1: Setup mamba-ssm, implement Hi-Mamba baseline
Week 2: Debug stability, add MTKD Stage 1
Week 3: Add direction-aware MTKD Stage 2 + FAKD
Week 4: Add cross-direction consistency
```

### Option 3: Parallel Training (Both Simultaneously)
```
GPU Schedule:
- Batch A: Model A (Daytime, shorter iterations)
- Batch B: Model B (Overnight, longer iterations)
- Weekend: Ensemble distillation to final student
```

### Option 4: Sequential Ensemble (Full Pipeline)
```
Phase 1 (Weeks 1-3): Train Model A to completion
Phase 2 (Weeks 4-6): Train Model B to completion  
Phase 3 (Week 7): Ensemble + Final student distillation
```

---

## VRAM Budget Analysis (8GB RTX 4000 Mobile)

| Configuration | Peak VRAM | Batch Size | Notes |
|---------------|-----------|------------|-------|
| Model A (SPAN) | 6.5 GB | 8 | Comfortable headroom |
| Model B (Mamba) | 7.2 GB | 4 | Requires gradient checkpointing |
| Both Parallel | 7.5 GB | 4+4 | Alternate batches, not simultaneous forward |
| Ensemble Train | 6.0 GB | 8 | Teachers frozen, only student trains |

---

## Risk Assessment & Mitigation

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Mamba training instability | 30% | Start with NTIRE; Mamba as stretch goal |
| Teacher model availability | 20% | Pre-download EDSR, RCAN, SwinIR weights |
| CUDA compatibility (mamba-ssm) | 40% | Test in Docker first; fallback to NTIRE-only |
| 8GB OOM | 15% | Enable gradient checkpointing; reduce patch size |
| Training time exceeds estimate | 25% | Set iteration limits; use EMA for early stopping |

---

## Decision Log

**Approved**: Progressive Ensemble Distillation  
**Rationale**: Maximizes quality through complementary architectures while maintaining deployment flexibility.

**Primary Path**: Model A (NTIRE) first — proven, fast, lower risk  
**Stretch Goal**: Model B (Mamba-PAN) — pursue if time permits and Mamba stabilizes  
**Ensemble**: Always viable once both models trained

---

## Next Steps

1. **Immediate**: Implement SPAN baseline (Model A)
2. **Short-term**: Integrate MTKD Stage 1 (Knowledge Aggregation)
3. **Medium-term**: Add FAKD feature distillation
4. **Long-term**: Experiment with Mamba-PAN variant
5. **Final**: Ensemble distillation to ultra-lightweight student

---

*Analysis completed. Ready for implementation.*
