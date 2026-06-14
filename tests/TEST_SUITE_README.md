# Comprehensive Test Suite Documentation

This test suite provides complete coverage for all phases, stages, and features of the Anime Super-Resolution project with real data validation and fallback mechanisms.

## Quick Start

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_dataset_validation.py -v

# Run with data source report
python -m pytest tests/test_dataset_validation.py::test_data_source_report -v -s

# Run slow tests (E2E tests)
python -m pytest tests/ -v --runslow

# Run only unit tests (fast)
python -m pytest tests/ -v -m "not slow"
```

## Data Strategy

### Real Data Sources
- **Validation Data**: `data/val_hr/` (4 images)
- **Test Data**: `data/test_hr/` (4 images)
- **Total Real Images**: 8 unique images

### Fallback Mechanism
If real data is not available, tests automatically create dummy images using:
- `tests/utils/test_data_manager.py` - Central data management utility
- Patterns: gradient, noise, checkerboard, anime_style, natural
- Sizes: 256x256 (configurable)

### Data Usage Report
Each test suite reports which data it's using:
```python
# In any test file
from tests.utils.test_data_manager import print_data_summary
print_data_summary()
```

## Test Suite Structure

### Phase 1: Missing Critical Tests (NEW)

| Test File | Purpose | Data Used | Run Time |
|-----------|---------|-----------|----------|
| `test_dataset_validation.py` | Dataset cleaning, duplicate detection, blur detection | data/val_hr/ (4 imgs) | ~10s |
| `test_video_processing.py` | Video extraction, temporal windows | data/anime_vid/ or synthetic | ~30s |
| `test_model_export.py` | ONNX, TorchScript, quantized export | Mock checkpoint | ~20s |
| `test_benchmarking.py` | PSNR, SSIM, LPIPS, DISTS, FPS | data/test_hr/ (4 imgs) | ~15s |
| `test_meta_learning.py` | MAML inner/outer loop, fast adaptation | Synthetic tasks | ~30s |
| `test_test_time_adaptation.py` | Per-image TTA, online learning | Single image | ~20s |

### Phase 2: Enhanced Existing Tests

| Test File | Purpose | Data Used | Run Time |
|-----------|---------|-----------|----------|
| `test_stage_automation_enhanced.py` | Real data stage transitions | data/val_hr/ | ~60s |
| `test_early_stopping_enhanced.py` | Mini training convergence | data/val_hr/ | ~45s |
| `test_augmentation_enhanced.py` | Mixup/CutMix on real images | data/val_hr/ | ~15s |
| `test_aggregation_enhanced.py` | Teacher aggregation on real data | data/val_hr/ | ~30s |

### Phase 3: End-to-End Integration Tests

| Test File | Purpose | Data Used | Run Time |
|-----------|---------|-----------|----------|
| `test_e2e_training_pipeline.py` | Full Stage 1→2 training | data/val_hr/ (4 imgs) | ~5min |
| `test_e2e_inference_pipeline.py` | Single/batch/TTA/EMA inference | data/test_hr/ (4 imgs) | ~30s |
| `test_e2e_data_pipeline.py` | Loading → degradation → batches | data/val_hr/ (4 imgs) | ~20s |
| `test_e2e_small_dataset_pipeline.py` | All 10 phases end-to-end | data/val_hr/ (4 imgs) | ~10min |

## Test Configuration

### Test Markers
- `unit` - Fast unit tests (no data needed)
- `integration` - Component integration tests
- `e2e` - End-to-end tests with real data
- `slow` - Tests > 30 seconds
- `requires_data` - Needs real image data
- `requires_gpu` - Needs CUDA
- `requires_teachers` - Needs teacher models

### Running with Markers
```bash
# Run only unit tests
python -m pytest tests/ -v -m "unit"

# Run E2E tests
python -m pytest tests/ -v -m "e2e"

# Skip slow tests
python -m pytest tests/ -v -m "not slow"

# Run tests that need real data
python -m pytest tests/ -v -m "requires_data"
```

## Utility Modules

### test_data_manager.py
Central utility for test data management:

```python
from tests.utils.test_data_manager import (
    ensure_test_data,      # Get or create test data
    get_val_hr_path,       # Get validation data path
    get_test_hr_path,      # Get test data path
    create_temp_dataset,   # Create temporary dataset
    cleanup_temp_data,     # Clean up temporary data
    print_data_summary     # Show data availability
)

# Ensure data exists
data_dir = ensure_test_data(min_images=5, pattern='gradient')

# Create fallback data
temp_dir = create_temp_dataset(num_images=5, pattern='anime_style')

# Report data usage
print_data_summary()
```

### Dummy Image Patterns
- `gradient` - Smooth gradients (default)
- `noise` - Random texture
- `checkerboard` - Sharp edges
- `anime_style` - Simple line art + flat colors
- `natural` - Perlin noise texture

## Individual Test Suites

### Dataset Validation Tests
```bash
python -m pytest tests/test_dataset_validation.py -v
```

Tests:
- Perceptual hash duplicate detection
- Blur detection (Laplacian variance)
- Resolution validation
- Compression artifact estimation
- Validation report generation
- Clean dataset output

### Video Processing Tests
```bash
python -m pytest tests/test_video_processing.py -v
```

Tests:
- Video metadata extraction
- Pre-extraction mode
- On-demand extraction
- Quality threshold filtering
- Duplicate frame removal
- Temporal window creation
- Scene change detection

### Model Export Tests
```bash
python -m pytest tests/test_model_export.py -v
```

Tests:
- TorchScript export
- ONNX export
- Quantized model export
- Export compatibility
- File size comparison
- Inference with exported models

### Benchmarking Tests
```bash
python -m pytest tests/test_benchmarking.py -v
```

Tests:
- PSNR calculation
- SSIM calculation
- LPIPS calculation
- DISTS calculation
- Inference speed measurement
- Memory usage tracking
- Report generation

### Meta-Learning Tests
```bash
python -m pytest tests/test_meta_learning.py -v
```

Tests:
- MAML initialization
- Inner loop adaptation
- Outer loop meta-update
- Fast adaptation on few samples
- Meta-gradient computation
- Checkpoint save/load

### Test-Time Adaptation Tests
```bash
python -m pytest tests/test_test_time_adaptation.py -v
```

Tests:
- TTA initialization
- Online learning loop
- Reconstruction loss
- Convergence detection
- Checkpointing
- Quality improvement

## E2E Integration Tests

### Training Pipeline E2E
```bash
python -m pytest tests/test_e2e_training_pipeline.py -v --runslow
```

Tests complete training workflows:
- Stage 1 mini training (3 epochs)
- Stage 2 mini training (3 epochs)
- Auto-stage pipeline
- Model A/B training
- Checkpoint save/resume

### Inference Pipeline E2E
```bash
python -m pytest tests/test_e2e_inference_pipeline.py -v
```

Tests complete inference workflows:
- Single image inference
- Batch inference
- TTA inference
- EMA model inference
- Output quality validation
- Speed benchmarking

### Small Dataset Pipeline E2E
```bash
python -m pytest tests/test_e2e_small_dataset_pipeline.py -v --runslow
```

Tests all 10 phases:
1. Dataset validation
2. Self-supervised pre-training
3. Transfer learning fine-tuning
4. Advanced augmentation
5. Meta-learning (MAML)
6. Test-time adaptation
7. Feature distillation
8. SWA/EMA
9. Benchmarking
10. Production export

## Continuous Integration

### GitHub Actions / CI Setup
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest
      - name: Run tests
        run: |
          python -m pytest tests/ -v -m "not slow and not requires_gpu"
```

## Troubleshooting

### Tests Failing Due to Missing Data
Tests automatically create fallback data. If you want to use real data:
```bash
# Ensure real data exists
ls data/val_hr/
ls data/test_hr/

# If missing, check if data was extracted from videos
python scripts/process_videos.py extract --input data/anime_vid --output data/anime_video_frames
```

### Tests Failing Due to Missing Modules
Some tests skip if modules aren't available:
```python
# In test output:
SKIPPED [1] tests/test_video_processing.py:15: Video processing module not available
```

This is expected behavior - not all features may be installed.

### Slow Tests Timing Out
Run without slow tests:
```bash
python -m pytest tests/ -v -m "not slow"
```

Or increase timeout:
```bash
python -m pytest tests/ -v --timeout=300
```

## Coverage Report

To generate coverage report:
```bash
pip install pytest-cov
python -m pytest tests/ --cov=src --cov-report=html
```

View report: `htmlcov/index.html`

## Adding New Tests

1. Create test file: `tests/test_<feature>.py`
2. Import test data manager: `from tests.utils.test_data_manager import ensure_test_data`
3. Use fixtures for setup/teardown
4. Add appropriate markers: `@pytest.mark.slow`, `@pytest.mark.e2e`
5. Run and verify: `python -m pytest tests/test_<feature>.py -v`

## Test Maintenance

- Update tests when adding new features
- Keep fallback data generation updated
- Monitor test execution times
- Review skipped tests periodically

---

**Last Updated**: May 2026
**Test Files**: 25+
**Total Tests**: 200+
**Coverage**: All 7 training modes, 10 small dataset phases
