# Test Suite

This directory contains the test suite for the anime super-resolution project.

## Structure

```
tests/
├── conftest.py              # Pytest configuration and shared fixtures
├── README.md                # This file
├── unit/                    # Unit tests (fast, isolated)
│   ├── models/              # Model architecture tests
│   ├── data/                # Data pipeline tests
│   ├── losses/              # Loss function tests
│   └── utils/               # Utility function tests
├── integration/             # Integration tests (multi-component)
│   ├── training/            # Training pipeline tests
│   ├── inference/           # Inference pipeline tests
│   └── api/                 # API tests
├── e2e/                     # End-to-end tests (full pipeline)
├── gui/                     # GUI tests (see note below)
└── utils/                   # Shared test utilities
    ├── test_data_manager.py # Test data management
    └── enhanced_dummy_generator.py  # Dummy data generation
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run only unit tests
```bash
pytest tests/unit/
```

### Run only integration tests
```bash
pytest tests/integration/
```

### Run only e2e tests
```bash
pytest tests/e2e/
```

### Run tests by marker
```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# E2E tests only
pytest -m e2e

# Skip slow tests
pytest -m "not slow"

# Skip GPU tests
pytest -m "not gpu"
```

### Run with coverage
```bash
pytest --cov=anime_sr --cov-report=term-missing
```

## Test Categories

### Unit Tests (`tests/unit/`)
Fast, isolated tests that test individual components:
- **models/**: Model architecture correctness, forward/backward passes
- **data/**: Data loading, augmentation, preprocessing
- **losses/**: Loss function correctness
- **utils/**: Utility functions, config management

### Integration Tests (`tests/integration/`)
Tests that verify multi-component interactions:
- **training/**: Training pipeline, callbacks, EMA, schedulers
- **inference/**: Inference engine, TTA, model loading
- **api/**: API endpoints (when applicable)

### End-to-End Tests (`tests/e2e/`)
Full pipeline tests that run complete workflows:
- Data pipeline end-to-end
- Training pipeline end-to-end
- Inference pipeline end-to-end

### GUI Tests (`tests/gui/`)
**Note:** GUI tests are not included in this test suite. The GUI component
(`apps/anime_upscaler_gui/) has its own test directory. GUI tests require
a display environment and are typically run separately.

## Test Data

Tests use a fallback data system defined in `tests/utils/test_data_manager.py`:
1. First tries to use real validation data (`data/val_hr/`)
2. Falls back to real test data (`data/test_hr/`)
3. Creates dummy data if no real data is available

This ensures tests can run without requiring the full dataset.

## Adding New Tests

1. **Choose the right directory**: Place your test in the appropriate category
   directory based on what it tests.

2. **Use proper imports**: Import from `anime_sr.*` (not `from models.*`):
   ```python
   from anime_sr.models.span import SPANModel
   from anime_sr.losses import L1Loss
   ```

3. **Use shared fixtures**: Leverage fixtures from `conftest.py`:
   ```python
   def test_something(test_data_path, small_test_image):
       # test_data_path: path to test images
       # small_test_image: torch tensor (1, 3, 64, 64)
       ...
   ```

4. **Add markers**: Tag your tests with appropriate markers:
   ```python
   @pytest.mark.unit
   def test_model_forward():
       ...
   
   @pytest.mark.slow
   def test_large_training():
       ...
   
   @pytest.mark.gpu
   def test_gpu_training():
       ...
   ```

5. **Clean up**: Use `tmp_path` fixture for temporary files, or rely on
   the automatic cleanup fixture for test data.

## Fixtures

Common fixtures available in `conftest.py`:

| Fixture | Scope | Description |
|---------|-------|-------------|
| `project_root` | session | Project root directory |
| `tests_dir` | session | Tests directory |
| `data_dir` | session | Data directory |
| `configs_dir` | session | Configs directory |
| `test_data_path` | session | Path to test images |
| `val_hr_path` | session | Path to validation HR images |
| `test_hr_path` | session | Path to test HR images |
| `temp_dataset` | function | Temporary dataset for a test |
| `small_test_image` | function | Small test image tensor |
| `tiny_model_config` | function | Minimal model config |
| `gpu_available` | session | Whether GPU is available |
| `mamba_available` | session | Whether mamba-ssm is available |

## Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.unit` | Unit test (fast, isolated) |
| `@pytest.mark.integration` | Integration test (multi-component) |
| `@pytest.mark.e2e` | End-to-end test (full pipeline) |
| `@pytest.mark.gui` | GUI test |
| `@pytest.mark.slow` | Slow test |
| `@pytest.mark.gpu` | Requires GPU |
| `@pytest.mark.mamba` | Requires mamba-ssm |

## Notes

- Tests that require GPU are marked with `@pytest.mark.gpu` and can be
  skipped with `pytest -m "not gpu"`.
- Tests that require mamba-ssm are marked with `@pytest.mark.mamba`.
- The `cleanup_temp_data` fixture runs automatically after each test to
  clean up temporary data.
- Some tests may be skipped if optional dependencies are not installed.
