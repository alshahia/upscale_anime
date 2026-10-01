"""
Pytest configuration and shared fixtures for the anime_sr test suite.

This module provides:
- Path setup for test utilities
- Common fixtures (test data, model configs, temp directories)
- Pytest markers for test categorization
"""

import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
# Ensure the tests/ directory is on sys.path so that test utilities
# (tests/utils/) can be imported as `from utils.test_data_manager import ...`.
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

# Ensure the project root is on sys.path so that `anime_sr` is importable
# even if the package is not installed in editable mode.
PROJECT_ROOT = TESTS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------------------------
# Collect ignore - fundamentally broken tests
# ---------------------------------------------------------------------------
# test_topiq_metric.py crashes the test runner (pyiqa segfault in native code).
# The crash happens at import time, before pytest's skip mechanism can engage.
collect_ignore = ["unit/utils/test_topiq_metric.py"]

# ---------------------------------------------------------------------------
# Pytest markers
# ---------------------------------------------------------------------------
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: marks tests as unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: marks tests as integration tests (multi-component)")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests (full pipeline)")
    config.addinivalue_line("markers", "gui: marks tests as GUI tests")
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with -m 'not slow')")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "mamba: marks tests that require mamba-ssm")


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def tests_dir():
    """Return the tests directory."""
    return TESTS_DIR


@pytest.fixture(scope="session")
def data_dir():
    """Return the data directory path."""
    return PROJECT_ROOT / "data"


@pytest.fixture(scope="session")
def configs_dir():
    """Return the configs directory path."""
    return PROJECT_ROOT / "configs"


@pytest.fixture(scope="session")
def temp_data_dir():
    """Return the temporary test data directory path."""
    return TESTS_DIR / "temp_test_data"


# ---------------------------------------------------------------------------
# Test data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_data_path():
    """Ensure test data exists and return the path.

    Uses real validation data if available, otherwise creates dummy data.
    """
    from utils.test_data_manager import ensure_test_data
    return ensure_test_data(min_images=4, verbose=False)


@pytest.fixture(scope="session")
def val_hr_path():
    """Return path to validation HR images (real or fallback)."""
    from utils.test_data_manager import get_val_hr_path
    return get_val_hr_path()


@pytest.fixture(scope="session")
def test_hr_path():
    """Return path to test HR images (real or fallback)."""
    from utils.test_data_manager import get_test_hr_path
    return get_test_hr_path()


@pytest.fixture
def temp_dataset(tmp_path):
    """Create a temporary dataset for a specific test.

    Returns a path to a directory with dummy images.
    """
    from utils.test_data_manager import create_fallback_data
    return create_fallback_data(
        num_images=3,
        image_size=(64, 64),
        pattern='gradient',
        verbose=False,
        output_dir=tmp_path / "test_data",
    )


@pytest.fixture
def small_test_image():
    """Create a small test image tensor.

    Returns a torch tensor of shape (1, 3, 64, 64).
    """
    import torch
    torch.manual_seed(42)
    return torch.randn(1, 3, 64, 64)


@pytest.fixture
def tiny_model_config():
    """Return a minimal model configuration for testing."""
    return {
        'model': {
            'name': 'span',
            'scale': 4,
            'in_channels': 3,
            'out_channels': 3,
            'num_features': 32,
            'num_blocks': 4,
        },
        'training': {
            'batch_size': 1,
            'num_epochs': 1,
            'learning_rate': 1e-4,
        },
    }


# ---------------------------------------------------------------------------
# Cleanup fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def cleanup_temp_data():
    """Automatically clean up temporary test data after each test."""
    yield
    from utils.test_data_manager import cleanup_temp_data as _cleanup
    _cleanup(verbose=False)


# ---------------------------------------------------------------------------
# GPU availability
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture(scope="session")
def mamba_available():
    """Check if mamba-ssm is available."""
    try:
        import mamba_ssm
        return True
    except ImportError:
        return False
