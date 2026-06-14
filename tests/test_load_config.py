"""
Tests for the standalone load_config() function in src/utils/config.py.

This function is used by:
  - tests/integration/test_full_pipeline.py
  - scripts/precompute_degradation.py
  - scripts/debug_transfer_learning.py
  - scripts/manage_stages.py

It returns a plain dict (NOT a Config instance) and supports base: inheritance
via recursive merge of nested dicts.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Invalidate any cached 'utils' module so pytest's collection order
# (where some other test file may have already imported a different utils)
# does not shadow the project's src/utils package.
for _cached in ("utils", "utils.config"):
    sys.modules.pop(_cached, None)

from utils.config import load_config  # noqa: E402


class TestLoadConfig:
    """Tests for the module-level load_config helper."""

    def test_load_config_plain_yaml_returns_dict(self, tmp_path):
        """A YAML file without a `base:` key is returned as a dict."""
        cfg_path = tmp_path / "plain.yaml"
        cfg_path.write_text(
            "model:\n"
            "  name: tiny\n"
            "  scale: 4\n"
            "training:\n"
            "  epochs: 10\n"
            "  batch_size: 4\n"
        )
        result = load_config(cfg_path)

        assert isinstance(result, dict)
        assert result == {
            "model": {"name": "tiny", "scale": 4},
            "training": {"epochs": 10, "batch_size": 4},
        }

    def test_load_config_with_base_inheritance(self, tmp_path):
        """A YAML file with a `base:` key is merged with the base file."""
        base_path = tmp_path / "base.yaml"
        base_path.write_text(
            "model:\n"
            "  name: base_model\n"
            "  scale: 2\n"
            "  channels: 32\n"
            "training:\n"
            "  epochs: 100\n"
            "  batch_size: 8\n"
        )
        override_path = tmp_path / "override.yaml"
        override_path.write_text(
            "base: base.yaml\n"
            "model:\n"
            "  name: override_model\n"
            "training:\n"
            "  epochs: 5\n"
        )
        result = load_config(override_path)

        assert isinstance(result, dict)
        assert result["model"]["name"] == "override_model"
        assert result["model"]["scale"] == 2
        assert result["model"]["channels"] == 32
        assert result["training"]["epochs"] == 5
        assert result["training"]["batch_size"] == 8
        assert "base" not in result

    def test_load_config_missing_path_raises(self, tmp_path):
        """A missing path raises FileNotFoundError."""
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError):
            load_config(missing)
