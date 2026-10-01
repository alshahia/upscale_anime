"""
Tests for the WP-5 configuration management system:
  - load_config with base: inheritance and DSH_* env overrides
  - merge_configs / get_config_value / validate_config
  - ConfigValidationError
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

for _cached in ("utils", "utils.config"):
    sys.modules.pop(_cached, None)

from anime_sr.utils.config import (  # noqa: E402
    Config,
    ConfigValidationError,
    apply_env_overrides,
    get_config_value,
    load_config,
    merge_configs,
    validate_config,
)

# ------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------
@pytest.fixture
def default_config():
    """The canonical default config, loaded without env overrides."""
    return load_config(PROJECT_ROOT / "configs" / "default.yaml", apply_env=False)

# ------------------------------------------------------------
# load_config: plain YAML
# ------------------------------------------------------------
class TestLoadConfigPlain:
    def test_plain_yaml_returns_dict(self, tmp_path):
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

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "does_not_exist.yaml")

    def test_empty_yaml_returns_empty_dict(self, tmp_path):
        cfg_path = tmp_path / "empty.yaml"
        cfg_path.write_text("")
        assert load_config(cfg_path) == {}

# ------------------------------------------------------------
# load_config: base: inheritance
# ------------------------------------------------------------
class TestLoadConfigInheritance:
    def test_base_inheritance_merges(self, tmp_path):
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

        assert result["model"]["name"] == "override_model"
        assert result["model"]["scale"] == 2
        assert result["model"]["channels"] == 32
        assert result["training"]["epochs"] == 5
        assert result["training"]["batch_size"] == 8
        assert "base" not in result

    def test_base_path_resolved_relative_to_child(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "root.yaml").write_text("model:\n  scale: 3\n")
        (tmp_path / "sub" / "child.yaml").write_text(
            "base: ../root.yaml\nmodel:\n  name: child\n"
        )
        result = load_config(tmp_path / "sub" / "child.yaml")
        assert result == {"model": {"scale": 3, "name": "child"}}

    def test_inheritance_chain(self, tmp_path):
        (tmp_path / "a.yaml").write_text("model:\n  scale: 2\n  x: 1\n")
        (tmp_path / "b.yaml").write_text("base: a.yaml\nmodel:\n  y: 2\n")
        (tmp_path / "c.yaml").write_text("base: b.yaml\nmodel:\n  z: 3\n")
        result = load_config(tmp_path / "c.yaml")
        assert result == {"model": {"scale": 2, "x": 1, "y": 2, "z": 3}}

    def test_child_list_replaces_base_list(self, tmp_path):
        (tmp_path / "base.yaml").write_text("data:\n  betas: [0.9, 0.99]\n")
        (tmp_path / "child.yaml").write_text("base: base.yaml\ndata:\n  betas: [0.8]\n")
        result = load_config(tmp_path / "child.yaml")
        assert result["data"]["betas"] == [0.8]

# ------------------------------------------------------------
# Project configs: inheritance from configs/default.yaml
# ------------------------------------------------------------
class TestProjectConfigs:
    TRAINING_CONFIGS = [
        "configs/training/model_a.yaml",
        "configs/training/model_b.yaml",
        "configs/training/ensemble.yaml",
        "configs/training/distill.yaml",
    ]

    @pytest.mark.parametrize("rel_path", TRAINING_CONFIGS)
    def test_training_config_inherits_defaults(self, rel_path):
        config = load_config(PROJECT_ROOT / rel_path, apply_env=False)

        # Inherited from configs/default.yaml
        assert config["model"]["scale"] == 4
        assert config["training"]["optimizer"] == "adam"
        assert config["logging"]["level"] == "INFO"
        assert "base" not in config

    def test_model_a_overrides(self):
        config = load_config(
            PROJECT_ROOT / "configs" / "training" / "model_a.yaml", apply_env=False
        )
        assert config["model"]["num_features"] == 26
        assert config["model"]["use_lora"] is True
        assert config["training"]["mode"] == "model_a"
        assert config["training"]["stage1"]["teachers"][0]["name"] == "edsr"

    def test_model_b_overrides(self):
        config = load_config(
            PROJECT_ROOT / "configs" / "training" / "model_b.yaml", apply_env=False
        )
        assert config["model"]["type"] == "mamba_pan"
        assert config["model"]["directions"] == ["h", "v", "rh", "rv"]
        assert config["training"]["batch_size"] == 4

    def test_ensemble_overrides(self):
        config = load_config(
            PROJECT_ROOT / "configs" / "training" / "ensemble.yaml", apply_env=False
        )
        assert config["model"]["type"] == "tiny_student"
        assert config["ensemble"]["weights"] == [0.6, 0.4]

    def test_inference_config(self):
        config = load_config(
            PROJECT_ROOT / "configs" / "inference" / "default.yaml", apply_env=False
        )
        assert config["inference"]["tile_size"] == 512
        assert config["inference"]["tta"] is False
        assert config["model"]["scale"] == 4  # inherited

    def test_api_config(self):
        config = load_config(
            PROJECT_ROOT / "configs" / "api" / "default.yaml", apply_env=False
        )
        assert config["api"]["port"] == 8000
        assert config["api"]["max_file_size_mb"] == 50
        assert config["logging"]["file"] == "api.log"

# ------------------------------------------------------------
# merge_configs
# ------------------------------------------------------------
class TestMergeConfigs:
    def test_deep_merge_dicts(self):
        base = {"a": {"x": 1, "y": 2}, "keep": True}
        override = {"a": {"y": 3, "z": 4}}
        assert merge_configs(base, override) == {
            "a": {"x": 1, "y": 3, "z": 4},
            "keep": True,
        }

    def test_override_replaces_scalars_and_lists(self):
        base = {"n": 1, "l": [1, 2], "s": "a"}
        override = {"n": 2, "l": [9], "s": "b"}
        assert merge_configs(base, override) == {"n": 2, "l": [9], "s": "b"}

    def test_inputs_not_mutated(self):
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}}
        merge_configs(base, override)
        assert base == {"a": {"x": 1}}
        assert override == {"a": {"y": 2}}

    def test_none_arguments(self):
        assert merge_configs(None, {"a": 1}) == {"a": 1}
        assert merge_configs({"a": 1}, None) == {"a": 1}
        assert merge_configs(None, None) == {}

    def test_non_dict_raises(self):
        with pytest.raises(TypeError):
            merge_configs({"a": 1}, [1, 2])

# ------------------------------------------------------------
# get_config_value
# ------------------------------------------------------------
class TestGetConfigValue:
    def test_dot_notation(self, default_config):
        assert get_config_value(default_config, "model.scale") == 4
        assert get_config_value(default_config, "training.early_stopping.patience") == 15

    def test_list_index_access(self, default_config):
        assert get_config_value(default_config, "data.datasets.0.name") == "div2k"
        assert get_config_value(default_config, "training.betas.1") == 0.99

    def test_default_on_missing(self, default_config):
        assert get_config_value(default_config, "no.such.key") is None
        assert get_config_value(default_config, "no.such.key", "dflt") == "dflt"
        assert get_config_value(default_config, "model.name", "dflt") == "span"

    def test_none_config_returns_default(self):
        assert get_config_value(None, "model.scale", 4) == 4

# ------------------------------------------------------------
# validate_config
# ------------------------------------------------------------
class TestValidateConfig:
    def test_default_config_is_valid(self, default_config):
        assert validate_config(default_config) is True

    def test_custom_schema(self):
        schema = {
            "model.scale": {"type": int, "required": True, "range": (2, 4)},
            "training.optimizer": {"type": str, "choices": ["adam", "adamw"]},
        }
        assert validate_config(
            {"model": {"scale": 4}, "training": {"optimizer": "adam"}}, schema
        ) is True

    def test_missing_required_raises(self):
        with pytest.raises(ConfigValidationError, match="missing required"):
            validate_config({"model": {"scale": 4}}, {
                "model.name": {"type": str, "required": True},
                "model.scale": {"type": int, "required": True},
            })

    def test_range_violation_raises(self):
        with pytest.raises(ConfigValidationError, match="outside valid range"):
            validate_config({"model": {"scale": 7}}, {
                "model.scale": {"type": int, "range": (2, 4)},
            })

    def test_choices_violation_raises(self):
        with pytest.raises(ConfigValidationError, match="not in allowed choices"):
            validate_config(
                {"training": {"optimizer": "rmsprop"}},
                {"training.optimizer": {"type": str, "choices": ["adam", "adamw"]}},
            )

    def test_type_mismatch_raises(self):
        with pytest.raises(ConfigValidationError, match="expected type"):
            validate_config(
                {"model": {"scale": "four"}},
                {"model.scale": {"type": int}},
            )

    def test_bool_not_accepted_as_int(self):
        with pytest.raises(ConfigValidationError, match="expected type"):
            validate_config(
                {"model": {"scale": True}},
                {"model.scale": {"type": int}},
            )

    def test_error_message_lists_all_problems(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(
                {"model": {"scale": 9}, "training": {"batch_size": -1}},
                {
                    "model.scale": {"type": int, "range": (2, 4)},
                    "training.batch_size": {"type": int, "range": (1, 4096)},
                },
            )
        msg = str(exc_info.value)
        assert "model.scale" in msg
        assert "training.batch_size" in msg
        assert "2 error(s)" in msg

    def test_none_value_skipped_when_not_required(self):
        assert validate_config(
            {"model": {"scale": 4}},
            {"model.pretrained_path": {"type": (str, type(None)), "default": None}},
        ) is True

# ------------------------------------------------------------
# apply_env_overrides

# ------------------------------------------------------------
class TestApplyEnvOverrides:
    def test_explicit_mapping(self, default_config):
        applied = apply_env_overrides(
            default_config, env={"DSH_MODEL_PATH": "/tmp/model.pth"}
        )
        assert default_config["model"]["path"] == "/tmp/model.pth"
        assert applied == {"model.path": "/tmp/model.pth"}

    def test_generic_dotted_mapping(self, default_config):
        apply_env_overrides(
            default_config,
            env={"DSH_TRAINING__BATCH_SIZE": "16", "DSH_MODEL__SCALE": "2"},
        )
        assert default_config["training"]["batch_size"] == 16
        assert default_config["model"]["scale"] == 2

    def test_type_coercion_matches_existing_value(self, default_config):
        apply_env_overrides(
            default_config,
            env={
                "DSH_TRAINING__BATCH_SIZE": "16",
                "DSH_TRAINING__MIXED_PRECISION": "false",
                "DSH_TRAINING__BETAS": "0.8,0.95",
            },
        )
        assert default_config["training"]["batch_size"] == 16
        assert isinstance(default_config["training"]["batch_size"], int)
        assert default_config["training"]["mixed_precision"] is False
        assert isinstance(default_config["training"]["mixed_precision"], bool)
        assert default_config["training"]["betas"] == ["0.8", "0.95"]

    def test_unknown_generic_key_ignored(self, default_config):
        applied = apply_env_overrides(
            default_config, env={"DSH_NOPE__NOT_A_KEY": "x"}
        )
        assert applied == {}

    def test_non_prefixed_env_ignored(self, default_config):
        applied = apply_env_overrides(
            default_config, env={"PATH": "/usr/bin", "HOME": "/root"}
        )
        assert applied == {}

    def test_env_override_via_load_config(self, monkeypatch):
        # load_config applies DSH_* env vars from os.environ by default
        monkeypatch.setenv("DSH_LOG_LEVEL", "DEBUG")
        config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
        assert config["logging"]["level"] == "DEBUG"

    def test_apply_env_false_disables(self, monkeypatch):
        monkeypatch.setenv("DSH_LOG_LEVEL", "DEBUG")
        config = load_config(
            PROJECT_ROOT / "configs" / "default.yaml", apply_env=False
        )
        assert config["logging"]["level"] == "INFO"

    def test_non_dict_raises(self):
        with pytest.raises(TypeError):
            apply_env_overrides([1, 2], env={})

# ------------------------------------------------------------
# Config class integration

# ------------------------------------------------------------
class TestConfigClass:
    def test_load_with_inheritance(self):
        cfg = Config.load(
            PROJECT_ROOT / "configs" / "training" / "model_a.yaml",
            apply_env=False,
        )
        assert cfg.get("model.num_features") == 26
        assert cfg.get("model.scale") == 4  # inherited

    def test_get_set_dot_notation(self):
        cfg = Config({"model": {"scale": 4}})
        cfg.set("training.batch_size", 16)
        assert cfg.get("training.batch_size") == 16
        assert cfg["training.batch_size"] == 16

    def test_to_dict_roundtrip(self, tmp_path):
        cfg = Config.load(
            PROJECT_ROOT / "configs" / "default.yaml", apply_env=False
        )
        out = tmp_path / "out.yaml"
        cfg.save(out)
        reloaded = load_config(out)
        assert reloaded == cfg.to_dict()

    def test_validate_method_still_works(self):
        cfg = Config.load(
            PROJECT_ROOT / "configs" / "default.yaml", apply_env=False
        )
        assert cfg.validate() is True

# ------------------------------------------------------------
# ConfigValidationError type

# ------------------------------------------------------------
class TestConfigValidationError:
    def test_is_value_error_subclass(self):
        assert issubclass(ConfigValidationError, ValueError)

    def test_caught_as_value_error(self):
        with pytest.raises(ValueError):
            validate_config({"model": {"scale": 9}}, {
                "model.scale": {"type": int, "range": (2, 4)},
            })
