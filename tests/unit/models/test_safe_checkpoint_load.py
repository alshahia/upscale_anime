"""
Unit tests for NeosrSPANFinetuner.load_checkpoint (PR-4, Issue #4).

The trainer must try `weights_only=True` first and only fall back to pickle
loading if the user has explicitly opted in via `security.allow_pickle_checkpoint`.
This is a security best practice (avoid loading arbitrary code from untrusted
checkpoint files).
"""
import sys
import os
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from pathlib import Path

from anime_sr.training.neosr_finetuner import NeosrSPANFinetuner


# Module-level class so PyTorch's pickler can resolve its qualified name.
class _NeedsPickleConfig:
    """Custom class that cannot be deserialized with weights_only=True."""
    def __init__(self):
        self.special_attr = "needs_pickle"


# ---------------------------------------------------------------------------
# Stub trainer factory
# ---------------------------------------------------------------------------

def _stub_trainer(config_extra: dict, tmp_path: Path) -> NeosrSPANFinetuner:
    """Minimal trainer shell with the attributes load_checkpoint touches."""
    trainer = NeosrSPANFinetuner.__new__(NeosrSPANFinetuner)
    trainer.config = {'training': {'mixed_precision': False}, **config_extra}
    trainer.finetune_cfg = {}
    trainer.checkpoint_manager = None
    trainer.checkpoint_dir = tmp_path
    trainer.device = 'cpu'

    trainer.model = nn.Linear(2, 2)
    trainer.optimizer = optim.AdamW(trainer.model.parameters(), lr=1e-4)
    trainer.scheduler = None
    trainer.scaler = None
    trainer.current_crop_size = 32
    trainer.use_progressive_crop = False
    trainer.phase1_epochs = 30
    trainer.best_loss = float('inf')
    trainer.early_stopping_counter = 0
    trainer.best_metric_value = float('-inf')
    return trainer


def _make_tensor_only_ckpt(path: Path, model: nn.Module, optimizer: optim.Optimizer):
    """Write a checkpoint that contains only tensors (weights-only-safe)."""
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    ckpt = {
        'epoch': 3,
        'model_state_dict': sd,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': None,
        'best_loss': 0.5,
        'early_stopping_counter': 0,
        'best_metric_value': 0.7,
        'config': {},
        'crop_size': 32,
        'progressive_crop_enabled': False,
        'phase1_epochs': 30,
    }
    torch.save(ckpt, path)


def _make_pickle_ckpt(path: Path, model: nn.Module, optimizer: optim.Optimizer):
    """Write a checkpoint that contains a non-tensor (custom class) that
    requires pickle deserialization when loaded with weights_only=True.
    """
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    ckpt = {
        'epoch': 3,
        'model_state_dict': sd,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': None,
        'best_loss': 0.5,
        'early_stopping_counter': 0,
        'best_metric_value': 0.7,
        'config': {'custom': _NeedsPickleConfig()},  # non-tensor; requires pickle
        'crop_size': 32,
        'progressive_crop_enabled': False,
        'phase1_epochs': 30,
    }
    torch.save(ckpt, path)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestLoadCheckpointWeightsOnlySafe:
    def test_load_succeeds_for_tensor_only_checkpoint(self, tmp_path, capsys):
        ckpt_path = tmp_path / "safe.pth"
        # Build a matching trainer, save its state, then build a fresh trainer
        # to load it. This guarantees model/optimizer shape compatibility.
        producer = _stub_trainer({}, tmp_path)
        _make_tensor_only_ckpt(ckpt_path, producer.model, producer.optimizer)
        loader = _stub_trainer({}, tmp_path)
        # Capture producer weights before load; loader has different (random) init
        producer_weights = producer.model.weight.clone()
        producer_bias = producer.model.bias.clone()

        epoch = loader.load_checkpoint(str(ckpt_path))

        # Epoch should be checkpoint.epoch + 1 = 4
        assert epoch == 4
        # Loader's weights must match the saved (producer) values
        assert torch.equal(loader.model.weight, producer_weights)
        assert torch.equal(loader.model.bias, producer_bias)
        captured = capsys.readouterr().out
        assert "Loaded model weights" in captured

    def test_load_does_not_require_opt_in_for_safe_checkpoint(self, tmp_path):
        """Tensor-only checkpoints must load even with no security config."""
        ckpt_path = tmp_path / "safe.pth"
        producer = _stub_trainer({}, tmp_path)
        _make_tensor_only_ckpt(ckpt_path, producer.model, producer.optimizer)
        # No security config at all
        loader = _stub_trainer({}, tmp_path)
        assert 'security' not in loader.config
        epoch = loader.load_checkpoint(str(ckpt_path))
        assert epoch == 4


class TestLoadCheckpointBlocksPickleWithoutOptIn:
    def test_pickle_ckpt_raises_without_opt_in(self, tmp_path):
        ckpt_path = tmp_path / "pickle.pth"
        producer = _stub_trainer({}, tmp_path)
        _make_pickle_ckpt(ckpt_path, producer.model, producer.optimizer)
        trainer = _stub_trainer({}, tmp_path)  # no allow_pickle_checkpoint

        with pytest.raises(RuntimeError, match="allow_pickle_checkpoint"):
            trainer.load_checkpoint(str(ckpt_path))

    def test_pickle_ckpt_raises_explicitly_disabled(self, tmp_path):
        ckpt_path = tmp_path / "pickle.pth"
        producer = _stub_trainer({}, tmp_path)
        _make_pickle_ckpt(ckpt_path, producer.model, producer.optimizer)
        trainer = _stub_trainer(
            {'security': {'allow_pickle_checkpoint': False}},
            tmp_path,
        )
        with pytest.raises(RuntimeError, match="allow_pickle_checkpoint"):
            trainer.load_checkpoint(str(ckpt_path))


class TestLoadCheckpointAllowsPickleWithOptIn:
    def test_pickle_ckpt_loads_with_opt_in(self, tmp_path, recwarn):
        ckpt_path = tmp_path / "pickle.pth"
        producer = _stub_trainer({}, tmp_path)
        _make_pickle_ckpt(ckpt_path, producer.model, producer.optimizer)
        trainer = _stub_trainer(
            {'security': {'allow_pickle_checkpoint': True}},
            tmp_path,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            epoch = trainer.load_checkpoint(str(ckpt_path))

        assert epoch == 4
        # Should have emitted a UserWarning about pickle fallback
        assert any(issubclass(x.category, UserWarning) for x in w), (
            "Expected a UserWarning when loading pickle checkpoint with opt-in"
        )
        # The warning should mention "pickle" or "trusted"
        pickle_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert any("pickle" in str(x.message).lower() or "trust" in str(x.message).lower()
                   for x in pickle_warnings)


class TestLoadCheckpointNonexistent:
    def test_missing_checkpoint_returns_zero(self, tmp_path, capsys):
        trainer = _stub_trainer({}, tmp_path)
        epoch = trainer.load_checkpoint(str(tmp_path / "does_not_exist.pth"))
        assert epoch == 0
        captured = capsys.readouterr().out
        assert "Checkpoint not found" in captured
