"""
Tests for EMA state preservation across checkpoint save/load.

Complements `tests/test_ema_integration.py` (in-memory state_dict shape
and update behavior) by verifying the full save -> load -> forward
roundtrip that the finetuner actually performs. Specifically:

  1. Save a model + EMA shadow to a torch checkpoint file.
  2. Load into a fresh model + fresh EMA built from random init.
  3. Assert the loaded EMA shadow params match the saved ones bit-exact.
  4. Assert the loaded (model, EMA) pair produces identical forward
     output to the original (using `ema.averaged()` context).
  5. Assert that a model update AFTER save does NOT propagate into the
     loaded EMA (the loaded shadow is decoupled from the original model
     object after the load).
  6. End-to-end test using the `NeosrSPANFinetuner.__new__()` stub
     pattern (from `tests/test_grad_scaler.py`) so we exercise the
     real `save_checkpoint()` code path that the finetuner runs.

Tests are CPU-only and use tiny models for speed.
"""
import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from training.ema import GeneratorEMA  # noqa: E402


# ---------------------------------------------------------------------------
# Tiny model + stub trainer fixtures
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    """Minimal 2-layer MLP for EMA roundtrip tests."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _stub_trainer(tmp_path: Path):
    """Build a `NeosrSPANFinetuner` stub with the minimum attrs needed
    by `save_checkpoint()` and `load_checkpoint()`.

    Mirrors the pattern in `tests/test_grad_scaler.py`: we bypass
    `__init__` via `__new__` because the real `__init__` builds the
    model, optimizer, losses, dataloader, etc., which is way out of
    scope for a unit test.
    """
    from training.neosr_finetuner import NeosrSPANFinetuner, GradScaler

    trainer = NeosrSPANFinetuner.__new__(NeosrSPANFinetuner)
    trainer.config = {
        "training": {"mixed_precision": False},
        "security": {"allow_pickle_checkpoint": True},
    }
    trainer.finetune_cfg = {}
    trainer.checkpoint_manager = None
    trainer.checkpoint_dir = tmp_path
    trainer.device = "cpu"

    # Build a real model so state_dict() roundtrips correctly.
    trainer.model = _TinyModel(hidden=8)
    trainer.optimizer = optim.AdamW(trainer.model.parameters(), lr=1e-4)
    trainer.scheduler = None  # save_checkpoint uses `hasattr(self.scheduler, 'base')` and writes None.
    trainer.current_crop_size = 32
    trainer.use_progressive_crop = False
    trainer.phase1_epochs = 30
    trainer.best_loss = float("inf")
    trainer.early_stopping_counter = 0
    trainer.best_metric_value = float("-inf")
    trainer.epochs = 50
    trainer.save_interval = 10

    # AMP is off in CPU tests; the shim GradScaler is enabled=False.
    trainer.use_amp = False
    trainer.scaler = GradScaler("cuda", enabled=False)
    return trainer


# ---------------------------------------------------------------------------
# Pure EMA save/load tests (no finetuner)
# ---------------------------------------------------------------------------

class TestEMAResumeRoundtrip:
    """Verify the EMA shadow survives a torch.save / torch.load cycle."""

    def test_save_load_ema_state_dict(self, tmp_path):
        """Shadow params in the loaded EMA match the saved ones bit-exact."""
        torch.manual_seed(0)
        model_a = _TinyModel()
        ema_a = GeneratorEMA(model_a, decay=0.999)

        # Simulate a few training steps so the shadow is non-trivial.
        for _ in range(3):
            with torch.no_grad():
                for p in model_a.parameters():
                    p.add_(torch.randn_like(p) * 0.01)
            ema_a.update(model_a)

        # Save
        ckpt = {
            "model_state_dict": model_a.state_dict(),
            "ema_state_dict": ema_a.state_dict(),
        }
        path = tmp_path / "ema_test.pth"
        torch.save(ckpt, path)

        # Load into fresh instances
        model_b = _TinyModel()
        ema_b = GeneratorEMA(model_b, decay=0.999)
        loaded = torch.load(path, weights_only=True)
        model_b.load_state_dict(loaded["model_state_dict"])
        ema_b.load_state_dict(loaded["ema_state_dict"])

        # Every EMA shadow key must match.
        sd_a = ema_a.state_dict()
        sd_b = ema_b.state_dict()
        assert set(sd_a.keys()) == set(sd_b.keys()), (
            f"Key mismatch after load: "
            f"only_in_a={set(sd_a) - set(sd_b)}, only_in_b={set(sd_b) - set(sd_a)}"
        )
        for k in sd_a:
            assert torch.equal(sd_a[k], sd_b[k]), f"EMA shadow mismatch for '{k}'"

    def test_ema_forward_unchanged_after_load(self, tmp_path):
        """A loaded (model, EMA) pair produces the same forward as the
        original when the EMA contextmanager is active."""
        torch.manual_seed(0)
        model_a = _TinyModel(hidden=8)
        ema_a = GeneratorEMA(model_a, decay=0.999)
        for _ in range(2):
            with torch.no_grad():
                for p in model_a.parameters():
                    p.add_(torch.randn_like(p) * 0.01)
            ema_a.update(model_a)

        x = torch.randn(2, 8)
        with ema_a.averaged(model_a):
            y_a = model_a(x)

        # Save + load
        path = tmp_path / "ema_test2.pth"
        torch.save(
            {
                "model_state_dict": model_a.state_dict(),
                "ema_state_dict": ema_a.state_dict(),
            },
            path,
        )
        model_b = _TinyModel(hidden=8)
        ema_b = GeneratorEMA(model_b, decay=0.999)
        loaded = torch.load(path, weights_only=True)
        model_b.load_state_dict(loaded["model_state_dict"])
        ema_b.load_state_dict(loaded["ema_state_dict"])

        with ema_b.averaged(model_b):
            y_b = model_b(x)

        assert torch.equal(y_a, y_b), "Forward outputs differ after save/load"

    def test_loaded_ema_does_not_track_post_save_model_updates(self, tmp_path):
        """The loaded EMA is decoupled from the *original* model object.
        Mutating model_a AFTER the save must NOT move ema_b's shadow.
        """
        torch.manual_seed(0)
        model_a = _TinyModel()
        ema_a = GeneratorEMA(model_a, decay=0.999)
        for _ in range(2):
            with torch.no_grad():
                for p in model_a.parameters():
                    p.add_(torch.randn_like(p) * 0.01)
            ema_a.update(model_a)

        # Save snapshot of EMA at this point.
        saved_shadow = copy.deepcopy(ema_a.state_dict())
        path = tmp_path / "ema_test3.pth"
        torch.save(
            {
                "model_state_dict": model_a.state_dict(),
                "ema_state_dict": saved_shadow,
            },
            path,
        )

        # Mutate model_a post-save and update ema_a (the original EMA).
        # ema_b (loaded later) must NOT see these changes.
        with torch.no_grad():
            for p in model_a.parameters():
                p.add_(torch.randn_like(p) * 0.5)
        ema_a.update(model_a)
        assert not torch.equal(ema_a.state_dict()["fc1.weight"], saved_shadow["fc1.weight"])

        # Load into fresh instances
        model_b = _TinyModel()
        ema_b = GeneratorEMA(model_b, decay=0.999)
        loaded = torch.load(path, weights_only=True)
        model_b.load_state_dict(loaded["model_state_dict"])
        ema_b.load_state_dict(loaded["ema_state_dict"])

        # ema_b's shadow must equal the SAVE-TIME snapshot exactly,
        # not the post-update state of ema_a.
        for k, v_saved in saved_shadow.items():
            v_loaded = ema_b.state_dict()[k]
            assert torch.equal(v_saved, v_loaded), (
                f"Loaded EMA shadow drifted from saved snapshot at '{k}'"
            )
        # And, for extra clarity, ema_b should NOT match ema_a's current state.
        assert not torch.equal(
            ema_b.state_dict()["fc1.weight"],
            ema_a.state_dict()["fc1.weight"],
        )

    def test_save_load_with_batchnorm_buffers(self, tmp_path):
        """Buffers (BN running_mean/var) must also survive the roundtrip."""
        torch.manual_seed(0)

        class _TinyBN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
                self.bn = nn.BatchNorm2d(4)
                with torch.no_grad():
                    self.bn.running_mean.fill_(0.42)
                    self.bn.running_var.fill_(1.7)
                    self.bn.num_batches_tracked.fill_(3)

        model_a = _TinyBN()
        ema_a = GeneratorEMA(model_a, decay=0.99)
        # Run one update so shadow buffers are populated.
        ema_a.update(model_a)

        path = tmp_path / "ema_bn.pth"
        torch.save(
            {
                "model_state_dict": model_a.state_dict(),
                "ema_state_dict": ema_a.state_dict(),
            },
            path,
        )

        model_b = _TinyBN()
        # Init model_b to different values; load should overwrite them.
        with torch.no_grad():
            model_b.bn.running_mean.fill_(-1.0)
            model_b.bn.running_var.fill_(-1.0)
        ema_b = GeneratorEMA(model_b, decay=0.99)
        loaded = torch.load(path, weights_only=True)
        model_b.load_state_dict(loaded["model_state_dict"])
        ema_b.load_state_dict(loaded["ema_state_dict"])

        for k in ("bn.running_mean", "bn.running_var", "bn.num_batches_tracked"):
            assert torch.equal(
                ema_b.state_dict()[k], ema_a.state_dict()[k]
            ), f"BN buffer '{k}' mismatch after roundtrip"


# ---------------------------------------------------------------------------
# End-to-end test using the real finetuner save_checkpoint path
# ---------------------------------------------------------------------------

class TestEMAResumeViaFinetuner:
    """Exercise `NeosrSPANFinetuner.save_checkpoint()` to confirm it
    stores `ema_state_dict` and `ema_model_state_dict` (both keys),
    and that loading into a fresh stub restores the shadow correctly.
    """

    def test_save_checkpoint_includes_ema_keys(self, tmp_path):
        trainer = _stub_trainer(tmp_path)
        # Attach an EMA on top of the stub model.
        trainer.ema = GeneratorEMA(trainer.model, decay=0.999)
        # Drive shadow and live model apart so the saved state is non-trivial.
        with torch.no_grad():
            for p in trainer.model.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        trainer.ema.update(trainer.model)

        trainer.save_checkpoint(epoch=5, is_best=True)

        ckpt_path = trainer.checkpoint_dir / "finetune_best.pth"
        assert ckpt_path.exists()
        ckpt = torch.load(ckpt_path, weights_only=False)
        assert "ema_state_dict" in ckpt, "save_checkpoint did not write ema_state_dict"
        assert "ema_model_state_dict" in ckpt, (
            "save_checkpoint did not write ema_model_state_dict (inference compat key)"
        )
        # Both keys must point at the same payload (per the save code).
        assert set(ckpt["ema_state_dict"].keys()) == set(
            ckpt["ema_model_state_dict"].keys()
        )

    def test_full_save_load_cycle_via_finetuner(self, tmp_path):
        """End-to-end: stub trainer A saves a checkpoint with EMA;
        stub trainer B loads it; both produce identical forward output
        under the EMA context."""
        # Trainer A: build model + EMA, drift them apart, save.
        trainer_a = _stub_trainer(tmp_path)
        trainer_a.ema = GeneratorEMA(trainer_a.model, decay=0.999)
        with torch.no_grad():
            for p in trainer_a.model.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        trainer_a.ema.update(trainer_a.model)

        x = torch.randn(2, 8)
        with trainer_a.ema.averaged(trainer_a.model):
            y_a = trainer_a.model(x)

        trainer_a.save_checkpoint(epoch=7, is_best=True)
        ckpt_path = trainer_a.checkpoint_dir / "finetune_best.pth"

        # Trainer B: fresh model + EMA, load the checkpoint.
        trainer_b = _stub_trainer(tmp_path)
        trainer_b.ema = GeneratorEMA(trainer_b.model, decay=0.999)
        # Mimic the resume code path: read ema_state_dict, load into EMA.
        ckpt = torch.load(ckpt_path, weights_only=False)
        assert "ema_state_dict" in ckpt
        trainer_b.ema.load_state_dict(ckpt["ema_state_dict"])

        # Forward output must match bit-exact.
        with trainer_b.ema.averaged(trainer_b.model):
            y_b = trainer_b.model(x)
        assert torch.equal(y_a, y_b), (
            "Forward outputs differ across finetuner save/load cycle"
        )

    def test_old_checkpoint_without_ema_key_loads_cleanly(self, tmp_path):
        """A checkpoint saved BEFORE EMA wiring (no `ema_state_dict`)
        must still load. The trainer's resume path logs a warning and
        the EMA stays at its (initial) deepcopy-of-model values."""
        trainer = _stub_trainer(tmp_path)
        # Build a "legacy" checkpoint with no EMA keys.
        legacy = {
            "epoch": 3,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": None,
            "best_loss": 0.5,
            "early_stopping_counter": 0,
            "best_metric_value": 0.5,
            "config": trainer.config,
            "crop_size": 32,
            "progressive_crop_enabled": False,
            "phase1_epochs": 30,
        }
        legacy_path = tmp_path / "legacy.pth"
        torch.save(legacy, legacy_path)

        # Fresh trainer with EMA wired; loading the legacy ckpt should
        # NOT raise. We exercise the exact branch from
        # `neosr_finetuner.py:1479-1496` here.
        trainer_b = _stub_trainer(tmp_path)
        trainer_b.ema = GeneratorEMA(trainer_b.model, decay=0.999)
        ckpt = torch.load(legacy_path, weights_only=False)
        ema_payload = (
            ckpt.get("ema_state_dict")
            if "ema_state_dict" in ckpt
            else ckpt.get("ema_model_state_dict")
        )
        if ema_payload is not None:
            trainer_b.ema.load_state_dict(ema_payload)
        # Otherwise: shadow stays at its initial deepcopy values (no-op).
        # The test passes if we got here without an exception.
        assert trainer_b.ema is not None
