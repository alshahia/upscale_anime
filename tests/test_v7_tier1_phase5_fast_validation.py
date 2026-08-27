"""
Unit tests for V7 Tier 1 Phase 5: `fast_validation` flag in
`NeosrSPANFinetuner._compute_total_loss`.

Covers the perf/v7-tier1-speedups Phase 5 change:

> Phase 5: `fast_validation` flag in `_compute_total_loss`
> `src/training/neosr_finetuner.py:_compute_total_loss` gained a
> `fast_validation: bool = False` parameter. When `True`, the heavy
> frozen-backbone losses (FDL/DINOv2, frequency/DCT, wavelet-guided) are
> skipped and only the cheap ones (pixel, perceptual, line_art, color,
> flat) are computed. `validate()` calls it with `fast_validation=True`.
> Cuts validation wall-clock ~5-10x.

These tests verify:
- Signature: `_compute_total_loss` accepts `fast_validation: bool`.
- Behavior: with `fast_validation=True`, FDL / frequency / wavelet_guided
  losses are NEVER invoked. With `fast_validation=False` (default), they
  ARE invoked (when their `use_*` flags and weight > 0).
- Both modes always compute pixel (the only loss without an enable flag).
- Both modes always include a `total` entry in `loss_dict`.
- Both modes return a dict whose `total` is a torch.Tensor (the loss
  accumulator must not crash on the result).
- `validate()` calls `_compute_total_loss` with `fast_validation=True`.
"""
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import inspect
import pytest
import torch
import torch.nn as nn

from training.neosr_finetuner import NeosrSPANFinetuner


# ---------------------------------------------------------------------------
# Helpers: build a stub trainer with call-counting loss modules.
# ---------------------------------------------------------------------------

class _CountingLoss(nn.Module):
    """A loss module that returns a tensor and tracks forward() calls.

    Returns a single scalar tensor (the shape that the trainer expects for
    perceptual_loss / pixel_loss / etc.). The trainer unwraps dict-returning
    losses inline (e.g. ``line_result['line_art'] if isinstance(...)``), so
    we don't need dict-style stubs to exercise the call-counting contract.
    """

    def __init__(self, value: float = 0.1):
        super().__init__()
        self._value = value
        self.calls = 0

    def forward(self, sr, hr, **kwargs):
        self.calls += 1
        return torch.tensor(self._value, requires_grad=True)


def _stub_trainer(tmp_path) -> NeosrSPANFinetuner:
    """Build a trainer with stubbed model + loss modules.

    The loss modules are real `_CountingLoss` instances so we can verify
    whether they were invoked. The trainer's `_compute_total_loss` does the
    routing.
    """
    trainer = NeosrSPANFinetuner.__new__(NeosrSPANFinetuner)
    trainer.config = {'training': {'mixed_precision': False}}
    trainer.finetune_cfg = {}
    trainer.checkpoint_manager = None
    trainer.checkpoint_dir = tmp_path
    trainer.device = 'cpu'

    # Tiny real model so `.to(device)` works if needed.
    trainer.model = nn.Linear(4, 4)
    trainer.optimizer = None
    trainer.scheduler = None
    trainer.current_crop_size = 32
    trainer.use_progressive_crop = False
    trainer.phase1_epochs = 30
    trainer.best_loss = float('inf')
    trainer.early_stopping_counter = 0
    trainer.best_metric_value = float('-inf')
    trainer.epochs = 50
    trainer.save_interval = 10

    # Phase weights: pixel/perceptual/line_art/flat/frequency/adversarial.
    # These are the keys _get_phase_weights() always returns.
    trainer.use_two_phase = False
    trainer.pixel_weight = 1.0
    trainer.perceptual_weight = 0.3
    trainer.line_art_weight = 1.0
    trainer.flat_weight = 0.05
    trainer.frequency_weight = 0.15
    trainer.adversarial_weight = 0.0
    trainer.adversarial_max_weight = 0.0
    trainer.phase_adversarial_ramp_epochs = 0
    trainer.adversarial_start_epoch = 0
    trainer.perceptual_warmup_epochs = 0  # skip warmup branch

    # Replace losses with counting stubs.
    trainer.pixel_loss = nn.L1Loss()  # standard, not counted
    trainer.perceptual_loss = _CountingLoss(0.2)
    trainer.line_art_loss = _CountingLoss(0.1)
    trainer.color_loss = _CountingLoss(0.05)
    trainer.flat_loss = _CountingLoss(0.01)
    trainer.frequency_loss = _CountingLoss(0.03)
    trainer.wavelet_guided_loss = _CountingLoss(0.04)
    trainer.fdl_loss = _CountingLoss(0.02)

    # Use_* flags: turn them all on so that the default (fast_validation=False)
    # path tries to call every loss.
    trainer.use_perceptual = True
    trainer.use_line_art = True
    trainer.use_color = True
    trainer.use_flat = True
    trainer.use_frequency = True
    trainer.use_wavelet_guided = True
    trainer.use_fdl = True

    # Adversarial: not invoked inside _compute_total_loss (it lives in
    # train_epoch), so it does not matter here.
    trainer.use_adversarial = False
    trainer.adversarial_loss_fn = None
    trainer.discriminator = None

    # FDL every-2-batches setting. Set to 1 so every batch runs FDL (simplest).
    trainer.fdl_compute_every = 1
    trainer.fdl_weight = 0.1
    trainer.fdl_conditional_enabled = False
    trainer.fdl_warmup_steps = 0
    trainer.global_step = 0
    trainer.wavelet_start_epoch = 0  # so wavelet_guided always fires
    trainer.wavelet_guided_weight = 0.05

    # Color weight attribute used by the fast_validation block.
    trainer.color_weight = 0.5

    # current_epoch for weight selection (used by _get_phase_weights).
    trainer.current_epoch = 0
    return trainer


@pytest.fixture
def trainer(tmp_path):
    return _stub_trainer(tmp_path)


# ---------------------------------------------------------------------------
# Test 1: signature accepts the new flag with a sane default
# ---------------------------------------------------------------------------

class TestFastValidationSignature:

    def test_compute_total_loss_accepts_fast_validation_kwarg(self):
        sig = inspect.signature(NeosrSPANFinetuner._compute_total_loss)
        assert 'fast_validation' in sig.parameters, (
            "Phase 5: _compute_total_loss must accept a `fast_validation` kwarg"
        )
        param = sig.parameters['fast_validation']
        assert param.default is False, (
            "Phase 5: fast_validation default must be False to preserve "
            "existing behavior"
        )

    def test_fast_validation_is_keyword_with_bool_annotation(self):
        sig = inspect.signature(NeosrSPANFinetuner._compute_total_loss)
        param = sig.parameters['fast_validation']
        # annotation should be `bool` (or `bool = False` if erasable).
        ann = str(param.annotation)
        assert 'bool' in ann, f"Expected bool annotation, got {ann!r}"


# ---------------------------------------------------------------------------
# Test 2: behavior -- fast path skips heavy losses, slow path includes them
# ---------------------------------------------------------------------------

class TestFastValidationBehavior:

    def test_default_does_not_skip_heavy_losses(self, trainer):
        """fast_validation=False (default) must still call FDL, frequency,
        and wavelet_guided. This preserves training behavior."""
        sr = torch.rand(1, 3, 16, 16)
        hr = torch.rand(1, 3, 16, 16)

        before = (
            trainer.fdl_loss.calls,
            trainer.frequency_loss.calls,
            trainer.wavelet_guided_loss.calls,
        )
        out = trainer._compute_total_loss(sr, hr, batch_idx=0)
        after = (
            trainer.fdl_loss.calls,
            trainer.frequency_loss.calls,
            trainer.wavelet_guided_loss.calls,
        )
        # Default path: heavy losses were called at least once each.
        assert after[0] > before[0], "FDL should be called in default path"
        assert after[1] > before[1], "frequency should be called in default path"
        assert after[2] > before[2], "wavelet_guided should be called in default path"
        assert 'total' in out
        assert isinstance(out['total'], torch.Tensor)

    def test_fast_validation_skips_heavy_losses(self, trainer):
        """fast_validation=True must NOT call FDL, frequency, or wavelet_guided.
        This is the whole point of Phase 5 -- those are the dominant costs in
        validation (DINOv2 forward, DCT, Haar)."""
        sr = torch.rand(1, 3, 16, 16)
        hr = torch.rand(1, 3, 16, 16)

        before = (
            trainer.fdl_loss.calls,
            trainer.frequency_loss.calls,
            trainer.wavelet_guided_loss.calls,
        )
        out = trainer._compute_total_loss(
            sr, hr, batch_idx=0, fast_validation=True,
        )
        after = (
            trainer.fdl_loss.calls,
            trainer.frequency_loss.calls,
            trainer.wavelet_guided_loss.calls,
        )
        assert after[0] == before[0], "FDL must be SKIPPED in fast_validation"
        assert after[1] == before[1], "frequency must be SKIPPED in fast_validation"
        assert after[2] == before[2], "wavelet_guided must be SKIPPED in fast_validation"

    def test_fast_validation_still_calls_cheap_losses(self, trainer):
        """Pixel (always), perceptual, line_art, color, flat must all be
        computed in fast_validation mode. Pixel is always there; the others
        are gated by use_* flags."""
        sr = torch.rand(1, 3, 16, 16)
        hr = torch.rand(1, 3, 16, 16)

        before = (
            trainer.perceptual_loss.calls,
            trainer.line_art_loss.calls,
            trainer.color_loss.calls,
            trainer.flat_loss.calls,
        )
        out = trainer._compute_total_loss(
            sr, hr, batch_idx=0, fast_validation=True,
        )
        after = (
            trainer.perceptual_loss.calls,
            trainer.line_art_loss.calls,
            trainer.color_loss.calls,
            trainer.flat_loss.calls,
        )
        assert after[0] > before[0], "perceptual must run in fast path"
        assert after[1] > before[1], "line_art must run in fast path"
        assert after[2] > before[2], "color must run in fast path"
        assert after[3] > before[3], "flat must run in fast path"

        # Loss dict must contain the cheap keys + total.
        assert 'pixel' in out, "pixel is always present"
        assert 'total' in out, "total must be present"
        assert isinstance(out['total'], torch.Tensor)

        # Heavy-loss keys must NOT be present in the fast_validation output.
        for heavy in ('fdl', 'frequency', 'wavelet_guided'):
            assert heavy not in out, (
                f"fast_validation must omit '{heavy}' from loss_dict, "
                f"got keys: {sorted(out.keys())}"
            )

    def test_fast_validation_loss_dict_is_subset_of_default(self, trainer):
        """The fast_validation output must be a strict subset of the keys
        the default path can produce (modulo per-batch gating)."""
        sr = torch.rand(1, 3, 16, 16)
        hr = torch.rand(1, 3, 16, 16)

        default_out = trainer._compute_total_loss(sr, hr, batch_idx=0)
        fast_out = trainer._compute_total_loss(
            sr, hr, batch_idx=0, fast_validation=True,
        )
        # Pixel + total are always present; cheap losses gated by use_*;
        # heavy losses gated by use_* + weight>0.
        for k in fast_out.keys():
            assert k in default_out, (
                f"fast_validation produced key '{k}' not in default "
                f"output {sorted(default_out.keys())}"
            )
        # The fast path must drop the heavy-loss keys (since they ran in
        # default mode above).
        assert 'fdl' in default_out
        assert 'wavelet_guided' in default_out
        assert 'fdl' not in fast_out
        assert 'wavelet_guided' not in fast_out

    def test_fast_validation_total_is_finite_tensor(self, trainer):
        sr = torch.rand(1, 3, 16, 16)
        hr = torch.rand(1, 3, 16, 16)
        out = trainer._compute_total_loss(
            sr, hr, batch_idx=0, fast_validation=True,
        )
        total = out['total']
        assert isinstance(total, torch.Tensor)
        assert torch.isfinite(total).all().item(), (
            f"fast_validation total loss is non-finite: {total}"
        )


# ---------------------------------------------------------------------------
# Test 3: fast path respects use_* disable flags
# ---------------------------------------------------------------------------

class TestFastValidationDisabledFlags:

    def test_disabled_use_perceptual_is_skipped(self, trainer):
        trainer.use_perceptual = False
        sr = torch.rand(1, 3, 16, 16)
        hr = torch.rand(1, 3, 16, 16)
        out = trainer._compute_total_loss(
            sr, hr, batch_idx=0, fast_validation=True,
        )
        assert 'perceptual' not in out
        assert trainer.perceptual_loss.calls == 0

    def test_disabled_use_line_art_is_skipped(self, trainer):
        trainer.use_line_art = False
        sr = torch.rand(1, 3, 16, 16)
        hr = torch.rand(1, 3, 16, 16)
        out = trainer._compute_total_loss(
            sr, hr, batch_idx=0, fast_validation=True,
        )
        assert 'line_art' not in out
        assert trainer.line_art_loss.calls == 0

    def test_disabled_use_color_is_skipped(self, trainer):
        trainer.use_color = False
        sr = torch.rand(1, 3, 16, 16)
        hr = torch.rand(1, 3, 16, 16)
        out = trainer._compute_total_loss(
            sr, hr, batch_idx=0, fast_validation=True,
        )
        assert 'color' not in out
        assert trainer.color_loss.calls == 0

    def test_disabled_use_flat_is_skipped(self, trainer):
        trainer.use_flat = False
        sr = torch.rand(1, 3, 16, 16)
        hr = torch.rand(1, 3, 16, 16)
        out = trainer._compute_total_loss(
            sr, hr, batch_idx=0, fast_validation=True,
        )
        assert 'flat' not in out
        assert trainer.flat_loss.calls == 0

    def test_zero_weight_suppresses_heavy_loss_in_default_path(self, trainer):
        """If a heavy loss weight is 0, _compute_total_loss in the default
        path must still call it (it's the trainer's job to multiply), but
        the contribution is 0. We just verify that the call happens and the
        key appears. (Weight-zero skipping is NOT part of fast_validation's
        contract; the fast path skips the call entirely.)"""
        trainer.frequency_weight = 0.0
        sr = torch.rand(1, 3, 16, 16)
        hr = torch.rand(1, 3, 16, 16)
        before = trainer.frequency_loss.calls
        out = trainer._compute_total_loss(sr, hr, batch_idx=0)
        # Default path: even with weight=0, the loss module is invoked
        # (this matches existing behavior).
        assert trainer.frequency_loss.calls > before
        assert 'frequency' in out


# ---------------------------------------------------------------------------
# Test 4: validate() invokes the fast path
# ---------------------------------------------------------------------------

class TestValidateUsesFastValidation:
    """The whole point of Phase 5 is that `validate()` calls
    `_compute_total_loss(..., fast_validation=True)`. Patch the heavy losses
    and assert that they are not invoked during a validation pass."""

    def test_validate_calls_compute_total_loss_with_fast_validation_true(self, trainer):
        """Direct call: validate() must pass fast_validation=True to
        _compute_total_loss."""
        sr = torch.rand(1, 3, 16, 16)
        hr = torch.rand(1, 3, 16, 16)

        captured = {}

        def fake_compute(sr_arg, hr_arg, batch_idx, include_adversarial=True,
                         fast_validation=False):
            captured['fast_validation'] = fast_validation
            captured['batch_idx'] = batch_idx
            return {'total': torch.tensor(0.5), 'pixel': torch.tensor(0.5)}

        # Build a one-batch val_loader: a list with a single batch dict.
        val_loader = [{'lr': sr, 'hr': hr}]

        with patch.object(trainer, '_compute_total_loss', side_effect=fake_compute):
            with patch.object(trainer, 'model', MagicMock(return_value=sr)):
                # EMA may be None; validate() handles that with getattr.
                with patch.object(trainer, 'ema', None, create=True):
                    trainer.validate(val_loader)

        assert captured.get('fast_validation') is True, (
            f"validate() must call _compute_total_loss with fast_validation=True; "
            f"got fast_validation={captured.get('fast_validation')!r}"
        )
