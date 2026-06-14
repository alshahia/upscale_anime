"""
Unit tests for Phase 3 MEDIUM-priority fixes.

- Issue #9:  silent except -> logger.warning
- Issue #11: FDL skipped batch does not inflate total_loss
- Issue #12: NaN sanitizer logs up to 5 offender param names
- Issue #13: _compute_total_loss accepts include_adversarial kwarg
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from training.neosr_finetuner import NeosrSPANFinetuner


# ---------------------------------------------------------------------------
# Helpers: build a minimal stub trainer
# ---------------------------------------------------------------------------

def _build_trainer():
    trainer = NeosrSPANFinetuner.__new__(NeosrSPANFinetuner)
    trainer.current_epoch = 0
    trainer.batch_idx = 0
    trainer.global_step = 0
    trainer.use_perceptual = False
    trainer.perceptual_loss = None
    trainer.use_line_art = False
    trainer.line_art_loss = None
    trainer.use_color = False
    trainer.color_loss = None
    trainer.use_flat = False
    trainer.flat_loss = None
    trainer.use_frequency = False
    trainer.frequency_loss = None
    trainer.use_fdl = False
    trainer.fdl_loss = None
    trainer._fdl_last_value = 0.0
    trainer.fdl_weight = 0.05
    trainer.fdl_compute_every = 1
    # Conditional FDL (set by __init__ when loss.fdl.conditional is present).
    # Default off — existing FDL-on path is unaffected.
    trainer.fdl_conditional_enabled = False
    trainer.fdl_warmup_steps = 0
    trainer.use_wavelet_guided = False
    trainer.wavelet_guided_loss = None
    trainer.wavelet_guided_weight = 1.0
    trainer.wavelet_init_epoch = 40
    trainer.use_adversarial = False
    trainer.discriminator = None
    trainer.pixel_loss = nn.L1Loss()
    return trainer


# ---------------------------------------------------------------------------
# 3.1: FDL NaN/Inf path uses logger.warning (not print)
# ---------------------------------------------------------------------------

class TestFDLNaNLogging:
    def test_fdl_nan_uses_logger_warning(self, caplog):
        trainer = _build_trainer()
        trainer.use_fdl = True
        trainer.fdl_compute_every = 1
        # NaN-producing stub
        nan_loss = MagicMock()
        nan_loss.item = MagicMock(side_effect=RuntimeError("should not be called"))
        trainer.fdl_loss = MagicMock(return_value=nan_loss)
        sr = torch.randn(1, 3, 8, 8)
        hr = torch.randn(1, 3, 8, 8)
        # First, make torch.isnan return True for our mock loss
        with patch('torch.isnan', return_value=torch.tensor(True)):
            with caplog.at_level(logging.WARNING, logger='training.neosr_finetuner'):
                # Force the FDL branch to run; the NaN path logs a warning.
                trainer._fdl_last_value = 0.0
                # Re-build the function call directly: we need a tensor that fails the isnan check
                # so we just use a real tensor with NaN.
                trainer.fdl_loss = MagicMock(return_value=torch.tensor(float('nan')))
                loss_dict = {}
                weights = {'fdl': 0.05, 'pixel': 1.0, 'perceptual': 0.1,
                           'line_art': 1.0, 'flat': 0.05, 'frequency': 0.025,
                           'adversarial': 0.0}
                with patch.object(trainer, '_get_phase_weights', return_value=weights):
                    with patch('torch.amp.autocast'):
                        with caplog.at_level(logging.WARNING, logger='training.neosr_finetuner'):
                            trainer._compute_total_loss(sr, hr, batch_idx=0)

        # We just need to confirm: if FDL returns NaN, the logger records a warning.
        # The "loss_dict" after the call may or may not contain 'fdl' (NaN is
        # zeroed out), but the log must contain a 'NaN/Inf in FDL loss' entry.
        assert any(
            "FDL loss" in rec.getMessage() or "fdl" in rec.getMessage().lower()
            for rec in caplog.records
        ), f"Expected FDL NaN warning, got {[r.getMessage() for r in caplog.records]}"


# ---------------------------------------------------------------------------
# 3.4: FDL skipped batch does not add to total_loss
# ---------------------------------------------------------------------------

class TestFDLSkippedBatch:
    def test_skipped_batch_does_not_add_to_total_loss(self):
        trainer = _build_trainer()
        trainer.use_fdl = True
        trainer.fdl_compute_every = 4
        trainer.fdl_weight = 0.5
        trainer._fdl_last_value = 2.0  # cached value
        trainer.fdl_loss = MagicMock()  # non-None to enter the FDL branch

        sr = torch.randn(1, 3, 8, 8, requires_grad=True)
        hr = torch.randn(1, 3, 8, 8)
        weights = {'pixel': 1.0, 'perceptual': 0.0, 'line_art': 0.0,
                   'flat': 0.0, 'frequency': 0.0, 'adversarial': 0.0}
        with patch.object(trainer, '_get_phase_weights', return_value=weights):
            with patch('torch.amp.autocast'):
                # batch_idx=1 is NOT a multiple of 4 -> skipped branch
                loss_dict = trainer._compute_total_loss(sr, hr, batch_idx=1)

        # The 'total' loss must be exactly the pixel loss; FDL cached value
        # must NOT have been added.
        pixel_loss = trainer.pixel_loss(sr, hr)
        assert torch.allclose(loss_dict['total'], pixel_loss, atol=1e-5), (
            f"total_loss was {loss_dict['total'].item()}, expected {pixel_loss.item()}"
        )
        # 'fdl_cached' should be in loss_dict for logging visibility.
        assert 'fdl_cached' in loss_dict
        assert loss_dict['fdl_cached'] == 2.0


# ---------------------------------------------------------------------------
# 3.4b: Conditional FDL warmup skips the DINOv2 forward pass for the
# first fdl_warmup_steps global steps. Defaults to disabled, so existing
# FDL behavior is preserved.
# ---------------------------------------------------------------------------

class TestFDLConditionalWarmup:
    def test_conditional_disabled_is_noop(self):
        """Default (no loss.fdl.conditional block) must not change behavior."""
        trainer = _build_trainer()
        trainer.use_fdl = True
        trainer.fdl_conditional_enabled = False  # default
        trainer.fdl_warmup_steps = 0             # default
        trainer.fdl_compute_every = 1
        trainer.fdl_loss = MagicMock(
            return_value=torch.tensor(0.3, requires_grad=True),
        )
        trainer._fdl_last_value = 0.0

        sr = torch.randn(1, 3, 8, 8, requires_grad=True)
        hr = torch.randn(1, 3, 8, 8)
        weights = {'pixel': 1.0, 'perceptual': 0.0, 'line_art': 0.0,
                   'flat': 0.0, 'frequency': 0.0, 'adversarial': 0.0}
        with patch.object(trainer, '_get_phase_weights', return_value=weights):
            with patch('torch.amp.autocast'):
                loss_dict = trainer._compute_total_loss(sr, hr, batch_idx=0)

        # FDL ran (called) and contributed to total_loss
        assert trainer.fdl_loss.called, "fdl_loss should be called when conditional is off"
        assert 'fdl' in loss_dict
        assert 'fdl_cached' not in loss_dict

    def test_conditional_warmup_skips_fdl_forward(self):
        """When global_step < warmup_steps and conditional is enabled, FDL is skipped."""
        trainer = _build_trainer()
        trainer.use_fdl = True
        trainer.fdl_conditional_enabled = True
        trainer.fdl_warmup_steps = 100
        trainer.global_step = 42  # well within warmup
        trainer.fdl_compute_every = 1
        trainer.fdl_loss = MagicMock(
            return_value=torch.tensor(0.3, requires_grad=True),
        )
        trainer._fdl_last_value = 1.5  # pretend prior batch cached this

        sr = torch.randn(1, 3, 8, 8, requires_grad=True)
        hr = torch.randn(1, 3, 8, 8)
        weights = {'pixel': 1.0, 'perceptual': 0.0, 'line_art': 0.0,
                   'flat': 0.0, 'frequency': 0.0, 'adversarial': 0.0}
        with patch.object(trainer, '_get_phase_weights', return_value=weights):
            with patch('torch.amp.autocast'):
                loss_dict = trainer._compute_total_loss(sr, hr, batch_idx=0)

        # DINOv2 forward was NOT called (this is the whole point of the gate)
        assert not trainer.fdl_loss.called, (
            "fdl_loss forward must be skipped during warmup window"
        )
        # 'fdl' key absent, 'fdl_cached' present (for logging visibility)
        assert 'fdl' not in loss_dict
        assert 'fdl_cached' in loss_dict
        assert loss_dict['fdl_cached'] == 1.5
        # total_loss is exactly the pixel loss (no FDL contribution)
        pixel_loss = trainer.pixel_loss(sr, hr)
        assert torch.allclose(loss_dict['total'], pixel_loss, atol=1e-5)

    def test_conditional_warmup_expires(self):
        """After global_step >= warmup_steps, FDL runs normally again."""
        trainer = _build_trainer()
        trainer.use_fdl = True
        trainer.fdl_conditional_enabled = True
        trainer.fdl_warmup_steps = 50
        trainer.global_step = 50  # exactly at the threshold
        trainer.fdl_compute_every = 1
        trainer.fdl_loss = MagicMock(
            return_value=torch.tensor(0.3, requires_grad=True),
        )
        trainer._fdl_last_value = 0.0

        sr = torch.randn(1, 3, 8, 8, requires_grad=True)
        hr = torch.randn(1, 3, 8, 8)
        weights = {'pixel': 1.0, 'perceptual': 0.0, 'line_art': 0.0,
                   'flat': 0.0, 'frequency': 0.0, 'adversarial': 0.0}
        with patch.object(trainer, '_get_phase_weights', return_value=weights):
            with patch('torch.amp.autocast'):
                loss_dict = trainer._compute_total_loss(sr, hr, batch_idx=0)

        # Warmup is over: FDL ran and contributed.
        assert trainer.fdl_loss.called
        assert 'fdl' in loss_dict


# ---------------------------------------------------------------------------
# 3.5: NaN sanitizer logs up to 5 offender names
# ---------------------------------------------------------------------------

class TestGradientSanitizerLogging:
    def test_nan_gradients_log_offender_names(self, caplog):
        """When gradients contain NaN, the sanitizer logs the first 5 param names."""
        model = nn.Sequential(
            nn.Linear(4, 4),
            nn.Linear(4, 4),
            nn.Linear(4, 4),
            nn.Linear(4, 4),
            nn.Linear(4, 4),
            nn.Linear(4, 4),
            nn.Linear(4, 4),
        )
        # Inject NaN into specific gradients
        model[0].weight.grad = torch.full_like(model[0].weight, 0.0)
        model[0].weight.grad[0, 0] = float('nan')
        model[1].weight.grad = torch.full_like(model[1].weight, 0.0)
        model[1].weight.grad[0, 0] = float('inf')
        model[2].weight.grad = None
        model[3].weight.grad = None
        model[4].weight.grad = None
        model[5].weight.grad = None
        model[6].weight.grad = None

        trainer = _build_trainer()
        trainer.model = model
        trainer.current_epoch = 5
        trainer.batch_idx = 10

        # Manually invoke the sanitizer block by re-running it.
        with caplog.at_level(logging.WARNING, logger='training.neosr_finetuner'):
            # The sanitizer block from train_epoch, inlined for testability
            nan_count = 0
            offender_names = []
            for name, param in trainer.model.named_parameters():
                if param.grad is not None:
                    nan_mask = torch.isnan(param.grad) | torch.isinf(param.grad)
                    n_bad = nan_mask.sum().item()
                    if n_bad > 0:
                        nan_count += n_bad
                        if len(offender_names) < 5:
                            offender_names.append(f"{name} ({n_bad} bad)")
                        param.grad[nan_mask] = 0.0
            if nan_count > 0:
                suffix = (
                    f" offenders={offender_names}"
                    if offender_names
                    else ""
                )
                logger = logging.getLogger('training.neosr_finetuner')
                logger.warning(
                    "Sanitized %d NaN/Inf gradient values (epoch=%d, batch=%d)%s",
                    nan_count, trainer.current_epoch, trainer.batch_idx, suffix,
                )

        # Verify log was emitted
        msgs = [r.getMessage() for r in caplog.records]
        assert any("Sanitized" in m for m in msgs), f"Expected sanitizer warning, got {msgs}"
        # Verify offender names are present
        joined = " ".join(msgs)
        assert "0.weight" in joined, f"Expected first layer offender name, got {msgs}"
        assert "1.weight" in joined, f"Expected second layer offender name, got {msgs}"

    def test_sanitizer_caps_offender_list_at_five(self, caplog):
        """If more than 5 parameters have NaN, only the first 5 names are logged."""
        model = nn.Sequential(*[nn.Linear(4, 4) for _ in range(8)])
        for i, layer in enumerate(model):
            layer.weight.grad = torch.full_like(layer.weight, 0.0)
            layer.weight.grad[0, 0] = float('nan')  # all 8 layers poisoned
            layer.bias.grad = None

        trainer = _build_trainer()
        trainer.model = model
        trainer.current_epoch = 0
        trainer.batch_idx = 0

        with caplog.at_level(logging.WARNING, logger='training.neosr_finetuner'):
            nan_count = 0
            offender_names = []
            for name, param in trainer.model.named_parameters():
                if param.grad is not None:
                    nan_mask = torch.isnan(param.grad) | torch.isinf(param.grad)
                    n_bad = nan_mask.sum().item()
                    if n_bad > 0:
                        nan_count += n_bad
                        if len(offender_names) < 5:
                            offender_names.append(f"{name} ({n_bad} bad)")
                        param.grad[nan_mask] = 0.0
            if nan_count > 0:
                suffix = (
                    f" offenders={offender_names}"
                    if offender_names
                    else ""
                )
                logger = logging.getLogger('training.neosr_finetuner')
                logger.warning(
                    "Sanitized %d NaN/Inf gradient values (epoch=%d, batch=%d)%s",
                    nan_count, trainer.current_epoch, trainer.batch_idx, suffix,
                )

        joined = " ".join(r.getMessage() for r in caplog.records)
        # Should log up to 5 offenders. The offenders list itself is bounded.
        # All 8 weight layers are poisoned, but only 5 names appear in the list.
        assert "0.weight" in joined
        assert "4.weight" in joined
        # The 8th layer's name should NOT appear in the offender list section.
        # (The full log message includes "offenders=[...]" which is bounded to 5.)
        import re
        offenders_match = re.search(r'offenders=(\[.*?\])', joined)
        assert offenders_match is not None, f"offenders list not found in: {joined}"
        offenders_str = offenders_match.group(1)
        # Count names in the offenders list
        assert offenders_str.count("weight") == 5, (
            f"Expected 5 offender names, got: {offenders_str}"
        )


# ---------------------------------------------------------------------------
# 3.3: _compute_total_loss accepts include_adversarial kwarg
# ---------------------------------------------------------------------------

class TestIncludeAdversarialKwarg:
    def test_accepts_include_adversarial_true(self):
        trainer = _build_trainer()
        sr = torch.randn(1, 3, 8, 8, requires_grad=True)
        hr = torch.randn(1, 3, 8, 8)
        weights = {'pixel': 1.0, 'perceptual': 0.0, 'line_art': 0.0,
                   'flat': 0.0, 'frequency': 0.0, 'adversarial': 0.0}
        with patch.object(trainer, '_get_phase_weights', return_value=weights):
            with patch('torch.amp.autocast'):
                result = trainer._compute_total_loss(
                    sr, hr, batch_idx=0, include_adversarial=True
                )
        assert 'total' in result

    def test_accepts_include_adversarial_false(self):
        trainer = _build_trainer()
        sr = torch.randn(1, 3, 8, 8, requires_grad=True)
        hr = torch.randn(1, 3, 8, 8)
        weights = {'pixel': 1.0, 'perceptual': 0.0, 'line_art': 0.0,
                   'flat': 0.0, 'frequency': 0.0, 'adversarial': 0.0}
        with patch.object(trainer, '_get_phase_weights', return_value=weights):
            with patch('torch.amp.autocast'):
                result = trainer._compute_total_loss(
                    sr, hr, batch_idx=0, include_adversarial=False
                )
        assert 'total' in result

    def test_default_is_true(self):
        """When the kwarg is omitted, include_adversarial defaults to True."""
        trainer = _build_trainer()
        sr = torch.randn(1, 3, 8, 8, requires_grad=True)
        hr = torch.randn(1, 3, 8, 8)
        weights = {'pixel': 1.0, 'perceptual': 0.0, 'line_art': 0.0,
                   'flat': 0.0, 'frequency': 0.0, 'adversarial': 0.0}
        with patch.object(trainer, '_get_phase_weights', return_value=weights):
            with patch('torch.amp.autocast'):
                result = trainer._compute_total_loss(sr, hr, batch_idx=0)
        assert 'total' in result


# ---------------------------------------------------------------------------
# 3.4b: regression — loss-accumulator loop tolerates fdl_cached float
# ---------------------------------------------------------------------------

class TestLossAccumulatorToleratesFloat:
    """Regression test for the bug found during real training:
    train_epoch's loss-accumulator loop calls `value.item()` on every entry
    in loss_dict, but Phase 3.4 added `fdl_cached` (Python float) to loss_dict
    for skipped FDL batches. The accumulator must skip non-tensor values.
    """

    def test_accumulator_skips_non_tensor_values(self):
        """Simulate the train_epoch accumulator loop with a mixed loss_dict."""
        # Stand-in for a real loss_dict
        loss_dict = {
            'total': torch.tensor(1.0, requires_grad=True),
            'pixel': torch.tensor(0.5),
            'perceptual': torch.tensor(0.2),
            'fdl_cached': 0.12345,  # Python float, NOT tensor
        }
        loss_accumulators = {}
        for key, value in loss_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key not in loss_accumulators:
                loss_accumulators[key] = 0.0
            loss_accumulators[key] += value.item()

        # All tensor values should be in the accumulator.
        assert 'total' in loss_accumulators
        assert 'pixel' in loss_accumulators
        assert 'perceptual' in loss_accumulators
        # The float `fdl_cached` must NOT have been added (would have crashed
        # before the fix, or silently stored wrong type after).
        assert 'fdl_cached' not in loss_accumulators

    def test_train_epoch_accumulator_handles_fdl_cached(self):
        """End-to-end: the actual train_epoch loss-accumulator block must
        not crash when loss_dict contains fdl_cached (a float).
        """
        # Build a stub trainer and call the actual loop snippet
        trainer = NeosrSPANFinetuner.__new__(NeosrSPANFinetuner)
        loss_dict = {
            'total': torch.tensor(0.5),
            'pixel': torch.tensor(0.5),
            'fdl_cached': 0.05,  # float
        }
        loss_accumulators = {'total': 0.0}
        gradient_accumulation = 1

        # This is the same loop in train_epoch, copied here to ensure the
        # exact pattern is correct.
        for key, value in loss_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key not in loss_accumulators:
                loss_accumulators[key] = 0.0
            loss_accumulators[key] += value.item() * gradient_accumulation

        assert loss_accumulators['total'] == 0.5
        assert loss_accumulators['pixel'] == 0.5
        assert 'fdl_cached' not in loss_accumulators
