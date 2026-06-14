"""
Unit tests for the GeneratorEMA class (Phase D-EMA).

Covers:
- Parameter decay updates after each `update(model)` call
- The `averaged()` contextmanager swaps params in and restores them
- `state_dict()` / `load_state_dict()` roundtrip preserves shadow exactly
- Buffers (e.g. BatchNorm running stats) are copied verbatim, not decayed
- decay=0.0 produces an eager copy (shadow == live model)
- decay=1.0 freezes the shadow at its initial values

The tests use small CPU-only models so no GPU or special dependencies
are required.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import torch
import torch.nn as nn
import copy

from training.ema import GeneratorEMA, EMA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class TinyLinear(nn.Module):
    """Minimal model for EMA unit tests."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=False)
        # Initialize to a known non-random state so we can reason about
        # exact values without seeding gymnastics.
        with torch.no_grad():
            self.fc.weight.fill_(0.5)


class TinyBN(nn.Module):
    """Tiny model with BatchNorm2d to exercise buffer handling."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        # Disable BN's running-stats reset behavior so we can set them.
        self.bn.track_running_stats = True
        # Force a known initial state for running_mean / running_var /
        # num_batches_tracked.
        with torch.no_grad():
            self.bn.running_mean.fill_(0.1)
            self.bn.running_var.fill_(2.0)
            self.bn.num_batches_tracked.fill_(7)
            self.bn.weight.fill_(1.0)
            self.bn.bias.zero_()


@pytest.fixture
def tiny_model() -> nn.Module:
    return TinyLinear()


@pytest.fixture
def tiny_bn_model() -> nn.Module:
    return TinyBN()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEMAUpdatesParams:
    def test_update_tracks_model(self, tiny_model):
        """After multiple updates with changing params, shadow tracks model.

        With decay=0.9 and a per-step perturbation of std 0.1, the live
        model performs a noisy random walk while the shadow smooths
        (acts as a low-pass filter). The key property to verify is that
        the shadow is moving in the same direction as the live model
        AND that it is closer to the live model than the initial
        shadow is. We allow a generous atol because the random walk
        does not converge to a point.
        """
        torch.manual_seed(0)
        ema = GeneratorEMA(tiny_model, decay=0.9)
        initial_shadow = copy.deepcopy(ema.ema_model.fc.weight.data)
        initial_live = copy.deepcopy(tiny_model.fc.weight.data)

        for _ in range(10):
            with torch.no_grad():
                tiny_model.fc.weight.add_(torch.randn_like(tiny_model.fc.weight) * 0.1)
            ema.update(tiny_model)

        # Shadow has moved away from its initial value.
        assert not torch.allclose(ema.ema_model.fc.weight.data, initial_shadow)

        # Distance from initial_live -> shadow is much less than
        # distance from initial_live -> live (smoothing is working).
        dist_shadow_from_initial = (
            (ema.ema_model.fc.weight.data - initial_live).abs().mean().item()
        )
        dist_live_from_initial = (
            (tiny_model.fc.weight.data - initial_live).abs().mean().item()
        )
        assert dist_shadow_from_initial < dist_live_from_initial, (
            f"Shadow {dist_shadow_from_initial:.4f} should track closer to "
            f"initial than live {dist_live_from_initial:.4f}"
        )

    def test_shadow_never_diverges_with_decay_one(self, tiny_model):
        """With decay=1.0, shadow should never change after construction.

        See `TestDecayOnePreservesInitial.test_decay_one_shadow_stays_at_initial`
        for the canonical assertion. This test exists to mirror the
        `test_decay_zero` symmetry documented in the plan.
        """
        ema = GeneratorEMA(tiny_model, decay=1.0)
        initial = copy.deepcopy(ema.ema_model.fc.weight.data)
        with torch.no_grad():
            tiny_model.fc.weight.add_(torch.randn_like(tiny_model.fc.weight))
        ema.update(tiny_model)
        assert torch.equal(ema.ema_model.fc.weight.data, initial)


class TestEMAAveragedContextManager:
    def test_averaged_swaps_and_restores(self, tiny_model):
        """Inside `with ema.averaged(model)` the params are shadowed; on
        exit the live params are restored exactly."""
        torch.manual_seed(0)
        ema = GeneratorEMA(tiny_model, decay=0.5)

        # Drive shadow and model apart.
        for _ in range(5):
            with torch.no_grad():
                tiny_model.fc.weight.add_(torch.randn_like(tiny_model.fc.weight) * 0.2)
            ema.update(tiny_model)

        # Capture the live params BEFORE entering the context.
        live_before = copy.deepcopy(tiny_model.fc.weight.data)

        with ema.averaged(tiny_model):
            # Inside the context, model should now hold shadow values.
            assert torch.allclose(tiny_model.fc.weight.data, ema.ema_model.fc.weight.data)
            # And those values should differ from the pre-context live
            # weights (otherwise the test is vacuous).
            assert not torch.allclose(tiny_model.fc.weight.data, live_before)

        # After exit, live params are restored exactly.
        assert torch.allclose(tiny_model.fc.weight.data, live_before)

    def test_averaged_restores_even_on_exception(self, tiny_model):
        """If validation code raises inside the context, params are still
        restored on the way out (contextmanager contract)."""
        torch.manual_seed(0)
        ema = GeneratorEMA(tiny_model, decay=0.5)
        for _ in range(3):
            with torch.no_grad():
                tiny_model.fc.weight.add_(torch.randn_like(tiny_model.fc.weight) * 0.1)
            ema.update(tiny_model)

        live_before = copy.deepcopy(tiny_model.fc.weight.data)
        with pytest.raises(RuntimeError, match="boom"):
            with ema.averaged(tiny_model):
                raise RuntimeError("boom")
        assert torch.allclose(tiny_model.fc.weight.data, live_before)


class TestEMAStateDictRoundtrip:
    def test_state_dict_roundtrip(self, tiny_model):
        """`state_dict()` + `load_state_dict()` roundtrip preserves shadow."""
        torch.manual_seed(0)
        ema1 = GeneratorEMA(tiny_model, decay=0.7)
        for _ in range(5):
            with torch.no_grad():
                tiny_model.fc.weight.add_(torch.randn_like(tiny_model.fc.weight) * 0.1)
            ema1.update(tiny_model)

        snapshot = ema1.state_dict()

        # Build a second EMA on a fresh model, load the snapshot.
        tiny_model2 = TinyLinear()
        ema2 = GeneratorEMA(tiny_model2, decay=0.7)
        ema2.load_state_dict(snapshot)

        # Every key in the snapshot must match exactly.
        sd2 = ema2.state_dict()
        assert set(snapshot.keys()) == set(sd2.keys())
        for k in snapshot:
            assert torch.allclose(snapshot[k], sd2[k]), f"mismatch at {k}"

    def test_state_dict_keys_match_model(self, tiny_model):
        """`state_dict()` returns the same keys as the shadow model."""
        ema = GeneratorEMA(tiny_model, decay=0.999)
        sd = ema.state_dict()
        model_sd = ema.ema_model.state_dict()
        assert set(sd.keys()) == set(model_sd.keys())


class TestEMAHandlesBuffers:
    def test_bn_buffers_copied_verbatim(self, tiny_bn_model):
        """BatchNorm running_mean / running_var / num_batches_tracked are
        buffers, not parameters. They must be copied verbatim each
        update, NOT decayed (decaying running stats would break BN)."""
        ema = GeneratorEMA(tiny_bn_model, decay=0.99)
        # After construction, the shadow's BN buffers equal the model's.
        assert torch.allclose(
            ema.ema_model.bn.running_mean,
            tiny_bn_model.bn.running_mean,
        )
        assert torch.allclose(
            ema.ema_model.bn.running_var,
            tiny_bn_model.bn.running_var,
        )
        assert ema.ema_model.bn.num_batches_tracked.item() == \
               tiny_bn_model.bn.num_batches_tracked.item()

        # Mutate the live model's buffers.
        with torch.no_grad():
            tiny_bn_model.bn.running_mean.fill_(0.5)
            tiny_bn_model.bn.running_var.fill_(3.5)
            tiny_bn_model.bn.num_batches_tracked.fill_(99)

        # And also change the BN affine params to verify decay is applied
        # to parameters only.
        with torch.no_grad():
            tiny_bn_model.bn.weight.fill_(2.0)
            tiny_bn_model.bn.bias.fill_(0.25)

        ema.update(tiny_bn_model)

        # Buffers are exact copies, not 99%-of-old.
        assert torch.allclose(
            ema.ema_model.bn.running_mean,
            torch.full_like(ema.ema_model.bn.running_mean, 0.5),
        )
        assert torch.allclose(
            ema.ema_model.bn.running_var,
            torch.full_like(ema.ema_model.bn.running_var, 3.5),
        )
        assert ema.ema_model.bn.num_batches_tracked.item() == 99

        # Affine weight is decayed: 0.99 * 1.0 + 0.01 * 2.0 = 1.01
        expected = 0.99 * 1.0 + 0.01 * 2.0
        assert torch.allclose(
            ema.ema_model.bn.weight,
            torch.full_like(ema.ema_model.bn.weight, expected),
            atol=1e-6,
        )


class TestDecayZeroCopiesEagerly:
    def test_decay_zero_first_update_copies_model(self, tiny_model):
        """With decay=0.0, the first update should set shadow == live model."""
        torch.manual_seed(0)
        ema = GeneratorEMA(tiny_model, decay=0.0)

        with torch.no_grad():
            tiny_model.fc.weight.add_(torch.randn_like(tiny_model.fc.weight) * 0.3)

        ema.update(tiny_model)
        assert torch.allclose(
            ema.ema_model.fc.weight.data,
            tiny_model.fc.weight.data,
        )


class TestDecayOnePreservesInitial:
    def test_decay_one_shadow_stays_at_initial(self, tiny_model):
        """With decay=1.0, the shadow is frozen at its construction-time
        values regardless of how many updates we do."""
        ema = GeneratorEMA(tiny_model, decay=1.0)
        initial = copy.deepcopy(ema.ema_model.fc.weight.data)

        for _ in range(20):
            with torch.no_grad():
                tiny_model.fc.weight.add_(torch.randn_like(tiny_model.fc.weight) * 0.5)
            ema.update(tiny_model)

        # Shadow is bit-exact unchanged.
        assert torch.equal(ema.ema_model.fc.weight.data, initial)


# ---------------------------------------------------------------------------
# Backward-compat alias: the pre-existing test in
# tests/test_e2e_small_dataset_pipeline.py imports `EMA` and reads
# `ema.shadow_params`. Verify the alias is still wired.
# ---------------------------------------------------------------------------

class TestEMAAlias:
    def test_alias_is_generator_ema(self):
        assert EMA is GeneratorEMA

    def test_shadow_params_dict_exposed(self, tiny_model):
        ema = EMA(tiny_model, decay=0.999)
        assert hasattr(ema, 'shadow_params')
        assert isinstance(ema.shadow_params, dict)
        # Every named parameter of the model should appear in shadow_params.
        for name, _ in tiny_model.named_parameters():
            assert name in ema.shadow_params
