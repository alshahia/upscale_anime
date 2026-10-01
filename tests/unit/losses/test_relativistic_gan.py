"""
Unit tests for RelativisticGANLoss D/G asymmetry (PR-3, Issue #3).

Regression tests for the fix that adds proper `discriminator_loss` and
`generator_loss` methods, replacing the symmetric `forward()` that returned
near-constant values for adversarial training.
"""
import sys
import os
import pytest
import torch
import warnings

from anime_sr.losses.adversarial_loss import RelativisticGANLoss, AdversarialLoss


@pytest.fixture
def loss() -> RelativisticGANLoss:
    return RelativisticGANLoss()


def _make_outputs(batch: int = 4, spatial: int = 8, seed: int = 0):
    """Build synthetic disc outputs with controlled gap between real and fake."""
    torch.manual_seed(seed)
    real = torch.randn(batch, 1, spatial, spatial) * 0.5 + 1.0  # centered at +1
    fake = torch.randn(batch, 1, spatial, spatial) * 0.5 - 1.0  # centered at -1
    return real, fake


class TestRelativisticDGAreDifferent:
    def test_d_and_g_loss_differ_for_separated_outputs(self, loss):
        real, fake = _make_outputs(seed=42)
        d_loss = loss.discriminator_loss(real, fake)
        g_loss = loss.generator_loss(real, fake)
        # D wants separation, G wants overlap -> they should NOT be equal.
        assert not torch.isclose(d_loss, g_loss, atol=1e-5), (
            f"D loss ({d_loss.item():.4f}) and G loss ({g_loss.item():.4f}) are equal; "
            "the symmetric forward() bug is back."
        )

    def test_d_and_g_loss_finite(self, loss):
        real, fake = _make_outputs(seed=1)
        d_loss = loss.discriminator_loss(real, fake)
        g_loss = loss.generator_loss(real, fake)
        assert torch.isfinite(d_loss)
        assert torch.isfinite(g_loss)

    def test_vanilla_gan_unchanged(self):
        """VanillaGANLoss should still be called with (output, is_real)."""
        from anime_sr.losses.adversarial_loss import VanillaGANLoss
        v = VanillaGANLoss()
        out = torch.randn(4, 1, 8, 8)
        loss_real = v(out, is_real=True)
        loss_fake = v(out, is_real=False)
        # These MUST differ.
        assert not torch.isclose(loss_real, loss_fake, atol=1e-5)


class TestRelativisticDMinimizesWhenSeparated:
    """When D(real) - D(fake) is large, the D loss should be small (good)."""

    def test_d_loss_lower_for_well_separated_outputs(self, loss):
        torch.manual_seed(0)
        # Real high, fake low -> D is winning -> D loss should be small
        separated_real = torch.full((4, 1, 4, 4), 5.0)
        separated_fake = torch.full((4, 1, 4, 4), -5.0)
        # Real and fake similar -> D is confused -> D loss should be large
        confused_real = torch.full((4, 1, 4, 4), 0.0)
        confused_fake = torch.full((4, 1, 4, 4), 0.0)

        d_loss_sep = loss.discriminator_loss(separated_real, separated_fake)
        d_loss_conf = loss.discriminator_loss(confused_real, confused_fake)

        assert d_loss_sep < d_loss_conf, (
            f"D loss when separated ({d_loss_sep.item():.4f}) should be < "
            f"confused ({d_loss_conf.item():.4f})"
        )


class TestRelativisticGMinimizesWhenOverlapping:
    """When D(real) ~ D(fake) (G has fooled D), the G loss should be small."""

    def test_g_loss_lower_when_outputs_overlap(self, loss):
        # Real and fake similar -> G has fooled D -> G loss should be small
        overlap_real = torch.full((4, 1, 4, 4), 0.5)
        overlap_fake = torch.full((4, 1, 4, 4), 0.5)
        # Real and fake far apart -> G is failing -> G loss should be large
        apart_real = torch.full((4, 1, 4, 4), 5.0)
        apart_fake = torch.full((4, 1, 4, 4), -5.0)

        g_loss_overlap = loss.generator_loss(overlap_real, overlap_fake)
        g_loss_apart = loss.generator_loss(apart_real, apart_fake)

        assert g_loss_overlap < g_loss_apart, (
            f"G loss when overlapping ({g_loss_overlap.item():.4f}) should be < "
            f"when apart ({g_loss_apart.item():.4f})"
        )


class TestDeprecationWarning:
    def test_forward_emits_deprecation_warning(self, loss):
        real, fake = _make_outputs(seed=7)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = loss(real, fake)
        assert any(issubclass(x.category, DeprecationWarning) for x in w), (
            "forward() should emit a DeprecationWarning on first call"
        )

    def test_forward_warns_only_once(self, loss):
        """Module should rate-limit the deprecation warning to once."""
        real, fake = _make_outputs(seed=8)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for _ in range(5):
                _ = loss(real, fake)
        deprec = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprec) == 1, f"Expected exactly 1 deprecation warning, got {len(deprec)}"


class TestAdversarialLossWrapper:
    """AdversarialLoss should expose both D and G methods for all loss types."""

    def test_relativistic_wrapper_routes_to_new_methods(self):
        adv = AdversarialLoss(loss_type='relativistic')
        real, fake = _make_outputs(seed=3)
        d = adv.discriminator_loss(real, fake)
        g = adv.generator_loss(real, fake)
        # Should NOT be equal (proves routing is asymmetric)
        assert not torch.isclose(d, g, atol=1e-5)

    def test_vanilla_wrapper_routes_via_is_real(self):
        adv = AdversarialLoss(loss_type='vanilla')
        real, fake = _make_outputs(seed=4)
        d = adv.discriminator_loss(real, fake)
        g = adv.generator_loss(real, fake)
        # Both should be finite and bounded
        assert torch.isfinite(d)
        assert torch.isfinite(g)
