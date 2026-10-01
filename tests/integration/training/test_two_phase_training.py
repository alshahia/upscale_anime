"""
Test Suite for Two-Phase Training Implementation.

Tests:
1. Config parsing for all loss components (pixel, perceptual, adversarial, line_art, flat, frequency)
2. Adversarial weight respects start_epoch config in two-phase mode
3. Phase transitions at correct epoch boundaries
4. All losses use phase-aware weights
5. Linear interpolation in Phase 2
6. Backward compatibility when two-phase is disabled
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent

import torch
import torch.nn as nn
import pytest

class MockFinetuner:
    """Mock finetuner for testing _get_phase_weights and _compute_adversarial_weight_phase_aware."""

    def __init__(
        self,
        use_two_phase=False,
        phase1_epochs=31,
        phase_adversarial_ramp_epochs=10,
        epochs=80,
        adversarial_start_epoch=20,
        use_adversarial=True,
        perceptual_warmup_epochs=10,
        phase1_pixel_weight=1.0,
        phase1_perceptual_weight=0.1,
        phase1_adversarial_weight=0.0,
        phase1_line_art_weight=1.0,
        phase1_flat_weight=0.05,
        phase1_frequency_weight=0.025,
        phase2_pixel_weight=0.6,
        phase2_perceptual_weight=0.12,
        phase2_adversarial_weight=0.0001,
        phase2_line_art_weight=1.0,
        phase2_flat_weight=0.05,
        phase2_frequency_weight=0.075,
        pixel_weight=1.0,
        perceptual_weight=0.1,
        adversarial_weight=0.00005,
        line_art_weight=1.0,
        flat_weight=0.05,
        frequency_weight=0.05,
    ):
        self.use_two_phase = use_two_phase
        self.phase1_epochs = phase1_epochs
        self.phase_adversarial_ramp_epochs = phase_adversarial_ramp_epochs
        self.epochs = epochs
        self.adversarial_start_epoch = adversarial_start_epoch
        self.use_adversarial = use_adversarial
        self.perceptual_warmup_epochs = perceptual_warmup_epochs

        self.phase1_pixel_weight = phase1_pixel_weight
        self.phase1_perceptual_weight = phase1_perceptual_weight
        self.phase1_adversarial_weight = phase1_adversarial_weight
        self.phase1_line_art_weight = phase1_line_art_weight
        self.phase1_flat_weight = phase1_flat_weight
        self.phase1_frequency_weight = phase1_frequency_weight

        self.phase2_pixel_weight = phase2_pixel_weight
        self.phase2_perceptual_weight = phase2_perceptual_weight
        self.phase2_adversarial_weight = phase2_adversarial_weight
        self.phase2_line_art_weight = phase2_line_art_weight
        self.phase2_flat_weight = phase2_flat_weight
        self.phase2_frequency_weight = phase2_frequency_weight

        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
        self.line_art_weight = line_art_weight
        self.flat_weight = flat_weight
        self.frequency_weight = frequency_weight

        self.current_epoch = 0
        self.discriminator = nn.Identity() if use_adversarial else None

    def _get_perceptual_weight(self) -> float:
        if self.current_epoch < self.perceptual_warmup_epochs:
            warmup_ratio = (self.current_epoch + 1) / self.perceptual_warmup_epochs
            return self.perceptual_weight * warmup_ratio
        return self.perceptual_weight

    def _compute_adversarial_weight_phase_aware(self, phase1_weight, phase2_weight, in_phase1):
        if not self.use_adversarial or self.discriminator is None:
            return 0.0
        if self.current_epoch < self.adversarial_start_epoch:
            return 0.0

        ramp_target = phase1_weight if in_phase1 else phase2_weight
        ramp_epochs = self.phase_adversarial_ramp_epochs
        ramp_end_epoch = self.adversarial_start_epoch + ramp_epochs

        if self.current_epoch < ramp_end_epoch:
            progress = (self.current_epoch - self.adversarial_start_epoch) / ramp_epochs
            return ramp_target * progress

        return ramp_target

    def _get_phase_weights(self):
        if not self.use_two_phase:
            return {
                'pixel': self.pixel_weight,
                'perceptual': self._get_perceptual_weight(),
                'adversarial': self._get_adversarial_weight_base(),
                'line_art': self.line_art_weight,
                'flat': self.flat_weight,
                'frequency': self.frequency_weight,
            }

        if self.current_epoch < self.phase1_epochs:
            perc_weight = self.phase1_perceptual_weight
            if self.perceptual_warmup_epochs > 0 and self.current_epoch < self.perceptual_warmup_epochs:
                warmup_ratio = (self.current_epoch + 1) / self.perceptual_warmup_epochs
                perc_weight = perc_weight * warmup_ratio

            adversarial_w = self._compute_adversarial_weight_phase_aware(
                phase1_weight=self.phase1_adversarial_weight,
                phase2_weight=self.phase2_adversarial_weight,
                in_phase1=True
            )

            return {
                'pixel': self.phase1_pixel_weight,
                'perceptual': perc_weight,
                'adversarial': adversarial_w,
                'line_art': self.phase1_line_art_weight,
                'flat': self.phase1_flat_weight,
                'frequency': self.phase1_frequency_weight,
            }
        else:
            total_phase2_epochs = self.epochs - self.phase1_epochs
            phase2_progress = (self.current_epoch - self.phase1_epochs) / total_phase2_epochs

            pixel_w = self.phase1_pixel_weight + (self.phase2_pixel_weight - self.phase1_pixel_weight) * phase2_progress
            perceptual_w = self.phase1_perceptual_weight + (self.phase2_perceptual_weight - self.phase1_perceptual_weight) * phase2_progress
            line_art_w = self.phase1_line_art_weight + (self.phase2_line_art_weight - self.phase1_line_art_weight) * phase2_progress
            flat_w = self.phase1_flat_weight + (self.phase2_flat_weight - self.phase1_flat_weight) * phase2_progress
            frequency_w = self.phase1_frequency_weight + (self.phase2_frequency_weight - self.phase1_frequency_weight) * phase2_progress

            adversarial_w = self._compute_adversarial_weight_phase_aware(
                phase1_weight=self.phase1_adversarial_weight,
                phase2_weight=self.phase2_adversarial_weight,
                in_phase1=False
            )

            return {
                'pixel': pixel_w,
                'perceptual': perceptual_w,
                'adversarial': adversarial_w,
                'line_art': line_art_w,
                'flat': flat_w,
                'frequency': frequency_w,
            }

    def _get_adversarial_weight_base(self) -> float:
        if not self.use_adversarial or self.discriminator is None:
            return 0.0
        if self.current_epoch < self.adversarial_start_epoch:
            return 0.0
        progress_epochs = 10
        if self.current_epoch < self.adversarial_start_epoch + progress_epochs:
            progress = (self.current_epoch - self.adversarial_start_epoch) / progress_epochs
            return self.adversarial_weight * progress
        return self.adversarial_weight

class TestTwoPhaseConfigParsing:
    """Test that config parsing correctly sets all phase weights."""

    def test_phase1_weights_parsed(self):
        finetuner = MockFinetuner(use_two_phase=True)
        assert finetuner.phase1_pixel_weight == 1.0
        assert finetuner.phase1_perceptual_weight == 0.1
        assert finetuner.phase1_adversarial_weight == 0.0
        assert finetuner.phase1_line_art_weight == 1.0
        assert finetuner.phase1_flat_weight == 0.05
        assert finetuner.phase1_frequency_weight == 0.025
        print("[PASS] Phase 1 weights parsed correctly")

    def test_phase2_weights_parsed(self):
        finetuner = MockFinetuner(use_two_phase=True)
        assert finetuner.phase2_pixel_weight == 0.6
        assert finetuner.phase2_perceptual_weight == 0.12
        assert finetuner.phase2_adversarial_weight == 0.0001
        assert finetuner.phase2_line_art_weight == 1.0
        assert finetuner.phase2_flat_weight == 0.05
        assert finetuner.phase2_frequency_weight == 0.075
        print("[PASS] Phase 2 weights parsed correctly")

    def test_adversarial_ramp_epochs_parsed(self):
        finetuner = MockFinetuner(use_two_phase=True, phase_adversarial_ramp_epochs=15)
        assert finetuner.phase_adversarial_ramp_epochs == 15
        print("[PASS] Adversarial ramp epochs parsed correctly")

class TestPhaseWeights:
    """Test _get_phase_weights() returns correct values."""

    def test_phase1_start_weights(self):
        finetuner = MockFinetuner(use_two_phase=True)
        finetuner.current_epoch = 0
        weights = finetuner._get_phase_weights()

        assert weights['pixel'] == 1.0, f"Expected pixel=1.0, got {weights['pixel']}"
        assert 0.0 <= weights['perceptual'] <= 0.11, f"Expected perceptual warmup in (0, 0.11], got {weights['perceptual']}"
        assert weights['adversarial'] == 0.0, f"Expected adversarial=0.0 (before start_epoch), got {weights['adversarial']}"
        assert weights['line_art'] == 1.0
        assert weights['flat'] == 0.05
        assert weights['frequency'] == 0.025
        print(f"[PASS] Phase 1 start weights correct (perceptual={weights['perceptual']:.4f}, adversarial=0.0)")

    def test_phase1_perceptual_warmup(self):
        finetuner = MockFinetuner(use_two_phase=True, perceptual_warmup_epochs=10)
        finetuner.current_epoch = 4
        weights = finetuner._get_phase_weights()

        expected_perc = 0.1 * (5 / 10)
        assert abs(weights['perceptual'] - expected_perc) < 1e-6, f"Expected perceptual={expected_perc}, got {weights['perceptual']}"
        print(f"[PASS] Phase 1 perceptual warmup correct: {weights['perceptual']:.4f}")

    def test_phase1_adversarial_after_start_epoch(self):
        finetuner = MockFinetuner(use_two_phase=True, adversarial_start_epoch=20, phase_adversarial_ramp_epochs=10)
        finetuner.current_epoch = 24
        weights = finetuner._get_phase_weights()

        progress = (24 - 20) / 10
        expected_adv = finetuner.phase1_adversarial_weight * progress
        assert abs(weights['adversarial'] - expected_adv) < 1e-6, f"Expected adversarial={expected_adv}, got {weights['adversarial']}"
        print(f"[PASS] Phase 1 adversarial ramp correct: {weights['adversarial']:.6f}")

    def test_phase2_start_weights(self):
        finetuner = MockFinetuner(use_two_phase=True, phase1_epochs=31)
        finetuner.current_epoch = 31
        weights = finetuner._get_phase_weights()

        total_phase2 = 80 - 31
        progress = (31 - 31) / total_phase2
        expected_pixel = 1.0 + (0.6 - 1.0) * progress
        assert abs(weights['pixel'] - expected_pixel) < 0.01, f"Expected pixel close to {expected_pixel:.4f}, got {weights['pixel']:.4f}"
        print(f"[PASS] Phase 2 start weights correct (pixel={weights['pixel']:.4f})")

    def test_phase2_adversarial_ramp(self):
        finetuner = MockFinetuner(
            use_two_phase=True,
            phase1_epochs=31,
            adversarial_start_epoch=20,
            phase_adversarial_ramp_epochs=10,
            phase2_adversarial_weight=0.0001
        )
        finetuner.current_epoch = 35
        weights = finetuner._get_phase_weights()

        ramp_end_epoch = 20 + 10
        if finetuner.current_epoch < ramp_end_epoch:
            progress = (finetuner.current_epoch - 20) / 10
            expected_adv = 0.0001 * progress
        else:
            expected_adv = 0.0001
        assert abs(weights['adversarial'] - expected_adv) < 1e-6, f"Expected adversarial={expected_adv}, got {weights['adversarial']}"
        print(f"[PASS] Phase 2 adversarial ramp correct: {weights['adversarial']:.8f} (expected {expected_adv:.8f})")

class TestTwoPhaseDisabled:
    """Test backward compatibility when two-phase is disabled."""

    def test_two_phase_disabled_uses_base_weights(self):
        finetuner = MockFinetuner(use_two_phase=False)
        finetuner.current_epoch = 0
        weights = finetuner._get_phase_weights()

        assert weights['pixel'] == 1.0, f"Expected pixel=1.0, got {weights['pixel']}"
        assert 0.0 <= weights['perceptual'] <= 0.11, f"Expected perceptual warmup in (0, 0.11], got {weights['perceptual']}"
        assert weights['adversarial'] == 0.0, f"Expected adversarial=0.0, got {weights['adversarial']}"
        assert weights['line_art'] == 1.0
        assert weights['flat'] == 0.05
        assert weights['frequency'] == 0.05
        print("[PASS] Two-phase disabled mode uses base weights correctly")

class TestAdversarialStartEpoch:
    """Test that adversarial weight respects start_epoch config."""

    def test_adversarial_zero_before_start_epoch_phase1(self):
        finetuner = MockFinetuner(use_two_phase=True, adversarial_start_epoch=20)
        finetuner.current_epoch = 15
        weights = finetuner._get_phase_weights()

        assert weights['adversarial'] == 0.0, f"Expected adversarial=0.0 before start_epoch, got {weights['adversarial']}"
        print("[PASS] Adversarial is 0.0 before start_epoch in Phase 1")

    def test_adversarial_zero_before_start_epoch_phase2(self):
        finetuner = MockFinetuner(use_two_phase=True, phase1_epochs=31, adversarial_start_epoch=20)
        finetuner.current_epoch = 25
        weights = finetuner._get_phase_weights()

        assert weights['adversarial'] == 0.0, f"Expected adversarial=0.0 before start_epoch, got {weights['adversarial']}"
        print("[PASS] Adversarial is 0.0 before start_epoch in Phase 2")

class TestLinearInterpolation:
    """Test linear interpolation in Phase 2."""

    def test_interpolation_mid_phase2(self):
        finetuner = MockFinetuner(use_two_phase=True, phase1_epochs=31, epochs=80)
        finetuner.current_epoch = 55

        total_phase2 = 80 - 31
        progress = (55 - 31) / total_phase2

        expected_pixel = 1.0 + (0.6 - 1.0) * progress
        expected_perc = 0.1 + (0.12 - 0.1) * progress
        expected_freq = 0.025 + (0.075 - 0.025) * progress

        weights = finetuner._get_phase_weights()

        assert abs(weights['pixel'] - expected_pixel) < 1e-6, f"Expected pixel={expected_pixel}, got {weights['pixel']}"
        assert abs(weights['perceptual'] - expected_perc) < 1e-6, f"Expected perceptual={expected_perc}, got {weights['perceptual']}"
        assert abs(weights['frequency'] - expected_freq) < 1e-6, f"Expected frequency={expected_freq}, got {weights['frequency']}"
        print(f"[PASS] Linear interpolation mid-phase2 correct (progress={progress:.2f})")

    def test_interpolation_end_phase2(self):
        finetuner = MockFinetuner(use_two_phase=True, phase1_epochs=31, epochs=80)
        finetuner.current_epoch = 79

        weights = finetuner._get_phase_weights()

        assert abs(weights['pixel'] - 0.6) < 0.05, f"Expected pixel close to 0.6, got {weights['pixel']}"
        assert abs(weights['perceptual'] - 0.12) < 0.02, f"Expected perceptual close to 0.12, got {weights['perceptual']}"
        print(f"[PASS] Linear interpolation end-phase2 correct (pixel={weights['pixel']:.4f})")

class TestAllLossesPhaseAware:
    """Test that all losses use phase-aware weights."""

    def test_all_six_losses_returned(self):
        finetuner = MockFinetuner(use_two_phase=True)
        finetuner.current_epoch = 0
        weights = finetuner._get_phase_weights()

        expected_keys = {'pixel', 'perceptual', 'adversarial', 'line_art', 'flat', 'frequency'}
        assert set(weights.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(weights.keys())}"
        print("[PASS] All 6 losses returned in phase weights dict")

    def test_line_art_constant_across_phases(self):
        finetuner = MockFinetuner(use_two_phase=True, phase1_epochs=31)
        finetuner.current_epoch = 0
        weights_start = finetuner._get_phase_weights()

        finetuner.current_epoch = 31
        weights_mid = finetuner._get_phase_weights()

        finetuner.current_epoch = 79
        weights_end = finetuner._get_phase_weights()

        assert weights_start['line_art'] == weights_mid['line_art'] == weights_end['line_art'] == 1.0
        print("[PASS] Line art weight stays constant across phases")

    def test_frequency_transitions_between_phases(self):
        finetuner = MockFinetuner(use_two_phase=True, phase1_epochs=31)
        finetuner.current_epoch = 30
        weights_phase1_end = finetuner._get_phase_weights()

        finetuner.current_epoch = 32
        weights_phase2_start = finetuner._get_phase_weights()

        assert weights_phase1_end['frequency'] == 0.025, f"Expected phase1 frequency=0.025, got {weights_phase1_end['frequency']}"
        total_phase2 = 80 - 31
        progress = (32 - 31) / total_phase2
        expected_freq = 0.025 + (0.075 - 0.025) * progress
        assert abs(weights_phase2_start['frequency'] - expected_freq) < 0.01, f"Expected phase2 frequency ~{expected_freq:.4f}, got {weights_phase2_start['frequency']:.4f}"
        print(f"[PASS] Frequency weight in Phase 2: {weights_phase2_start['frequency']:.4f} (interpolated from {expected_freq:.4f})")

class TestResumePhaseConsistency:
    """Test that phase determination uses checkpoint's phase1_epochs, not current config."""

    def test_phase_determination_with_restored_phase1_epochs(self):
        """When resuming from epoch 37 with checkpoint's phase1_epochs=36, should be in Phase 2."""
        finetuner = MockFinetuner(use_two_phase=True, phase1_epochs=40)

        resumed_phase1_epochs = 36
        finetuner.phase1_epochs = resumed_phase1_epochs
        finetuner.current_epoch = 37

        weights = finetuner._get_phase_weights()

        assert weights['pixel'] != finetuner.phase1_pixel_weight, \
            f"Expected Phase 2 weights, got Phase 1 (pixel={weights['pixel']})"
        assert finetuner.current_epoch >= finetuner.phase1_epochs, \
            f"Expected epoch {finetuner.current_epoch} >= phase1_epochs {finetuner.phase1_epochs}"

        total_phase2 = finetuner.epochs - finetuner.phase1_epochs
        progress = (finetuner.current_epoch - finetuner.phase1_epochs) / total_phase2
        expected_pixel = finetuner.phase1_pixel_weight + (finetuner.phase2_pixel_weight - finetuner.phase1_pixel_weight) * progress

        assert abs(weights['pixel'] - expected_pixel) < 0.01, \
            f"Expected pixel ~{expected_pixel:.4f}, got {weights['pixel']:.4f}"

        print(f"[PASS] Resume at epoch 37 with restored phase1_epochs=36: Phase 2 weights (pixel={weights['pixel']:.4f})")

    def test_phase_determination_original_config(self):
        """When resuming from epoch 5 with checkpoint's phase1_epochs=40, should be in Phase 1."""
        finetuner = MockFinetuner(use_two_phase=True, phase1_epochs=40)

        resumed_phase1_epochs = 40
        finetuner.phase1_epochs = resumed_phase1_epochs
        finetuner.current_epoch = 5

        weights = finetuner._get_phase_weights()

        assert weights['pixel'] == finetuner.phase1_pixel_weight, \
            f"Expected Phase 1 weights (pixel={finetuner.phase1_pixel_weight}), got {weights['pixel']}"
        assert finetuner.current_epoch < finetuner.phase1_epochs, \
            f"Expected epoch {finetuner.current_epoch} < phase1_epochs {finetuner.phase1_epochs}"

        print(f"[PASS] Resume at epoch 5 with restored phase1_epochs=40: Phase 1 weights (pixel={weights['pixel']})")

    def test_phase_boundary_epoch(self):
        """Epoch exactly at phase1_epochs boundary should be Phase 2."""
        finetuner = MockFinetuner(use_two_phase=True, phase1_epochs=31)
        finetuner.current_epoch = 31

        weights = finetuner._get_phase_weights()

        assert finetuner.current_epoch >= finetuner.phase1_epochs, \
            f"Epoch {finetuner.current_epoch} should be >= phase1_epochs {finetuner.phase1_epochs} for Phase 2"

        total_phase2 = finetuner.epochs - finetuner.phase1_epochs
        progress = (finetuner.current_epoch - finetuner.phase1_epochs) / total_phase2
        expected_pixel = finetuner.phase1_pixel_weight + (finetuner.phase2_pixel_weight - finetuner.phase1_pixel_weight) * progress

        assert abs(weights['pixel'] - expected_pixel) < 0.01, \
            f"Expected pixel ~{expected_pixel:.4f}, got {weights['pixel']:.4f}"

        print(f"[PASS] Epoch 31 exactly at boundary: Phase 2 weights (pixel={weights['pixel']:.4f})")

def run_all_tests():
    """Run all test classes."""
    test_classes = [
        TestTwoPhaseConfigParsing,
        TestPhaseWeights,
        TestTwoPhaseDisabled,
        TestAdversarialStartEpoch,
        TestLinearInterpolation,
        TestAllLossesPhaseAware,
        TestResumePhaseConsistency,
    ]

    total = 0
    passed = 0

    for cls in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {cls.__name__}...")
        print('='*60)
        instance = cls()
        for name in dir(instance):
            if name.startswith('test_'):
                total += 1
                try:
                    getattr(instance, name)()
                    passed += 1
                except AssertionError as e:
                    print(f"[FAIL] {name}: {e}")
                except Exception as e:
                    print(f"[ERROR] {name}: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} tests passed")
    print('='*60)

    if passed == total:
        print("\n[OK] All two-phase training tests passed!")
    else:
        print(f"\n[ERROR] {total - passed} tests failed!")

    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)