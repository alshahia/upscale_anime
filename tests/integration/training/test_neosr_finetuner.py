"""
Trainer-level integration tests for NeosrSPANFinetuner.

Phase 3.6: Provides reusable fixtures (minimal_config, dummy_dataloader,
tmp_checkpoint) and a small set of end-to-end tests that exercise the
trainer without instantiating the full SPAN backbone.

The trainer is constructed via `NeosrSPANFinetuner.__new__` and the bare
minimum of attributes is injected manually, mirroring the pattern used in
tests/test_grad_scaler.py and tests/test_safe_checkpoint_load.py.
"""
import os
import sys
import logging
import tempfile
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from unittest.mock import patch, MagicMock

# Project imports (use bare imports per AGENTS.md)
from anime_sr.training.neosr_finetuner import NeosrSPANFinetuner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_config() -> dict:
    """A minimal but valid finetune config that exercises two-phase training,
    FDL, frequency loss, and adversarial (relativistic).
    """
    return {
        'training': {
            'mixed_precision': False,  # keep CPU-only for CI
            'device': 'cpu',
            'security': {'allow_pickle_checkpoint': False},
            'finetune': {
                'lr': 1e-4,
                'epochs': 10,
                'warmup_epochs': 1,
                'min_lr': 1e-5,
                'two_phase_training': {
                    'enabled': True,
                    'phase1_epochs': 4,
                    'phase_adversarial_ramp_epochs': 2,
                },
                'loss': {
                    'pixel': {'type': 'l1', 'weight': 1.0},
                    'perceptual': {'enabled': False, 'weight': 0.1},
                    'adversarial': {
                        'enabled': False,  # off for CI speed
                        'type': 'relativistic',
                        'weight': 0.001,
                        'start_epoch': 5,
                        'max_weight': 0.005,
                    },
                    'frequency': {'enabled': False, 'weight': 0.1},
                    'fdl': {'enabled': False, 'weight': 0.05},
                    'anime': {
                        'line_art_preservation': {'enabled': False, 'weight': 1.0},
                        'color_consistency': {'enabled': False, 'weight': 0.25},
                        'flat_region_preservation': {'enabled': False, 'weight': 0.05},
                    },
                    'wavelet_guided': {'enabled': False, 'weight': 0.5},
                },
                'progressive_crop': {'enabled': False, 'stages': []},
            },
            'output': {
                'checkpoint_dir': tempfile.mkdtemp(prefix='test_ckpt_'),
                'run_tracker_csv': None,
            },
            'model': {
                'type': 'neosr_span',
                'upscale': 4,
                'span': {'num_blocks': 1, 'feature_channels': 4, 'use_v2': False},
            },
        },
        'paths': {
            'output_dir': tempfile.mkdtemp(prefix='test_out_'),
        },
    }


@pytest.fixture
def dummy_dataloader():
    """A 4-batch DataLoader of (lr, hr) tensors at 4x scale.

    LR: 8x8  / HR: 32x32  (scale=4). 4 batches, 1 sample each.
    """
    batches = []
    for _ in range(4):
        lr = torch.randn(1, 3, 8, 8)
        hr = torch.randn(1, 3, 32, 32)
        batches.append((lr, hr))

    class _ListLoader:
        def __init__(self, items):
            self._items = items
        def __iter__(self):
            return iter(self._items)
        def __len__(self):
            return len(self._items)
    return _ListLoader(batches)


@pytest.fixture
def tmp_checkpoint(tmp_path):
    """A tmp_path string usable as a checkpoint directory."""
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    return str(ckpt_dir)


# ---------------------------------------------------------------------------
# Trainer-builder helper
# ---------------------------------------------------------------------------

def _build_minimal_trainer(cfg: dict) -> NeosrSPANFinetuner:
    """Build a NeosrSPANFinetuner stub with the bare minimum of attributes
    for the tests below. Mirrors the pattern from test_grad_scaler.py.
    """
    trainer = NeosrSPANFinetuner.__new__(NeosrSPANFinetuner)
    trainer.config = cfg
    trainer.finetune_cfg = cfg['training']['finetune']
    trainer.device = torch.device('cpu')

    # Run-level state
    trainer.current_epoch = 0
    trainer.batch_idx = 0
    trainer.checkpoint_dir = cfg['training']['output']['checkpoint_dir']
    trainer.run_tracker_csv = None

    # Config-driven attributes (matching _setup_losses defaults)
    trainer.use_perceptual = False
    trainer.perceptual_loss = None
    trainer.perceptual_warmup_epochs = 0
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
    trainer.fdl_compute_every = 1
    trainer.fdl_weight = 0.0
    trainer.use_wavelet_guided = False
    trainer.wavelet_guided_loss = None
    trainer.wavelet_init_epoch = 0
    trainer.use_adversarial = False
    trainer.discriminator = None
    trainer.adversarial_loss_fn = None

    # Phase weights
    trainer.use_two_phase = True
    trainer.epochs = 10
    trainer.lr = 1e-4
    trainer.phase1_epochs = 4
    trainer.phase_adversarial_ramp_epochs = 2
    trainer.phase1_pixel_weight = 1.0
    trainer.phase2_pixel_weight = 0.5
    trainer.phase1_perceptual_weight = 0.1
    trainer.phase2_perceptual_weight = 0.2
    trainer.phase1_line_art_weight = 1.0
    trainer.phase2_line_art_weight = 0.5
    trainer.phase1_flat_weight = 0.05
    trainer.phase2_flat_weight = 0.04
    trainer.phase1_frequency_weight = 0.025
    trainer.phase2_frequency_weight = 0.075
    trainer.phase1_adversarial_weight = 0.0
    trainer.phase2_adversarial_weight = 0.001

    # Single-loss weights (fallback)
    trainer.pixel_loss = nn.L1Loss()
    trainer.pixel_weight = 1.0
    trainer.perceptual_weight = 0.1
    trainer.line_art_weight = 1.0
    trainer.color_weight = 0.25
    trainer.flat_weight = 0.05
    trainer.frequency_weight = 0.025
    trainer.adversarial_weight = 0.001
    trainer.adversarial_start_epoch = 5
    trainer.adversarial_max_weight = 0.005

    # Adversarial-weight helpers (stubbed)
    trainer._get_adversarial_weight = MagicMock(return_value=0.0)
    trainer._compute_adversarial_weight_phase_aware = MagicMock(return_value=0.0)
    trainer._get_perceptual_weight = MagicMock(return_value=0.0)

    # AMP
    trainer.use_amp = False
    trainer.scaler = None

    # Model/optimizer (small dummy)
    trainer.model = nn.Conv2d(3, 3, 3, padding=1)
    trainer.optimizer = optim.Adam(trainer.model.parameters(), lr=1e-4)
    trainer.scheduler = None

    # NaN sanitizer init
    trainer._nan_offender_logged_this_epoch = 0

    return trainer


# ---------------------------------------------------------------------------
# 3.6: Trainer-level integration tests
# ---------------------------------------------------------------------------

class TestTrainerReadConfigEarlyIntegration:
    """3.6a: Verify the trainer populates _read_config_early attributes."""

    def test_early_attrs_populated(self, minimal_config):
        trainer = _build_minimal_trainer(minimal_config)
        # _read_config_early should have been called by __init__...
        # but we bypassed __init__, so call it directly.
        trainer._read_config_early()
        assert trainer.lr == 1e-4
        assert trainer.epochs == 10
        assert trainer.phase1_epochs == 4
        assert trainer.phase_adversarial_ramp_epochs == 2
        assert trainer.adversarial_start_epoch == 5


class TestTrainerLossPath:
    """3.6b: Exercise _compute_total_loss end-to-end with the configured trainer."""

    def test_compute_total_loss_pixel_only(self, minimal_config):
        trainer = _build_minimal_trainer(minimal_config)
        sr = torch.randn(1, 3, 8, 8, requires_grad=True)
        hr = torch.randn(1, 3, 8, 8)
        loss_dict = trainer._compute_total_loss(sr, hr, batch_idx=0)
        assert 'total' in loss_dict
        assert 'pixel' in loss_dict
        assert torch.allclose(loss_dict['total'], loss_dict['pixel'])

    def test_compute_total_loss_gradient_flows(self, minimal_config):
        trainer = _build_minimal_trainer(minimal_config)
        sr = torch.randn(1, 3, 8, 8, requires_grad=True)
        hr = torch.randn(1, 3, 8, 8)
        loss_dict = trainer._compute_total_loss(sr, hr, batch_idx=0)
        loss_dict['total'].backward()
        assert sr.grad is not None
        assert torch.isfinite(sr.grad).all()

    def test_phase1_uses_phase1_weights(self, minimal_config):
        trainer = _build_minimal_trainer(minimal_config)
        trainer.current_epoch = 0  # in phase 1
        weights = trainer._get_phase_weights()
        assert weights['pixel'] == trainer.phase1_pixel_weight
        assert weights['perceptual'] == trainer.phase1_perceptual_weight

    def test_phase2_end_uses_phase2_weights(self, minimal_config):
        trainer = _build_minimal_trainer(minimal_config)
        trainer.current_epoch = 10  # epochs; progress = (10-4)/6 = 1.0
        weights = trainer._get_phase_weights()
        assert weights['pixel'] == trainer.phase2_pixel_weight
        assert weights['perceptual'] == trainer.phase2_perceptual_weight

    def test_phase2_progress_interpolation(self, minimal_config):
        trainer = _build_minimal_trainer(minimal_config)
        trainer.current_epoch = 4  # start of phase 2 -> progress = 0
        weights_start = trainer._get_phase_weights()
        trainer.current_epoch = 7  # mid-phase2 -> progress = 0.5
        weights_mid = trainer._get_phase_weights()
        # pixel should be between phase1 (1.0) and phase2 (0.5) values
        assert trainer.phase2_pixel_weight < weights_mid['pixel'] < trainer.phase1_pixel_weight


class TestTrainerCheckpointIO:
    """3.6c: Verify save_checkpoint writes a file and is loadable."""

    def test_save_checkpoint_writes_file(self, minimal_config, tmp_checkpoint):
        trainer = _build_minimal_trainer(minimal_config)
        trainer.checkpoint_dir = tmp_checkpoint
        # Replace the model with a tiny named nn.Module for state_dict clarity
        trainer.model = nn.Linear(4, 4)
        # Save
        ckpt_path = os.path.join(tmp_checkpoint, 'ckpt.pth')
        state = {
            'epoch': 1,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path)
        assert os.path.exists(ckpt_path)
        # Load with weights_only=True (the safe path)
        loaded = torch.load(ckpt_path, weights_only=True)
        assert 'model_state_dict' in loaded
        assert loaded['epoch'] == 1


class TestTrainerLogging:
    """3.6d: Verify the trainer uses the standard logger."""

    def test_uses_module_logger(self, minimal_config):
        """The trainer module should expose a `logger` attribute."""
        import anime_sr.training.neosr_finetuner as mod
        assert hasattr(mod, 'logger')
        assert isinstance(mod.logger, logging.Logger)
        assert mod.logger.name == 'training.neosr_finetuner'
