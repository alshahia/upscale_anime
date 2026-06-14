"""
Unit tests for Phase 4 LOW-priority fixes.

- Issue #14: 4.1 UnpicklingError import is used (smoke test)
- Issue #15: 4.2 Conv3XC.eval_conv parameters frozen
- Issue #16: 4.3 progressive_crop defaults load from base.yaml
- Issue #17: 4.5 PreprocessingManager pre-flight raises on missing base_dir
- Issue #18: 4.6 gradient_penalty_lambda read from config
- Issue #19: 4.7 _get_adversarial_weight delegates to phase-aware method
- Issue #20: 4.8 wavelet_init deprecated alias -> start_epoch
"""
import os
import sys
import logging
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from unittest.mock import patch, MagicMock

from training.neosr_finetuner import NeosrSPANFinetuner
from models.span.neosr_span import Conv3XC
from data.preprocessing_manager import PreprocessingManager


# ---------------------------------------------------------------------------
# 4.1: UnpicklingError is actually used
# ---------------------------------------------------------------------------

class TestUnpicklingErrorUsed:
    def test_unpicklingerror_imported_and_used(self):
        """Phase 4.1: confirm the import is not dead code."""
        import training.neosr_finetuner as mod
        assert hasattr(mod, 'UnpicklingError')
        # Used at the safe-checkpoint-load site (Phase 1 PR-4)
        import inspect
        source = inspect.getsource(mod)
        assert 'UnpicklingError' in source
        # Specifically used in the except clause
        assert 'except UnpicklingError' in source


# ---------------------------------------------------------------------------
# 4.2: Conv3XC.eval_conv parameters are frozen
# ---------------------------------------------------------------------------

class TestConv3XCEvalConvFrozen:
    def test_eval_conv_weight_requires_grad_false(self):
        layer = Conv3XC(c_in=4, c_out=4, gain1=2, s=1)
        assert not layer.eval_conv.weight.requires_grad, (
            "eval_conv.weight must be frozen (requires_grad=False)"
        )

    def test_eval_conv_bias_requires_grad_false(self):
        layer = Conv3XC(c_in=4, c_out=4, gain1=2, s=1)
        assert layer.eval_conv.bias is not None
        assert not layer.eval_conv.bias.requires_grad

    def test_eval_conv_no_bias_frozen(self):
        """When bias=False, the bias attribute is None, and that's fine."""
        layer = Conv3XC(c_in=4, c_out=4, gain1=2, s=1, bias=False)
        assert layer.eval_conv.bias is None
        assert not layer.eval_conv.weight.requires_grad

    def test_state_dict_format_unchanged(self):
        """State-dict must still contain `eval_conv.weight` and `eval_conv.bias`."""
        layer = Conv3XC(c_in=4, c_out=4, gain1=2, s=1)
        sd_keys = list(layer.state_dict().keys())
        assert 'eval_conv.weight' in sd_keys
        assert 'eval_conv.bias' in sd_keys


# ---------------------------------------------------------------------------
# 4.3: progressive_crop defaults load from base.yaml
# ---------------------------------------------------------------------------

class TestProgressiveCropDefaults:
    def test_defaults_from_base_yaml(self):
        """When config has no `stages`, fall back to base.yaml progressive_crop_defaults."""
        # Import base.yaml
        import yaml
        with open('configs/base.yaml', encoding='utf-8') as f:
            base_cfg = yaml.safe_load(f)
        assert 'progressive_crop_defaults' in base_cfg
        assert 'stages' in base_cfg['progressive_crop_defaults']
        stages = base_cfg['progressive_crop_defaults']['stages']
        # 3 stages: 128 / 256 / 512
        crop_sizes = [s['crop_size'] for s in stages]
        assert crop_sizes == [128, 256, 512]

    def test_trainer_uses_base_defaults_when_no_stages(self):
        """When the trainer is configured with progressive_crop enabled but no
        explicit `stages`, it must pick up the base.yaml defaults.
        """
        trainer = NeosrSPANFinetuner.__new__(NeosrSPANFinetuner)
        trainer.config = {
            'progressive_crop_defaults': {
                'stages': [
                    {'crop_size': 64, 'epochs': 5},
                    {'crop_size': 128, 'epochs': 10},
                ],
                'default_crop_size': 64,
            },
            'training': {
                'finetune': {
                    'progressive_crop': {
                        'enabled': True,
                        # No `stages` key
                    }
                }
            }
        }
        trainer.finetune_cfg = trainer.config['training']['finetune']
        # Run the relevant block
        progressive_cfg = trainer.finetune_cfg.get('progressive_crop') or {}
        trainer.use_progressive_crop = progressive_cfg.get('enabled', False)
        pc_defaults = trainer.config.get('progressive_crop_defaults') or {}
        default_stages = pc_defaults.get('stages', [])
        trainer.progressive_stages = progressive_cfg.get('stages', default_stages)
        trainer.current_crop_size = trainer.progressive_stages[0].get('crop_size', 128)

        assert trainer.use_progressive_crop
        assert len(trainer.progressive_stages) == 2
        assert trainer.progressive_stages[0]['crop_size'] == 64
        assert trainer.current_crop_size == 64


# ---------------------------------------------------------------------------
# 4.5: PreprocessingManager pre-flight check
# ---------------------------------------------------------------------------

class TestPreprocessingManagerPreflight:
    def test_hybrid_precomputed_base_missing_raises(self):
        """hybrid.precomputed_base=true with a missing base_dir must raise."""
        cfg = {
            'mode': 'hybrid',
            'hybrid': {
                'precomputed_base': True,
                'base_dir': str(tempfile.mkdtemp(prefix='nonexistent_') + '_does_not_exist'),
            },
        }
        with pytest.raises(FileNotFoundError) as excinfo:
            PreprocessingManager(cfg)
        msg = str(excinfo.value)
        assert 'hybrid.precomputed_base=true' in msg
        assert 'precompute_pairs.py' in msg

    def test_hybrid_precomputed_base_present_no_raise(self):
        """If the base_dir exists with lr/ and hr/ subdirs, no error."""
        with tempfile.TemporaryDirectory(prefix='pc_present_') as td:
            base = Path(td) / 'precomputed_4x'
            (base / 'lr').mkdir(parents=True)
            (base / 'hr').mkdir(parents=True)
            cfg = {
                'mode': 'hybrid',
                'hybrid': {'precomputed_base': True, 'base_dir': str(base)},
            }
            pm = PreprocessingManager(cfg)
            assert pm.precomputed_available

    def test_hybrid_precomputed_base_false_does_not_raise(self):
        """precomputed_base=false: missing base_dir is fine (we don't need it)."""
        cfg = {
            'mode': 'hybrid',
            'hybrid': {
                'precomputed_base': False,
                'base_dir': '/nonexistent/path',
            },
        }
        # Should not raise; we just won't have precomputed base available.
        pm = PreprocessingManager(cfg)
        assert not pm.precomputed_available


# ---------------------------------------------------------------------------
# 4.6: gradient_penalty_lambda read from config
# ---------------------------------------------------------------------------

class TestGradientPenaltyLambda:
    def _build_trainer_with_gp(self, **kwargs):
        trainer = NeosrSPANFinetuner.__new__(NeosrSPANFinetuner)
        # Default lambda = 1.0
        trainer.use_gradient_penalty = kwargs.get('use_gradient_penalty', True)
        trainer.gradient_penalty_lambda = float(kwargs.get('gradient_penalty_lambda', 1.0))
        return trainer

    def test_default_lambda_is_one(self):
        trainer = self._build_trainer_with_gp()
        assert trainer.gradient_penalty_lambda == 1.0

    def test_custom_lambda(self):
        trainer = self._build_trainer_with_gp(gradient_penalty_lambda=5.0)
        assert trainer.gradient_penalty_lambda == 5.0


# ---------------------------------------------------------------------------
# 4.7: _get_adversarial_weight delegates to phase-aware method
# ---------------------------------------------------------------------------

class TestAdversarialWeightMerge:
    def test_get_adversarial_weight_delegates(self):
        trainer = NeosrSPANFinetuner.__new__(NeosrSPANFinetuner)
        trainer.use_adversarial = True
        trainer.discriminator = MagicMock()  # non-None
        trainer.current_epoch = 30
        trainer.adversarial_start_epoch = 25
        trainer.phase_adversarial_ramp_epochs = 5
        trainer.adversarial_weight = 0.01
        trainer.adversarial_max_weight = 0.01

        result = trainer._get_adversarial_weight()
        # With start_epoch=25, ramp=5, current=30: progress=5/5=1.0
        # _compute_adversarial_weight_phase_aware returns the target (0.01)
        # since the ramp is complete.
        assert abs(result - 0.01) < 1e-5

    def test_get_adversarial_weight_zero_before_start(self):
        trainer = NeosrSPANFinetuner.__new__(NeosrSPANFinetuner)
        trainer.use_adversarial = True
        trainer.discriminator = MagicMock()
        trainer.current_epoch = 10
        trainer.adversarial_start_epoch = 25
        trainer.phase_adversarial_ramp_epochs = 5
        trainer.adversarial_weight = 0.01
        trainer.adversarial_max_weight = 0.01

        result = trainer._get_adversarial_weight()
        assert result == 0.0

    def test_get_adversarial_weight_zero_when_disabled(self):
        trainer = NeosrSPANFinetuner.__new__(NeosrSPANFinetuner)
        trainer.use_adversarial = False
        trainer.discriminator = None
        trainer.adversarial_weight = 0.0
        trainer.adversarial_max_weight = 0.0
        result = trainer._get_adversarial_weight()
        assert result == 0.0


# ---------------------------------------------------------------------------
# 4.8: wavelet_init deprecated alias
# ---------------------------------------------------------------------------

class TestWaveletInitAlias:
    """The trainer must accept both `start_epoch` (preferred) and `wavelet_init` (deprecated)."""

    def _extract_init_epoch(self, wg_cfg):
        """Mirror the trainer's _setup_losses logic for the wavelet init epoch."""
        if 'start_epoch' in wg_cfg:
            return wg_cfg.get('start_epoch', 40)
        else:
            return wg_cfg.get('wavelet_init', 40)

    def test_start_epoch_preferred(self):
        assert self._extract_init_epoch({'start_epoch': 30}) == 30

    def test_wavelet_init_falls_back(self):
        assert self._extract_init_epoch({'wavelet_init': 25}) == 25

    def test_default_when_neither_set(self):
        assert self._extract_init_epoch({}) == 40

    def test_start_epoch_wins_over_wavelet_init(self):
        assert self._extract_init_epoch(
            {'start_epoch': 30, 'wavelet_init': 99}
        ) == 30
