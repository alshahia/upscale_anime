"""
Unit tests for the DegradationPipeline PRESETS and the new 'apisr' preset.

The 'apisr' preset (Phase G.2) bundles the v7 APISR-aligned degradation
recipe under a single named mode. It must:

  - Instantiate via ``DegradationPipeline(mode='apisr', ...)`` without crash
  - Default ``two_stage`` to True
  - Default ``shuffled_resize`` to True
  - Default ``degrade_before_crop`` to True
  - Default ``compression_stage2`` to ``[avif, h264, h265, jpeg]`` (the v7 list)

Tests use small (32x32) random uint8 images and never invoke ffmpeg. The
CompressionPipeline codec list is stored on the DegradationPipeline but
not actually exercised against video codecs; the v7 codec list lives in
``pipe._stage2_types`` and we only assert membership.

Tests skip gracefully if ``data.degradation_pipeline`` cannot be imported
(handled by the ``pytest.importorskip`` style guard at the top).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest


# Skip the whole module if the import path is broken in this environment.
try:
    from data.degradation_pipeline import DegradationPipeline, PRESETS
except Exception as _exc:  # pragma: no cover - environment-only guard
    pytest.skip(
        f"data.degradation_pipeline not importable: {_exc}",
        allow_module_level=True,
    )


def _make_img(h: int = 32, w: int = 32, seed: int = 0) -> np.ndarray:
    """Build a deterministic uint8 RGB image for smoke-testing pipelines."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# 1. The 'apisr' preset is registered in PRESETS
# ---------------------------------------------------------------------------

class TestApisrPresetRegistered:
    def test_apisr_in_presets(self):
        """The 'apisr' preset must be a key in the PRESETS dict."""
        assert 'apisr' in PRESETS

    def test_apisr_preset_instantiates(self):
        """Creating a DegradationPipeline(mode='apisr', ...) does not crash."""
        pipe = DegradationPipeline(mode='apisr', scale=4, seed=0)
        assert pipe.mode == 'apisr'
        # Smoke: forward pass on a 32x32 random image must not raise.
        out = pipe(_make_img(seed=0))
        assert out.dtype == np.uint8
        assert out.ndim == 3
        assert out.shape[2] == 3


# ---------------------------------------------------------------------------
# 2. 'apisr' preset defaults match the v7 recipe
# ---------------------------------------------------------------------------

class TestApisrPresetDefaults:
    def test_apisr_codec_list(self):
        """apisr preset's compression_stage2 is [avif, h264, h265, jpeg]."""
        pipe = DegradationPipeline(mode='apisr', scale=4, seed=0)
        assert pipe._stage2_types == ['avif', 'h264', 'h265', 'jpeg']

    def test_apisr_two_stage_default(self):
        """apisr preset sets two_stage=True by default."""
        pipe = DegradationPipeline(mode='apisr', scale=4, seed=0)
        assert pipe.two_stage is True

    def test_apisr_shuffled_default(self):
        """apisr preset sets shuffled_resize=True by default."""
        pipe = DegradationPipeline(mode='apisr', scale=4, seed=0)
        assert pipe.shuffled_resize is True

    def test_apisr_degrade_before_crop_default(self):
        """apisr preset sets degrade_before_crop=True by default."""
        pipe = DegradationPipeline(mode='apisr', scale=4, seed=0)
        assert pipe.degrade_before_crop is True

    def test_apisr_explicit_two_stage_override(self):
        """Passing two_stage=False explicitly must override the preset default."""
        pipe = DegradationPipeline(mode='apisr', two_stage=False, scale=4, seed=0)
        assert pipe.two_stage is False

    def test_apisr_explicit_compression_stage2_override(self):
        """Passing an explicit compression_stage2 must override the preset list."""
        pipe = DegradationPipeline(
            mode='apisr',
            compression_stage2=['jpeg'],
            scale=4,
            seed=0,
        )
        assert pipe._stage2_types == ['jpeg']

    def test_apisr_stage1_default(self):
        """apisr preset's compression_stage1 is [jpeg, webp] (matches v7)."""
        pipe = DegradationPipeline(mode='apisr', scale=4, seed=0)
        assert pipe._stage1_types == ['jpeg', 'webp']


# ---------------------------------------------------------------------------
# 3. Backward compat: existing presets still work
# ---------------------------------------------------------------------------

class TestExistingPresetsStillWork:
    @pytest.mark.parametrize("preset_name", [
        'bicubic', 'light', 'medium', 'heavy', 'anime', 'anime_heavy',
    ])
    def test_existing_preset_instantiates(self, preset_name):
        """All pre-existing presets must still construct without error."""
        pipe = DegradationPipeline(mode=preset_name, scale=4, seed=0)
        assert pipe.mode == preset_name

    @pytest.mark.parametrize("preset_name", [
        'bicubic', 'light', 'medium', 'heavy', 'anime', 'anime_heavy', 'apisr',
    ])
    def test_existing_preset_forward_pass(self, preset_name):
        """Smoke: each preset must accept a 32x32 random image."""
        pipe = DegradationPipeline(mode=preset_name, two_stage=False, scale=4, seed=0)
        out = pipe(_make_img(seed=0))
        assert out.dtype == np.uint8
        assert out.ndim == 3
        assert out.shape[2] == 3

    def test_anime_heavy_backward_compat_default_two_stage(self):
        """anime_heavy (the v6 default) must still default to two_stage=False
        when no preset metadata is present."""
        pipe = DegradationPipeline(mode='anime_heavy', scale=4, seed=0)
        assert pipe.two_stage is False
        assert pipe.shuffled_resize is False
        assert pipe.degrade_before_crop is False

    def test_anime_heavy_explicit_two_stage(self):
        """anime_heavy with explicit two_stage=True still works."""
        pipe = DegradationPipeline(
            mode='anime_heavy',
            two_stage=True,
            compression_stage1=['jpeg', 'webp'],
            compression_stage2=['avif', 'h264', 'h265', 'jpeg'],
            scale=4,
            seed=0,
        )
        assert pipe.two_stage is True
        assert pipe._stage1_types == ['jpeg', 'webp']
        assert pipe._stage2_types == ['avif', 'h264', 'h265', 'jpeg']


# ---------------------------------------------------------------------------
# 4. cfg introspection: the preset dict is consumed cleanly
# ---------------------------------------------------------------------------

class TestApisrCfgIntrospection:
    def test_apisr_cfg_does_not_leak_preset_metadata(self):
        """The preset-level metadata keys must be popped from self.cfg so
        that callers that introspect ``self.cfg`` only see algorithmic
        (probability) knobs."""
        pipe = DegradationPipeline(mode='apisr', scale=4, seed=0)
        # Algorithmic keys ARE present
        assert 'blur_prob' in pipe.cfg
        assert 'noise_prob' in pipe.cfg
        assert 'jpeg_prob' in pipe.cfg
        # Preset-metadata keys are NOT present in cfg (they were popped)
        assert 'two_stage' not in pipe.cfg
        assert 'shuffled_resize' not in pipe.cfg
        assert 'degrade_before_crop' not in pipe.cfg
        assert 'compression_stage1' not in pipe.cfg
        assert 'compression_stage2' not in pipe.cfg

    def test_apisr_preset_dict_preserves_its_metadata(self):
        """Importing PRESETS should see the apisr preset WITH its metadata
        (the pop happens in __init__, not at module import)."""
        apisr = PRESETS['apisr']
        assert apisr['two_stage'] is True
        assert apisr['shuffled_resize'] is True
        assert apisr['degrade_before_crop'] is True
        assert apisr['compression_stage1'] == ['jpeg', 'webp']
        assert apisr['compression_stage2'] == ['avif', 'h264', 'h265', 'jpeg']
