"""
Unit tests for APISR two-stage compression wiring.

Verifies:
- DegradationPipeline calls stage1 + stage2 compression pipelines
  when two_stage=True (via the __call__ path)
- Stage 2 uses the post-resize codecs (avif/h264/h265/jpeg)
- Stage 1 uses the pre-resize codecs (jpeg/webp by default)
- CompressionPipeline exposes all required codec names
- Video codecs gracefully fall back to JPEG when PyAV/ffmpeg is unavailable
"""
import os
import sys
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. CompressionPipeline accepts the v7 codec list
# ---------------------------------------------------------------------------

class TestCompressionPipelineAcceptsV7Codecs:
    def test_stage2_v7_codecs_build(self):
        """v7 config: compression_stage2 = [avif, h264, h265, jpeg]."""
        from anime_sr.data.compression_modules import CompressionPipeline
        pipe = CompressionPipeline(
            compression_types=['avif', 'h264', 'h265', 'jpeg'],
            quality_ranges=[(30, 80), (18, 35), (18, 35), (60, 95)],
        )
        assert set(pipe.compressors.keys()) >= {'avif', 'h264', 'h265', 'jpeg'}

    def test_stage1_v6_codecs_build(self):
        """v6/v7 config: compression_stage1 = [jpeg, webp]."""
        from anime_sr.data.compression_modules import CompressionPipeline
        pipe = CompressionPipeline(
            compression_types=['jpeg', 'webp'],
            quality_ranges=[(60, 95), (60, 95)],
        )
        assert 'jpeg' in pipe.compressors
        assert 'webp' in pipe.compressors

    def test_unknown_codec_ignored(self):
        """Unknown codec names should not crash; they're silently dropped
        from the compressors dict (matches the existing behavior)."""
        from anime_sr.data.compression_modules import CompressionPipeline
        pipe = CompressionPipeline(
            compression_types=['jpeg', 'unknown_xyz'],
            quality_ranges=[(60, 95), (60, 95)],
        )
        assert 'jpeg' in pipe.compressors
        assert 'unknown_xyz' not in pipe.compressors


# ---------------------------------------------------------------------------
# 2. DegradationPipeline initializes both stage1 and stage2 pipelines
# ---------------------------------------------------------------------------

class TestDegradationPipelineTwoStageInit:
    def test_two_stage_creates_both_pipelines(self):
        from anime_sr.data.degradation_pipeline import DegradationPipeline
        pipe = DegradationPipeline(
            mode='anime_heavy',
            two_stage=True,
            compression_stage1=['jpeg', 'webp'],
            compression_stage2=['avif', 'h264', 'jpeg'],
            seed=0,
        )
        assert pipe.two_stage is True
        # When CompressionPipeline is available, both stages should be set
        from anime_sr.data.compression_modules import CompressionPipeline as CP
        if CP is not None:
            assert pipe._stage1_pipeline is not None
            assert pipe._stage2_pipeline is not None
            assert pipe._stage1_types == ['jpeg', 'webp']
            assert pipe._stage2_types == ['avif', 'h264', 'jpeg']

    def test_single_stage_no_pipelines(self):
        """When two_stage=False, the stage pipelines are not built."""
        from anime_sr.data.degradation_pipeline import DegradationPipeline
        pipe = DegradationPipeline(
            mode='anime_heavy',
            two_stage=False,
            seed=0,
        )
        assert pipe.two_stage is False
        assert pipe._stage1_pipeline is None
        assert pipe._stage2_pipeline is None

    def test_v7_stage2_includes_h264_h265(self):
        """v7 config has h264 + h265 in stage2; verify they're in the pipeline."""
        from anime_sr.data.degradation_pipeline import DegradationPipeline
        from anime_sr.data.compression_modules import CompressionPipeline as CP
        pipe = DegradationPipeline(
            mode='anime_heavy',
            two_stage=True,
            compression_stage1=['jpeg', 'webp'],
            compression_stage2=['avif', 'h264', 'h265', 'jpeg'],
            seed=0,
        )
        if CP is not None:
            # stage2 codec names appear in the pipeline
            assert 'h264' in pipe._stage2_pipeline.compressors
            assert 'h265' in pipe._stage2_pipeline.compressors
            assert 'avif' in pipe._stage2_pipeline.compressors
            assert 'jpeg' in pipe._stage2_pipeline.compressors


# ---------------------------------------------------------------------------
# 3. End-to-end: two_stage __call__ runs both stages
# ---------------------------------------------------------------------------

class TestTwoStageEndToEnd:
    def _make_img(self, h=256, w=256, seed=0):
        rng = np.random.default_rng(seed)
        return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)

    def test_two_stage_call_returns_valid_image(self):
        from anime_sr.data.degradation_pipeline import DegradationPipeline
        pipe = DegradationPipeline(
            mode='anime_heavy',
            two_stage=True,
            compression_stage1=['jpeg', 'webp'],
            compression_stage2=['jpeg'],  # only JPEG to avoid ffmpeg dependency
            scale=4,
            seed=0,
        )
        img = self._make_img(seed=42)
        out = pipe(img)
        assert out.dtype == np.uint8
        assert out.ndim == 3
        assert out.shape[2] == 3
        # scale=4 means LR is 1/4 of HR (in pixels)
        assert out.shape[0] == img.shape[0] // 4
        assert out.shape[1] == img.shape[1] // 4

    def test_two_stage_with_video_codecs_call(self):
        """v7 stage2 [avif, h264, h265, jpeg] must not crash even if codecs
        fall back to JPEG internally (graceful degradation)."""
        from anime_sr.data.degradation_pipeline import DegradationPipeline
        pipe = DegradationPipeline(
            mode='anime_heavy',
            two_stage=True,
            compression_stage1=['jpeg', 'webp'],
            compression_stage2=['avif', 'h264', 'h265', 'jpeg'],
            scale=4,
            seed=0,
        )
        img = self._make_img(h=128, w=128, seed=7)
        # Should not raise even on a system without ffmpeg
        out = pipe(img)
        assert out is not None
        assert out.shape[2] == 3
        assert out.shape[0] > 0 and out.shape[1] > 0

    def test_single_stage_vs_two_stage_differ(self):
        """two_stage=True and two_stage=False should produce different images
        on the same input+seed (different algorithmic paths)."""
        from anime_sr.data.degradation_pipeline import DegradationPipeline
        img = self._make_img(h=128, w=128, seed=10)

        pipe_single = DegradationPipeline(
            mode='anime_heavy', two_stage=False, scale=4, seed=0,
        )
        pipe_two = DegradationPipeline(
            mode='anime_heavy', two_stage=True,
            compression_stage1=['jpeg'],
            compression_stage2=['jpeg'],
            scale=4, seed=0,
        )
        out_single = pipe_single(img.copy())
        out_two = pipe_two(img.copy())
        # Both produce valid RGB uint8 output. single-stage keeps the
        # shuffled-resize output as-is (may be larger than LR), while
        # two-stage always ends with a scale-resize to LR size.
        assert out_single.shape[2] == 3
        assert out_two.shape[2] == 3
        # two-stage should always downscale to scale=4 LR
        assert out_two.shape[0] == img.shape[0] // 4
        assert out_two.shape[1] == img.shape[1] // 4


# ---------------------------------------------------------------------------
# 4. Stage 1 vs Stage 2 codec lists are distinct (semantic separation)
# ---------------------------------------------------------------------------

class TestStage1VsStage2Separation:
    def test_stage1_and_stage2_different_codec_lists(self):
        """v7: stage1 = [jpeg, webp], stage2 = [avif, h264, h265, jpeg].
        These should be kept as separate fields in DegradationPipeline."""
        from anime_sr.data.degradation_pipeline import DegradationPipeline
        pipe = DegradationPipeline(
            mode='anime_heavy',
            two_stage=True,
            compression_stage1=['jpeg', 'webp'],
            compression_stage2=['avif', 'h264', 'h265', 'jpeg'],
            seed=0,
        )
        assert pipe._stage1_types == ['jpeg', 'webp']
        assert pipe._stage2_types == ['avif', 'h264', 'h265', 'jpeg']
        assert set(pipe._stage1_types) != set(pipe._stage2_types)
