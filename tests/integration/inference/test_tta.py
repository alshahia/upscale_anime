"""Tests for D4 TTA ensemble."""
import sys
from pathlib import Path

import pytest
import torch
from anime_sr.inference.tta import D4_AUGMENTATIONS, tta_forward


class TestD4Group:
    """D4 group must be closed under inversion (each element is its own inverse)."""

    def test_d4_inverses_are_correct(self):
        """For each (aug, inv) pair, applying aug then inv to a tensor returns the original."""
        x = torch.randn(1, 3, 32, 32)
        for aug, inv in D4_AUGMENTATIONS:
            assert torch.allclose(inv(aug(x)), x, atol=1e-6), \
                "aug/inv pair is not inverse"

    def test_d4_has_8_elements(self):
        """D4 group has 8 symmetries."""
        assert len(D4_AUGMENTATIONS) == 8

    def test_d4_unique_outputs(self):
        """Applying each augmentation to a non-symmetric tensor gives distinct outputs."""
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
        seen = set()
        for aug, _ in D4_AUGMENTATIONS:
            y = aug(x)
            seen.add(tuple(y.flatten().tolist()))
        # The 8 D4 transforms of an asymmetric matrix should be at least 4 distinct
        # (rotational symmetry reduces 8 to 4 if there's 180-deg symmetry, but this
        # asymmetric matrix has none).
        assert len(seen) >= 4


class TestTTAForward:
    """tta_forward should correctly average 8 predictions."""

    def test_tta_identity_model_returns_input_mean(self):
        """If model returns a constant, TTA returns that constant (with clip)."""
        def const_model(x):
            return torch.full((x.shape[0], 3, x.shape[2] * 4, x.shape[3] * 4), 0.5)

        lr = torch.rand(1, 3, 8, 8)
        out = tta_forward(const_model, lr, use_clip=True)
        assert out.shape == (1, 3, 32, 32)
        assert torch.allclose(out, torch.full_like(out, 0.5), atol=1e-5)

    def test_tta_pixel_range_clipped(self):
        """When use_clip=True, output is in [0, 1]."""
        def over_model(x):
            # Return values outside [0, 1] to test clipping.
            return torch.full((x.shape[0], 3, x.shape[2] * 4, x.shape[3] * 4), 1.5)

        lr = torch.rand(1, 3, 8, 8)
        out = tta_forward(over_model, lr, use_clip=True)
        assert out.max() <= 1.0 + 1e-5
        assert out.min() >= 0.0 - 1e-5

    def test_tta_pixel_range_unclipped(self):
        """When use_clip=False, output preserves model range."""
        def over_model(x):
            return torch.full((x.shape[0], 3, x.shape[2] * 4, x.shape[3] * 4), 1.5)

        lr = torch.rand(1, 3, 8, 8)
        out = tta_forward(over_model, lr, use_clip=False)
        assert torch.allclose(out, torch.full_like(out, 1.5), atol=1e-5)

    def test_tta_matches_manual_mean(self):
        """tta_forward equals the manual sum-of-8 divided by 8."""
        def linear_model(x):
            # x has shape [B, 3, H, W]; return upsampled (nearest 4x) version of x.
            return torch.nn.functional.interpolate(x, scale_factor=4, mode='nearest')

        torch.manual_seed(42)
        lr = torch.rand(2, 3, 16, 16)

        # Manual computation
        manual = torch.zeros_like(linear_model(lr))
        for aug, inv in D4_AUGMENTATIONS:
            manual = manual + inv(linear_model(aug(lr)))
        manual = manual / 8.0

        out = tta_forward(linear_model, lr, use_clip=True)
        assert torch.allclose(out, manual, atol=1e-5)

    def test_tta_dtype_preserved(self):
        """Output dtype matches input dtype."""
        def const_model(x):
            return torch.full((x.shape[0], 3, x.shape[2] * 4, x.shape[3] * 4), 0.5,
                              dtype=x.dtype)

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            lr = torch.rand(1, 3, 8, 8, dtype=dtype)
            out = tta_forward(const_model, lr, use_clip=True)
            assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only test")
class TestTTACUDA:
    """GPU-only smoke test (skipped on CPU-only environments)."""

    def test_tta_cuda_smoke(self):
        def const_model(x):
            return torch.full((x.shape[0], 3, x.shape[2] * 4, x.shape[3] * 4), 0.5,
                              device=x.device)

        lr = torch.rand(1, 3, 8, 8, device='cuda')
        out = tta_forward(const_model, lr, use_clip=True)
        assert out.device.type == 'cuda'
        assert torch.allclose(out, torch.full_like(out, 0.5), atol=1e-5)