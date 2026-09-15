"""Tests for the VRAM budget estimator."""
from pathlib import Path


def test_vram_dtype_bytes():
    from apps.anime_upscaler_gui.anime_upscaler_gui.vram import dtype_bytes
    assert dtype_bytes(True) == 2
    assert dtype_bytes(False) == 4


def test_vram_estimate_small_does_not_trigger():
    import torch
    from apps.anime_upscaler_gui.anime_upscaler_gui.vram import estimate
    # A 64x64 -> 256x256 upscale is tiny; should never trigger regardless of GPU size
    b = estimate(64, 64, 4, True)
    assert b.output_bytes == 256 * 256 * 3 * 2
    assert b.triggered is False  # 1 MB is well within any GPU's VRAM
    if not torch.cuda.is_available():
        assert b.free_bytes == 0


def test_vram_suggest_downscale_returns_tuple():
    from apps.anime_upscaler_gui.anime_upscaler_gui.vram import suggest_downscale
    h, w = suggest_downscale(64, 64, 4, True)
    assert isinstance(h, int) and isinstance(w, int)
    assert h > 0 and w > 0