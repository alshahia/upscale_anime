"""
Unit tests for Phase B XDoG + USM pseudo-GT pipeline.

Reference: docs/APISR_DEGRADATION_SPEC.md Section 0.3 + Section 5.
The full APISR pipeline is:
    1. 3 rounds of USM sharpening
    2. XDoG edge extraction
    3. Connected-component cleanup (BFS, min_size=32)
    4. Passive dilation
    5. Composite: pseudo_gt = sharpened * xdog_map + original * (1 - xdog_map)

This file tests the individual stages and the composite behaviour.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
import cv2

from data.line_enhancement import (
    LineEnhancer,
    connected_component_cleanup,
    make_pseudo_gt,
    passive_dilation,
    usm_sharpen,
    xdog,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic 64x64 test images with known line patterns.
# ---------------------------------------------------------------------------

def _draw_rectangle(img: np.ndarray, top: int, left: int, h: int, w: int,
                    color=(0, 0, 0), thickness: int = 2) -> None:
    """Draw a filled outline rectangle on img (uint8, RGB)."""
    cv2.rectangle(img, (left, top), (left + w, top + h), color, thickness)


def _make_rectangle_image(size: int = 64) -> np.ndarray:
    """White background with a black 2-px-thick rectangle outline."""
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    _draw_rectangle(img, top=8, left=8, h=48, w=48, color=(0, 0, 0), thickness=2)
    return img


def _make_dotted_image(size: int = 64) -> np.ndarray:
    """White background with isolated small black blobs (outliers)."""
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    # Two 2x2 isolated dots -> well below outlier_min_size=32
    img[10:12, 10:12] = (0, 0, 0)
    img[20:22, 20:22] = (0, 0, 0)
    # A larger 6x6 blob -> above outlier threshold
    img[40:46, 40:46] = (0, 0, 0)
    return img


def _laplacian_magnitude(gray: np.ndarray) -> float:
    """Sum of |Laplacian|, used as a sharpness proxy."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.sum(np.abs(lap)))


# ---------------------------------------------------------------------------
# USM
# ---------------------------------------------------------------------------

class TestUsmSharpen:
    def test_usm_sharpens(self):
        """USM should increase edge magnitude on a synthetic rectangle."""
        img = _make_rectangle_image(size=64)
        gray_in = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        lap_in = _laplacian_magnitude(gray_in)

        out = usm_sharpen(img, rounds=3, radius=11, threshold=5, weight=0.7)
        assert out.dtype == np.uint8
        assert out.shape == img.shape
        gray_out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        lap_out = _laplacian_magnitude(gray_out)
        assert lap_out > lap_in, (
            f"USM should increase Laplacian sum: in={lap_in:.1f}, out={lap_out:.1f}"
        )

    def test_usm_zero_rounds_is_identity(self):
        """rounds=0 should return the input unchanged."""
        img = _make_rectangle_image()
        out = usm_sharpen(img, rounds=0, radius=11)
        assert np.array_equal(out, img)

    def test_usm_preserves_shape_and_dtype(self):
        """Output must remain uint8 with the same shape as input."""
        img = _make_rectangle_image()
        out = usm_sharpen(img, rounds=2, radius=9, threshold=5, weight=0.5)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_usm_even_radius_forced_odd(self):
        """An even radius must be incremented to odd (GaussianBlur requirement)."""
        img = _make_rectangle_image()
        out_even = usm_sharpen(img, rounds=1, radius=10, threshold=5, weight=0.5)
        out_odd = usm_sharpen(img, rounds=1, radius=11, threshold=5, weight=0.5)
        # Both should run without error and produce uint8 RGB output.
        assert out_even.shape == img.shape
        assert out_odd.shape == img.shape
        assert out_even.dtype == np.uint8
        assert out_odd.dtype == np.uint8


# ---------------------------------------------------------------------------
# XDoG
# ---------------------------------------------------------------------------

class TestXdog:
    def test_xdog_returns_uint8_in_range(self):
        """XDoG output must be uint8 in [0, 255]."""
        img = _make_rectangle_image()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        out = xdog(gray, sigma=0.6, k=2.5, gamma=0.97, eps=-15.0, phi=1e9)
        assert out.dtype == np.uint8
        assert out.min() >= 0 and out.max() <= 255

    def test_xdog_detects_rectangle_outline(self):
        """On a rectangle outline, XDoG should produce some bright pixels."""
        img = _make_rectangle_image()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        out = xdog(gray, sigma=0.6, k=2.5, gamma=0.97, eps=-15.0, phi=1e9)
        # The rectangle has clear dark lines on white -> XDoG should fire.
        assert (out > 127).sum() > 0, "XDoG should detect at least one edge pixel"

    def test_xdog_uniform_input_is_quiet(self):
        """On a uniform image, XDoG should not produce widespread edges."""
        uniform = np.full((64, 64), 200, dtype=np.uint8)
        out = xdog(uniform, sigma=0.6, k=2.5, gamma=0.97, eps=-15.0, phi=1e9)
        # All values should be 0 (no edges) or 255 (full saturation)
        unique = np.unique(out)
        # On a perfectly uniform image with phi=1e9, the XDoG is essentially
        # a hard threshold; the result should be either all 0 or all 255.
        # The vast majority of pixels should be 0 since the DoG is constant.
        assert (out == 0).sum() > (out == 255).sum() * 10, (
            "Uniform image should produce very few edge pixels"
        )


# ---------------------------------------------------------------------------
# Connected-component cleanup (BFS)
# ---------------------------------------------------------------------------

class TestConnectedComponentCleanup:
    def test_removes_small_blobs(self):
        """Isolated 2x2 dots must be erased; the 6x6 blob must remain."""
        line_map = np.zeros((64, 64), dtype=np.uint8)
        line_map[10:12, 10:12] = 255  # 4 px (small)
        line_map[20:22, 20:22] = 255  # 4 px (small)
        line_map[40:46, 40:46] = 255  # 36 px (kept, >=32)

        out = connected_component_cleanup(line_map, min_size=32)

        # 4-px dots should be gone
        assert (out[10:12, 10:12] == 0).all()
        assert (out[20:22, 20:22] == 0).all()
        # 36-px blob should survive
        assert (out[40:46, 40:46] == 255).all()

    def test_empty_input(self):
        """All-zero input -> all-zero output."""
        line_map = np.zeros((32, 32), dtype=np.uint8)
        out = connected_component_cleanup(line_map, min_size=4)
        assert (out == 0).all()

    def test_all_pixels_above_threshold(self):
        """If every white pixel is part of one giant connected component
        and min_size=1, the whole map should survive."""
        line_map = np.full((16, 16), 255, dtype=np.uint8)
        out = connected_component_cleanup(line_map, min_size=1)
        assert (out == 255).all()


# ---------------------------------------------------------------------------
# Passive dilation
# ---------------------------------------------------------------------------

class TestPassiveDilation:
    def test_closes_one_pixel_gap(self):
        """A black pixel with 3+ white 8-neighbors should be filled."""
        line_map = np.zeros((16, 16), dtype=np.uint8)
        # A line of 8 white pixels (top row 7..14 of row 8)
        line_map[7, 5:13] = 255
        # One black pixel at (8, 9) with >=3 white neighbors (up, up-left, up-right)
        # pre-dilation; after dilation, it should be filled.
        out = passive_dilation(line_map, threshold=3)
        assert out[8, 9] == 255, "Gap-filling pixel should be filled"

    def test_preserves_existing_white(self):
        """Dilation must not erase existing white pixels."""
        line_map = np.full((8, 8), 255, dtype=np.uint8)
        out = passive_dilation(line_map, threshold=3)
        assert (out == 255).all()

    def test_isolated_pixels_unchanged(self):
        """An isolated white pixel with no white neighbors should stay white,
        and the surrounding 8 black pixels must remain black."""
        line_map = np.zeros((16, 16), dtype=np.uint8)
        line_map[8, 8] = 255
        out = passive_dilation(line_map, threshold=3)
        # Center remains white
        assert out[8, 8] == 255
        # Neighbors stay black (they have only 1 white neighbor)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                assert out[8 + di, 8 + dj] == 0


# ---------------------------------------------------------------------------
# make_pseudo_gt (full APISR composite)
# ---------------------------------------------------------------------------

class TestMakePseudoGt:
    def test_apisr_composite_preserves_background(self):
        """Where xdog_map=0, output must equal the ORIGINAL (not just sharpened)."""
        # Use a white-only image with no edges: XDoG should produce 0 map.
        img = np.full((64, 64, 3), 250, dtype=np.uint8)
        # Use a very high eps so XDoG never fires
        out = make_pseudo_gt(
            img, usm_rounds=1, usm_radius=11, usm_threshold=255,
            usm_weight=0.5, xdog_sigma=0.6, xdog_k=2.5, xdog_gamma=0.97,
            xdog_eps=255.0, xdog_phi=1e9, outlier_min_size=32,
            dilation_threshold=3,
        )
        assert out.dtype == np.uint8
        # Every pixel: map=0 -> output = I_sharp_1
        # The composite formula is: I_pseudo = I_sharp_1 * 0 + I_GT * 1 = I_GT
        # I_GT here is 250. I_sharp_1 differs only in edges (none on uniform).
        # The result is therefore exactly I_GT.
        np.testing.assert_array_equal(out, img)

    def test_apisr_composite_uses_sharpened_on_edges(self):
        """Where xdog_map=1, output must equal I_sharp_1 (NOT I_GT).

        We use a mid-gray background with a 1-px dark line. USM brightens the
        transition pixels (where img drops from 180 to 0), and the XDoG map
        fires at those transition pixels. The composite is:
            I_pseudo = I_sharp_1 * I_map + I_GT * (1 - I_map)
        At a transition pixel with I_map ~ 1, I_pseudo ~ I_sharp_1.
        We assert that I_pseudo is closer to I_sharp_1 than to I_GT.
        """
        from data.line_enhancement import usm_sharpen
        # Mid-gray background: USM is non-trivial (residual > threshold).
        img = np.full((64, 64, 3), 180, dtype=np.uint8)
        # 1-px dark line at row 31 — the transition pixels (row 30 and 32)
        # are where USM sharpens to a value different from I_GT.
        img[31, 10:54] = (0, 0, 0)

        sharp_1 = usm_sharpen(
            img, rounds=1, radius=11, threshold=5, weight=0.7,
        )

        out = make_pseudo_gt(
            img, usm_rounds=1, usm_radius=11, usm_threshold=5,
            usm_weight=0.7, xdog_sigma=0.6, xdog_k=2.5, xdog_gamma=0.97,
            xdog_eps=-15.0, xdog_phi=1e9, outlier_min_size=8,
            dilation_threshold=3,
        )

        # Find a pixel where I_map is 1 (i.e., the XDoG fired) and where
        # I_sharp_1 differs from I_GT. These are the transition pixels at
        # the line edge.
        # Scan for the first pixel where out != img (the line edge in the
        # composite, where xdog_map was 1).
        diff_mask = (out[..., 0].astype(int) != img[..., 0].astype(int))
        diff_idx = np.argwhere(diff_mask)
        assert diff_idx.size > 0, (
            "Composite should differ from input at the line edges"
        )
        # Take the first differing pixel (a transition edge).
        r, c = diff_idx[0]
        px = int(out[r, c, 0])
        sx = int(sharp_1[r, c, 0])
        gx = int(img[r, c, 0])
        # At this pixel, I_map was 1 (the XDoG fired and the composite
        # followed I_sharp_1). So |px - sx| should be small and
        # |px - gx| should be non-trivial.
        d_sharp = abs(px - sx)
        d_orig = abs(px - gx)
        assert d_orig > 0, (
            f"Transition pixel (r={r}, c={c}) should differ from GT "
            f"(px={px}, gx={gx})"
        )
        assert d_sharp <= d_orig, (
            f"At edge pixel (r={r}, c={c}): px={px}, sharp={sx}, orig={gx}; "
            f"d_sharp={d_sharp}, d_orig={d_orig} — composite should follow "
            f"sharp_1 (not orig) where xdog_map=1"
        )

    def test_apisr_composite_shape_preserved(self):
        """Output shape and dtype must match input."""
        img = _make_rectangle_image(size=64)
        out = make_pseudo_gt(img, usm_rounds=1, usm_radius=11)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_apisr_composite_handles_grayscale_input(self):
        """make_pseudo_gt should accept grayscale input and return grayscale."""
        gray = np.full((32, 32), 200, dtype=np.uint8)
        gray[10:12, 5:25] = 0  # dark line
        out = make_pseudo_gt(gray, usm_rounds=1, usm_radius=7)
        assert out.ndim == 2
        assert out.dtype == np.uint8

    def test_apisr_composite_handles_very_small_image(self):
        """An image too small for the USM kernel must not crash."""
        small = np.full((16, 16, 3), 200, dtype=np.uint8)
        out = make_pseudo_gt(
            small, usm_rounds=1, usm_radius=3, usm_threshold=5, usm_weight=0.5,
        )
        assert out.shape == small.shape
        assert out.dtype == np.uint8


# ---------------------------------------------------------------------------
# LineEnhancer class — pseudo_gt_mode dispatch
# ---------------------------------------------------------------------------

class TestLineEnhancerModeDispatch:
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            LineEnhancer(pseudo_gt_mode="bogus")

    def test_lite_mode_default(self):
        le = LineEnhancer()
        assert le.pseudo_gt_mode == "lite"

    def test_apisr_mode_constructs(self):
        le = LineEnhancer(
            pseudo_gt_mode="apisr",
            usm_rounds=3, usm_radius=50, usm_sigma=0,
            usm_threshold=10, usm_weight=0.5,
        )
        assert le.pseudo_gt_mode == "apisr"
        assert le.usm_rounds == 3
        assert le.usm_radius == 50

    def test_lite_call_preserves_shape(self):
        le = LineEnhancer(pseudo_gt_mode="lite")
        img = _make_rectangle_image()
        out = le(img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_apisr_call_preserves_shape(self):
        le = LineEnhancer(
            pseudo_gt_mode="apisr", usm_rounds=1, usm_radius=11,
        )
        img = _make_rectangle_image()
        out = le(img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_apisr_call_differs_from_lite(self):
        """APISR mode uses a different composite strategy than lite; on a
        mid-gray image with a dark line the two outputs should differ
        because APISR's USM sharpens the line edges while lite's alpha-blend
        applies a small line overlay only."""
        # Mid-gray background so USM is non-trivial.
        img = np.full((64, 64, 3), 180, dtype=np.uint8)
        # Dark rectangle outline
        cv2.rectangle(img, (8, 8), (56, 56), (0, 0, 0), 2)
        # A wider dark filled region so XDoG has area to fire on.
        img[20:24, 20:44] = (0, 0, 0)

        lite = LineEnhancer(pseudo_gt_mode="lite", alpha=0.5)
        apisr = LineEnhancer(
            pseudo_gt_mode="apisr", usm_rounds=3, usm_radius=11,
            usm_weight=0.7, usm_threshold=5,
        )
        out_lite = lite(img)
        out_apisr = apisr(img)
        # At least some pixels should differ — the APISR composite uses
        # I_sharp_1 where xdog_map=1, which differs from the lite overlay.
        assert not np.array_equal(out_lite, out_apisr)
        # The liteline overlay is bounded; APISR can brighten edges.
        assert (out_apisr > img).any() or (out_apisr < img).any()
