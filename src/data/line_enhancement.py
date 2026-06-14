"""
XDoG-based line enhancement for anime super-resolution.
Creates pseudo-ground-truth with enhanced hand-drawn lines.

Reference: APISR (CVPR 2024) - Pseudo-GT with enhanced hand-drawn lines
Uses XDoG (Extended Difference of Gaussians) for edge extraction,
outlier filtering, and passive dilation to sharpen anime lines.

Supports two pseudo-GT modes:
- "lite" (default): existing behavior - XDoG edges blended via alpha overlay
- "apisr": full APISR pipeline - 3 rounds of USM sharpening + XDoG +
           connected-component cleanup + passive dilation + composite via
           APISR blending: I_pseudo = I_sharp * I_map + I_GT * (1 - I_map)
"""
import numpy as np
import cv2
from typing import Tuple


def usm_sharpen(
    img: np.ndarray,
    rounds: int = 3,
    radius: int = 50,
    sigma: float = 0,
    threshold: int = 10,
    weight: float = 0.5,
) -> np.ndarray:
    """
    APISR-style unsharp mask (USM) sharpening.

    Applies `rounds` sequential USM operations to the input image. Each round:
        blur     = cv2.GaussianBlur(img, (radius, radius), sigma)
        residual = img - blur
        mask     = (abs(residual) * 255 > threshold).astype(float32)
        soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)
        sharp    = clip(img + weight * residual, 0, 255)
        result   = soft_mask * sharp + (1 - soft_mask) * img

    Args:
        img: HxWx3 uint8 RGB (or grayscale) image.
        rounds: Number of USM rounds to apply (APISR default: 3).
        radius: Gaussian blur kernel size (forced odd; APISR default: 50).
        sigma: Gaussian blur stddev (0 = auto; APISR default: 0).
        threshold: |I - B|*255 threshold for the per-pixel mask (APISR: 10).
        weight: Sharpening amount (APISR: 0.5).

    Returns:
        Sharpened uint8 image with the same shape/dtype as `img`.
    """
    if rounds < 1:
        return img
    r = radius if (radius % 2) == 1 else radius + 1
    out = img
    for _ in range(int(rounds)):
        blur = cv2.GaussianBlur(out, (r, r), sigma)
        residual = out.astype(np.float32) - blur.astype(np.float32)
        mask = (np.abs(residual) * 255.0 > float(threshold)).astype(np.float32)
        soft_mask = cv2.GaussianBlur(mask, (r, r), 0)
        sharp = np.clip(out.astype(np.float32) + float(weight) * residual, 0.0, 255.0)
        out = (soft_mask * sharp + (1.0 - soft_mask) * out.astype(np.float32))
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def xdog(
    img: np.ndarray,
    sigma: float = 0.6,
    k: float = 2.5,
    gamma: float = 0.97,
    eps: float = -15.0,
    phi: float = 1e9,
) -> np.ndarray:
    """
    APISR-style XDoG edge extraction on a single-channel (grayscale) image.

    Pipeline (mirrors APISR scripts/anime_strong_usm.py):
        g1 = GaussianBlur(gray, sigma=sigma)
        g2 = GaussianBlur(gray, sigma=sigma*k)
        d  = (g1 - gamma * g2) / 255
        e  = 1 + tanh(phi * (d - eps/255))
        e[e >= 1] = 1
        sketch = 1 - e   # 0=background, 1=line

    Args:
        img: HxW uint8 grayscale image.
        sigma: stddev of g1 (APISR: 0.6).
        k: ratio g2_sigma/g1_sigma (APISR: 2.5).
        gamma: DoG mixing (APISR: 0.97 + U(0, 0.01)).
        eps: XDoG threshold in raw 0-255 units (APISR: -15).
        phi: tanh steepness (APISR: 1e9, near-hard threshold).

    Returns:
        uint8 HxW line map in [0, 255] where 255 = detected line.
    """
    g1 = cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    g2 = cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma) * float(k),
                          sigmaY=float(sigma) * float(k))
    d = (g1.astype(np.float32) - float(gamma) * g2.astype(np.float32)) / 255.0
    e = 1.0 + np.tanh(float(phi) * (d - float(eps) / 255.0))
    e = np.where(e >= 1.0, 1.0, e)
    sketch = 1.0 - e  # float in [0, 1]
    return (sketch * 255.0).astype(np.uint8)


def connected_component_cleanup(line_map: np.ndarray, min_size: int = 32) -> np.ndarray:
    """
    APISR-style outlier removal using 8-connected BFS.

    Pixels with value > 127 are "white" (line). Connected components with fewer
    than `min_size` pixels are erased (set to 0).

    Args:
        line_map: HxW uint8 line map (0 or 255).
        min_size: Minimum region size to keep (APISR: 32).

    Returns:
        HxW uint8 cleaned line map.
    """
    h, w = line_map.shape
    out = np.zeros_like(line_map)
    visited = np.zeros((h, w), dtype=bool)

    for i in range(h):
        for j in range(w):
            if visited[i, j]:
                continue
            if int(line_map[i, j]) <= 127:
                visited[i, j] = True
                continue
            # BFS
            stack = [(i, j)]
            region = [(i, j)]
            visited[i, j] = True
            while stack:
                ci, cj = stack.pop()
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < h and 0 <= nj < w and not visited[ni, nj]:
                            visited[ni, nj] = True
                            if int(line_map[ni, nj]) > 127:
                                stack.append((ni, nj))
                                region.append((ni, nj))
            if len(region) >= int(min_size):
                for (ri, rj) in region:
                    out[ri, rj] = 255
    return out


def passive_dilation(line_map: np.ndarray, threshold: int = 3) -> np.ndarray:
    """
    APISR-style passive dilation: fill a black pixel if >= `threshold` of its
    8 white neighbors are white. Thickens lines to bridge small gaps.

    Args:
        line_map: HxW uint8 line map (0 or 255).
        threshold: Minimum count of white 8-neighbors to fill (APISR: 3).

    Returns:
        HxW uint8 dilated line map.
    """
    h, w = line_map.shape
    out = line_map.copy()
    th = int(threshold)
    for i in range(h):
        for j in range(w):
            if int(out[i, j]) > 127:
                continue
            n_white = 0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w and int(line_map[ni, nj]) > 127:
                        n_white += 1
            if n_white >= th:
                out[i, j] = 255
    return out


def make_pseudo_gt(
    img: np.ndarray,
    usm_rounds: int = 3,
    usm_radius: int = 50,
    usm_sigma: float = 0,
    usm_threshold: int = 10,
    usm_weight: float = 0.5,
    xdog_sigma: float = 0.6,
    xdog_k: float = 2.5,
    xdog_gamma: float = 0.97,
    xdog_eps: float = -15.0,
    xdog_phi: float = 1e9,
    outlier_min_size: int = 32,
    dilation_threshold: int = 3,
) -> np.ndarray:
    """
    APISR full pseudo-GT pipeline.

    Steps:
        1. USM sharpen the GT (3 rounds) -> I_sharp_1 (after 1st round) and
           I_sharp_3 (after all rounds) - I_sharp_1 is the "I_sharp" used in
           the composite per APISR scripts/anime_strong_usm.py:548.
        2. XDoG on I_sharp_3 grayscale -> sketch in [0, 1].
        3. Connected-component cleanup (BFS, min_size=32).
        4. Passive dilation (>=3 of 8 white neighbors).
        5. Composite: I_pseudo = I_sharp_1 * I_map + I_GT * (1 - I_map)
           (uses I_sharp_1 = sharpened GT after 1 USM round, NOT after all 3,
           matching APISR's source comment in scripts/anime_strong_usm.py).

    Args:
        img: HxWx3 uint8 RGB image (or HxW grayscale).
        usm_rounds: Number of USM rounds (APISR: 3).
        usm_radius: USM Gaussian radius (APISR: 50).
        usm_sigma: USM Gaussian sigma (APISR: 0).
        usm_threshold: USM threshold (APISR: 10).
        usm_weight: USM weight (APISR: 0.5).
        xdog_sigma, xdog_k, xdog_gamma, xdog_eps, xdog_phi: XDoG params.
        outlier_min_size: BFS region-size cutoff (APISR: 32).
        dilation_threshold: Passive dilation neighbor threshold (APISR: 3).

    Returns:
        HxWx3 uint8 pseudo-GT image (same shape/dtype as `img`).
    """
    if img.ndim == 2:
        # Grayscale: synthesize 3-channel for USM, then return grayscale
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img
    else:
        rgb = img

    # USM rounds: 1 -> I_sharp_1 (used in composite)
    sharp_1 = usm_sharpen(
        rgb, rounds=1, radius=usm_radius, sigma=usm_sigma,
        threshold=usm_threshold, weight=usm_weight,
    )
    # Full rounds on top of sharp_1 (matches APISR: f^2, f^3)
    sharp_3 = usm_sharpen(
        sharp_1, rounds=max(0, int(usm_rounds) - 1), radius=usm_radius,
        sigma=usm_sigma, threshold=usm_threshold, weight=usm_weight,
    )

    # XDoG on grayscale of fully-sharpened
    if sharp_3.ndim == 3:
        gray = cv2.cvtColor(sharp_3, cv2.COLOR_RGB2GRAY)
    else:
        gray = sharp_3
    sketch = xdog(
        gray, sigma=xdog_sigma, k=xdog_k, gamma=xdog_gamma,
        eps=xdog_eps, phi=xdog_phi,
    )
    sketch = connected_component_cleanup(sketch, min_size=outlier_min_size)
    sketch = passive_dilation(sketch, threshold=dilation_threshold)

    # Composite: I_pseudo = I_sharp_1 * I_map + I_GT * (1 - I_map)
    map_f = (sketch.astype(np.float32) / 255.0)
    if rgb.ndim == 3:
        map_3 = np.stack([map_f] * 3, axis=-1)
    else:
        map_3 = map_f
    pseudo = (sharp_1.astype(np.float32) * map_3
              + rgb.astype(np.float32) * (1.0 - map_3))
    pseudo = np.clip(pseudo, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        pseudo = cv2.cvtColor(pseudo, cv2.COLOR_RGB2GRAY)
    return pseudo


class LineEnhancer:
    """
    Enhances hand-drawn lines in anime images for pseudo-GT preparation.

    Two modes:
    - "lite" (default): XDoG edge extraction + outlier filter + passive
      dilation; overlay-blend back onto original.
    - "apisr": Full APISR pipeline (3-round USM + XDoG + cleanup + dilation)
      with composite `pseudo_gt = sharpened * xdog_map + original * (1 - xdog_map)`.

    Reference: APISR (CVPR 2024), Section 3.3
    https://arxiv.org/abs/2403.01598
    """

    def __init__(
        self,
        sigma1: float = 1.0,
        sigma2: float = 16.0,
        alpha: float = 0.1,
        gamma: float = 0.5,
        phi: float = 100.0,
        dilation_size: int = 2,
        outlier_threshold: int = 5,
        pseudo_gt_mode: str = "lite",
        usm_rounds: int = 3,
        usm_radius: int = 50,
        usm_sigma: float = 0,
        usm_threshold: int = 10,
        usm_weight: float = 0.5,
        xdog_sigma: float = 0.6,
        xdog_k: float = 2.5,
        xdog_gamma: float = 0.97,
        xdog_eps: float = -15.0,
        xdog_phi: float = 1e9,
        outlier_min_size: int = 32,
        dilation_threshold: int = 3,
    ):
        if pseudo_gt_mode not in ("lite", "apisr"):
            raise ValueError(
                f"pseudo_gt_mode must be 'lite' or 'apisr', got {pseudo_gt_mode!r}"
            )
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.alpha = alpha
        self.gamma = gamma
        self.phi = phi
        self.dilation_size = dilation_size
        self.outlier_threshold = outlier_threshold
        self.pseudo_gt_mode = pseudo_gt_mode

        # APISR-mode parameters. xdog_gamma is independent of the lite `gamma`
        # parameter to avoid the lite-mode gamma (0.5) silently overriding
        # the APISR XDoG gamma (0.97 + jitter).
        self.usm_rounds = usm_rounds
        self.usm_radius = usm_radius
        self.usm_sigma = usm_sigma
        self.usm_threshold = usm_threshold
        self.usm_weight = usm_weight
        self.xdog_sigma = xdog_sigma
        self.xdog_k = xdog_k
        self.xdog_gamma = xdog_gamma
        self.xdog_eps = xdog_eps
        self.xdog_phi = xdog_phi
        self.outlier_min_size = outlier_min_size
        self.dilation_threshold = dilation_threshold

    def _xdog(self, gray: np.ndarray) -> np.ndarray:
        """Extended Difference of Gaussians for edge extraction (lite mode)."""
        g1 = cv2.GaussianBlur(gray, (0, 0), self.sigma1)
        g2 = cv2.GaussianBlur(gray, (0, 0), self.sigma2)

        dog = g1 - self.gamma * g2
        dog = dog / 255.0

        edges = np.where(dog > 0, 1.0, 1.0 + np.tanh(self.phi * dog))
        edges = np.clip(edges, 0, 1)
        return (edges * 255).astype(np.uint8)

    def _filter_outliers(self, edges: np.ndarray) -> np.ndarray:
        """Remove isolated noise pixels (outliers)."""
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered = np.zeros_like(edges)
        for contour in contours:
            if cv2.contourArea(contour) >= self.outlier_threshold:
                cv2.drawContours(filtered, [contour], -1, 255, -1)

        return filtered

    def _passive_dilation(self, edges: np.ndarray) -> np.ndarray:
        """Slightly thicken lines for better visibility."""
        if self.dilation_size <= 0:
            return edges

        kernel = np.ones((self.dilation_size, self.dilation_size), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)

        lines_only = cv2.subtract(dilated, edges)
        lines_only = cv2.erode(lines_only, kernel, iterations=1)

        result = cv2.add(edges, lines_only)
        return result

    def _lite_call(self, img: np.ndarray) -> np.ndarray:
        """Existing lite-mode behavior (alpha-blended overlay)."""
        if img.ndim == 2:
            img_in = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_in = img

        gray = cv2.cvtColor(img_in, cv2.COLOR_RGB2GRAY)

        edges = self._xdog(gray)
        edges = self._filter_outliers(edges)
        edges = self._passive_dilation(edges)

        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        line_overlay = (255 - edges_3ch).astype(np.float32) / 255.0
        line_overlay = line_overlay * self.alpha * 255

        enhanced = img_in.astype(np.float32) + line_overlay
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        if img.ndim == 2:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
        return enhanced

    def _apisr_call(self, img: np.ndarray) -> np.ndarray:
        """Full APISR pseudo-GT pipeline."""
        return make_pseudo_gt(
            img,
            usm_rounds=self.usm_rounds,
            usm_radius=self.usm_radius,
            usm_sigma=self.usm_sigma,
            usm_threshold=self.usm_threshold,
            usm_weight=self.usm_weight,
            xdog_sigma=self.xdog_sigma,
            xdog_k=self.xdog_k,
            xdog_gamma=self.xdog_gamma,
            xdog_eps=self.xdog_eps,
            xdog_phi=self.xdog_phi,
            outlier_min_size=self.outlier_min_size,
            dilation_threshold=self.dilation_threshold,
        )

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply line enhancement to create pseudo-GT.

        Args:
            img: Input anime image [H, W, C] (RGB or grayscale), uint8.

        Returns:
            Enhanced image with sharpened lines, uint8, same shape as `img`.
        """
        if self.pseudo_gt_mode == "apisr":
            return self._apisr_call(img)
        return self._lite_call(img)
