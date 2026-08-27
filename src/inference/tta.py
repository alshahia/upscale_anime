"""D4-group 8x test-time augmentation ensemble for inference.

The D4 dihedral group has 8 symmetries of a square. The canonical 8-element
listing used here is:

    1. identity
    2. horizontal flip (fliplr)
    3. vertical flip (flipud)
    4. both flips (flipboth)
    5. 90-degree rotation CCW (rot90_ccw)
    6. 90-degree rotation CW  (rot90_cw)
    7. rot90_ccw + fliplr   (diagonal reflection)
    8. rot90_ccw + flipud   (anti-diagonal reflection)

The four reflections and the two diagonal reflections are order-2 (each is
its own inverse); the two 90-degree rotations are order-4 (rot90_ccw and
rot90_cw are inverses of each other).

Note that `torch.flip(x, dims=[-1, -2])` (flipboth) and
`torch.rot90(x, k=2, dims=[-2, -1])` (rot180) produce identical outputs on
both square and non-square tensors, so we list only `flipboth` here (using
`rot180` instead would also work but adds a redundant code path).

`tta_forward` runs the model 8 times on the 8 augmented inputs and returns
the average of the inverse-augmented predictions, putting every prediction
back into the canonical orientation before averaging.

This is a "Tier 1" quality boost documented in
docs/survey_2026/recommendations.md (LPIPS -10% on average for anisotropic SR
models, free at inference time aside from 8x compute).

Distinct from `engine.TTA_TRANSFORMS` (a flip-only 4x ensemble). The D4 8x is
the recommended setting for the v7 anime config.
"""
import torch


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _flip_lr(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-1])


def _flip_ud(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-2])


def _flip_both(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-1, -2])


def _rot90_ccw(x: torch.Tensor) -> torch.Tensor:
    return torch.rot90(x, k=1, dims=[-2, -1])


def _rot90_cw(x: torch.Tensor) -> torch.Tensor:
    return torch.rot90(x, k=-1, dims=[-2, -1])


def _rot90_ccw_fliplr(x: torch.Tensor) -> torch.Tensor:
    """rot90 CCW then horizontal flip. Self-inverse (order-2 D4 element)."""
    return torch.flip(torch.rot90(x, k=1, dims=[-2, -1]), dims=[-1])


def _rot90_ccw_flipud(x: torch.Tensor) -> torch.Tensor:
    """rot90 CCW then vertical flip. Self-inverse (order-2 D4 element)."""
    return torch.flip(torch.rot90(x, k=1, dims=[-2, -1]), dims=[-2])


# (aug_fn, inv_fn) - both operate on [B, C, H, W].
# Verified empirically: each pair satisfies inv(aug(x)) == x.
D4_AUGMENTATIONS = [
    (_identity,           _identity),
    (_flip_lr,            _flip_lr),
    (_flip_ud,            _flip_ud),
    (_flip_both,          _flip_both),
    (_rot90_ccw,          _rot90_cw),
    (_rot90_cw,           _rot90_ccw),
    (_rot90_ccw_fliplr,   _rot90_ccw_fliplr),
    (_rot90_ccw_flipud,   _rot90_ccw_flipud),
]


def tta_forward(model, lr: torch.Tensor, use_clip: bool = True) -> torch.Tensor:
    """Average 8 D4 augmentations of the SR output.

    Args:
        model: callable returning SR tensor of shape [B, C, H*scale, W*scale].
        lr: [B, C, H, W] input tensor (any dtype, on any device).
        use_clip: if True, clamp each prediction to [0, 1] before averaging.

    Returns:
        sr: [B, C, H*scale, W*scale] averaged prediction in float32, cast back
        to `lr.dtype` for downstream tensor-type compatibility.
    """
    out = None
    n = len(D4_AUGMENTATIONS)
    for aug, inv in D4_AUGMENTATIONS:
        with torch.no_grad():
            aug_lr = aug(lr)
            sr = model(aug_lr)
            if use_clip:
                sr = sr.clamp(0.0, 1.0)
            sr = inv(sr).float()
        out = sr if out is None else out + sr
    return (out / n).to(lr.dtype)