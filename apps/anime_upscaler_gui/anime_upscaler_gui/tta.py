"""D4 group test-time augmentation (TTA) for SR models.

VENDORED COPY of anime_upscaler/tta.py so the GUI package remains self-contained.
The training pipeline version is the source of truth; this copy must stay in sync.

D4 is the dihedral group of order 8: 4 rotations (0/90/180/270) plus 4 reflections
(horizontal, vertical, main-diagonal transpose, anti-diagonal transpose). For
super-resolution we apply each transform to LR, run the model, apply the inverse
to the SR output, and average the n results. Typically adds ~0.05-0.15 dB PSNR
at 4x-8x the inference cost.

Shape policy:
  * Square LR (h == w): use all 8 transforms (full D4).
  * Rectangular LR (h != w): use the 6 shape-preserving transforms
    (id, R90, R180, R270, H-flip, V-flip). The diagonal reflections swap
    h and w, so they are skipped.

References:
  - Wang et al. (NTIRE2017 winner trick) - mean-of-augmented inference.
  - Real-ESRGAN inference_realesrgan.py uses the same shape-aware pattern
    with the diagonal reflections gated on h == w.
"""
import torch


# --------------------------------------------------------------------------- #
# Transform primitives
# --------------------------------------------------------------------------- #
# Every fwd/inv takes a (1, 3, H, W) tensor and returns the same-shape tensor.
# Diagonal reflections must .contiguous() because torch.transpose returns a
# view with non-contiguous strides, which convs reject.

def _fwd_transpose(x):
    return torch.transpose(x, -2, -1).contiguous()


def _inv_transpose(x):
    return torch.transpose(x, -2, -1).contiguous()


def _fwd_antitranspose(x):
    # antitranspose = transpose then flip both axes (involutive).
    return torch.flip(torch.transpose(x, -2, -1).contiguous(), dims=(-2, -1)).contiguous()


def _inv_antitranspose(x):
    return torch.flip(torch.transpose(x, -2, -1).contiguous(), dims=(-2, -1)).contiguous()


# Identity transforms clone the input when it is non-contiguous so that
# downstream code can rely on every augmented LR being a real (not aliased) tensor.
def _identity(x):
    return x.contiguous()


# --------------------------------------------------------------------------- #
# Transform lists
# --------------------------------------------------------------------------- #
_AXIS_ALIGNED = [
    (_identity, _identity, "id"),
    (lambda x: torch.rot90(x, 1, (-2, -1)), lambda x: torch.rot90(x, -1, (-2, -1)), "R90"),
    (lambda x: torch.rot90(x, 2, (-2, -1)), lambda x: torch.rot90(x, 2, (-2, -1)), "R180"),
    (lambda x: torch.rot90(x, 3, (-2, -1)), lambda x: torch.rot90(x, -3, (-2, -1)), "R270"),
    (lambda x: torch.flip(x, (-1,)),         lambda x: torch.flip(x, (-1,)),         "Fh"),
    (lambda x: torch.flip(x, (-2,)),         lambda x: torch.flip(x, (-2,)),         "Fv"),
]

_DIAGONAL_SUFFIX = [
    (_fwd_transpose, _inv_transpose, "T"),
    (_fwd_antitranspose, _inv_antitranspose, "AT"),
]


def _build_transforms(h, w):
    """Return (forward, inverse, name) triples appropriate for input shape.

    Square (h == w): 8 transforms (full D4).
    Rectangle:       6 transforms (axis-aligned only -- diagonals swap h, w).
    """
    if h == w:
        return _AXIS_ALIGNED + _DIAGONAL_SUFFIX
    return list(_AXIS_ALIGNED)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
@torch.no_grad()
def tta_forward(model, lr):
    """D4 test-time augmentation forward pass.

    Args:
        model: callable (1, 3, H, W) -> (1, 3, 4H, 4W); should be in eval mode.
        lr: input tensor (1, 3, H, W); any float dtype/device.

    Returns:
        (sr_mean, n_aug, names) where:
          - sr_mean is the per-element mean of n_aug augmented forward passes
            (each inverse-transformed back to the original orientation),
            shape (1, 3, 4H, 4W) on the same device/dtype as `lr`.
          - n_aug is 8 for square inputs, 6 for rectangular.
          - names is the list of transform labels applied.
    """
    _, _, h, w = lr.shape
    transforms = _build_transforms(h, w)
    sum_sr = None
    for fwd, inv, _name in transforms:
        x_aug = fwd(lr)
        y_aug = model(x_aug)
        y = inv(y_aug)
        sum_sr = y if sum_sr is None else sum_sr + y
    return sum_sr / len(transforms), len(transforms), [n for _, _, n in transforms]


__all__ = ["tta_forward"]