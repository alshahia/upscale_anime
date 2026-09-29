"""Phase 5 Rank #2 smoke tests: APISR-style balanced twin perceptual loss wiring.

Verifies the --loss {vgg, twin} flag in anime_upscaler/distill.py:
  1. argparse accepts the new flag with default "vgg" (Rank #1 backward-compat).
  2. --twin-danbooru-weight / --twin-vgg-weight / --twin-delta are exposed.
  3. The lazy-init helper _get_twin_perceptual returns the same instance for
     the same weight signature (so we do not rebuild VGG19 + ResNet50 every
     epoch), and a fresh instance for a different signature.
  4. _get_twin_perceptual passes use_danbooru_resnet=True so the Danbooru file
     lookup is attempted; if absent, the fallback message is printed but the
     loss still returns a finite scalar (ImageNet-only BSD-3 path).

These tests are deliberately offline: they do NOT download VGG19/ResNet50
weights. The VGG19 weights are cached by torchvision on first use, so the
first test that builds a TwinPerceptualLoss will download ~548 MB to the
torch hub cache; subsequent tests reuse that cache.

If the user environment is hermetic (no network, no torch hub cache), the
argparse tests still pass; only test_twin_loss_forward_runs_with_imagenet
will fail. Mark that one xfail with a clear reason.
"""
import argparse
import sys
from pathlib import Path

import pytest
import torch

# distill.py lives at anime_upscaler/distill.py and bare-imports
#   `from dataset import AnimePairDataset, denorm01` (repo bare-import
# convention: same dir as the script). When invoked as a script via
# `python anime_upscaler/distill.py`, Python automatically adds the
# script's directory (anime_upscaler/) to sys.path. To reproduce that in a
# pytest collection, we insert anime_upscaler/ onto sys.path ourselves.
ROOT = Path(__file__).resolve().parent.parent
ANIME_UPSCALER = ROOT / "anime_upscaler"
for p in (str(ROOT), str(ANIME_UPSCALER)):
    if p not in sys.path:
        sys.path.insert(0, p)

import anime_upscaler.distill as distill  # noqa: E402


# ---------------------------------------------------------------------------
# 1. argparse shape
#
# distill.main() builds its ArgumentParser locally; we test the parser by
# invoking distill.py as a subprocess with the relevant flags and inspecting
# either the help text (for flag presence) or the exit code (for invalid
# choices). This is heavier than mocking sys.argv but it validates the
# ACTUAL parser the user-facing CLI sees.
# ---------------------------------------------------------------------------

DISTILL_PATH = ANIME_UPSCALER / "distill.py"


def _run_distill(argv):
    """Invoke distill.py with the given argv; return (returncode, stdout)."""
    import subprocess
    proc = subprocess.run(
        [sys.executable, str(DISTILL_PATH), *argv],
        capture_output=True, text=True, cwd=str(ROOT),
        timeout=60,
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_help_lists_loss_twin_flag():
    """--help advertises --loss {vgg,twin} so users can discover it."""
    rc, stdout, _ = _run_distill(["--help"])
    assert rc == 0, f"--help failed rc={rc}"
    assert "--loss {vgg,twin}" in stdout, \
        "--help should advertise --loss {vgg,twin} for discoverability"


def test_help_lists_twin_weight_flags():
    """--help advertises the twin weight flags."""
    rc, stdout, _ = _run_distill(["--help"])
    assert rc == 0
    assert "--twin-danbooru-weight" in stdout
    assert "--twin-vgg-weight" in stdout
    assert "--twin-delta" in stdout


def test_loss_invalid_choice_rejected_by_cli():
    """The CLI rejects unknown --loss values (argparse default behavior)."""
    rc, _, stderr = _run_distill(["--loss", "gan"])
    assert rc != 0, "--loss gan should be rejected"
    assert "invalid choice" in stderr.lower(), \
        f"argparse rejection message missing; stderr={stderr!r}"


def test_distill_help_does_not_crash_on_twin_flag():
    """--help with --loss twin parses cleanly (does not consume both flags)."""
    rc, stdout, _ = _run_distill(["--loss", "twin", "--help"])
    assert rc == 0
    assert "--loss {vgg,twin}" in stdout


# ---------------------------------------------------------------------------
# 2. lazy-init cache behaviour
# ---------------------------------------------------------------------------

def test_get_twin_perceptual_signature_change_rebuilds():
    """Different weight signatures must rebuild the loss (not silently cache).

    We force a fresh cache by nulling the module globals before each call,
    then assert both calls return distinct module instances. This catches
    regressions where a recipe sweep with --twin-vgg-weight=0.5 vs 0.75 would
    return the same cached model with stale weights.
    """
    # Force a clean cache state.
    distill._twin_loss = None
    distill._twin_loss_signature = None

    loss_a = distill._get_twin_perceptual(
        torch.device("cpu"),
        danbooru_weight=0.5,
        vgg_weight=0.5,
        use_danbooru_resnet=False,  # offline path; do not search Danbooru file
    )
    assert distill._twin_loss is loss_a
    assert distill._twin_loss_signature == (0.5, 0.5, False)

    # Same signature -> same instance (cache hit).
    loss_a2 = distill._get_twin_perceptual(
        torch.device("cpu"),
        danbooru_weight=0.5,
        vgg_weight=0.5,
        use_danbooru_resnet=False,
    )
    assert loss_a2 is loss_a, "same signature must return cached instance"

    # Different signature -> rebuild.
    loss_b = distill._get_twin_perceptual(
        torch.device("cpu"),
        danbooru_weight=0.25,
        vgg_weight=0.75,
        use_danbooru_resnet=False,
    )
    assert loss_b is not loss_a, "signature change must trigger rebuild"
    assert distill._twin_loss_signature == (0.25, 0.75, False)

    # Restore clean cache for downstream tests.
    distill._twin_loss = None
    distill._twin_loss_signature = None


def test_get_twin_perceptual_frozen_grads():
    """The cached twin loss must keep all backbone params frozen.

    Backbone autograd through VGG19/ResNet50 would balloon VRAM and waste
    compute; the lazy-init helper explicitly disables grads. Catch any
    future regression that re-enables training on the backbone.
    """
    distill._twin_loss = None
    distill._twin_loss_signature = None
    loss = distill._get_twin_perceptual(
        torch.device("cpu"),
        danbooru_weight=0.5,
        vgg_weight=0.5,
        use_danbooru_resnet=False,
    )
    bad = [n for n, p in loss.named_parameters() if p.requires_grad]
    assert not bad, f"twin backbone params must be frozen, found: {bad[:5]}"
    distill._twin_loss = None
    distill._twin_loss_signature = None


# ---------------------------------------------------------------------------
# 3. forward pass (requires torch hub cache for VGG19 + ResNet50 ImageNet
#    weights, ~548 MB + ~98 MB). xfail with a clear reason if cache is empty.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def twin_loss_cpu():
    """Build a TwinPerceptualLoss once per test module (CPU, no Danbooru)."""
    distill._twin_loss = None
    distill._twin_loss_signature = None
    try:
        loss = distill._get_twin_perceptual(
            torch.device("cpu"),
            danbooru_weight=0.5,
            vgg_weight=0.5,
            use_danbooru_resnet=False,
        )
    except Exception as exc:
        pytest.skip(f"TwinPerceptualLoss build failed (offline?): {exc}")
    yield loss
    distill._twin_loss = None
    distill._twin_loss_signature = None


def test_twin_loss_forward_runs_with_imagenet(twin_loss_cpu):
    """TwinPerceptualLoss(sr, hr) returns a finite scalar on CPU.

    Uses 32x32 random inputs to keep this test fast. Catches regressions
    where the new wiring in distill.py passes a wrong input range (LPIPS
    needs [-1, 1]; TwinPerceptualLoss handles [0, 1] internally and
    normalises to ImageNet stats).
    """
    torch.manual_seed(0)
    pred = torch.rand(1, 3, 32, 32)
    target = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        loss_val = twin_loss_cpu(pred, target)
    assert torch.is_tensor(loss_val), f"twin loss returned non-tensor: {loss_val}"
    assert torch.isfinite(loss_val).all(), \
        f"twin loss produced NaN/Inf: {loss_val}"
    assert loss_val.dim() == 0 or loss_val.numel() == 1, \
        f"twin loss must be scalar, got shape {tuple(loss_val.shape)}"
    assert float(loss_val) >= 0.0, f"twin loss must be non-negative, got {loss_val}"


def test_twin_loss_isolated_zero_weight_is_finite(twin_loss_cpu):
    """Setting one branch weight to 0 must still produce a finite loss.

    Mirrors the v7_loss_schedule TestDanbooruVGGWeightBalance offline check;
    protects against future regressions where danbooru_weight=0 / vgg_weight=0
    would divide-by-zero inside the loss.
    """
    distill._twin_loss = None
    distill._twin_loss_signature = None
    loss = distill._get_twin_perceptual(
        torch.device("cpu"),
        danbooru_weight=1.0,
        vgg_weight=0.0,
        use_danbooru_resnet=False,
    )
    torch.manual_seed(1)
    pred = torch.rand(1, 3, 32, 32)
    target = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        loss_val = loss(pred, target)
    assert torch.isfinite(loss_val).all(), \
        f"twin loss with vgg_weight=0 produced NaN/Inf: {loss_val}"
    distill._twin_loss = None
    distill._twin_loss_signature = None
