"""
Unit tests for FDLLoss (Frequency Distribution Loss).

Regression tests for the gradient-flow fix:
- extract_features must NOT wrap DINOv2 forward in torch.no_grad();
  otherwise FDL contributes zero gradient to the SR prediction.
- DINOv2 params must remain frozen regardless.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import torch
from losses.fdl_loss import FDLLoss


@pytest.fixture(scope="module")
def fdl_cpu() -> FDLLoss:
    """Real FDLLoss with DINOv2-small on CPU. Module-scoped to avoid re-download."""
    torch.set_num_threads(1)
    loss = FDLLoss(num_proj=8, dino_variant='small')
    loss.eval()
    return loss


class TestFDLGradientFlow:
    """Issue #1: FDL must propagate gradients to the prediction input."""

    def test_fdl_gradient_flows_to_input(self, fdl_cpu: FDLLoss) -> None:
        pred = torch.randn(1, 3, 28, 28, requires_grad=True)
        target = torch.randn(1, 3, 28, 28)

        loss = fdl_cpu(pred, target)
        loss.backward()

        assert pred.grad is not None, "FDL loss produced no gradient on prediction input"
        assert pred.grad.abs().sum() > 0, "FDL gradient on prediction is all zeros"

    def test_fdl_gradient_finite(self, fdl_cpu: FDLLoss) -> None:
        """Gradient must be finite (no NaN/Inf from DINOv2)."""
        pred = torch.randn(1, 3, 28, 28, requires_grad=True)
        target = torch.randn(1, 3, 28, 28)

        loss = fdl_cpu(pred, target)
        loss.backward()

        assert torch.isfinite(pred.grad).all(), "FDL gradient contains NaN or Inf"

    def test_fdl_loss_finite_within_step_range(self, fdl_cpu: FDLLoss) -> None:
        pred = torch.randn(1, 3, 28, 28)
        target = torch.randn(1, 3, 28, 28)

        loss = fdl_cpu(pred, target)

        assert torch.isfinite(loss), f"FDL loss is not finite: {loss.item()}"


class TestFDLFrozenDino:
    """DINOv2 must remain frozen after FDL backward pass."""

    def test_dino_params_remain_frozen(self, fdl_cpu: FDLLoss) -> None:
        # All DINOv2 params must have requires_grad=False from init.
        trainable = [n for n, p in fdl_cpu.feature_extractor.named_parameters() if p.requires_grad]
        assert trainable == [], f"DINOv2 has trainable params: {trainable[:3]}..."

    def test_dino_grads_remain_none_after_backward(self, fdl_cpu: FDLLoss) -> None:
        pred = torch.randn(1, 3, 28, 28, requires_grad=True)
        target = torch.randn(1, 3, 28, 28)

        loss = fdl_cpu(pred, target)
        loss.backward()

        grads_set = [n for n, p in fdl_cpu.feature_extractor.named_parameters() if p.grad is not None]
        assert grads_set == [], f"DINOv2 params got gradients: {grads_set[:3]}..."


class TestFDLEvalMode:
    """FDL should be in eval mode by default for inference/eval passes."""

    def test_fdl_dino_in_eval_mode(self, fdl_cpu: FDLLoss) -> None:
        assert not fdl_cpu.feature_extractor.training, "DINOv2 should be in eval mode"
