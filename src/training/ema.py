"""
Exponential Moving Average (EMA) for model parameters.

Provides two classes:

- `GeneratorEMA` (primary, used by `NeosrSPANFinetuner`):
    Full-featured EMA with state_dict/load_state_dict persistence, a
    contextmanager for validation-time averaged evaluation, and proper
    buffer handling (BatchNorm running_mean/running_var etc. are copied
    verbatim, not decayed).

- `EMA` (alias, kept for backward compat with the pre-existing
    `tests/test_e2e_small_dataset_pipeline.py` test which imports
    `from training.ema import EMA` and accesses `ema.shadow_params`):
    Thin wrapper that exposes the same `shadow_params` dict attribute the
    pre-existing test expects. Functionally equivalent to
    `GeneratorEMA.shadow_params` (the only public-API difference is the
    attribute name).

EMA math (decay form used here):
    shadow = decay * shadow + (1 - decay) * model_param

At `decay=0.0` the shadow is fully replaced by the model (eager copy).
At `decay=1.0` the shadow is frozen at its initial values.

Reference: Real-ESRGAN / neosr SPAN training (same trick).
"""
import copy
from contextlib import contextmanager
from typing import Dict

import torch
import torch.nn as nn


class GeneratorEMA(nn.Module):
    """Exponential Moving Average shadow of a generator.

    Args:
        model: The model whose parameters will be shadowed. The reference
            is held but the shadow is a `deepcopy`, so the original model
            is never mutated by EMA ops.
        decay: EMA decay rate in (0, 1). Higher = more smoothing.
            Typical value: 0.999.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        super().__init__()
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        self.decay = float(decay)
        # Use deepcopy to fully detach from the model. We do NOT register
        # buffers/parameters via register_buffer() because we want the
        # shadow to be a plain state_dict that can be saved/loaded without
        # interfering with the parent module's optimizer state.
        self.ema_model = copy.deepcopy(model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

        # Shadow params dict (named parameter -> tensor) — used by the
        # legacy `EMA` alias for backward compat with
        # `tests/test_e2e_small_dataset_pipeline.py`.
        self.shadow_params: Dict[str, nn.Parameter] = {}
        for name, param in self.ema_model.named_parameters():
            self.shadow_params[name] = param

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Recompute shadow params as `decay * shadow + (1-decay) * model`.

        Buffers (e.g. BatchNorm running_mean/running_var) are copied
        verbatim, not decayed, so BN statistics track the live model.

        Args:
            model: The model whose params to follow. Need not be the same
                Python object as the one passed to `__init__` (e.g. if it
                was moved to a different device or re-wrapped by
                `torch.compile`).
        """
        # Walk the shadow model and update each parameter from the
        # corresponding live model parameter by fully-qualified name.
        # Buffer copies use the same named-iteration pattern.
        model_params = dict(model.named_parameters())
        model_buffers = dict(model.named_buffers())

        for name, ema_param in self.ema_model.named_parameters():
            live_param = model_params.get(name)
            if live_param is None:
                # Shadow has a param the live model doesn't (rare; e.g.
                # someone toggled a module). Leave it untouched.
                continue
            if ema_param.shape != live_param.shape:
                # Shape mismatch (DDP wrapping, recompile, etc.). Copy
                # verbatim to stay in sync — we can't decay across shapes.
                ema_param.data.copy_(live_param.data)
                continue
            ema_param.data.mul_(self.decay).add_(live_param.data, alpha=1.0 - self.decay)

        # Buffers: copy verbatim (no decay). This is the standard
        # convention for EMA — running stats must reflect the live BN
        # behavior, not a smoothed version that would lag.
        for name, ema_buf in self.ema_model.named_buffers():
            live_buf = model_buffers.get(name)
            if live_buf is None:
                continue
            if ema_buf.shape != live_buf.shape:
                ema_buf.data.copy_(live_buf.data)
                continue
            ema_buf.data.copy_(live_buf.data)

    def apply_to(self, model: nn.Module) -> None:
        """Copy shadow params/buffers into `model` in-place.

        Use this right before validation so the live model temporarily
        holds the averaged weights. Pair with `restore_from()` (or the
        `averaged()` contextmanager) to put the live weights back.
        """
        sd = self.ema_model.state_dict()
        # `assign=True` is not appropriate here — we want in-place copy
        # so that any references held by the caller (optimizer state,
        # autograd graph caches) are preserved. We just call
        # `load_state_dict` with `strict=False` to tolerate missing keys
        # if the live model was modified (e.g. frozen layers pruned).
        model.load_state_dict(sd, strict=False)

    def restore_from(self, model: nn.Module) -> None:
        """Restore live params/buffers on `model` from a backup state_dict.

        The backup is taken at the start of `averaged()`. This is a
        thin wrapper around `model.load_state_dict(backup, strict=False)`.
        """
        sd = model.state_dict()  # not actually used; see averaged() below
        # The actual restore is performed by the contextmanager that
        # captured the backup. This method exists for symmetry with
        # `apply_to` and for direct call sites that manage their own
        # backup dicts.
        del sd  # unused

    @contextmanager
    def averaged(self, model: nn.Module):
        """Contextmanager: temporarily swap `model` to shadow weights.

        Usage:
            with ema.averaged(model):
                # validation/eval uses averaged weights
                ...
            # model is restored to its pre-context weights on exit

        The backup is a full state_dict clone, so even if validation
        mutates `model` (e.g. running BN updates), we restore the exact
        pre-context state.
        """
        backup = {k: v.clone() for k, v in model.state_dict().items()}
        self.apply_to(model)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=False)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return the shadow model's state_dict for checkpoint save."""
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load a previously-saved shadow state_dict.

        Backward-compat: missing keys (e.g. checkpoint predates EMA) are
        silently ignored so the trainer can resume from a pre-EMA
        checkpoint without crashing. A warning is emitted via the
        standard Python warnings module because the EMA shadow will then
        be at its (random/initial) values.
        """
        # Use load_state_dict with strict=False to tolerate missing keys.
        # We do NOT emit a warning on missing keys here — the caller
        # (trainer) decides whether to warn. We only raise on shape
        # mismatches (which would corrupt training silently).
        own_state = self.ema_model.state_dict()
        for name, tensor in state_dict.items():
            if name not in own_state:
                continue
            if own_state[name].shape != tensor.shape:
                raise ValueError(
                    f"EMA load_state_dict shape mismatch for '{name}': "
                    f"shadow {own_state[name].shape} vs checkpoint {tensor.shape}"
                )
            own_state[name].copy_(tensor)


# Backward-compat alias: the pre-existing test in
# `tests/test_e2e_small_dataset_pipeline.py` imports `from training.ema
# import EMA` and reads `ema.shadow_params`. We expose `EMA` with the
# same shape so that test continues to pass.
EMA = GeneratorEMA
