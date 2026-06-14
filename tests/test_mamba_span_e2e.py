"""
End-to-end integration test for the v7 opt-in MambaIRv2 path (Phase G.5).

Verifies the full NeosrSPAN model can be built with ``sab_type: mamba_v2``
(v7 opt-in) and runs through a complete forward + backward pass on CPU,
using the pure-PyTorch ``selective_scan_ref`` fallback (mamba-ssm is NOT
installed in this venv).

Background
----------
- v7 (commit d0f7f23) introduces ``sab_type`` in
  ``configs/finetune_neosr_span_v7_anime.yaml`` and the factory in
  ``src/models/span/neosr_span.py`` that swaps SPAB for MambaSPAB.
- The v7 default is ``sab_type: conv3xc`` (NOT ``mamba_v2``) so that v6
  warm-start checkpoints continue to load into the same parameter layout.
- Per-block param count: MambaSPAB ~19,712 vs SPAB ~285,000 (~14x smaller
  per block). Full model (feature_channels=48, upscale=4): 275K vs 2.24M
  (~8x smaller end-to-end). This is the headline efficiency gain.
- v6 checkpoints do NOT warm-start a ``sab_type: mamba_v2`` model -- the
  per-block parameter names differ (``block_*.c1_r.*`` / ``c2_r.*`` /
  ``c3_r.*`` for SPAB vs ``block_*.gate`` / ``block_*.assm.*`` for
  MambaSPAB). This is a NEGATIVE compatibility property that we document
  here so the user is not surprised at warm-start time.

CPU-only. 8 tests across 1 class.
"""
import os
import sys
import io
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import torch
import torch.nn as nn

from models.span.neosr_span import NeosrSPAN, SPAB, create_neosr_span
from models.span.mambair_v2 import MambaSPAB, HAS_MAMBA_SSM


V7_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), '..',
    'configs', 'finetune_neosr_span_v7_anime.yaml',
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build(model_kwargs):
    """Build a NeosrSPAN and return it (no GPU required)."""
    return NeosrSPAN(
        num_in_ch=3, num_out_ch=3,
        feature_channels=48, upscale=4, bias=True, norm=False,
        **model_kwargs,
    )


def _v6_shaped_state_dict():
    """Return a freshly-initialized NeosrSPAN(sab_type=conv3xc) state_dict.

    Mimics what a v6-trained checkpoint would look like: a strict key-shape
    match against a NeosrSPAN(sab_type=conv3xc) model. 206 keys total,
    block_*.c1_r.* / c2_r.* / c3_r.* / eval_conv.* for every attention
    block. See ``tmp/inspect_state.py`` for the full enumeration.
    """
    m = _build({})
    return m.state_dict()


def _write_tmp_checkpoint(state_dict):
    """Serialize a state_dict to a tmp file and return the path.

    The file lives under the workspace's gitignored ``tmp/`` directory and
    is cleaned up at the end of the test that created it. Pattern mirrors
    the safe-fallback ``torch.load`` convention from AGENTS.md.
    """
    tmp_dir = os.path.join(os.path.dirname(__file__), '..', 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, '_e2e_v6_ckpt.pth')
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    with open(path, 'wb') as f:
        f.write(buf.getvalue())
    return path


# ---------------------------------------------------------------------------
# End-to-end tests
# ---------------------------------------------------------------------------
class TestMambaSpanE2E:
    """End-to-end tests for NeosrSPAN with sab_type: mamba_v2 (v7 opt-in)."""

    # ------------------------------------------------------------------
    # 1. Full model builds and runs a 4x forward pass.
    # ------------------------------------------------------------------
    def test_full_model_with_mamba_v2(self):
        """Build a full NeosrSPAN with sab_type='mamba_v2' and run 4x forward.

        Input  (1, 3, 32, 32)  ->  Output (1, 3, 128, 128).
        All six attention blocks must be MambaSPAB instances (not SPAB).
        """
        torch.manual_seed(0)
        model = _build({'sab_type': 'mamba_v2'})
        model.eval()

        # All six blocks are MambaSPAB.
        for i in range(1, 7):
            blk = getattr(model, f'block_{i}')
            assert isinstance(blk, MambaSPAB), (
                f"block_{i} is {type(blk).__name__}, expected MambaSPAB"
            )

        # 4x forward: 32x32 -> 128x128.
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            y = model(x)

        assert y.shape == (1, 3, 128, 128), f"got {tuple(y.shape)}"
        assert y.dtype == torch.float32
        assert torch.isfinite(y).all(), "output has NaN/Inf"
        # Random init + sigmoid-gated scan output should be bounded and small
        # in magnitude (gate starts at 0 -> output is near zero). We only
        # require it to be finite and not absurdly large.
        assert y.abs().max() < 100.0, "output magnitude is implausibly large"

    # ------------------------------------------------------------------
    # 2. Mamba_v2 is dramatically smaller than conv3xc.
    # ------------------------------------------------------------------
    def test_mamba_v2_param_count_smaller_than_conv3xc(self):
        """MambaIRv2 should reduce total param count by ~8x end-to-end.

        Reference numbers (feature_channels=48, upscale=4, bias=True):
          - conv3xc: 2,236,731 total / 1,840,488 trainable
          - mamba_v2: 275,355 total / 253,224 trainable
          - ratio: ~8.12x smaller

        We use a conservative lower-bound (4x) so the test does not become
        brittle if module-internal additions nudge the ratio by a few
        percent, while still catching the "we accidentally built a much
        larger model" regression.
        """
        m_conv = _build({})
        m_mamba = _build({'sab_type': 'mamba_v2'})

        n_conv = sum(p.numel() for p in m_conv.parameters())
        n_mamba = sum(p.numel() for p in m_mamba.parameters())

        # Document actual values for diagnostics.
        assert n_conv > 1_000_000, f"conv3xc has only {n_conv} params"
        assert n_mamba < 500_000, f"mamba_v2 has {n_mamba} params (expected < 500K)"
        # mamba_v2 should be at most 25% the size of conv3xc (i.e., >=4x smaller).
        assert n_mamba * 4 < n_conv, (
            f"mamba_v2 ({n_mamba}) is not 4x smaller than conv3xc ({n_conv})"
        )

    # ------------------------------------------------------------------
    # 3. Gradient flows back to the input through the whole model.
    # ------------------------------------------------------------------
    def test_mamba_v2_gradient_flows(self):
        """Backward through the mamba_v2 stack must produce non-zero
        gradients on the input AND on the MambaSPAB parameters.

        This guards against e.g. an accidentally-detached tensor, a
        ``torch.no_grad()`` leak, or a broken autograd path inside the
        selective scan.
        """
        torch.manual_seed(0)
        model = _build({'sab_type': 'mamba_v2'})
        model.train()  # ensure Conv3XC takes the training (non-fused) path

        x = torch.randn(1, 3, 32, 32, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Input gradient present and non-zero.
        assert x.grad is not None, "no grad on input"
        assert torch.any(x.grad != 0), "input grad is all-zero"
        assert torch.isfinite(x.grad).all(), "input grad has NaN/Inf"

        # At least one MambaSPAB parameter must have received a gradient.
        # The gated scan's path is non-trivial; if any of them are
        # disconnected the test will catch it.
        mamba_param_grads = [
            p.grad for n, p in model.named_parameters()
            if 'block_' in n and 'assm' in n and p.grad is not None
        ]
        assert len(mamba_param_grads) > 0, "no MambaSPAB parameters got grads"
        assert any(g.abs().sum() > 0 for g in mamba_param_grads), (
            "all MambaSPAB gradients are zero -- autograd is broken"
        )

    # ------------------------------------------------------------------
    # 4. Regression: sab_type=conv3xc default still works.
    # ------------------------------------------------------------------
    def test_mamba_v2_inherits_v6_default(self):
        """sab_type='conv3xc' (the v6 default) must still build SPAB blocks.

        This is the regression guard for v6 warm-start compatibility. If
        the default ever flipped to 'mamba_v2' (or if SPAB was accidentally
        renamed), every previously-saved v6 checkpoint would silently fail
        to load. The v7 config keeps the default at 'conv3xc'; verify the
        model reflects that.
        """
        # Default constructor (no explicit sab_type) -> conv3xc.
        model_default = _build({})
        for i in range(1, 7):
            blk = getattr(model_default, f'block_{i}')
            assert isinstance(blk, SPAB), (
                f"default block_{i} is {type(blk).__name__}, expected SPAB"
            )

        # Explicit conv3xc -> SPAB.
        model_conv = _build({'sab_type': 'conv3xc'})
        for i in range(1, 7):
            blk = getattr(model_conv, f'block_{i}')
            assert isinstance(blk, SPAB), (
                f"conv3xc block_{i} is {type(blk).__name__}, expected SPAB"
            )

        # The two models should be architecturally identical.
        assert model_default.load_state_dict(model_conv.state_dict())

    # ------------------------------------------------------------------
    # 5. Factory kwargs (d_state, num_tokens) actually change the model.
    # ------------------------------------------------------------------
    def test_mamba_v2_factory_kwargs(self):
        """Different factory kwargs (d_state, num_tokens) should produce
        structurally different models (different param shapes/counts).

        This guards the factory plumbing: if the v7 config's
        ``model.mamba.{d_state, num_tokens, inner_rank, mlp_ratio}`` block
        is silently dropped on the way to ``MambaSPAB``, this test will
        catch it.
        """
        m_default = _build({
            'sab_type': 'mamba_v2',
            'mamba_d_state': 16,
            'mamba_num_tokens': 64,
        })
        m_larger = _build({
            'sab_type': 'mamba_v2',
            'mamba_d_state': 32,     # 2x d_state
            'mamba_num_tokens': 128, # 2x token table
        })

        n_default = sum(p.numel() for p in m_default.parameters())
        n_larger = sum(p.numel() for p in m_larger.parameters())

        # Larger factory kwargs -> more parameters (d_state and num_tokens
        # both widen the scan state / token table).
        assert n_larger > n_default, (
            f"larger kwargs produced {n_larger} params, expected > {n_default}"
        )

        # And the two models are not state-dict compatible (different
        # shapes for ``block_*.assm.scan.A_log`` and
        # ``block_*.assm.token.weight``).
        sd_default = m_default.state_dict()
        sd_larger = m_larger.state_dict()
        assert sd_default['block_1.assm.scan.A_log'].shape == (48, 16)
        assert sd_larger['block_1.assm.scan.A_log'].shape == (48, 32)
        assert sd_default['block_1.assm.token.weight'].shape == (64, 16)
        assert sd_larger['block_1.assm.token.weight'].shape == (128, 32)

    # ------------------------------------------------------------------
    # 6. mamba-ssm is NOT installed; we use the pure-PyTorch fallback.
    # ------------------------------------------------------------------
    def test_mamba_v2_pure_torch_fallback(self):
        """Confirm the v7 opt-in path runs without the CUDA mamba-ssm kernel.

        This is a guard for the README claim that v7 works on Windows
        (where mamba-ssm typically fails to build). If the mamba-ssm
        package ever sneaks into requirements.txt, this test will fail
        and force a deliberate decision about Windows support.
        """
        assert HAS_MAMBA_SSM is False, (
            "mamba-ssm is unexpectedly importable; verify the Windows "
            "build story still holds (Triton/mamba-ssm do not build on Win)."
        )

        # Forward must still succeed without the CUDA kernel.
        torch.manual_seed(0)
        model = _build({'sab_type': 'mamba_v2'})
        model.eval()
        x = torch.randn(1, 3, 16, 16)  # tiny crop -> fast reference scan
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 3, 64, 64)
        assert torch.isfinite(y).all()

    # ------------------------------------------------------------------
    # 7. v6 checkpoint keys DO NOT load into a mamba_v2 model.
    # ------------------------------------------------------------------
    def test_v6_warm_start_incompatible_with_mamba_v2(self):
        """Architecture mismatch: a v6 (sab_type=conv3xc) state_dict
        cannot warm-start a NeosrSPAN(sab_type=mamba_v2).

        What we expect:
          - load_state_dict(strict=True) raises RuntimeError.
          - load_state_dict(strict=False) returns 114 missing keys
            (the mamba-specific ones) and 180 unexpected keys (the
            conv3xc-specific ones).
          - The legacy ``load_neosr_weights`` helper reports ~26 loaded,
            ~180 skipped, 0 mismatched (it only loads exact-shape matches
            and silently drops the rest).
        """
        v6_state = _v6_shaped_state_dict()
        model = _build({'sab_type': 'mamba_v2'})

        # Strict load: must fail because every block's c1_r/c2_r/c3_r key
        # is missing in the mamba_v2 model.
        with pytest.raises(RuntimeError):
            model.load_state_dict(v6_state, strict=True)

        # Non-strict load: surface the architecture mismatch clearly.
        model = _build({'sab_type': 'mamba_v2'})
        result = model.load_state_dict(v6_state, strict=False)
        assert len(result.missing_keys) > 100, (
            f"expected many missing keys; got {len(result.missing_keys)}"
        )
        assert len(result.unexpected_keys) > 100, (
            f"expected many unexpected keys; got {len(result.unexpected_keys)}"
        )

        # Spot-check: mamba keys are missing, conv3xc keys are unexpected.
        assert any('block_1.assm.scan.A_log' in k for k in result.missing_keys)
        assert any('block_1.c1_r.sk.weight' in k for k in result.unexpected_keys)

        # The legacy warm-start helper should also report a clear
        # incompatibility signal (skipped > 0, loaded < total).
        tmp_ckpt = _write_tmp_checkpoint(v6_state)
        try:
            model = _build({'sab_type': 'mamba_v2'})
            info = model.load_neosr_weights(tmp_ckpt, strict=False)
            assert info['loaded'] < info['total_ckpt_keys'], (
                "load_neosr_weights unexpectedly reported a full load"
            )
            assert info['skipped'] > 100, (
                f"expected many skipped v6 keys; got {info['skipped']}"
            )
        finally:
            if os.path.exists(tmp_ckpt):
                os.remove(tmp_ckpt)

    # ------------------------------------------------------------------
    # 8. Regression: v6 checkpoint keys DO load into a conv3xc model.
    # ------------------------------------------------------------------
    def test_v6_warm_start_compatible_with_conv3xc(self):
        """A v6-shaped (sab_type=conv3xc) state_dict must warm-start a
        NeosrSPAN(sab_type=conv3xc) with zero missing / unexpected keys.

        This is the v6-warm-start regression: the v7 default must keep the
        exact same parameter layout as v6 so that
        ``finetune_latest.pth`` from a v6 run resumes cleanly.
        """
        v6_state = _v6_shaped_state_dict()
        model = _build({})  # sab_type=conv3xc (default)

        result = model.load_state_dict(v6_state, strict=False)
        assert result.missing_keys == [], (
            f"unexpected missing keys for conv3xc warm-start: "
            f"{result.missing_keys[:5]}..."
        )
        assert result.unexpected_keys == [], (
            f"unexpected extra keys for conv3xc warm-start: "
            f"{result.unexpected_keys[:5]}..."
        )

        # And the legacy helper must report a full load.
        tmp_ckpt = _write_tmp_checkpoint(v6_state)
        try:
            model = _build({})
            info = model.load_neosr_weights(tmp_ckpt, strict=False)
            assert info['loaded'] == info['total_ckpt_keys'], (
                f"warm-start did not load all keys: {info}"
            )
            assert info['skipped'] == 0
            assert info['mismatched'] == 0
        finally:
            if os.path.exists(tmp_ckpt):
                os.remove(tmp_ckpt)

        # Sanity: a forward pass after warm-start produces the right shape.
        model.eval()
        with torch.no_grad():
            y = model(torch.randn(1, 3, 32, 32))
        assert y.shape == (1, 3, 128, 128)


# ---------------------------------------------------------------------------
# Optional config-level integration check (v7 YAML).
# Skipped silently if the config is absent (e.g. in a fresh checkout where
# the file was added but not yet committed).
# ---------------------------------------------------------------------------
class TestMambaSpanV7Config:
    """v7 config wiring: create_neosr_span consumes the YAML correctly."""

    def test_v7_config_default_is_conv3xc(self):
        if not os.path.isfile(V7_CONFIG_PATH):
            pytest.skip(f"v7 config not present at {V7_CONFIG_PATH}")
        import yaml
        with open(V7_CONFIG_PATH, 'r') as f:
            cfg = yaml.safe_load(f)
        model_cfg = cfg.get('model', {})
        assert model_cfg.get('sab_type') == 'conv3xc'
        # Factory can build both variants from this config.
        for sab in ('conv3xc', 'mamba_v2'):
            cfg_copy = copy.deepcopy(cfg)
            cfg_copy['model']['sab_type'] = sab
            m = create_neosr_span(cfg_copy)
            assert isinstance(m, NeosrSPAN)
            expected_cls = MambaSPAB if sab == 'mamba_v2' else SPAB
            assert isinstance(m.block_1, expected_cls)
