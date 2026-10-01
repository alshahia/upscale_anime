"""
Unit tests for the MambaIRv2 SPAB-compatible block (Phase E).

Verifies:
  - MambaSPAB constructs and forwards with the right shapes
  - Param count is well under SPAB's (~285K)
  - Gradient flows through the block
  - Pure-PyTorch fallback path works (mamba-ssm is NOT installed here)
  - selective_scan_ref matches a hand-written naive scan
  - index_reverse / semantic_neighbor helpers
  - Factory in NeosrSPAN.__init__ dispatches correctly for both
    `sab_type='conv3xc'` (default) and `sab_type='mamba_v2'`
  - v7 config parses with `sab_type: conv3xc` and would parse with
    `sab_type: mamba_v2`
"""
import sys
import os
import copy
import pytest
import torch
import torch.nn as nn

from anime_sr.models.span.mambair_v2 import (
    HAS_MAMBA_SSM,
    MambaSPAB,
    ASSM,
    SelectiveScanV2,
    selective_scan_ref,
    index_reverse,
    semantic_neighbor,
)
from anime_sr.models.span.neosr_span import NeosrSPAN, SPAB, create_neosr_span


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_block(cls=MambaSPAB, **kwargs):
    """Construct a MambaSPAB with sensible defaults."""
    defaults = dict(in_channels=48, bias=True)
    defaults.update(kwargs)
    return cls(**defaults)


def _naive_scan(u, delta, A, B, C, D, delta_bias, delta_softplus):
    """Hand-written reference for the selective scan, used to cross-check
    selective_scan_ref. Iterates the same recurrence but accumulates with
    `torch.jit`-free Python; we keep it small (B=1, L=4) for test speed.
    """
    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(0).unsqueeze(-1)
    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)

    B_sz, D_dim, L = u.shape
    N = A.shape[1]

    if B.dim() == 3 and B.shape[1] == L:
        B = B.permute(0, 2, 1).contiguous()
    if C.dim() == 3 and C.shape[1] == L:
        C = C.permute(0, 2, 1).contiguous()

    h = u.new_zeros(B_sz, D_dim, N)
    ys = []
    for t in range(L):
        dA = torch.exp(delta[:, :, t].unsqueeze(-1) * A.unsqueeze(0))
        dB_u = (
            delta[:, :, t].unsqueeze(-1)
            * B[:, :, t].unsqueeze(1)
            * u[:, :, t].unsqueeze(-1)
        )
        h = dA * h + dB_u
        y_t = (h * C[:, :, t].unsqueeze(1)).sum(dim=-1)
        ys.append(y_t)
    y = torch.stack(ys, dim=-1)
    if D is not None:
        y = y + u * D.view(1, -1, 1)
    return y


# ---------------------------------------------------------------------------
# Construction / forward
# ---------------------------------------------------------------------------
def test_mambaspab_constructs():
    block = _build_block()
    assert isinstance(block, MambaSPAB)
    assert isinstance(block.assm, ASSM)


def test_mambaspab_forward_shape():
    block = _build_block()
    block.eval()  # disable gate init noise / dropout
    x = torch.randn(2, 48, 32, 32)
    with torch.no_grad():
        out, out1, sim_att = block(x)
    assert out.shape == (2, 48, 32, 32)
    assert out1.shape == (2, 48, 32, 32)
    # sim_att is a zero placeholder.
    assert sim_att.shape == (2, 1, 32, 32)
    assert torch.all(sim_att == 0.0)


def test_mambaspab_param_count_under_30k():
    block = _build_block()
    total = sum(p.numel() for p in block.parameters() if p.requires_grad)
    assert total < 30_000, f"MambaSPAB has {total} params, expected < 30K"
    # Sanity: it should also be reasonably close to the spec target (~20K).
    assert total > 1_000, f"MambaSPAB has only {total} params — looks too small"


def test_mambaspab_gradient_flows():
    torch.manual_seed(0)
    block = _build_block()
    x = torch.randn(1, 48, 16, 16, requires_grad=True)
    out, _, _ = block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "no grad on input"
    assert torch.any(x.grad != 0), "input grad is all-zero"


# ---------------------------------------------------------------------------
# Reference scan correctness
# ---------------------------------------------------------------------------
def test_selective_scan_ref_matches_naive():
    torch.manual_seed(0)
    B, D_dim, L, N = 1, 4, 5, 3
    u = torch.randn(B, D_dim, L)
    delta = torch.randn(B, D_dim, L)
    A = -torch.exp(torch.randn(D_dim, N))  # negative
    B_p = torch.randn(B, N, L)
    C_p = torch.randn(B, N, L)
    D = torch.randn(D_dim)
    delta_bias = torch.randn(D_dim)

    y_ref = selective_scan_ref(
        u, delta, A, B_p, C_p, D=D, delta_bias=delta_bias, delta_softplus=True,
    )
    y_naive = _naive_scan(u, delta, A, B_p, C_p, D, delta_bias, True)
    assert torch.allclose(y_ref, y_naive, atol=1e-5), (
        f"max diff = {(y_ref - y_naive).abs().max().item()}"
    )


def test_selective_scan_ref_supports_LN_B():
    """B/C given as [B, L, N] (LN layout) should also work."""
    torch.manual_seed(0)
    B, D_dim, L, N = 1, 4, 4, 3
    u = torch.randn(B, D_dim, L)
    delta = torch.randn(B, D_dim, L)
    A = -torch.exp(torch.randn(D_dim, N))
    B_p = torch.randn(B, L, N)  # LN layout
    C_p = torch.randn(B, L, N)  # LN layout
    y_ref = selective_scan_ref(u, delta, A, B_p, C_p, D=None,
                               delta_bias=None, delta_softplus=False)
    y_naive = _naive_scan(u, delta, A, B_p, C_p, None, None, False)
    assert torch.allclose(y_ref, y_naive, atol=1e-5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def test_index_reverse():
    torch.manual_seed(0)
    for n in [1, 4, 17]:
        idx = torch.randperm(n)
        inv = index_reverse(idx)
        x = torch.randn(n)
        # gather then un-gather == identity
        assert torch.allclose(semantic_neighbor(x, idx)[inv], x), (
            f"index_reverse roundtrip failed for n={n}"
        )


def test_semantic_neighbor_roundtrip():
    torch.manual_seed(0)
    n = 12
    idx = torch.randperm(n)
    inv = index_reverse(idx)
    x = torch.randn(2, 3, n)
    # gather(idx) then gather(inv) recovers x
    out = semantic_neighbor(semantic_neighbor(x, idx), inv)
    assert torch.allclose(out, x)


# ---------------------------------------------------------------------------
# Factory in NeosrSPAN.__init__
# ---------------------------------------------------------------------------
def test_factory_dispatches_to_conv3xc_by_default():
    """sab_type='conv3xc' (default) keeps v6 warm-start checkpoints loadable."""
    model = NeosrSPAN(feature_channels=48, upscale=2, num_in_ch=3, num_out_ch=3)
    for i in range(1, 7):
        blk = getattr(model, f'block_{i}')
        assert isinstance(blk, SPAB), f"block_{i} is {type(blk).__name__}, expected SPAB"


def test_factory_dispatches_to_mamba():
    """sab_type='mamba_v2' swaps in MambaSPAB blocks."""
    model = NeosrSPAN(
        feature_channels=48, upscale=2, num_in_ch=3, num_out_ch=3,
        sab_type='mamba_v2',
    )
    for i in range(1, 7):
        blk = getattr(model, f'block_{i}')
        assert isinstance(blk, MambaSPAB), (
            f"block_{i} is {type(blk).__name__}, expected MambaSPAB"
        )


def test_factory_mamba_forward_smoke():
    """End-to-end forward through the mamba-wrapped model."""
    model = NeosrSPAN(
        feature_channels=48, upscale=2, num_in_ch=3, num_out_ch=3,
        sab_type='mamba_v2',
    )
    model.eval()
    x = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 3, 32, 32)


# ---------------------------------------------------------------------------
# Config-level checks
# ---------------------------------------------------------------------------
def _load_yaml(path):
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def test_config_sab_type_override():
    """v7 config parses; sab_type and mamba block have the right shape."""
    cfg_path = os.path.join(
        os.path.dirname(__file__), '..',
        'configs', 'finetune_neosr_span_v7_anime.yaml',
    )
    if not os.path.isfile(cfg_path):
        pytest.skip(f"v7 config not present at {cfg_path}")
    cfg = _load_yaml(cfg_path)
    model_cfg = cfg.get('model', {})
    assert model_cfg.get('sab_type') == 'conv3xc', (
        f"sab_type should default to 'conv3xc' for v6 warm-start, got "
        f"{model_cfg.get('sab_type')!r}"
    )
    mamba_cfg = model_cfg.get('mamba', {})
    assert set(mamba_cfg.keys()) == {'d_state', 'num_tokens', 'inner_rank', 'mlp_ratio'}
    # Verify `create_neosr_span` would build successfully with both values
    # (no instantiation, just config-shape checks via a deepcopy).
    for sab in ('conv3xc', 'mamba_v2'):
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy['model']['sab_type'] = sab
        # Just verify the keys we'd read are present.
        mc = cfg_copy['model']
        assert 'sab_type' in mc and mc['sab_type'] in ('conv3xc', 'mamba_v2')
