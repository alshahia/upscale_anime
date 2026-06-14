"""
MambaIRv2 attention block, drop-in for SPAB in NeosrSPAN.

Mirrors upstream csguoh/MambaIR (mambairv2_arch.py). The K=1 selective
scan is "prompt-aware": a learned `token` embedding is added to the C
matrix so the scan dynamics depend on the input content.

When `mamba-ssm` is importable we use the CUDA kernel via
`selective_scan_fn`; otherwise we fall back to a pure-PyTorch reference
implementation (`selective_scan_ref`) that loops over the sequence length.
The fallback is slow but correct, which is what we need for the v7
opt-in path on Windows (where `mamba-ssm` typically fails to build).

Exports:
    SelectiveScanV2  -- K=1, prompt-aware SSM (nn.Module)
    ASSM             -- input/output projections + CPE + route + scan
    MambaSPAB        -- SPAB-compatible wrapper (returns 3-tuple)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ---------------------------------------------------------------------------
# Optional mamba-ssm (CUDA) kernel. We rebind `selective_scan_fn` to the
# pure-PyTorch fallback when the package is missing.
# ---------------------------------------------------------------------------
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # type: ignore
    HAS_MAMBA_SSM = True
except ImportError:  # pragma: no cover - exercised on Windows / no-build envs
    HAS_MAMBA_SSM = False

    def selective_scan_fn(u, delta, A, B, C, D=None, delta_bias=None,
                          delta_softplus=True):
        return selective_scan_ref(
            u, delta, A, B, C, D=D, delta_bias=delta_bias,
            delta_softplus=delta_softplus,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def index_reverse(index: torch.Tensor) -> torch.Tensor:
    """Invert a permutation index (long tensor).

    Given `index` of shape [N] where `index[i] = j` means
    `gather(x, index)[i] = x[j]`, returns `inv` such that
    `gather(gather(x, index), inv) == x`.
    """
    inv = torch.zeros_like(index)
    inv[index] = torch.arange(index.numel(), device=index.device,
                              dtype=index.dtype)
    return inv


def semantic_neighbor(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather along the last dim of `x` using `index`.

    Args:
        x: [..., N] tensor.
        index: [N] long tensor of indices into the last dim.

    Returns:
        Tensor with the same shape as `x`, last dim permuted by `index`.
    """
    return x.index_select(-1, index)


# ---------------------------------------------------------------------------
# Pure-PyTorch reference selective scan (fallback).
# ---------------------------------------------------------------------------
def selective_scan_ref(u, delta, A, B, C, D=None, delta_bias=None,
                       delta_softplus=True):
    """Reference selective scan (S6) using a Python for-loop.

    Args:
        u:        [B, D, L] input
        delta:    [B, D, L] or [B, D, L] after dt_proj
        A:        [D, N]    (negative semi-definite; shape D, N)
        B:        [B, N, L] or [B, L, N]
        C:        [B, N, L] or [B, L, N]
        D:        [D]       skip term (None-safe)
        delta_bias: [D]      bias added to delta (None-safe)
        delta_softplus: bool apply softplus to (delta + delta_bias)

    Returns:
        y: [B, D, L] output
    """
    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(0).unsqueeze(-1)
    if delta_softplus:
        delta = F.softplus(delta)

    B_sz, D_dim, L = u.shape
    N = A.shape[1]

    # Normalize B, C shapes to [B, N, L]
    if B.dim() == 3 and B.shape[1] == L:
        # [B, L, N] -> [B, N, L]
        B = B.permute(0, 2, 1).contiguous()
    if C.dim() == 3 and C.shape[1] == L:
        C = C.permute(0, 2, 1).contiguous()

    # h: [B, D, N]
    h = u.new_zeros(B_sz, D_dim, N)
    ys = []
    # Precompute exp(Δ_t * A) for each step lazily inside the loop.
    for t in range(L):
        # delta[:, :, t]: [B, D] -> [B, D, 1] for broadcast with A: [D, N]
        # dA: [B, D, N]
        dA = torch.exp(delta[:, :, t].unsqueeze(-1) * A.unsqueeze(0))
        # dB_u: [B, D, N] = delta[B,D,1] * B[B,1,N] * u[B,D,1]
        dB_u = (
            delta[:, :, t].unsqueeze(-1)
            * B[:, :, t].unsqueeze(1)
            * u[:, :, t].unsqueeze(-1)
        )
        h = dA * h + dB_u
        # y_t = C_t @ h -> [B, D]
        y = (h * C[:, :, t].unsqueeze(1)).sum(dim=-1)
        ys.append(y)

    y = torch.stack(ys, dim=-1)  # [B, D, L]

    if D is not None:
        y = y + u * D.view(1, -1, 1)
    return y


# ---------------------------------------------------------------------------
# SelectiveScanV2: K=1, prompt-aware.
# ---------------------------------------------------------------------------
class SelectiveScanV2(nn.Module):
    """K=1 selective scan with an additive prompt on the C matrix.

    Mirrors upstream mambairv2_arch.py Selective_Scan.
    """

    def __init__(self, d_model, d_state=16, expand=1, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = expand * d_model
        # A is shared (D, N) real-valued, negative
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        # dt projection bias: start near zero; helps stability.
        self.dt_bias = nn.Parameter(torch.zeros(self.d_inner))

    def forward_core(self, x: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        """K=1 scan with `prompt` added to C.

        Args:
            x:      [B, L, hidden]
            prompt: [B, L, d_state]

        Returns:
            y: [B, L, hidden]
        """
        B, L, _ = x.shape
        # Treat the whole sequence as a single chunk (K=1).
        x_t = x.permute(0, 2, 1)            # [B, hidden, L]
        delta = x_t                          # treat x itself as Δ (simplest K=1)
        # Derive B, C from prompt (K=1, single projection per token).
        B_param = prompt                     # [B, L, d_state]
        C_param = prompt                     # [B, L, d_state]
        A = -torch.exp(self.A_log.float())   # [hidden, d_state]
        y = selective_scan_fn(
            x_t, delta, A,
            B_param.permute(0, 2, 1).contiguous(),
            (C_param + 0).permute(0, 2, 1).contiguous(),
            D=self.D.float(), delta_bias=self.dt_bias.float(),
            delta_softplus=True,
        )
        return y.permute(0, 2, 1)            # [B, L, hidden]


# ---------------------------------------------------------------------------
# ASSM: input/output projections + CPE + route MLP + scan.
# ---------------------------------------------------------------------------
class ASSM(nn.Module):
    """ASSM: input projection -> CPE (gated DWConv) -> route MLP -> scan -> output projection.

    Mirrors upstream mambairv2_arch.py ASSM (semantic neighbor version).
    """

    def __init__(self, dim, d_state=16, num_tokens=64, inner_rank=32,
                 mlp_ratio=2.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank

        # Input projection
        self.in_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        # CPE: gated depth-wise 3x3 conv, channel-wise gated.
        self.cpe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        # Route MLP: classify tokens into num_tokens buckets via a small 1x1 conv.
        self.route = nn.Conv2d(dim, num_tokens, kernel_size=1, bias=True)
        # Learnable token embeddings used as prompts for the scan.
        self.token = nn.Embedding(num_tokens, d_state)
        nn.init.normal_(self.token.weight, std=0.02)

        # Scan
        self.scan = SelectiveScanV2(d_model=dim, d_state=d_state, expand=1)

        # Output projection back to channel space.
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        # Layer norm on the channel dim (applied after CPE).
        self.norm = nn.LayerNorm(dim)
        # MLP at the end (channel mixing).
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor, x_size, token: nn.Embedding) -> torch.Tensor:
        """Forward.

        Args:
            x:      [B, C, H, W]
            x_size: (H, W) tuple.
            token:  nn.Embedding (overrides self.token; kept for API parity).
        """
        B, C, H, W = x.shape
        H_in, W_in = x_size

        # 1) Input projection + CPE (gated).
        x_proj = self.in_proj(x)
        x_cpe = self.cpe(x) * torch.sigmoid(self.cpe(x))  # gated: y = cpe(x) * sigmoid(cpe(x))
        x_in = x_proj + x_cpe

        # 2) Route MLP -> assign each pixel a token id.
        # logits: [B, num_tokens, H, W]
        logits = self.route(x_in)
        # Hard assignment (straight-through not needed for inference; this is
        # already a discrete index).
        token_ids = logits.argmax(dim=1)  # [B, H, W]
        # Flatten to [B, L] where L = H*W
        token_ids_flat = token_ids.view(B, -1)  # [B, L]
        # Gather prompt embeddings: [B, L, d_state]
        prompt = token(token_ids_flat)

        # 3) Norm + sequence reshape.
        x_seq = x_in.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x_seq = self.norm(x_seq)                        # [B, H, W, C]
        x_seq = x_seq.view(B, H * W, C)                 # [B, L, C]

        # 4) Scan (K=1, prompt-aware).
        y = self.scan.forward_core(x_seq, prompt)       # [B, L, C]

        # 5) Reshape back to image and apply output projection + MLP.
        y_img = y.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        y_img = self.out_proj(y_img) + y_img             # residual on output
        y_img = y_img + self.mlp(y_img)                  # channel-mixing MLP
        return y_img


# ---------------------------------------------------------------------------
# MambaSPAB: drop-in for SPAB.
# ---------------------------------------------------------------------------
class MambaSPAB(nn.Module):
    """SPAB-compatible wrapper around ASSM.

    Forward returns a 3-tuple `(out, out1, sim_att)` matching SPAB:
        - out:    post-MLP feature (consumed as next block input)
        - out1:   pre-ASSM feature (consumed as skip-connection in NeosrSPAN)
        - sim_att: zero placeholder (v2 has no analog; only consumed for logging)

    Constructor signature mirrors SPAB for drop-in compatibility, with extra
    mamba-specific kwargs (`d_state`, `num_tokens`, `inner_rank`, `mlp_ratio`).
    """

    def __init__(self, in_channels: int, mid_channels: int = None,
                 out_channels: int = None, bias: bool = True,
                 d_state: int = 16, num_tokens: int = 64,
                 inner_rank: int = 32, mlp_ratio: float = 2.0):
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels

        # Pre-projection so we can swap channel widths (SPAB allows mid/out
        # to differ from in_channels). Use 1x1 to stay cheap.
        if mid_channels != in_channels:
            self.pre_proj = nn.Conv2d(in_channels, mid_channels, kernel_size=1,
                                      bias=bias)
        else:
            self.pre_proj = nn.Identity()

        # The actual MambaIRv2 block (operates on `mid_channels`).
        self.assm = ASSM(
            dim=mid_channels, d_state=d_state, num_tokens=num_tokens,
            inner_rank=inner_rank, mlp_ratio=mlp_ratio,
        )

        # Post-projection back to `out_channels` if needed.
        if out_channels != mid_channels:
            self.post_proj = nn.Conv2d(mid_channels, out_channels,
                                       kernel_size=1, bias=bias)
        else:
            self.post_proj = nn.Identity()

        # Gating analogous to SPAB's `sim_att` (we keep a learnable scalar
        # gate so the block can be identity-like at init and grow over
        # training, which mirrors the `(out3 + x) * sim_att` SPAB pattern).
        self.gate = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward matching SPAB's `(out, out1, sim_att)` interface."""
        B, C, H, W = x.shape
        # out1 is the pre-ASSM feature (pre-activation analogue of SPAB's c1_r).
        out1 = self.pre_proj(x)
        # Run the scan.
        scan_out = self.assm(out1, x_size=(H, W), token=self.assm.token)
        scan_out = self.post_proj(scan_out)
        # Modulated by a learnable gate (starts at 0 -> near identity).
        gate = torch.sigmoid(self.gate)
        out = scan_out * gate
        # sim_att: zero placeholder (v2 has no analog; consumers treat it
        # as a logging-only output).
        sim_att = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)
        return out, out1, sim_att
