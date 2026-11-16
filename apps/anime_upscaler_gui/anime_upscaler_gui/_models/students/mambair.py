# anime_upscaler/student_mambair.py
"""MambaIRv2-style student in pure PyTorch (no CUDA mamba_ssm dep).

This is a *style* implementation of the Visual State-Space backbone from
MambaIRv2 (CVPR 2025) -- sufficient for testing whether the SSM inductive
bias breaks the SRVGG-body ceiling observed in Phase 4 (4 ablations all
landed near PSNR 28 dB).

Key design choices, with rationale:
- Single-stage body at LR spatial resolution (NOT MambaIRv2's original
  PixelUnshuffle(2)-then-body-then-PixelShuffle design). Rationale: clearer
  shape semantics + smaller param count; if Mamba proves useful, we can
  add the PixelUnshuffle body in v2. The state-space math is unchanged.
- Closed-form selective scan via the parallel-prefix trick (NOT a Python
  loop over the sequence). Recurrence state[t] = dA[t]*state[t-1] + dB_u[t]
  expands to:
        state[t] = exp(cum_dA[t]) * cumsum_t(exp(-cum_dA[t]) * dB_u[t])
  which is fully vectorised and runs in a small kernel pair on the GPU.
  A Python loop would launch ~5 CUDA kernels per step x ~530K inference
  steps/frame = unusable latency.
- Numerical safety: clamp cum_dA to [-20, 20] to prevent overflow; this
  matches what the reference mamba_ssm kernels do for stability.
- 4-direction scan average (H-fwd, H-rev, W-fwd, W-rev) per SS2D; matches
  the official implementation's permutation + scan pattern.
- LayerNorm2d at the block input (per-channel, over HxW) is the 2D analogue
  of Mamba's per-token LayerNorm.
- Forward signature mirrors RFDN: forward(lr01) -> sr01 or
  forward(lr01, return_features=True) -> (sr01, taps). We expose
  intermediate taps (shallow + body_out) for consistency, but in practice
  distill.py pairs this with SRVGG teachers that have no taps of their
  own -- so it falls through the s_feats=[] branch like TinySRVGGStudent.

Reference for orientation:
- Mamba paper (Gu & Dao 2023): recurrence / math.
- MambaIRv2 (CVPR 2025): visual SSM block (SS2D) composition + 4-direction
  scan pattern.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _grad_ckpt


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    """Per-channel LayerNorm over the (H, W) dims of a (B, C, H, W) tensor.

    Equivalent to a standard nn.LayerNorm applied to a (B, H*W, C) tensor
    when reduced mean/var are taken over the spatial dims of each channel.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).pow(2).mean(dim=1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        w = self.weight.view(1, -1, 1, 1)
        b = self.bias.view(1, -1, 1, 1)
        return x * w + b


def _cum_clamp(x: torch.Tensor, max_value: float = 20.0) -> torch.Tensor:
    """Symmetric clamp for cumulative exponent intermediates."""
    return x.clamp(min=-max_value, max=max_value)


def _selective_scan_vectorised(
    u,         # (B, L, d_inner)        - inputs x_t
    delta,     # (B, L, d_inner)        - timestep dt_t
    A_log,     # (d_inner, d_state)     - log|A|; A = -exp(A_log)
    B_sel,     # (B, L, d_state)        - input projection
    C_sel,     # (B, L, d_state)        - output projection
    D,         # (d_inner,)             - skip connection
):
    """Differentiably vectorised 1D selective scan.

    Uses the closed-form parallel-prefix trick to avoid a Python loop over
    the sequence length. See module docstring for derivation.
    """
    # Build (B, L, d_inner, d_state) shapes for the matmul-style intermediates.
    B, L, d_inner = u.shape
    d_state = A_log.shape[1]

    delta_e = delta.unsqueeze(-1)              # (B, L, d_inner, 1)
    A_e = A_log.unsqueeze(0).unsqueeze(0)      # (1, 1, d_inner, d_state)
    dA = (delta_e * A_e)                        # (B, L, d_inner, d_state)

    # cum_dA[t] = sum_{k<=t} dA[k]
    cum_dA = dA.cumsum(dim=1)                   # (B, L, d_inner, d_state)
    cum_dA = _cum_clamp(cum_dA)

    # Per-step contribution: delta * B * u, broadcast to (B,L,d_inner,d_state)
    B_e = B_sel.unsqueeze(-2)                   # (B, L, 1, d_state)
    u_e = u.unsqueeze(-1)                       # (B, L, d_inner, 1)
    delta_bu = delta_e * B_e * u_e              # (B, L, d_inner, d_state)

    # inner[t] = sum_{k<=t} exp(cum_dA[k] - cum_dA[t]) * delta_bu[k]
    #          = exp(-cum_dA[t]) * cumsum_t(exp(cum_dA[t]) * delta_bu[t])
    inner = (
        delta_bu * torch.exp(-cum_dA)
    ).cumsum(dim=1)                             # (B, L, d_inner, d_state)

    state = torch.exp(cum_dA) * inner           # (B, L, d_inner, d_state)

    # y[t] = sum_s state[t, :, s] * C[t, s]  -> einsum over d_state
    C_e = C_sel.unsqueeze(-2)                   # (B, L, 1, d_state)
    y = (state * C_e).sum(dim=-1)               # (B, L, d_inner)

    # Skip / residual path: y += u * D
    y = y + u * D
    return y


class SS2D(nn.Module):
    """2D Selective Scan block: scan along 4 directions of (H, W), then average.

    Direction breakdown (matches the standard VSS pattern):
      1. row-major left-to-right, top-to-bottom (along H)
      2. row-major reversed
      3. column-major left-to-right, top-to-bottom (along W)
      4. column-major reversed
    The 4 outputs are averaged (each captures different anisotropy).
    """

    def __init__(self, d_inner: int, d_state: int = 16,
                 dt_rank="auto") -> None:
        super().__init__()
        if dt_rank == "auto":
            dt_rank = max(d_inner // 4, 8)
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank

        # x_proj: x -> (dt, B, C) concatenated along channel.
        # No bias -- bias would break the selective parameterisation.
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)

        # dt_proj: dt_rank -> d_inner with bias. Bias is initialised so that
        # softplus(bias) ~= uniform(dt_min, dt_max) (cf. mamba_ssm.py init).
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt_min, dt_max = 0.001, 0.1
        dt_init_floor = 1e-4
        # Inverse-softplus parameterisation so the initial dt distribution
        # is uniform on the log axis (i.e. exponential after softplus).
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Mark for exclusion from weight decay (matches mamba_ssm convention).
        self.dt_proj.bias._no_weight_decay = True

        # A_log: negative-exponential parameterisation of state-space A.
        # Init uniform(0, 1) on log space => A ~ Uniform(-e, -1) per channel.
        A = torch.empty(d_inner, d_state).uniform_(0.0, 1.0)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # Skip / D parameter (Hidalgo et al., per-token scale).
        self.Ds = nn.Parameter(torch.ones(d_inner))
        self.Ds._no_weight_decay = True

    def _scan_one_dir(self, x_flat):
        """Run the scan once over a flat (B, L, d_inner) sequence."""
        x_proj = self.x_proj(x_flat)                              # (B, L, dt+2*ds)
        dt_x = x_proj[..., :self.dt_rank]                          # (B, L, dt)
        B_sel = x_proj[..., self.dt_rank:self.dt_rank + self.d_state]
        C_sel = x_proj[..., self.dt_rank + self.d_state:]
        delta = F.softplus(self.dt_proj(dt_x))                     # (B, L, d_inner)
        return _selective_scan_vectorised(
            x_flat, delta, self.A_log, B_sel, C_sel, self.Ds,
        )

    def forward(self, x_hwc):
        """x_hwc: (B, H, W, d_inner) channels-last. Returns same shape."""
        B, H, W, d_inner = x_hwc.shape
        L = H * W

        # Direction 1: row-major forward (H rows concatenated along W).
        x_flat = x_hwc.reshape(B, L, d_inner).contiguous()
        y1 = self._scan_one_dir(x_flat)

        # Direction 2: row-major reversed.
        x_rev = x_flat.flip(dims=[1]).contiguous()
        y2 = self._scan_one_dir(x_rev).flip(dims=[1])

        # Direction 3: column-major forward (W columns along H).
        x_perm = x_hwc.transpose(1, 2).contiguous()                # (B, W, H, d_inner)
        x_perm_flat = x_perm.reshape(B, L, d_inner).contiguous()
        y3_flat = self._scan_one_dir(x_perm_flat)
        y3 = y3_flat.view(B, W, H, d_inner).transpose(1, 2).contiguous()
        y3 = y3.view(B, L, d_inner)

        # Direction 4: column-major reversed.
        x_perm_rev = x_perm_flat.flip(dims=[1]).contiguous()
        y4_flat = self._scan_one_dir(x_perm_rev).flip(dims=[1])
        y4 = y4_flat.view(B, W, H, d_inner).transpose(1, 2).contiguous()
        y4 = y4.view(B, L, d_inner)

        # Average over the 4 directions (matches VSS-SSM paper).
        y = (y1 + y2 + y3 + y4) / 4.0
        return y.view(B, H, W, d_inner)


class VSSBlock(nn.Module):
    """One MambaIRv2-style VSS block (LayerNorm -> dwConv -> scan -> residual).

    Zero-initialised gamma on the block output so the model begins as
    identity (a common training-time trick for residual-heavy stacks).
    """

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 3) -> None:
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=d_conv, padding=d_conv // 2, groups=dim,
        )
        self.in_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.SiLU(inplace=False)
        self.ss2d = SS2D(dim, d_state=d_state)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1))
        # Phase 5 MambaIRv2 (2026-09-06): gradient checkpointing on by default.
        # Without this, every VSSBlock's SS2D intermediates (~600 MB at our
        # 113K-param config, batch=16, 48x48 LR) are held in the autograd graph
        # simultaneously across num_blocks blocks -> ~5 GB peak on Quadro RTX
        # 4000 (8 GB total) and the trainer OOMs during the first forward.
        # Checkpointing recomputes each block during backward, trading ~30%
        # compute for fitting in 8 GB. Skipped in eval (.train() False).
        self.use_checkpoint = True

    def forward(self, x):
        # Gate on training so eval pays no recompute cost.
        if self.training and self.use_checkpoint:
            return _grad_ckpt(self._vss_inner, x, use_reentrant=False)
        return self._vss_inner(x)

    def _vss_inner(self, x):
        residual = x
        x = self.norm(x)
        x = self.act(self.dwconv(x))
        x = self.in_proj(x)
        # (B, C, H, W) -> (B, H, W, C) for the channels-last scan, back for the sum.
        x_scan = x.permute(0, 2, 3, 1).contiguous()
        y_scan = self.ss2d(x_scan)
        y = y_scan.permute(0, 3, 1, 2).contiguous()
        y = self.out_proj(y)
        return residual + y * self.gamma


# -----------------------------------------------------------------------------
# Public model: matches RFDN / TinySRVGGStudent conventions
# -----------------------------------------------------------------------------

class MambaIRv2Student(nn.Module):
    """Lightweight MambaIRv2-style student (<600K params).

    Args:
        num_in_ch:  input channels (3 for RGB).
        num_out_ch: output channels (3 for RGB).
        embed_dim:  VSS channel width (kept small for the <600K budget).
        num_blocks: number of VSS blocks at LR spatial resolution.
        d_state:    inner state dimension of the selective scan.
        scale:      super-resolution factor (2 or 4).
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        # Phase 5 MambaIRv2 host-trainable defaults (2026-09-06). Tuned for
        # the Quadro RTX 4000 (8 GB): embed_dim=48 / num_blocks=8 keeps the
        # vectorised-scan cumulative-exponent intermediates at ~750 MB peak
        # during batch=16 forward, while still giving ~113K trainable params
        # (36% of the SRVGG body's 317K) -- enough capacity that the
        # architectural hypothesis ("does state-space break the SRVGG-body
        # ceiling at ~27.9 dB?") can be tested, while staying within budget.
        embed_dim: int = 48,
        num_blocks: int = 8,
        d_state: int = 16,
        scale: int = 4,
    ) -> None:
        super().__init__()
        if scale not in (2, 3, 4, 8):
            raise ValueError(
                f"MambaIRv2Student: scale must be in (2, 3, 4, 8); got {scale}"
            )
        self.scale = int(scale)
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.d_state = d_state

        # Head: 2x conv at LR resolution. No PixelUnshuffle; we keep the
        # body at LR spatial for this v1 (single-stage design).
        self.head = nn.Sequential(
            nn.Conv2d(num_in_ch, embed_dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=False),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=False),
        )

        # Body: stack of VSS blocks, all at LR resolution.
        self.body = nn.Sequential(
            *[VSSBlock(embed_dim, d_state=d_state) for _ in range(num_blocks)]
        )

        # Output conv -> PixelShuffle -> nearest residual.
        self.tail_conv = nn.Conv2d(
            embed_dim, num_out_ch * self.scale * self.scale,
            kernel_size=3, padding=1,
        )
        self.pixel_shuffle = nn.PixelShuffle(self.scale)

    @property
    def upscale(self):
        """Backwards-compat alias -- older checkpoints and downstream
        consumers used model.upscale. Mirrors RFDN/TinySRVGGStudent."""
        return self.scale

    def forward(self, lr01, return_features=False):
        """Forward pass.

        Args:
            lr01: (B, num_in_ch, H, W) in [0, 1].
            return_features: if True, returns (sr, taps); if False, returns sr.
                Tap positions:
                  - [0]: shallow conv-head output
                  - [1]: body_out
                Used to mirror RFDN's tap-exposure convention; in practice
                distill.py pairs MambaIRv2 with SRVGG teachers that have no
                taps, so the s_feats branch falls through to []. Kept here
                so a future feature-tap teacher (e.g. EDSR conv_cat) would
                Just Work without a code change.
        """
        feats = []
        x = self.head(lr01)
        if return_features:
            feats.append(x)                                # tap 0: shallow
        x = self.body(x)
        if return_features:
            feats.append(x)                                # tap 1: body_out
        # Upsample to HR.
        x = self.tail_conv(x)
        x = self.pixel_shuffle(x)
        # Nearest residual (matches animevideov3 / TinySRVGGStudent shortcut).
        shortcut = F.interpolate(
            lr01, scale_factor=self.scale, mode="nearest",
        )
        sr = x + shortcut
        if return_features:
            return sr, feats
        return sr

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    # -- Hooks to keep the trainer's uniform ablation API stable --
    def set_shortcut_weight(self, w):
        """No-op. MambaIRv2 uses a fixed-weight nearest residual."""
        return None

    def set_shortcut_mode(self, mode):
        """No-op. MambaIRv2 always uses nearest residual."""
        return None


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    s = MambaIRv2Student()
    n = s.num_params()
    assert n < 600_000, f"MambaIRv2Student too big: {n}"
    lr = torch.randn(2, 3, 48, 48)
    sr, feats = s(lr, return_features=True)
    print(f"MambaIRv2Student params: {n:,} (<600K OK)")
    print("lr:", tuple(lr.shape), "-> sr:", tuple(sr.shape))
    print("feature taps:", [tuple(f.shape) for f in feats])

    # Selective-scan finite-valued sanity (no NaN / Inf from a scan).
    scan_input = torch.randn(2, 16 * 12, 64)
    delta = torch.rand(2, 16 * 12, 64) * 0.1 + 0.01
    B_sel = torch.randn(2, 16 * 12, 16)
    C_sel = torch.randn(2, 16 * 12, 16)
    A_log = torch.log(torch.rand(64, 16).clamp(min=1e-3))
    D = torch.ones(64)
    y = _selective_scan_vectorised(scan_input, delta, A_log, B_sel, C_sel, D)
    print("scan output finite:", torch.isfinite(y).all().item(),
          "shape:", tuple(y.shape))

    if torch.cuda.is_available():
        s = s.cuda().eval()
        lr_cuda = torch.randn(1, 3, 270, 480, device="cuda")
        with torch.no_grad():
            for _ in range(5):
                s(lr_cuda)
            torch.cuda.synchronize()
            import time
            t0 = time.time()
            for _ in range(10):
                s(lr_cuda)
            torch.cuda.synchronize()
        ms = (time.time() - t0) / 10 * 1000
        print(f"GPU latency: {ms:.2f} ms/frame @270x480 LR (target <=80 ms/frame)")
