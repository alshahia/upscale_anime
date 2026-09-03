# anime_upscaler/student.py
"""Lightweight student: RFDN-style Residual Feature Distillation network.

Target < 600K parameters for real-time 4x anime upscaling. Architecture follows
the RFDN concept (Sun et al., AIM 2020): each block progressively distills
features through cheap 1x1 conv branches while a 3x3 branch keeps capacity;
pixel attention gates the block output; global + local residual learning.

return_features=True exposes [shallow, blk1..blkN, body_out] (all nf channels,
LR resolution) for feature distillation against the teacher taps.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelAttention(nn.Module):
    """Channel-wise pixel attention: 1x1 conv + sigmoid gate."""

    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))


class RFDBlock(nn.Module):
    """Residual Feature Distillation block.

    d* branches are cheap 1x1 'distillation' paths; r* branches are 3x3.
    Distilled features are concatenated with the refined remainder and fused
    back to nf channels, then locally residual + pixel-attention gated.
    """

    def __init__(self, nf):
        super().__init__()
        d1, d2 = nf // 2, nf // 4
        self.d1 = nn.Conv2d(nf, d1, 1)
        self.r1 = nn.Conv2d(nf, d1, 3, padding=1)
        self.d2 = nn.Conv2d(d1, d2, 1)
        self.r2 = nn.Conv2d(d1, d2, 3, padding=1)
        self.fuse = nn.Conv2d(d1 + d2 * 2, nf, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.pa = PixelAttention(nf)

    def forward(self, x):
        dist1 = self.act(self.d1(x))
        rem1 = self.act(self.r1(x))
        dist2 = self.act(self.d2(rem1))
        rem2 = self.act(self.r2(rem1))
        out = self.fuse(torch.cat([dist1, dist2, rem2], dim=1))
        return self.pa(out + x)


class RFDN(nn.Module):
    """Student network (scale=2 or 4). Params stay under 600K with defaults."""

    # `scale` is the upscaling factor (2 or 4). The architecture (head, blocks,
    # body_tail, pa) is scale-agnostic; only the upsampler convolution's out
    # channels and the PixelShuffle factor depend on `scale`. Bicubic residual
    # shortcut uses `scale_factor=self.scale` for stability.
    def __init__(self, num_in_ch=3, num_out_ch=3, nf=52, num_blocks=6, scale=4,
                 shortcut_mode: str = "bicubic"):
        super().__init__()
        self.scale = int(scale)
        self.head = nn.Conv2d(num_in_ch, nf, 3, padding=1)
        self.blocks = nn.ModuleList(RFDBlock(nf) for _ in range(num_blocks))
        self.body_tail = nn.Conv2d(nf, nf, 3, padding=1)
        self.pa = PixelAttention(nf)
        self.upsampler = nn.Sequential(
            nn.Conv2d(nf, num_out_ch * self.scale * self.scale, 3, padding=1),
            nn.PixelShuffle(self.scale),
        )
        # Residual shortcut interpolation mode (Phase 4 I1).
        # "bicubic" = legacy v1 (smooth low-pass anchor); "nearest" = animevideov3
        # style (sharp pixel-replicate anchor; body learns the residuals).
        if shortcut_mode not in {"bicubic", "nearest"}:
            raise ValueError(
                f"RFDN: shortcut_mode must be 'bicubic' or 'nearest', got {shortcut_mode!r}")
        self.shortcut_mode = shortcut_mode
        # Bicubic residual shortcut weight (Phase 3 handoff A.4 / gotchas).
        # 1.0 = full shortcut (legacy v1 behavior); the Phase 3 trainer anneals
        # this down to 0.0 over the first 5 epochs so the network is forced to
        # learn detail rather than anchor output near the (mode-selected) anchor.
        self.shortcut_weight = 1.0

    @property
    def upscale(self):
        """Backwards-compat alias for `scale` (older checkpoints and downstream
        consumers used `model.upscale` for the bicubic scale_factor)."""
        return self.scale

    def forward(self, lr01, return_features=False):
        feats = []
        x = self.head(lr01)
        if return_features:
            feats.append(x)                       # tap 0: shallow (vs teacher conv_1)
        res = x
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if return_features and i in (1, 3):   # taps vs teacher block_2 / block_4
                feats.append(x)
        x = self.body_tail(x) + res
        if return_features:
            feats.append(x)                       # tap 3: body_out (vs teacher conv_cat)
        # PyTorch raises if `align_corners` is set for non-interpolating modes
        # (nearest/area/nearest-exact). Only pass it when the mode supports it.
        if self.shortcut_mode in ("bicubic", "bilinear", "linear", "trilinear"):
            shortcut = F.interpolate(
                lr01, scale_factor=self.scale, mode=self.shortcut_mode,
                align_corners=False)
        else:
            shortcut = F.interpolate(
                lr01, scale_factor=self.scale, mode=self.shortcut_mode)
        # Phase 3 handoff A.4: shortcut_weight anneals 1.0 -> 0.0 in epoch 1..5
        # so the student must learn high-frequency detail rather than rest on
        # the shortcut anchor. When shortcut_weight == 0 the residual branch
        # contributes nothing; the student must produce the full SR itself.
        sr = self.upsampler(self.pa(x)) + self.shortcut_weight * shortcut
        if return_features:
            return sr, feats
        return sr

    def set_shortcut_weight(self, w: float) -> None:
        """Phase 3 annealing hook: set the residual shortcut weight.

        Trainer calls this once per epoch. w=1.0 == full shortcut (anchor
        output near the upsampled LR); w=0.0 disables the shortcut entirely.
        """
        self.shortcut_weight = float(w)

    def set_shortcut_mode(self, mode: str) -> None:
        """Phase 4 I1 hook: switch the residual shortcut interpolation mode.

        Valid modes: "bicubic" (legacy v1) or "nearest" (animevideov3 style
        pixel-replicate). Switching at runtime is safe; the next forward
        pass picks up the new mode.
        """
        if mode not in {"bicubic", "nearest"}:
            raise ValueError(
                f"set_shortcut_mode: invalid mode {mode!r} (expected 'bicubic' or 'nearest')")
        self.shortcut_mode = mode

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class TinySRVGGStudent(nn.Module):
    """SRVGG-body student at our 315K budget (Phase 4 I3).

    Mirrors the inductive bias of RealESRGAN's animevideov3 (SRVGGNetCompact):
    a stack of plain 3x3 convs with per-conv PReLU activations + PixelShuffle
    head + nearest residual. There are no skip taps and no attention gates --
    the body learns the residual end-to-end against the nearest-upsampled LR.

    Architectural differences vs RFDN:
    - No feature-distillation branches (no 1x1 d1/d2 splits); distill.py
      treats this student's forward as tap-less (s_feats=[]) regardless of
      which teacher is paired with it.
    - No shortcut anneal (single-path architecture; the residual is fixed at
      weight 1.0 and is not annealed).
    - No return_features path (no meaningful intermediate features at any
      particular depth that would correspond to teacher taps).
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=52, num_conv=12, scale=4):
        super().__init__()
        if num_conv < 1:
            raise ValueError(f"TinySRVGGStudent: num_conv must be >= 1, got {num_conv}")
        self.scale = int(scale)
        self.num_feat = int(num_feat)
        self.num_conv = int(num_conv)
        # Match animevideov3's SRVGGNetCompact exactly: conv -> PReLU -> conv
        # -> PReLU -> ... -> conv (no activation after the final conv, which
        # is then fed straight into PixelShuffle). Each PReLU is its own
        # module instance (per-channel learnable slope), as in RealESRGAN.
        layers = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1),
                  nn.PReLU(num_feat)]
        for _ in range(num_conv):
            layers.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            layers.append(nn.PReLU(num_feat))
        layers.append(nn.Conv2d(num_feat, num_out_ch * self.scale * self.scale, 3, 1, 1))
        self.body = nn.Sequential(*layers)
        self.upsampler = nn.PixelShuffle(self.scale)

    @property
    def upscale(self):
        """Backwards-compat alias for `scale` (downstream consumers + older
        ckpt loaders used `model.upscale`)."""
        return self.scale

    def forward(self, lr01, return_features=False):
        # SRVGG has no meaningful intermediate features for distillation;
        # `return_features` is accepted only to keep a uniform call signature
        # with RFDN so distill.py's dispatch is trivial.
        out = self.body(lr01)
        out = self.upsampler(out)
        out = out + F.interpolate(lr01, scale_factor=self.scale, mode="nearest")
        if return_features:
            return out, []
        return out

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    # The next two hooks mirror RFDN so a uniform ablation harness (e.g. a
    # future shortcut-anneal experiment on SRVGG) could call them without
    # dispatching on architecture. They are no-ops today.
    def set_shortcut_weight(self, w: float) -> None:
        """No-op (SRVGG has no annealable shortcut)."""
        return

    def set_shortcut_mode(self, mode: str) -> None:
        """No-op (SRVGG always uses nearest residual)."""
        if mode not in {"bicubic", "nearest"}:
            raise ValueError(
                f"TinySRVGGStudent.set_shortcut_mode: invalid mode {mode!r} "
                f"(SRVGG uses nearest; this is a no-op)")
        return


if __name__ == "__main__":
    torch.manual_seed(42)
    s = RFDN()
    n = s.num_params()
    assert n < 600_000, f"student too big: {n}"
    lr = torch.randn(2, 3, 48, 48)
    sr, feats = s(lr, return_features=True)
    print(f"RFDN params: {n:,} (<600K OK)")
    print("lr:", tuple(lr.shape), "-> sr:", tuple(sr.shape))
    print("feature taps:", [tuple(f.shape) for f in feats])
    # Latency sanity check on GPU if available.
    if torch.cuda.is_available():
        s = s.cuda().eval()
        lr = torch.randn(1, 3, 270, 480, device="cuda")   # ~1080p output
        with torch.no_grad():
            for _ in range(10):
                s(lr)
            torch.cuda.synchronize()
            import time
            t0 = time.time()
            for _ in range(30):
                s(lr)
            torch.cuda.synchronize()
        print(f"GPU latency: {(time.time()-t0)/30*1000:.2f} ms/frame @270x480 LR")

    # Phase 4 I3: SRVGG-body student smoke check (mirrors the RFDN block above).
    # Note: PReLU's slope is initialised randomly so the first forward differs
    # from a deterministic nn.ReLU; we only check shape + param ceiling here.
    s2 = TinySRVGGStudent()
    n2 = s2.num_params()
    assert n2 < 600_000, f"srvgg student too big: {n2}"
    lr2 = torch.randn(1, 3, 48, 48)
    sr2 = s2(lr2)
    print(f"\nTinySRVGGStudent params: {n2:,} (<600K OK)")
    print("lr:", tuple(lr2.shape), "-> sr:", tuple(sr2.shape))
    # return_features path returns (sr, []) to keep dispatch uniform.
    sr2f, feats2 = s2(lr2, return_features=True)
    assert sr2.shape == sr2f.shape
    assert feats2 == [], f"SRVGG must expose no taps, got {len(feats2)}"
