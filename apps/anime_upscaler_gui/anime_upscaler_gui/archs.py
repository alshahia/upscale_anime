"""Vendored model architectures for the anime-upscaler-gui.

This file is a self-contained copy of the inference-time code from:
  - scripts/compare_pretrained_models.py   (SRVGGNetCompact, MSRSWVSR, ERANet)
  - src/models/span/neosr_span.py          (NeosrSPAN, Conv3XC, SPAB)

Why vendored? So the GUI can be moved to its own project (apps/anime_upscaler_gui)
without dragging the rest of upscale_anime's training pipeline. Edit here only.

Supported architectures (kind strings used by registry.py and the GUI dropdown):
  - "srvgg"        SRVGGNetCompact (Real-ESRGAN compact: animevideov3, LSDIRCompactv2)
  - "span"         NeosrSPAN (Phhofm SPAN pretrains)
  - "era"          ERANet (NevermindNilas; 2x only)
  - "animesr"      MSRSWVSR (TencentARC AnimeSR; recurrent 3-frame)
  - "rfdn_student" RFDN teacher-distilled student (~315K params, 4x only)

NOTE: MambaIRv2 (sab_type='mamba_v2') is intentionally NOT vendored -- the
MambaSPAB class requires mamba-ssm which is Linux-only and only relevant for
finetuning. Inference on a conv3xc checkpoint is the supported path here.
"""
import cv2
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


# ============================================================================ #
# SRVGGNetCompact -- Real-ESRGAN compact variant
# ============================================================================ #
class SRVGGNetCompact(nn.Module):
    """Real-ESRGAN compact (animevideov3, LSDIRCompactv2).

    body is a ModuleList of (Conv, Act) pairs + final Conv; the forward adds
    nearest-upsampled input as a residual so the network learns high-frequency
    detail only.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        super().__init__()
        if act_type == 'prelu':
            Act = lambda: nn.PReLU(num_feat)
        elif act_type == 'leakyrelu':
            Act = lambda: nn.LeakyReLU(0.1, inplace=True)
        else:
            Act = lambda: nn.ReLU(inplace=True)
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(Act())
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(Act())
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)
        self.upscale = upscale

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        out = out + F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        return out


# ============================================================================ #
# AnimeSR (TencentARC MSRSWVSR) -- 3-frame recurrent 4x
# ============================================================================ #
class _ResidualBlockNoBN(nn.Module):
    def __init__(self, num_feat=64):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


def _pixel_unshuffle(x, scale):
    b, c, hh, hw = x.size()
    h, w = hh // scale, hw // scale
    return (x.view(b, c, h, scale, w, scale)
             .permute(0, 1, 3, 5, 2, 4)
             .reshape(b, c * scale * scale, h, w))


class _RightAlignMSConvResidualBlocks(nn.Module):
    def __init__(self, num_in_ch, num_state_ch, num_out_ch, num_block=(5, 3, 2)):
        super().__init__()
        self.num_block = num_block
        self.conv_s1_first = nn.Sequential(
            nn.Conv2d(num_in_ch, num_state_ch, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True))
        self.conv_s2_first = nn.Sequential(
            nn.Conv2d(num_state_ch, num_state_ch, 3, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True))
        self.conv_s4_first = nn.Sequential(
            nn.Conv2d(num_state_ch, num_state_ch, 3, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True))
        self.body_s1_first = nn.ModuleList(_ResidualBlockNoBN(num_state_ch) for _ in range(num_block[0]))
        self.body_s2_first = nn.ModuleList(_ResidualBlockNoBN(num_state_ch) for _ in range(num_block[1]))
        self.body_s4_first = nn.ModuleList(_ResidualBlockNoBN(num_state_ch) for _ in range(num_block[2]))
        self.upsample_x2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_x4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.fusion = nn.Sequential(
            nn.Conv2d(3 * num_state_ch, 2 * num_out_ch, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(2 * num_out_ch, num_out_ch, 3, 1, 1, bias=True))

    def up(self, x, scale):
        if isinstance(x, int):
            return x
        return self.upsample_x2(x) if scale == 2 else self.upsample_x4(x)

    def forward(self, x):
        x_s1 = self.conv_s1_first(x)
        x_s2 = self.conv_s2_first(x_s1)
        x_s4 = self.conv_s4_first(x_s2)
        flag_s2 = False
        flag_s4 = False
        for i in range(self.num_block[0]):
            x_s1 = self.body_s1_first[i](
                x_s1 + (self.up(x_s2, 2) if flag_s2 else 0) + (self.up(x_s4, 4) if flag_s4 else 0))
            if i >= self.num_block[0] - self.num_block[1]:
                x_s2 = self.body_s2_first[i - self.num_block[0] + self.num_block[1]](
                    x_s2 + (self.up(x_s4, 2) if flag_s4 else 0))
                flag_s2 = True
            if i >= self.num_block[0] - self.num_block[2]:
                x_s4 = self.body_s4_first[i - self.num_block[0] + self.num_block[2]](x_s4)
                flag_s4 = True
        return self.fusion(torch.cat((x_s1, self.upsample_x2(x_s2), self.upsample_x4(x_s4)), dim=1))


class MSRSWVSR(nn.Module):
    """AnimeSR architecture (4x recurrent multi-frame)."""
    def __init__(self, num_feat=64, num_block=(5, 3, 2), netscale=4):
        super().__init__()
        self.num_feat = num_feat
        self.netscale = netscale
        num_in = 3 * 3 + 3 * netscale * netscale + num_feat
        num_out = num_feat + 3 * netscale * netscale
        self.recurrent_cell = _RightAlignMSConvResidualBlocks(num_in, num_feat, num_out, num_block)
        self.lrelu = nn.LeakyReLU(0.1)
        self.pixel_shuffle = nn.PixelShuffle(netscale)

    def cell(self, x, fb, state):
        res = x[:, 3:6]
        inp = torch.cat((x, _pixel_unshuffle(fb, self.netscale), state), dim=1)
        out = self.recurrent_cell(inp)
        out_img = self.pixel_shuffle(out[:, :3 * self.netscale * self.netscale]) + F.interpolate(
            res, scale_factor=self.netscale, mode='bilinear', align_corners=False)
        out_state = self.lrelu(out[:, 3 * self.netscale * self.netscale:])
        return out_img, out_state

    def forward(self, x):
        b, n, c, h, w = x.size()
        out = x.new_zeros(b, c, h * self.netscale, w * self.netscale)
        state = x.new_zeros(b, self.num_feat, h, w)
        out_l = []
        for i in range(n):
            if i == 0:
                prev = x[:, i]
                nxt = x[:, i + 1] if i + 1 < n else x[:, i]
            elif i == n - 1:
                prev = x[:, i - 1]
                nxt = x[:, i]
            else:
                prev = x[:, i - 1]
                nxt = x[:, i + 1]
            cur = x[:, i]
            inp = torch.cat((prev, cur, nxt), dim=1)
            out, state = self.cell(inp, out, state)
            out_l.append(out)
        return torch.stack(out_l, dim=1)


# ============================================================================ #
# ERANet (NevermindNilas) -- 2x reparameterized CNN
# ============================================================================ #
def _edge_kernels():
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    lp = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
    return torch.stack([sx, sy, lp])


class _EdgeReparamConv(nn.Module):
    def __init__(self, C, gain=2, edge=True):
        super().__init__()
        self.C = C
        self.edge = edge
        self.sk = nn.Conv2d(C, C, 1)
        self.core = nn.Sequential(
            nn.Conv2d(C, C * gain, 1, bias=False),
            nn.Conv2d(C * gain, C * gain, 3, padding=0),
            nn.Conv2d(C * gain, C, 1),
        )
        if edge:
            self.edge_1x1 = nn.ModuleList([nn.Conv2d(C, C, 1, bias=False) for _ in range(3)])
            self.edge_scale = nn.ParameterList(
                [nn.Parameter(torch.randn(C) * 0.02) for _ in range(3)])
            self.register_buffer("ek", _edge_kernels())
        self.eval_conv = nn.Conv2d(C, C, 3, padding=1)
        self.eval_conv.requires_grad_(False)
        self._loaded_train_branches = False
        self.register_load_state_dict_post_hook(_EdgeReparamConv._post_load)

    @torch.no_grad()
    def _fuse(self):
        w1 = self.core[0].weight
        w2, b2 = self.core[1].weight, self.core[1].bias
        w3, b3 = self.core[2].weight, self.core[2].bias
        w = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2).flip(2, 3).permute(1, 0, 2, 3)
        wc = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0).flip(2, 3).permute(1, 0, 2, 3)
        bc = (w3 * b2.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3
        W = wc + F.pad(self.sk.weight, [1, 1, 1, 1])
        b = bc + self.sk.bias
        if self.edge:
            for i in range(3):
                w1e = self.edge_1x1[i].weight[:, :, 0, 0]
                ker = self.edge_scale[i].view(self.C, 1, 1) * self.ek[i].view(1, 3, 3)
                W = W + torch.einsum("oi,okl->oikl", w1e, ker)
        self.eval_conv.weight.copy_(W)
        self.eval_conv.bias.copy_(b)

    def forward(self, x):
        if self.training:
            xp = F.pad(x, (1, 1, 1, 1))
            out = self.core(xp) + self.sk(x)
            if self.edge:
                for i in range(3):
                    y = self.edge_1x1[i](x)
                    dw = (self.edge_scale[i].view(self.C, 1, 1, 1) * self.ek[i].view(1, 1, 3, 3)).to(y.dtype)
                    out = out + F.conv2d(F.pad(y, (1, 1, 1, 1)), dw, groups=self.C)
            return out
        return self.eval_conv(x)

    def _post_load(self, *_):
        if not self.training and self._loaded_train_branches:
            self._fuse()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        self._loaded_train_branches = (prefix + "core.1.weight") in state_dict
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class _ERABlock(nn.Module):
    def __init__(self, C, gain=2, gate=False, edge=True):
        super().__init__()
        self.conv = _EdgeReparamConv(C, gain, edge)
        self.act = nn.SiLU(inplace=True)
        self.gate = gate

    def forward(self, x):
        if self.training or torch.is_grad_enabled():
            f = x + self.act(self.conv(x))
            if self.gate:
                f = f * (0.5 + torch.sigmoid(f))
            return f
        t = self.act(self.conv(x))
        t.add_(x)
        if self.gate:
            t = t * (0.5 + torch.sigmoid(t))
        return t


class ERANet(nn.Module):
    def __init__(self, scale=2, C=32, N=12, gain=2, line_gate=False, gate_every=4, edge=True):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(3, C, 3, 1, 1)
        self.body = nn.Sequential(*[
            _ERABlock(C, gain, gate=(line_gate and (i + 1) % gate_every == 0), edge=edge)
            for i in range(N)
        ])
        self.tail = nn.Conv2d(C, 3 * scale * scale, 3, 1, 1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)
        self.ps = nn.PixelShuffle(scale)

    def forward(self, x):
        f = self.head(x)
        f = self.body(f)
        hr = self.tail(f) + x.repeat_interleave(self.scale * self.scale, dim=1)
        return self.ps(hr).clamp(0, 1)


# ============================================================================ #
# NeosrSPAN -- SPAN (Phhofm / neosr)
# ============================================================================ #
def _conv_layer(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True) -> nn.Conv2d:
    """Convolution with adaptive padding."""
    padding = (kernel_size - 1) // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


class Conv3XC(nn.Module):
    """Conv3XC from neosr SPAN.

    Training: out = conv(x_padded) + sk(x)
    Eval: fuses sk + conv into eval_conv (3x3), out = eval_conv(x)
    """

    def __init__(self, c_in: int, c_out: int, gain1: int = 1, gain2: int = 0,
                 s: int = 1, bias: bool = True, relu: bool = False):
        super().__init__()
        self.stride = s
        self.has_relu = relu
        gain = gain1

        self.sk = nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, stride=s, bias=bias)
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_in * gain, kernel_size=1, padding=0, bias=bias),
            nn.Conv2d(c_in * gain, c_out * gain, kernel_size=3, stride=s, padding=0, bias=bias),
            nn.Conv2d(c_out * gain, c_out, kernel_size=1, padding=0, bias=bias),
        )
        self.eval_conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=s, bias=bias)
        self.eval_conv.weight.requires_grad_(False)
        if bias:
            self.eval_conv.bias.requires_grad_(False)
        self._fused = False

    def update_params(self):
        """Fuse sk + conv into eval_conv (called automatically in eval mode)."""
        w1 = self.conv[0].weight.data.clone().detach()
        b1 = self.conv[0].bias.data.clone().detach()
        w2 = self.conv[1].weight.data.clone().detach()
        b2 = self.conv[1].bias.data.clone().detach()
        w3 = self.conv[2].weight.data.clone().detach()
        b3 = self.conv[2].bias.data.clone().detach()

        w = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2
        self.weight_concat = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.data.clone().detach()
        sk_w = F.pad(sk_w, [1, 1, 1, 1])

        self.eval_conv.weight.data = self.weight_concat + sk_w
        self.eval_conv.bias.data = self.bias_concat + sk_b

    def train(self, mode: bool = True):
        result = super().train(mode)
        if mode:
            self._fused = False
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            if not self._fused:
                self.update_params()
                self._fused = True
            out = self.eval_conv(x)

        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out


class SPAB(nn.Module):
    """SPAN Attention Block from neosr."""

    def __init__(self, in_channels: int, mid_channels: int = None,
                 out_channels: int = None, bias: bool = False):
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = Conv3XC(in_channels, mid_channels, gain1=2, s=1)
        self.c2_r = Conv3XC(mid_channels, mid_channels, gain1=2, s=1)
        self.c3_r = Conv3XC(mid_channels, out_channels, gain1=2, s=1)
        self.act1 = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor):
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)
        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)
        out3 = self.c3_r(out2_act)
        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att
        return out, out1, sim_att


class NeosrSPAN(nn.Module):
    """SPAN model matching neosr exactly (conv3xc SAB only)."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        feature_channels: int = 48,
        upscale: int = 4,
        bias: bool = True,
        norm: bool = False,
        img_range: float = 1.0,
        rgb_mean: tuple = (0.5, 0.5, 0.5),
    ):
        super().__init__()
        self.upscale = upscale
        self.img_range = img_range
        self.mean = nn.Parameter(torch.Tensor(rgb_mean).view(1, 3, 1, 1), requires_grad=False)

        if not norm:
            self.register_buffer("no_norm", torch.zeros(1))
        else:
            self.no_norm = None

        self.conv_1 = Conv3XC(num_in_ch, feature_channels, gain1=2, s=1)

        SAB_CLS = SPAB
        sab_kwargs = {}

        self.block_1 = SAB_CLS(feature_channels, bias=bias, **sab_kwargs)
        self.block_2 = SAB_CLS(feature_channels, bias=bias, **sab_kwargs)
        self.block_3 = SAB_CLS(feature_channels, bias=bias, **sab_kwargs)
        self.block_4 = SAB_CLS(feature_channels, bias=bias, **sab_kwargs)
        self.block_5 = SAB_CLS(feature_channels, bias=bias, **sab_kwargs)
        self.block_6 = SAB_CLS(feature_channels, bias=bias, **sab_kwargs)

        self.conv_cat = _conv_layer(feature_channels * 4, feature_channels, kernel_size=1, bias=True)
        self.conv_2 = Conv3XC(feature_channels, feature_channels, gain1=2, s=1)

        self.upsampler = nn.Sequential(
            _conv_layer(feature_channels, num_out_ch * (upscale ** 2), kernel_size=3),
            nn.PixelShuffle(upscale),
        )

    @property
    def is_norm(self):
        return self.no_norm is None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_norm:
            self.mean = self.mean.type_as(x)
            x = (x - self.mean) * self.img_range

        out_feature = self.conv_1(x)
        out_b1, _, _ = self.block_1(out_feature)
        out_b2, _, _ = self.block_2(out_b1)
        out_b3, _, _ = self.block_3(out_b2)
        out_b4, _, _ = self.block_4(out_b3)
        out_b5, _, _ = self.block_5(out_b4)
        out_b6, out_b5_2, _ = self.block_6(out_b5)

        out_b6 = self.conv_2(out_b6)
        out = self.conv_cat(torch.cat([out_feature, out_b6, out_b1, out_b5_2], 1))
        output = self.upsampler(out)
        return output

    def load_neosr_weights(self, checkpoint_path: str, strict: bool = False) -> dict:
        """Load weights from neosr/Phhofm checkpoint format."""
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except Exception:
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if isinstance(ckpt, dict):
            for key in ('params', 'params_ema', 'state_dict', 'model_state_dict'):
                if key in ckpt:
                    state = ckpt[key]
                    break
            else:
                state = ckpt
        else:
            state = ckpt

        model_state = self.state_dict()
        loaded = []
        skipped = []
        mismatched = []

        for ckpt_key, ckpt_val in state.items():
            if ckpt_key in model_state:
                if ckpt_val.shape == model_state[ckpt_key].shape:
                    model_state[ckpt_key].copy_(ckpt_val)
                    loaded.append(ckpt_key)
                else:
                    mismatched.append((ckpt_key, str(ckpt_val.shape), str(model_state[ckpt_key].shape)))
            else:
                skipped.append(ckpt_key)

        self.load_state_dict(model_state, strict=strict)

        return {
            'loaded': len(loaded),
            'skipped': len(skipped),
            'mismatched': len(mismatched),
            'total_model_keys': len(model_state),
            'total_ckpt_keys': len(state),
        }


def create_neosr_span(config: dict) -> NeosrSPAN:
    """Factory: build a NeosrSPAN from a config dict (mirrors src/models/span)."""
    model_config = config.get('model', config)
    return NeosrSPAN(
        num_in_ch=model_config.get('num_in_ch', 3),
        num_out_ch=model_config.get('num_out_ch', 3),
        feature_channels=model_config.get('feature_channels', 48),
        upscale=model_config.get('upscale', 4),
        bias=model_config.get('bias', True),
        norm=model_config.get('norm', False),
    )


# ============================================================================ #
# RFDN student -- teacher-distilled 4x anime upscaler (~315K params)
# Vendored copy of anime_upscaler/student.py so the GUI stays self-contained.
# Inference only (no return_features path needed here; that's for training).
# ============================================================================ #
class _RFDN_PixelAttention(nn.Module):
    """Channel-wise pixel attention: 1x1 conv + sigmoid gate."""

    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))


class _RFDN_Block(nn.Module):
    """Residual Feature Distillation block (Sun et al., AIM 2020).

    Two cheap 1x1 distillation branches plus a 3x3 refine branch; outputs
    are concatenated and fused back to nf channels with a 3x3 conv, then
    added back to the input and gated by pixel attention.
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
        self.pa = _RFDN_PixelAttention(nf)

    def forward(self, x):
        dist1 = self.act(self.d1(x))
        rem1 = self.act(self.r1(x))
        dist2 = self.act(self.d2(rem1))
        rem2 = self.act(self.r2(rem1))
        out = self.fuse(torch.cat([dist1, dist2, rem2], dim=1))
        return self.pa(out + x)


class RFDN(nn.Module):
    """RFDN-style student network (scale=2 or 4).

    Defaults match the production v1 student: nf=52, 6 blocks, ~315K params.
    Residual to bicubic-upsampled LR for a stable starting point.
    `scale` controls the upscale factor and final PixelShuffle output shape.

    Phase 4 I1: optional `shortcut_mode` arg (default "bicubic" for backward
    compat with v1 ckpts). When "nearest", the residual shortcut uses
    nearest-neighbor upsampling (matches animevideov3's anchor); the GUI's
    `build()` auto-sniffs this from the ckpt's saved `args.shortcut_mode`.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, nf=52, num_blocks=6, scale=4,
                 shortcut_mode: str = "bicubic"):
        super().__init__()
        if shortcut_mode not in {"bicubic", "nearest"}:
            raise ValueError(
                f"RFDN: shortcut_mode must be 'bicubic' or 'nearest', got {shortcut_mode!r}")
        self.scale = int(scale)
        self.shortcut_mode = shortcut_mode
        self.head = nn.Conv2d(num_in_ch, nf, 3, padding=1)
        self.blocks = nn.ModuleList(_RFDN_Block(nf) for _ in range(num_blocks))
        self.body_tail = nn.Conv2d(nf, nf, 3, padding=1)
        self.pa = _RFDN_PixelAttention(nf)
        self.upsampler = nn.Sequential(
            nn.Conv2d(nf, num_out_ch * self.scale * self.scale, 3, padding=1),
            nn.PixelShuffle(self.scale),
        )

    @property
    def upscale(self):
        """Backwards-compat alias for `scale`."""
        return self.scale

    def forward(self, lr01):
        x = self.head(lr01)
        res = x
        for blk in self.blocks:
            x = blk(x)
        x = self.body_tail(x) + res
        # Phase 4 I1: honor shortcut_mode. PyTorch only accepts `align_corners`
        # for interpolating modes (bicubic/bilinear); nearest/area raise.
        if self.shortcut_mode in ("bicubic", "bilinear"):
            sc = F.interpolate(lr01, scale_factor=self.scale,
                               mode=self.shortcut_mode, align_corners=False)
        else:
            sc = F.interpolate(lr01, scale_factor=self.scale,
                               mode=self.shortcut_mode)
        return self.upsampler(self.pa(x)) + sc


# ============================================================================ #
# TinySRVGGStudent -- vendored copy of anime_upscaler/student.py::TinySRVGGStudent
# Phase 4 I3: the distilled student that mirrors animevideov3's SRVGG-body
# inductive bias at our 315K budget. Always uses nearest residual (matches the
# teacher it's distilled from). Edit here only when anime_upscaler/student.py
# changes; keep them in lockstep.
# ============================================================================ #
class TinySRVGGStudent(nn.Module):
    """SRVGG-body student at our 315K budget (Phase 4 I3).

    Architecture matches animevideov3's SRVGGNetCompact exactly: stacked 3x3
    convs with per-conv PReLU activations + PixelShuffle head + nearest
    residual. No return_features path (no taps to distill against).
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=52, num_conv=12, scale=4):
        super().__init__()
        self.scale = int(scale)
        self.num_feat = int(num_feat)
        self.num_conv = int(num_conv)
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
        """Backwards-compat alias for `scale`."""
        return self.scale

    def forward(self, lr01):
        out = self.body(lr01)
        out = self.upsampler(out)
        return out + F.interpolate(lr01, scale_factor=self.scale, mode="nearest")

    def num_params(self):
        """Sum of all parameter counts. Mirrors the canonical
        anime_upscaler/student.py::TinySRVGGStudent so the GUI parity test
        (tests/test_tiny_srvgg.py::test_vendored_archs_matches_student_py)
        can compare against the same definition."""
        return sum(p.numel() for p in self.parameters())


# ============================================================================ #
# Loader helpers
# ============================================================================ #
def _load_state_dict(path):
    """Load a checkpoint and extract its state dict (handles params/state_dict wrappers).

    Accepts any of these top-level keys: params, params_ema, state_dict,
    model_state_dict, student (KD pipeline), params_ema.
    """
    sd = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict):
        for k in ('params', 'params_ema', 'state_dict', 'model_state_dict', 'student'):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    return sd


def _save_sr(out, path):
    """Save a (1, 3, H, W) tensor in [0,1] as a uint8 BGR PNG/JPG via cv2."""
    arr = (out.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    path = __import__('pathlib').Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), arr)


# ============================================================================ #
# Extensible architecture registry
# ---------------------------------------------------------------------------- #
# Adding a new model kind is a single registration point: build an ArchSpec
# and call register_arch(). Nothing else in the codebase needs to change --
# pipeline.py, trt_engine.py, registry.py and the GUI all query the spec
# table instead of hard-coding kind-string checks.
# ============================================================================ #


@dataclass(frozen=True)
class Capability:
    """Runtime abilities the pipeline / TRT / VRAM-guard layers query.

    Defaults describe the common case (plain feed-forward 2D CNN: tiling,
    TensorRT, and video batching all allowed). Models with unusual behavior
    override fields at registration time so downstream code never needs a
    kind-string check.
    """

    tiled: bool = True
    tensorrt: bool = True
    batch_video: bool = True
    # >0: the model wants a (B, N, 3, H, W) input built by repeating the
    # frame N times; the backend takes the center frame of the output.
    recurrent_frames: int = 0
    # Pad the input H/W to this pixel multiple (0 = no padding needed).
    pad_multiple: int = 0
    # >0: crop the SR output back to input_h * this after padded inference.
    out_multiple: int = 0


@dataclass(frozen=True)
class ArchSpec:
    """How to build, detect and query one architecture kind.

    detect: takes a state_dict, returns a kind string or None for a
        different kind. Mutually precise: a spec only claims checkpoints
        it recognizes.
    loader: takes a checkpoint path, returns an eval-ready nn.Module.
    """

    kind: str
    detect: Callable[[dict], Optional[str]]
    loader: Callable[[str], nn.Module]
    default_scale: int = 4
    capability: Capability = Capability()


_REGISTRY: Dict[str, ArchSpec] = {}


def register_arch(spec: ArchSpec) -> ArchSpec:
    """Register (or replace) an architecture spec. Last registration wins."""
    _REGISTRY[str(spec.kind)] = spec
    return spec


def unregister_arch(kind: str) -> None:
    """Remove a registered kind (mainly for tests/experiments)."""
    _REGISTRY.pop(str(kind), None)


def registered_kinds() -> List[str]:
    """Kinds currently registered, in registration order."""
    return list(_REGISTRY)


def spec_of(kind: str) -> Optional[ArchSpec]:
    """Look up the spec for `kind`, or None."""
    return _REGISTRY.get(str(kind))


def capabilities(kind: str) -> Capability:
    """Capability flags for `kind`; unknown kinds get the safe defaults."""
    s = spec_of(kind)
    return s.capability if s else Capability()


def is_supported_kind(kind: str) -> bool:
    """True when the registry knows how to build this kind."""
    return str(kind) in _REGISTRY


def detect_kind_from_state(state: dict) -> Optional[str]:
    """Sniff the state_dict to identify the architecture.

    Registered in order; the baseline detectors are mutually precise, so
    ordering is not load-bearing. Detection never raises regardless of the
    state dict content.
    """
    if not isinstance(state, dict):
        return None
    for s in _REGISTRY.values():
        try:
            if s.detect(state):
                return s.kind
        except Exception:
            continue
    return None


def build(kind: str, ckpt_path) -> nn.Module:
    """Build + load + set eval() on the right architecture.

    kind identifiers live entirely in the registry: registering a new
    ArchSpec (see _register_builtin_archs below) is the only change needed
    to support a new checkpoint family.
    """
    s = spec_of(kind)
    if s is None:
        raise ValueError("unsupported arch kind: %r. registered kinds: %s"
                         % (kind, ", ".join(sorted(_REGISTRY))))
    m = s.loader(str(ckpt_path))
    m.eval()
    return m


# ---- per-kind detection helpers (moved verbatim from registry.py) -------- #

def _detect_span(state: dict) -> Optional[str]:
    # SPAN: conv_1 / conv_2 / block_1..6 / upsampler
    keys = list(state.keys())
    keyset = set(keys)
    if "conv_1.sk.weight" in keyset and "upsampler.0.weight" in keyset:
        return "span"
    return None


def _detect_srvgg(state: dict) -> Optional[str]:
    keys = list(state.keys())
    keyset = set(keys)
    # SRVGGNetCompact: body.0.weight ... body.34.weight. The PixelShuffle
    # upsampler has no parameters so it never shows in the state_dict.
    if any(k == "body.0.weight" for k in keys) and "body.34.weight" in keyset:
        return "srvgg"
    return None



def _load_srvgg(ckpt_path: str) -> nn.Module:
    m = SRVGGNetCompact()
    m.load_state_dict(_load_state_dict(ckpt_path), strict=False)
    return m


def _load_srvgg_student(ckpt_path: str) -> nn.Module:
    """SRVGG-body distilled student (Phase 4 I3).

    We sniff num_feat from body.0.weight (input conv) and num_conv from
    counting the conv layers in body. Scale is sniffed from the last body
    conv's out_ch (= num_out_ch * scale * scale). Defaults match the
    production run (num_feat=52, num_conv=12, scale=4).
    """
    import math
    sd = _load_state_dict(ckpt_path)
    num_feat = 52
    first_w = sd.get('body.0.weight') if isinstance(sd, dict) else None
    if first_w is not None and hasattr(first_w, 'shape') and len(first_w.shape) == 4:
        num_feat = int(first_w.shape[0])
    # body.<i>.weight keys: even i are conv weights (0, 2, 4, ..., 2*(num_conv+1))
    body_conv_keys = sorted([k for k in sd.keys()
                             if k.startswith('body.') and k.endswith('.weight')
                             and (k.split('.')[1].isdigit()
                                  and int(k.split('.')[1]) % 2 == 0)],
                            key=lambda k: int(k.split('.')[1]))
    # First is head, last is the upsample pre-shuffle conv; the middle
    # [len(body_conv_keys) - 2] are the internal body convs.
    num_conv = max(len(body_conv_keys) - 2, 1)
    scale = 4
    if len(body_conv_keys) >= 2:
        last_w = sd[body_conv_keys[-1]]
        if hasattr(last_w, 'shape') and len(last_w.shape) == 4:
            out_ch = int(last_w.shape[0])
            ratio = out_ch // 3
            sq = int(round(math.sqrt(max(ratio, 1))))
            if sq * sq == ratio and sq in (2, 3, 4, 8):
                scale = sq
    m = TinySRVGGStudent(num_feat=num_feat, num_conv=num_conv, scale=scale)
    m.load_state_dict(sd, strict=False)
    return m


def _load_era(ckpt_path: str) -> nn.Module:
    m = ERANet(scale=2, C=32, N=12)
    m.load_state_dict(_load_state_dict(ckpt_path), strict=False)
    return m


def _load_animesr(ckpt_path: str) -> nn.Module:
    m = MSRSWVSR()
    m.load_state_dict(_load_state_dict(ckpt_path), strict=False)
    return m


def _load_rfdn_student(ckpt_path: str) -> nn.Module:
    """RFDN teacher-distilled student (Phase 4 I1).

    Sniff shortcut_mode from the full ckpt's saved args (default bicubic for
    backward compat with v1 ckpts that have no args dict), and scale from the
    upsampler's conv weight shape: scale=2 -> out_ch = 3*4 = 12; scale=4 ->
    48 (the v1_4x ckpt in the repo); scale=3 -> 27 (legal). Uses
    sqrt(out_ch / num_out_ch) so future scales (e.g. 8x) also work.
    """
    import math
    shortcut_mode = "bicubic"
    try:
        full = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(full, dict) and isinstance(full.get('args'), dict):
            sm = full['args'].get('shortcut_mode')
            if sm in ('bicubic', 'nearest'):
                shortcut_mode = sm
    except Exception:
        pass
    sd = _load_state_dict(ckpt_path)
    num_out_ch = 3
    ups_w = sd.get('upsampler.0.weight') if isinstance(sd, dict) else None
    if ups_w is not None and hasattr(ups_w, 'shape') and len(ups_w.shape) == 4:
        out_ch = int(ups_w.shape[0])
        ratio = out_ch // num_out_ch
        scale_sqrt = int(round(math.sqrt(max(ratio, 1))))
        if scale_sqrt * scale_sqrt == ratio and scale_sqrt in (2, 3, 4, 8):
            m = RFDN(scale=scale_sqrt, shortcut_mode=shortcut_mode)
        else:
            m = RFDN(shortcut_mode=shortcut_mode)  # defaults to 4
    else:
        m = RFDN(shortcut_mode=shortcut_mode)  # weight-shape sniff failed
    m.load_state_dict(sd, strict=False)
    return m


def _load_span(ckpt_path: str) -> nn.Module:
    m = create_neosr_span({'type': 'neosr_span', 'scale': 4})
    m.load_neosr_weights(ckpt_path, strict=False)
    return m


def _detect_era(state: dict) -> Optional[str]:
    # ERANet: head/body.N.conv.{ek,sk,core,eval_conv}/tail
    keys = list(state.keys())
    keyset = set(keys)
    if "head.weight" in keyset and "tail.weight" in keyset and any(
        k.startswith("body.") and ".conv.ek" in k for k in keys
    ):
        return "era"
    return None


def _detect_animesr(state: dict) -> Optional[str]:
    # AnimeSR (MSRSWVSR): recurrent_cell, body_sN_first, fusion
    keys = list(state.keys())
    keyset = set(keys)
    if "recurrent_cell.conv_s1_first.0.weight" in keyset or any(
        k.startswith("recurrent_cell.") for k in keys
    ):
        return "animesr"
    return None


def _detect_rfdn_student(state: dict) -> Optional[str]:
    """RFDN student (anime_upscaler.student.RFDN): head.* ->
    blocks.N.{d1,r1,d2,r2,fuse,pa}*, body_tail.*, pa.pa_conv.*, upsampler.0.*.
    The blocks.<N>.pa.pa_conv pattern is unique to our RFDN (vs. SPAN's
    block_<N>.c1_r.*) so it is safe to detect.
    """
    keys = list(state.keys())
    keyset = set(keys)
    if (
        "head.weight" in keyset
        and "body_tail.weight" in keyset
        and "pa.pa_conv.weight" in keyset
        and "upsampler.0.weight" in keyset
        and any(
            k.startswith("blocks.") and ".d1.weight" in k for k in keys
        )
    ):
        return "rfdn_student"
    return None


def _detect_srvgg_student(state: dict) -> Optional[str]:
    """TinySRVGGStudent (anime_upscaler/student.py) shares the SRVGG
    Sequential-body layout but with a different depth, so it never has
    body.34.weight. Convs sit at even body indices; the final conv is the
    last even index with out_ch = 3 * scale^2 (scale in 2/3/4/8). Detected
    structurally (even-index convs, no body.34); the srvgg spec above catches
    num_conv=16 SRVGG (which has body.34).
    """
    keys = list(state.keys())
    keyset = set(keys)
    if "body.0.weight" in keyset and "body.34.weight" not in keyset:
        even_w = [k for k in keys if re.fullmatch(r"body\.\d+\.weight", k)
                  and int(k.split(".")[1]) % 2 == 0]
        body_last = max((int(k.split(".")[1]) for k in keys if k.startswith("body."))
                        if any(k.startswith("body.") for k in keys) else [],
                        default=-1)
        if (len(even_w) >= 2 and even_w[-1].split(".")[1] == str(body_last)
                and body_last % 2 == 0):
            return "srvgg_student"
    return None


def _register_builtin_archs() -> None:
    """Register the kinds shipped with the GUI. Registration order is the
    detection order; the detectors are mutually precise so ordering is not
    load-bearing.

    How to add a model family: write a _load_x / _detect_x pair and register
    an ArchSpec here -- no other module needs touching. Extension modules can
    also register kinds at their own import time via register_arch().
    """
    register_arch(ArchSpec(kind="span", detect=_detect_span,
                           loader=_load_span, default_scale=4))
    register_arch(ArchSpec(kind="srvgg", detect=_detect_srvgg,
                           loader=_load_srvgg, default_scale=4))
    register_arch(ArchSpec(kind="srvgg_student", detect=_detect_srvgg_student,
                           loader=_load_srvgg_student, default_scale=4))
    register_arch(ArchSpec(kind="era", detect=_detect_era,
                           loader=_load_era, default_scale=2))
    register_arch(ArchSpec(kind="animesr", detect=_detect_animesr,
                           loader=_load_animesr, default_scale=4,
                           capability=Capability(
                               tiled=False, tensorrt=False,
                               batch_video=False, recurrent_frames=3,
                               pad_multiple=4, out_multiple=4)))
    register_arch(ArchSpec(kind="rfdn_student", detect=_detect_rfdn_student,
                           loader=_load_rfdn_student, default_scale=4))


_register_builtin_archs()


# Public surface
__all__ = [
    "SRVGGNetCompact", "MSRSWVSR", "ERANet", "NeosrSPAN", "Conv3XC", "SPAB",
    "RFDN", "TinySRVGGStudent",
    "create_neosr_span", "build", "_save_sr", "_load_state_dict",
    "ArchSpec", "Capability", "register_arch", "unregister_arch",
    "registered_kinds", "spec_of", "capabilities", "is_supported_kind",
    "detect_kind_from_state",
]
