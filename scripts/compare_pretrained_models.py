#!/usr/bin/env python3
r"""Run every downloaded pretrained anime SR model on val_sr images and dump PNGs.

Models covered (all under pretrained/):
  - SPAN (4x)            : span_pix_pretrain_4x.pth  via create_neosr_span
  - Real-ESRGAN compact  : realesr-animevideov3.pth, 4xLSDIRCompactv2.pth  (4x)
  - ERANet (2x)          : eranet_N12_pretrain_325k.pth
  - AnimeSR (4x, 3-frame recurrent) : AnimeSR_v2.pth, AnimeSR_v1-PaperModel.pth

Usage:
  .venv\Scripts\python.exe scripts\compare_pretrained_models.py
  .venv\Scripts\python.exe scripts\compare_pretrained_models.py --input results/val_sr/2.png --output results/compare_v1
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from models.span import create_neosr_span  # noqa: E402


# --------------------------------------------------------------------------- #
# Arch stubs (inline because realesrgan/basicsr/animesr/eranet packages are not
# installed; only the bits needed to run inference are included)
# --------------------------------------------------------------------------- #

class SRVGGNetCompact(nn.Module):
    """Real-ESRGAN compact (animevideov3, LSDIRCompactv2). body is a ModuleList
    of (Conv, Act) pairs + final Conv; the forward adds nearest-upsampled input
    as a residual so the network learns the high-frequency detail only.
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


# ---- AnimeSR (TencentARC MSRSWVSR) --------------------------------------- #
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


# ---- ERANet (NevermindNilas/eranet) -------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Loader dispatch
# --------------------------------------------------------------------------- #
def _load_state_dict(path):
    sd = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict):
        for k in ('params', 'params_ema', 'state_dict', 'model_state_dict'):
            if k in sd:
                sd = sd[k]
                break
    return sd


def _save_sr(out, path):
    arr = (out.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), arr)


MODELS = [
    ('SPAN_pix_4x',         'span_pix_pretrain_4x.pth',     'span',    4, 'image'),
    ('RealESRGAN_v3_4x',    'realesr-animevideov3.pth',     'srvgg',   4, 'image'),
    ('RealESRGAN_LSDIR_4x', '4xLSDIRCompactv2.pth',         'srvgg',   4, 'image'),
    ('ERANet_N12_2x',       'eranet_N12_pretrain_325k.pth', 'era',     2, 'image'),
    ('AnimeSR_v2_4x',       'AnimeSR_v2.pth',               'animesr', 4, 'video'),
    ('AnimeSR_v1_4x',       'AnimeSR_v1-PaperModel.pth',    'animesr', 4, 'video'),
]


def _build(name, ckpt_path):
    if name == 'span':
        m = create_neosr_span({'type': 'neosr_span', 'scale': 4})
        m.load_neosr_weights(str(ckpt_path), strict=False)
    elif name == 'srvgg':
        sd = _load_state_dict(ckpt_path)
        m = SRVGGNetCompact()
        m.load_state_dict(sd, strict=False)
    elif name == 'era':
        sd = _load_state_dict(ckpt_path)
        m = ERANet(scale=2, C=32, N=12)
        m.load_state_dict(sd, strict=False)
        m.eval()
    elif name == 'animesr':
        sd = _load_state_dict(ckpt_path)
        m = MSRSWVSR()
        m.load_state_dict(sd, strict=False)
    else:
        raise ValueError(name)
    m.eval()
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', nargs='+', default=['results/val_sr/2.png', 'results/val_sr/1_sr.png'])
    p.add_argument('--output', default='results/compare_pretrained')
    p.add_argument('--pretrained-dir', default='pretrained')
    args = p.parse_args()

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    inputs = [Path(x) for x in args.input]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Compare] device={device}  out={out_root.resolve()}')
    print(f'[Compare] {len(inputs)} input images, {len(MODELS)} models\n')

    summary = []
    for label, ckpt_name, kind, scale, mode in MODELS:
        ckpt_path = Path(args.pretrained_dir) / ckpt_name
        if not ckpt_path.exists():
            print(f'[SKIP] {label}: {ckpt_path} not found')
            continue
        try:
            m = _build(kind, ckpt_path).to(device)
        except Exception as e:
            print(f'[FAIL] {label}: {e}')
            continue

        print(f'=== {label}  ({ckpt_name}, {scale}x, {mode}) ===')
        for img_path in inputs:
            if not img_path.exists():
                print(f'  [SKIP] missing {img_path}')
                continue
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                print(f'  [SKIP] cv2 failed: {img_path}')
                continue
            x = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                                .transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
            t0 = time.time()
            with torch.no_grad():
                if mode == 'video':
                    x = x.unsqueeze(1).expand(-1, 3, -1, -1, -1).contiguous()  # (B,3,C,H,W)
                    y = m(x)
                    y = y[:, y.shape[1] // 2]  # middle frame
                else:
                    y = m(x)
            dt_ms = (time.time() - t0) * 1000
            out_path = out_root / label / img_path.name
            _save_sr(y, out_path)
            print(f'  {img_path.name:<10} {bgr.shape[1]}x{bgr.shape[0]} -> {y.shape[3]}x{y.shape[2]}  '
                  f'{dt_ms:6.1f} ms  mean={float(y.mean()):.3f} std={float(y.std()):.3f}')
            summary.append((label, img_path.name, int(y.shape[3]), int(y.shape[2]), round(dt_ms, 1)))

    print(f'\n[Summary] wrote {len(summary)} outputs under {out_root}/')
    for label, fname, w, h, ms in summary:
        print(f'  {label:24} {fname:12} -> {w}x{h}  ({ms} ms)')


if __name__ == '__main__':
    main()