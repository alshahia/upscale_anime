"""ERANet (NevermindNilas) -- 2x reparameterized CNN.

Moved from apps/anime_upscaler_gui/anime_upscaler_gui/archs.py to eliminate
vendored code. This is the single source of truth for the ERANet architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
