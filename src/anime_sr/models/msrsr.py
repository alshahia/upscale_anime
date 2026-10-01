"""AnimeSR (TencentARC MSRSWVSR) -- 3-frame recurrent 4x.

Moved from apps/anime_upscaler_gui/anime_upscaler_gui/archs.py to eliminate
vendored code. This is the single source of truth for the MSRSWVSR architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
