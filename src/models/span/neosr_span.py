"""
Neosr-compatible SPAN model.
Matches the EXACT architecture from neosr-project/neosr (neosr/archs/span_arch.py).

Key components:
- Conv3XC: Training uses sk + conv paths; eval fuses them into eval_conv (3x3)
- SPAB: 3x Conv3XC + SiLU activations + sigmoid attention modulation
- span: conv_1 -> 6x SPAB -> conv_cat -> conv_2 -> upsampler
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from typing import Dict, Tuple

from models.span.mambair_v2 import MambaSPAB


def conv_layer(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True) -> nn.Conv2d:
    """Convolution with adaptive padding."""
    padding = (kernel_size - 1) // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


class Conv3XC(nn.Module):
    """
    Conv3XC from neosr SPAN.

    Training mode: out = conv(x_padded) + sk(x)
    Eval mode: fuses sk + conv into eval_conv (3x3), out = eval_conv(x)

    Structure:
    - sk: 1x1 conv
    - conv: 1x1 -> 3x3 (no padding!) -> 1x1
    - eval_conv: 3x3 conv (filled at eval time via update_params)

    Caching (Phase 2 #6): the original code called `update_params()` on every
    forward in eval mode, fusing ~20 Conv3XC layers per pass. We add a
    `_fused` flag set in eval and cleared when the module transitions to
    train mode. `train(mode)` is overridden to invalidate the cache so that
    any later switch to eval re-fuses with the new (possibly fine-tuned)
    weights.
    """

    def __init__(self, c_in: int, c_out: int, gain1: int = 1, gain2: int = 0,
                 s: int = 1, bias: bool = True, relu: bool = False):
        super().__init__()
        self.stride = s
        self.has_relu = relu
        gain = gain1

        # SK branch: 1x1 conv
        self.sk = nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, stride=s, bias=bias)

        # Conv bottleneck: 1x1 -> 3x3 -> 1x1
        # NOTE: conv[1] (3x3) has padding=0!
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_in * gain, kernel_size=1, padding=0, bias=bias),
            nn.Conv2d(c_in * gain, c_out * gain, kernel_size=3, stride=s, padding=0, bias=bias),
            nn.Conv2d(c_out * gain, c_out, kernel_size=1, padding=0, bias=bias),
        )

        # Eval conv: 3x3 (filled at eval time)
        # Phase 4.2: parameters are frozen. update_params() only writes to
        # `.data` (no gradient tracking) and the optimizer must never touch
        # these weights. State-dict format is unchanged: still stores
        # `eval_conv.weight` and `eval_conv.bias` keys (same as before).
        self.eval_conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=s, bias=bias)
        self.eval_conv.weight.requires_grad_(False)
        if bias:
            self.eval_conv.bias.requires_grad_(False)

        # Fusion cache flag. False = need to re-fuse on next eval forward.
        # Not a Parameter/buffer, so it doesn't show up in state_dict.
        self._fused = False

    def update_params(self):
        """Fuse sk + conv into eval_conv (called automatically in eval mode)."""
        w1 = self.conv[0].weight.data.clone().detach()
        b1 = self.conv[0].bias.data.clone().detach()
        w2 = self.conv[1].weight.data.clone().detach()
        b2 = self.conv[1].bias.data.clone().detach()
        w3 = self.conv[2].weight.data.clone().detach()
        b3 = self.conv[2].bias.data.clone().detach()

        # Fuse conv[0] + conv[1]
        w = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        # Fuse result + conv[2]
        self.weight_concat = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        # Add sk (padded to 3x3)
        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.data.clone().detach()
        sk_w = F.pad(sk_w, [1, 1, 1, 1])

        self.eval_conv.weight.data = self.weight_concat + sk_w
        self.eval_conv.bias.data = self.bias_concat + sk_b

    def train(self, mode: bool = True):
        """Override to invalidate the eval fusion cache on mode switch."""
        # super().train() will set self.training = mode
        result = super().train(mode)
        # Whenever we leave eval (mode=True is train), drop the cache so the
        # next eval forward re-fuses with current weights.
        if mode:
            self._fused = False
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            # Fuse at most once per eval session.
            if not self._fused:
                self.update_params()
                self._fused = True
            out = self.eval_conv(x)

        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out


class SPAB(nn.Module):
    """
    SPAN Attention Block from neosr.
    
    Structure:
    x -> Conv3XC(c1_r) -> SiLU -> Conv3XC(c2_r) -> SiLU -> Conv3XC(c3_r)
    sim_att = sigmoid(out3) - 0.5
    out = (out3 + x) * sim_att
    
    Returns: (out, out1, sim_att)
    - out: final output
    - out1: output of c1_r (used for concatenation in main model)
    - sim_att: attention map
    """
    
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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)
        
        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)
        
        out3 = self.c3_r(out2_act)
        
        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att
        
        return out, out1, sim_att


class NeosrSPAN(nn.Module):
    """
    SPAN model matching neosr exactly.
    
    Forward flow:
    1. conv_1(x) -> out_feature
    2. block_1(out_feature) -> out_b1, _, _
    3. block_2(out_b1) -> out_b2
    4. block_3(out_b2) -> out_b3
    5. block_4(out_b3) -> out_b4
    6. block_5(out_b4) -> out_b5
    7. block_6(out_b5) -> out_b6, out_b5_2, _
    8. conv_2(out_b6)
    9. conv_cat([out_feature, out_b6, out_b1, out_b5_2])
    10. upsampler(out)
    """
    
    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        feature_channels: int = 48,
        upscale: int = 4,
        bias: bool = True,
        norm: bool = False,
        img_range: float = 1.0,
        rgb_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        sab_type: str = 'conv3xc',
        mamba_d_state: int = 16,
        mamba_num_tokens: int = 64,
        mamba_inner_rank: int = 32,
        mamba_mlp_ratio: float = 2.0,
    ):
        super().__init__()
        
        self.upscale = upscale
        self.img_range = img_range
        self.mean = nn.Parameter(torch.Tensor(rgb_mean).view(1, 3, 1, 1), requires_grad=False)
        
        if not norm:
            self.register_buffer("no_norm", torch.zeros(1))
        else:
            self.no_norm = None
        
        # Shallow feature extraction
        self.conv_1 = Conv3XC(num_in_ch, feature_channels, gain1=2, s=1)
        
        # Factory: pick attention block class. Default `conv3xc` keeps v6
        # warm-start checkpoints loadable; opt-in `mamba_v2` swaps in
        # MambaIRv2 ASSM blocks (v7 opt-in).
        if sab_type == 'mamba_v2':
            SAB_CLS = MambaSPAB
            sab_kwargs = {
                'd_state': mamba_d_state,
                'num_tokens': mamba_num_tokens,
                'inner_rank': mamba_inner_rank,
                'mlp_ratio': mamba_mlp_ratio,
            }
        else:
            SAB_CLS = SPAB
            sab_kwargs = {}
        
        # 6 attention blocks (named individually, not ModuleList)
        self.block_1 = SAB_CLS(feature_channels, bias=bias, **sab_kwargs)
        self.block_2 = SAB_CLS(feature_channels, bias=bias, **sab_kwargs)
        self.block_3 = SAB_CLS(feature_channels, bias=bias, **sab_kwargs)
        self.block_4 = SAB_CLS(feature_channels, bias=bias, **sab_kwargs)
        self.block_5 = SAB_CLS(feature_channels, bias=bias, **sab_kwargs)
        self.block_6 = SAB_CLS(feature_channels, bias=bias, **sab_kwargs)
        
        # After blocks
        self.conv_cat = conv_layer(feature_channels * 4, feature_channels, kernel_size=1, bias=True)
        self.conv_2 = Conv3XC(feature_channels, feature_channels, gain1=2, s=1)
        
        # Upsampler
        self.upsampler = nn.Sequential(
            conv_layer(feature_channels, num_out_ch * (upscale ** 2), kernel_size=3),
            nn.PixelShuffle(upscale),
        )

        self._use_checkpointing = False

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for SPAB blocks to save VRAM."""
        self._use_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self._use_checkpointing = False

    @property
    def use_checkpointing(self):
        return self._use_checkpointing
    
    @property
    def is_norm(self):
        return self.no_norm is None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_norm:
            self.mean = self.mean.type_as(x)
            x = (x - self.mean) * self.img_range

        out_feature = self.conv_1(x)

        if self._use_checkpointing and self.training:
            out_b1, _, att1 = checkpoint(self.block_1, out_feature, use_reentrant=False)
            out_b2, _, att2 = checkpoint(self.block_2, out_b1, use_reentrant=False)
            out_b3, _, att3 = checkpoint(self.block_3, out_b2, use_reentrant=False)
            out_b4, _, att4 = checkpoint(self.block_4, out_b3, use_reentrant=False)
            out_b5, _, att5 = checkpoint(self.block_5, out_b4, use_reentrant=False)
            out_b6, out_b5_2, att6 = checkpoint(self.block_6, out_b5, use_reentrant=False)
        else:
            out_b1, _, att1 = self.block_1(out_feature)
            out_b2, _, att2 = self.block_2(out_b1)
            out_b3, _, att3 = self.block_3(out_b2)
            out_b4, _, att4 = self.block_4(out_b3)
            out_b5, _, att5 = self.block_5(out_b4)
            out_b6, out_b5_2, att6 = self.block_6(out_b5)

        out_b6 = self.conv_2(out_b6)
        out = self.conv_cat(torch.cat([out_feature, out_b6, out_b1, out_b5_2], 1))
        output = self.upsampler(out)

        return output
    
    def load_neosr_weights(self, checkpoint_path: str, strict: bool = False) -> Dict:
        """Load weights from neosr/Phhofm checkpoint format."""
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except Exception:
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if isinstance(ckpt, dict):
            for key in ['params', 'params_ema', 'state_dict', 'model_state_dict']:
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
            'skipped_keys': skipped[:5] if skipped else [],
            'mismatched_keys': mismatched[:3] if mismatched else [],
        }


def create_neosr_span(config: Dict) -> NeosrSPAN:
    """Create neosr-compatible SPAN from config"""
    model_config = config.get('model', config)
    mamba_cfg = model_config.get('mamba', {}) or {}
    return NeosrSPAN(
        num_in_ch=model_config.get('num_in_ch', 3),
        num_out_ch=model_config.get('num_out_ch', 3),
        feature_channels=model_config.get('feature_channels', 48),
        upscale=model_config.get('upscale', 4),
        bias=model_config.get('bias', True),
        norm=model_config.get('norm', False),
        sab_type=model_config.get('sab_type', 'conv3xc'),
        mamba_d_state=mamba_cfg.get('d_state', 16),
        mamba_num_tokens=mamba_cfg.get('num_tokens', 64),
        mamba_inner_rank=mamba_cfg.get('inner_rank', 32),
        mamba_mlp_ratio=mamba_cfg.get('mlp_ratio', 2.0),
    )
