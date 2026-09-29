import sys, warnings, torch
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "apps" / "anime_upscaler_gui"))
import anime_upscaler_gui.archs as A
m = A._load_animesr(str(ROOT / "pretrained" / "AnimeSR_v1-PaperModel.pth")).half().eval()
x = torch.zeros(1, 1, 9, 48, 85, dtype=torch.half)
with torch.inference_mode():
    b, n, c, h, w = x.size()
    out = x.new_zeros(b, c, h * m.netscale, w * m.netscale)
    state = x.new_zeros(b, m.num_feat, h, w)
    i = 0
    prev = x[:, 0]; nxt = x[:, 0]; cur = x[:, 0]
    inp = torch.cat((prev, cur, nxt), dim=1)
    print("inp", tuple(inp.shape))
    print("out(new_zeros)", tuple(out.shape))
    print("p_u(out)", tuple(A._pixel_unshuffle(out, m.netscale).shape))
    print("state", tuple(state.shape))
    print("cell inp", tuple(torch.cat((inp, A._pixel_unshuffle(out, m.netscale), state), dim=1).shape))
    conv = m.recurrent_cell.conv_s1_first[0]
    print("conv", tuple(conv.weight.shape), "inp.size(1) ==", inp.size(1))
    # directly try run one cell
    def cell(x, fb, state):
        res = x[:, 3:6]
        inp2 = torch.cat((x, A._pixel_unshuffle(fb, m.netscale), state), dim=1)
        print("cell inp2:", tuple(inp2.shape))
        return m.recurrent_cell(inp2)
    with torch.inference_mode():
        o = cell(inp, out, state);
