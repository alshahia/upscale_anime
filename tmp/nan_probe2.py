import sys, warnings, os
warnings.filterwarnings("ignore")
import numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"apps"/"anime_upscaler_gui"))
from anime_upscaler_gui.archs import build

frame = None
import imageio.v2 as imageio
a = imageio.imread(str(ROOT/"results/bench_all/lr_frames/00045.png"))
t = torch.from_numpy(a).permute(2,0,1)[None].to("cuda").half()

for name, kind in [("realesr-animevideov3.pth", "srvgg"),
                   ("results/style_mix/weights/stylemix_a0.50.pth", "srvgg"),
                   ("results/style_mix/weights/stylemix_a0.50.pth", "srvgg_fp32_probe")]:
    p = name if name.startswith("pretrained") else str(ROOT/name)
    if not p.startswith(ROOT.as_posix()) and not os.path.isabs(p):
        p = str(ROOT/"pretrained"/name)
    model = build("srvgg", str(ROOT/(p if not os.path.isabs(p) else p))).to("cuda").eval()
    is_fp32 = kind.endswith("fp32_probe")
    if not is_fp32:
        model = model.half()
        tt = t
    else:
        tt = torch.from_numpy(a).permute(2,0,1)[None].to("cuda").float()
    with torch.inference_mode():
        y = model(tt)
    print(name, kind, "nan:", bool(torch.isnan(y).any()), "min", y.min().item(), "max", y.max().item())
