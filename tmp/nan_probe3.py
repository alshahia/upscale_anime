import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"apps"/"anime_upscaler_gui"))
from anime_upscaler_gui.archs import build
import imageio.v2 as imageio
a = imageio.imread(str(ROOT/"results/bench_all/lr_frames/00045.png"))
for name, fp32 in [(r"pretrained/realesr-animevideov3.pth", False),
                   (r"results/style_mix/weights/stylemix_a0.50.pth", False),
                   (r"results/style_mix/weights/stylemix_a0.50.pth", True),
                   (r"results/style_mix/weights/stylemix_a1.00.pth", False)]:
    model = build("srvgg", str(ROOT/name)).to("cuda").eval()
    t = torch.from_numpy(a).permute(2,0,1)[None].to("cuda")
    if not fp32:
        model = model.half(); t = t.half()
    else:
        t = t.float()
    with torch.inference_mode():
        y = model(t)
    print(Path(name).name, "fp32" if fp32 else "fp16", "nan:", bool(torch.isnan(y).any()), tuple(y.shape))
