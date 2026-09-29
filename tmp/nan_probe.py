import sys, warnings
warnings.filterwarnings("ignore")
import av, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"apps"/"anime_upscaler_gui"))
from anime_upscaler_gui.archs import build
model = build("srvgg", str(ROOT/"results/style_mix/weights/stylemix_a0.50.pth")).to("cuda").eval().half()
c = av.open(str(ROOT/"results/style_video/src_clip_2040_2050.mp4"))
s = c.streams.video[0]
frames = [f.to_ndarray(format="rgb24") for f in c.decode(s)]
nan_frames = []
with torch.inference_mode():
    for i in (0, 45, 90, 179):
        t = torch.from_numpy(frames[i]).permute(2,0,1)[None].to("cuda").half()
        y = model(t)
        if torch.isnan(y).any(): nan_frames.append(i)
        print(i, "ok" if not torch.isnan(y).any() else "NAN", tuple(y.shape), "min/max", y.min().item(), y.max().item())
print("nan_frames", nan_frames)
