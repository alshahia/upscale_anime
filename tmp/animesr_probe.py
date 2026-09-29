import sys, warnings, torch
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "apps" / "anime_upscaler_gui"))
from anime_upscaler_gui.archs import build, _load_animesr
m = _load_animesr(str(ROOT / "pretrained" / "AnimeSR_v1-PaperModel.pth"))
print("num_feat", getattr(m, "num_feat", None), "netscale", getattr(m, "netscale", None))
for name, p in list(m.named_parameters())[:3]:
    print(name, tuple(p.shape))
x = torch.zeros(1, 1, 9, 48, 85, dtype=torch.half)
m = m.half().eval()
try:
    with torch.inference_mode():
        y = m(x)
    print("ok", y.shape)
except Exception as e:
    import traceback; traceback.print_exc()
