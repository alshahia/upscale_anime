import sys, os, shutil
sys.path.insert(0, "apps/anime_upscaler_gui")
# Clear cache
cache = os.path.expandvars("$APPDATA/anime_upscaler_gui/cache/trt")
if os.path.isdir(cache): shutil.rmtree(cache)
os.makedirs(cache, exist_ok=True)

import torch
from anime_upscaler_gui.archs import build
from anime_upscaler_gui.pipeline import _make_backend, _RunJob

device = torch.device("cuda")
m4 = build("rfdn_student", "pretrained/RFDN_distill_v1_4x_student.pth").to(device).half().eval()
b, name = _make_backend(m4, "pretrained/RFDN_distill_v1_4x_student.pth", "rfdn_student", device, True, True, False, batch_size=1)
print("v1 4x backend:", name, "scale=", b.model.scale)
x = torch.zeros(1, 3, 480, 854, device=device, dtype=torch.float16)
with torch.no_grad():
    y = b(x)
print("v1 4x forward:", tuple(y.shape))
print("v1 4x backend INIT + forward PASS")

m2 = build("rfdn_student", "pretrained/RFDN_distill_v2_2x_student.pth").to(device).half().eval()
b2, name2 = _make_backend(m2, "pretrained/RFDN_distill_v2_2x_student.pth", "rfdn_student", device, True, True, False, batch_size=1)
print("v2 2x backend:", name2, "scale=", b2.model.scale)
with torch.no_grad():
    y2 = b2(x)
print("v2 2x forward:", tuple(y2.shape))
print("v2 2x backend INIT + forward PASS")
