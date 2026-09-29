import sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "apps" / "anime_upscaler_gui"))
import numpy as np
import torch
import imageio
import torch.nn.functional as F
from anime_upscaler_gui.archs import build

torch.cuda.set_per_process_memory_fraction(0.9)
torch.backends.cudnn.benchmark = True
DEV = torch.device("cuda")

# Timing loop mimicking per-frame capture: model at 854x480 -> 4x
model = build("srvgg_student", str(ROOT / "pretrained" / "SRVGG_distill_v1_4x_student.pth")).to(DEV).eval().half()
x = torch.zeros(1, 3, 480, 854, device=DEV, dtype=torch.half)
with torch.inference_mode():
    for i in range(3):
        y = model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(10):
        y = model(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 10
print("student 480p->3416x1920 fp16:", round(dt*1000, 1), "ms =", round(1/dt, 1), "fps")
print("peak GB:", round(torch.cuda.max_memory_allocated() / 1024**3, 2))
