# scripts/step2_visual_pair.py — direct bicubic vs student comparison from tensors
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
from anime_sr.models.students import TinySRVGGStudent

device = "cuda"
video_src = ROOT / "data" / "anime_vid" / (
    "[Anime3rb.com] A-Rank Party wo Ridatsu shita Ore wa Moto Oshiego-tachi "
    "to Meikyuu Shinbu wo Mezasu. - 2 [1080p].mp4")
ckpt = torch.load(str(ROOT / "runs/step2_srvgg_hfa_v1/student_best.pt"),
                  map_location="cpu", weights_only=False)
net = TinySRVGGStudent(scale=4)
net.load_state_dict(ckpt["student"], strict=True)
net.half()
net.eval().to(device)

import av
cont = av.open(str(video_src))
st = cont.streams.video[0]
cont.seek(int(45 / st.time_base), stream=st)
frame = next(iter(cont.decode(st))).to_image().convert("RGB")
cont.close()
img = np.asarray(frame, np.float32) / 255.0
H, W = img.shape[:2]
# center crop 960x540 as the LR (stream-size), 4x -> 3840x2160 comparable window
crop = img[H // 2 - H // 4: H // 2 + H // 4, W // 2 - W // 4: W // 2 + W // 4]
lr = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(device)
with torch.inference_mode():
    stu = net(lr.half()).float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
bic = F.interpolate(lr, scale_factor=4, mode="bicubic",
                    align_corners=False).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
y0, x0, h, w = 900, 1600, 900, 900
sep = np.full((h, 24, 3), 255, np.uint8)
strip = np.concatenate([bic[y0:y0 + h, x0:x0 + w], sep,
                        stu[y0:y0 + h, x0:x0 + w].squeeze() if stu.ndim == 4 else stu[y0:y0 + h, x0:x0 + w]], axis=1)
Image.fromarray((strip * 255).round().astype(np.uint8)).save(
    str(ROOT / "results" / "step2_video_pair_still.png"))
print("saved results/step2_video_pair_still.png (left bicubic | right student)")
