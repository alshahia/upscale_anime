# scripts/step2_video_test.py
"""Upscale a 1-second clip with the Step-2 student.

Grabs 1 second of a 1080p source episode, simulates a low-res stream
(half-res), runs the TinySRVGG student 4x, and writes:
  results/step2_video_test_student.mp4   (upscaled clip)
  results/step2_video_test_pair.mp4      (bicubic | student side-by-side)
and prints per-frame latency / fps.
"""
import sys
import time
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "anime_upscaler"))
from student import TinySRVGGStudent  # noqa: E402

device = "cuda"
video_src = ROOT / "data" / "anime_vid" / (
    "[Anime3rb.com] A-Rank Party wo Ridatsu shita Ore wa Moto Oshiego-tachi "
    "to Meikyuu Shinbu wo Mezasu. - 2 [1080p].mp4")
out_dir = ROOT / "results"
out_dir.mkdir(exist_ok=True)

ckpt = torch.load(str(ROOT / "runs/step2_srvgg_hfa_v1/student_best.pt"),
                  map_location="cpu", weights_only=False)
net = TinySRVGGStudent(scale=4)
net.load_state_dict(ckpt["student"], strict=True)
net.half()
net.eval().to(device)

container = av.open(str(video_src))
stream = container.streams.video[0]
fps = float(stream.average_rate or 25)
n = max(1, int(round(fps)))
container.seek(int(45 / stream.time_base), stream=stream)  # jump to a real scene at 45 s
frames = []
for frame in container.decode(stream):
    frames.append(frame.to_image().convert("RGB"))
    if len(frames) >= n:
        break
container.close()
print(f"captured {len(frames)} frames @ {fps:.1f} fps, src {stream.width}x{stream.height}")

lrs = []
for img in frames:
    a = np.asarray(img, np.float32) / 255.0
    t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)
    lrs.append(F.interpolate(t, scale_factor=0.5, mode="bicubic",
                             align_corners=False).clamp(0, 1))

H, W = frames[0].height, frames[0].width
times = []
outs, bics = [], []
with torch.inference_mode():
    for t_in in lrs:
        x = t_in.half()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        sr = net(x).float().clamp(0, 1)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        outs.append(sr)
        bics.append(F.interpolate(x.float(), scale_factor=4, mode="bicubic",
                                  align_corners=False).clamp(0, 1))
outs_t = torch.cat(outs, dim=0)
print("eager fp16: %.1f ms/frame -> %.1f fps (target 25 fps real-time)"
      % (1000 * np.mean(times), 1.0 / np.mean(times)))


def to_img(t01):
    a = (t01.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(a)


import cv2
import imageio
stu_path = out_dir / "step2_video_test_student.mp4"
pair_path = out_dir / "step2_video_test_pair.mp4"
for path, side_by_side in ((stu_path, False), (pair_path, True)):
    w = outs_t.shape[3] * (2 if side_by_side else 1)
    h = outs_t.shape[2]
    with imageio.get_writer(str(path), fps=fps, macro_block_size=8) as vw:
        for i in range(len(lrs)):
            pair = (torch.cat([bics[i], outs[i]], dim=2) if side_by_side else outs[i])
            a = pair[0].permute(1, 2, 0).cpu().numpy()
            vw.append_data((a * 255).round().astype(np.uint8))
print("wrote", stu_path.name, "and", pair_path.name, "(pair = bicubic | student)")
