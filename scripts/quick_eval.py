# anime_upscaler/_quick_eval_ckpt.py - throwaway interim checkpoint scorer
import sys

import torch
from torch.utils.data import DataLoader

from anime_sr.data.datasets.image import AnimePairDataset, denorm01
from anime_sr.training.distillation.distill import psnr01, ssim01
from anime_sr.models.students import RFDN

run_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/distill_v2_grl"
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 99
allow_pickle = "--allow-pickle" in sys.argv
dev = "cuda"
try:
    ck = torch.load(run_dir + "/student_best.pt", map_location=dev,
                    weights_only=True)
except Exception as e:
    if not allow_pickle:
        raise RuntimeError("Checkpoint requires pickle loading. Use --allow-pickle to allow. Only use with trusted checkpoints!")
    import warnings
    warnings.warn("Loading checkpoint with pickle fallback - only use with trusted sources!", UserWarning, stacklevel=2)
    ck = torch.load(run_dir + "/student_best.pt", map_location=dev,
                    weights_only=False)
print("checkpoint epoch:", ck["epoch"], "val_psnr:", round(ck["val_psnr"], 2))
net = RFDN().to(dev).eval()
net.load_state_dict(ck["student"])
ds = AnimePairDataset("data/anime_video_frames", split="test")
if limit < 99:
    ds.files = ds.files[:limit]
dl = DataLoader(ds, batch_size=8, num_workers=0)
ps, ss, pb = [], [], []
with torch.no_grad():
    for lr_n, hr_n in dl:
        lr, hr = denorm01(lr_n).to(dev), denorm01(hr_n).to(dev)
        bic = torch.nn.functional.interpolate(lr, scale_factor=4,
                                              mode="bicubic",
                                              align_corners=False)
        sr = net(lr).clamp(0, 1)
        ps.append(psnr01(sr, hr)); ss.append(ssim01(sr, hr))
        pb.append(psnr01(bic.clamp(0, 1), hr))
import numpy as np
print("interim TEST split (%d imgs): student %.2f dB / SSIM %.4f | bicubic %.2f dB" % (len(ds), float(np.mean(ps)), float(np.mean(ss)), float(np.mean(pb))))
