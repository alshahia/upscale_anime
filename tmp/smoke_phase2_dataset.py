import sys
sys.path.insert(0, "anime_upscaler")
from dataset import AnimePairDataset

ds4 = AnimePairDataset("data/anime_video_frames", "train", scale=4, max_files=8)
lr, hr = ds4[0]; print(f"scale=4: lr={tuple(lr.shape)}, hr={tuple(hr.shape)}  expected lr=(3,48,48), hr=(3,192,192)")

ds2 = AnimePairDataset("data/anime_video_frames", "train", scale=2, max_files=8)
lr, hr = ds2[0]; print(f"scale=2: lr={tuple(lr.shape)}, hr={tuple(hr.shape)}  expected lr=(3,96,96), hr=(3,192,192)")

# val (deterministic center crop, larger)
vds4 = AnimePairDataset("data/anime_video_frames", "val", scale=4, max_files=2)
lr, hr = vds4[0]; print(f"val scale=4: lr={tuple(lr.shape)}, hr={tuple(hr.shape)}  expected lr=(3,96,96), hr=(3,384,384)")

vds2 = AnimePairDataset("data/anime_video_frames", "val", scale=2, max_files=2)
lr, hr = vds2[0]; print(f"val scale=2: lr={tuple(lr.shape)}, hr={tuple(hr.shape)}  expected lr=(3,192,192), hr=(3,384,384)")

print("DATASET SMOKE PASS")
