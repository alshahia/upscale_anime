import av, numpy as np, sys, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
bench = ROOT / "results" / "style_mix" / "bench"
ids = ["stylemix_a0.00", "stylemix_a0.25", "stylemix_a0.50", "stylemix_a0.75", "stylemix_a1.00"]

def grab(mp4, idx=45):
    c = av.open(str(mp4)); s = c.streams.video[0]
    for i, f in enumerate(c.decode(s)):
        if i == idx:
            a = f.to_ndarray(format="rgb24"); c.close(); return a
    c.close()

# crop a detail region (center-left, face/line area) from 3416x1920
def crop(a):
    return a[1200:1716, 400:916]

tiles = [crop(grab(bench / "models" / m / "upscaled.mp4")) for m in ids]
lr = np.asarray(__import__("imageio").imread(str(bench / "lr_frames" / "00045.png")))
import imageio, torch.nn.functional as F
t = torch.from_numpy(lr).permute(2,0,1)[None].float()
up = F.interpolate(t, scale_factor=4, mode="bicubic", antialias=True)[0].permute(1,2,0).numpy().clip(0,255).astype(np.uint8)
tiles = [crop(up)] + tiles
gap = 8
h, w = tiles[0].shape[:2]
row = np.full((h*2 + gap, w*3 + gap*2, 3), 30, np.uint8)
for i, t in enumerate(tiles):
    r, cc = divmod(i, 3)
    row[r*(h+gap):r*(h+gap)+h, cc*(w+gap):cc*(w+gap)+w] = t
imageio.imwrite(str(ROOT / "results" / "style_mix" / "compare_strip.png"), row)
print("wrote compare_strip.png", row.shape)
