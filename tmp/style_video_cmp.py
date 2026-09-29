import av, torch, numpy as np, imageio
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
V = ROOT / "results" / "style_video"

def grab(mp4, idx=90):
    c = av.open(str(mp4)); s = c.streams.video[0]
    for i, f in enumerate(c.decode(s)):
        if i == idx:
            a = f.to_ndarray(format="rgb24"); c.close(); return a
    c.close(); return None

ids = ["base_a050", "cinema_a050", "pastel_a050"]
def crop(a): return a[250:950, 700:1300]
tiles = [crop(grab(V / f"upscaled_{m}.mp4")) for m in ids]
gap = 8
h, w = tiles[0].shape[:2]
row = np.full((h, w*3 + gap*2, 3), 30, np.uint8)
for i, t in enumerate(tiles):
    row[:, i*(w+gap):i*(w+gap)+w] = t
imageio.imwrite(str(V / "style_ref_compare.png"), row)
print("ok", row.shape)
