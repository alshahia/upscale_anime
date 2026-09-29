import sys
from pathlib import Path
import numpy as np, imageio.v2 as imageio
ROOT = Path(r"E:\python projects\upscale_anime")
def load_style_stats(style_dir):
    chunks = []
    for p in sorted(Path(style_dir).glob("*")):
        if p.suffix.lower() in (".png",".jpg",".jpeg",".webp"):
            a = np.asarray(imageio.imread(str(p)), np.float32)/255.0
            if a.ndim == 2: a = np.repeat(a[...,None],3,axis=2)
            chunks.append(a[...,:3].reshape(-1,3))
    ref = np.concatenate(chunks,axis=0); return ref.mean(0), ref.std(0)
def ct(img, m_r, s_r, s):
    m, sd = img.mean((0,1)), img.std((0,1))
    tm = m + s*(m_r-m); ts = sd*(1.0+s*(s_r/sd-1.0))
    return np.clip((img-m)/sd*ts+tm,0,1), tm, ts
src = ROOT/"data"/"val_hr"/"mp4upload - Easy Way to Backup and Share your Videos_frame026310_q0.74.png"
a = np.asarray(imageio.imread(str(src)), np.float32)/255.0
a = a[...,:3]
ys = [(ys := [1,2,3]) for _ in []] if False else None
m_r, s_r = load_style_stats(ROOT/"data"/"style")
print("ref mean", m_r, "ref std", s_r)
print("img mean", a.mean((0,1)), "std", a.std((0,1)))
for s in (1.0, 2.2, 4.0):
    y, tm, ts = ct(a, m_r, s_r, s)
    print("strength", s, "-> target mean", tm, "target std", ts, "mean|diff| = %.1f" % (np.abs(y-a).mean()*255))
