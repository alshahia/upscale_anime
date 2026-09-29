import av, numpy as np, imageio
from pathlib import Path
V = Path(r"E:/python projects/upscale_anime/results/style_video")
def grab(mp4, idx=90):
    c = av.open(str(mp4)); s = c.streams.video[0]
    for i, f in enumerate(c.decode(s)):
        if i == idx:
            a = f.to_ndarray(format="rgb24"); c.close(); return a
    c.close()
a = [grab(V/"upscaled_base_a050.mp4"), grab(V/"upscaled_datastyle_a050.mp4")]
def crop(x): return x[400:1000, 300:1200]
img = np.concatenate([crop(a[0]), crop(a[1])], axis=1)
imageio.imwrite(str(V/"video_style_compare.png"), img)
print("ok")
