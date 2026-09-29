import numpy as np, imageio.v2 as imageio
from pathlib import Path
SI = Path(r"E:/python projects/upscale_anime/results/style_image")
imgs = {}
for a in ["0.05", "0.20", "0.50"]:
    im = imageio.imread(str(SI / f"styled_a{a}.png"))[..., :3]
    imgs[a] = im
# face crop at 4x output: approx eyes/nose region -> alpha fine-detail differences
def crop(a): return a[150:550, 300:1000]
tiles = [crop(imgs[k]) for k in imgs]
import numpy as np
print("pixel diff 0.05 vs 0.50 (max abs):", int(np.abs(imgs["0.05"].astype(int) - imgs["0.50"].astype(int)).max()),
      "mean:", round(float(np.abs(imgs["0.05"].astype(int) - imgs["0.50"].astype(int)).mean()), 3))
# also diff 0.05 vs 0.50 raw model output: need to model that - use styled only for now
out = np.concatenate(tiles, axis=1)
imageio.imwrite(str(SI / "alpha_zoom_crop.png"), out)
print("alpha_zoom_crop ok", out.shape)
