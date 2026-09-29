import numpy as np, imageio.v2 as imageio
from pathlib import Path
chunks = []
for p in sorted(Path(r"E:/python projects/upscale_anime/data/style").glob("*")):
    if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
        a = np.asarray(imageio.imread(str(p)), np.float32)[..., :3]
        chunks.append(a.reshape(-1, 3))
ref = np.concatenate(chunks)
sel = ref[np.linspace(0, len(ref) - 1, 512 * 512).astype(np.int64)].reshape(512, 512, 3)
imageio.imwrite(r"E:/python projects/upscale_anime/results/style_video/ref_data_style.png", np.clip(sel, 0, 255).astype(np.uint8))
print("ref written")
