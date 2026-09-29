import numpy as np, imageio.v2 as imageio, av
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
grid = imageio.imread(str(ROOT / "results" / "style_image" / "style_dial.png"))
imageio.imwrite(str(ROOT / "tmp" / "style_dial_small.png"), grid[::2, ::2][:2880, :2880])
print("small ok")
