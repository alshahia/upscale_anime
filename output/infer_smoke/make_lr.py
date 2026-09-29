import sys
from pathlib import Path
from PIL import Image
sys.path.insert(0, r"E:/python projects/upscale_anime/anime_upscaler")
from dataset import AnimePairDataset
root = Path(r"E:/python projects/upscale_anime")
ds = AnimePairDataset(str(root / "data" / "anime_video_frames"), "test")
out = root / "output" / "infer_smoke" / "lr"
out.mkdir(parents=True, exist_ok=True)
for f in ds.files[:3]:
    hr = Image.open(f).convert("RGB")
    lr = hr.resize((hr.width // 4, hr.height // 4), Image.BICUBIC)
    lr.save(out / f.name)
    print("LR made:", f.name, lr.size)
