import sys, imageio, av
from pathlib import Path
ROOT = Path(r"E:\python projects\upscale_anime")
sys.path.insert(0, str(ROOT / "apps" / "anime_upscaler_gui"))
out = ROOT/"results"/"style_video"; out.mkdir(exist_ok=True)
src = r"C:/Users/Ahmad Mahmoud/Downloads/Video/[Anime3rb.com] Yi Ren Zhi Xia - 1 [480p].mp4"
DEV = __import__("torch").device("cuda")
model = __import__("anime_upscaler_gui.archs", fromlist=["build"]).build("srvgg", str(ROOT/"results/style_ft4/styled_student.pth")).to(DEV).eval().half()
clip = out / "src_probe_withmodel.mp4"
def parse_hms(s):
    t=0.0
    for p in s.split(":"): t = t*60 + float(p)
    return t
t0, t1 = parse_hms("20:45:00"), parse_hms("20:55:00")
print("t0", t0, "t1", t1)
c = av.open(src); s = c.streams.video[0]
fps = float(s.average_rate or 25)
c.seek(int(t0 / s.time_base), stream=s)
n = int(round((t1 - t0) * fps))
got = 0
import numpy as np
with imageio.get_writer(str(clip), fps=round(fps), macro_block_size=8) as vw:
    for frame in c.decode(s):
        if frame.time is not None and frame.time < t0 - 1e-4: continue
        vw.append_data(frame.to_ndarray(format="rgb24"))
        got += 1
        if got >= n: break
c.close()
print("n", n, "got", got, "exists", clip.exists())
