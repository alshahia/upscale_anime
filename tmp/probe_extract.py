import av, imageio.v2 as imageio, sys
from pathlib import Path
out = Path(r"E:\python projects\upscale_anime\results\style_video"); out.mkdir(exist_ok=True)
src = r"C:/Users/Ahmad Mahmoud/Downloads/Video/[Anime3rb.com] Yi Ren Zhi Xia - 1 [480p].mp4"
clip = out / "src_probe_test.mp4"
c = av.open(src); s = c.streams.video[0]
fps = float(s.average_rate or 25)
t0 = 1245.0
c.seek(int(t0 / s.time_base), stream=s)
n = int(round((1245+10 - 1245) * fps)) if False else int(round(10 * fps))
got = 0
with imageio.get_writer(str(clip), fps=round(fps), macro_block_size=8) as vw:
    for frame in c.decode(s):
        if frame.time is not None and frame.time < t0 - 1e-4: continue
        vw.append_data(frame.to_ndarray(format="rgb24"))
        got += 1
        if got >= n: break
c.close()
print("got", got, "exists", clip.exists(), clip.stat().st_size if clip.exists() else 0)
