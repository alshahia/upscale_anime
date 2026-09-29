import av, imageio, numpy as np
c = av.open(r"E:/python projects/upscale_anime/results/style_video/src_clip_2040_2050.mp4")
s = c.streams.video[0]
frames = [f.to_ndarray(format="rgb24") for f in c.decode(s)]
c.close()
out = r"E:/python projects/upscale_anime/results/style_video/src_clip_2040_2050_re.mp4"
with imageio.get_writer(out, fps=18, macro_block_size=8) as w:
    for f in frames: w.append_data(f)
import os
os.replace(out, out.replace("_re.mp4", ".mp4"))
c = av.open(r"E:/python projects/upscale_anime/results/style_video/src_clip_2040_2050.mp4")
s = c.streams.video[0]
print("fps", float(s.average_rate), "frames", s.frames, "dur", round(float(s.duration*s.time_base), 2))
