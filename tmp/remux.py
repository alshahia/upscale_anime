import av
from fractions import Fraction
jobs = [
 (r"E:/python projects/upscale_anime/results/style_video/src_clip_2040_2050.mp4", Fraction(18,1)),
 (r"E:/python projects/upscale_anime/results/style_video/upscaled_base_a050.mp4", Fraction(18,1)),
 (r"E:/python projects/upscale_anime/results/style_video/upscaled_cinema_a050.mp4", Fraction(18,1)),
 (r"E:/python projects/upscale_anime/results/style_video/upscaled_pastel_a050.mp4", Fraction(18,1)),
]
for src_path, rate in jobs:
    c = av.open(src_path); s = c.streams.video[0]
    frames = [f.reformat(width=s.codec_context.width, height=s.codec_context.height, format="yuv420p") for f in c.decode(s)]
    c.close()
    out_path = src_path.replace(".mp4", "_fixed.mp4")
    with av.open(out_path, "w") as oc:
        ov = oc.add_stream("libx264", rate=rate)
        ov.width = frames[0].width; ov.height = frames[0].height
        ov.pix_fmt = "yuv420p"
        ov.time_base = Fraction(1, rate.numerator)
        for i, f in enumerate(frames):
            f.pts = i
            f.pict_type = 0
            for pkt in ov.encode(f): oc.mux(pkt)
        for pkt in ov.encode(): oc.mux(pkt)
    print("fixed", out_path)
