import re
p = r"E:\python projects\upscale_anime\scripts\step_style_video.py"
src = open(p, encoding="utf-8").read()
src = src.replace("import av\n", "import av\nfrom fractions import Fraction\n", 1)
src = src.replace(
    'ov = oc.add_stream("libx264", rate=round(fps))',
    'ov = oc.add_stream("libx264", rate=round(fps))\n        ov.time_base = Fraction(1, round(fps))'
)
open(p, "w", encoding="utf-8").write(src)
print("patched")
