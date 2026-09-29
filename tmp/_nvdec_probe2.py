import av
print("av.__version__:", av.__version__)
# List hwaccels
print("has h264_cuvid:", any(c.name == "h264_cuvid" for c in av.codecs_available))
# Try via container hwaccel_options
c = av.open("E:/python projects/upscale_anime/tmp/mid2s_job.py".replace("E:/python projects/upscale_anime/tmp/mid2s_job.py", ""))
print("av OK")
