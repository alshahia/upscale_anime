import av
c = av.Codec('h264', 'r').create('h264_cuvid', mode='cuda')
print("hwaccel cuda ok:", c)
c.close()
