import numpy as np, imageio
H, W = 256, 256
yy, xx = np.mgrid[0:H, 0:W]
v = (xx + yy) / (H + W)
v3 = np.stack([v, v, v], axis=-1)
warm = np.stack([0.95 - 0.3*v, 0.35 + 0.2*np.sin(v*3), 0.15 + 0.1*np.cos(v)], axis=-1)
warm = (np.clip(((warm - warm.mean()) / (warm.std()+1e-6) * 0.18 + 0.55), 0, 1)*255).astype(np.uint8)
imageio.imwrite(r"E:/python projects/upscale_anime/results/style_video/ref_cinema_warm.png", warm)
past = np.clip(0.6 + 0.12*np.sin(v3*2.0) + 0.05*np.cos(v3*4.0), 0, 1)
past = (past*255).astype(np.uint8)
imageio.imwrite(r"E:/python projects/upscale_anime/results/style_video/ref_pastel_soft.png", past)
print("ok")
