# scripts/step2_still.py — extract a bicubic|student detail pair from the pair video
import imageio.v3 as iio
import numpy as np

fr = iio.imread("results/step2_video_test_pair.mp4", index=12)
print("pair frame", fr.shape)
H, W = fr.shape[:2]
y0, h = H // 4, 800
x0 = W // 8
w = 900
strip = np.concatenate([
    fr[y0:y0 + h, x0:x0 + w],
    np.full((h, 20, 3), 255, np.uint8),
    fr[y0:y0 + h, W // 2 + x0:W // 2 + x0 + w],
], axis=1)
iio.imwrite("results/step2_video_test_still.png", strip)
print("still", strip.shape)
