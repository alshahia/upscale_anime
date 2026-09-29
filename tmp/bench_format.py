import sys, time
sys.path.insert(0, r"E:\python projects\upscale_anime\apps\anime_upscaler_gui")
sys.path.insert(0, r"E:\python projects\upscale_anime")
import torch
from anime_upscaler_gui.archs import build
torch.cuda.set_per_process_memory_fraction(0.9)
torch.backends.cudnn.benchmark = True
net = build('srvgg_student',
            r"E:\python projects\upscale_anime\pretrained\SRVGG_distill_v1_4x_student.pth").to("cuda").half().eval()
print("params", sum(p.numel() for p in net.parameters()))
x = torch.zeros(1, 3, 540, 960, device="cuda", dtype=torch.half)
for _ in range(3): y = net(x)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(10): y = net(x)
torch.cuda.synchronize(); d1 = (time.perf_counter() - t0) / 10 * 1000
print("NCHW", round(d1, 1), "ms", round(1000 / d1, 1), "fps")
net2 = net.to(memory_format=torch.channels_last)
x2 = x.to(memory_format=torch.channels_last)
for _ in range(3): y = net2(x2)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(10): y = net2(x2)
torch.cuda.synchronize(); d2 = (time.perf_counter() - t0) / 10 * 1000
print("NHWC", round(d2, 1), "ms", round(1000 / d2, 1), "fps")
xsmall = torch.zeros(1, 3, 360, 640, device="cuda", dtype=torch.half)
for _ in range(3): y = net(xsmall)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(10): y = net(xsmall)
torch.cuda.synchronize(); d3 = (time.perf_counter() - t0) / 10 * 1000
print("NCHW-640", round(d3, 1), "ms", round(1000 / d3, 1), "fps")