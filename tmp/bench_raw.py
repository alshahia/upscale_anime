import torch, time
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
a = torch.randn(8192, 8192, device="cuda", dtype=torch.half)
b = torch.randn(8192, 8192, device="cuda", dtype=torch.half)
for _ in range(3): c = a @ b
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(10): c = a @ b
torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / 10
print("fp16 matmul 8192^3:", round(dt * 1000, 2), "ms =", round(2 * 8192**3 / dt / 1e12, 2), "TFLOPS")
x = torch.randn(1, 64, 540, 960, device="cuda", dtype=torch.half)
conv = torch.nn.Conv2d(64, 64, 3, padding=1).cuda().half()
for _ in range(5): y = conv(x)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(20): y = conv(x)
torch.cuda.synchronize(); df = (time.perf_counter() - t0) / 20
flops = 2 * 64*64*9 * 540*960
print("fp16 conv 64ch 3x3 540x960:", round(df * 1000, 2), "ms =", round(flops / df / 1e12, 2), "TFLOPS")
import subprocess
r = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu", "--format=csv"], capture_output=True, text=True)
print(r.stdout)