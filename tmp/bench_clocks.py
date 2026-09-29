import torch, time, subprocess, threading
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
a = torch.randn(8192, 8192, device="cuda", dtype=torch.half)
b = torch.randn(8192, 8192, device="cuda", dtype=torch.half)
stop = False
def poll():
    for _ in range(6):
        r = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,utilization.gpu",
                            "--format=csv,noheader"], capture_output=True, text=True)
        print("during-load:", r.stdout, flush=True)
        if stop:
            break
        time.sleep(0.7)
for _ in range(5):
    c = a @ b
torch.cuda.synchronize()
thr = threading.Thread(target=poll)
thr.start()
t0 = time.perf_counter()
for _ in range(30):
    c = a @ b
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / 30
stop = True
thr.join()
print("matmul:", round(dt * 1000, 2), "ms =", round(2 * 8192**3 / dt / 1e12, 2), "TFLOPS")
