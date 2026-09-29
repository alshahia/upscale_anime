import sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "apps" / "anime_upscaler_gui"))
import numpy as np
import torch
import onnxruntime as ort
from PIL import Image
from anime_upscaler_gui.archs import build
import onnx

torch.cuda.set_per_process_memory_fraction(0.9)
torch.backends.cudnn.benchmark = True
DEV = torch.device("cuda")
CKPT = ROOT / "pretrained" / "SRVGG_distill_v1_4x_student.pth"

def fix_io_binding(sess, X, Y, odtype):
    pass

src = sorted((ROOT / "data" / "anime_fullframes").glob("*.jpg"))[7]
im = Image.open(src).convert("RGB")
lr540 = np.asarray(im.resize((960, 540), Image.BICUBIC), np.float32).transpose(2, 0, 1)[None] / 255.0
lr360 = np.asarray(im.resize((640, 360), Image.BICUBIC), np.float32).transpose(2, 0, 1)[None] / 255.0

# static session (960x540)
net16 = build("srvgg_student", str(CKPT)).to(DEV).half().eval()
sess540 = ort.InferenceSession(str(ROOT / "tmp" / "probe_fp16.onnx"),
                               providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

def make_dyn(path):
    torch.onnx.export(net16,
                      torch.zeros(1, 3, 16, 16, device=DEV, dtype=torch.half),
                      str(path), opset_version=17,
                      input_names=["lr"], output_names=["hr"],
                      dynamic_axes={"lr": {2: "H", 3: "W"}}, dynamo=False)
    return ort.InferenceSession(str(path),
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

sess360 = make_dyn(ROOT / "tmp" / "probe_fp16_dyn.onnx") if not (ROOT / "tmp" / "probe_fp16_dyn.onnx").exists() else ort.InferenceSession(str(ROOT / "tmp" / "probe_fp16_dyn.onnx"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

def bench(sess, X, tag, n=30, binding=False):
    if not binding:
        for _ in range(5):
            sess.run(["hr"], {"lr": X})
        t0 = time.perf_counter()
        for _ in range(n):
            sess.run(["hr"], {"lr": X})
        dt = (time.perf_counter() - t0) / n
        print(tag, round(dt * 1000, 1), "ms =", round(1 / dt, 1), "fps")
        return
    xd = torch.from_numpy(X).cuda()
    yd = torch.empty(1, 3, X.shape[2] * 4, X.shape[3] * 4, device="cuda", dtype=torch.half)
    io = sess.io_binding()
    io.bind_input("lr", "cuda", 0, np.float16, xd.shape, xd.data_ptr())
    io.bind_output("hr", "cuda", 0, np.float16, yd.shape, yd.data_ptr())
    for _ in range(5):
        sess.run_with_iobinding(io)
    t0 = time.perf_counter()
    for _ in range(n):
        sess.run_with_iobinding(io)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    print(tag, round(dt * 1000, 1), "ms =", round(1 / dt, 1), "fps")

X540 = np.ascontiguousarray(lr540).astype(np.float16)
X360 = np.ascontiguousarray(lr360).astype(np.float16)
bench(sess540, X540, "ORT fp16 540x960 hostIO  -> 4K:")
bench(sess540, X540, "ORT fp16 540x960 cudaIO  -> 4K:", binding=True)
bench(sess360, X360, "ORT fp16 360x640 cudaIO  -> 1440p:", binding=True)
bench(sess360, X360, "ORT fp16 360x640 hostIO  -> 1440p:")

# eager channels_last
netcl = net16.to(memory_format=torch.channels_last)
for (h, w, Xh) in ((540, 960, X540), (360, 640, X360)):
    x = torch.from_numpy(Xh).cuda().to(memory_format=torch.channels_last)
    with torch.inference_mode():
        for _ in range(3):
            y = netcl(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            y = netcl(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 10
    print(f"eager fp16 CL {h}x{w}:", round(dt * 1000, 1), "ms =", round(1 / dt, 1), "fps")
