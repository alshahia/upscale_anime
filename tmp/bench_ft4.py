import json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import imageio.v2 as imageio
import av
ROOT = Path(r"E:\python projects\upscale_anime"); sys.path.insert(0, str(ROOT/"apps"/"anime_upscaler_gui"))
import sys as _s; _s.path.insert(0, str(ROOT/"apps"/"anime_upscaler_gui"))
torch.backends.cudnn.benchmark = True
from anime_upscaler_gui.archs import build
target = ROOT/"results"/"bench_ft4"; target.mkdir(parents=True, exist_ok=True)
CKPT = ROOT/"results/style_ft4/styled_student.pth"
model = build("srvgg", str(CKPT)).cuda().eval().half()

# ---- IQA metrics (CLIP-IQA+ / NIQE via pyiqa if available) ----
try:
    import pyiqa
    have_iqa = True
    iqi = pyiqa.create_metric("clipiqa+"); niq = pyiqa.create_metric("niqe")
except Exception as e:
    print("pyiqa unavailable:", e); have_iqa = False

rows = {}
def record(name, klips, kfps, kvram, clipiqa=None, niqe=None):
    rows[name] = {"ms_per_frame": round(klips,1), "fps": round(kfps,1), "peak_vram_GB": round(kvram,2)}
    if clipiqa is not None: rows[name]["clipiqa+"] = round(float(clipiqa),4)
    if niqe is not None: rows[name]["niqe"] = round(float(niqe),4)

# ---------- Part A: image benchmark (the val_hr frame, degraded 4x) ----------
src = ROOT/"data"/"val_hr"/"mp4upload - Easy Way to Backup and Share your Videos_frame026310_q0.74.png"
a = np.asarray(imageio.imread(str(src)), np.float32)/255.0
im = torch.from_numpy(a[...,:3]).permute(2,0,1)[None]
lr = F.interpolate(im, scale_factor=0.25, mode="bicubic", antialias=True)
lr = F.avg_pool2d(lr,3,1,1); lr = (lr + 0.01*torch.randn_like(lr)).clamp(0,1)
lrD = lr.cuda().half()
for _ in range(5): model(lrD)  # warm-up
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(30): y = model(lrD)
torch.cuda.synchronize(); dt = (time.perf_counter()-t0)/30*1000
vram = torch.cuda.max_memory_allocated()/1e9; torch.cuda.reset_peak_memory_stats()
y32 = y.clamp(0,1)[0].permute(1,2,0).float().detach().cpu().numpy()
p1 = target/"bench_image_ft4.png"; imageio.imwrite(str(p1), (y32*255).astype(np.uint8))
c1 = n1 = None
if have_iqa:
    import imageio.v2 as iio2
    c1 = iqi(str(p1)); n1 = niq(str(p1))
record("image_ft4baked", dt, 1000/dt, vram, c1, n1)
print("[img] %.1f ms/frame, %.1f fps, vram %.2f GB" % (dt, 1000/dt, vram), flush=True)

# ---------- Part B: 10 s video benchmark (exact 20:45:00-20:55:00 clip) ----------
clip = ROOT/"results"/"style_video"/"src_clip_204500_205500.mp4"
mp4 = target/"bench_video_ft4.mp4"
cc = av.open(str(clip)); cs = cc.streams.video[0]
fps = float(cs.average_rate)
torch.cuda.reset_peak_memory_stats()
wf = None; writer = None; n_fg = 0; lat = []; t_start = time.perf_counter()
for frame in cc.decode(cs):
    img = frame.to_ndarray(format="rgb24")
    x = torch.from_numpy(img).float().div(255).permute(2,0,1)[None].cuda().half()
    if x.shape[1] == 3: pass
    with torch.no_grad():
        y = model(x)
    out = (y.clamp(0,1)[0].permute(1,2,0).float().detach().cpu().numpy()*255).astype(np.uint8)
    if writer is None:
        writer = imageio.get_writer(str(mp4), fps=round(fps), macro_block_size=8)
    writer.append_data(out); n_fg += 1
writer.close(); cc.close()

total = time.perf_counter()-t_start
ms = total/n_fg*1000
print("[vid] %d frames, %.1f ms/frame (%.1f fps), wall %.1f s" % (n_fg, ms, n_fg/total, total), flush=True)
c2 = n2 = None
if have_iqa:
    # IQA on 6 evenly spaced decoded frames
    from PIL import Image
    rr = av.open(str(mp4)); samples=[]; i=0
    for fr in rr.decode(rr.streams.video[0]):
        if i % 30 == 0:
            samples.append(Image.fromarray(fr.to_ndarray(format="rgb24")))
        i+=1
        if len(samples)>=6: break
    rr.close()
    import tempfile, os
    scores_c=[]; scores_n=[]
    for s_ in samples:
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False); s_.save(f.name); f.close()
        scores_c.append(float(iqi(f.name))); scores_n.append(float(niq(f.name))); os.unlink(f.name)
    c2 = np.mean(scores_c); n2 = np.mean(scores_n)
record("video_10s_ft4baked", ms, n_fg/total, torch.cuda.max_memory_allocated()/1e9, c2, n2)

# comparison rows from earlier benches (realesr-animevideov3 & style_mix alpha0.5)
rows["reference_base_animevideov3"] = {"ms_per_frame": 43.0, "fps": 23.6, "peak_vram_GB": 0.4}
rows["reference_stylemix_a050"]    = {"ms_per_frame": 42.4, "fps": 23.6, "peak_vram_GB": 0.4}
(ROOT/"results"/"bench_ft4"/"bench.json").write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
