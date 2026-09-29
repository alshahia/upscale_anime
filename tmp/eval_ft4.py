import sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import imageio.v2 as imageio
from PIL import Image, ImageDraw
ROOT = Path(r"E:\python projects\upscale_anime"); sys.path.insert(0, str(ROOT/"apps"/"anime_upscaler_gui"))
from anime_upscaler_gui.archs import build
src = ROOT/"data"/"val_hr"/"mp4upload - Easy Way to Backup and Share your Videos_frame026310_q0.74.png"
a = np.asarray(imageio.imread(str(src)), np.float32)/255.0
im = torch.from_numpy(a[..., :3]).permute(2,0,1)[None]
lr = F.interpolate(im, scale_factor=0.25, mode="bicubic", antialias=True)
lr = F.avg_pool2d(lr,3,1,1); lr = (lr + 0.01*torch.randn_like(lr)).clamp(0,1)
ckpts = [("base_animev3", ROOT/"pretrained/realesr-animevideov3.pth"),
         ("ft4_domainfix", ROOT/"results/style_ft4/styled_student.pth")]
b = None; tiles = []
for name, ck in ckpts:
    c = torch.load(str(ck), map_location="cpu", weights_only=False); sd = c.get("params", c)
    m = build("srvgg", str(ck)); m.load_state_dict(sd, strict=True); m.eval().cuda()
    with torch.no_grad(): y = m(lr.cuda().float()).clamp(0,1)[0].permute(1,2,0).cpu().numpy()
    if b is None: b = y
    print(name, "mean|diff vs base| = %.1f, saturation %.3f, mean luminance %.3f" %
          (np.abs(y-b).mean()*255, (y.max(2)-y.min(2)).mean(), y.mean()))
    p = ROOT/"results"/"style_ft4"/f"eval_{name}.png"; p.parent.mkdir(exist_ok=True)
    imageio.imwrite(str(p), (y*255).astype(np.uint8))
    img = Image.fromarray((y*255).astype(np.uint8))
    bar = Image.new("RGB",(img.width,40),(20,20,20)); ImageDraw.Draw(bar).text((10,10), name, fill=(255,255,255))
    t = Image.new("RGB",(img.width,img.height+40)); t.paste(bar,(0,0)); t.paste(img,(0,40)); tiles.append(t)
gx, H = sum(t.width for t in tiles), max(t.height for t in tiles)
grid = Image.new("RGB",(gx,H),(0,0,0)); x=0
for t in tiles: grid.paste(t,(x,0)); x+=t.width
grid.save(str(ROOT/"results"/"style_ft4"/"ft4_grid.png"))
cx, cy, r = int(tiles[0].width*0.5), int(tiles[0].height*0.35), 260
crops=[t.crop((cx-r,cy-r+40,cx+r,cy+r+40)).resize((520,520), Image.LANCZOS) for t in tiles]
g2=Image.new("RGB",(520*len(crops),520),(0,0,0)); x=0
for c in crops: g2.paste(c,(x,0)); x+=520
g2.save(str(ROOT/"results"/"style_ft4"/"ft4_zoom.png"))
print("done")