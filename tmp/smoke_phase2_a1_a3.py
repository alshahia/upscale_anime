import torch, sys
sys.path.insert(0, "anime_upscaler")
sys.path.insert(0, "apps")
sys.path.insert(0, "apps/anime_upscaler_gui")
from student import RFDN

# Build both scales
s4 = RFDN(scale=4); print("scale=4: params=", s4.num_params(), "upscale=", s4.upscale)
s2 = RFDN(scale=2); print("scale=2: params=", s2.num_params(), "upscale=", s2.upscale)

# Smoke forward
x = torch.randn(1, 3, 96, 96)
y4 = s4(x); print("scale=4 forward:", tuple(y4.shape), "expected (1,3,384,384)")
y2 = s2(x); print("scale=2 forward:", tuple(y2.shape), "expected (1,3,192,192)")

# Load existing v1 4x ckpt (default scale=4 path)
ck = torch.load("pretrained/RFDN_distill_v1_4x_student.pth", map_location="cpu", weights_only=False)
sd = ck["student"] if isinstance(ck, dict) and "student" in ck else ck
m = RFDN(scale=4)
missing, unexpected = m.load_state_dict(sd, strict=False)
print(f"missing keys: {len(missing)}, unexpected: {len(unexpected)}")
if missing:
    print("  e.g. missing:", missing[:3])
if unexpected:
    print("  e.g. unexpected:", unexpected[:3])
print("v1 4x ckpt loaded with default scale=4 -- OK")

# And verify build() in archs.py sniffs scale correctly
from anime_upscaler_gui.archs import build
m2 = build("rfdn_student", "pretrained/RFDN_distill_v1_4x_student.pth")
print(f"archs.build() -> scale={m2.scale}, upscale={m2.upscale}")
