import torch, sys
from pathlib import Path

print("REGISTRY:")
try:
    from apps.anime_upscaler_gui.anime_upscaler_gui.registry import PRESET_CATALOG
    for k, v in PRESET_CATALOG.items():
        desc = v.get("description", "")
        ckpt = v.get("ckpt_path", "")
        print(f"  {k}: {desc[:60]} | ckpt={ckpt}")
except Exception as e:
    print(f"  registry import err: {e}")

print()
print("PARAMS:")
files = [
    "pretrained/span_pix_pretrain_4x.pth",
    "pretrained/span_mssim_pretrain_4x.pth",
    "pretrained/realesr-animevideov3.pth",
    "pretrained/RFDN_distill_v1_4x_student.pth",
    "pretrained/RFDN_distill_v2_2x_student.pth",
    "pretrained/4xLSDIRCompactv2.pth",
    "pretrained/AnimeSR_v1-PaperModel.pth",
    "pretrained/AnimeSR_v2.pth",
    "pretrained/eranet_N12_pretrain_325k.pth",
]
for f in files:
    try:
        ckpt = torch.load(f, map_location="cpu", weights_only=False)
        keys = list(ckpt.keys())
        n_params = 0
        for key in ["params", "params_ema", "state_dict", "params_dict"]:
            if key in ckpt:
                n_params = sum(
                    v.numel() for v in ckpt[key].values() if isinstance(v, torch.Tensor)
                )
                break
        if n_params == 0 and keys and isinstance(ckpt[keys[0]], torch.Tensor):
            n_params = sum(
                v.numel() for v in ckpt.values() if isinstance(v, torch.Tensor)
            )
        print(f"{Path(f).name:35s}: {n_params/1e6:7.3f}M | keys: {keys[:3]}")
    except Exception as e:
        print(f"{Path(f).name}: ERR {type(e).__name__}: {str(e)[:80]}")
