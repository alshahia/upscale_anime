# anime_upscaler/infer.py
"""Upscale image(s) 4x with the distilled student (PyTorch checkpoint or ONNX).

Inputs are treated as LOW-RES images (the model's expected domain); outputs are
saved at 4x size. When --gt points at a folder of original frames whose
basenames match the inputs, PSNR/SSIM against ground truth are reported.

Usage:
    .venv/Scripts/python.exe anime_upscaler/infer.py --input lr.png
    .venv/Scripts/python.exe anime_upscaler/infer.py --input lr_folder/ \
        --out output/upscaled --gt data/anime_video_frames
    .venv/Scripts/python.exe anime_upscaler/infer.py --input lr.png \
        --onnx anime_upscaler/export/student_fp16.onnx   # tests the deploy artifact
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from distill import psnr01, ssim01
from student import build_student

ROOT = Path(__file__).resolve().parent.parent
EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def load_student(run_dir, device, arch=None, allow_pickle=False):
    try:
        state = torch.load(Path(run_dir) / "student_best.pt",
                          map_location="cpu", weights_only=True)
    except Exception as e:
        if not allow_pickle:
            raise RuntimeError("Checkpoint requires pickle loading. Use --allow-pickle to allow. Only use with trusted checkpoints!")
        import warnings
        warnings.warn("Loading checkpoint with pickle fallback - only use with trusted sources!", UserWarning, stacklevel=2)
        state = torch.load(Path(run_dir) / "student_best.pt",
                          map_location="cpu", weights_only=False)
    model = build_student(state, arch=arch)
    model.load_state_dict(state["student"])
    return model.eval().to(device)


def upscale_tiled(model, lr01, tile, overlap=16):
    """Full-frame forward, or overlapping tiles for inputs too large for VRAM."""
    _, _, h, w = lr01.shape
    if tile <= 0 or (h <= tile and w <= tile):
        return model(lr01)
    out = torch.zeros(1, 3, h * 4, w * 4, device=lr01.device)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            y0, y1 = max(0, y - overlap), min(h, y + tile + overlap)
            x0, x1 = max(0, x - overlap), min(w, x + tile + overlap)
            sr = model(lr01[:, :, y0:y1, x0:x1])
            # keep only the core region so tile seams stay out of the result
            cy0, ch = (y - y0) * 4, (min(y + tile, h) - y) * 4
            cx0, cw = (x - x0) * 4, (min(x + tile, w) - x) * 4
            out[:, :, y * 4:y * 4 + ch, x * 4:x * 4 + cw] = \
                sr[:, :, cy0:cy0 + ch, cx0:cx0 + cw]
    return out


def ort_session(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="LR image file or folder")
    ap.add_argument("--out", default=str(ROOT / "output" / "upscaled"))
    ap.add_argument("--run-dir", default=str(ROOT / "runs" / "distill_v1"))
    ap.add_argument("--onnx", default=None, help="use an exported ONNX instead of the ckpt")
    ap.add_argument("--gt", default=None, help="folder of HR frames for PSNR/SSIM scoring")
    ap.add_argument("--allow-pickle", action="store_true",
                    help="allow pickle checkpoint loading (only use with trusted checkpoints)")
    ap.add_argument("--tile", type=int, default=0, help="LR tile size, 0 = whole frame")
    ap.add_argument("--tta", action="store_true",
                    help="apply D4 TTA (8x square, 6x rectangle); averaged mean output")
    ap.add_argument("--arch", choices=["auto", "rfdn", "srvgg"], default="auto",
                    help="student architecture (default: auto-detect from checkpoint)")
    args = ap.parse_args()

    if args.tta:
        from tta import tta_forward

    in_path = Path(args.input)
    files = sorted(p for p in (in_path.iterdir() if in_path.is_dir() else [in_path])
                   if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"no images found under {in_path}")

    gt_dir = Path(args.gt) if args.gt else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.out).mkdir(parents=True, exist_ok=True)

    sess = None
    model = None
    if args.onnx:
        sess = ort_session(args.onnx)
        print(f"[mode] ONNX {Path(args.onnx).name} ({device})")
    else:
        model = load_student(args.run_dir, device,
                                 arch=None if args.arch == "auto" else args.arch, allow_pickle=args.allow_pickle)
        print(f"[mode] PyTorch checkpoint {Path(args.run_dir).name} ({device})")

    for i, f in enumerate(files):
        lr = Image.open(f).convert("RGB")
        arr = np.asarray(lr, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        if sess is not None:
            dtype = np.float16 if sess.get_inputs()[0].type == "tensor(float16)" else np.float32
            sr_np = sess.run(["sr"], {"lr": arr.astype(dtype)})[0].astype(np.float32)
            sr = torch.from_numpy(sr_np).to(device).clamp(0, 1)
        else:
            with torch.no_grad():
                if args.tta:
                    _call = lambda x: upscale_tiled(model, x, args.tile)
                    sr, n_aug, names = tta_forward(_call, torch.from_numpy(arr).to(device))
                    print(f"[tta] {n_aug}x ({','.join(names)})")
                    sr = sr.clamp(0, 1)
                else:
                    sr = upscale_tiled(model, torch.from_numpy(arr).to(device),
                                       args.tile).clamp(0, 1)

        ext = f.suffix.lower() if f.suffix.lower() != '.jpg' else '.png'
        out_path = Path(args.out) / f"{f.stem}_x4{ext}"
        sr_img = Image.fromarray(
            (sr[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8))
        sr_img.save(out_path)

        line = (f"[{i + 1}/{len(files)}] {f.name} {lr.width}x{lr.height} -> "
                f"{sr_img.width}x{sr_img.height}")
        if gt_dir is not None:
            gt_file = gt_dir / f.name
            if gt_file.exists():
                gta = np.asarray(Image.open(gt_file).convert("RGB"), dtype=np.float32)
                gta = gta.transpose(2, 0, 1)[None] / 255.0
                hr = torch.from_numpy(gta).to(device)
                if hr.shape[-2:] != sr.shape[-2:]:
                    hr = hr[..., :sr.shape[-2], :sr.shape[-1]]
                    sr_c = sr[..., :hr.shape[-2], :hr.shape[-1]]
                else:
                    sr_c = sr
                line += (f" | PSNR {psnr01(sr_c, hr):.2f} dB "
                         f"SSIM {ssim01(sr_c, hr):.4f} -> {out_path.name}")
            else:
                line += f" | (no GT named {f.name}) -> {out_path.name}"
        else:
            line += f" -> {out_path.name}"
        print(line)

    print(f"[done] {len(files)} image(s) under {args.out}")


if __name__ == "__main__":
    main()
