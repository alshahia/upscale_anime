# anime_upscaler/validate.py
"""Stage-6 validation: side-by-side visual inspection on held-out frames.

For K test-split frames saves a labelled strip [bicubic | student | teacher]
plus metrics; reports line-art sharpness (Laplacian variance) and flags
possible texture hallucination when the student's high-frequency energy far
exceeds the teacher's.

Usage: python anime_upscaler/validate.py [--run-dir runs/distill_v1] [--frames 5]
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from dataset import AnimePairDataset, denorm01
from distill import psnr01, ssim01
from student import RFDN
from teacher import SPANTeacher

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"


def lap_var(img01):
    """Variance of the 3x3 Laplacian response - line-art sharpness proxy."""
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
    k = k.view(1, 1, 3, 3).expand(3, 1, 3, 3)
    resp = F.conv2d(img01, k.to(img01.device), padding=1, groups=3)
    return resp.var().item()


def label(img, text):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
    d.text((6, 5), text, fill=(255, 255, 255))
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=str(ROOT / "runs" / "distill_v1"))
    ap.add_argument("--data", default=str(ROOT / "data" / "anime_video_frames"))
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--allow-pickle", action="store_true",
                    help="allow pickle checkpoint loading (only use with trusted checkpoints)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(exist_ok=True)

    try:
        state = torch.load(Path(args.run_dir) / "student_best.pt",
                          map_location="cpu", weights_only=True)
    except Exception as e:
        if not args.allow_pickle:
            raise RuntimeError("Checkpoint requires pickle loading. Use --allow-pickle to allow. Only use with trusted checkpoints!")
        import warnings
        warnings.warn("Loading checkpoint with pickle fallback - only use with trusted sources!", UserWarning, stacklevel=2)
        state = torch.load(Path(args.run_dir) / "student_best.pt",
                          map_location="cpu", weights_only=False)
    student = RFDN()
    student.load_state_dict(state["student"])
    student.eval().to(device)
    teacher = SPANTeacher(device=device)

    ds = AnimePairDataset(args.data, "test")
    idxs = np.linspace(0, len(ds) - 1, args.frames).astype(int)

    rows = [["frame", "psnr_bicubic", "psnr_student", "psnr_teacher",
             "ssim_student", "lap_bicubic", "lap_student", "lap_teacher",
             "hallucination_flag"]]
    print(f"validating {len(idxs)} held-out frames ...")
    for j, i in enumerate(idxs):
        lr_n, hr_n = ds[int(i)]
        lr = denorm01(lr_n)[None].to(device)
        hr = denorm01(hr_n)[None].to(device)
        with torch.no_grad():
            bic = F.interpolate(lr, scale_factor=4, mode="bicubic",
                                align_corners=False).clamp(0, 1)
            sr = student(lr).clamp(0, 1)
            tt = teacher(lr).clamp(0, 1)
        p_b, p_s, p_t = (psnr01(x, hr) for x in (bic, sr, tt))
        s_ssim = ssim01(sr, hr)
        l_b, l_s, l_t = lap_var(bic), lap_var(sr), lap_var(tt)
        # Hallucination heuristic: much more HF energy than the teacher.
        flag = "YES" if l_s > 1.8 * max(l_t, 1e-6) else "no"
        rows.append([ds.files[i].name[:60], "%.2f" % p_b, "%.2f" % p_s,
                     "%.2f" % p_t, "%.4f" % s_ssim,
                     "%.4f" % l_b, "%.4f" % l_s, "%.4f" % l_t, flag])

        strip_h, strip_w = hr.shape[-2], hr.shape[-1]
        strip = Image.new("RGB", (strip_w * 3 + 20, strip_h), (30, 30, 30))
        for slot, (pred, name) in enumerate([(bic, "bicubic"), (sr, "student"),
                                             (tt, "teacher")]):
            arr = (pred[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            im = Image.fromarray(arr)
            strip.paste(label(im, name), (slot * (strip_w + 10), 0))
        out_path = OUT / f"validate_frame_{j:02d}.png"
        strip.save(out_path)
        print(f"  [{j}] PSNR b/s/t = {p_b:.2f}/{p_s:.2f}/{p_t:.2f} dB | "
              f"lap b/s/t = {l_b:.3f}/{l_s:.3f}/{l_t:.3f} | "
              f"hallucination: {flag} -> {out_path.name}")

    with open(OUT / "validation_metrics.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[done] strips + validation_metrics.csv under {OUT}")


if __name__ == "__main__":
    main()
