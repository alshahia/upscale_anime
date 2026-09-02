#!/usr/bin/env python3
"""Quality-only comparison of our distilled students vs pretrained anime SR models.

Inputs: a single real-anime frame from tmp/real_video_1sec.mp4 (frame 8).

Models compared (all output 4x):
  - bicubic                       (baseline, no model)
  - v1_student_4x                 RFDN_distill_v1_4x_student.pth, scale=4
  - v2_student_cascade_2x2        RFDN_distill_v2_2x_student.pth applied twice (2x -> 2x)
  - RealESRGAN_animevideov3       realesr-animevideov3.pth (canonical anime x4)
  - RealESRGAN_LSDIR              4xLSDIRCompactv2.pth (general compact x4)
  - AnimeSR_v2                    AnimeSR_v2.pth (3-frame recurrent x4)

Outputs:
  results/quality_compare_students_vs_pretrained/
    source_frame.png              854x480 LR
    <model>_4x.png                native 4x PNG per model
    crops/<model>_{hair,eye}_2x.png   zoom crops (2x) of high-detail regions
    grid_full.png                 all 4x outputs at half-scale, vertical
    grid_zoom.png                 zoom crops in a grid for visual diff
    metrics.csv / metrics.md      per-model objective quality

Usage:
  .venv/Scripts/python.exe scripts/compare_students_vs_pretrained.py
"""

import argparse, csv, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "apps" / "anime_upscaler_gui"))

from anime_upscaler_gui.archs import build  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIDEO   = REPO_ROOT / "tmp" / "real_video_1sec.mp4"
OUTDIR  = REPO_ROOT / "results" / "quality_compare_students_vs_pretrained"

# (label, kind, ckpt_relpath, scale, mode)
MODELS = [
    ("bicubic",                  "bicubic",      None,                                              4, "image"),
    ("v1_student_4x",            "rfdn_student", "pretrained/RFDN_distill_v1_4x_student.pth",       4, "image"),
    ("v2_student_cascade_2x2",   "rfdn_student", "pretrained/RFDN_distill_v2_2x_student.pth",       2, "image_cascade"),
    ("RealESRGAN_animevideov3",  "srvgg",        "pretrained/realesr-animevideov3.pth",             4, "image"),
    ("RealESRGAN_LSDIR",         "srvgg",        "pretrained/4xLSDIRCompactv2.pth",                 4, "image"),
    # ("AnimeSR_v2",               "animesr",      "pretrained/AnimeSR_v2.pth",                       4, "video"),
]


def _add_v3_row(models, v3_ckpt):
    """Phase 3 handoff A.5: append a v3 row if --v3-ckpt is provided.

    Accepts both absolute paths and paths relative to REPO_ROOT. Skipped
    silently when v3_ckpt is None (the default) or when the file doesn't
    exist on disk yet (Phase C will produce it).
    """
    if not v3_ckpt:
        return models
    p = Path(v3_ckpt)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        print(f"[info] --v3-ckpt {v3_ckpt} does not exist; skipping v3 row")
        return models
    # v3 ckpt path is the literal arg so the row prints it verbatim
    rel = v3_ckpt if Path(v3_ckpt).is_absolute() else str(p.relative_to(REPO_ROOT))
    return models + [("v3_student_4x", "rfdn_student", rel, 4, "image")]

CROPS = {
    "hair": dict(x0=470, y0=20,  w=180, h=180),
    "eye":  dict(x0=540, y0=140, w=140, h=100),
}


def extract_frame(video, out_png, frame_idx=8):
    import subprocess
    if out_png.exists() and out_png.stat().st_size > 1000:
        return True
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-i", str(video),
           "-vf", f"select=eq(n\\,{frame_idx})",
           "-vframes", "1", str(out_png)]
    subprocess.run(cmd, check=True, timeout=30)
    return out_png.exists()


def load_lr(p):
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cv2 failed to read {p}")
    return bgr


def to_tensor(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    return t


def from_tensor(t):
    arr = (t.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def upscale_bicubic(bgr, scale):
    h, w = bgr.shape[:2]
    return cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def laplacian_variance(gray):
    f = np.float32(gray)
    lap = cv2.Laplacian(f, cv2.CV_32F, ksize=3)
    return float(lap.var())


def mean_grad_magnitude(gray):
    f = np.float32(gray)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.abs(gx) + np.abs(gy)))


def saturation_std(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].std())


def channel_stats(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return dict(
        mean=float(bgr.mean()),
        std=float(bgr.std()),
        gray_mean=float(gray.mean()),
        gray_std=float(gray.std()),
        lap_var=laplacian_variance(gray),
        grad_mag=mean_grad_magnitude(gray),
        sat_std=saturation_std(bgr),
    )


def crop_region(bgr, region, scale):
    x0, y0, w, h = region["x0"], region["y0"], region["w"], region["h"]
    sx0, sy0 = x0 * scale, y0 * scale
    sx1, sy1 = (x0 + w) * scale, (y0 + h) * scale
    return bgr[sy0:sy1, sx0:sx1].copy()


def save_zoom(crop, out_path, zoom=2):
    if zoom > 1:
        h, w = crop.shape[:2]
        crop = cv2.resize(crop, (w * zoom, h * zoom), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(out_path), crop)


def make_grid_full(images):
    rows = []
    for label, bgr in images:
        h, w = bgr.shape[:2]
        rs = cv2.resize(bgr, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        band_w = 240
        band = np.full((rs.shape[0], band_w, 3), 30, dtype=np.uint8)
        cv2.putText(band, label, (8, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(np.hstack([band, rs]))
    return np.vstack(rows)


def make_grid_zoom(crops):
    regions = ["hair", "eye"]
    panels = []
    for region in regions:
        row = [c for c in crops if c[1] == region]
        if not row:
            continue
        max_w = max(im.shape[1] for _, _, im in row)
        h0 = row[0][2].shape[0]
        # band spans the full row width (max_w per panel, * number of tiles)
        band = np.full((36, max_w * len(row), 3), 30, dtype=np.uint8)
        x = 8
        for lab, _, _ in row:
            cv2.putText(band, lab, (x, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
            x += 220
        padded = []
        for _, _, im in row:
            pad = np.full((h0, max_w, 3), 0, dtype=np.uint8)
            pad[:im.shape[0], :im.shape[1]] = im
            padded.append(pad)
        panel = np.vstack([band, np.hstack(padded)])
        panels.append(panel)
    # Normalize widths across panels (hair vs eye region sizes differ)
    if len(panels) > 1:
        max_w = max(p.shape[1] for p in panels)
        norm = []
        for p in panels:
            if p.shape[1] < max_w:
                pad = np.full((p.shape[0], max_w, 3), 0, dtype=np.uint8)
                pad[:, :p.shape[1]] = p
                norm.append(pad)
            else:
                norm.append(p)
        return np.vstack(norm)
    return panels[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-idx", type=int, default=8)
    # Phase 3 handoff A.5: --v3-ckpt appends a v3 row to MODELS (no-op when
    # the file does not exist yet). --output overrides the default OUTDIR.
    ap.add_argument("--v3-ckpt", default=None,
                    help="path to v3 student .pth (skipped if not on disk)")
    ap.add_argument("--output", default=None,
                    help="output directory (default: results/quality_compare_students_vs_pretrained)")
    args = ap.parse_args()

    out_dir = Path(args.output) if args.output else OUTDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)

    lr_png = out_dir / "source_frame.png"
    if not extract_frame(VIDEO, lr_png, args.frame_idx):
        print(f"[FATAL] failed to extract frame from {VIDEO}")
        return 1
    lr_bgr = load_lr(lr_png)
    H, W = lr_bgr.shape[:2]
    print(f"[Setup] LR: {W}x{H}  device={DEVICE}  out={out_dir}")

    metrics_rows = []
    full_outputs = []
    zoom_crops   = []

    models = _add_v3_row(MODELS, args.v3_ckpt)
    for label, kind, ckpt_rel, scale, mode in models:
        t0 = time.perf_counter()
        try:
            if kind == "bicubic":
                sr_bgr = upscale_bicubic(lr_bgr, 4)
                mode_eff = "bicubic 4x"
            else:
                ckpt_path = REPO_ROOT / ckpt_rel
                if not ckpt_path.exists():
                    print(f"[SKIP] {label}: ckpt not found {ckpt_path}")
                    continue
                model = build(kind, str(ckpt_path)).to(DEVICE)
                x_lr = to_tensor(lr_bgr)
                with torch.no_grad():
                    if mode == "image_cascade":
                        mid = model(x_lr)         # 2x
                        y   = model(mid)          # 2x -> 4x
                        mode_eff = "rfdn cascade (2x x2)"
                    elif mode == "video":
                        x5 = x_lr.unsqueeze(1).expand(-1, 3, -1, -1, -1).contiguous()
                        yy = model(x5)
                        y  = yy[:, yy.shape[1] // 2]
                        mode_eff = f"{kind} ({scale}x, 3-frame middle)"
                    else:
                        y = model(x_lr)
                        mode_eff = f"{kind} ({scale}x)"
                sr_bgr = from_tensor(y)
                del model
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[FAIL] {label}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            continue

        dt_ms = (time.perf_counter() - t0) * 1000
        sr_path = out_dir / f"{label}_4x.png"
        cv2.imwrite(str(sr_path), sr_bgr)
        sh, sw = sr_bgr.shape[:2]
        stats = channel_stats(sr_bgr)
        metrics_rows.append(dict(label=label, mode=mode_eff, w=sw, h=sh,
                                 dt_ms=round(dt_ms, 1), **stats))
        full_outputs.append((label, sr_bgr))
        print(f"  [{label:24}] {sw}x{sh}  {dt_ms:7.1f} ms  "
              f"mean={stats['mean']:.3f} std={stats['std']:.3f}  "
              f"lap={stats['lap_var']:7.1f} grad={stats['grad_mag']:5.2f}  "
              f"sat_std={stats['sat_std']:5.2f}")

        for region_name, region in CROPS.items():
            crop_sr = crop_region(sr_bgr, region, scale=4)
            zoom_path = out_dir / "crops" / f"{label}_{region_name}_2x.png"
            save_zoom(crop_sr, zoom_path, zoom=2)
            zoom_crops.append((label, region_name, cv2.imread(str(zoom_path))))

    grid_full_path = out_dir / "grid_full.png"
    grid_zoom_path = out_dir / "grid_zoom.png"
    cv2.imwrite(str(grid_full_path), make_grid_full(full_outputs))
    cv2.imwrite(str(grid_zoom_path), make_grid_zoom(zoom_crops))
    print(f"[Grid] full: {grid_full_path}")
    print(f"[Grid] zoom: {grid_zoom_path}")

    csv_path = out_dir / "metrics.csv"
    md_path  = out_dir / "metrics.md"
    fieldnames = ["label", "mode", "w", "h", "dt_ms",
                  "mean", "std", "gray_mean", "gray_std",
                  "lap_var", "grad_mag", "sat_std"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in metrics_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    with open(md_path, "w") as f:
        f.write("# Quality metrics (real-anime frame 8, LR 854x480 -> SR 3416x1920)\n\n")
        f.write("Sorted by **laplacian variance** (desc) — higher = sharper.\n\n")
        f.write("| Model | Mode | Res | mean | std | lap_var | grad_mag | sat_std |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|\n")
        for r in sorted(metrics_rows, key=lambda x: -x["lap_var"]):
            f.write(f"| {r['label']} | {r['mode']} | {r['w']}x{r['h']} | "
                    f"{r['mean']:.3f} | {r['std']:.3f} | {r['lap_var']:.1f} | "
                    f"{r['grad_mag']:.2f} | {r['sat_std']:.2f} |\n")
        f.write("\n**Laplacian variance**: higher = sharper. Bicubic on anime LR is usually 50-200, ")
        f.write("good SR typically 200-600, over-sharpened SR can exceed 1000.\n\n")
        f.write("**Mean gradient magnitude**: higher = more visible edges / detail.\n\n")
        f.write("**Saturation std**: higher = more chromatic variation preserved.\n")
    print(f"[Metrics] {csv_path}")
    print(f"[Metrics] {md_path}")
    print(f"[Done] {len(metrics_rows)} models processed -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
