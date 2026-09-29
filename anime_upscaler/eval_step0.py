# anime_upscaler/eval_step0.py
r"""Step-0 model shootout: pretrained champions vs our from-scratch models.

Quality: pyiqa CLIPIQA + NIQE + Laplacian sharpness on real mp4upload frames
(center 384x384 input crops, 4x upscale), no reference needed.
Speed: fp16 (when supported) forward latency at 640x360 and 960x540 LR inputs.
Input convention [0,1] — probe_domain() in eval_teachers.py chose "01" for
every benchmarked candidate.

Usage:
    .venv\Scripts\python.exe anime_upscaler/eval_step0.py --limit 3   # smoke
    .venv\Scripts\python.exe anime_upscaler/eval_step0.py             # full
"""
import argparse
import csv
import sys
import time
from pathlib import Path


_LOADER = None


def load_model_with_scale(path, device):
    """Spandrel loader returning (nn.Module, scale, supports_half)."""
    global _LOADER
    from spandrel import ImageModelDescriptor, ModelLoader
    if _LOADER is None:
        import spandrel_extra_arches
        spandrel_extra_arches.install()
        _LOADER = ModelLoader()
    desc = _LOADER.load_from_file(str(path))
    if not isinstance(desc, ImageModelDescriptor):
        raise TypeError(str(path.name) + ": not an image model")
    desc.to(device).eval()
    return desc.model, int(desc.scale), bool(desc.supports_half)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_teachers import lap_var01  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PRETRAINED = ROOT / "pretrained"
CANDIDATES = ROOT / "checkpoints" / "candidates"

MODELS = {
    "realesr-animevideov3": PRETRAINED / "realesr-animevideov3.pth",
    "4x_APISR_GRL_GAN": CANDIDATES / "4x_APISR_GRL_GAN_generator.pth",
    "4x_APISR_DAT_GAN": CANDIDATES / "4x_APISR_DAT_GAN_generator.pth",
    "4x_APISR_RRDB_GAN": CANDIDATES / "4x_APISR_RRDB_GAN_generator.pth",
    "4x-AnimeSharp": CANDIDATES / "4x-AnimeSharp.pth",
    "4xHFA2k_plksr": CANDIDATES / "4xHFA2k_ludvae_realplksr_dysample.pth",
    "RFDN_distill_v1_student": PRETRAINED / "RFDN_distill_v1_4x_student.pth",
    "step2_srvgg_hfa_v1": ROOT / "runs" / "step2_srvgg_hfa_v1" / "student_best.pt",
    "NEOSR_SPAN_V7_ANIME_best": ROOT / "checkpoints" / "NEOSR_SPAN_V7_ANIME_001" / "finetune_best.pth",
}
SPEED_SIZES = [(640, 360), (960, 540)]
QUALITY_CROP = 384
SAMPLE_FRAMES = 2


def load_rfdn_student(path, device):
    """Our RFDN student as saved by anime_upscaler/distill.py."""
    from student import RFDN
    try:
        ck = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as e:
        if not args.allow_pickle:
            raise RuntimeError("Checkpoint requires pickle loading. Use --allow-pickle to allow. Only use with trusted checkpoints!")
        import warnings
        warnings.warn("Loading checkpoint with pickle fallback - only use with trusted sources!", UserWarning, stacklevel=2)
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = ck.get("student", ck.get("model", ck.get("state_dict", ck)))
    if not any(torch.is_tensor(v) for v in sd.values() if isinstance(v, torch.Tensor)):
        raise ValueError("no tensor state dict in " + str(path))
    net = RFDN(scale=4)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    net.eval().to(device)
    params = sum(p.numel() for p in net.parameters()) / 1e6
    note = f"missing={len(missing)} unexpected={len(unexpected)}"
    return net, params, note


def try_half(net, device):
    try:
        net.half()
        net(torch.rand(1, 3, 64, 64, device=device).half())
        return True
    except Exception:
        net.float()
        return False


def jpeg_degrade(lr, quality=60):
    """Simulate web-stream compression on a [0,1] BHWC float tensor crop."""
    import io
    from PIL import Image as _I
    arr = (lr[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    buf = io.BytesIO()
    _I.fromarray(arr).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    back = np.asarray(_I.open(buf).convert("RGB"), np.float32) / 255.0
    return torch.from_numpy(back).permute(2, 0, 1).unsqueeze(0).to(lr.device)


@torch.inference_mode()
def measure_speed(net, device, use_half):
    out = {}
    net.half() if use_half else net.float()
    for w, h in SPEED_SIZES:
        x = torch.rand(1, 3, h, w, device=device)
        if use_half:
            x = x.half()
        try:
            for _ in range(3):
                net(x)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            for _ in range(8):
                net(x)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 8 * 1000.0
            vram = torch.cuda.max_memory_allocated(device) / 2**20
            out[f"fps_{w}x{h}"] = round(1000.0 / ms, 1)
            out[f"vram_{w}x{h}"] = round(vram, 0)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            # tiled timing (accurate per-tile speed estimate, larger VRAM headroom)
            try:
                scale_hint = 4
                for _ in range(2):
                    tiled_run(net, x, scale_hint, tile=256, pad=8)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
                t0 = time.perf_counter()
                for _ in range(4):
                    tiled_run(net, x, scale_hint, tile=256, pad=8)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) / 4 * 1000.0
                out[f"fps_{w}x{h}"] = round(1000.0 / ms, 1)
                out[f"vram_{w}x{h}"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 0)
                out[f"mode_{w}x{h}"] = "tiled"
            except RuntimeError as e2:
                if "out of memory" not in str(e2).lower():
                    raise
                out[f"fps_{w}x{h}"] = "FAIL:" + type(e2).__name__
                out[f"vram_{w}x{h}"] = ""
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            out[f"fps_{w}x{h}"] = "FAIL:" + type(e).__name__
            out[f"vram_{w}x{h}"] = ""
        torch.cuda.empty_cache()
    return out


@torch.inference_mode()
def tiled_run(net, lr, scale, tile=192, pad=8):
    """Tile-and-paste fallback for models that OOM on a whole-crop forward."""
    b, c, h, w = lr.shape
    out = torch.zeros(b, c, h * scale, w * scale, device=lr.device)
    ys = list(range(0, h, tile))
    xs = list(range(0, w, tile))
    for y0 in ys:
        for x0 in xs:
            torch.cuda.empty_cache()
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            ty0, tx0 = max(y0 - pad, 0), max(x0 - pad, 0)
            ty1, tx1 = min(y1 + pad, h), min(x1 + pad, w)
            patch = net(lr[:, :, ty0:ty1, tx0:tx1]).clamp(0, 1)
            py0, px0 = (y0 - ty0) * scale, (x0 - tx0) * scale
            out[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = (
                patch[
                    :, :, py0 : py0 + (y1 - y0) * scale, px0 : px0 + (x1 - x0) * scale,
                ]
            )
    return out


def run_with_oom_fallback(run_once, lr, scale):
    try:
        with torch.inference_mode():
            return run_once(lr)
    except Exception as e:  # noqa: BLE001
        if "out of memory" not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        # shrink tile until a tile forward fits VRAM
        tile = 192
        while tile >= 32:
            try:
                print(f"  OOM on full crop -> tiled fallback (tile={tile})")
                return tiled_run(run_once, lr, scale, tile=tile)
            except RuntimeError as te:
                if "out of memory" not in str(te).lower():
                    raise
                torch.cuda.empty_cache()
                tile //= 2
        raise


def center_crop(img, size):
    H, W = img.shape[:2]
    cy, cx = H // 2, W // 2
    return img[cy - size // 2: cy + size // 2, cx - size // 2: cx + size // 2]


def main():
    ap = argparse.ArgumentParser(description="Step-0 model shootout")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--frames-dir", default=str(ROOT / "data" / "anime_hr"))
    ap.add_argument("--frames-glob", default="*mp4upload*.png")
    ap.add_argument("--models", default=None, help="comma list; default=all")
    ap.add_argument("--iqa", default="clipiqa,niqe")
    ap.add_argument("--skip-speed", action="store_true")
    ap.add_argument("--quality-crop", type=int, default=QUALITY_CROP,
                    help="input side length for the quality pass; shrink if OOM")
    ap.add_argument("--degrade", type=int, default=None,
                    help="JPEG quality for stream-artifact simulation before upscale")
    ap.add_argument("--out-csv", default=str(ROOT / "results" / "step0_model_shootout.csv"))
    ap.add_argument("--allow-pickle", action="store_true",
                    help="allow pickle checkpoint loading (only use with trusted checkpoints)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    if device == "cuda":
        # Hard-cap PyTorch's allocator to ~90% of the DEDICATED VRAM so heavy
        # models spill into tiled fallback instead of Windows shared GPU
        # memory (which cripples throughput).
        total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(0.9, 7.2 / total_gb), 0)
        print(f"allocator capped to ~{min(0.9, 7.2 / total_gb):.2f} of {total_gb:.1f} GB dedicated VRAM")

    frames = sorted(Path(args.frames_dir).glob(args.frames_glob))
    if not frames:
        raise SystemExit("no frames matched " + str(Path(args.frames_dir) / args.frames_glob))
    idx = np.linspace(0, len(frames) - 1, min(args.limit, len(frames))).round().astype(int)
    frames = [frames[i] for i in idx]
    print(f"{len(frames)} frames, e.g. {frames[0].name}")

    import pyiqa
    iqa_names = [m for m in args.iqa.split(",") if m]
    metrics = {m: pyiqa.create_metric(m, device=device).to(device) for m in iqa_names}

    want = args.models.split(",") if args.models else list(MODELS.keys())
    rows = []
    for name in want:
        row = {"model": name, "frame_count": len(frames)}
        try:
            note = ""
            scale = 4
            if name == "bicubic":
                net, params, half_ok = None, 0.0, False

                def run(x, _n=None):
                    return F.interpolate(x, scale_factor=4, mode="bicubic",
                                         align_corners=False).clamp(0, 1)
            elif name == "RFDN_distill_v1_student":
                net, params, note = load_rfdn_student(MODELS[name], device)
            elif name == "step2_srvgg_hfa_v1":
                from student import TinySRVGGStudent
                try:
                    ck = torch.load(str(MODELS[name]), map_location="cpu",
                                    weights_only=True)
                except Exception as e:
                    if not args.allow_pickle:
                        raise RuntimeError("Checkpoint requires pickle loading. Use --allow-pickle to allow. Only use with trusted checkpoints!")
                    import warnings
                    warnings.warn("Loading checkpoint with pickle fallback - only use with trusted sources!", UserWarning, stacklevel=2)
                    ck = torch.load(str(MODELS[name]), map_location="cpu",
                                    weights_only=False)
                sd = ck.get("student", ck.get("model", ck.get("state_dict", ck)))
                net = TinySRVGGStudent(scale=4)
                msg = net.load_state_dict(sd, strict=False)
                note = ("missing=%d unexpected=%d val_psnr=%.2f" %
                        (len(msg.missing_keys), len(msg.unexpected_keys),
                         float(ck.get("val_psnr", 0) or 0)))
                params = sum(p.numel() for p in net.parameters()) / 1e6
                net.eval().to(device)
                half_ok = try_half(net, device)
            else:
                path = MODELS[name]
                if not path.exists():
                    raise FileNotFoundError(str(path))
                net, scale, desc_half = load_model_with_scale(path, device)
                params = sum(p.numel() for p in net.parameters()) / 1e6
                half_ok = desc_half and try_half(net, device)
                run = (lambda x: net(x.half()).float().clamp(0, 1)) if half_ok \
                      else (lambda x: net(x).clamp(0, 1))

            if net is None:
                def run(x, _n=None):
                    return F.interpolate(x, scale_factor=4, mode="bicubic",
                                         align_corners=False).clamp(0, 1)
            elif half_ok:
                run = (lambda x, n=net: n(x.half()).float().clamp(0, 1))
            else:
                run = (lambda x, n=net: n(x).clamp(0, 1))
            row.update(status="ok", params_M=round(params, 2), half=half_ok,
                       scale=scale, note=note)

            # --- quality on real frames (optional JPEG stream degradation)
            vals = {m: [] for m in metrics}
            laps, t0 = [], time.perf_counter()
            for f in frames:
                img = np.asarray(Image.open(str(f)).convert("RGB"), np.float32) / 255.0
                lr = torch.from_numpy(center_crop(img, args.quality_crop).copy())
                lr = lr.permute(2, 0, 1).unsqueeze(0).to(device)
                if args.degrade is not None:
                    lr = jpeg_degrade(lr, args.degrade)
                sr = run_with_oom_fallback(run, lr, scale)
                for mname in metrics:
                    v = float(metrics[mname](sr[:, :, :sr.shape[2], :sr.shape[3]].clamp(0, 1)))
                    vals[mname].append(v)
                laps.append(lap_var01(sr))
                del sr, lr
                torch.cuda.empty_cache()
            dt = time.perf_counter() - t0
            for mname in metrics:
                row[mname] = round(float(np.mean(vals[mname])), 4)
            row["laplacian"] = round(float(np.mean(laps)), 6)
            row["iqa_s_total"] = round(dt, 1)

            # --- speed
            if not args.skip_speed and net is not None:
                row.update(measure_speed(net, device, half_ok))
        except Exception as e:  # noqa: BLE001 report and continue
            row["status"] = ("FAIL: " + type(e).__name__ + ": " + str(e))[:220]
        rows.append(row)
        keep = ("model", "status", "clipiqa", "niqe", "fps_640x360", "fps_960x540")
        print({k: row.get(k) for k in keep})

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["model", "status", "scale", "params_M", "half", "clipiqa", "niqe", "laplacian",
            "fps_640x360", "vram_640x360", "mode_640x360",
            "fps_960x540", "vram_960x540", "mode_960x540",
            "iqa_s_total", "note", "frame_count"]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)

    ok = [r for r in rows if r.get("status") == "ok" and r.get("clipiqa") is not None]
    if ok:
        best_q = max(ok, key=lambda r: r.get("clipiqa", -1))
        print("BEST quality :", best_q["model"], "CLIPIQA", best_q.get("clipiqa"))
        rt = [r["model"] for r in ok
              if isinstance(r.get("fps_640x360"), (int, float)) and r["fps_640x360"] >= 25]
        print("REALTIME @ 640x360 LR input (>=25fps):", rt)


if __name__ == "__main__":
    main()
