# anime_upscaler/export.py
"""Export the distilled student to ONNX (opset 17, dynamic shapes) with
post-training FP16 and calibrated INT8 variants, then report size/latency.

Usage:
    python anime_upscaler/export.py [--run-dir runs/distill_v1]
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from onnxruntime.quantization import CalibrationDataReader

from anime_sr.models.students import RFDN, build_student

ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXPORT_DIR = ROOT / "anime_upscaler" / "export"


def load_student(run_dir, device="cuda", arch=None, allow_pickle=False):
    ckpt_path = Path(run_dir) / "student_best.pt"
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as e:
        if not allow_pickle:
            raise RuntimeError("Checkpoint requires pickle loading. Use --allow-pickle to allow. Only use with trusted checkpoints!")
        import warnings
        warnings.warn("Loading checkpoint with pickle fallback - only use with trusted sources!", UserWarning, stacklevel=2)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_student(state, arch=arch)
    model.load_state_dict(state["student"])
    model.eval().to(device)
    print(f"[load] {ckpt_path.name}: epoch {state.get('epoch')} "
          f"val_psnr {state.get('val_psnr', -1):.2f} dB")
    return model


def export_onnx(model, out_path, half=False):
    """Opset 17 export with dynamic batch/height/width."""
    # deepcopy so .half() never mutates the caller's fp32 model
    import copy
    m = copy.deepcopy(model).half() if half else model
    lr = torch.randn(1, 3, 48, 48, device=next(m.parameters()).device)
    if half:
        lr = lr.half()
    torch.onnx.export(
        m, lr, str(out_path),
        input_names=["lr"], output_names=["sr"],
        dynamic_axes={"lr": {0: "batch", 2: "height", 3: "width"},
                      "sr": {0: "batch", 2: "height", 3: "width"}},
        opset_version=17, dynamo=False)
    print(f"[onnx] wrote {out_path.name} ({out_path.stat().st_size/1e6:.2f} MB)")


def ort_session(path, providers):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=so, providers=providers)


def _frame_size(path):
    with Image.open(path) as im:
        return im.size


def make_calibration_reader(n_samples=32, crop=192):
    """Feeds representative anime LR patches (bicubic of HR crops) to INT8 calibration."""
    frames = sorted((ROOT / "data" / "anime_video_frames").glob("*.png"))
    # Skip frames too small for the calibration crop: w < crop or h < crop
    # would make the crop box invalid and the modulo below negative.
    frames = [f for f in frames
              if _frame_size(f)[0] >= crop and _frame_size(f)[1] >= crop]
    if not frames:
        raise RuntimeError(
            f"no frames of at least {crop}x{crop} under data/anime_video_frames "
            f"for INT8 calibration")
    step = max(1, len(frames) // n_samples)

    class AnimeCalibrationReader(CalibrationDataReader):
        def __init__(self):
            self.idx = 0

        def get_next(self):
            if self.idx >= n_samples:
                return None
            im = Image.open(frames[(self.idx * step) % len(frames)]).convert("RGB")
            w, h = im.size
            x = (w // 3 + self.idx * 37) % (w - crop)
            y = (h // 3) % (h - crop)
            hr = im.crop((x, y, x + crop, y + crop))
            lr = hr.resize((crop // 4, crop // 4), Image.BICUBIC)
            arr = np.asarray(lr, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
            self.idx += 1
            return {"lr": arr}

        def rewind(self):
            self.idx = 0

    return AnimeCalibrationReader()


def verify_parity(torch_model, onnx_path, providers, tag, half=False):
    """Compare ONNX runtime output against PyTorch on a fixed anime patch."""
    import glob
    frames = sorted(glob.glob(str(ROOT / "data" / "anime_video_frames" / "*.png")))
    if len(frames) < 8:
        raise RuntimeError(
            f"verify_parity needs >= 8 frames under data/anime_video_frames, "
            f"found {len(frames)}")
    hr = Image.open(frames[7]).convert("RGB").crop((256, 256, 256 + 320, 256 + 192))
    lr = hr.resize((80, 48), Image.BICUBIC)
    arr = np.asarray(lr, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    # Reference always comes from the fp32 torch model so we isolate
    # quantization error from any dtype mismatch.
    t_in = torch.from_numpy(arr).to(torch.float32).to("cuda")
    with torch.no_grad():
        ref = torch_model(t_in).float().cpu().numpy()
    sess = ort_session(onnx_path, providers)
    out = sess.run(["sr"], {"lr": arr.astype(np.float16 if half else np.float32)})[0]
    max_diff = float(np.abs(out.astype(np.float32) - ref).max())
    mse = ((np.clip(out, 0, 1) - np.clip(ref, 0, 1)) ** 2).mean()
    psnr_delta = -10 * np.log10(max(mse, 1e-12)) if mse > 0 else 99.0
    print(f"[parity] {tag}: max|diff|={max_diff:.4f} (PSNR-equivalent {psnr_delta:.1f} dB)")
    return max_diff


def bench(fn, warmup=10, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=str(ROOT / "runs" / "distill_v1"))
    ap.add_argument("--allow-pickle", action="store_true",
                    help="allow pickle checkpoint loading (only use with trusted checkpoints)")
    ap.add_argument("--arch", choices=["auto", "rfdn", "srvgg"], default="auto",
                    help="student architecture (default: auto-detect from checkpoint)")
    args = ap.parse_args()

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_student(args.run_dir, device,
                         arch=None if args.arch == "auto" else args.arch, allow_pickle=args.allow_pickle)
    n_params = sum(p.numel() for p in model.parameters())

    fp32_path = EXPORT_DIR / "student_fp32.onnx"
    fp16_path = EXPORT_DIR / "student_fp16.onnx"
    int8_path = EXPORT_DIR / "student_int8.onnx"

    print("== exporting fp32 ==")
    export_onnx(model, fp32_path)
    verify_parity(model, fp32_path, ["CUDAExecutionProvider"], "fp32")

    print("== exporting fp16 ==")
    export_onnx(model, fp16_path, half=True)
    try:
        verify_parity(model, fp16_path, ["CUDAExecutionProvider"], "fp16", half=True)
    except Exception as e:
        print(f"[fp16] verification issue: {type(e).__name__}: {str(e)[:200]}")

    print("== building calibrated int8 ==")
    try:
        from onnxruntime.quantization import (CalibrationMethod, QuantFormat,
                                              QuantType, quantize_static)
        reader = make_calibration_reader(n_samples=32)
        # Entropy (KL) calibration over real anime patches avoids flat-region banding.
        quantize_static(
            model_input=str(fp32_path), model_output=str(int8_path),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
            per_channel=True,
            calibrate_method=CalibrationMethod.Entropy,
            op_types_to_quantize=["Conv"])
        print(f"[onnx] wrote {int8_path.name} ({int8_path.stat().st_size/1e6:.2f} MB)")
        verify_parity(model, int8_path, ["CPUExecutionProvider"], "int8")
    except Exception as e:
        print(f"[int8] FAILED: {type(e).__name__}: {str(e)[:300]}")

    # ---- latency + size report ----
    rows = []
    H, W = 270, 480   # -> 1080x1920 output
    xp = torch.randn(1, 3, H, W, device=device)
    with torch.no_grad():
        ms = bench(lambda: model(xp))
    rows.append(("pytorch-fp32-gpu", n_params, fp32_path.stat().st_size / 1e6, ms))

    for path, provs, tag in [
        (fp32_path, ["CUDAExecutionProvider", "CPUExecutionProvider"], "ort-cuda-fp32"),
        (fp16_path, ["CUDAExecutionProvider", "CPUExecutionProvider"], "ort-cuda-fp16"),
        (int8_path, ["CPUExecutionProvider"], "ort-cpu-int8"),
    ]:
        if not path.exists():
            continue
        try:
            sess = ort_session(path, provs)
            feed_np = xp.cpu().numpy()
            if "fp16" in tag:
                feed_np = feed_np.astype(np.float16)
            ms = bench(lambda: sess.run(["sr"], {"lr": feed_np}))
            rows.append((tag, "-", path.stat().st_size / 1e6, ms))
        except Exception as e:
            print(f"[bench] {tag} skipped: {type(e).__name__}: {str(e)[:160]}")

    print("\n=== EXPORT REPORT ===")
    print(f"{'variant':<18} {'params':>12} {'size MB':>9} {'ms/frame':>10}")
    for tag, p, mb, ms in rows:
        ps = f"{p:,}" if isinstance(p, int) else "-"
        print(f"{tag:<18} {ps:>12} {mb:>9.2f} {ms:>10.2f}")
    with open(EXPORT_DIR / "export_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "params", "size_mb", "latency_ms_per_frame_270x480"])
        for tag, p, mb, ms in rows:
            w.writerow([tag, p, "%.2f" % mb, "%.2f" % ms])
    print(f"[done] report: {EXPORT_DIR / 'export_report.csv'}")


if __name__ == "__main__":
    main()
