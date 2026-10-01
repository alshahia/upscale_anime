#!/usr/bin/env python3
"""
Side-by-side checkpoint comparison harness for v6/v7 evaluation.

Runs two (or more) checkpoints over the same set of input images, computes
full-reference metrics (PSNR/SSIM/LPIPS) when a ground-truth directory is
provided, and no-reference metrics (NIQE/MANIQA/CLIPIQA/TOPIQ) on every
output. Per-image rows are written to a CSV and a mean +/- std summary
table is printed at the end.

Usage:
    # v6 vs v7 (after v7 has been trained):
    python scripts/compare_checkpoints.py ^
        --baseline checkpoints/NEOSR_SPAN_V6_ANIME/finetune_best.pth ^
        --v7 checkpoints/NEOSR_SPAN_V7_ANIME/finetune_best.pth ^
        --input data/val_hr ^
        --gt data/val_hr ^
        --output results/comparison_v6_vs_v7/ ^
        --metrics psnr ssim lpips clipiqa maniqa niqe topiq_nr

    # Baseline-only (single checkpoint):
    python scripts/compare_checkpoints.py ^
        --baseline pretrained/span_pix_pretrain_4x.pth ^
        --input data/val_hr ^
        --output results/baseline/ ^
        --metrics clipiqa maniqa niqe topiq_nr

The script reuses metric functions from src/utils/metrics.py and the
checkpoint loader pattern from scripts/inference.py. It is intentionally
self-contained (no dependency on inference.py subprocess) so the harness
is fast and CI-friendly.
"""
import sys
import argparse
import logging
import csv
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Match the project's bare-import style: add src/ to sys.path so that
# `from models.span...` works (same convention as scripts/inference.py).
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from anime_sr.models.span import create_neosr_span, create_span_model
from anime_sr.utils.config import Config
from anime_sr.inference.tta import tta_forward as _tta_forward
from anime_sr.utils.metrics import (
    PYIQA_DIRECTION,
    calculate_clipiqa,
    calculate_lpips,
    calculate_maniqa,
    calculate_musiq,
    calculate_niqe,
    calculate_psnr,
    calculate_ssim,
    calculate_topiq,
)

logger = logging.getLogger(__name__)

# Map CLI metric name -> no-reference function (full-ref metrics handled
# separately so the dispatch stays explicit).
NR_METRIC_FNS = {
    'niqe': calculate_niqe,
    'maniqa': calculate_maniqa,
    'clipiqa': calculate_clipiqa,
    'topiq_nr': calculate_topiq,
    'musiq': calculate_musiq,
}

# Default config path: v7 config declares the exact model architecture
# (neosr_span with feature_channels=48, upscale=4, bias=True, norm=False).
# Override with --config to evaluate a different architecture.
DEFAULT_CONFIG = 'configs/finetune_neosr_span_v7_anime.yaml'


def load_checkpoint(
    checkpoint_path: str,
    config_path: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, str]:
    """Load a SPAN/neosr_span checkpoint, returning (model, label).

    The label includes checkpoint name + epoch for traceability in the CSV.
    Falls back to a default neosr_span if --config is not provided.
    """
    model_cfg_path = Path(config_path)
    if model_cfg_path.is_file():
        try:
            config = Config.load(str(model_cfg_path))
            model_cfg = config.get('model', {}) or {}
        except Exception as e:
            logger.warning(f"Could not load config {config_path}: {e}; using defaults")
            model_cfg = {}
    else:
        logger.warning(
            f"Config not found at {config_path}; using default neosr_span 4x."
        )
        model_cfg = {'type': 'neosr_span', 'upscale': 4}

    if model_cfg.get('type') == 'neosr_span':
        model = create_neosr_span(model_cfg)
    else:
        model = create_span_model(model_cfg)

    # Match the inference.py load path: try weights_only first, then fall
    # back. (Unpickling error is the most common cause of fallback.)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            raise

    # Common checkpoint layouts seen in this repo:
    #   - {'model_state_dict': ...}    (finetune checkpoints)
    #   - {'state_dict': ...}          (legacy / distilled)
    #   - {'params': ...}              (neosr/SPAN stock pretrained)
    #   - bare state_dict              (raw)
    state = None
    if isinstance(ckpt, dict):
        for key in ('model_state_dict', 'state_dict', 'params_ema', 'params', 'g_ema'):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
        if state is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state = ckpt
    else:
        state = ckpt

    if state is None:
        raise RuntimeError(
            f"Could not locate model state_dict in checkpoint: {checkpoint_path}"
        )

    # Some checkpoints store 'block.N' keys; some store 'features.block.N'.
    # Try a strict load first; if it fails, attempt a permissive load that
    # warns about missing/unexpected keys (better than crashing).
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
    except Exception as e:
        logger.warning(f"Strict load failed for {checkpoint_path}: {e}")
        raise

    if missing:
        logger.warning(
            f"{Path(checkpoint_path).name}: {len(missing)} missing keys (first 3: "
            f"{missing[:3]})"
        )
    if unexpected:
        logger.warning(
            f"{Path(checkpoint_path).name}: {len(unexpected)} unexpected keys (first 3: "
            f"{unexpected[:3]})"
        )

    model = model.to(device).eval()

    epoch = None
    if isinstance(ckpt, dict):
        epoch = ckpt.get('epoch')
    label = f"{Path(checkpoint_path).stem}"
    if epoch is not None:
        label += f"_e{epoch}"

    print(
        f"  Loaded: {Path(checkpoint_path).name}  "
        f"({sum(p.numel() for p in model.parameters()):,} params)"
    )
    return model, label


def preprocess_image(image_path: Path, scale: int = 4, max_side: Optional[int] = None) -> torch.Tensor:
    """Load an image as a normalized RGB float tensor [1, C, H, W] in [0, 1].

    Identical to scripts/inference.py:preprocess_image so output sizes match
    for downstream metric computation.

    If `max_side` is set, the image is center-cropped (or downscaled) so its
    larger spatial dimension is <= max_side. This is a guard for the
    4x-upscale case where 1920x1080 inputs become 7680x4320 SR outputs and
    dominate the runtime budget.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if max_side is not None and max(h, w) > max_side:
        # Center crop so max side == max_side. Divisibility by `scale` keeps
        # the resulting SR size divisible by the upsampler.
        if h >= w:
            top = (h - max_side) // 2
            img = img[top:top + max_side, :, :]
        else:
            left = (w - max_side) // 2
            img = img[:, left:left + max_side, :]
        # Re-align to a multiple of `scale` (rare but safer).
        h, w = img.shape[:2]
        h2 = (h // scale) * scale
        w2 = (w // scale) * scale
        if h2 != h or w2 != w:
            top = (h - h2) // 2
            left = (w - w2) // 2
            img = img[top:top + h2, left:left + w2, :]
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    return tensor.unsqueeze(0)


def postprocess_to_tensor(model_output: torch.Tensor) -> torch.Tensor:
    """Clamp model output to [0, 1] and return as 4D float32 tensor.

    The finetuner uses AMP (autocast fp16) so the raw model output is fp16
    with values potentially outside [0, 1]. We must:
      1) Move to float32 (LPIPS, SSIM, pyiqa all expect float32 inputs)
      2) Clamp to [0, 1] (LPIPS and pyiqa both enforce a strict range)
    """
    return model_output.squeeze(0).float().clamp(0.0, 1.0).unsqueeze(0)


def run_inference(
    model: torch.nn.Module,
    lr_tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run model inference with AMP where appropriate (matches inference.py)."""
    lr_tensor = lr_tensor.to(device)
    use_amp = device.type == 'cuda'
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=use_amp):
            sr = model(lr_tensor)
    return sr


def run_inference_tta(
    model: torch.nn.Module,
    lr_tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run model inference with 8x D4 test-time augmentation ensemble.

    Delegates to `inference.tta.tta_forward`, which averages 8 D4-augmented
    predictions. The 8x compute cost is acceptable here because this harness
    is for off-line comparison (not real-time inference).
    """
    lr_tensor = lr_tensor.to(device)
    return _tta_forward(lambda x: run_inference(model, x, device), lr_tensor)


def compute_metrics(
    sr_tensor: torch.Tensor,
    metrics: List[str],
    gt_tensor: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute the requested metrics for a single (sr, gt?) pair.

    Full-reference metrics (psnr/ssim/lpips) are skipped silently if no
    ground-truth tensor is provided. No-reference metrics return NaN on
    failure (handled by the underlying functions in utils.metrics).
    """
    results: Dict[str, float] = {}

    # Force both tensors to float32 on CPU-friendly device. pyiqa / LPIPS
    # / our SSIM all require float32; a fp16 input from autocast will
    # crash SSIM (dtype mismatch) and trip pyiqa's strict range check
    # (max can be slightly above 1 after a fp16 resize).
    sr_tensor = sr_tensor.float()
    if gt_tensor is not None:
        gt_tensor = gt_tensor.float()

    # Resize SR to match GT shape if needed (LPIPS/PSNR/SSIM require
    # matching spatial dims; SR is 4x LR by construction).
    if gt_tensor is not None and sr_tensor.shape != gt_tensor.shape:
        sr_tensor = torch.nn.functional.interpolate(
            sr_tensor, size=gt_tensor.shape[2:],
            mode='bicubic', align_corners=False,
        )

    for metric in metrics:
        if metric == 'psnr' and gt_tensor is not None:
            try:
                results['psnr'] = calculate_psnr(sr_tensor, gt_tensor)
            except Exception as e:
                logger.warning(f"PSNR failed: {e}")
                results['psnr'] = float('nan')
        elif metric == 'ssim' and gt_tensor is not None:
            try:
                results['ssim'] = calculate_ssim(sr_tensor, gt_tensor)
            except Exception as e:
                logger.warning(f"SSIM failed: {e}")
                results['ssim'] = float('nan')
        elif metric == 'lpips' and gt_tensor is not None:
            try:
                results['lpips'] = calculate_lpips(sr_tensor, gt_tensor)
            except Exception as e:
                logger.warning(f"LPIPS failed: {e}")
                results['lpips'] = float('nan')
        elif metric in NR_METRIC_FNS:
            try:
                results[metric] = NR_METRIC_FNS[metric](sr_tensor)
            except Exception as e:
                logger.warning(f"{metric.upper()} failed: {e}")
                results[metric] = float('nan')
        else:
            # Either a full-ref metric requested without --gt, or unknown.
            if metric in ('psnr', 'ssim', 'lpips') and gt_tensor is None:
                # Skip silently — common when running NR-only.
                continue
            logger.warning(f"Unknown metric: {metric}")

    return results


def _summarize(values: List[float]) -> Dict[str, float]:
    """mean / median / std / min / max / count, ignoring NaNs and None."""
    finite = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not finite:
        nan = float('nan')
        return {'mean': nan, 'median': nan, 'std': nan, 'min': nan, 'max': nan, 'count': 0}
    return {
        'mean': float(np.mean(finite)),
        'median': float(np.median(finite)),
        'std': float(np.std(finite)) if len(finite) > 1 else 0.0,
        'min': float(np.min(finite)),
        'max': float(np.max(finite)),
        'count': len(finite),
    }


def _direction_for(metric: str) -> str:
    """Look up 'higher is better' / 'lower is better' for a metric name."""
    if metric in PYIQA_DIRECTION:
        return PYIQA_DIRECTION[metric]
    if metric == 'lpips':
        return 'lower'
    if metric in ('psnr', 'ssim'):
        return 'higher'
    return 'n/a'


def discover_images(input_dir: Path, smoke: Optional[int] = None) -> List[Path]:
    """Return sorted image files in input_dir (optionally capped to N for smoke)."""
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    files = sorted(f for f in input_dir.iterdir() if f.suffix.lower() in exts and f.is_file())
    if smoke is not None and smoke > 0:
        files = files[:smoke]
    return files


def main():
    parser = argparse.ArgumentParser(
        description='Compare two (or one) SR checkpoints on a common image set.',
    )
    parser.add_argument(
        '--baseline', '-b', type=str, required=True,
        help='Path to baseline checkpoint (.pth).',
    )
    parser.add_argument(
        '--v7', type=str, default=None,
        help='Path to v7 checkpoint (.pth). If omitted, only --baseline is evaluated.',
    )
    parser.add_argument(
        '--config', type=str, default=DEFAULT_CONFIG,
        help=f'Model config YAML used to instantiate both models (default: {DEFAULT_CONFIG}).',
    )
    parser.add_argument(
        '--input', '-i', type=str, required=True,
        help='Directory with LR or HR input images (file stems must match --gt if FR metrics used).',
    )
    parser.add_argument(
        '--gt', type=str, default=None,
        help='Optional ground-truth directory for full-reference metrics (PSNR/SSIM/LPIPS).',
    )
    parser.add_argument(
        '--output', '-o', type=str, default='results/comparison/',
        help='Output directory for per-image SR outputs and CSV report.',
    )
    parser.add_argument(
        '--metrics', '-m', type=str, nargs='+',
        default=['psnr', 'ssim', 'lpips', 'clipiqa', 'maniqa', 'niqe', 'topiq_nr'],
        help='Metrics to compute. FR metrics silently skipped if --gt not provided.',
    )
    parser.add_argument(
        '--device', '-d', type=str, default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for inference and metric computation.',
    )
    parser.add_argument(
        '--scale', type=int, default=4,
        help='Upscale factor (used for input preprocessing; default 4).',
    )
    parser.add_argument(
        '--smoke', '-s', type=int, default=None,
        help='Process only the first N images (fast CI mode).',
    )
    parser.add_argument(
        '--max-side', type=int, default=None,
        help='Center-crop input images so max(H, W) <= this value. '
             'Recommended for 4x upscale (e.g. --max-side 480 keeps SR output at 1920).',
    )
    parser.add_argument(
        '--save-sr', action='store_true',
        help='Save SR outputs as PNGs into <output>/<label>/ (off by default to save disk).',
    )
    parser.add_argument(
        '--tta', action='store_true',
        help='Enable 8x D4 test-time augmentation ensemble (8x inference cost).',
    )
    args = parser.parse_args()

    if args.tta:
        print("[TTA] Enabled: 8x D4 flip/rot ensemble (8x inference cost)")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )

    input_path = Path(args.input)
    if not input_path.is_dir():
        print(f"Error: --input must be a directory: {input_path}")
        return 1
    if not Path(args.baseline).is_file():
        print(f"Error: baseline checkpoint not found: {args.baseline}")
        return 1
    if args.v7 and not Path(args.v7).is_file():
        print(f"Error: v7 checkpoint not found: {args.v7}")
        return 1

    gt_path = Path(args.gt) if args.gt else None
    if gt_path and not gt_path.is_dir():
        print(f"Warning: --gt path is not a directory ({gt_path}); disabling FR metrics.")
        gt_path = None

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    )
    print(f"Device: {device}")
    print(f"Metrics: {', '.join(args.metrics)}")
    if args.gt:
        print(f"Ground-truth: {gt_path} (FR metrics enabled)")
    else:
        print("No --gt provided (FR metrics will be skipped).")

    image_files = discover_images(input_path, smoke=args.smoke)
    if not image_files:
        print(f"No images found in {input_path}")
        return 1
    print(f"Found {len(image_files)} image(s) to process.\n")

    # Define checkpoint set: always baseline; v7 optional.
    ckpt_specs = [('baseline', args.baseline)]
    if args.v7:
        ckpt_specs.append(('v7', args.v7))

    models: Dict[str, Tuple[torch.nn.Module, str]] = {}
    for name, path in ckpt_specs:
        print(f"Loading {name}: {path}")
        try:
            model, label = load_checkpoint(path, args.config, device)
        except Exception as e:
            logger.error(f"Could not load {name} ({path}): {e}")
            if name == 'baseline':
                # Baseline is required; abort.
                return 2
            else:
                # v7 failure is non-fatal: just skip it and continue baseline-only.
                logger.warning(f"Skipping {name} due to load failure.")
                continue
        models[name] = (model, label)

    if not models:
        print("Error: no checkpoints loaded successfully.")
        return 2

    # Per-image, per-checkpoint metric storage.
    # rows[filename] = { 'baseline': {metric: val, ...}, 'v7': {...} }
    rows: Dict[str, Dict[str, Dict[str, float]]] = {}
    metric_keys: List[str] = []
    # We need the order of metric keys for the CSV header. Use the input list,
    # but filter out any full-ref metrics that won't be computed.
    for m in args.metrics:
        if m in ('psnr', 'ssim', 'lpips') and gt_path is None:
            continue
        if m not in metric_keys:
            metric_keys.append(m)

    print(f"\nProcessing {len(image_files)} image(s) x {len(models)} checkpoint(s)...\n")
    overall_start = time.time()
    for img_path in tqdm(image_files, desc="Images"):
        try:
            lr_tensor = preprocess_image(img_path, scale=args.scale, max_side=args.max_side)
        except Exception as e:
            logger.warning(f"Skipping {img_path.name}: {e}")
            continue

        gt_tensor = None
        if gt_path is not None:
            gt_file = gt_path / img_path.name
            if gt_file.is_file():
                try:
                    gt_tensor = preprocess_image(gt_file, scale=args.scale, max_side=args.max_side).to(device)
                except Exception as e:
                    logger.warning(f"Could not load GT for {img_path.name}: {e}")
            else:
                logger.debug(f"No GT for {img_path.name}; FR metrics will be NaN.")

        rows[img_path.name] = {}
        for name, (model, label) in models.items():
            try:
                if args.tta:
                    sr_tensor = run_inference_tta(model, lr_tensor, device)
                else:
                    sr_tensor = run_inference(model, lr_tensor, device)
                sr_for_metrics = postprocess_to_tensor(sr_tensor)
                metrics_for_ckpt = compute_metrics(
                    sr_for_metrics, args.metrics, gt_tensor=gt_tensor,
                )
            except Exception as e:
                logger.warning(f"{name} inference failed for {img_path.name}: {e}")
                metrics_for_ckpt = {m: float('nan') for m in metric_keys}

            rows[img_path.name][name] = metrics_for_ckpt

            if args.save_sr:
                ckpt_out_dir = output_dir / label
                ckpt_out_dir.mkdir(parents=True, exist_ok=True)
                sr_np = (sr_for_metrics.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255)
                sr_np = np.clip(sr_np, 0, 255).astype(np.uint8)
                sr_bgr = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(ckpt_out_dir / f"{img_path.stem}_sr.png"),
                    sr_bgr,
                )

    elapsed = time.time() - overall_start
    print(f"\nDone in {elapsed:.1f}s ({elapsed / max(1, len(image_files)):.2f}s/image).")

    # ---- Write per-image CSV ----
    csv_path = output_dir / 'comparison.csv'
    fieldnames = ['filename']
    for name in models.keys():
        for m in metric_keys:
            fieldnames.append(f'{name}__{m}')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for filename, ckpt_results in rows.items():
            row = {'filename': filename}
            for name, metrics_dict in ckpt_results.items():
                for m in metric_keys:
                    val = metrics_dict.get(m, float('nan'))
                    row[f'{name}__{m}'] = val
            writer.writerow(row)
    print(f"Per-image CSV: {csv_path}")

    # ---- Print summary table ----
    print("\n" + "=" * 80)
    print("CHECKPOINT COMPARISON SUMMARY")
    print("=" * 80)
    header = f"  {'CHECKPOINT':<14} {'METRIC':<10} {'N':>4} {'MEAN':>10} {'STD':>10} {'MIN':>10} {'MAX':>10}  DIR"
    print(header)
    print("  " + "-" * (len(header) - 2))

    summaries: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name in models.keys():
        summaries[name] = {}
        for m in metric_keys:
            values = [
                rows[fn][name].get(m, float('nan'))
                for fn in rows
                if name in rows[fn]
            ]
            summaries[name][m] = _summarize(values)

    for name in models.keys():
        for m in metric_keys:
            s = summaries[name][m]
            print(
                f"  {name:<14} {m.upper():<10} {s['count']:>4d} "
                f"{s['mean']:>10.4f} {s['std']:>10.4f} "
                f"{s['min']:>10.4f} {s['max']:>10.4f}  "
                f"({_direction_for(m)} is better)"
            )
        print()

    # ---- JSON dump for downstream tooling ----
    json_path = output_dir / 'comparison_summary.json'
    import json
    with open(json_path, 'w') as f:
        json.dump(
            {
                'checkpoints': {
                    name: {
                        'label': models[name][1],
                        'path': str(ckpt_specs[i][1]) if i < len(ckpt_specs) else None,
                    }
                    for i, (name, _) in enumerate(models.items())
                    for name in [name]
                },
                'n_images': len(rows),
                'metrics': {
                    name: {
                        m: summaries[name][m] for m in metric_keys
                    }
                    for name in models.keys()
                },
                'per_image': {
                    fn: {name: rows[fn][name] for name in models.keys()}
                    for fn in rows
                },
            },
            f,
            indent=2,
            default=float,
        )
    print(f"Summary JSON: {json_path}")

    # ---- Interpretation guide ----
    print("\nMetric interpretation guide:")
    print("  PSNR:    >25 dB good,  >30 dB excellent   (higher is better)")
    print("  SSIM:    >0.85 good,   >0.95 excellent    (higher is better)")
    print("  LPIPS:   <0.25 good,   <0.1 excellent     (lower is better)")
    print("  NIQE:    <5.0 good,    <3.0 excellent     (lower is better)")
    print("  MANIQA:  >0.7 good,    >0.85 excellent    (higher is better)")
    print("  CLIPIQA: >0.7 good,    >0.85 excellent    (higher is better)")
    print("  TOPIQ:   >0.5 good,    >0.7 excellent     (higher is better)")
    print("  MUSIQ:   higher is better (range ~0-100)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
