#!/usr/bin/env python3
"""Multi-checkpoint comparison harness.

Like `scripts/compare_checkpoints.py`, but supports N>=2 checkpoints in a
single run. Each `--ckpt label:path` pair adds one model to the eval matrix.
Per-image, per-checkpoint metrics are written to CSV and a summary table
is printed. NR metrics only by default; pass `--gt` for full-ref.

Usage:
    python scripts/compare_n_checkpoints.py \
        --config configs/finetune_neosr_span_v7_anime.yaml \
        --input data/val_hr --gt data/val_hr \
        --smoke 5 --max-side 256 \
        --output results/v7_soup_comparison/ \
        --metrics psnr ssim lpips maniqa clipiqa niqe \
        --ckpt baseline:pretrained/span_pix_pretrain_4x.pth \
        --ckpt v7_best:checkpoints/NEOSR_SPAN_V7_ANIME_001/finetune_best.pth \
        --ckpt v7_latest:checkpoints/NEOSR_SPAN_V7_ANIME_001/finetune_latest.pth \
        --ckpt soup_all:checkpoints/SOUP_V7_ANIME/soup_v7_all_7ckpts.pth \
        --ckpt soup_converged:checkpoints/SOUP_V7_ANIME/soup_v7_converged_4ckpts.pth
"""
import sys, csv, json, time, statistics, logging, argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
import cv2
import numpy as np
import torch
from tqdm import tqdm

from anime_sr.models.span import create_neosr_span, create_span_model
from anime_sr.utils.config import Config
from anime_sr.inference.tta import tta_forward as _tta_forward
from anime_sr.utils.metrics import (
    PYIQA_DIRECTION,
    calculate_clipiqa, calculate_lpips, calculate_maniqa,
    calculate_musiq, calculate_niqe, calculate_psnr,
    calculate_ssim, calculate_topiq,
)

logger = logging.getLogger(__name__)
NR_METRIC_FNS = {
    'niqe': calculate_niqe, 'maniqa': calculate_maniqa,
    'clipiqa': calculate_clipiqa, 'topiq_nr': calculate_topiq,
    'musiq': calculate_musiq,
}
DEFAULT_CONFIG = 'configs/finetune_neosr_span_v7_anime.yaml'

def load_one_ckpt(path, config_path, device):
    cfgp = Path(config_path)
    if cfgp.is_file():
        try:
            cfg = Config.load(str(cfgp))
            mcfg = cfg.get('model', {}) or {}
        except Exception as e:
            logger.warning(f'config load failed: {e}'); mcfg = {}
    else:
        logger.warning(f'config {config_path} not found; defaults'); mcfg = {'type':'neosr_span','upscale':4}
    model = create_neosr_span(mcfg) if mcfg.get('type') == 'neosr_span' else create_span_model(mcfg)
    try:
        ck = torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        logger.warning(f'weights_only=False failed: {e}; retrying weights_only=True')
        ck = torch.load(path, map_location=device, weights_only=True)
    state = None
    if isinstance(ck, dict):
        for k in ('ema_state_dict','model_state_dict','state_dict','params_ema','params','g_ema'):
            if k in ck and isinstance(ck[k], dict):
                state = ck[k]; break
        if state is None and all(isinstance(v, torch.Tensor) for v in ck.values()):
            state = ck
    else:
        state = ck
    if state is None: raise RuntimeError(f'no state_dict in {path}')
    miss, unexp = model.load_state_dict(state, strict=False)
    if miss: logger.warning(f'{Path(path).name}: {len(miss)} missing keys')
    if unexp: logger.warning(f'{Path(path).name}: {len(unexp)} unexpected keys')
    model = model.to(device).eval()
    epoch = ck.get('epoch') if isinstance(ck, dict) else None
    label_extra = f"_e{epoch}" if epoch is not None else ""
    return model, label_extra

def preprocess_image(image_path, scale=4, max_side=None):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None: raise ValueError(f'failed: {image_path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if max_side and max(h,w) > max_side:
        if h >= w: img = img[(h-max_side)//2:(h-max_side)//2+max_side, :, :]
        else: img = img[:, (w-max_side)//2:(w-max_side)//2+max_side, :]
        h,w = img.shape[:2]
        h2,w2 = (h//scale)*scale, (w//scale)*scale
        if h2!=h or w2!=w:
            img = img[(h-h2)//2:(h-h2)//2+h2, (w-w2)//2:(w-w2)//2+w2, :]
    t = torch.from_numpy(img.transpose(2,0,1)).float()/255.0
    return t.unsqueeze(0)

def postprocess(sr):
    return sr.squeeze(0).float().clamp(0,1).unsqueeze(0)

def infer(model, lr, device):
    lr = lr.to(device)
    use_amp = device.type == 'cuda'
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=use_amp):
            return model(lr)

def infer_tta(model, lr, device):
    lr = lr.to(device)
    return _tta_forward(lambda x: infer(model, x, device), lr)

def compute_metrics(sr, metric_names, gt=None):
    out = {}
    sr = sr.float()
    if gt is not None:
        gt = gt.float()
        if sr.shape != gt.shape:
            sr = torch.nn.functional.interpolate(sr, size=gt.shape[2:], mode='bicubic', align_corners=False)
    for m in metric_names:
        if m == 'psnr' and gt is not None:
            try: out['psnr'] = calculate_psnr(sr, gt)
            except Exception as e: out['psnr'] = float('nan')
        elif m == 'ssim' and gt is not None:
            try: out['ssim'] = calculate_ssim(sr, gt)
            except Exception as e: out['ssim'] = float('nan')
        elif m == 'lpips' and gt is not None:
            try: out['lpips'] = calculate_lpips(sr, gt)
            except Exception as e: out['lpips'] = float('nan')
        elif m in NR_METRIC_FNS:
            try: out[m] = NR_METRIC_FNS[m](sr)
            except Exception as e: out[m] = float('nan')
        elif m in ('psnr','ssim','lpips') and gt is None:
            continue
        else:
            logger.warning(f'unknown metric: {m}')
    return out

def summary(values):
    finite = [v for v in values if v is not None and not (isinstance(v,float) and np.isnan(v))]
    if not finite: return {'mean':float('nan'),'median':float('nan'),'std':0,'min':float('nan'),'max':float('nan'),'count':0}
    return {'mean':float(np.mean(finite)),'median':float(np.median(finite)),'std':float(np.std(finite)) if len(finite)>1 else 0.0,'min':float(np.min(finite)),'max':float(np.max(finite)),'count':len(finite)}

def direction(metric):
    if metric in PYIQA_DIRECTION: return PYIQA_DIRECTION[metric]
    if metric == 'lpips': return 'lower'
    if metric in ('psnr','ssim'): return 'higher'
    return 'n/a'

def main():
    ap = argparse.ArgumentParser(description='Compare N SR checkpoints.')
    ap.add_argument('--ckpt', action='append', required=True, help='label:path (repeatable)')
    ap.add_argument('--config', default=DEFAULT_CONFIG)
    ap.add_argument('--input', required=True)
    ap.add_argument('--gt', default=None)
    ap.add_argument('--output', default='results/v7_soup_comparison/')
    ap.add_argument('--metrics', nargs='+', default=['psnr','ssim','lpips','maniqa','clipiqa','niqe'])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--scale', type=int, default=4)
    ap.add_argument('--smoke', type=int, default=None)
    ap.add_argument('--max-side', type=int, default=None)
    ap.add_argument('--tta', action='store_true')
    args = ap.parse_args()

    if args.tta:
        print('[TTA] 8x D4 ensemble enabled (8x inference cost)', flush=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    # Parse --ckpt label:path pairs
    ckpts = []
    for spec in args.ckpt:
        if ':' not in spec:
            print(f'Error: --ckpt must be label:path, got: {spec}'); return 1
        label, path = spec.split(':', 1)
        if not Path(path).is_file():
            print(f'Error: ckpt not found: {path}'); return 1
        ckpts.append((label, path))

    input_path = Path(args.input)
    if not input_path.is_dir():
        print(f'Error: --input must be a directory: {input_path}'); return 1
    gt_path = Path(args.gt) if args.gt else None
    if gt_path and not gt_path.is_dir():
        print(f'Warning: --gt not a dir, disabling FR'); gt_path = None

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu')
    print(f'Device: {device}', flush=True)
    print(f'Checkpoints: {len(ckpts)}', flush=True)
    for label, path in ckpts:
        print(f'  {label}: {path}', flush=True)

    exts = {'.png','.jpg','.jpeg','.bmp','.tiff','.webp'}
    files = sorted(f for f in input_path.iterdir() if f.suffix.lower() in exts and f.is_file())
    if args.smoke: files = files[:args.smoke]
    print(f'Images: {len(files)}', flush=True)

    # Load all models
    models = {}
    for label, path in ckpts:
        print(f'Loading {label} from {path}...', flush=True)
        try:
            m, _ = load_one_ckpt(path, args.config, device)
            models[label] = m
        except Exception as e:
            logger.error(f'Could not load {label} ({path}): {e}')
            return 2

    metric_keys = []
    for m in args.metrics:
        if m in ('psnr','ssim','lpips') and gt_path is None: continue
        if m not in metric_keys: metric_keys.append(m)

    rows = {}
    print(f'\nProcessing {len(files)} image(s) x {len(models)} checkpoint(s)...', flush=True)
    t0 = time.time()
    for img_path in tqdm(files, desc='Images'):
        try:
            lr = preprocess_image(img_path, scale=args.scale, max_side=args.max_side)
        except Exception as e:
            logger.warning(f'skip {img_path.name}: {e}'); continue
        gt = None
        if gt_path:
            gtf = gt_path / img_path.name
            if gtf.is_file():
                try: gt = preprocess_image(gtf, scale=args.scale, max_side=args.max_side).to(device)
                except Exception as e: logger.warning(f'GT load failed: {e}')
        rows[img_path.name] = {}
        for label, model in models.items():
            try:
                sr = infer_tta(model, lr, device) if args.tta else infer(model, lr, device)
                sr_pp = postprocess(sr)
                met = compute_metrics(sr_pp, args.metrics, gt)
            except Exception as e:
                logger.warning(f'{label} infer fail on {img_path.name}: {e}')
                met = {m: float('nan') for m in metric_keys}
            rows[img_path.name][label] = met
    elapsed = time.time() - t0
    print(f'\nDone in {elapsed:.1f}s ({elapsed/max(1,len(files)):.2f}s/img)', flush=True)

    # CSV
    csv_path = output_dir / 'comparison.csv'
    fields = ['filename']
    for label in models.keys():
        for m in metric_keys:
            fields.append(f'{label}__{m}')
    with open(csv_path,'w',newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for fn, ck in rows.items():
            r = {'filename': fn}
            for label, md in ck.items():
                for m in metric_keys:
                    r[f'{label}__{m}'] = md.get(m, float('nan'))
            writer.writerow(r)
    print(f'CSV: {csv_path}', flush=True)

    # Summary
    print('\n' + '='*80)
    print('MULTI-CHECKPOINT COMPARISON SUMMARY')
    print('='*80)
    hdr = f"  {'CHECKPOINT':<22} {'METRIC':<10} {'N':>4} {'MEAN':>10} {'STD':>10} {'MIN':>10} {'MAX':>10}  DIR"
    print(hdr); print('  ' + '-'*(len(hdr)-2))
    sums = {}
    for label in models.keys():
        sums[label] = {}
        for m in metric_keys:
            vals = [rows[fn][label].get(m, float('nan')) for fn in rows if label in rows[fn]]
            sums[label][m] = summary(vals)
    for label in models.keys():
        for m in metric_keys:
            s = sums[label][m]
            print(f"  {label:<22} {m.upper():<10} {s['count']:>4d} {s['mean']:>10.4f} {s['std']:>10.4f} {s['min']:>10.4f} {s['max']:>10.4f}  ({direction(m)} is better)")
        print()

    # JSON
    json_path = output_dir / 'comparison_summary.json'
    with open(json_path,'w') as f:
        json.dump({'checkpoints':{l:{'path':p} for l,p in ckpts},'n_images':len(rows),'metrics':{l:sums[l] for l in models.keys()},'per_image':{fn:{l:rows[fn][l] for l in models.keys()} for fn in rows},'tta':args.tta,'elapsed_s':elapsed}, f, indent=2, default=float)
    print(f'JSON: {json_path}', flush=True)
    print('\nMetric interpretation: PSNR>30 excellent, SSIM>0.95 excellent, LPIPS<0.1 excellent, NIQE<3 excellent, MANIQA>0.85 excellent, CLIPIQA>0.85 excellent')
    return 0

if __name__ == '__main__':
    sys.exit(main())
