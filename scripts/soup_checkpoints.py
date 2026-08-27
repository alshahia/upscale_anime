#!/usr/bin/env python3
"""Soup two (or more) checkpoint EMA state-dicts.

Averages `ema_state_dict` from multiple fine-tuned checkpoints into a single
"soup" checkpoint. Per Wortsman et al. (ICML 2022), this often beats the
best single finetune when the checkpoints share a basin (same warm-start,
similar data).

Usage:
    python scripts/soup_checkpoints.py \
        --inputs checkpoints/NEOSR_SPAN_V6_ANIME/finetune_best.pth \
                 checkpoints/NEOSR_SPAN_V7_ANIME/finetune_best.pth \
        --output checkpoints/SOUP_V6_V7/soup.pth
"""
import argparse
import sys
import time
from pathlib import Path

import torch


def load_state_dict(checkpoint_path: str) -> tuple:
    """Load a checkpoint and pick the best state-dict.

    Preference order:
      1. ema_state_dict (if present)
      2. model_state_dict
      3. params (Phhofm/neosr convention)
      4. state_dict
      5. raw checkpoint (if dict-shaped)

    Returns: (state_dict, source_name, full_checkpoint)
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except Exception:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint is not a dict: {type(ckpt)}")

    for key in ['ema_state_dict', 'model_state_dict', 'params', 'state_dict']:
        if key in ckpt:
            return ckpt[key], key, ckpt

    # Assume the whole checkpoint IS the state_dict.
    return ckpt, '<raw>', ckpt


def soup_checkpoints(
    input_paths: list,
    output_path: str,
    use_ema: bool = True,
    greedy: bool = False,
    greedy_eval_dir: str = None,
) -> dict:
    """Average state-dicts from multiple checkpoints.

    Args:
        input_paths: list of .pth paths to soup.
        output_path: where to write the soup.
        use_ema: prefer ema_state_dict; fall back to model_state_dict if absent.
        greedy: placeholder for future greedy-soup driver.
        greedy_eval_dir: placeholder for future greedy-soup driver.

    Returns:
        dict with keys: keys, output, sources, input_paths, epoch, dropped_keys.
    """
    if len(input_paths) < 2:
        raise ValueError(f"Need >= 2 checkpoints to soup, got {len(input_paths)}")

    del greedy, greedy_eval_dir  # reserved for future work

    states = []
    sources = []
    full_ckpts = []
    for p in input_paths:
        sd, src, full = load_state_dict(p)
        states.append(sd)
        sources.append(src)
        full_ckpts.append(full)

    # Verify all state-dicts have the same keys.
    key_sets = [set(sd.keys()) for sd in states]
    ref_keys = key_sets[0]
    for i, ks in enumerate(key_sets[1:], 1):
        if ks != ref_keys:
            missing = ref_keys - ks
            extra = ks - ref_keys
            raise ValueError(
                f"State-dict key mismatch between {input_paths[0]} and {input_paths[i]}: "
                f"missing={list(missing)[:5]}, extra={list(extra)[:5]}"
            )

    # Average.
    soup_sd = {}
    for k in ref_keys:
        ref_dtype = states[0][k].dtype
        accum = torch.zeros_like(states[0][k], dtype=torch.float32)
        for sd in states:
            accum = accum + sd[k].float()
        soup_sd[k] = (accum / len(states)).to(ref_dtype)

    # Build the output checkpoint. Preserve metadata from the most-recent
    # (highest epoch) input if available.
    epochs = [fc.get('epoch', 0) if isinstance(fc, dict) else 0 for fc in full_ckpts]
    max_epoch_idx = max(range(len(epochs)), key=lambda i: epochs[i])
    out = {
        'ema_state_dict': soup_sd,
        'epoch': epochs[max_epoch_idx],
        'soup_inputs': input_paths,
        'soup_sources': sources,
        'soup_n': len(input_paths),
        'soup_timestamp': int(time.time()),
    }
    if isinstance(full_ckpts[max_epoch_idx], dict) and 'config' in full_ckpts[max_epoch_idx]:
        out['config'] = full_ckpts[max_epoch_idx]['config']

    # Explicitly NOT carried over (per AGENTS.md "checkpoint souping" pattern):
    #   - scaler_state_dict (GradScaler is per-checkpoint)
    #   - optimizer_state_dict (Adam moments are per-checkpoint)
    # The user must re-init these on resume.

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output_path)

    return {
        'keys': len(soup_sd),
        'output': output_path,
        'sources': sources,
        'input_paths': input_paths,
        'epoch': epochs[max_epoch_idx],
        'dropped_keys': ['scaler_state_dict', 'optimizer_state_dict'],
    }


def main():
    ap = argparse.ArgumentParser(
        description='Soup two or more fine-tuned checkpoint EMA state-dicts.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument('--inputs', nargs='+', required=True,
                   help='Two or more checkpoint paths to soup.')
    ap.add_argument('--output', required=True,
                   help='Output path for the souped checkpoint.')
    ap.add_argument('--no-ema', action='store_true',
                   help='Use model_state_dict instead of ema_state_dict.')
    args = ap.parse_args()

    info = soup_checkpoints(args.inputs, args.output, use_ema=not args.no_ema)

    print(f"[Soup] Averaged {info['keys']} keys from {len(args.inputs)} checkpoints")
    print(f"[Soup] Source keys used: {info['sources']}")
    print(f"[Soup] Dropped: {info['dropped_keys']} (re-init on resume)")
    print(f"[Soup] Output epoch: {info['epoch']}")
    print(f"[Soup] Written to: {info['output']}")


if __name__ == '__main__':
    main()
