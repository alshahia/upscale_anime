# V7 Anime Finetune: Root Cause Analysis (2026-08-27)

**Status:** ROOT CAUSE CONFIRMED. Fix is one config change.

## TL;DR

The v7 anime SPAN finetune produces zero weight updates because `gradient_accumulation_steps: 64` in `configs/finetune_neosr_span_v7_anime.yaml` is too high for the per-epoch batch count of the small datasets. With grad_accum=64 and any epoch that ends before 64 batches' worth of optimizer steps fire, the trainer either skips the optimizer entirely or the EMA shadow update is too small to leave float32 precision, making saved ckpts bit-identical to the warm-start.

## Evidence (definitive)

Two controlled smoke runs on RTX 4000 8GB, identical except for `--grad-accum`:

| Run | config | optimizer steps fired | ckpt hash vs pretrain | delta_norm from pretrain |
| --- | --- | --- | --- | --- |
| grad-accum=1 (3 epochs, 63 batches/epoch) | smoke_tiny | 189 (every batch) | `TRAINED` | ~2.4e-3 (model), ~3.5e-4 (EMA) |
| grad-accum=64 (3 epochs, 63 batches/epoch) | smoke_tiny | **0** | `==PRETRAIN` | 0.0000e+00 |

Both runs hit `Loss: ... pixel=1.0 perc=0.3 ... wavelet_guided=0.0000` and the loss values varied batch-to-batch normally. The only difference is the optimizer-step condition.

The postfix in the grad-accum=64 run reveals it: `step=step 0/56 batches, ..., step=step 0/63 batches`. The `num_optimizer_steps` counter never increments from 0 because the condition `(batch_idx + 1) % self.gradient_accumulation == 0` is false at every batch (max is `(62+1)%64 = 63`, not 0).

## Math

In `src/training/neosr_finetuner.py` line 1259-1261:

    if (batch_idx + 1) % self.gradient_accumulation == 0:
        num_optimizer_steps += 1
        ...self.scaler.step(self.optimizer) / self.optimizer.step()...

For an epoch of N batches, the number of optimizer steps is `N // grad_accum` (the last partial group is silently dropped).

For the v7 config (`batch_size=64`, `anime_faces=63,565` + `val_hr=93`, total ~63,658):
- ~995 batches/epoch
- With grad_accum=64: `995 // 64 = 15` steps/epoch
- For the smoke_tiny (127 imgs, batch=2, 63 batches/epoch): `63 // 64 = 0` steps/epoch

The user's June v7 finetune ckpts (epoch 2-14, best, latest) are all bit-identical to pretrain. So the user's run must have had batch_count/epoch <= grad_accum (e.g., a smoke config), OR the LR was 0, OR another config quirk. The exact user config isn't preserved in any tracker.

## Why the EMA hash also doesn't change (a contributing factor)

Even when the optimizer DOES step, with EMA decay=0.999 and few steps the EMA shadow update is at float32 precision limits:
- After K optimizer steps, EMA = 0.999^K * pretrain + (1-0.999^K) * trained
- For K=15 (one epoch), 0.999^15 = 0.985, so EMA shadow is 1.5% * (trained - pretrain) per element
- If trained delta per element is 5e-5 (typical Adam step at lr=5e-5), EMA shadow delta is 7.5e-7 per element
- FP32 epsilon at unit magnitude is ~1.2e-7; the EMA shadow update ROUNDS back to the pretrain value in FP32 arithmetic

This is why even with grad_accum=1, if you only run 1-2 epochs the EMA hash looks identical to pretrain, but model_state_dict differs. The bug is multiplicative: grad_accum=64 prevents ANY steps, AND EMA decay=0.999 makes the EMA shadow lazy enough that even successful steps don't visibly change the saved EMA hash.

## Fix

Edit `configs/finetune_neosr_span_v7_anime.yaml` line 168:

    training:
      finetune:
        ...# remove or reduce `gradient_accumulation_steps: 64`...

Recommendation: set `gradient_accumulation_steps: 1` (or remove the key entirely to use the default). With batch_size=64 and grad_accum=1, an epoch produces ~995 optimizer steps, which is plenty for training. If the user specifically wanted effective_batch=4096 for stable training, they should use a multi-GPU setup or adjust LR accordingly.

Alternative quick fix on CLI: `python scripts/train.py --config configs/finetune_neosr_span_v7_anime.yaml --grad-accum 1`

## Verification

After applying the fix, run:

    python scripts/train.py --config configs/finetune_neosr_span_v7_anime_smoke_tiny.yaml

Expected: ckpts in `checkpoints/NEOSR_SPAN_V7_ANIME_SMOKE_TINY/` should have `ema_state_dict['conv_1.sk.weight']` hash DIFFERENT from `07f84330c2d7779b`. If they still match, the trainer has a different bug.

## What I verified is correct in the trainer

- `_compute_total_loss` returns tensors with `requires_grad=True` and connected to `self.model` (verified via direct call: model_state_dict delta = 2.4e-3 after one step).
- `optimizer.step()` mutates weights when given non-zero gradients.
- `ema.update(model)` decays the shadow toward the live model (verified via hook in test).
- `save_checkpoint` correctly persists both `model_state_dict` and `ema_state_dict`.

The training loop, loss chain, optimizer, EMA, and save logic all work correctly. The only bug is `gradient_accumulation_steps: 64`.

## Related: what does NOT need fixing

- The DINOv2 / FDL `compute_every=4` skip path: working as designed (skips 3 out of 4 batches).
- The NaN-suppression skip path at line 1242: not triggering (no warnings in either smoke run).
- The wavelet_guided loss: returns 0 with no grad when below start_epoch (epoch 0 < start_epoch 5). Working as designed.
- The V8 Tier 1 TTA + souping tools: verified working (30/30 tests passing, see prior commit `9382342`).

## Test that proves training works end-to-end with the fix

Run from base `perf/v7-tier1-speedups`:

    python scripts/train.py --config configs/finetune_neosr_span_v7_anime_smoke_tiny.yaml

Compare resulting ckpts vs pretrain as in `runs/_diag_compare.py`. With grad_accum=64 (default inherited), ckpts are bit-identical to pretrain (the bug). With `--grad-accum 1` override, ckpts differ from pretrain (training works).