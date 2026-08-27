# V7 Anime Finetune: Weight-Update Bug (2026-08-27)

Status: Bug confirmed systemic, root cause suspected but not fully diagnosed.
Impact: All v7 anime finetune runs produce checkpoints bit-identical to the warm-start. No actual training occurs.

## TL;DR

The v7 anime SPAN finetuner does not update model weights, despite the training loop reporting varying loss values per batch. Every v7 checkpoint in checkpoints/NEOSR_SPAN_V7_ANIME_001/ (epoch 2, 4, 6, 8, 10, 12, 14, latest, best) is bit-exact identical to the warm-start pretrained/span_pix_pretrain_4x.pth for both ema_state_dict and model_state_dict. Same hash: 07f84330c2d7779b (conv_1.sk.weight, MD5 prefix).

## Evidence (Empirical)

| Checkpoint | SHA256 (prefix) | vs Pretrain |
| --- | --- | --- |
| pretrained/span_pix_pretrain_4x.pth | 07f84330c2d7779b | baseline |
| checkpoints/NEOSR_SPAN_V7_ANIME_001/finetune_best.pth | 07f84330c2d7779b | identical |
| checkpoints/NEOSR_SPAN_V7_ANIME_001/finetune_latest.pth | 07f84330c2d7779b | identical |
| checkpoints/NEOSR_SPAN_V7_ANIME_001/finetune_epoch_2.pth | 07f84330c2d7779b | identical |
| checkpoints/NEOSR_SPAN_V7_ANIME_001/finetune_epoch_4.pth | 07f84330c2d7779b | identical |
| checkpoints/NEOSR_SPAN_V7_ANIME_001/finetune_epoch_6.pth | 07f84330c2d7779b | identical |
| checkpoints/NEOSR_SPAN_V7_ANIME_001/finetune_epoch_8.pth | 07f84330c2d7779b | identical |
| checkpoints/NEOSR_SPAN_V7_ANIME_001/finetune_epoch_10.pth | 07f84330c2d7779b | identical |
| checkpoints/NEOSR_SPAN_V7_ANIME_001/finetune_epoch_12.pth | 07f84330c2d7779b | identical |
| checkpoints/NEOSR_SPAN_V7_ANIME_001/finetune_epoch_14.pth | 07f84330c2d7779b | identical |
| checkpoints/finetune_best.pth (root, May 19 2026) | 869161edef7ec100 | different — old finetune trained OK |

A 3-epoch smoke run on local RTX 4000 8GB (finetune_neosr_span_v7_anime_smoke_tiny.yaml, 127 imgs/epoch, grad_accum=1) also produced ckpts bit-identical to pretrain despite per-batch loss varying normally (0.18 -> 1.46 -> 0.56 -> 0.09 -> 1.05 -> 0.64 -> 0.35).

## Where the trainer does work

- The loss is computed and varies batch-to-batch (forward + loss eval works).
- The discriminator backward at line 853 (d_loss.backward()) is the ONLY .backward() call in 1681 lines.
- The generator's self.scaler.scale(total_loss).backward() exists at line 1255.
- self.optimizer.step() is called at line 1300/1303.
- self.ema.update(self.model) is called at line 1314.
- self.save_checkpoint() correctly writes both model_state_dict and ema_state_dict.

## What is broken

- The optimizer's .step() either (a) does not actually mutate the model params, (b) all gradients are zero/NaN-suppressed before stepping, (c) something detaches the graph between sr and self.model.parameters().
- Both ema_state_dict AND model_state_dict end up identical to pretrain, so the bug is upstream of EMA — the live model itself never updates.
- gradient_accumulation_steps: 64 in the v7 config. With 63 batches/epoch the optimizer only steps once per epoch. With grad_accum=1 (smoke), the bug persists, ruling out grad-accum as the cause.
- 18% of params are frozen (eval_conv.weight/bias, rgb_mean), but 1.84M params remain trainable and would still produce non-zero updates with non-zero grads.

## Not yet ruled out

- _compute_total_loss might return a total that is detached from sr (would need to inspect each component loss for .detach() calls).
- Some loss might be NaN every batch, tripping the NaN-suppression path at line 1242 (Skipping batch and zeroing gradients) — but no NaN warning was printed during the smoke.
- The DINOv2 / FDL forward pass on fdl_compute_every=4 might short-circuit in a way that leaves the model detached.

## Next-step diagnostic (not yet run)

Add a one-line print at line 1255 in src/training/neosr_finetuner.py to log total.grad_fn and total.requires_grad, plus a one-line print at line 1303 to log the model conv_1.sk.weight delta before/after step.

## Impact on prior work

- The user's June 14, 2026 v7 finetune was effectively a no-op — 7 epochs x ~36s/epoch = 4 minutes of compute, zero learning.
- V8 Tier 1 TTA + souping work (already committed) still ships correctly — those tools operate on whatever checkpoint you point them at, and they will work once the trainer bug is fixed and real trained checkpoints exist.
- Soup of v7_001 checkpoints (already cleaned up from checkpoints/SOUP_V7_ANIME/) would have been pretrain x N = still pretrain. Deleted because it was misleading.
- The root checkpoints/finetune_best.pth (May 19, 2026, v3-architecture) is from an older trained run but uses feature_channels=32 so it does NOT load into the v7 (feature_channels=48) model.

## What does work

The May 19 root ckpt + the v7 trainer (when configured for a different architecture) proves the trainer CAN train. The bug is specific to the current v7 architecture / config combination.
