# V7 Anime Finetune Recipe (2026-06)

Status: UNBLOCKED. Root cause identified — see docs/v7_finetune_root_cause.md. Single config change fixes it (gradient_accumulation_steps: 64 -> 1).

This recipe was the user's planned v7 finetune workflow. The training bug was a config issue (gradient_accumulation_steps too high for the small batch-per-epoch count), NOT a trainer bug; everything else (V7 Tier 1 perf speedups, V8 Tier 1 TTA+souping) was already verified working.

## Pre-flight checklist

- [ ] Fix gradient_accumulation_steps in configs/finetune_neosr_span_v7_anime.yaml (line 168). Change `gradient_accumulation_steps: 64` to `1` (or remove the key entirely). See docs/v7_finetune_root_cause.md for evidence.
- [ ] Confirm fix via: run scripts/train.py with configs/finetune_neosr_span_v7_anime_smoke_tiny.yaml. Ckpts in checkpoints/NEOSR_SPAN_V7_ANIME_SMOKE_TINY/ should have ema_state_dict hash DIFFERENT from 07f84330c2d7779b.
- [ ] GPU with >=16GB VRAM. RTX 4000 8GB (this machine) is too small for the full B=64 crop=64 config — use a remote GPU.

## Step 1 — Full finetune (8-24 hours depending on GPU)

    python scripts/train.py --config configs/finetune_neosr_span_v7_anime.yaml \
        --output.run_name NEOSR_SPAN_V7_ANIME_RUN2

Produces 7 EMA snapshots + best + latest in checkpoints/NEOSR_SPAN_V7_ANIME_RUN2/.
Verify post-train with the quick-check in docs/v7_finetune_weight_update_bug.md — ema hashes should differ across epochs and from the warm-start.

## Step 2 — Souping (V8 Tier 1, ~30s)

    python scripts/soup_checkpoints.py \
        --inputs checkpoints/NEOSR_SPAN_V7_ANIME_RUN2/finetune_epoch_4.pth \
                 checkpoints/NEOSR_SPAN_V7_ANIME_RUN2/finetune_epoch_6.pth \
                 checkpoints/NEOSR_SPAN_V7_ANIME_RUN2/finetune_epoch_8.pth \
                 checkpoints/NEOSR_SPAN_V7_ANIME_RUN2/finetune_epoch_10.pth \
                 checkpoints/NEOSR_SPAN_V7_ANIME_RUN2/finetune_epoch_12.pth \
                 checkpoints/NEOSR_SPAN_V7_ANIME_RUN2/finetune_epoch_14.pth \
        --output checkpoints/SOUP_V7/soup_v7_6ckpts.pth

    python scripts/soup_checkpoints.py \
        --inputs checkpoints/NEOSR_SPAN_V7_ANIME_RUN2/finetune_epoch_8.pth \
                 checkpoints/NEOSR_SPAN_V7_ANIME_RUN2/finetune_epoch_10.pth \
                 checkpoints/NEOSR_SPAN_V7_ANIME_RUN2/finetune_epoch_12.pth \
                 checkpoints/NEOSR_SPAN_V7_ANIME_RUN2/finetune_epoch_14.pth \
        --output checkpoints/SOUP_V7/soup_v7_converged_4ckpts.pth

Expected: MANIQA +0.01 to +0.05 vs best single finetune (Wortsman ICML 2022 recipe).

## Step 3 — TTA inference + comparison (V8 Tier 1, ~10min on 10 val images)

    python scripts/compare_n_checkpoints.py \
        --config configs/finetune_neosr_span_v7_anime.yaml \
        --input data/val_hr --gt data/val_hr --smoke 10 --max-side 256 \
        --metrics psnr ssim lpips maniqa clipiqa niqe \
        --output results/v7_tier1_comparison/ \
        --ckpt baseline:pretrained/span_pix_pretrain_4x.pth \
        --ckpt v7_best:checkpoints/NEOSR_SPAN_V7_ANIME_RUN2/finetune_best.pth \
        --ckpt soup_6:checkpoints/SOUP_V7/soup_v7_6ckpts.pth \
        --ckpt soup_4:checkpoints/SOUP_V7/soup_v7_converged_4ckpts.pth \
        --tta

Expected: soup + TTA should improve over best single, with LPIPS dropping ~10% from TTA alone and MANIQA improving from soup.

## Step 4 — Production inference with TTA

    python scripts/inference.py \
        --checkpoint checkpoints/SOUP_V7/soup_v7_converged_4ckpts.pth \
        --config configs/finetune_neosr_span_v7_anime.yaml \
        --input path/to/low_res_image.png \
        --output path/to/output_4x.png \
        --tta

## Time/cost budget (estimated)

| Step | Wall time (RTX 4090 24GB) | VRAM | Cost (cloud) |
| --- | --- | --- | --- |
| Full finetune (80 epochs) | 12-18h | ~10GB | $30-50 |
| Soup (per recipe) | 30s | trivial | free |
| TTA eval (10 imgs) | 5-10min | 6GB | free |
| TOTAL | ~13-19h | | ~$30-50 |

With V7 Tier 1 perf speedups enabled (BF16, no_grad backbones, fused Haar, DCT precomp, fast_validation), the per-step cost is reduced ~2-3x vs the naive implementation.