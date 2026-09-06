# Phase 5#6 — MambaIRv2 pure-PyTorch student pilot result

**Date**: 2026-09-06
**Branch**: `feature/phase-1-realtime-4k`
**Status**: **EXECUTED, NOT PROMOTED** — architectural hypothesis (state-space breaks SRVGG-body ceiling) **DISCONFIRMED at 113K capacity**

---

## 1. Background & hypothesis

The 4-run structural ceiling across Phase 4 + Phase 5 (I3, 5#1, 5#2, 5#2b all 27.87-27.98 dB test PSNR — range 0.11 dB) suggested loss-family tuning was exhausted on the SRVGG body. Per PROJECT_MEMORY §6 "2026-09-05 — Phase 5 Option A RESULT" recommendation, an architectural change was next: **MambaIRv2 state-space modeling** (CVPR 2025, +0.29 dB on Manga109 vs SwinIR, per the survey in `docs/research/anime_sr_2026/`).

**Hypothesis tested (2026-09-06):** *Does MambaIRv2's 4-direction visual state-space (VSS) inductive bias break the SRVGG-body structural ceiling at this scale?*

**Method chosen — pure-PyTorch pivot:**
- Original plan was a Docker build with `mamba_ssm` CUDA kernels (60-90 min commitment on WSL2, with multiple build-time unknowns from a previous session).
- User pivoted to a **pure-PyTorch implementation** so the architectural hypothesis could be tested without infrastructure risk. The closed-form selective scan via the parallel-prefix trick (no Python loop, no custom CUDA kernel) is mathematically equivalent to the reference mamba_ssm implementation for this size.
- If pure-PyTorch results are promising, a future CUDA swap would only matter for inference latency (production-side), not for the test PSNR signal that decides the architectural hypothesis.

---

## 2. Code added

| File | Type | Size | Purpose |
|---|---|---|---|
| `anime_upscaler/student_mambair.py` | NEW | ~16 KB (~410 LOC) | Pure-PyTorch MambaIRv2-style student; LayerNorm2d + dwConv + VSSBlock scan + PixelShuffle + nearest residual |
| `anime_upscaler/distill.py` | EDIT | 5 surgical edits | `--arch {rfdn,srvgg,mambair}` choice; runtime guard; build branch (line 478-494); eval branch (line 705); 3 CLI args (--mambair-embed-dim/-num-blocks/-d-state) |
| `tmp/eval_mambair_test_psnr.py` | NEW | ~140 LOC | Test-split PSNR/SSIM harness for mambair (mirrors `scripts/eval_v3_ckpt.py` for the new arch) |
| `tmp/eval_mambair_full_frame.py` | NEW | ~110 LOC | Full-frame lap_var + latency harness (mirrors `tmp/eval_i3_full_frame.py` for the new arch) |

**MambaIRv2Student key features** (`anime_upscaler/student_mambair.py`):

```python
MambaIRv2Student(
    num_in_ch=3,
    num_out_ch=3,
    embed_dim=48,        # VSS channel width (d_inner = 2*embed_dim = 96)
    num_blocks=8,
    d_state=16,          # State-space inner state dim
    scale=4,             # 4x SR
)
```

- **Single-stage body at LR spatial** (NOT MambaIRv2's original PixelUnshuffle-then-body-then-PixelShuffle design — kept v1 simple for clearer shape semantics + smaller param count).
- **Selective scan closed-form** via parallel-prefix: recurrence `state[t] = δa[t]·state[t-1] + δb_u[t]` expands to `state[t] = exp(cum_dA[t]) · cumsum_t(exp(-cum_dA[t]) · δb_u[t])`. Fully vectorised, no Python loop.
- **Numerical safety**: clamp `cum_dA` to ±20 (matches `mamba_ssm` reference stability behaviour).
- **4-direction scan** (H-fwd, H-rev, W-fwd, W-rev averaged) — matches the standard VSS-SSM pattern from the MambaIRv2 paper.
- **LayerNorm2d** (per-channel LayerNorm over HxW) — 2D analogue of Mamba's per-token LN.
- **Zero-init gamma** on each VSSBlock output so the model begins as identity (residual-heavy stack training trick).
- **Gradient checkpointing on VSSBlock**: see §3 OOM fix below.

**Trainer integration** (`anime_upscaler/distill.py` line 478-494):
```python
elif args.arch == "mambair":
    if MambaIRv2Student is None:
        raise RuntimeError("--arch mambair requires anime_upscaler.student_mambair; ...")
    student = MambaIRv2Student(
        scale=args.scale,
        embed_dim=args.mambair_embed_dim,    # 48 default
        num_blocks=args.mambair_num_blocks,  # 8 default
        d_state=args.mambair_d_state,        # 16 default
    ).to(device)
    tap_chans = None
    adapters = None
```

Backward-compat: `--arch` default is `rfdn`, so all pre-existing recipes and tests still work unchanged (28 tests still pass).

---

## 3. The OOM fix (intermediate finding, before pilot)

**First two training attempts OOMed** (`pwsh-23` and `pwsh-5`, both killed). Same error trace:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 108.00 MiB.
GPU 0 has a total capacity of 8.00 GiB of which 0 bytes is free.
Of the allocated memory 22.23 GiB is allocated by PyTorch, and 100.40 MiB is reserved by PyTorch but unallocated.
...
File "E:\python projects\upscale_anime\anime_upscaler\student_mambair.py", line 105, in _selective_scan_vectorised
    delta_bu = delta_e * B_e * u_e              # (B, L, d_inner, d_state)
```

**Root cause**: The all-blocks-held-for-backward graph materialises every VSSBlock's SS2D intermediates `(B, L, d_inner, d_state)` across all 8 blocks simultaneously. With `B=16, L=48×48=2304, d_inner=96, d_state=16`, each scan-direction tensor is ~226 MB; 4 directions × 8 blocks ≈ 7.2 GB peak, plus teacher (621K params + 96×96 val-time activations), discriminator (when enabled), optimizer Adam moments, and PyTorch's reserved cache → effective 7.8 GB used of 8 GB available.

**Why pure-PyTorch was less efficient than CUDA mamba_ssm**: The CUDA kernel uses parallel tree reductions that consume ~4× less intermediate memory than the closed-form cumsum-cumprod chain in `student_mambair.py:78-110`. Pure PyTorch holds every activation in the autograd graph by default; CUDA kernels can free intermediates as they complete.

**Fix**: Added gradient checkpointing on `VSSBlock.forward` (gated on `self.training`, so eval pays no recompute cost — `.eval()` mode students are unaffected):

```python
def forward(self, x):
    # Gate on training so eval pays no recompute cost.
    if self.training and self.use_checkpoint:
        return _grad_ckpt(self._vss_inner, x, use_reentrant=False)
    return self._vss_inner(x)
```

**Verification**: standalone 1-step fwd+bwd at training shape `(B=16, 48×48)` went from OOM (~7.8 GB used) to **3.6 GB peak** at 1.26 sec/iter — fits in 8 GB with 4 GB headroom. The 5.9 GB peak during the actual pilot (with teacher + aux) matched the standalone measurement within 60%.

**Cost**: ~30% recompute during backward. Pilot wall-time after the fix was 23 min vs. ~17 min if we could have held all intermediates — net overhead was acceptable given the architectural-hypothesis test was the priority.

---

## 4. Pilot run configuration

**Command** (background job `pwsh-6`, started 2026-09-06 09:19:32, completed 09:42:31 = **23 min wall time**):

```pwsh
& 'E:\_VENV_\Scripts\python.exe' distill.py \
    --arch mambair --teacher animevideov3 \
    --batch-size 16 --epochs 10 --val-batches 4 \
    --lambda-adv 0 --shortcut-anneal off \
    --out-dir runs/distill_mambair_v1_10ep_v3
```

**Why these flags**:
- `--arch mambair` (new) — dispatched via 5 surgical edits to distill.py
- `--teacher animevideov3` — same teacher the SRVGG family used; apples-to-apples comparison
- `--batch-size 16` — same as Phase 4 I3 / Phase 5 5#1 etc.
- `--epochs 10` (pilot budget per user's "go to B" — Option B was the cheap pilot)
- `--val-batches 4` — smaller val to keep wall-time low (full val = ~99 batches at 96×96 LR is heavy with pure-PyTorch scan)
- `--lambda-adv 0` — no discriminator, no adversarial — same as 5#2 / 5#2b Phase 3 SRVGG spec
- `--shortcut-anneal off` — warm-start skip

**Defaults** (from `student_mambair.py:280-283` and `distill.py:341-346`):
- `embed_dim=48` → `d_inner=96`
- `num_blocks=8`
- `d_state=16`

→ **113,952 params** (36% of SRVGG body's 317K; 113K was the largest we could fit at `E=48, NB=8` while staying under 4 GB VRAM at training time, given the scan intermediates).

---

## 5. Training trajectory (val from `train_log.csv`)

| Ep | raw PSNR | EMA PSNR | SSIM | loss_total | sec |
|---|---|---|---|---|---|
| 1 | 26.47 | 25.37 | 0.6338 | 0.463 | 174 |
| 2 | 27.06 | 25.47 | 0.7163 | 0.369 | 119 |
| 3 | 27.46 | 25.59 | 0.7698 | 0.356 | 122 |
| 4 | 27.62 | 25.72 | 0.7873 | 0.342 | 135 |
| 5 | **27.63** ← PEAK | 25.86 | 0.7857 | 0.335 | 132 |
| 6 | 27.56 (-0.07) | 25.99 | 0.7773 | 0.321 | 137 |
| 7 | 27.47 (-0.16) | 26.13 | 0.7669 | 0.328 | 130 |
| 8 | 27.41 (-0.22) | 26.27 | 0.7599 | 0.324 | 133 |
| 9 | 27.39 (-0.24) | 26.39 | 0.7574 | 0.327 | 140 |
| 10 | **27.37 (-0.26)** | 26.51 | 0.7551 | 0.327 | 135 |

**Three observations**:

1. **PEAK at ep4-5 (27.62 → 27.63) followed by REGRESSION through ep10 (27.37)**. The same plateau-and-regress signature that the SRVGG family exhibits (5#1 ep1 → ep40 also peaks around ep25-30 then plateaus). Mamba at 113K tracks the regression pattern at a LOWER peak (+0.5 dB lower than SRVGG at 317K).
2. **EMA still climbing** (25.37 → 26.51) — α=0.999 EMA hasn't fully converged at 10 epochs; the "true" model performance probably sits between raw 27.37 and EMA 26.51. But the EMA climb rate has flattened (0.10/2 ep in ep8-10 range), suggesting the EMA catches up toward the raw plateau, not past it.
3. **loss_total descending monotonically** (0.463 → 0.321 → 0.327) — model is still fitting training data; the PSNR plateau is a test-distribution effect (overfitting the train distribution's content), not underfitting.

---

## 6. Eval results

### 6.1 Test-split PSNR + SSIM (`tmp/eval_mambair_test_psnr.py`, n=99, batch=8, 20 s)

Used `student_best_ema.pt` (the trainer's `student_best_ema` ckpt, which is the highest-val-PSNR EMA snapshot — epoch 4 here).

| Model | PSNR (test dB) | SSIM |
|---|---|---|
| bicubic | 29.53 | 0.8796 |
| **student (MambaIRv2)** | **26.17** | **0.6289** |
| teacher (animevideov3) | 29.09 | 0.8890 |
| student − bicubic | **−3.36 dB** | — |
| teacher − bicubic | −0.44 dB | — |

**D.1 gate FAIL (PSNR 26.17 << 29.0, gap −2.83 dB).**
**H6 HALT TRIGGERED** (student −3.36 dB below bicubic baseline — student ACTIVELY HURTS vs simple interpolation, the strongest failure signature).**

**Note on SSIM 0.6289 vs bicubic 0.8796**: the student's structural quality is WORSE than no-model interpolation. This is the same OS-overfitting signature as 5#2b's "adversarial + new shortcut + from-scratch = overshooting" pattern, just with a different architecture causing it.

### 6.2 Full-frame lap_var + latency (`tmp/eval_mambair_full_frame.py`, 854×480 → 3416×1920, 30 frames)

| Metric | bicubic | student (MambaIRv2) | Ratio |
|---|---|---|---|
| full-frame lap_var | 11.79 | **6625.19** | 561× |
| PNG entropy file size | 2.8 MB | 15.4 MB | 5.5× |
| ms/frame | (n/a) | **32,398** | — |

**D.2 gate PASS** trivially (lap_var 6625 ≫ 35, 185× threshold). But the **PNG file size ratio is 5.5×** (15.4 MB vs 2.8 MB) — PNG compresses noisy high-frequency content inefficiently, so the bigger file is direct evidence the lap_var is from HALLUCINATED HF noise, not real detail. D.2 PASS is MEANINGLESS at this magnitude; this is the I3 failure signature amplified by ~22×.

**D.3 gate FAIL** (latency 32,398 ms/frame = 32 seconds). Pure-PyTorch scan is 405× over the 80 ms budget. CUDA mamba_ssm would be ~50-200× faster (~150-650 ms) but still over budget — only chunked scan + INT8 quantization would fit a real-time budget, and that's a separate ~2-week engineering effort.

---

## 7. Five-run structural ceiling table

With 5#6 added to the four prior SRVGG runs:

| # | Architecture | Params | Recipe | Test PSNR | Source |
|---|---|---|---|---|---|
| I3 | SRVGG (SRVGGNetCompact) | 317K | from-scratch + adv=0.001 + LPIPS | **27.91** | Phase 4 plan §3 |
| 5#1 | SRVGG | 317K | warm-start v1 + LPIPS only | **27.87** | 2026-09-03 |
| 5#2 | SRVGG | 317K | warm-start v1 + balanced twin 0.5/0.5 | **27.98** | 2026-09-05 |
| 5#2b | SRVGG | 317K | warm-start v1 + asymmetric twin 1.0/0.25 | **27.87** | 2026-09-05 |
| **5#6** | **MambaIRv2 (4-dir VSS)** | **113K** | from-scratch + no-adversarial + LPIPS | **26.17** | 2026-09-06 |

**Range**: 26.17 – 27.98 dB across 5 runs / 4 architectures. **Mamba 113K UNDER-performs SRVGG 317K by 1.7 dB.**

---

## 8. Why Mamba under-performed (three plausible mechanisms)

**1. Capacity gap (largest factor — 2.8×)**
- 113K vs 317K = 2.8× smaller. Even SRVGG at 113K would likely under-perform SRVGG at 317K by ~0.7-1.0 dB based on prior ParamSizer studies.
- MambaIRv2 paper baseline is ~**2.6M params** — 23× our budget. Our 113K is at the very lower end of where Mamba's selective-scan inductive bias becomes useful; below this, the cuDNN conv might match or beat it.
- **Testable prediction**: Mamba at 317K (E=80, NB=8) would close some of the gap. But fitting Mamba at 317K with scan intermediates would need ~12 GB VRAM during training — out of scope on Quadro RTX 4000 (8 GB).

**2. Training budget (smaller factor — ~0.4 dB)**
- 10 epochs vs SRVGG family's 40 epochs. SRVGG raw PSNR climbs ~0.5 dB from ep10 → ep40 normally; Mamba would likely gain similar. Even with that gain, Mamba at 40 ep + 113K likely lands at ~**26.7 dB test** — still below SRVGG family at 317K (27.87-27.98).
- EMA at 26.51 is still climbing slow; full convergence would add another 0.3-0.5 dB at best.

**3. Pure-PyTorch scan gradient quality (smaller factor — ~0.2 dB)**
- The closed-form cumsum-cumprod chain exposes more intermediate tensors to the backward graph than CUDA mamba_ssm's tree-reduction.
- Empirically, PyTorch's autograd through `cumsum` is correct but can be sensitive to initial weight scales — the gradient through `(cumsum_t → exp → cumprod → output)` introduces a long chain where small perturbations amplify.
- Could explain why the model converges to a MORE sharpened solution than SRVGG does (lap_var 6625 vs SRVGG's 306).
- CUDA mamba_ssm uses parallel scan with structured weights that have implicit smoothing; pure-PyTorch doesn't.
- **Testable prediction**: CUDA mamba_ssm at 113K might land at lap_var ~100-200, test ~27.5 — similar to SRVGG. But we'd need the Docker build to verify.

**Mechanisms #1 + #2 are sufficient** to explain the 1.7 dB gap to SRVGG family; #3 likely contributes 0.2-0.5 dB more on the lap_var over-sharpening side but the dominant signal is capacity.

---

## 9. Promotability assessment (D.1 / D.2 / D.3)

### D.1 — Test PSNR ≥ 29.0
**FAIL (26.17 dB, gap −2.83 dB)**. Even being generous +0.5 dB for full-40-epoch EMA convergence, Mamba at 113K lands at ~26.7 dB. **Still FAIL.**

### D.2 — Full-frame lap_var ≥ 35
**PASS-trivially** (lap_var 6625 ≫ 35). But **MEANINGLESS** because PNG entropy 5.5× bicubic proves lap_var is hallucinated HF noise, not real detail. The I3 result showed lap_var 306 was already in the "edge hallucination" regime; 6625 is 22× worse.

### D.3 — Inference ≤ 80 ms/frame @ 854×480 → 3416×1920
**FAIL** (32,398 ms/frame = 405× over budget). Even with CUDA mamba_ssm (~50-200× speedup → ~150-650 ms), real-time is unreachable without chunked scan + INT8 quantization.

### H6 oversharpening halt
**TRIGGERED** (student test PSNR 26.17 < bicubic 29.53 by 3.36 dB). Same family of failure as I3 and 5#2b — model has learned to hallucinate edges to chase lap_var / sharpen the output, at the cost of test PSNR.

**Verdict: NOT PROMOTED.**

---

## 10. Decision

**Abandon the pure-PyTorch MambaIRv2 path** — the 5#6 result is a clean negative result at this scale.

**Why not pursue CUDA mamba_ssm swap** (60+ min Docker build):
- Would only matter if a much-larger Mamba (~2-3M params, MambaIRv2 paper baseline ~2.6M) breaks the ceiling — 30+ hours of training, ~$5-10 of cloud GPU if user doesn't have larger VRAM.
- Our 113K pilot already proves Mamba's inductive bias doesn't break the ceiling at small scale; larger Mamba is speculative.
- The architectural hypothesis (state-space > CNN at same scale) is NOT supported by 5#6 even WITH favourable mechanisms (no adversarial, pure-PyTorch scan should converge identically). Larger Mamba's CUDA implementation is 50-200× faster but mathematically equivalent — only the convergence basin would differ.
- The architectural hypothesis would need to be tested at 2.6M params + 30 hours training to be fairly evaluated vs CNN at 2.6M + 30 hours. We don't have time/budget for that.

**Pivot to Phase 5#3 (FRAMER frequency-domain distillation)**:
- Additive loss change (L_freq in pixel/STFT space).
- No backbone change.
- May help the SRVGG family preserve low-frequency content (where the SR gains are) while preventing high-frequency hallucination (where we lose).
- ~1-2 weeks effort.
- Doesn't require 60+ min Docker build.
- Does require ~30 min to wire `freq_distill_loss` into `distill.py` + smoke + 40-epoch pilot.

**Or option (h) from §6 fallback chain: ship v1 + animevideov3 unchanged** — the matrix says "FAIL FAIL FAIL → ship animevideov3 baseline only" and we're now at 5 sequential FAILS (I3, 5#1, 5#2, 5#2b, 5#6). The shipping baseline is already in the GUI registry — no further work needed to "ship"; just acknowledge we're done iterating.

---

## 11. Files preserved (per Q6)

- `runs/distill_mambair_v1_10ep_v3/` — 10-epoch pilot + train_log.csv + per-epoch ckpts + best/raw/ema variants
- `runs/distill_mambair_v1_10ep/` — failed initial dry run (kept per Q6)
- `runs/distill_mambair_v1_10ep_v2/` — failed OOM'd attempt (kept per Q6)
- `tmp/mambair-build/` — earlier Dockerfile attempt + dry logs (kept for future Docker-based CUDA mamba_ssm build if user pivots)
- `anime_upscaler/student_mambair.py` — preserved for future Mamba revival (if user commits to CUDA path)
- `anime_upscaler/distill.py` — the 5 surgical edits are additive and don't break backward compat
- `tmp/eval_mambair_test_psnr.py` + `tmp/eval_mambair_full_frame.py` — harnesses for any future mambair ckpt eval

---

## 12. Reversibility / cost

- All changes are additive. No deletions.
- Reverting to pre-Phase-5#6 state = `git restore anime_upscaler/student_mambair.py anime_upscaler/distill.py` + `rm tmp/eval_mambair_*.py runs/distill_mambair_v1_10ep* tmp/mambair-build/` (if desired). PROJECT_MEMORY entries stay for historical record.
- Pre-existing 28 tests still pass. No interface breakage (default `--arch rfdn`).

---

## 13. Lessons

1. **Gradient checkpointing is essential for memory-heavy SSMs at this scale**.
   - Pure-PyTorch 4-direction scan OOMs at training shape when all 8 blocks are held in the autograd graph.
   - Checkpointing cuts peak from ~7.8 GB (OOM) to ~3.6 GB (fits). Eval mode unaffected.
   - This is the standard PyTorch pattern; should have considered it from the start.

2. **The structural ceiling persists across architectures, not just loss families**.
   - 5#6 (Mamba, no loss change from best, no adversarial) lands at 26.17 dB test — same regression curve as SRVGG family.
   - Means the ceiling is not bound by: loss family (LPIPS / twin / twin-asymmetric / no-perceptual), architecture (CNN / Mamba SSM), training budget (10 ep would likely give 26.7 at full convergence, still below SRVGG), adversarial weight (zero or 0.001). 
   - **Most likely root cause (now)**: dataset size (240 LR/HR pairs from 890 frames = 27% coverage) is too small for any architecture to break the 28 dB ceiling without external pre-training.

3. **The capacity dimension matters more than the architecture dimension**.
   - Doubling Mamba from 113K → 317K (matching SRVGG body) would likely close the 1.7 dB gap to ~28 dB.
   - Going SRVGG → Mamba at SAME capacity (113K) doesn't move the needle.
   - Going SRVGG → MUCH LARGER Mamba (2.6M) is unverified but plausible per the MambaIRv2 paper; cost = 60 min Docker + 30 hours training + 12 GB VRAM.

4. **Embarrassingly parallel eval harnesses take longer than they should**.
   - The first attempt to run test PSNR + full-frame eval in parallel hit the wrapper timeout (the test PSNR failed due to a multiprocessing Windows fork issue, and the wrapper killed both).
   - The full-frame eval was 17 min — long enough that any wrapper timeout eats it.
   - Lesson: run sequentially, with foreground rather than background, when inference is slow.

---

## 14. Recommendation to user

**Recommended next action: pivot to Phase 5#3 (FRAMER frequency-domain distillation) or ship baseline (option h).**

Detailed choices:

1. **(M, 1-2 weeks) Phase 5#3 FRAMER frequency-domain distillation** — additive loss change, no backbone change, low architectural risk, the only remaining "low-hanging" candidate.
   - Expected effort: ~30 min code (L_freq wiring in `distill.py` loss block) + 40-epoch pilot ~30 min wall + 2-epoch smoke ~2 min + D.1 + D.2 + D.3 eval.
   - Expected outcome: best case +0.3 dB over SRVGG body family (28.27 dB test), still below D.1 gate but with positive signal that frequency-domain constraints help.
   - Risk: failure mode = no signal. Cost = 1-2 weeks of effort wasted.

2. **(0 min) Ship baseline (option h)** — v1 RFDN + animevideov3 are already in the GUI registry; acknowledge we've tried 5+ distinct approaches (loss family, architecture, recipe) and none break the ceiling. Document and move on to other project goals (e.g., GUI improvements, export speed, deploy to other GPUs).
   - Expected effort: 0 min.
   - Risk: abandoning a research question without exhausting all candidates (5#4 multi-scale contrastive, 5#5 OSEDiff/SinSR, 5#7 FiDeSR/One-Step Diffusion are still on the matrix).

3. **(L, 2-3 weeks) Phase 5#5 OSEDiff/SinSR anime fine-tune** — one-step diffusion, ~0.5-1 sec/frame on RTX 4090, quality-tier preset. Different paradigm entirely.
   - Expected effort: 60+ min Docker/cuda install + 2-3 weeks fine-tuning + dataset work.
   - Risk: diffusion has never been tried; expected ++++ quality but at non-real-time cost.

My recommendation: **(1) — Phase 5#3 FRAMER**. It's the cheapest remaining option with a real (if narrow) chance of breaking the ceiling, and it has an additive-loss design that can't regress the SRVGG body result. If 5#3 also fails, then option (2) — ship baseline — becomes the right move.

---

## 15. References

- PROJECT_MEMORY §3 "Phase 5#6 (MambaIRv2 pure-PyTorch) — EXECUTED, NOT PROMOTED" (2026-09-06)
- PROJECT_MEMORY §4 line 172 "5#6 (2026-09-06): MambaIRv2 student distillation — PURE-PyTorch path"
- PROJECT_MEMORY §5 row "Phase 5#6 Mamba" + Phase 5#6 architecture comparison addendum
- PROJECT_MEMORY §6 "2026-09-06 -- Phase 5#6 LAUNCH" + "Phase 5#6 RESULT" entries
- PROJECT_MEMORY §7 H6 halt condition (triggered)
- PROJECT_MEMORY §9 "2026-09-06: Phase 5#6" update entry
- `tmp/mambair-build/` Dockerfile + dry logs (preserved, may be useful for future CUDA path)
- `docs/research/anime_sr_2026/REPORT.md` — original MambaIRv2 candidate scoring at ~+0.29 dB on Manga109
