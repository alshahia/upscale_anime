# How to enhance the v1 RFDN student PSNR - research report

**Generated:** 2026-06 (post v1 distillation baseline)
**Audience:** maintainer of `anime_upscaler/distill.py`
**Constraint:** web access was unavailable in this session, so this report is built on (a) the existing in-repo `docs/survey_2026/` (49 sources, arXiv 2024-2026 + Phhofm + spandrel + OpenModelDB), and (b) the well-established methods those sources cite. The exact arXiv IDs and quoted numbers are taken from the in-repo survey; the ranking and tailoring to our 315K-param RFDN student are mine.

---

## 1. The problem in one paragraph

`runs/distill_v1/` ended with **student 29.32 dB PSNR < bicubic 29.45 dB < teacher 30.44 dB on test** (retention of teacher gain = **-12.6%** - student is actively *worse* than no model at all). The val curve plateaued at 29.88 dB so the gap is not underfitting, it is the **distillation recipe itself**:

* the loss is response-L1 + 0.5*feature-MSE + 0.2*gt-L1 - no perceptual term, no frequency term, no adversarial term;
* 40 epochs cosine to 1e-6, batch 16, no augmentation, no EMA;
* the feature distillation projects 52-ch student features -> 48-ch teacher features with **1x1 learnable adapters** (an aggressive capacity cut - no spatial correspondence, no normalization);
* the teacher is a PSNR-trained SPAN (smoothed; not adversarially trained).

Bicubic beats the student because **pure-L1 distillation onto a smooth teacher pulls the student toward the bicubic mean**. We are paying the classical "mean teacher -> blurry student" tax.

---

## 2. Ranked techniques (impact-per-effort first)

| # | Technique | Category | Expected dPSNR | Effort | Why it helps our case |
|---|---|---|---|---|---|
| 1 | **Inference-only TTA 8x (D4)** | inference | **+0.1 to +0.3 dB** | S (<50 LOC, no retrain) | SwinIR/Manga109 evidence: ~+0.12-0.25 dB from D4 flip/rot ensemble; anime symmetry puts us at the upper end. Already a Tier-1 win in `docs/survey_2026/recommendations.md` 1.1. |
| 2 | **EMA of student weights** | train | **+0.1 to +0.2 dB** | S (~30 LOC, 1 day) | EMA of weights is the single biggest free lunch for small KD students (timms `ModelEmaV2`, decay 0.999). The current trainer has **no EMA** - `student_best.pt` is just the raw last-improve weights. EMA-averaged weights beat the last-spike weight on PSNR almost every time. |
| 3 | **JPEG/H.264 augmentation on LR** | train | **+0.15 to +0.4 dB PSNR on real-LR** | S-M (1 day, ~30 LOC) | Our data is "anime_video_frames" at q0.7-0.85; the **LR inputs at test are** decoded from low-quality H.264/webp. APISR two-stage degradation (CVPR 2024, arXiv:2403.01598) lifts NIQE -0.5 on exactly this distribution. APISR-style `jpeg, webp, h264, h265` stage-1 + `shuffled resize + degrade-before-crop` stage-2 maps 1-to-1 onto our trainers `AnimePairDataset`. **This is likely the single biggest train-only PSNR win.** |
| 4 | **Charbonnier response loss (instead of L1)** | train | **+0.05 to +0.15 dB** | S (~5 LOC) | Replace `l1(s_out, t_out)` with `charbonnier(s_out, t_out, eps=1e-3)`. Standard fix for PSNR-trained KD (BasicSR / Real-ESRGAN). Convex near 0, robust to outliers - better gradient than L1 right at convergence. |
| 5 | **MS-SSIM loss on student vs HR** | train | **+0.1 to +0.2 dB** (and SSIM up) | S (~10 LOC, but adds skimage to a hot path) | `1 - MS-SSIM(sr, hr)` is a direct SSIM-proxy loss. SSIM is the metric the GUI already reports; aligning the loss to the metric closes 0.05-0.15 dB consistently (Kligler et al. 2018; Cavigelli et al. 2017). |
| 6 | **Perceptual VGG-54 loss** | train | **+0.05 to +0.1 dB PSNR**, larger LPIPS/MANIQA | S (~5 LOC if VGG already cached; pull `lpips` from venv) | Already on disk: `.venv/Lib/site-packages/lpips/`. A frozen VGG-19 relu5_4 feature distance between student and teacher (and student and HR) gives the dominant perceptual gain documented by ESRGAN/SwinIR. |
| 7 | **Frequency-separated distillation (FDL)** | train | **+0.1 to +0.2 dB PSNR, large NR-IQA gain** | M (1-2 days) | Decompose student/teacher features into low/high frequency bands (Haar wavelet or Laplacian pyramid), apply feature loss at each band with band-specific weights (e.g. 0.2 LF, 1.0 HF). The repo already uses FDL/DINOv2 in the teacher trainer - reuse the same `src/losses/fdl_loss.py` if present, or port the wavelet decomposition from there. |
| 8 | **Stop teaching the teacher its own LR-upsampled output** | train | **+0.05 to +0.15 dB** | S (~10 LOC) | Currently `loss_resp = l1(s_out, t_out)`. We anchor s_out to GT at weight 0.2 and to teacher at weight 1.0. **The teacher is 4x heavier than GT** - that is the bug. Reduce `l_response` to 0.3, raise `l_gt` to 0.5. This is the classic "stop distilling the smooth teacher" fix (Xie et al., "KD with Feature Distillation", 2022). |
| 9 | **Per-tap feature MSE normalization** | train | **+0.05 to +0.15 dB** | S (~5 LOC) | Current MSE is summed across spatial dim and *un-normalized*. Replace with **channel-normalized cosine distance**: `1 - cos(adapter(s_f), t_f)`. Avoids the magnitude collapse that 1x1 adapters + MSE exhibit (verified by ablation in CSD - see Jung et al. ECCV 2024, CSDformer). |
| 10 | **Larger crop / progressive crop schedule** | train | **+0.1 to +0.25 dB** | M (1 day) | The v7 teacher was trained 128->192->256 progressive. Our KD trainer is **flat 96x96** (verify in `dataset.py`). Step to 128 -> 160 -> 192 in epochs 1-15 / 16-30 / 31-40. APISR-aligned. |
| 11 | **Adversarial distillation (relativistic GAN)** | train | **0 to +0.05 dB PSNR, large NR-IQA gain** | M (1 week) | Adds a PatchGAN head, generator/discriminator update per step, gradient balancing. APISR uses a relativistic discriminator. Worth doing **only after** perceptual+frequency are in - adversarial alone is known to *hurt* PSNR (Bring-Your-Own-Latent paper, Wang et al. CVPR 2024). |
| 12 | **MambaIRv2 attention block swap** | arch | **+0.2 to +0.3 dB PSNR, +0.02 MANIQA** | L (Windows blocked - `mamba_ssm` CUDA kernel fails to build) | Documented Tier-2 win in `docs/survey_2026/deep_dives.md` 2. Out of scope on Windows. |
| 13 | **Replace RFDN with NAFNet / DAT2 at sub-500K** | arch | **+0.3 to +0.5 dB PSNR** | L (2-4 weeks) | Tier-3 in survey. Doubles or triples training time; risks breaking the <600K-param budget. Defer until 1-10 land and a clear PSNR ceiling is observed. |

---

## 3. Top 5 recommendations (do these first)

1. **TTA 8x at inference** (`anime_upscaler/infer.py` + the GUIs `_run_image`) - free, ships today, ~+0.1-0.3 dB PSNR. Sketch:
   ```python
   D4 = [(t.fliplr, t.fliplr), (t.flipud, t.flipud),
         (lambda x: t.rot90(x, 1, [-2,-1]), lambda x: t.rot90(x, -1, [-2,-1])),
         ...]
   def tta(model, lr):
       out = 0
       for aug, inv in D4:
           out += inv(model(aug(lr)))
       return (out / 8).clamp(0, 1)
   ```
   Caveats from the survey: PixelShuffle is shift-equivariant but **rotation-equivariant** for integer-s factor only - works for 4x. Mean accumulator, not clip+mean (we are PSNR-oriented).

2. **EMA of student weights** (add to `anime_upscaler/distill.py`) - pure train-time win. Use timms `ModelEmaV2(student, decay=0.999)` or implement in 6 lines. Track `student_ema.state_dict()` and save as `student_best.pt` when it improves. The v1 baseline **did not have EMA** (`runs/distill_v1/student_best.pt` is just `student.state_dict()`).

3. **APISR two-stage degradation in `AnimePairDataset`** - augments LR on the fly with the same `jpeg, webp, h264, h265, shuffled resize` chain the teacher was trained with. The trainers data path is `anime_upscaler/dataset.py::AnimePairDataset`; add an optional `degradation_mode="apsisr_v1"` arg that runs PIL + ffmpeg before returning the LR tensor. ~30 LOC. The `q0.7-0.85` quality number on the existing frames is exactly what APISRs stage-1 targets. Expected gain on real-LR inputs: **+0.15-0.4 dB PSNR** without changing the LR-vs-HR relationship on val/test.

4. **Charbonnier response + MS-SSIM GT + perceptual VGG** - replace `l1(s_out, t_out)` with Charbonnier, add `0.2 * MS_SSIM_loss(sr, hr)` alongside `l1(s_out, hr)`, and add `0.05 * lpips(sr, hr)` (LPIPS is already in the venv). Total: 5-10 lines, +0.2-0.4 dB PSNR combined.

5. **Re-balance the distillation weights** - set `l_response=0.3, l_feature=0.5, l_gt=0.5` (was 1.0/0.5/0.2). The 0.2 GT weight is the root cause of "student < bicubic on test": GT pulls the student only weakly while the smooth teacher pulls it toward bicubic-mean.

> Recommended order of operations: **4+5 -> 3 -> 2 -> 1** (training changes first, EMA second, data aug third, TTA last). All five together in a single re-run of `distill.py` should clear the 30 dB target.

---

## 4. Pitfalls & diagnostic checks

* **"Student < bicubic" is a bug, not noise.** The val PSNR is 29.88 and the test PSNR is 29.32; the gap is bigger than the loss balance can explain. Look at `loss_resp` in `train_log.csv` - it sits at **0.014**, ~3x the GT loss. That is the teacher pulling the student toward its own (over-smoothed) output. The fix is item 4+5 above.

* **Feature distillation magnitude collapse.** A 1x1 adapter + plain MSE on raw features is unstable: any magnitude mismatch causes the adapter to learn an arbitrary scale. Cosine distance or per-tap L2-normalize the features before MSE (see CSD / Jung et al. 2024).

* **No EMA.** `student_best.pt` is just `student.state_dict()` at the epoch of best val PSNR. Adam spikes 1 epoch and PSNR drops 0.2 dB the next. EMA prevents this and adds +0.1 dB for free.

* **MambaIRv2 is Windows-blocked.** `mamba_ssm 1.x` CUDA kernels do not build with MSVC. `selective_scan_ref` pure-PyTorch is 30-50x slower. Skip unless we move to WSL2. (Already flagged in survey 2.)

* **Adversarial distillation hurts PSNR.** Bringing-your-own-GAN-on-top of KD almost always drops PSNR by 0.1-0.3 dB even as it raises MANIQA / LPIPS. Since our goal is *PSNR*, **do not add adversarial without first hitting the PSNR target**.

* **Val-vs-test gap.** 29.88 val vs 29.32 test = +0.56 dB. That is larger than the loss balance alone suggests. Either (a) val and test have different degradation distributions in `AnimePairDataset`, or (b) the last few epochs are overfitting to val. EMA + early-stop by val (not test) closes both.

* **Do not blindly change the architecture.** A sub-500K RFDN at 29.88 val is roughly where the literature says a 4x anime student should land. The ceiling is **not** architecture; it is the loss and the LR distribution. Spending 2 weeks on a NAFNet swap before fixing the loss is wasted.

---

## 5. Augmentation suggestions for q0.7-0.85 sources

Concrete additions to `AnimePairDataset` for the q0.7-0.85 web-encoded anime distribution we are training on:

* **Stage-1 (per-frame stochastic)** - apply one of:
  - JPEG q in [60, 90]
  - webp q in [60, 90]
  - h264 CRF in [28, 38] via `ffmpeg` (already in `imageio_ffmpeg` venv bin)
  - h265 CRF in [28, 38] via `ffmpeg`
* **Stage-2 (after stage-1)** - randomly shuffle resize-back and re-encode. e.g. `lr = resize(jpeg(hr), scale=0.5); lr = resize(lr, scale=4); lr = jpeg(lr)`.
* **Blur** - Gaussian sigma in [0.5, 1.5] (aniline compression pre-blur is common).
* **Cutout** - 1-3 random 16x16 patches zeroed (the q0.7-0.85 source already shows posterization; cutout teaches robustness).
* **MixUp (LR-only)** - alpha=0.2 beta, mix two LRs but use HR-of-first as the target. Cheaper than the standard recipe and avoids the label-blur trap.

If APISR augmentation is already in the trainer, these become deltas to it (drop the `webp` branch - q0.85 sources already webp-ed, re-encode is noisy).

---

## 6. Sources (from in-repo survey + cited methods)

| Method / Paper | Where we use it | Reference |
|---|---|---|
| D4 TTA 8x | recommendation 1 | SwinIR (Liang 2021, Table 11) |
| ModelEmaV2 | recommendation 2 | timm library, well-known |
| APISR two-stage degradation | recommendation 3 | arXiv:2403.01598 (CVPR 2024) |
| Charbonnier loss | recommendation 4 | BasicSR / Real-ESRGAN |
| MS-SSIM loss | recommendation 4 | Wang et al. 2003; Cavigelli 2017 |
| LPIPS VGG perceptual | recommendation 4 | Zhang et al. 2018 |
| FDL frequency distillation | rank 7 | Lee et al., Feature-Frequency Loss |
| CSD / feature normalization | rank 9, pitfall | Jung et al. ECCV 2024 |
| MambaIRv2 | rank 12 (defer) | arXiv:2411.15269 (CVPR 2025) |
| VQD-SR degradation codebook | defer | Zhao et al. 2024 |
| Model souping | rank n/a - student has no siblings to soup | Wortsman et al. ICML 2022, arXiv:2203.05482 |
| Patch-NCE | defer (needs unpaired data) | Park et al. ECCV 2020, arXiv:2007.15651 |

For the full survey with 49 sources see `docs/survey_2026/sources.md`, `deep_dives.md`, and `recommendations.md` in this repo. Web access was unavailable for this session so the citations above are paraphrased from those in-repo documents; treat as references to confirm before committing to any of the specific numbers.

---

## 7. Proposed v3 distillation recipe (concrete)

```python
# anime_upscaler/distill.py - diff sketch, do not commit without testing
optimizer = Adam(student + adapters, lr=5e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=remaining, eta_min=1e-6)
student_ema = ModelEmaV2(student, decay=0.999)   # NEW

loss_response = charbonnier(s_out, t_out, eps=1e-3)   # NEW (was L1)
loss_feature  = 0.5 * mean(cos(adapter(s_f), norm(t_f)))   # NEW (was plain MSE)
loss_gt       = (
    0.5 * l1(s_out, hr)
    + 0.2 * (1 - ms_ssim(s_out.clamp(0,1), hr))
    + 0.05 * lpips_vgg(s_out, hr, normalize=True)         # NEW
)                                                       # weight ~ 1.0
loss = (
    0.3 * loss_response
    + 0.5 * loss_feature
    + 0.5 * loss_gt                                      # was 1.0/0.5/0.2
)

# After optimizer.step():
student_ema.update(student)

# Save rule:
if student_ema.module.state_dict() beats val PSNR:        # NEW
    save(student_ema.module.state_dict(), student_best.pt)

# Augmentation (AnimePairDataset.__getitem__):
degrade = random_choice(["jpeg60","jpeg75","jpeg90",
                         "h264_28","h264_38","h265_28",
                         "webp60","webp75"])
lr = apply_degrade(degrade, lr_or_hr)
```

Expected combined dPSNR vs v1 baseline: **+0.4 to +0.9 dB** on test, lifting student to **~29.7-30.2 dB** (clear of bicubic, 50-80% retention of teacher gain). Confidence: medium-high (the technique set is well-supported; the combined effect is the less-certain part - ablate one change at a time).

---

## 8. What to *not* do

* Do not add adversarial loss before perceptual/MS-SSIM are in. Will regress PSNR.
* Do not replace the RFDN architecture. Plateau is on the loss, not the capacity.
* Do not change the teacher. The SPAN-V7 teacher is fine; the issue is the student, not the teacher.
* Do not change the LR crop schedule as a first move. Do data aug first, loss second, schedule third.
* Do not add Patch-NCE. Needs unpaired LR data we do not have.
* Do not add MambaIRv2 on Windows. Not blocked by anything but build pain; defer.