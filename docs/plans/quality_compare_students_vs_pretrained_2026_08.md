# Quality Comparison - Distilled Students vs Pretrained Anime SR

**Status (2026-08-31):** DONE. Single-frame comparison of our two distilled
students against canonical anime pre-trained models on a real 1-second anime
video.

**Branch:** `feature/phase-1-realtime-4k`

**Hardware:** Quadro RTX 4000, torch 2.12.0+cu126

**Goal of this step:** provide a visual + objective evidence baseline for
how the distilled students compare against the RealESRGAN anime family so
the user can decide whether to retrain, change the loss, or ship as-is.
**Speed is out of scope for this report.**

---

## Headline finding

**Both distilled students are visibly over-smoothed compared to the
ReaLESRGAN anime family.** On the real-anime frame (mid-shot of an Attack
on Titan scene), the v1 student 4x output is perceptually indistinguishable
from bicubic on the hair strands and only marginally sharper on the eye.
The v2 cascade 2x+2x is slightly better but still soft. The RealESRGAN
animevideov3 weight produces the canonical crisp-anime-line look the user
asked about ("smooth of the anime images"); RealESRGAN-LSDIR is even
sharper but introduces a more photo-real / painterly rendering that is
less anime-true.

**Recommended short-term action:** ship a pre-trained RealESRGAN
`realesr-animevideov3.pth` (or `4xLSDIRCompactv2.pth` if a faster
alternative is needed) behind the existing pipeline so the user gets the
quality they expect. The student path remains the long-term bet because of
parameter count, but it needs adversarial / GAN training to close the gap.

---

## What was run

- Script: `scripts/compare_students_vs_pretrained.py`
- Input: frame 8 of `tmp/real_video_1sec.mp4` (854x480 anime shot)
- All 4x outputs at 3416x1920, then 2x zoom crops of hair / eye regions
  saved under `results/quality_compare_students_vs_pretrained/`
- Models compared:
  - `bicubic`                       (baseline, no model)
  - `v1_student_4x`                 `RFDN_distill_v1_4x_student.pth`, scale=4
  - `v2_student_cascade_2x2`        same RFDN scale=2 applied twice (cascade)
  - `RealESRGAN_animevideov3`       `realesr-animevideov3.pth`
  - `RealESRGAN_LSDIR`              `4xLSDIRCompactv2.pth`
  - `AnimeSR_v2` (skipped: recurrent 3-frame arch fails on single-frame
    replicated 3x; its weight expects real temporal context. RealESRGAN
    already covers the user's "pre-trained model" request.)
- AnimeSR_v2 left as a commented-out entry in the script so a video-based
  evaluation can add it later.

---

## Objective metrics (real-anime frame 8, LR 854x480 -> SR 3416x1920)

Sorted by **Laplacian variance** (higher = sharper).

| Model | Mode | Res | mean | std | lap_var | grad_mag | sat_std |
|---|---|---|---:|---:|---:|---:|---:|
| RealESRGAN_LSDIR        | srvgg (4x)            | 3416x1920 | 44.13 | 44.48 | **132.6** | 6.70 | **65.77** |
| RealESRGAN_animevideov3 | srvgg (4x)            | 3416x1920 | 45.42 | 45.24 |   58.1    | 6.36 |   61.14   |
| v2_student_cascade_2x2  | rfdn cascade (2x x2)  | 3416x1920 | 45.76 | 44.21 |   30.8    | 6.74 |   56.36   |
| v1_student_4x           | rfdn_student (4x)     | 3416x1920 | 45.36 | 44.59 |   21.0    | 6.85 |   58.51   |
| bicubic                 | bicubic 4x            | 3416x1920 | 45.65 | 44.47 |   11.8    | 5.82 |   58.92   |

Reading the table:

- **lap_var** (variance of Laplacian) is the cleanest sharpness proxy.
  v1 student is **1.78x bicubic**, v2 cascade **2.61x bicubic**, but
  RealESRGAN-LSDIR is **11.3x bicubic** and animevideov3 is **4.9x
  bicubic**. The pretrained models produce 2-6x more high-frequency
  detail than the students.
- **grad_mag** (mean Sobel magnitude) is roughly flat across models
  (~6.4-6.85). Bicubic is the lowest at 5.82, meaning the students do
  add SOME edge energy but not enough to be visually distinct.
- **sat_std** is highest for RealESRGAN-LSDIR (65.77), then
  animevideov3 (61.14), then our students and bicubic clustered around
  56-59. Pretrained models preserve more chromatic range.
- **mean / std** are nearly identical across all outputs (~45 / 44)
  which is expected: the bicubic residual shortcut in our students
  guarantees the student output is anchored near the bicubic baseline.
  If you remove that shortcut, mean/std will drift; on this evidence it
  has drifted toward preserving bicubic rather than adding detail.

Numbers above are also persisted as
`results/quality_compare_students_vs_pretrained/metrics.csv`.

---

## Visual evidence

All under `results/quality_compare_students_vs_pretrained/`.

- **Full 4x outputs** (3416x1920, half-scale grid in `grid_full.png`):
  - `bicubic_4x.png`
  - `v1_student_4x_4x.png`
  - `v2_student_cascade_2x2_4x.png`
  - `RealESRGAN_animevideov3_4x.png`
  - `RealESRGAN_LSDIR_4x.png`
- **Hair-region zoom crops** (2x of 180x180 LR region) under `crops/`:
  - `crops/bicubic_hair_2x.png`                  -- soft, strands bleed
  - `crops/v1_student_4x_hair_2x.png`            -- near-identical to bicubic
  - `crops/v2_student_cascade_2x2_hair_2x.png`   -- marginally sharper edge
  - `crops/RealESRGAN_animevideov3_hair_2x.png`  -- **clean line art**
  - `crops/RealESRGAN_LSDIR_hair_2x.png`         -- very sharp but painterly
- **Eye-region zoom crops** (same naming with `_eye_2x.png`).
- `grid_full.png`: vertical half-scale stack of all 4x outputs.
- `grid_zoom.png`: hair + eye zoom panels with label band per row.

### What you see in the zoom crops

On the hair region (top-right of the frame, ~470-650 x, ~20-200 y):

- **bicubic and v1 student**: each hair strand is a soft brush stroke;
  the eye iris is a flat gray disk with no texture.
- **v2 cascade**: slightly more edge definition (you can see the
  individual hair strands start to separate) but the overall feel is
  still painted, not drawn.
- **RealESRGAN_animevideov3**: the canonical anime look -- each hair
  stroke is a distinct dark line, eye iris has iris striations, eyebrow
  hairs visible, lip line crisp. Skin is smooth (no plastic / no
  over-sharpening). This is what anime upscaling should produce.
- **RealESRGAN_LSDIR**: highest raw sharpness (lap_var 132.6 vs 58.1)
  but the look is photo-real: skin has subtle texture, hair strands
  look painted. Sharp but slightly off-style for anime.

---

## Why the students underperform (root cause)

Two compounding causes:

### 1. Loss formulation

Our distillation recipes (`anime_upscaler/distill.py`) train with
pixel loss + perceptual loss. Neither loss penalises "looks too smooth"
directly. The optimal L1/L2 solution is the conditional mean, which for
natural images is biased toward the low-pass filtered ground truth.
GAN / adversarial loss is what produces the high-frequency, anime-line
detail that RealESRGAN and AnimeSR ship with.

### 2. Bicubic residual shortcut

Both our RFDN students output

    sr = self.upsampler(...) + F.interpolate(lr, scale_factor=scale, mode="bicubic")

This is great for **training stability** (the network only learns the
difference vs bicubic) and **safety** (if the student fails to learn, you
still get a bicubic-quality output rather than artifacts). But it also
makes the model contribution small in regions where bicubic is already
close to optimal -- which is most of an anime frame. The 1.78x
lap_var improvement of v1 over bicubic means the student is contributing
a small fraction of the high-frequency band.

If we remove the shortcut, training becomes less stable (more risk of
instability / color shift in early epochs), and we need stronger
adversarial loss to compensate.

### 3. Teacher quality / data

ReaLESRGAN was trained on anime-specific data with an anime-tuned
adversarial objective. Our distillation teacher is generally a
photo-real SR model (e.g. EDSR / RCAN / SPAN). Distilling a photo
teacher onto a tiny student and expecting anime-line fidelity is
uphill; we are inheriting the photo teacher blur bias.

---

## Recommendations

1. **(Quality fix, near-term, ~0.5 day)** Ship a RealESRGAN
   `realesr-animevideov3.pth` (or `4xLSDIRCompactv2.pth` if speed
   matters more than style) as an alternative model in the GUI so the
   user has a quality option today. Use the existing `srvgg` kind in
   `archs.build` -- no new code needed. The model files are already
   in `pretrained/`. We do **not** need to re-export ONNX/TRT for these
   since they will be loaded with the existing student pipeline
   (via `srvgg` kind dispatch).

2. **(Speed fix, near-term)** If real-time 4K is still the goal, the
   SRVGG compact net (`realesr-animevideov3.pth`) is much faster on
   GPU than our RFDN student at the same quality -- this is exactly
   the engineering trade RealESRGAN was designed for.

3. **(Quality gap closer, medium-term)** Add an adversarial loss to
   distillation (`anime_upscaler/losses/adversarial_loss.py` already
   exists; we would need a PatchGAN discriminator trained on
   anime LR/HR pairs). Re-distill v1 with adversarial weight ~0.005
   and compare against the current RealESRGAN-LSDIR baseline.

4. **(Loss architecture, medium-term)** Consider removing or reducing
   the bicubic residual for the adversarial-trained student so the
   network is not anchored toward the blurry solution. Train with a
   small warm-up where the shortcut is full weight, then anneal it
   down to e.g. 0.3 over the first 5 epochs.

5. **(Defer)** v2 cascade: per the existing `phase2_cascade_plan.md`,
   cascade was already a negative result on speed and quality. With
   the new evidence above, **drop the v2 cascade path entirely** and
   redirect that compute to training a single 4x student with
   adversarial loss.

---

## Actionable TODOs (next steps)

- [ ] A.1  Decide: ship RealESRGAN as an alternative model in the GUI
  (option 1 above) or invest in retraining the student first (option 3).
  Recommend A.1 now and A.3 in parallel as a longer bet.
- [ ] A.2  Add `srvgg` entries for `realesr-animevideov3.pth` and
  `4xLSDIRCompactv2.pth` to `apps/.../registry.py` (preset entries
  with scale=4, kind="srvgg").
- [ ] A.3  Add an end-to-end quality comparison harness that runs on
  `--quality-only` mode in the existing test_e2e scripts, so we can
  re-run quality after each student retraining and not regress.
- [ ] A.4  Investigate adversarial loss integration with our
  `AnimeSR_v2` (it is the closest weight in our corpus to the
  animevideov3 quality target). Use it as a teacher or as an
  adversarial target for the next student.
- [~] A.5  This document and the comparison artifacts are committed on
  `feature/phase-1-realtime-4k` (pending user approval).

---

## Files in this drop

- `scripts/compare_students_vs_pretrained.py`
   (the comparison harness; arch stubs already in `archs.py`)
- `docs/plans/quality_compare_students_vs_pretrained_2026_08.md`
   (this file)
- `results/quality_compare_students_vs_pretrained/`
   - `source_frame.png`            -- 854x480 LR
   - `<model>_4x.png`              -- 5 native 4x outputs
   - `crops/<model>_<region>_2x.png`  -- 10 zoom crops (2 regions x 5 models)
   - `grid_full.png`, `grid_zoom.png` -- visual comparison composites
   - `metrics.csv`, `metrics.md`   -- per-model objective numbers
- `tmp/quality_compare_run.log`    -- captured stdout from the run

## Risks / known limitations

- Single-frame evaluation. Anime is a video domain; PSNR-style metrics
  on one frame cannot capture temporal stability. A second-pass eval on
  1-2 sec of `tmp/real_video_1sec.mp4` is recommended before any
  ship decision.
- The crops were picked to expose sharpness issues. There may be
  regions (uniform skin, plain sky) where the student is competitive.
  The full `grid_full.png` shows the overall balance.
- All models run in fp32. fp16 / TRT may shift the metrics slightly
  but is unlikely to change the ranking.
- AnimeSR_v2 was intentionally skipped; the harness still has the
  entry commented-out so a future video-based eval can add it.
