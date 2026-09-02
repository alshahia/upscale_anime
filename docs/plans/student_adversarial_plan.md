# Plan: Close the Quality Gap -- Adversarial Student Distilled from RealESRGAN-animevideov3

**Status:** PROPOSED (waiting for user approval)
**Author / driver:** MiniMax-M3
**Created:** 2026-08-31
**Branch:** feature/phase-1-realtime-4k (continues Phase 2.D, commit 324649f)
**Related report:** docs/plans/quality_compare_students_vs_pretrained_2026_08.md

---

## 1. Goal

Train a **single 4x RFDN student** whose perceptual quality is within
striking distance of realesr-animevideov3.pth while staying under
~600K parameters and a single forward pass on the existing pipeline.

Concrete target on the existing single-frame quality harness
(scripts/compare_students_vs_pretrained.py):

  - Laplacian variance on the SR output **>= 35** (animevideov3 = 58.1,
    current v1 student = 21.0, bicubic = 11.8). This is the primary
    sharpness gate.
  - Subjective visual review: hair strands distinct, eye iris textured,
    no plastic / over-smoothed skin, comparable to animevideov3 zoom
    crops.
  - PSNR drop vs animevideov3 <= 1.5 dB (animevideov3 sets the upper
    bound; the student will not match it on PSNR but should be close).
  - LPIPS vs animevideov3 SR <= 0.12 (perceptual gap; lower is better).

Non-goals:

  - Beating animevideov3 on PSNR / speed. We are a smaller distilled
    student; that comparison is unfair.
  - Producing a general-purpose photo SR model. Anime-only.
  - Replacing the existing pipeline or GUI. Pure additive change.

---

## 2. Background and decision inputs

The Phase 2.D quality report found:

  - v1 student lap_var = 21.0 (1.78x bicubic), v2 cascade = 30.8 (2.61x),
    RealESRGAN-animevideov3 = 58.1 (4.93x), RealESRGAN-LSDIR = 132.6 (11.26x).
  - Root causes:
    1. **Loss formulation** -- pixel L1 + VGG perceptual only.
       Optimal L1/L2 solution is the conditional mean, biased toward
       the low-pass filtered ground truth. RealESRGAN uses adversarial.
    2. **Bicubic residual shortcut** -- anchors student output near
       bicubic; safe but caps contribution to high-frequency band.
    3. **Teacher mismatch** -- distillation teacher is photo-real
       (EDSR / RCAN / SPAN); we inherit its blur bias.
  - animevideov3 produces the canonical anime look the user wants.
    LSDIR is sharper but photo-real (off-style for anime).

---

## 3. Teacher decision

**Primary teacher: realesr-animevideov3.pth**

  - Compact SRVGG (16 convs, nf=64, scale=4), 2.5 MB.
  - Adversarially trained on anime video frames. Produces the
    'anime line art' look the user requested.
  - Available locally under pretrained/.
  - Forward pass: ~25 ms on RTX 4000 for 480x270 LR. Cheap to use
    on-line during student training.

**Secondary / ablation teacher: 4xLSDIRCompactv2.pth**

  - Same arch as animevideov3, trained on LSDIR (broader / photo).
  - Used only as a 'what does a non-anime teacher give us' control
    in the final eval table, not for the primary training run.

Why not the bigger teachers?

  - AnimeSR_v2.pth (TencentARC MSRSWVSR): recurrent 3-frame,
    needs motion-compensated frames; not a good single-image teacher.
  - EDSR_x4.pt / RCAN_x4.pt / SwinIR_x4.pt: photo-real teachers
    with the same blur bias we are trying to escape.
  - span_pix_pretrain_4x.pth / span_mssim_pretrain_4x.pth:
    documented 'warm-start taint' issue (bias_mean ~0.43, flat output).
    AGENTS.md flags these as risky without re-initialization.

---

## 4. Reuse the existing student, or train from scratch?

**Recommendation: do both. Try warm-start first; from-scratch as a fallback.**

### 4a. Warm-start from RFDN_distill_v1_4x_student.pth (PRIMARY)

  - Pros: faster convergence (already converged to a local minimum on
    pixel loss; we just need to push out of that basin with adversarial).
    Reuses 315K params of prior work; safe baseline at epoch 0.
  - Cons: the warm-start encodes the prior teacher's distribution.
    If the bicubic shortcut bias is too strong, the new adversarial
    gradient cannot escape it and we plateau.
  - Mitigation: **anneal the bicubic shortcut from 1.0 to 0 over the
    first 5 epochs** so the network is forced to find an alternative
    path to produce the SR image.

### 4b. From-scratch (FALLBACK)

  - Triggered if warm-start stalls (lap_var < 25 after epoch 15).
  - Same architecture, same loss, same schedule; just init from random.
  - Converges slower (no useful prior) but starts from a clean loss
    landscape. We expect ~5-10 extra epochs to reach the same quality.
  - Use a different run tag (v3_from_scratch) so we can compare.

### 4c. What we will NOT do

  - Continue distillation from the *photo-real* teacher (EDSR/SPAN).
    That is exactly the failure mode we are fixing.
  - Keep the 2x+2x cascade path. Phase 2.C showed it is slower AND
    lower-quality for the RFDN body. Drop it permanently.
  - Touch the GUI / pipeline / TRT cache. Out of scope for this plan.

---

## 5. Architecture: keep / add / remove / avoid

### 5a. KEEP (proven, do not touch)

  - **RFDN body** (anime_upscaler/student.py::RFDN, vendored copy in
    apps/.../archs.py::RFDN): parameter-efficient, body-shape matches
    anime line patterns.
  - **RFDBlock**: 1x1 distillation branches + 3x3 remainder + pixel
    attention. This is the right inductive bias for anime.
  - **PixelAttention** at the body tail.
  - **Local residual inside each block** + **global residual around the
    body** (head to body_tail). Both contribute to stable training.
  - **nf=52, num_blocks=6**: 315K params at scale=4. Comfortably
    under our 600K budget; room for a 4-block variant if adversarial
    needs more capacity (see 5d).
  - **PixelShuffle upsampler**: cheap, no checkerboard, GPU-friendly.
  - **scale = 4 single forward pass.** No cascade.
  - **fp16 inference path** (already in pipeline). Training will use
    fp32 mixed-precision (autocast) for stability.

### 5b. ADD

  - **PatchGAN discriminator** (new file anime_upscaler/discriminator.py).
    - 70x70 receptive field, 4 strided conv blocks (C64, C128, C256,
      C512) + 1x1 final conv to a single-channel logit map.
    - Spectral normalization on every conv (stabilizes GAN training).
    - Hinge loss. Trained 1 D step per 1 G step (TTUR).
    - Input: concat(LR_upscaled_to_HR, SR_or_HR), 6 channels.
  - **Edge / gradient L1 loss** (new file anime_upscaler/losses/edge_loss.py).
    Sobel-x + Sobel-y magnitude, L1 against HR. Cheap,
    preserves hair/eye edges that pure pixel loss misses.
  - **L_distill** term against animevideov3 SR (already in
    distill.py; just rewire the teacher path).
  - **EMA of student weights** (existing training/ema.py); already
    in the codebase, just enable it.
  - **LPIPS** in the eval harness (new dep; needs pip install lpips).
    Used only for the per-epoch val gate, not as a training loss.

### 5c. REMOVE or REDUCE

  - **Bicubic residual shortcut**: anneal 1.0 -> 0.5 (ep 1-2) ->
    0.2 (ep 3-4) -> 0.0 (ep 5+). This is the single biggest fix.
  - **Pure pixel L1 weight**: drop from default (~1.0) to 0.5 so the
    adversarial + perceptual gradients are not drowned out.
  - **VGG perceptual 'isolation'**: keep at 1.0 but pair it with the
    adversarial signal so it does not bias toward blur.

### 5d. ARCHITECTURE VARIANTS TO TRY (in order)

  1. **A**: keep v1 arch exactly (nf=52, 6 blocks), warm-start from
     v1, anneal shortcut. *This is the baseline attempt.*
  2. **B**: same as A, but warm-start + from-scratch ablation (4b).
  3. **C** (only if A and B both stall): bump num_blocks=4 (drop
     redundant distillation capacity; rely on adversarial for the
     high-freq), keep nf=52. ~250K params.
  4. **D** (only if A/B/C all stall on capacity): try nf=64,
     num_blocks=4. ~470K params. Still under the 600K budget.
  5. **E (NOT recommended)**: switch to SRVGG-style body (matches
     animevideov3 shape, makes distillation easier). Loses RFDN's
     cheap body. Defer unless all of A-D fail.

### 5e. AVOID

  - **StableSR / ControlNet / diffusion-based conditioning**: massive
    scope creep; we do not have paired control signals.
  - **Photo-real teachers (EDSR / RCAN / SPAN / SwinIR)**: these are
    what caused the blur in the first place.
  - **AnimeSR_v2 as a teacher**: recurrent 3-frame, expects motion
    context; misaligned with single-frame student.
  - **Increasing model size past 600K**: kills the real-time budget
    that Phase 1 was designed around.
  - **Self-distillation loops (student -> teacher)**: known unstable
    unless the teacher is much larger. animevideov3 is only ~3x bigger
    than our student; it is the teacher, not a co-student.
  - **Removing the global residual inside RFDN body**: the network
    will diverge in the first epoch. Keep the body's internal residual.

---

## 6. Training recipe

### 6a. Data

  - HR: anime video frames, existing data/anime_video_frames/.
    Add a held-out val split (200 frames) for the per-epoch gate.
  - LR synthesis: bicubic-downsample of HR, **plus** JPEG compression
    at q=70-95 (random per batch). Real-world compressed LRs bridge
    the synthetic->real gap RealESRGAN handles.
  - Patch size: 128x128 LR (512x512 HR). Random crop + horizontal flip.
  - Batch size: 16 (RTX 4000 8 GB, fp32). Fallback to 8 if OOM.
  - Workers: 4.

### 6b. Losses

    L_total = 0.5 * L_pix(SR, HR)
            + 1.0 * L_perc(SR, HR)               # VGG-19 perceptual
            + 0.5 * L_distill(SR, T_SR)          # T = animevideov3
            + 0.1 * L_grad(SR, HR)               # Sobel-edge L1
            + lambda_adv(t) * L_adv(SR, HR)      # hinge GAN

Where:

  - L_pix: L1
  - L_perc: VGG-19 conv1_2..conv5_4 features, MSE; existing
    losses/perceptual_loss.py.
  - L_distill: L1(student_SR, teacher_SR). Teacher gets the same LR
    batch; SR is generated on-the-fly (single forward ~25 ms).
  - L_grad: |Sobel(SR)| - |Sobel(HR)| in L1.
  - L_adv: hinge, -E[min(0, -1 + D(SR, LR_up))] for G.
  - lambda_adv(t): schedule (see 6d).

### 6c. Optimizers (TTUR)

  - G: AdamW, lr=2e-4, betas=(0.9, 0.99), weight_decay=1e-4,
    cosine annealing to 1e-6 over 30 epochs.
  - D: Adam, lr=4e-4, betas=(0.9, 0.99). (2x G LR per TTUR.)
  - Grad clipping: max_norm=1.0 on both G and D.

### 6d. Schedules

  - Epochs: 30 (~15 h on RTX 4000).
  - Bicubic shortcut weight:
    ep 1 -> 1.0 (warm-up; student starts near current behavior)
    ep 2-3 -> 0.5
    ep 4-5 -> 0.2
    ep 6+ -> 0.0
  - lambda_adv:
    ep 1-5 -> 0.0 (pure distillation; let the student find a good
    starting point before adversarial kicks in)
    ep 6-10 -> 0.001
    ep 11-20 -> 0.005
    ep 21-30 -> 0.01
  - D step: 1 per G step (no D pre-training; relies on the warm-up).
  - EMA: shadow update every step, decay=0.999; eval/ckpt from EMA.
  - **Checkpoint every epoch** to runs/distill_v3_4x/epoch_N.pt (N=1..30). Never overwrite; rotate. Keep last 5 epochs on disk to recover from a late-training spike.
  - **Checkpoint every epoch** to runs/distill_v3_4x/epoch_N.pt (N=1..30). Never overwrite; rotate. Keep last 5 epochs on disk to recover from a late-training spike.

### 6e. Per-epoch eval gate (must pass to continue training)

  - val_PSNR > 30 dB (current v1 is ~30.1 dB on val_sr)
  - val_LPIPS < 0.20 (proxy for perceptual gap closing)
  - val_lap_var > 25 by epoch 10, > 35 by epoch 20, > 40 by epoch 30
  - D loss and G adv loss both non-zero, neither diverging
  - If val_PSNR drops > 0.5 dB for 3 consecutive epochs: halt
    (mode collapse signal).
  - If val_LPIPS does not improve for 5 consecutive epochs: halt
    (adversarial saturation).

---

## 7. End-of-training eval

  1. Re-run scripts/compare_students_vs_pretrained.py adding
     v3_student to MODELS. Expect lap_var in [35, 60] range.
  2. Full-video eval on tmp/real_video_1sec.mp4: re-run
     tmp/test_e2e_cascade_real_video.py with the new checkpoint
     swapped in. Watch for temporal stability (anime flicker).
  3. LPIPS vs animevideov3 SR per frame on val_sr/2.png: target
     <= 0.12.
  4. Manual zoom-crop review (hair, eye, lip, skin).
  5. Visual gate decision: accept / iterate.

---

## 8. Risks and mitigations

  - **Adversarial instability**: spectral norm + TTUR + grad clipping +
    slow lambda_adv ramp. If still unstable, drop lambda_adv peak to
    0.005 (not 0.01).
  - **Mode collapse**: per-epoch PSNR/SSIM/LPIPS monitoring, automatic
    halt on plateau. Keep an EMA copy of weights to recover from a bad
    late-epoch spike.
  - **Discriminator overfit**: held-out val set, early stopping.
  - **VRAM**: batch=16 + patch=128 + D fp32 ~ 6.5 GB. If OOM: drop
    batch to 8 (accum=2 to keep effective LR).
  - **Time**: 15 h is long; we may want a 15-epoch proof-of-concept
    first (see Q3 below).
  - **Real-world LR distribution shift**: if our bicubic LR is too
    different from real compressed LR, the student overfits synthetic.
    The JPEG-q augmentation in 6a mitigates this; if it still
    underperforms on real video, add the existing anime_degradation
    pipeline (already in src/data/anime_degradation.py).

---

## 9. Phases and TODOs

### Phase A -- Code (0.5 day)

  - [ ] A.1 Create anime_upscaler/discriminator.py:
        PatchGAN 70x70, spectral norm, Hinge loss.
  - [ ] A.2 Create anime_upscaler/losses/edge_loss.py:
        Sobel-x + Sobel-y, L1 against HR.
  - [ ] A.3 Edit anime_upscaler/distill.py:
        - add --teacher animevideov3|lsdir arg
        - add --lambda-adv schedule arg (default ramp 0->0.01)
        - add --shortcut-anneal schedule arg (default 1->0)
        - add --resume path/to/v1.pth arg
        - wire in L_distill, L_adv, L_grad
        - enable EMA
  - [ ] A.4 Edit scripts/eval_cascade_vs_single.py (or add a new
        scripts/eval_lpips.py): add LPIPS per-frame.
  - [ ] A.5 Add scripts/train_v3_smoke.py: 2-epoch sanity harness
        that runs with adversarial ON at lambda=0.001 and prints
        G/D loss curves + a sample SR PNG.

### Phase A -- Tests

  - [ ] A.6 py_compile all touched files (PASS).
  - [ ] A.7 train_v3_smoke.py runs 2 epochs without NaN,
        discriminator loss non-zero, sample SR saves correctly.
  - [ ] A.8 Load v1 ckpt via --resume; epoch 0 loss in the same
        range as a from-scratch epoch 0 (sanity).

### Phase B -- Validation runs (0.5 day)

  - [ ] B.1 Run train_v3_smoke.py --epochs 2 --lambda-adv 0 --
        confirm student does not regress vs v1.
  - [ ] B.2 Run train_v3_smoke.py --epochs 2 --lambda-adv 0.001 --
        confirm D loss > 0 and G adv loss in a stable range
        (target: |G_adv| < 1.0 after 100 steps).
  - [ ] B.3 Run train_v3_smoke.py --epochs 2 --shortcut-anneal 1to0 --
        confirm student still produces non-NaN SR at epoch 2.

### Phase C -- Training (~1 day unattended)

  - [ ] C.0 Implement per-epoch checkpoint rotation in distill.py before
        starting the run. Every epoch save:
        - runs/distill_v3_4x/epoch_N.pt            (raw G weights, for resume)
        - runs/distill_v3_4x/epoch_N_ema.pt        (EMA copy, for eval/ship)
        - runs/distill_v3_4x/epoch_N_metrics.json  (val PSNR/SSIM/LPIPS/lap_var)
        - runs/distill_v3_4x/latest.pt and latest_ema.pt (rotating)
        Never overwrite. Keep last 5 epochs on disk; archive older to
        runs/distill_v3_4x/archive/ if disk is tight.
  - [ ] C.1 Run full 30-epoch training (warm-start from v1):
        python anime_upscaler/distill.py --teacher animevideov3
        --resume pretrained/RFDN_distill_v1_4x_student.pth
        --epochs 30 --batch-size 16 --lr 2e-4
        --out-dir runs/distill_v3_4x
  - [ ] C.2 If warm-start stalls (lap_var < 25 at epoch 15): kill,
        re-run as runs/distill_v3_4x_scratch with no --resume.
  - [ ] C.3 If still stalls: try variant C (nf=52, 4 blocks) then D
        (nf=64, 4 blocks). Each is a separate --out-dir.
  - [ ] C.4 After Phase D eval PASSES the lap_var >= 35 gate, copy the
        specific epoch_N_ema.pt that produced the best result ->
        pretrained/RFDN_distill_v3_4x_student.pth. **Do NOT delete** the
        per-epoch ckpts in runs/distill_v3_4x/ until the user explicitly
        confirms acceptance (they are the only fallback if a regression
        is later discovered).

### Phase D -- Eval (0.5 day)

  - [ ] D.1 Re-run scripts/compare_students_vs_pretrained.py with
        v3 added to MODELS. Save artifacts to
        results/quality_compare_v3_2026_08/.
  - [ ] D.2 Full-video eval on tmp/real_video_1sec.mp4 -- check
        flicker (frame-to-frame pixel diff std in a static region).
  - [ ] D.3 Manual zoom-crop review (hair, eye, lip, skin) at
        ep 10, 20, 30 snapshots.
  - [ ] D.4 Decide: accept / iterate / abandon (record decision and
        reasoning in the eval doc).

### Phase E -- Ship RealESRGAN baseline (parallel to Phase C; 0.5 day)

  Independent of retraining the student, ship the RealESRGAN baseline
  so the user has a quality option today. This does NOT block on the
  adversarial student working.

  - [ ] E.1 Add srvgg preset entries to apps/.../registry.py for
        realesr-animevideov3.pth and 4xLSDIRCompactv2.pth.
  - [ ] E.2 Update apps/.../widgets/models_panel.py so the new
        entries appear in the model selector.
  - [ ] E.3 Run apps/anime_upscaler_gui/scripts/build_assets.py
        if there is a static asset catalog to refresh.
  - [ ] E.4 Manual smoke: load animevideov3 via the GUI on
        tmp/real_video_1sec.mp4; verify it produces a 4x output
        without errors.
  - [ ] E.5 (if v3 student ships) ONNX export v3 student; TRT engine
        build via the existing pipeline; add to registry.

### Phase F -- Documentation (0.25 day)

  - [ ] F.1 Update docs/plans/quality_compare_students_vs_pretrained_2026_08.md
        with the v3 row in the metrics table.
  - [ ] F.2 Write docs/plans/student_v3_result_2026_08.md with the
        training curves, eval results, ship/no-ship decision.
  - [ ] F.3 Update AGENTS.md if any new conventions are introduced
        (e.g. the --shortcut-anneal flag, the PatchGAN location).

---

## 10. Time / cost estimate

  - A + B (code + smoke tests): 1 day
  - C (training): 15 hours unattended
  - D (eval): 0.5 day
  - E (ship RealESRGAN baseline, parallel): 0.5 day
  - F (docs): 0.25 day
  - **Total wall-clock: ~2.5 days end-to-end** (C can run overnight;
    E can run in parallel with C).

GPU: Quadro RTX 4000 8 GB. fp32 training. No multi-GPU needed.

---

## 11. Open questions (need user answer before coding)

  - **Q1.** Approve realesr-animevideov3.pth as the primary teacher?
    (LSDIR as ablation teacher is OK regardless.)
  - **Q2.** Approve warm-start from v1 first, with from-scratch as a
    fallback if the warm-start stalls? (vs going straight to
    from-scratch.)
  - **Q3.** Approve 30 epochs at ~15 h on RTX 4000 as the primary
    training run? (vs a 15-epoch proof-of-concept first, then a full
    run only if POC looks promising.)
  - **Q4.** Approve shipping realesr-animevideov3.pth in the GUI
    in parallel with the student retraining (Phase E)? This gives the
    user a quality option today without waiting for the student to
    finish training.
  - **Q5.** Budget for adding LPIPS as a dep (pip install lpips)?
    It pulls torch + torchvision variants and is ~50 MB. If no, we
    fall back to gradient-magnitude + SSIM-only eval.

Once Q1-Q4 are approved (Q5 is nice-to-have), we can start Phase A.

---

## 12. Definition of done

  - [ ] Phase A code merged (324649f -> new commit on
        feature/phase-1-realtime-4k).
  - [ ] Phase B smoke tests all PASS.
  - [ ] Phase C training run completed; per-epoch ckpts preserved at
        runs/distill_v3_4x/epoch_*.pt and runs/distill_v3_4x/epoch_*_ema.pt.
        Best EMA ckpt copied to pretrained/RFDN_distill_v3_4x_student.pth
        only after D gate passes.
  - [ ] Phase D eval PASS on lap_var >= 35 gate, visual review
        accepted.
  - [ ] Phase E RealESRGAN baseline shipped (independent of C/D).
  - [ ] Phase F docs updated.
  - [ ] New commit(s) on feature/phase-1-realtime-4k.
  - [ ] No regression in Phase 1 / Phase 2 E2E numbers (re-run the
        Phase 1 smoke and confirm single-4x path still works).

---

## 13. References

  - docs/plans/quality_compare_students_vs_pretrained_2026_08.md --
    Phase 2.D quality findings that motivated this plan.
  - docs/plans/phase2_cascade_plan.md -- the 2x+2x cascade negative
    result; confirms cascade is not the answer.
  - docs/plans/realtime_4k_plan.md -- Phase 1 real-time 4K target;
    defines the speed / VRAM budget this plan must respect.
  - anime_upscaler/student.py -- current RFDN student definition.
  - anime_upscaler/distill.py -- current distillation loop.
  - anime_upscaler/losses/perceptual_loss.py -- VGG perceptual loss.
  - anime_upscaler/losses/adversarial_loss.py -- existing adversarial
    loss scaffolding (likely needs minor updates for Hinge + PatchGAN).
  - apps/anime_upscaler_gui/anime_upscaler_gui/archs.py -- vendored
    arch stubs (SRVGGNetCompact, RFDN) used by the eval harness.
  - scripts/compare_students_vs_pretrained.py -- the eval harness
    that gates this plan.
