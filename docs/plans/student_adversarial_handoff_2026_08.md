# Handoff: Adversarial Student Distilled from RealESRGAN-animevideov3

> **Read this first.** This is the resume document for Phase 3 work.
> It contains every command, file path, decision, gate, and halt
> criterion you need to continue from this point with no prior context.
> If you are a fresh agent (or future me) and the session memory is
> empty, working through the TODOs in this file is sufficient to
> complete the plan end-to-end.

**Last updated:** 2026-08-31
**Branch:** feature/phase-1-realtime-4k
**Driver / author:** MiniMax-M3
**Parent plan:** docs/plans/student_adversarial_plan.md (read this
  alongside the handoff; the plan is the why, this doc is the how)
**Parent report:** docs/plans/quality_compare_students_vs_pretrained_2026_08.md

---

## 0. One-paragraph context

Our distilled students (v1 4x and v2 2x+2x) are visibly over-smoothed
vs RealESRGAN-animevideov3 (lap_var 21.0 vs 58.1). Root cause:
(a) pixel + VGG perceptual loss only -- no adversarial signal;
(b) bicubic residual shortcut anchors student output near bicubic;
(c) distillation teacher is photo-real SPAN, not anime-trained. Phase 3
trains a new v3 student with adversarial loss against the animevideov3
teacher, annealed bicubic shortcut, and per-epoch checkpointing so no
epoch is lost on interruption.

---

## 1. Approved decisions (user signed off)

  - Q1. **Teacher**: realesr-animevideov3.pth (primary), 4xLSDIRCompactv2.pth (ablation).
  - Q2. **Warm-start first** from RFDN_distill_v1_4x_student.pth, with
       from-scratch as fallback if warm-start stalls (lap_var < 25 at
       epoch 15).
  - Q3. **30 epochs** at ~15 h on RTX 4000 as primary training run.
  - Q4. **Ship realesr-animevideov3.pth in the GUI in parallel** (Phase E).
  - Q5. **LPIPS dep OK** (already imported in distill.py -- no install
       needed; verify in step B.0 below).
  - Q6 (added by user). **Checkpoint every epoch** -- never lose work to
       a crash or kill.

---

## 2. Current state of the repo (as of this handoff)

```
branch:       feature/phase-1-realtime-4k
HEAD:         0c79241 plan: checkpoint every epoch (do not lose any on interruption)
prior commits:
              ea9486d plan: checkpoint every epoch (do not lose any)
              8e7d353 Phase 3 plan: adversarial student distilled from RealESRGAN-animevideov3
              324649f Phase 2.D: quality comparison students vs RealESRGAN baselines
              5285312 Phase 2.C2: cascade real-video confirmation
              b8fb25a Phase 2.C.doc cascade 2x+2x hypothesis did not pan out
              27856d1 Phase 2.C v2_2x student trained; cascade measurement
              e6e2da3 Phase 2.B cascade plumbing + quality eval script
              90d10b4 Phase 2.A cascade-2x+2x student infrastructure
              a3d54a9 Phase 1 (Real-time 4K): TRT + Batching + NVENC
uncommitted:  none (working tree clean for tracked files; large untracked
              backlog of untracked files in anime_upscaler/ etc. -- out
              of scope for this work)
```

---

## 3. Critical pre-flight (run before starting Phase A)

These are guards that catch the most common reasons for wasting time.
Run all of them once, fix any FAIL, then proceed.

  - [ ] F.1  Working dir is the repo root and branch is feature/phase-1-realtime-4k
        ```pwsh
        git rev-parse --show-toplevel
        git branch --show-current    # MUST be feature/phase-1-realtime-4k
        git status --short           # tracked files clean
        ```

  - [ ] F.2  Required weights and inputs are on disk
        ```pwsh
        Test-Path pretrained/RFDN_distill_v1_4x_student.pth          # must exist
        Test-Path pretrained/realesr-animevideov3.pth                # must exist
        Test-Path pretrained/4xLSDIRCompactv2.pth                    # must exist
        Test-Path tmp/real_video_1sec.mp4                           # 18-frame test clip
        Test-Path data/anime_video_frames                            # training data
        ```
        If any are missing, halt and ask the user. Do NOT fabricate weights.

  - [ ] F.3  Python deps (most are pre-installed per the CLAUDE.md env report)
        ```pwsh
        .venv\Scripts\python.exe -c "import torch, numpy, cv2, piq, lpips; import torch.nn.functional as F; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
        ```
        Expected: torch 2.12.0+cu126, cuda True. LPIPS may need `pip install lpips` if missing.

  - [ ] F.4  CWD + PYTHONPATH for anime_upscaler/ modules
        ```pwsh
        cd <repo root>
        $env:PYTHONPATH = "$PWD\anime_upscaler;$env:PYTHONPATH"
        .venv\Scripts\python.exe -c "from student import RFDN; from teacher import SPANTeacher; from dataset import AnimePairDataset; print('imports OK')"
        ```

  - [ ] F.5  Confirm the **current teacher in distill.py is SPAN**, not RealESRGAN
        ```pwsh
        Select-String -Path anime_upscaler\distill.py -Pattern "^from teacher import"
        ```
        Expected: `from teacher import SPANTeacher, FEATURE_CHANNELS, TAP_ORDER`
        This is why Phase A.3 must add a RealESRGAN teacher module.

---

## 4. Phase A -- Code (0.5 day)

### A.0 Pre-flight (already in F.1-F.5)

### A.1 Create anime_upscaler/losses/adversarial.py (NEW FILE)

  - What: PatchGAN discriminator + Hinge GAN loss with spectral norm.
  - Where: anime_upscaler/losses/__init__.py (create this if missing),
           anime_upscaler/losses/adversarial.py.
  - Spec:
    - class PatchGAN70(nn.Module): 4 strided conv blocks C64,C128,C256,C512
      kernel=4 stride=2 padding=1, LeakyReLU(0.2), spectral norm on
      every conv. Final 1x1 to 1-channel logit map. Input is
      concat(LR_upscaled_to_HR, SR_or_HR) = 6 channels.
    - class HingeGANLoss(nn.Module): forward(D(real), D(fake)) returns
      (d_loss, g_loss). D loss = mean(relu(1 - D(real))) + mean(relu(1 + D(fake))).
      G loss = -mean(D(fake)).
    - Apply torch.nn.utils.spectral_norm on every conv layer at __init__.
  - Reference: src/losses/adversarial_loss.py::SRDiscriminator for shape;
    REPLACE BatchNorm with no-norm (use spec-norm only) and Hinge loss.
  - Smoke test:
    ```pwsh
    .venv\Scripts\python.exe -c "from losses.adversarial import PatchGAN70, HingeGANLoss; d = PatchGAN70().cuda(); import torch; x = torch.randn(2,6,256,256).cuda(); print('D out', d(x).shape)"
    ```
    Expected: D out torch.Size([2, 1, 16, 16]) (for 256x256 input).

### A.2 Create anime_upscaler/losses/edge_loss.py (NEW FILE)

  - What: Sobel-x + Sobel-y magnitude, L1 against HR.
  - Where: anime_upscaler/losses/edge_loss.py
  - Spec:
    - class EdgeLoss(nn.Module):
        forward(sr, hr) returns mean(|Sobel(sr)| - |Sobel(hr)|) in L1.
    - Sobel kernels are fixed buffers (no learnable params).
    - Operate on Y channel only (cv2-style RGB->Y) to focus on luma edges.
  - Smoke test:
    ```pwsh
    .venv\Scripts\python.exe -c "from losses.edge_loss import EdgeLoss; import torch; e = EdgeLoss().cuda(); sr=torch.randn(2,3,128,128).cuda(); hr=torch.randn(2,3,128,128).cuda(); print('edge loss', e(sr,hr).item())"
    ```
    Expected: a small positive number (~0.5-2.0 range).

### A.3 Edit anime_upscaler/teacher.py -- add RealESRGAN teacher

  - What: New class RealESRTeacher(nn.Module) that loads
    realesr-animevideov3.pth via the existing SRVGGNetCompact stub
    (already in apps/.../archs.py, copy into anime_upscaler/teacher.py).
  - Why: distill.py imports `from teacher import ...`; the cleanest
    extension is to add the new class to that module.
  - Spec:
    - class RealESRTeacher(nn.Module):
        def __init__(self, ckpt_path, scale=4):
            build SRVGGNetCompact (16 convs, nf=64, scale=scale)
            load_state_dict from ckpt_path (handle params / params_ema
            wrappers -- same logic as src/losses/_load_state_dict or
            apps/.../archs.py::_load_state_dict)
            eval(), freeze params
            register to device on .cuda()
        def forward(self, lr): return self.model(lr)
  - Also add a module-level registry dict:
    TEACHERS = {'span': (SPANTeacher, {...}), 'animevideov3': (RealESRTeacher, {'ckpt': 'pretrained/realesr-animevideov3.pth'}), 'lsdir': (RealESRTeacher, {'ckpt': 'pretrained/4xLSDIRCompactv2.pth'})}
    so distill.py can dispatch by --teacher name.
  - Smoke test:
    ```pwsh
    .venv\Scripts\python.exe -c "from teacher import RealESRTeacher; t = RealESRTeacher('pretrained/realesr-animevideov3.pth').cuda().eval(); import torch; y = t(torch.randn(1,3,64,64).cuda()); print('animevideov3 SR', y.shape)"
    ```
    Expected: torch.Size([1, 3, 256, 256]) (64x64 -> 256x256 = scale=4).

### A.4 Edit anime_upscaler/distill.py -- new CLI flags + loop

  - What: Add the new teacher choice, anneal schedule for the bicubic
    shortcut, per-epoch checkpoint rotation, and the four new loss
    terms (L_distill from animevideov3, L_adv, L_grad, modified L_pix).
  - New CLI args (add to existing argparse):
    - `--teacher {span,animevideov3,lsdir}` (default: animevideov3)
    - `--lambda-adv 0.001`        (peak adversarial weight; ramp is hard-coded)
    - `--shortcut-anneal 1to0`    (ep 1:1.0 -> ep 6+:0.0)
    - `--resume path/to/v1.pth`   (warm-start)
    - `--no-ema`                  (disable EMA for ablation)
  - Loss wiring (replace existing single-loss section):
    - L_pix weight: 0.5 (down from 1.0 if currently 1.0)
    - L_perc weight: 1.0 (unchanged if already 1.0)
    - L_distill: 0.5 * L1(student_SR, teacher_SR) where teacher = --teacher
    - L_grad: 0.1 * EdgeLoss()(student_SR, HR)
    - L_adv: lambda_adv(t) * HingeGANLoss.G(D(G_output))
  - Bicubic shortcut annealing:
    - Modify student forward OR add a wrapper that takes shortcut_weight.
    - Easiest: monkey-patch the residual in anime_upscaler/student.py::RFDN.forward
      OR add a `set_shortcut_weight(w)` method on RFDN.
    - Recommended: add `set_shortcut_weight(w: float)` to RFDN, store on
      self.shortcut_weight (default 1.0), multiply the bicubic add in forward.
  - Per-epoch checkpoint rotation (CRITICAL -- user requirement):
    ```python
    # At end of each epoch in train loop:
    save_path = out_dir / f'epoch_{epoch}.pt'
    save_ema_path = out_dir / f'epoch_{epoch}_ema.pt'
    torch.save({'student': student.state_dict(), 'D': D.state_dict(), 'epoch': epoch}, save_path)
    torch.save({'student': ema_student.state_dict(), 'epoch': epoch}, save_ema_path)
    with open(out_dir / f'epoch_{epoch}_metrics.json', 'w') as f:
        json.dump({'val_psnr': ..., 'val_ssim': ..., 'val_lpips': ..., 'val_lap_var': ...}, f)
    # Rotate 'latest' symlink/copy:
    shutil.copy(save_path, out_dir / 'latest.pt')
    shutil.copy(save_ema_path, out_dir / 'latest_ema.pt')
    # Keep last 5, archive older:
    epochs = sorted(out_dir.glob('epoch_[0-9]*.pt'))
    if len(epochs) > 5:
        archive = out_dir / 'archive'; archive.mkdir(exist_ok=True)
        for old in epochs[:-5]:
            shutil.move(str(old), str(archive / old.name))
            ema_old = old.with_name(old.stem + '_ema.pt')
            if ema_old.exists(): shutil.move(str(ema_old), str(archive / ema_old.name))
    ```

### A.5 Edit scripts/compare_students_vs_pretrained.py -- add v3 entry

  - What: Add a new MODELS row for the v3 student that loads from
    pretrained/RFDN_distill_v3_4x_student.pth once it exists.
  - The existing harness is the gate for Phase D -- keep it as-is
    except for the new row.
  - Set the output dir to results/quality_compare_v3_2026_08/ when v3
    is the focus run.

### A.6 Add scripts/train_v3_smoke.py (NEW FILE)

  - What: 2-epoch smoke harness that loads distill.py as a library,
    runs 2 epochs with adversarial on at lambda=0.001, prints G/D loss
    curves and saves a sample SR PNG to runs/smoke_v3/sample.png.
  - Why: gives Phase B a single entrypoint instead of ad-hoc invocations.

### A.7 Tests for Phase A

  - [ ] A.7a `py_compile` every new/edited file:
        ```pwsh
        .venv\Scripts\python.exe -m py_compile anime_upscaler/losses/adversarial.py anime_upscaler/losses/edge_loss.py anime_upscaler/teacher.py anime_upscaler/distill.py scripts/compare_students_vs_pretrained.py scripts/train_v3_smoke.py
        ```
        Expected: no output, exit code 0.
  - [ ] A.7b PatchGAN shape check (see A.1 smoke).
  - [ ] A.7c EdgeLoss positive scalar (see A.2 smoke).
  - [ ] A.7d RealESRTeacher forward (see A.3 smoke).
  - [ ] A.7e distill.py --help shows --teacher, --lambda-adv, --shortcut-anneal, --resume.

### A.8 Commit at end of Phase A

  ```pwsh
  git add anime_upscaler/losses/__init__.py anime_upscaler/losses/adversarial.py anime_upscaler/losses/edge_loss.py anime_upscaler/teacher.py anime_upscaler/distill.py scripts/compare_students_vs_pretrained.py scripts/train_v3_smoke.py
  git commit -m "Phase 3.A: PatchGAN + edge loss + RealESRTeacher + per-epoch ckpt"
  ```

---

## 5. Phase B -- Smoke validation (0.5 day)

Each smoke test is 2 epochs. If any FAILs, halt Phase A work and fix.

### B.1 Adversarial off (sanity)

  ```pwsh
  .venv\Scripts\python.exe scripts/train_v3_smoke.py --epochs 2 --lambda-adv 0 --out-dir runs/smoke_v3_noadv
  ```
  Expect: G loss drops; val_PSNR ~ current v1 (~30 dB); D loss is unused.
  Halt if: val_PSNR drops > 0.5 dB vs v1 baseline.

### B.2 Adversarial on at low weight

  ```pwsh
  .venv\Scripts\python.exe scripts/train_v3_smoke.py --epochs 2 --lambda-adv 0.001 --out-dir runs/smoke_v3_adv
  ```
  Expect: D loss > 0 and bounded (~0.3-1.0); G_adv loss in stable range.
  Halt if: any loss is NaN, D loss collapses to 0, or sample SR has visible checkerboard / mode collapse.

### B.3 Bicubic shortcut anneal

  ```pwsh
  .venv\Scripts\python.exe scripts/train_v3_smoke.py --epochs 2 --shortcut-anneal 1to0 --out-dir runs/smoke_v3_anneal
  ```
  Expect: sample SR at epoch 2 is NOT dominated by bicubic (visually inspect
  runs/smoke_v3_anneal/sample.png -- should look sharper than bicubic).
  Halt if: sample SR is nearly identical to bicubic-upsampled LR, or NaN.

### B.4 Commit at end of Phase B (only if any smoke FAIL was fixed)

  If you had to fix anything in A.1-A.6 to pass B.1-B.3, commit those fixes here.

---

## 6. Phase C -- Training (~1 day unattended, 15 h)

### C.0 Verify per-epoch checkpoint rotation is in distill.py

  Confirm the rotation snippet from A.4 is in place. If missing, halt and
  re-do A.4 before starting C.1.

### C.1 Warm-start primary run

  Run from PowerShell with output redirected to a log so you can monitor:

  ```pwsh
  $ts = Get-Date -Format 'yyyyMMdd_HHmmss'
  .venv\Scripts\python.exe anime_upscaler/distill.py `
    --teacher animevideov3 `
    --resume pretrained/RFDN_distill_v1_4x_student.pth `
    --epochs 30 `
    --batch-size 16 `
    --lr 2e-4 `
    --lambda-adv 0.001 `
    --shortcut-anneal 1to0 `
    --out-dir runs/distill_v3_4x `
    2>&1 | Tee-Object -FilePath runs/distill_v3_4x_$ts.log
  ```

  Estimated wall time: ~15 h. Safe to leave overnight.

### C.2 Monitor (separate terminal while C.1 runs)

  ```pwsh
  # Tail latest log line every minute (PowerShell)
  Get-Content runs/distill_v3_4x_$ts.log -Wait -Tail 20
  # Or just refresh the runs dir listing:
  Get-ChildItem runs/distill_v3_4x | Sort-Object LastWriteTime | Select-Object -Last 10 Name, Length
  ```

  Check after epoch 1, 5, 10, 20, 30:
  - runs/distill_v3_4x/epoch_N.pt and epoch_N_ema.pt exist (1.3MB each).
  - runs/distill_v3_4x/epoch_N_metrics.json has val_psnr / val_ssim /
    val_lpips / val_lap_var.
  - val_PSNR is monotonically non-decreasing (within noise).
  - val_LPIPS is decreasing.

### C.3 Halt criteria (during C.1)

  Stop the run (Ctrl-C is safe -- per-epoch ckpts are on disk) if:
  - val_PSNR drops > 0.5 dB for 3 consecutive epochs (mode collapse).
  - val_LPIPS does not improve for 5 consecutive epochs (saturation).
  - D loss or G_adv loss is NaN or diverges to >10.
  - val_lap_var < 25 at epoch 15 (warm-start not helping -- switch to C.4).

### C.4 From-scratch fallback (only if C.3 halt)

  ```pwsh
  .venv\Scripts\python.exe anime_upscaler/distill.py `
    --teacher animevideov3 `
    --epochs 30 `
    --batch-size 16 `
    --lr 2e-4 `
    --lambda-adv 0.001 `
    --shortcut-anneal 1to0 `
    --out-dir runs/distill_v3_4x_scratch
  ```

### C.5 Variant C (only if C.4 also stalls)

  Edit anime_upscaler/student.py to construct RFDN with num_blocks=4
  (down from 6). Re-run C.4 with --out-dir runs/distill_v3_4x_v3c.

### C.6 Variant D (only if C.5 also stalls)

  Edit student.py to use nf=64, num_blocks=4 (~470K params).
  Re-run with --out-dir runs/distill_v3_4x_v3d.

### C.7 Commit at end of Phase C

  ```pwsh
  # Training logs go into git? NO -- gitignore already excludes *.log.
  # But commit any code changes made during C (e.g. student.py edits in C.5/C.6).
  git status --short
  # If student.py or distill.py changed, commit them.
  ```

---

## 7. Phase D -- Eval (0.5 day)

### D.1 Re-run the quality harness with v3 added

  Pre-req: A.5 must have added v3 to MODELS in
  scripts/compare_students_vs_pretrained.py.

  ```pwsh
  .venv\Scripts\python.exe scripts/compare_students_vs_pretrained.py --output results/quality_compare_v3_2026_08 --v3-ckpt runs/distill_v3_4x/latest_ema.pt
  ```
  (Exact flag depends on how you wired it in A.5; default to whatever is
  least invasive.)

### D.2 Gates (must pass to accept v3)

  - [ ] val_lap_var on results/quality_compare_v3_2026_08/v3_student_4x_4x.png
        >= 35  (target was 35-60; current v1 is 21; animevideov3 is 58)
  - [ ] Visual: hair strands distinct, eye iris textured, no plastic skin.
        Open results/quality_compare_v3_2026_08/grid_zoom.png and compare
        hair_2x column to the animevideov3 hair_2x column.
  - [ ] If from-scratch run also exists (runs/distill_v3_4x_scratch),
        compare which is better; pick winner.

### D.3 Full-video eval (temporal stability)

  ```pwsh
  # Temporarily symlink the v3 ckpt into pretrained/ for the existing
  # test_e2e_cascade_real_video.py to find it:
  Copy-Item runs/distill_v3_4x/latest_ema.pt pretrained/RFDN_distill_v3_4x_student.pth -Force
  # Run the existing 4-endpoint E2E test:
  .venv\Scripts\python.exe tmp/test_e2e_cascade_real_video.py
  # Compare single_pt_b1 numbers vs the previous baseline (recorded in
  # docs/plans/todos_phase2_cascade_real_video_2026_08.md).
  ```
  Expect: similar fps, similar visual quality, NO new flicker.

### D.4 Manual zoom-crop review

  Open results/quality_compare_v3_2026_08/crops/v3_student_*_2x.png and
  compare against the existing crops/v1_student_4x_*_2x.png and
  crops/RealESRGAN_animevideov3_*_2x.png. Decision: accept or iterate.

### D.5 Promote to pretrained/

  Only after D.2 + D.3 + D.4 all PASS:

  ```pwsh
  # Use the EMA ckpt from the BEST epoch (not 'latest'), per epoch_N_metrics.json
  $best = (Get-Content runs/distill_v3_4x/epoch_*_metrics.json | ConvertFrom-Json |
           Sort-Object -Property val_lap_var -Descending | Select-Object -First 1).epoch
  Copy-Item runs/distill_v3_4x/epoch_${best}_ema.pt pretrained/RFDN_distill_v3_4x_student.pth -Force
  Get-ChildItem pretrained/RFDN_distill_v3_4x_student.pth
  ```

  **DO NOT delete** runs/distill_v3_4x/*.pt until the user explicitly
  accepts v3 in chat. They are the only fallback if a regression is
  later discovered.

### D.6 Commit at end of Phase D

  ```pwsh
  git add scripts/compare_students_vs_pretrained.py  # only if MODELS changed
  git commit -m "Phase 3.D: v3 eval + promote to pretrained/" -m "Best epoch: $best  val_lap_var: <value>"
  ```

---

## 8. Phase E -- Ship RealESRGAN baseline in parallel (0.5 day)

  This can run in parallel with Phase C. Independent of whether v3
  student training succeeds.

### E.1 Add srvgg preset to apps/.../registry.py

  Find where RealESRGAN_x4plus_anime_6B or any srvgg preset is registered.
  Add two new entries:
  - display name 'RealESRGAN AnimeVideo v3 (4x)' -> filename 'realesr-animevideov3.pth', kind='srvgg', scale=4
  - display name 'RealESRGAN LSDIR Compact (4x)' -> filename '4xLSDIRCompactv2.pth', kind='srvgg', scale=4

  If registry.py is fully script-generated, regenerate instead of editing.

### E.2 Refresh the models panel dropdown

  Edit apps/anime_upscaler_gui/anime_upscaler_gui/widgets/models_panel.py
  to surface the new entries. If the panel reads from registry at runtime,
  no edit needed -- just verify by launching the GUI.

### E.3 Manual GUI smoke

  ```pwsh
  .venv\Scripts\python.exe -m apps.anime_upscaler_gui
  ```
  Pick 'RealESRGAN AnimeVideo v3' from the model selector; load
  tmp/real_video_1sec.mp4; run. Verify:
  - 4x output 3416x1920 produced.
  - No CUDA errors, no NaN frames.
  - Visual quality matches the crops in
    results/quality_compare_students_vs_pretrained/.

### E.4 Commit at end of Phase E

  ```pwsh
  git add apps/anime_upscaler_gui/anime_upscaler_gui/registry.py apps/anime_upscaler_gui/anime_upscaler_gui/widgets/models_panel.py
  git commit -m "Phase 3.E: ship RealESRGAN-animevideov3 + LSDIR in model picker"
  ```

---

## 9. Phase F -- Documentation (0.25 day)

### F.1 Update the Phase 2.D quality report with v3 row

  Edit docs/plans/quality_compare_students_vs_pretrained_2026_08.md,
  add v3 to the metrics table once Phase D has the numbers.

### F.2 Write the v3 result doc

  New file docs/plans/student_v3_result_2026_08.md with:
  - Training curves (PSNR/SSIM/LPIPS/lap_var per epoch, plot or table).
  - Final eval table (v1, v2, animevideov3, v3).
  - Ship / iterate / abandon decision with reasoning.
  - Lessons learned (which variant worked, what to do next time).

### F.3 Update AGENTS.md if new conventions emerged

  If Phase A introduced new conventions (e.g. the bicubic shortcut
  anneal schedule, the PatchGAN spec-norm Hinge recipe), record them.

### F.4 Commit at end of Phase F

  ```pwsh
  git add docs/plans/quality_compare_students_vs_pretrained_2026_08.md docs/plans/student_v3_result_2026_08.md AGENTS.md
  git commit -m "Phase 3.F: v3 result doc + report update + AGENTS update"
  ```

---

## 10. Decision points (where to halt and re-plan)

  - **B.x FAIL** -- fix in Phase A, do not start C until B passes.
  - **C.3 halt triggers** -- switch to C.4 / C.5 / C.6.
  - **D.2 lap_var < 35** -- do not promote; iterate (try C.4-C.6; if all
    fail, fall back to shipping only the RealESRGAN baseline from E).
  - **D.3 visual regression** (flicker / color shift) -- likely the
    adversarial ramp was too aggressive. Re-run C with peak lambda_adv
    capped at 0.005 (not 0.01).
  - **Anything that costs > 0.5 day to fix** -- stop and ask the user.

---

## 11. File map (where every artifact lives)

| Artifact | Path | Git? |
|---|---|---|
| Plan | docs/plans/student_adversarial_plan.md | yes |
| Quality report | docs/plans/quality_compare_students_vs_pretrained_2026_08.md | yes |
| Eval harness | scripts/compare_students_vs_pretrained.py | yes |
| Student arch | anime_upscaler/student.py | yes |
| Distillation loop | anime_upscaler/distill.py | yes |
| Existing teacher (SPAN) | anime_upscaler/teacher.py | yes |
| Existing dataset | anime_upscaler/dataset.py | yes |
| Adversarial loss (NEW) | anime_upscaler/losses/adversarial.py | yes |
| Edge loss (NEW) | anime_upscaler/losses/edge_loss.py | yes |
| RealESR teacher (NEW) | anime_upscaler/teacher.py (add RealESRTeacher) | yes |
| Smoke harness (NEW) | scripts/train_v3_smoke.py | yes |
| Reference arch stubs | apps/anime_upscaler_gui/anime_upscaler_gui/archs.py | yes |
| Reference PatchGAN | src/losses/adversarial_loss.py | yes (use as shape reference) |
| v1 4x ckpt (warm-start) | pretrained/RFDN_distill_v1_4x_student.pth | gitignored |
| animevideov3 ckpt (teacher) | pretrained/realesr-animevideov3.pth | gitignored |
| LSDIR ckpt (ablation) | pretrained/4xLSDIRCompactv2.pth | gitignored |
| v3 ckpt (PROMOTED) | pretrained/RFDN_distill_v3_4x_student.pth | gitignored |
| Training output dir | runs/distill_v3_4x/ | not in repo |
| Per-epoch ckpts | runs/distill_v3_4x/epoch_N.pt, epoch_N_ema.pt | local only |
| Eval artifacts | results/quality_compare_v3_2026_08/ | gitignored |
| Training log | runs/distill_v3_4x_<ts>.log | gitignored |
| v3 result doc | docs/plans/student_v3_result_2026_08.md | yes |
| Real test video | tmp/real_video_1sec.mp4 | committed |
| Reference crops | results/quality_compare_students_vs_pretrained/crops/ | gitignored |

---

## 12. Commands cheat sheet (one place)

### Setup

```pwsh
cd <repo root>
$env:PYTHONPATH = "$PWD\anime_upscaler;$env:PYTHONPATH"
```

### Smoke (Phase B)

```pwsh
.venv\Scripts\python.exe scripts/train_v3_smoke.py --epochs 2 --lambda-adv 0 --out-dir runs/smoke_v3_noadv
.venv\Scripts\python.exe scripts/train_v3_smoke.py --epochs 2 --lambda-adv 0.001 --out-dir runs/smoke_v3_adv
.venv\Scripts\python.exe scripts/train_v3_smoke.py --epochs 2 --shortcut-anneal 1to0 --out-dir runs/smoke_v3_anneal
```

### Train (Phase C)

```pwsh
# Warm-start (primary)
.venv\Scripts\python.exe anime_upscaler/distill.py --teacher animevideov3 --resume pretrained/RFDN_distill_v1_4x_student.pth --epochs 30 --batch-size 16 --lr 2e-4 --lambda-adv 0.001 --shortcut-anneal 1to0 --out-dir runs/distill_v3_4x

# From-scratch (fallback)
.venv\Scripts\python.exe anime_upscaler/distill.py --teacher animevideov3 --epochs 30 --batch-size 16 --lr 2e-4 --lambda-adv 0.001 --shortcut-anneal 1to0 --out-dir runs/distill_v3_4x_scratch
```

### Eval (Phase D)

```pwsh
.venv\Scripts\python.exe scripts/compare_students_vs_pretrained.py --output results/quality_compare_v3_2026_08
.venv\Scripts\python.exe tmp/test_e2e_cascade_real_video.py
```

### Promote (Phase D.5)

```pwsh
$best = (Get-Content runs/distill_v3_4x/epoch_*_metrics.json | ConvertFrom-Json | Sort-Object -Property val_lap_var -Descending | Select-Object -First 1).epoch
Copy-Item runs/distill_v3_4x/epoch_${best}_ema.pt pretrained/RFDN_distill_v3_4x_student.pth -Force
```

### GUI smoke (Phase E)

```pwsh
.venv\Scripts\python.exe -m apps.anime_upscaler_gui
```

### Commit cadence

```pwsh
git add <files>
git commit -m "Phase 3.<phase>: <one-line summary>"
git log --oneline -5   # verify
```

---

## 13. Definition of done (full checklist)

  - [ ] F.1-F.5 pre-flight all PASS.
  - [ ] Phase A.1-A.7: code merged, all smoke shape tests pass.
  - [ ] Phase B.1-B.3: 3 smoke runs pass with no NaN / collapse.
  - [ ] Phase C.1: 30-epoch run started; per-epoch ckpts visible in
        runs/distill_v3_4x/ after epoch 1.
  - [ ] Phase C.2: training completes (or is halted per C.3 criteria).
  - [ ] Phase C.4 / C.5 / C.6: only if C.3 halt.
  - [ ] Phase D.2: val_lap_var >= 35.
  - [ ] Phase D.3: no new flicker, fps in expected range.
  - [ ] Phase D.4: visual review accepted.
  - [ ] Phase D.5: pretrained/RFDN_distill_v3_4x_student.pth exists.
  - [ ] Phase E.1-E.4: RealESRGAN baseline shipped in the GUI.
  - [ ] Phase F.1-F.3: docs updated.
  - [ ] No regression in Phase 1 single-4x path (smoke test).
  - [ ] **Per-epoch ckpts in runs/distill_v3_4x/ NOT deleted** until the
        user explicitly confirms acceptance.
  - [ ] New commit(s) on feature/phase-1-realtime-4k.

---

## 14. Gotchas / things that have bitten us before

  - **PowerShell comma in ffmpeg args**: `eq(n,8)` is split by PS array
    parsing. Use `eq(n\,8)` (backslash escape). Same rule may apply to
    subprocess.run argv if PowerShell pre-parses before python -- if you
    see 'No such filter: 8)' the escape is missing.
  - **PowerShell `-FilePath` vs `-FileObject`** in Tee-Object: the
    parameter is `-FilePath`. `-FileObject` is invalid.
  - **CUDA context pollution**: after a crash, fresh process. Use
    `torch.cuda.empty_cache()` between models if sharing a process.
  - **Bicubic residual shortcut MUST be annealed**: leaving it at 1.0
    forever reproduces the current blur. Verify student.py::RFDN has
    set_shortcut_weight and the loop actually calls it.
  - **Per-epoch ckpts are large** (~1.3 MB x 30 = 40 MB). Disk is fine,
    but if running on a constrained drive, archive older than epoch-25
    to runs/distill_v3_4x/archive/.
  - **D not pre-trained**: rely on the warm-up (ep 1-5 lambda_adv=0) to
    let the G stabilize before turning D on. Skipping this almost
    always causes D to overpower and the run to collapse.
  - **EMA only on G**, not on D. D must track the latest G behavior.

---

## 15. End-of-handoff summary

  - Plan doc: docs/plans/student_adversarial_plan.md
  - This handoff: docs/plans/student_adversarial_handoff_2026_08.md
  - Quality report: docs/plans/quality_compare_students_vs_pretrained_2026_08.md
  - Phase A code: anime_upscaler/losses/{adversarial,edge_loss}.py +
    edits to anime_upscaler/teacher.py and anime_upscaler/distill.py +
    scripts/train_v3_smoke.py (new) + scripts/compare_students_vs_pretrained.py
    (edit for v3 entry)
  - Phase E code: apps/anime_upscaler_gui/.../registry.py + widgets/models_panel.py
  - Phase D outputs: runs/distill_v3_4x/, results/quality_compare_v3_2026_08/
  - Phase F output: docs/plans/student_v3_result_2026_08.md

  Resume from any phase by reading Section 3 (pre-flight) and then the
  corresponding phase section above. Halts and decision points are
  explicitly called out in Sections 6.3, 7.2, 10, and 13.
