# Phase 2 TODO Tracker — Cascade 2× + 2× + Batching + NVENC

**Plan:** `docs/plans/phase2_cascade_plan.md`
**Started:** 2026-08-31 (continuation from Phase 1 commit `a3d54a9`)
**Last updated:** 2026-08-31
**Branch:** `feature/phase-1-realtime-4k` (continues Phase 1)
**Hardware:** Quadro RTX 4000 8 GB, single GPU
**Goal:** ≥ 30 fps @ 4K end-to-end with cascade 2×+2× + Phase 1 stack (TRT + b4 + NVENC).
**Status (2026-08-31 EOD):** GOAL NOT MET. Cascade is functional but slower
than single 4× and -1.39 dB on PSNR. See Section "Findings" at end. Commits:
  * 90d10b4 -- Phase 2.A code (RFDN scale + dataset + distill)
  * e6e2da3 -- Phase 2.B (cascade plumbing + eval script)
  * 27856d1 -- Phase 2.C (training + measurement, this commit)

---

## Status legend
- `[ ]` pending
- `[~]` in progress
- `[x]` done
- `[!]` blocked
- `[-]` cancelled / deferred

---

## A. Code (small changes, ~ 0.5 day)

- [ ] A.1  Edit `anime_upscaler/student.py`: rename `upscale` to `scale` on `RFDN`; document usage
- [ ] A.1b Edit `apps/.../archs.py`: mirror A.1 in the vendored RFDN class
- [ ] A.1c Edit `apps/.../archs.py::build`: sniff scale from `upsampler.0.weight` shape, build `RFDN(scale=sniffed)`
- [ ] A.2  Edit `anime_upscaler/dataset.py`: move `SCALE = 4` into `AnimePairDataset.__init__(self, scale=4)`
- [ ] A.3  Edit `anime_upscaler/distill.py`: add `--scale {2,4}` arg, pass to student + dataset
- [ ] A.4  Create `scripts/eval_cascade_vs_single.py`: load val pairs; compute bicubic / single-4x / cascade-2x2x; print PSNR+SSIM table
- [ ] A.5a Edit `apps/.../pipeline.py`: add `_RunJob.cascade_mode` and `_CascadeMode` enum; in `_process_*` paths, run `y = backend(y)` once when `cascade_mode == CASCADE_2X2X`
- [ ] A.5b Edit `apps/.../pipeline.py`: confirm TR engine cache works for cascade (2 distinct HxW keys → 2 engines, each cached)
- [ ] A.5c Edit `apps/.../settings.py`: add `_Defaults.cascade_mode: str = "single_4x"` with `"single_4x" | "cascade_2x2x"` validator
- [ ] A.5d Edit `apps/.../widgets/<settings or models panel>`: add Cascade mode combobox + tooltip
- [ ] A.5e Edit `apps/.../app.py`: populate `cascade_mode` into every `_RunJob`
- [ ] A.6  Edit `apps/.../registry.py`: add `rfdn_distill_v2_2x` preset entry (scale=2, source=trained)

### A. Tests
- [ ] A.7  py_compile all touched files (PASS)
- [ ] A.8  Smoke: load existing v1 4x ckpt with default `scale=4` (PASS — backwards compat)
- [ ] A.9  Smoke: `RFDN(scale=2)` forward — output shape (N,3,2H,2W), no NaN
- [ ] A.10 Smoke: distill.py --smoke with `--scale 2` — train loop runs 2 epochs, no shape mismatch

---

## B. Training (~ 1-2 days unattended)

- [ ] B.1  Run `python anime_upscaler/distill.py --scale 2 --epochs 30 --batch-size 16 --lr 5e-5 --out-dir runs/distill_v2_2x --data data/anime_video_frames`
- [ ] B.2  Watch first 5 epochs; if val_psnr < 28 dB, halt and re-tune
- [ ] B.3  Copy `runs/distill_v2_2x/student_best.pt` → `pretrained/RFDN_distill_v2_2x_student.pth`
- [ ] B.4  Sanity load 2x ckpt via `archs.build("rfdn_student", path)` and a 480x270 LR — output 960x540, no NaN

---

## C. Eval & Validation (~ 0.5 day)

- [ ] C.1  `python scripts/eval_cascade_vs_single.py --val-batches 50 --ckpt-2x pretrained/RFDN_distill_v2_2x_student.pth --ckpt-4x pretrained/RFDN_distill_v1_4x_student.pth --val-dir data/anime_video_frames` — cascade PSNR ≥ 29.6 dB; |Δ| vs single 4x ≥ 0.3 dB
- [ ] C.2  Visual sanity on `.venv/test_30s_TTA.mp4` (or generate 30s clip if missing) — no obvious artifacts vs single 4x
- [ ] C.3  Image job: cascade mode, batch=1, sample LR — output shape (1,3,4H,4W), no NaN
- [ ] C.4  Cascade-mode ERFDN throughput at fixed batch=1 vs single 4x throughput — expect ~1.5× cascade speedup on inference-only bench

---

## D. E2E bench (~ 0.25 day)

- [ ] D.1  Extend `tmp/test_e2e.py` CONFIGS with `trt_b4_nvenc_cascade` and `trt_b1_nvenc_cascade`
- [ ] D.2  Run E2E on `.venv/test_4k_540p_input.mp4` with all cascade configs
- [ ] D.3  Record fps + per-frame infer + CPU% + file size into `tmp/e2e_out/results.json`
- [ ] D.4  Pass criterion: end-to-end fps ≥ 30 (`cascade b4 NVENC`); if 25–30, ship with "experimental" tag

---

## E. Docs (~ 0.25 day)

- [ ] E.1  Append `Phase 2 measurements` section to `docs/rfdn_realtime_report.md`
- [ ] E.2  Add `Phase 2 Shipped (2026-08-31)` entry to `AGENTS.md`
- [ ] E.3  Commit `--  Phase 2 (Real-time 4K): Cascade 2×+2× + NVDEC shipped` (or two commits: code-only + cascade-traffic)

---

## Cross-cutting

- [ ] X.1  Engine cache LRU now touches 2× more engines per job — verify 1 GB cap holds for clip at 854×480 + 1708×960 (per batch=4 ~600 MB combined)

---

## Reference

- **Plan:** `docs/plans/phase2_cascade_plan.md`
- **Phase 1 plan:** `docs/plans/realtime_4k_plan.md`
- **Phase 1 ship banner:** `docs/rfdn_realtime_report.md` section 10
- **Phase 1 commit:** `a3d54a9` on `feature/phase-1-realtime-4k`
- **v3 distill recipe:** see `distill.py` docstring → `.claude/plans/distill_v3_recipe.md`
- **E2E harness:** `tmp/test_e2e.py`
- **Test clip:** `.venv/test_4k_540p_input.mp4`

---

## Findings (2026-08-31 EOD)

**The cascade 2×+2× hypothesis was wrong for this architecture.**

| Metric | single 4× | cascade 2×2× | delta |
|---|---:|---:|---:|
| Test PSNR (val, 25 batches) | 30.10 dB | 28.71 dB | **-1.39 dB** |
| PyTorch E2E fps @ 4K (b1)   | 10.02 fps | 2.86 fps  | **-7.16 fps** |
| PyTorch infer ms/frame      | 83.44 ms  | 310 ms    | **3.7× slower** |

The body-heavy RFDN architecture dominates the compute cost at LR
resolution. A single 4× stage runs body convolutions once at LR
resolution; cascade 2×+2× runs body convolutions **at LR (stage 1) +
4× more pixels at stage 2** -- both stages are non-trivial. Total
compute is roughly 5-8× LR-units vs single 4× ~ 2× LR-units.

For RFDN specifically:
- Single 4× at LR=480x854: 24 ms (TRT) inference, 18.80 fps E2E
- Cascade 2×+2× at same LR: ~85-150 ms inference, 2.86 fps E2E

**The 30 fps goal requires a different architectural approach.**

### Open follow-ups (not actioned in this sprint)

- [ ] CUDA graph capture of the TRT 4× engine (saves 5-10 ms of
      per-frame kernel-launch overhead on a small model).
- [ ] NVDEC decode integration that actually engages (the
      `_PyAvReader` had a hwaccel attempt but didn't engage in
      tests; same decoder was 17.80 fps vs cv2 18.80 fps, no win).
- [ ] Re-evaluate the v2_2x student in a different role: Phase 2B's
      Option-2 self-distilled stage 2 could close the quality gap.
- [ ] Smaller student ("distilled-tiny" -- ~50K params): the
      existing 4× at 24 ms is hard to beat; a smaller model might
      fit two in cascade within the same latency.

